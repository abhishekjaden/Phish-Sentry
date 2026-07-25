from dotenv import load_dotenv
load_dotenv()

import os
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import requests as http_requests
import json
from datetime import datetime

from url_features import extract_features
from database import db
from gemini_helper import PhishingQuiz
from logging_config import configure_logging, get_logger
import uuid
import time
import structlog
from prometheus_flask_exporter import PrometheusMetrics
from tracing_config import setup_tracing
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

configure_logging()
logger = get_logger("phishguard.app")

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY')
if not app.secret_key:
    raise ValueError(
        "FLASK_SECRET_KEY environment variable is not set. "
        "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\" "
        "and add it to your .env file."
    )

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri=os.environ.get('REDIS_URL', 'memory://'),
    default_limits=["200 per hour", "50 per minute"]
)

from flask_talisman import Talisman
from flask_cors import CORS

# Security headers via Talisman.
# force_https=False because TLS is terminated upstream (Nginx/ingress) in production,
# and disabling it avoids breaking local HTTP testing.
csp = {
    'default-src': "'self'",
    'script-src': ["'self'", "'unsafe-inline'", 'https://cdn.tailwindcss.com', 'https://cdnjs.cloudflare.com'],
    'style-src': ["'self'", "'unsafe-inline'", 'https://cdnjs.cloudflare.com', 'https://fonts.googleapis.com', 'https://cdn.jsdelivr.net'],
    'font-src': ["'self'", 'https://fonts.gstatic.com', 'https://cdnjs.cloudflare.com', 'https://cdn.jsdelivr.net'],
    'img-src': ["'self'", 'data:'],
}
Talisman(
    app,
    content_security_policy=csp,
    force_https=False,
    session_cookie_secure=False,  # set True once HTTPS is in front in production
    frame_options='DENY'
)

# CORS - only applies to /api/* routes (used by the browser extension / API clients).
ALLOWED_ORIGINS = os.environ.get('ALLOWED_ORIGINS', 'http://localhost:8001').split(',')
CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGINS}})

# Prometheus metrics - exposes /metrics with request counts, latency histograms, status codes
metrics = PrometheusMetrics(app)
metrics.info('phishguard_app_info', 'PhishGuard application info', version='1.0.0')

# Exempt the Prometheus /metrics endpoint from rate limiting so the monitoring
# scrape is never throttled (it is internal, not user traffic).
@limiter.request_filter
def exempt_metrics():
    from flask import request
    return request.path == '/metrics' 

# Distributed tracing
setup_tracing("phishguard-app")
FlaskInstrumentor().instrument_app(app)
RequestsInstrumentor().instrument()


@app.before_request
def start_request_context():
    """Assign each request a unique ID and bind it to the logging context."""
    request_id = request.headers.get('X-Request-ID', str(uuid.uuid4()))
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        path=request.path,
        method=request.method,
    )
    request._start_time = time.time()


@app.after_request
def log_request(response):
    """Log every request with its latency and status."""
    latency_ms = None
    if hasattr(request, '_start_time'):
        latency_ms = round((time.time() - request._start_time) * 1000, 2)
    uid = session.get('user_id') if 'user_id' in session else None
    logger.info(
        "request_completed",
        status=response.status_code,
        latency_ms=latency_ms,
        user_id=uid,
    )
    response.headers['X-Request-ID'] = structlog.contextvars.get_contextvars().get('request_id', '')
    return response

# URL of the inference microservice (set in docker-compose)
INFERENCE_URL = os.environ.get('INFERENCE_URL', 'http://localhost:9000')
INFERENCE_TIMEOUT = 30  # seconds

# URL of the RAG retrieval microservice (set in docker-compose)
RAG_URL = os.environ.get('RAG_URL', 'http://localhost:9100')
RAG_TIMEOUT = 30  # seconds

from tasks import compute_shap_explanation_task
from jwt_auth import generate_tokens, decode_token, jwt_required
from email_helper import send_verification_email, send_reset_email, send_welcome_email
from celery.result import AsyncResult
# ---- Input validation helpers ----
import re as _re_validate

MAX_URL_LENGTH = 2048
MAX_TEXT_LENGTH = 10000

# Block known throwaway/disposable email providers at signup.
# (A blocklist is more production-correct than a strict allowlist, which would
#  wrongly reject legitimate company/university addresses.)
DISPOSABLE_EMAIL_DOMAINS = {
    'mailinator.com', 'guerrillamail.com', 'tempmail.com', '10minutemail.com',
    'throwawaymail.com', 'yopmail.com', 'trashmail.com', 'getnada.com',
    'temp-mail.org', 'fakeinbox.com', 'sharklasers.com', 'maildrop.cc',
}

def is_disposable_email(email):
    try:
        domain = email.split('@', 1)[1].lower().strip()
    except (IndexError, AttributeError):
        return False
    return domain in DISPOSABLE_EMAIL_DOMAINS

def validate_url(url):
    """Returns (is_valid, error_message)"""
    if not url or not isinstance(url, str):
        return False, "No URL provided"
    url = url.strip()
    if len(url) > MAX_URL_LENGTH:
        return False, f"URL too long (max {MAX_URL_LENGTH} characters)"
    url_pattern = _re_validate.compile(
        r'^(https?://)?'
        r'([\w\-]+\.)+[\w\-]+'
        r'(/[^\s]*)?$',
        _re_validate.IGNORECASE
    )
    if not url_pattern.match(url):
        return False, "Invalid URL format"
    return True, None

def validate_text(text):
    """Returns (is_valid, error_message)"""
    if not text or not isinstance(text, str):
        return False, "No text provided"
    if len(text) > MAX_TEXT_LENGTH:
        return False, f"Text too long (max {MAX_TEXT_LENGTH} characters)"
    return True, None


# ---- Inference service calls ----
def call_inference_text(text):
    """Call the inference microservice for text prediction."""
    try:
        resp = http_requests.post(
            f"{INFERENCE_URL}/predict/text",
            json={"text": text},
            timeout=INFERENCE_TIMEOUT
        )
        resp.raise_for_status()
        return resp.json()
    except http_requests.exceptions.RequestException as e:
        logger.error("inference_call_failed", service="text", error=str(e))
        return {"error": "Detection service unavailable. Please try again."}

def call_inference_explain(text):
    """Fetch SHAP word attributions from the inference service.
    Best-effort: returns {} on any error/timeout so it never breaks a prediction."""
    try:
        resp = http_requests.post(
            f"{INFERENCE_URL}/explain/text",
            json={"text": text},
            timeout=20,
        )
        data = resp.json()
        if isinstance(data, dict) and data.get("status") == "success":
            return {
                "top_risky_words": data.get("top_risky_words", []),
                "top_safe_words": data.get("top_safe_words", []),
            }
    except Exception as _e:
        pass
    return {}


def call_inference_url(url):
    """Call the inference microservice for URL prediction."""
    try:
        resp = http_requests.post(
            f"{INFERENCE_URL}/predict/url",
            json={"url": url},
            timeout=INFERENCE_TIMEOUT
        )
        resp.raise_for_status()
        return resp.json()
    except http_requests.exceptions.RequestException as e:
        logger.error("inference_call_failed", service="url", error=str(e))
        return {"error": "Detection service unavailable. Please try again."}


def call_rag_retrieve(query, top_k=5, candidates=20):
    """Call the RAG microservice to retrieve reranked threat-intel chunks."""
    try:
        resp = http_requests.post(
            f"{RAG_URL}/retrieve",
            json={"query": query, "top_k": top_k, "candidates": candidates},
            timeout=RAG_TIMEOUT
        )
        resp.raise_for_status()
        return resp.json()
    except http_requests.exceptions.RequestException as e:
        logger.error("rag_call_failed", error=str(e))
        return {"error": "Knowledge service unavailable. Please try again."}


@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        role = request.form.get('role')

        if not email or not password or not role:
            flash('Please fill in all fields', 'error')
            return render_template('login.html')

        user = db.authenticate_user(email, password, role)
        if user and not user.get('is_verified', False):
            db.log_event('login_blocked_unverified', user_id=user['id'], detail='unverified login attempt', ip_address=request.remote_addr)
            flash('Please verify your email before logging in - check your inbox for the link.', 'error')
            return render_template('login.html')
        if user:
            session['user_id'] = user['id']
            session['user_role'] = user['role']
            session['user_name'] = f"{user['first_name']} {user['last_name']}"
            db.log_event('login', user_id=user['id'], detail='web login', ip_address=request.remote_addr)
            logger.info("user_login", user_id=user['id'], method="web", role=user['role'])

            flash(f'Welcome back, {user["first_name"]}!', 'success')

            if user['role'] == 'admin':
                return redirect(url_for('admin_dashboard'))
            else:
                return redirect(url_for('dashboard'))
        else:
            db.log_event('login_failed', detail=f'web login attempt: {email}', ip_address=request.remote_addr)
            logger.warning("login_failed", method="web", email=email)
            flash('Invalid credentials. Please try again.', 'error')

    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def signup():
    if request.method == 'POST':
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if not all([first_name, last_name, username, email, password, confirm_password]):
            flash('Please fill in all fields', 'error')
            return render_template('signup.html')

        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('signup.html')

        if is_disposable_email(email):
            flash('Please use a permanent email address (disposable providers are not allowed).', 'error')
            return render_template('signup.html')

        if db.check_email_exists(email):
            flash('Email already exists', 'error')
            return render_template('signup.html')

        if db.check_username_exists(username):
            flash('Username already exists', 'error')
            return render_template('signup.html')

        user_id = db.create_user(first_name, last_name, username, email, password)
        if user_id:
            token = db.create_verification_token(user_id, 'verify', ttl_hours=24)
            if token:
                send_verification_email(email, token)
            db.log_event('signup', user_id=user_id, detail='account created (unverified)', ip_address=request.remote_addr)
            logger.info("user_signup", user_id=user_id, email=email)
            flash('Account created! Check your email for a verification link before logging in.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Error creating account. Please try again.', 'error')

    return render_template('signup.html')

@app.route('/verify/<token>')
@limiter.limit("20 per minute")
def verify_email(token):
    info = db.get_valid_token(token, 'verify')
    if not info:
        flash('This verification link is invalid or has expired.', 'error')
        return redirect(url_for('login'))

    db.set_user_verified(info['user_id'])
    db.mark_token_used(token)

    user = db.get_user_by_id(info['user_id'])
    if user:
        send_welcome_email(user['email'], user['first_name'])
        db.log_event('email_verified', user_id=user['id'], detail='email verified', ip_address=request.remote_addr)
        logger.info("email_verified", user_id=user['id'])

    flash('Your email is verified! You can now log in.', 'success')
    return redirect(url_for('login'))


@app.route('/forgot-password', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        if email:
            user = db.get_user_by_email(email.strip())
            if user:
                token = db.create_verification_token(user['id'], 'reset', ttl_hours=1)
                if token:
                    send_reset_email(user['email'], token)
                db.log_event('password_reset_requested', user_id=user['id'], detail='reset requested', ip_address=request.remote_addr)
        # Always the same message - never reveal whether an email is registered.
        flash('If an account exists for that email, a reset link has been sent.', 'success')
        return redirect(url_for('login'))
    return render_template('forgot_password.html')


@app.route('/reset-password/<token>', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def reset_password(token):
    info = db.get_valid_token(token, 'reset')
    if not info:
        flash('This password reset link is invalid or has expired.', 'error')
        return redirect(url_for('login'))

    if request.method == 'POST':
        password = request.form.get('password')
        confirm = request.form.get('confirm_password')
        if not password or not confirm:
            flash('Please fill in both password fields.', 'error')
            return render_template('reset_password.html', token=token)
        if password != confirm:
            flash('Passwords do not match.', 'error')
            return render_template('reset_password.html', token=token)
        if len(password) < 8:
            flash('Password must be at least 8 characters.', 'error')
            return render_template('reset_password.html', token=token)

        db.update_user_password(info['user_id'], password)
        db.mark_token_used(token)
        db.log_event('password_reset', user_id=info['user_id'], detail='password reset', ip_address=request.remote_addr)
        logger.info("password_reset", user_id=info['user_id'])
        flash('Your password has been reset. You can now log in.', 'success')
        return redirect(url_for('login'))

    return render_template('reset_password.html', token=token)

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully.', 'success')
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user = db.get_user_by_id(session['user_id'])
    recent_scans = db.get_user_scans(session['user_id'], limit=5)

    total_scans = len(db.get_user_scans(session['user_id'], limit=1000))
    phishing_detected = len([s for s in db.get_user_scans(session['user_id'], limit=1000) if s['is_phishing']])
    legitimate_detected = total_scans - phishing_detected
    # "Clean scan rate" = share of this user's scans that came back legitimate.
    # NB: this is NOT a measure of the user's safety — someone who deliberately
    # scans many phishing samples will (correctly) see a low rate.
    clean_scan_rate = int((legitimate_detected / total_scans * 100)) if total_scans > 0 else 100

    user_stats = {
        'total_scans': total_scans,
        'phishing_detected': phishing_detected,
        'legitimate_detected': legitimate_detected,
        'clean_scan_rate': clean_scan_rate,
        'safety_score': clean_scan_rate  # back-compat alias for the template
    }

    return render_template('dashboard.html', user=user, recent_scans=recent_scans, user_stats=user_stats)

@app.route('/detection')
def detection():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user = db.get_user_by_id(session['user_id'])
    return render_template('detection.html', user=user)

@app.route('/results')
def results():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user = db.get_user_by_id(session['user_id'])
    return render_template('results.html', user=user)

def _autosave_scan(user_id, scan_type, content, result, source):
    """Persist a scan result to history. Best-effort: never breaks the response.
    `result` is the dict returned by the inference service.
    De-dupes: skips saving if an identical (content, source) scan was recorded in
    the last 60s — the extension's onUpdated listener can fire several times per
    page load, which would otherwise create duplicate rows."""
    try:
        if not user_id or not isinstance(result, dict) or 'error' in result:
            return
        # De-dupe against very recent identical scans (same user/content/source).
        try:
            from datetime import datetime, timezone
            recent = db.get_user_scans(user_id, limit=10)
            content_head = (content or '')[:100]
            for s in recent:
                if s.get('source') != source:
                    continue
                # get_user_scans truncates content to 100 chars + '...'; compare heads
                existing_head = (s.get('content') or '').rstrip('.')[:100]
                if existing_head and existing_head == content_head.rstrip('.')[:100]:
                    ts = s.get('created_at')
                    if ts:
                        try:
                            dt = datetime.fromisoformat(ts)
                            now = datetime.utcnow()
                            if (now - dt.replace(tzinfo=None)).total_seconds() < 60:
                                return  # duplicate within 60s — skip save
                        except Exception:
                            pass
        except Exception:
            pass  # if the dedupe check fails, fall through and save normally
        is_phishing = bool(result.get('is_phishing'))
        confidence = result.get('confidence', 0)
        verdict = 'Phishing' if is_phishing else 'Legitimate'
        db.save_scan(
            user_id,
            scan_type,
            content,
            verdict,
            confidence,
            is_phishing,
            shap_explanation=None,
            source=source
        )
    except Exception:
        pass  # history persistence must never fail the user's scan


@app.route('/predict-text', methods=['POST'])
@limiter.limit("30 per minute")
def predict_text():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid request'}), 400
    text = data.get('text')

    is_valid, error = validate_text(text)
    if not is_valid:
        return jsonify({'error': error}), 400

    result = call_inference_text(text)
    # Best-effort SHAP overlay: merge word attributions into the result.
    # If SHAP is slow or fails, the prediction still returns unaffected.
    if isinstance(result, dict) and 'error' not in result:
        result.update(call_inference_explain(text))
    _autosave_scan(session.get('user_id'), 'email', text, result, source='web')
    return jsonify(result)
@app.route('/explain-text', methods=['POST'])
@limiter.limit("15 per minute")
def explain_text():
    """Kick off an async SHAP explanation job. Returns a job ID immediately."""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid request'}), 400
    text = data.get('text')

    is_valid, error = validate_text(text)
    if not is_valid:
        return jsonify({'error': error}), 400

    job = compute_shap_explanation_task.delay(text)
    return jsonify({'job_id': job.id, 'status': 'queued'})


@app.route('/explain-status/<job_id>', methods=['GET'])
def explain_status(job_id):
    """Poll for the result of a SHAP explanation job."""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    result = AsyncResult(job_id)
    if result.ready():
        return jsonify({'status': 'done', 'result': result.get()})
    else:
        return jsonify({'status': 'pending'})

@app.route('/analyze', methods=['POST'])
@limiter.limit("30 per minute")
def analyze():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    url = request.form.get('url')

    is_valid, error = validate_url(url)
    if not is_valid:
        return jsonify({'error': error}), 400

    result = call_inference_url(url.strip())
    if 'error' in result:
        return jsonify(result), 503

    _autosave_scan(session.get('user_id'), 'url', url.strip(), result, source='web')

    return jsonify({
        'is_phishing': result.get('is_phishing'),
        'confidence': result.get('confidence'),
        'prediction_value': result.get('prediction_value'),
        'top_features': []
    })

@app.route('/save-scan', methods=['POST'])
def save_scan():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    data = request.get_json()
    scan_id = db.save_scan(
        session['user_id'],
        data['scan_type'],
        data['content'],
        data['result'],
        data['confidence'],
        data['is_phishing'],
        data.get('shap_explanation')
    )

    if scan_id:
        return jsonify({'success': True, 'scan_id': scan_id})
    else:
        return jsonify({'success': False}), 500

@app.route('/quiz')
def quiz():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user = db.get_user_by_id(session['user_id'])
    return render_template('quiz.html', user=user)

@app.route('/generate-quiz', methods=['GET', 'POST'])
def generate_quiz_route():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    try:
        quiz_instance = PhishingQuiz()
        questions = quiz_instance.generate_quiz_questions()
        session['quiz_questions'] = questions
        return jsonify({'questions': questions})
    except Exception as e:
        print(f"Error generating quiz: {e}")
        return jsonify({'error': 'Error generating quiz. Please try again.'}), 500

@app.route('/submit-quiz', methods=['POST'])
def submit_quiz():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    data = request.get_json()
    user_answers = data['answers']
    questions = data['questions']

    try:
        quiz_instance = PhishingQuiz()
        score, feedback_details = quiz_instance.calculate_quiz_score(user_answers, questions)
        total_questions = len(questions)
        feedback = quiz_instance.get_quiz_feedback(score, total_questions, feedback_details)

        return jsonify({
            'score': score,
            'feedback': feedback,
            'results': feedback_details
        })
    except Exception as e:
        print(f"Error in /submit-quiz: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/chatbot', methods=['GET', 'POST'])
@limiter.limit("20 per minute")
def chatbot():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user = db.get_user_by_id(session['user_id'])
    if request.method == 'GET':
        return render_template('chatbot.html', user=user)

    user_message = request.json.get('message')
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    try:
        quiz_instance = PhishingQuiz()
        response = quiz_instance.get_chatbot_response(user_message)
        return jsonify(response)
    except Exception as e:
        print(f"Error in chatbot route: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/rag/query', methods=['POST'])
@limiter.limit("20 per minute")
def rag_query():
    """
    Retrieval-augmented answer: pull reranked threat-intel chunks from the RAG
    service and have Gemini answer grounded strictly in them, with citations.
    """
    if 'user_id' not in session:
        return redirect(url_for('login'))

    payload = request.json or {}
    question = payload.get('question') or payload.get('message')
    is_valid, err = validate_text(question)
    if not is_valid:
        return jsonify({'error': err}), 400

    chunks = call_rag_retrieve(question.strip())
    # call_rag_retrieve returns a list on success, or {"error": ...} on failure.
    if isinstance(chunks, dict) and 'error' in chunks:
        return jsonify(chunks), 503
    if not chunks:
        return jsonify({
            'answer': "I couldn't find anything relevant in the knowledge base for that.",
            'sources': []
        })

    try:
        quiz_instance = PhishingQuiz()
        result = quiz_instance.get_grounded_answer(question.strip(), chunks)
    except Exception as e:
        logger.error("rag_answer_failed", error=str(e))
        return jsonify({'error': str(e)}), 500

    sources = [
        {
            'n': i + 1,
            'title': c.get('title'),
            'doc_id': c.get('doc_id'),
            'chunk_index': c.get('chunk_index'),
            'rerank_score': c.get('rerank_score'),
        }
        for i, c in enumerate(chunks)
    ]
    return jsonify({'answer': result.get('answer'), 'sources': sources})


@app.route('/agent/analyze', methods=['POST'])
@limiter.limit("10 per minute")
def agent_analyze():
    """
    Agentic phishing analysis: an LLM agent autonomously decides which tools to
    call (URL classifier, text classifier, threat-intel retriever) and returns a
    reasoned, knowledge-base-grounded verdict.
    """
    if 'user_id' not in session:
        return redirect(url_for('login'))

    payload = request.json or {}
    user_input = payload.get('input') or payload.get('text') or payload.get('url')
    is_valid, err = validate_text(user_input)
    if not is_valid:
        return jsonify({'error': err}), 400

    try:
        result = PhishingQuiz().get_agent_analysis(user_input.strip())
    except Exception as e:
        logger.error("agent_analyze_failed", error=str(e))
        return jsonify({'error': str(e)}), 500
    return jsonify(result)


@app.route('/about')
def about():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user = db.get_user_by_id(session['user_id'])
    return render_template('about.html', user=user)

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user = db.get_user_by_id(session['user_id'])
    # Optional ?source=web|extension filter. Default view = on-site scans only,
    # so the page doesn't open into a wall of background extension page-checks.
    # 'all' shows everything; the template's toggle switches between them.
    source = request.args.get('source', 'web')
    if source == 'all':
        scans = db.get_user_scans(session['user_id'], limit=500)
    else:
        scans = db.get_user_scans(session['user_id'], limit=500, source=source)
    return render_template('history.html', user=user, scans=scans, active_source=source)


@app.route('/scan-details/<int:scan_id>')
def scan_details(scan_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    scan = db.get_scan_details(scan_id, session['user_id'])
    if not scan:
        return jsonify({'error': 'Scan not found'}), 404
    return jsonify(scan)

@app.route('/admin_dashboard')
def admin_dashboard():
    if 'user_id' not in session or session['user_role'] != 'admin':
        return redirect(url_for('login'))

    user = db.get_user_by_id(session['user_id'])
    stats = db.get_admin_stats()

    return render_template('admin_dashboard.html', user=user, stats=stats)

# ============================================================
# JWT API layer - for programmatic clients (browser extension, etc.)
# Separate from the session-based web UI above.
# ============================================================

@app.route('/api/login', methods=['POST'])
@limiter.limit("10 per minute")
def api_login():
    """Authenticate and receive a JWT access + refresh token pair."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid request'}), 400

    email = data.get('email')
    password = data.get('password')
    role = data.get('role', 'user')

    if not email or not password:
        return jsonify({'error': 'Email and password required'}), 400

    user = db.authenticate_user(email, password, role)
    if not user:
        db.log_event('login_failed', detail=f'api login attempt: {email}', ip_address=request.remote_addr)
        return jsonify({'error': 'Invalid credentials'}), 401

    db.log_event('login', user_id=user['id'], detail='api login', ip_address=request.remote_addr)
    access_token, refresh_token = generate_tokens(user['id'], user['role'])
    return jsonify({
        'access_token': access_token,
        'refresh_token': refresh_token,
        'token_type': 'Bearer',
        'expires_in': 1800
    })


@app.route('/api/refresh', methods=['POST'])
@limiter.limit("20 per minute")
def api_refresh():
    """Exchange a valid refresh token for a new access token."""
    data = request.get_json()
    if not data or not data.get('refresh_token'):
        return jsonify({'error': 'Refresh token required'}), 400

    payload = decode_token(data['refresh_token'])
    if payload is None or 'error' in payload:
        msg = payload.get('error', 'Invalid token') if payload else 'Invalid token'
        return jsonify({'error': msg}), 401

    if payload.get('type') != 'refresh':
        return jsonify({'error': 'Wrong token type'}), 401

    user = db.get_user_by_id(payload['user_id'])
    if not user:
        return jsonify({'error': 'User not found'}), 401

    access_token, _ = generate_tokens(user['id'], user['role'])
    return jsonify({
        'access_token': access_token,
        'token_type': 'Bearer',
        'expires_in': 1800
    })


@app.route('/api/predict/text', methods=['POST'])
@limiter.limit("30 per minute")
@jwt_required
def api_predict_text():
    """JWT-protected text prediction for API clients."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid request'}), 400
    text = data.get('text')

    is_valid, error = validate_text(text)
    if not is_valid:
        return jsonify({'error': error}), 400

    result = call_inference_text(text)
    _autosave_scan(getattr(request, 'jwt_user_id', None), 'email', text, result, source='extension')
    return jsonify(result)


@app.route('/api/predict/url', methods=['POST'])
@limiter.limit("30 per minute")
@jwt_required
def api_predict_url():
    """JWT-protected URL prediction for API clients."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid request'}), 400
    url = data.get('url')

    is_valid, error = validate_url(url)
    if not is_valid:
        return jsonify({'error': error}), 400

    result = call_inference_url(url.strip())
    if 'error' in result:
        return jsonify(result), 503
    _autosave_scan(getattr(request, 'jwt_user_id', None), 'url', url.strip(), result, source='extension')
    return jsonify(result)

@app.route('/health')
def health():
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    print("Starting PhishGuard Application...")
    print(f"Inference service: {INFERENCE_URL}")
    print(f"Database initialized: {db is not None}")
    app.run(debug=True, host='0.0.0.0', port=8001)
