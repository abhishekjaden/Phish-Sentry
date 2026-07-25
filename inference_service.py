"""
PhishGuard Inference Microservice
Holds the ML models and exposes prediction endpoints over HTTP.
The Flask app calls this service instead of loading models itself.
"""
import os
import re
import json
import numpy as np
import torch
import shap
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from tracing_config import setup_tracing
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from url_features import extract_features
import homograph     # Day 29: IDN / homograph lookalike detection (heuristic overlay)
import url_resolver  # Day 29: shortener / redirect expansion to the true destination


app = FastAPI(title="PhishGuard Inference Service")
setup_tracing("phishguard-inference")
FastAPIInstrumentor.instrument_app(app)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[inference] Loading models on device: {device}")

# ---- Load text model (RoBERTa) ----
try:
    email_tokenizer = RobertaTokenizer.from_pretrained('models/roberta_tokenizer')
    email_model = RobertaForSequenceClassification.from_pretrained('models/roberta_base_local', num_labels=2)
    email_model.load_state_dict(torch.load('models/roberta_phishing_model.pth', map_location=device))
    email_model.to(device)
    email_model.eval()
    print("[inference] Email model loaded successfully!")
except Exception as e:
    print(f"[inference] Error loading email model: {e}")
    email_model = None
    email_tokenizer = None

# ---- Load URL model (XGBoost, trained in-house) + serving config ----
try:
    _here = os.path.dirname(__file__)
    with open(os.path.join(_here, 'models', 'url_model_config.json')) as _f:
        URL_CFG = json.load(_f)
    url_booster = xgb.Booster()
    url_booster.load_model(os.path.join(_here, 'models', 'url_xgb_model.json'))
    URL_FEATURES = URL_CFG['features']
    URL_THRESHOLD = float(URL_CFG['threshold'])
    URL_CALIB_X = np.array(URL_CFG['calibration']['x'], dtype=float)
    URL_CALIB_Y = np.array(URL_CFG['calibration']['y'], dtype=float)
    print("[inference] URL XGBoost model loaded successfully!")
except Exception as e:
    print(f"[inference] Error loading URL model: {e}")
    url_booster = None


def preprocess_email_text(text):
    # NOTE: must match the normalization used during model training
    # (strip Subject: prefix + HTML, replace URLs with httpaddr, collapse whitespace).
    # Do NOT lowercase and do NOT delete URLs/punctuation, or the model sees
    # a different distribution than it was trained on.
    import html as _html
    t = _html.unescape(str(text))
    if '<' in t and '>' in t:
        t = re.sub(r'<[^>]+>', ' ', t)
    t = re.sub(r'^\s*subject\s*:\s*', '', t, flags=re.IGNORECASE)
    t = re.sub(r'https?://\S+|www\.\S+', ' httpaddr ', t)
    t = re.sub(r'\s+', ' ', t)
    return t.strip()[:5000]


# ---- SHAP explainer for the text model ----
# Built lazily on first use because it's expensive to construct.
_text_explainer = None

def get_text_explainer():
    """Build (once) and return a SHAP explainer wrapping the RoBERTa model.
    model_predict is BATCHED (one forward pass over all perturbations) so SHAP
    is fast enough for interactive use."""
    global _text_explainer
    if _text_explainer is not None:
        return _text_explainer
    if not email_model or not email_tokenizer:
        return None

    def model_predict(texts):
        # texts: list/array of strings -> array of phishing probabilities.
        # BATCHED: tokenize all at once and run a single forward pass (chunked),
        # instead of a Python loop of one-at-a-time full-length passes.
        import numpy as _np
        texts = [str(t) for t in texts]
        out_probs = []
        BATCH = 32
        for i in range(0, len(texts), BATCH):
            chunk = texts[i:i + BATCH]
            enc = email_tokenizer(
                chunk, truncation=True, padding=True,
                max_length=64, return_tensors='pt'
            )
            with torch.no_grad():
                input_ids = enc['input_ids'].to(device)
                attention_mask = enc['attention_mask'].to(device)
                logits = email_model(input_ids=input_ids,
                                     attention_mask=attention_mask).logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()[:, 1]
            out_probs.extend(probs.tolist())
        return _np.array(out_probs)

    masker = shap.maskers.Text(email_tokenizer)
    _text_explainer = shap.Explainer(model_predict, masker)
    return _text_explainer


def _clean_tok(tok):
    """Strip RoBERTa's space marker and whitespace so words render cleanly."""
    return str(tok).replace("\u0120", "").replace("Ġ", "").strip()


def _is_real_word(w):
    """Keep only clean, meaningful words for display. Drops mangled-unicode
    tokens, pure numbers, and short sub-word fragments that read as noise."""
    if not w or len(w) < 3:
        return False
    # must be mostly ASCII letters (drops mangled unicode like 'âĢĵ')
    if not all(ord(c) < 128 for c in w):
        return False
    # drop pure numbers / tokens with no alphabetic content
    if not any(c.isalpha() for c in w):
        return False
    # drop all-caps sub-word fragments of <=3 chars (e.g. 'ENT', 'UR')
    if len(w) <= 3 and w.isupper():
        return False
    return True


def compute_text_shap(text, max_words=20):
    """Compute token-level SHAP values, fast, in the shape the frontend reads.

    Returns top_risky_words / top_safe_words as [word, value] pairs (what
    results.html expects), plus legacy risky_tokens/safe_tokens for compat.
    """
    explainer = get_text_explainer()
    if explainer is None:
        return {"error": "Explainer not available"}

    processed = preprocess_email_text(text)
    words = processed.split()
    if len(words) > max_words:
        processed = " ".join(words[:max_words])

    try:
        # max_evals=80: balanced quality/speed (~10-12s on CPU).
        shap_values = explainer([processed], max_evals=80)
        tokens = shap_values.data[0]
        values = shap_values.values[0]

        token_scores = []
        for tok, val in zip(tokens, values):
            w = _clean_tok(tok)
            if _is_real_word(w):
                token_scores.append((w, float(val)))

        risky_pairs = sorted([t for t in token_scores if t[1] > 0],
                             key=lambda x: x[1], reverse=True)[:5]
        safe_pairs = sorted([t for t in token_scores if t[1] < 0],
                            key=lambda x: x[1])[:5]

        # Normalize to relative importance (0-100) for meaningful display.
        # Raw SHAP magnitudes are tiny (model is very confident), so we scale
        # each side against its own max |value|. This is the true relative
        # ranking, just rescaled for readability -- not a fabricated number.
        max_risky = max([v for (_, v) in risky_pairs], default=0.0)
        max_safe = max([abs(v) for (_, v) in safe_pairs], default=0.0)

        def rel(v, m):
            return round(100.0 * abs(v) / m, 1) if m > 0 else 0.0

        # frontend reads [word, value]; we make value the 0-100 relative weight
        # (results.html shows this directly), and keep the raw value separately.
        top_risky = [[w, rel(v, max_risky)] for (w, v) in risky_pairs]
        top_safe = [[w, rel(v, max_safe)] for (w, v) in safe_pairs]

        return {
            "status": "success",
            # frontend-shaped ([word, relative_importance_0_100] pairs):
            "top_risky_words": top_risky,
            "top_safe_words": top_safe,
            # raw SHAP values kept for transparency / compat:
            "risky_tokens": [{"token": w, "shap_value": v} for (w, v) in risky_pairs],
            "safe_tokens": [{"token": w, "shap_value": v} for (w, v) in safe_pairs],
        }
    except Exception as e:
        return {"error": f"SHAP computation failed: {str(e)}"}


# ---- Request models ----
class TextRequest(BaseModel):
    text: str

class URLRequest(BaseModel):
    url: str


# ---- Endpoints ----
@app.get("/health")
def health():
    return {
        "status": "ok",
        "text_model": email_model is not None,
        "url_model": url_booster is not None
    }


@app.post("/predict/text")
def predict_text(req: TextRequest):
    if not email_model or not email_tokenizer:
        return {"error": "Email model not available"}
    try:
        processed_text = preprocess_email_text(req.text)
        encoding = email_tokenizer(
            processed_text, truncation=True, padding='max_length',
            max_length=256, return_tensors='pt'
        )
        with torch.no_grad():
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            outputs = email_model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

        is_phishing = bool(probabilities[1] > 0.5)
        confidence = float(probabilities[1] * 100 if is_phishing else probabilities[0] * 100)

        return {
            "is_phishing": is_phishing,
            "confidence": confidence,
            "prediction_value": float(probabilities[1])
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/predict/url")
def predict_url(req: URLRequest):
    if url_booster is None:
        return {"error": "URL model not available"}
    try:
        # 0. Expand shorteners / follow redirects to the TRUE destination, then
        #    classify that. Phishers hide behind bit.ly-style links and redirect
        #    chains; judging the raw link would miss the real target. Resolution
        #    is hard-bounded and SSRF-guarded (see url_resolver).
        resolved = url_resolver.resolve_final_url(req.url)
        target_url = resolved["final_url"]

        # 1. Extract the 20 features on the FINAL url (same order the model trained on)
        feats = extract_features(target_url)
        dm = xgb.DMatrix(np.array([feats], dtype=float), feature_names=URL_FEATURES)

        # 2. Raw model score -> calibrated probability -> thresholded decision
        raw = float(url_booster.predict(dm)[0])
        calibrated = float(np.interp(raw, URL_CALIB_X, URL_CALIB_Y))
        model_is_phishing = bool(calibrated >= URL_THRESHOLD)

        # 3. Heuristic overlay: IDN / homograph lookalike on the FINAL domain.
        #    The trained model's length/character features cannot capture
        #    script-confusion attacks (e.g. Cyrillic "аpple.com"), so a confirmed
        #    homograph overrides the verdict to phishing. The model's own call is
        #    kept in `model_is_phishing` for transparency.
        homo = homograph.analyze(target_url)

        is_phishing = model_is_phishing
        reason = None
        confidence = calibrated * 100 if model_is_phishing else (1.0 - calibrated) * 100

        if homo["is_homograph"]:
            is_phishing = True
            reason = f"Homograph/IDN lookalike domain — {homo['reason']}"
            confidence = 99.0

        # 4. Fresh-domain risk overlay. The model already consumes domain age as
        #    a feature; here we additionally surface a clear flag for very
        #    recently registered domains (a common phishing trait). It is
        #    informational and does NOT by itself flip the verdict, to avoid
        #    penalising the many legitimate newly-registered domains.
        FRESH_DAYS = 30
        try:
            age_days = float(feats[URL_FEATURES.index("time_domain_activation")])
        except Exception:
            age_days = -1.0
        is_fresh = 0 <= age_days < FRESH_DAYS
        fresh_domain = {
            "is_fresh": bool(is_fresh),
            "age_days": int(age_days) if age_days >= 0 else None
        }

        return {
            "is_phishing": is_phishing,
            "confidence": float(confidence),
            "prediction_value": calibrated,      # honest model probability
            "model_is_phishing": model_is_phishing,
            "homograph": homo,                   # {is_homograph, is_idn, reason, decoded}
            "resolved": resolved,                # {original_url, final_url, redirected, redirect_chain, was_shortener, blocked}
            "fresh_domain": fresh_domain,        # {is_fresh, age_days}
            "reason": reason                      # set when a heuristic changed the verdict
        }
    except Exception as e:
        return {"error": str(e)}


class ExplainRequest(BaseModel):
    text: str

@app.post("/explain/text")
def explain_text(req: ExplainRequest):
    return compute_text_shap(req.text)
