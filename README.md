# PhishSentry

**An explainable, production-deployed phishing detection platform for email and URLs.**

PhishSentry classifies emails and URLs as phishing or legitimate using machine
learning, explains each decision with SHAP, and runs as a live web application
plus a browser extension. It is the production engineering build-out of an
IEEE-published research paper (iGET 2026, Paper ID 35).

The emphasis of this project is not just building models, but evaluating them
honestly — including finding and fixing real-world failure modes, and measuring
(rather than hiding) the limitations that remain.

---

## What it does

- **Email analysis** — a fine-tuned RoBERTa model classifies email text as
  phishing/spam or legitimate, with a probability and a token-level SHAP
  explanation.
- **URL analysis** — an XGBoost model scores links using 20 lexical and
  host-based features (URL structure, domain age, DNS TTL, ASN, response timing),
  with SHAP feature attributions.
- **Browser extension** — real-time page checking while browsing, with a curated
  reputation allow-list layered over the model to suppress false positives on
  trusted domains.
- **Supporting features** — a Gemini-powered security chatbot (RAG over a
  threat-intel corpus), an interactive phishing-awareness quiz, and per-user
  scan history.

---

## Evaluation (the core of this project)

### Email model (RoBERTa)
- **Held-out test (n = 570):** accuracy 0.997, precision 0.993
  (95% CI [0.977, 1.000]), recall 0.993 (95% CI [0.976, 1.000]).
- **The retrain story:** the original model (trained on a 2000s-era spam corpus)
  false-flagged modern legitimate security email — 81.8% false positives on a
  gold set of 11 real modern security emails. After retraining on ~700 modern
  labeled emails, this dropped to **0/11**, while spam detection was preserved.

### URL model (XGBoost)
- **Held-out test (n = 17,730):** precision 0.956 (95% CI [0.951, 0.961]),
  recall 0.963 (95% CI [0.958, 0.968]), ROC-AUC 0.996.
- **Modern legitimate URLs (n = 276):** 0.4% false positives.
- **Baseline:** +23.8 F1 points over a naive length/age heuristic.
- **Documented limitation (measured):** the model false-flags deep links (long,
  high-entropy paths like session IDs / UUIDs) at **37.5%**. This is mitigated at
  the extension layer with a reputation allow-list (trusted domains are checked
  by domain, not by the full-URL lexical features); unknown domains still get the
  full model check. It is reported here as a measured residual, not hidden.

Full details in [EVALUATION.md](EVALUATION.md) and [MODEL_CARDS.md](MODEL_CARDS.md).
Reproduce with `evaluate_model.py` (email) and `evaluate_url_model.py` (URL).

---

## Architecture

```
                          +--------------+
   Browser extension ---> |              |
   Web UI (Flask) ------> |  app (Flask  | --> inference (FastAPI)
                          |  + Gunicorn) |      - RoBERTa email model
                          |              |      - XGBoost URL model
                          |              |      - SHAP explanations
                          +------+-------+
                                 |
              +------------------+-------------------+
              v                  v                   v
        Celery + Redis      RDS PostgreSQL      rag (FastAPI)
        (async SHAP,        16, Multi-AZ        Gemini + pgvector
         WHOIS/DNS)         (users, scans)      threat-intel chatbot

   Observability: Prometheus + Grafana (request rate, latency p50/p95/p99,
   status codes), alert rules, structured JSON logging (structlog).
```

### Tech stack
- **ML:** RoBERTa (PyTorch / HuggingFace), XGBoost, SHAP, ONNX Runtime
- **Backend:** Flask + Gunicorn, FastAPI (inference + RAG), Celery + Redis
- **Data:** PostgreSQL 16 on Amazon RDS (Multi-AZ), pgvector for RAG
- **Infra:** Kubernetes (AWS EKS), Docker multi-stage builds, AWS ECR,
  Horizontal Pod Autoscaling
- **Observability:** Prometheus, Grafana, structlog (JSON logs with request IDs)
- **Security:** Flask-Limiter (rate limiting), Flask-Talisman (headers), JWT auth
- **Extension:** JavaScript (Manifest V3), two-tier reputation allow-list

---

## Repository layout

```
app.py                  Flask web app (routes, auth, scan history)
inference_service.py    FastAPI model-serving (RoBERTa + XGBoost + SHAP)
rag_service.py          FastAPI RAG chatbot (Gemini + pgvector)
url_features.py         20-feature URL extractor (live WHOIS/DNS/ASN/HTTP)
database.py             SQLAlchemy models + queries
evaluate_model.py       Email model evaluation (with bootstrap CIs)
evaluate_url_model.py   URL model evaluation (CIs, deep-link characterization,
                        baseline comparison)
EVALUATION.md           Full evaluation report
MODEL_CARDS.md          Model cards (both models)
k8s/                    Kubernetes manifests (services, HPA, monitoring)
models/                 URL model + configs + tokenizer (email .pth: see below)
templates/              Web UI templates
```

**Model weights:** the URL XGBoost model is included (small JSON). The email
RoBERTa weights (~476MB) exceed GitHub's file limit and are available on request
/ via HuggingFace.

---

## Running locally

```bash
# 1. clone + create env
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. configure secrets (copy the template, fill in your own values)
cp .env.example .env
#   set FLASK_SECRET_KEY, JWT_SECRET_KEY, GEMINI_API_KEY, DATABASE_URL, etc.

# 3. run the stack (app + inference + rag + redis + postgres)
docker-compose up
```

The app defaults to a local SQLite DB if `DATABASE_URL` is unset. For the full
stack (Postgres, Redis, observability), use `docker-compose up`.

## Deploying to Kubernetes

Manifests are in `k8s/` (numbered in apply order). Secrets are created from your
`.env` — see `k8s/01-secrets.example.yaml` for the required shape. **Never commit
real secrets;** this repo's `.env` and real secret files are gitignored.

---

## Research

This platform builds on a phishing-detection paper accepted at **IEEE iGET 2026**
(Paper ID 35). The models, explainability approach, and evaluation methodology
extend that work into a deployed system.

---

## Status & honest notes

- The system has been deployed on AWS EKS with RDS and observability. To manage
  cost it is not kept running 24/7; it is brought up for testing and demos.
- **User testing:** a beta-user phase is planned but has not yet been conducted.
  Aggregate usage metrics will be added here once it has.
- Limitations are documented honestly in EVALUATION.md and the model cards rather
  than omitted.

## Author

Abhishek Jaden — [github.com/abhishekjaden](https://github.com/abhishekjaden)
