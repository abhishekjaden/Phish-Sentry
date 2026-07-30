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
- **Current model: v4.1.** Held-out test (in-distribution): accuracy 0.996, F1 0.997.
- **Out-of-distribution evaluation (n=150 per category):** phishing recall
  **87.3%** mean (overt 92.7%, subtle/BEC 82.0%); legitimate mail passed **92.2%**
  mean. Reproducible via `eval_email_models.py`, which scores the current model
  and its predecessor on the same corpus.
- **Known weakness, stated plainly:** ~18% of legitimate *security
  notifications* (2FA codes, password resets, new-device alerts) are flagged as
  phishing. This is unresolved and is the most significant open problem in the
  email model.
- **The retrain story (four iterations, honestly):** the original model, trained
  on a 2000s-era spam corpus, missed modern phishing and false-flagged modern
  legitimate security email (81.8% FP). A first retrain fixed that but
  over-corrected - flagging ~50% of legitimate *transactional* email (order
  confirmations, receipts) as phishing. A second retrain on a real, balanced
  multi-corpus dataset - with format normalization, oversampled legitimate
  transactional mail, and a strict validation gate - resolved both: it catches
  novel phishing and passes legitimate transactional email. A fourth iteration
  followed after broader measurement (n=150 per category rather than 12) showed
  the third model caught only ~25% of business-email-compromise phishing --
  a failure the small validation gate could not detect. The value is that each
  failure mode was found, measured, and closed rather than hidden; and that the
  measurement apparatus itself needed correcting three times along the way. Full
  history in [EVALUATION.md](EVALUATION.md).

### URL model (XGBoost)
- **Held-out test (n = 17,730):** precision 0.956, recall 0.963, ROC-AUC 0.996 --
  but these are **in-distribution** figures. Measured against live phishing feeds,
  recall is **38.3%**. The gap is blindness to phishing hosted on legitimate
  platforms (`*.pages.dev`, `*.blogspot.com`), where every feature the model
  consumes reports benign. Four measured iterations established this as a
  representational ceiling rather than a tuning problem.
- **The browser extension therefore ships a deterministic rule-based lookalike
  detector, not this model** -- 91.5% typosquat recall (post-audit) with structurally zero
  false positives on genuine brand domains. See EVALUATION.md.
- **Modern legitimate URLs (n = 276):** 0.4% false positives.
- **Baseline:** +23.8 F1 points over a naive length/age heuristic.
- **Documented limitation (measured):** the model false-flags deep links (long,
  high-entropy paths like session IDs / UUIDs) at **37.5%**. This is mitigated at
  the extension layer with a reputation allow-list (trusted domains are checked
  by domain, not by the full-URL lexical features); unknown domains still get the
  full model check. It is reported here as a measured residual, not hidden.

Full details in [EVALUATION.md](EVALUATION.md) and [MODEL_CARDS.md](MODEL_CARDS.md).
Reproduce with `eval_email_models.py` (email, current model vs predecessor),
`measure_url_recall.py` (URL model on live phishing) and
`eval_lookalike_detector.py` (the rule-based detector the extension ships).
Earlier scripts (`evaluate_model_v3.py`, `evaluate_model.py`) are retained as
history; their gates were too small to support the claims made from them.

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
evaluate_model_v3.py    Email model evaluation (current v3 - OOD gate + probe)
evaluate_model.py       Earlier email evaluation (v1/v2 held-out + gold set)
evaluate_url_model.py   URL model evaluation (CIs, deep-link characterization,
                        baseline comparison)
EVALUATION.md           Full evaluation report
MODEL_CARDS.md          Model cards (both models)
k8s/                    Kubernetes manifests (services, HPA, monitoring)
models/                 URL model + configs + tokenizer (email .pth: see below)
templates/              Web UI templates
```

**Model weights:** the URL XGBoost model is included (small JSON). The email
RoBERTa weights (~476MB, current v3 model) exceed GitHub's file limit and are
available on request / via HuggingFace.

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
