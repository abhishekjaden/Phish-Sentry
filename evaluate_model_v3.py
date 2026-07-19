#!/usr/bin/env python3
"""
PhishSentry — RoBERTa email-model evaluation harness (v3)
=========================================================
Reproduces the v3 model's out-of-distribution claims against the *exact*
production model + preprocessing:

  1. VALIDATION GATE — a novel phishing set (never in training) that the model
     must catch, and a diverse legitimate set that INCLUDES modern transactional
     mail (orders, receipts, statements, subscriptions, refunds, flights) that
     the model must pass. This is the check that v1 and v2 evaluations lacked:
     v1 missed modern security notifications, v2 missed transactional mail.

  2. TRANSACTIONAL FALSE-POSITIVE PROBE — the specific legitimate emails that
     the v2 model flagged as phishing (~50% FP). v3 must pass them.

What this DOES reproduce locally (needs only the model files in models/):
  - the OOD gate (7 phishing / 12 legit) and the transactional probe.
What this does NOT reproduce here:
  - the held-out in-distribution F1 (0.996). That is measured on the ~160k-email
    multi-corpus training split, which lives in the retrain notebook, not on
    disk. It is cited in EVALUATION.md as "from the v3 training run", not as a
    number this script prints — that is deliberate, to keep the doc honest.

Run from the project folder, in the venv with torch/transformers:
    cd "C:\\Users\\ABHISHEK\\Downloads\\Phishing attack detection updated"
    python evaluate_model_v3.py

It loads the model the SAME way inference_service.py does:
    models/roberta_tokenizer , models/roberta_base_local , models/roberta_phishing_model.pth
Pure local eval — no network, no live site.
"""
import os
import re
import html as _html
import sys

import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# ---------------------------------------------------------------------------
# Config — paths match inference_service.py
# ---------------------------------------------------------------------------
MODEL_DIR       = "models"
TOKENIZER_PATH  = os.path.join(MODEL_DIR, "roberta_tokenizer")
BASE_MODEL_PATH = os.path.join(MODEL_DIR, "roberta_base_local")
WEIGHTS_PATH    = os.path.join(MODEL_DIR, "roberta_phishing_model.pth")
MAX_LENGTH      = 256    # same as production
THRESHOLD       = 0.5    # production rule: P(phishing) > 0.5 => phishing

# ---------------------------------------------------------------------------
# EXACT production preprocessing (matches the CURRENT inference_service.py:
# strip Subject:/HTML, URLs -> httpaddr, collapse whitespace, no lowercasing).
# This MUST match training normalization or results are understated.
# ---------------------------------------------------------------------------
def preprocess_email_text(text):
    t = _html.unescape(str(text))
    if '<' in t and '>' in t:
        t = re.sub(r'<[^>]+>', ' ', t)
    t = re.sub(r'^\s*subject\s*:\s*', '', t, flags=re.IGNORECASE)
    t = re.sub(r'https?://\S+|www\.\S+', ' httpaddr ', t)
    t = re.sub(r'\s+', ' ', t)
    return t.strip()[:5000]

# ---------------------------------------------------------------------------
# Test sets (all OUT-OF-DISTRIBUTION — none of these are in any training set)
# ---------------------------------------------------------------------------
NOVEL_PHISHING = [
    "URGENT: Your account has been suspended. Click here to verify your password immediately: http://secure-verify.tk/login",
    "Dear Customer, suspicious login detected. Verify your identity now to avoid permanent suspension: http://account-secure-verify.tk/login",
    "Congratulations! You won $1,000,000. Send your bank account number and social security number to claim.",
    "Your bank account will be suspended. Enter your account number, PIN and online banking password: http://bank-secure-verify.tk/unlock",
    "Your PayPal account has been limited. Confirm your information within 24 hours: http://paypal-verify-account.tk/login",
    "We noticed unusual activity. Verify your card details immediately or your account will be locked.",
    "Your parcel could not be delivered. Pay a small customs fee to release it: http://parcel-release.tk/pay",
]

DIVERSE_LEGIT = [
    # personal / work
    "Hi team, attaching the notes from today's standup. Let me know if I missed anything. Thanks!",
    "Reminder: our 1:1 is scheduled for 3pm tomorrow. Let me know if you need to reschedule.",
    "The quarterly report is ready for review. Let me know your thoughts.",
    "Welcome to the team! Here's your onboarding checklist.",
    "Meeting rescheduled to Friday 2pm. Calendar updated.",
    # transactional / commercial  (the category that broke v2)
    "Your order #29481 has shipped and should arrive Thursday. Track it in your account.",
    "Thanks for your purchase! Your receipt is attached. Total: $45.",
    "Your Amazon package was delivered. Rate your experience.",
    "Your monthly statement is now available in your account.",
    "Your subscription renews on the 15th. No action needed.",
    "Your refund of $45 has been processed and will appear in 3-5 business days.",
    "Your flight AI302 to Delhi is on schedule. Boarding begins at 6:40 PM.",
]

# The exact emails that the v2 model flagged as phishing (~50% FP). v3 must pass.
V2_REGRESSION_PROBE = [
    "Your order #29481 has shipped and should arrive Thursday.",
    "Thanks for your purchase! Your receipt is attached.",
    "Your Amazon package was delivered. Rate your experience.",
    "Your monthly statement is now available in your account.",
    "Your subscription renews on the 15th. No action needed.",
]

# ---------------------------------------------------------------------------
def load_model():
    for p in (TOKENIZER_PATH, BASE_MODEL_PATH, WEIGHTS_PATH):
        if not os.path.exists(p):
            sys.exit(f"[eval] missing required path: {p}\n"
                     f"       run this from the project root with the model files present.")
    device = torch.device("cpu")
    tok = RobertaTokenizer.from_pretrained(TOKENIZER_PATH)
    model = RobertaForSequenceClassification.from_pretrained(BASE_MODEL_PATH, num_labels=2)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.to(device).eval()
    return tok, model, device


def phishing_prob(text, tok, model, device):
    enc = tok(preprocess_email_text(text), truncation=True, max_length=MAX_LENGTH,
              padding="max_length", return_tensors="pt")
    with torch.no_grad():
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        logits = model(input_ids=ids, attention_mask=mask).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return float(probs[1])


def run():
    tok, model, device = load_model()

    print("=" * 68)
    print("PhishSentry v3 email-model evaluation (out-of-distribution)")
    print("=" * 68)

    # --- novel phishing: want prob > THRESHOLD ---
    print("\n[1] NOVEL PHISHING  (want P(phishing) > %.2f)" % THRESHOLD)
    caught = 0
    for t in NOVEL_PHISHING:
        p = phishing_prob(t, tok, model, device)
        ok = p > THRESHOLD
        caught += ok
        print(f"    {'OK  ' if ok else 'MISS'}  {p:0.3f}  {t[:52]}")
    print(f"    -> caught {caught}/{len(NOVEL_PHISHING)}")

    # --- diverse legit (incl. transactional): want prob < THRESHOLD ---
    print("\n[2] DIVERSE LEGIT incl. transactional  (want P(phishing) < %.2f)" % THRESHOLD)
    passed = 0
    for t in DIVERSE_LEGIT:
        p = phishing_prob(t, tok, model, device)
        ok = p < THRESHOLD
        passed += ok
        print(f"    {'OK   ' if ok else 'FP!! '} {p:0.3f}  {t[:52]}")
    print(f"    -> passed {passed}/{len(DIVERSE_LEGIT)}")

    # --- v2 regression probe ---
    print("\n[3] v2 TRANSACTIONAL REGRESSION PROBE  (these broke v2; v3 must pass)")
    probe_pass = 0
    for t in V2_REGRESSION_PROBE:
        p = phishing_prob(t, tok, model, device)
        ok = p < THRESHOLD
        probe_pass += ok
        print(f"    {'OK   ' if ok else 'FP!! '} {p:0.3f}  {t[:52]}")
    print(f"    -> passed {probe_pass}/{len(V2_REGRESSION_PROBE)}")

    # --- gate verdict ---
    gate = (caught >= 0.9 * len(NOVEL_PHISHING)
            and passed >= 0.9 * len(DIVERSE_LEGIT)
            and probe_pass == len(V2_REGRESSION_PROBE))
    print("\n" + "=" * 68)
    print("GATE:", "PASS  (v3 claims reproduced)" if gate else "FAIL")
    print("=" * 68)
    return 0 if gate else 1


if __name__ == "__main__":
    sys.exit(run())
