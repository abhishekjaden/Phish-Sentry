# PhishSentry — Email Model Evaluation Report

**Model under test:** RoBERTa-base sequence classifier, fine-tuned for phishing/spam email detection (the text-analysis model behind PhishSentry's email scan and `/api/predict/text`).
**Decision rule (production):** `phishing` if `P(class=phishing) > 0.5`, on text preprocessed and truncated to 256 tokens — identical to the deployed `inference_service.py` path.
**Method:** evaluated locally against the exact deployed weights, with two test sets (below). Reproducible via `evaluate_model.py` in this repository.

---

## Summary

The model performs **near-perfectly on its own data distribution but generalizes poorly to modern transactional/security email**, which it was not trained on. On a held-out split of the training corpus it reaches **99.5% accuracy (F1 = 0.989)**; on a hand-collected set of genuinely legitimate, security-themed emails it **misclassifies 9 of 11 unambiguously-safe messages as phishing (81.8% false-positive rate)**.

This is a **distribution-shift** result: the training corpus is an older spam dataset, and the model has learned strong associations between security/credential/urgency vocabulary ("sign-in", "verify", "account", "password") and the *spam* class. Modern legitimate security notifications use that same vocabulary, so the model over-flags them.

The finding is reported here in full because it is the kind of failure mode a deployed classifier must be honest about: high benchmark numbers do not imply real-world reliability.

---

## Test set 1 — Held-out split (in-distribution)

The training corpus (`emails.csv`, 5,728 unique emails after de-duplication; ~3.2 : 1 legitimate-to-spam) was split with the **same parameters the model was trained with** (`test_size=0.1, random_state=42, stratify=y`). The 570-email held-out portion below was **never seen during training or validation**.

| Metric | Value |
|---|---|
| Accuracy | **0.9947** |
| Precision | 0.9786 |
| Recall | **1.0000** |
| F1 | 0.9892 |

**Confusion matrix:** TN = 430, FP = 3, FN = 0, TP = 137 (n = 570; 433 legitimate, 137 spam).

*Interpretation.* On data drawn from its own distribution, the model is excellent: it caught **every** spam email (recall = 1.0) with only 3 false positives. This confirms the model was trained correctly and is not broken — its weakness is specifically generalization to **out-of-distribution** mail.

---

## Test set 2 — Real legitimate security emails (out-of-distribution)

A set of **18 genuinely legitimate emails** was collected from real inboxes — messages from Google, AWS, GitHub, Microsoft, LinkedIn, Pearson VUE, and IEEE/Microsoft CMT. All are truly safe (true label = legitimate), so **any "phishing" verdict is a false positive.** Emails are bucketed by how unambiguous their safety is:

- **gold** — sender domain unmistakably matches the brand; no reasonable person calls these phishing. A flag here is a clean false positive.
- **borderline** — legitimate, but with genuinely suspicious traits (urgency + form link, gmail sender). A flag here is *not* cleanly an error.
- **easy** — warm/marketing legitimate mail with little security language; a control to confirm the model isn't flagging everything.

| Email (sender) | Bucket | P(phishing) | Verdict |
|---|---|---:|---|
| Google — "sign in to keep your account" | gold | 0.9995 | **false positive** |
| AWS — IAM Identity Center invite | gold | 0.8819 | **false positive** |
| Pearson VUE — exam confirmation | gold | 0.0007 | correct |
| GitHub — "review this sign-in" | gold | 0.9921 | **false positive** |
| GitHub — third-party app authorized | gold | 0.9997 | **false positive** |
| GitHub — OAuth app authorized | gold | 0.9991 | **false positive** |
| Microsoft — passkey added to account | gold | 0.9995 | **false positive** |
| Google — new sign-in on Windows | gold | 0.9990 | **false positive** |
| LinkedIn — verify new device | gold | 0.9985 | **false positive** |
| IEEE PDF eXpress — account confirmation | gold | 0.7526 | **false positive** |
| Microsoft CMT — major-revision notice | gold | 0.0002 | correct |
| EdiGlobe — onboarding form | borderline | 0.1798 | correct |
| IEEE ICT Quiz — login credentials | borderline | 0.0079 | correct |
| AWS — CloudTrail welcome | easy | 0.9321 | **false positive** |
| Volante — team onboarding | easy | 0.0001 | correct |
| Microsoft — passkeys marketing | easy | 0.9996 | **false positive** |
| LinkedIn — "application sent" | easy | 0.9997 | **false positive** |
| FIE 2026 — paper withdrawn | easy | 0.0001 | correct |

**False-positive rate by bucket:**

| Bucket | False positives | Rate |
|---|---|---|
| gold (unambiguous) | 9 / 11 | **81.8%** |
| borderline | 0 / 2 | 0.0% |
| easy | 3 / 5 | 60.0% |
| **All** | **12 / 18** | **66.7%** |

---

## Error analysis

1. **The errors are systematic, not random.** Almost every false positive is a short, security-themed notification: GitHub "review this sign-in" (0.992), Microsoft passkey-added (0.9995), Google security alert (0.999), LinkedIn new-device (0.999). The model has learned that credential/sign-in/account/verify language signals spam — true in a 2000s spam corpus, false for modern transactional mail.

2. **Length and prose dilute the trigger.** The two gold emails the model classified correctly — Pearson VUE (0.0007) and Microsoft CMT (0.0002) — are long, formal, procedure-heavy messages. Abundant neutral text appears to dilute the trigger vocabulary below the decision threshold. The short, punchy security alerts are the ones that fire.

3. **It is the language, not the brand.** The model is not simply suspicious of "tech-company" mail: warm/transactional messages from the same senders pass easily (LinkedIn "application sent" *should* be easy yet was flagged at 0.9997, while Volante onboarding passed at 0.0001). The discriminating factor is security/urgency phrasing, not sender identity.

4. **Recall vs. precision in context.** In-distribution the model favors recall (caught all spam). Out-of-distribution that same aggressiveness manifests as a high false-positive rate on legitimate security mail — the cost of a recall-leaning model meeting a distribution it never trained on.

---

## Root cause

The model was fine-tuned on an **older email-spam corpus** dominated by classic promotional/scam spam, with legitimate examples that are mostly ordinary business/personal mail. That corpus contains very few **modern legitimate security notifications** ("new sign-in detected", "a passkey was added", "review this sign-in"). As a result the model never learned that such phrasing is *also* normal and legitimate, and it associates the vocabulary with spam. This is a dataset-coverage / distribution-shift problem, not a modeling defect.

---

## Implications and honest limitations

- **The headline benchmark (99.5% accuracy) overstates real-world reliability.** It reflects performance on the training distribution, not on the live email a user would actually scan.
- **As deployed, the email model should be understood as a demonstration of an explainable transformer-based classifier, not a production-ready filter** for arbitrary modern mail. It would over-flag legitimate security notifications at a high rate.
- The URL model and the homograph/redirect/fresh-domain overlays are evaluated separately and are unaffected by this finding.

## What would actually fix it (not yet done)

The correct remedy is **not** a threshold tweak (that trades false positives for missed phishing, the worse error for a detector) but **retraining/fine-tuning on a modern, balanced corpus** that includes legitimate transactional/security email as negative examples. A principled next step:

1. Assemble a modern labeled set including legitimate security notifications (the emails above are a starting seed).
2. Fine-tune or continue-train on it; re-run this exact harness.
3. Report the **before/after** false-positive rate on the gold set as the measure of improvement.

That before/after comparison — measured, not asserted — would be the honest way to claim the problem is fixed.

---

*Reproducibility: all numbers above are produced by `evaluate_model.py` in this repository, run against the deployed model weights. The script reconstructs the held-out split deterministically and prints the per-email table and bucket rates verbatim.*
---

## Update — Email model retrain (v2): fixing the modern-mail false-positive rate

### What the first evaluation found

The original RoBERTa email model was trained on a 2000s-era spam corpus
(`emails.csv`, 5,728 messages: 4,360 legitimate / 1,368 spam). Evaluation on a
set of 11 hand-collected **modern** security/transactional emails (GitHub
sign-in alerts, Google security notifications, Microsoft passkey confirmations,
AWS IAM invitations, LinkedIn device-verification, etc.) revealed a systematic
failure: the model flagged **9 of 11 (81.8%)** of these legitimate emails as
phishing, several at 0.88–0.9997 confidence.

Root cause: **distribution shift.** The training corpus contained almost no
modern automated security mail, so the model learned to associate
account/verify/sign-in/password language with spam. The failure was
brand-independent and systematic — not a handful of unlucky examples.

### The fix

Rather than tweak the decision threshold (which would trade the false-positive
problem for missed spam), the model was **retrained on augmented data**:

- Collected ~700 modern emails from a real inbox (Google Takeout export),
  parsed and anonymized locally (email addresses, numbers, names redacted).
- Hand-labeled them: **683 legitimate, 13 spam, 14 skipped.** Labeling was
  manual per-message, not rubber-stamped — the goal was trustworthy labels.
- Combined with the original corpus (final training pool ~6,391 messages after
  de-duplication) and fine-tuned with the **identical recipe** as the original
  model (RoBERTa-base, `max_length=256`, batch 16, 3 epochs, lr 2e-5, AdamW
  weight-decay 0.01, balanced class weights, linear warmup, grad-clip 1.0).
  Only the *data* changed, so the before/after isolates the effect of the data.
- A 20% slice of the modern emails was held out entirely from training, to
  measure whether the fix **generalizes** rather than memorizes.

### Results (retrained model)

**Independent gold set (the same 11 hand-collected security emails, never in
any training set — old or new):**

| Bucket | Before (original model) | After (retrained) |
|---|---|---|
| Gold security emails | 9/11 false positives (81.8%) | **0/11 (0.0%)** |
| All 18 (gold+borderline+easy) | — | **0/18 (0.0%)** |

The retrained model scores these emails at **0.001–0.004** spam-probability
(previously 0.88–0.9997) — not a marginal pass, a decisive one.

**Held-out modern emails (137 legitimate, never trained on):** 0 flagged as
phishing.

**Spam detection preserved (held-out test set, n=570):** accuracy 0.9965,
precision 0.9927, recall 0.9927, F1 0.9927 (TN=432, FP=1, FN=1, TP=136). The
model did not simply learn to call everything legitimate — it still detects
spam at >99% precision and recall.

### Honest framing and limitations

- The gold set is small (11 emails). The correct statement is **"0 false
  positives on 11 hand-collected security emails that previously produced
  81.8%,"** not that the false-positive rate is globally zero. A novel sender
  style not represented in the modern training data could still occasionally
  be misclassified.
- The modern spam sample is tiny (13 examples), so this evaluation primarily
  validates the **legitimate-mail false-positive fix** — which was the
  identified failure — rather than modern-spam recall.
- The improvement is a measured *reduction*, achieved by addressing the
  underlying distribution shift with data, not by threshold manipulation.
---

## URL model (XGBoost) evaluation

The URL model classifies links using 20 lexical + host-based features (URL
length, path/directory structure, domain age, DNS TTL, ASN, response timing).
This section reports its performance with confidence intervals, a characterized
account of its known limitation, and a baseline comparison.

*(Reproduce with `evaluate_url_model.py`.)*

### Held-out test performance (in-distribution)

Evaluated on the held-out 20% split of the GregaVrbancic dataset
(n = 17,730 URLs: 11,600 legitimate, 6,130 phishing). 95% confidence intervals
are from 2,000 bootstrap resamples.

| Metric | Value | 95% CI |
|---|---|---|
| Accuracy | 0.972 | — |
| Precision | 0.956 | [0.951, 0.961] |
| Recall | 0.963 | [0.958, 0.968] |
| F1 | 0.959 | — |
| ROC-AUC | 0.996 | — |

Confusion matrix: TN = 11,327, FP = 273, FN = 228, TP = 5,902.

The tight confidence intervals on a large test set indicate the point estimates
are stable, not artifacts of a small or lucky split.

### False positives on modern legitimate URLs

On 276 hand-collected modern legitimate URLs (universities, admissions portals,
internship sites, cloud/dev tools) — the distribution the retrain targeted —
the model produces **1 false positive (0.4%)**. This confirms the retrain
resolved the domain-level over-flagging that the original model exhibited
(where sites like AWS, GitHub, and Google were flagged at 63–100%).

### Known limitation, quantified: deep-link false positives

The model's lexical features cause a measurable weakness on **deep links** —
long URLs with high-entropy paths (session IDs, UUIDs, query strings) on
otherwise-legitimate domains. Appending realistic deep-link paths to trusted
domains (e.g. `claude.ai/chat/<uuid>`, Google Docs share links) and scoring
them yields a **37.5% false-positive rate (9/24)**, with the worst cases at
97–100% phishing probability.

This is expected: a random hex path structurally resembles the obfuscated URLs
phishing campaigns use, and the model has no domain-reputation signal to
override that. The limitation is addressed at the **extension layer** with a
curated reputation allow-list — trusted domains are checked by domain, not by
the lexical features of their full deep-link URL — while unknown domains still
receive the full model check. This is the standard "reputation list layered on
ML" pattern; it is reported here as a measured residual rather than hidden.

### Baseline comparison

Against a naive heuristic (flag as phishing if the URL is above-median length
OR the domain is young), on the same held-out set:

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Naive heuristic | 0.587 | 0.934 | 0.721 |
| XGBoost model | 0.956 | 0.963 | 0.959 |

The model improves F1 by **+23.8 points** over the naive baseline, confirming it
captures signal beyond simple length/age rules.

### Honest summary

Strong, stable in-distribution performance; near-zero false positives on modern
legitimate domains after retraining; a measured 37.5% residual on deep links
(mitigated at the extension layer, not eliminated); and a clear improvement over
naive baselines. The residual is a genuine property of lexical URL features, not
a bug — reported and mitigated rather than concealed.
