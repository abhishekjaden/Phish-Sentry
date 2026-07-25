# PhishSentry — Model Cards

Model cards for the two production models. Each documents intended use, training
data, performance, limitations, and ethical considerations, following the model-
card convention (Mitchell et al., 2019).

---

# Model Card 1 — Email Phishing Classifier (RoBERTa)

## Model details
- **Architecture:** `roberta-base` fine-tuned for binary sequence classification
  (legitimate vs. phishing/spam).
- **Framework:** PyTorch / HuggingFace Transformers.
- **Input:** raw email text (subject + body). Normalized before inference
  (strip `Subject:`/HTML, URLs -> `httpaddr`, collapse whitespace) to match the
  training distribution, then truncated to 256 tokens. Production preprocessing
  in `inference_service.py` matches this normalization exactly.
- **Output:** phishing probability in [0, 1]; threshold 0.5.
- **Version:** v4.1 (current production). v1 was trained on a 2000s-era spam
  corpus only; v2 added modern legitimate security mail but over-corrected into
  flagging legitimate *transactional* email; v3 fixed both on a real
  multi-corpus dataset; v4.1 added BEC/spear-phishing coverage after measurement
  showed v3 caught only ~25% of it. See EVALUATION.md for the full history and
  the measurement corrections along the way.

## Intended use
- **Primary:** flag phishing/spam content in email text and provide a
  SHAP-based, token-level explanation of the decision.
- **Users:** individuals checking suspicious emails; a demonstration of
  explainable phishing detection.
- **Out of scope:** not a replacement for enterprise email security; not a
  guarantee of safety; not designed for languages other than English or for
  detecting malware attachments (text-only).

## Training data
- **v4.1 (current)** adds to the v3 corpus: ~4,000 generated BEC / spear-phishing
  examples (wire-transfer requests, fake invoices, vendor bank-detail changes,
  portal credential lures) in the phishing class, and ~4,000 additional
  legitimate security-notification phrasings plus expanded transactional
  templates in the legitimate class. The BEC augmentation is **synthetic** --
  real BEC corpora are scarce, which is why the training set lacked them.
  Training used template pool A; evaluation uses structurally disjoint pool B,
  verified programmatically to share no template.
- **v3 base corpus:** a balanced multi-corpus dataset (~160,000 emails, ~50/50
  phishing/legitimate) combining real modern phishing (Nazario, Nigerian/419
  fraud, CEAS-08) and real legitimate mail (SpamAssassin ham, Enron ham, Ling),
  **plus** ~700 hand-labeled modern emails from a real inbox (Google Takeout;
  locally parsed and anonymized) oversampled x15, **plus** 600 curated realistic
  legitimate transactional templates (orders, receipts, statements,
  subscriptions, refunds, bookings). Text is format-normalized before training.
- The transactional templates and oversampling were added specifically to teach
  the model that transactional language is not inherently phishing - the failure
  mode that v2 exhibited (see Limitations / EVALUATION.md).
- *(v1 used only a ~5,700-email 2000s spam corpus; v2 added ~700 modern emails to
  a mostly-legacy pool. Both are documented in EVALUATION.md.)*

## Performance
- **Held-out test (in-distribution, v4.1):** accuracy 0.9964, F1 0.9965.
- **Out-of-distribution evaluation (n=150 per category, 95% Wilson intervals),
  v3 vs v4.1 on the same corpus via `eval_email_models.py`:**

  | Category | v3 | v4.1 |
  |---|---:|---:|
  | Phishing - overt | 76.7% [69, 83] | **92.7%** [87, 96] |
  | Phishing - subtle / BEC | 24.7% [18, 32] | **82.0%** [75, 87] |
  | Legitimate - transactional | 94.0% [89, 97] | 90.7% [85, 94] |
  | Legitimate - security notices | 82.0% [75, 87] | 81.3% [74, 87] |
  | Legitimate - work / personal | 100% [98, 100] | 100% [98, 100] |
  | Legitimate - marketing | 99.2% [95, 100] | 96.7% [92, 99] |
  | *mean phishing recall* | *50.7%* | ***87.3%*** |
  | *mean legitimate passed* | *93.8%* | *92.2%* |

- **These are coverage results, not production rates.** The evaluation emails
  are templated: they measure consistency across a phrasing space and detect
  whole-category failures, but are not a random sample of real mail.
- **Superseded:** the earlier "7/7 phishing, 12/12 legitimate" gate is retained
  in EVALUATION.md as history only. At that size a 0/12 result has a 95% upper
  bound near 24%, and it failed to detect either of the two weaknesses above.

## Limitations
- **Legitimate security notifications are flagged ~18% of the time, in both v3
  and v4.1.** Two-factor codes, password-reset confirmations and new-device
  alerts score as high as p=0.99. This is the most significant open problem: it
  is the same category that caused the v1 failure, was declared fixed on eleven
  emails, and remains substantially unresolved. Genuine security mail and
  credential-harvest phishing share nearly all their vocabulary; what separates
  them is whether the recipient initiated the action, which the text does not
  reveal.
- **BEC detection is validated only against synthetic augmentation.** Passing
  demonstrates generalisation across BEC phrasing structures, not validated
  real-world BEC detection. A BEC email is linguistically near-identical to a
  legitimate internal request; the discriminating evidence is sender identity and
  thread history, which a text-only classifier cannot observe.
- The evaluation sets are templated, so reported rates are coverage results
  rather than production estimates.
- v3 is deliberately aggressive on genuinely ambiguous transactional-phishing
  lures (e.g. "parcel undelivered - pay a customs fee"), which it flags as
  phishing. Arguably the correct bias for a security tool, but a legitimate email
  closely mimicking a known lure structure could occasionally be flagged.
- Modern-spam recall is validated mainly via the phishing corpora; the
  hand-labeled modern-spam sample remains small.
- English-only; text-only (no attachment/header/link analysis in this model).
- **Evaluation history (honest):** v1's test set missed modern security mail
  (81.8% FP found); v2's missed transactional mail (50% FP found); v3 added an
  explicit gate covering both. Each fix was measured, not asserted.

## Ethical considerations
- Training data came from a personal inbox; it was anonymized locally and never
  uploaded to third parties. No email content is shared in the repository
  (excluded via `.gitignore`).
- A false negative (missed phishing) has higher real-world cost than a false
  positive; users are shown probabilities and explanations rather than a bare
  verdict, and are advised the tool is assistive, not authoritative.

---

# Model Card 2 — URL Phishing Classifier (XGBoost)

## Model details
- **Architecture:** XGBoost gradient-boosted trees (binary:logistic), ~600
  trees, on 20 lexical + host-based features.
- **Features:** URL/domain/directory/file length and character counts,
  `time_response`, `asn_ip`, domain activation/expiration age, `ttl_hostname`.
- **Post-processing:** raw score → isotonic calibration curve → threshold 0.5,
  applied at inference.
- **Version:** v2 (retrained with modern legitimate URLs added).

## Intended use
- **Primary:** classify a URL as phishing or legitimate using structural and
  host features, with SHAP feature attributions.
- **Users:** individuals checking suspicious links; the browser extension's
  real-time page check.
- **Out of scope:** not a content/reputation service; does not fetch or render
  page content; not a replacement for Google Safe Browsing or similar.

## Training data
- GregaVrbancic phishing-URL dataset (88,647 URLs, 112 features) reduced to the
  20 deployed features, **plus** 276 hand-collected modern legitimate URLs run
  through the production feature extractor (universities, admissions portals,
  internship boards, cloud/dev tools). Features are used raw (trees need no
  scaling), matching the inference path.

## Performance
- **Held-out test (n = 17,730):** accuracy 0.972, precision 0.956
  (95% CI [0.951, 0.961]), recall 0.963 (95% CI [0.958, 0.968]), ROC-AUC 0.996.
- **Modern legitimate URLs (n = 276):** 1 false positive (0.4%).
- **Baseline:** +23.8 F1 points over a naive length/age heuristic.

## Limitations (measured)
- **Deep-link false positives:** on long, high-entropy URLs (session IDs, UUIDs,
  query strings) on trusted domains, the model false-flags at a measured
  **37.5% (9/24)** — a random hex path structurally resembles obfuscated
  phishing URLs, and the model has no domain-reputation signal to override it.
  Mitigated at the extension layer via a curated reputation allow-list (trusted
  domains checked by domain, not full-URL lexical features); unknown domains
  still get the full check.
- "Young domain = risky" is genuinely predictive, so lowering false positives
  trades against recall on brand-new legitimate domains; the retrain was tuned
  to reduce FP while holding phishing recall at ~0.96.
- Features depend on live WHOIS/DNS/HTTP lookups, which can fail (returning
  sentinel values) and can be slow; timeouts bound the impact.

## Ethical considerations
- The model can false-positive on legitimate but unusual URLs; users see the
  probability and SHAP explanation, and the extension trusts high-reputation
  domains to avoid crying wolf on everyday sites.
- No page content is fetched or stored beyond the URL and extracted features.

---

*Model-card format after Mitchell et al., "Model Cards for Model Reporting"
(FAT* 2019).*
