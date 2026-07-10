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
- **Input:** raw email text (subject + body), truncated to 256 tokens.
- **Output:** phishing probability in [0, 1]; threshold 0.5.
- **Version:** v2 (retrained). v1 was trained on a 2000s-era spam corpus only.

## Intended use
- **Primary:** flag phishing/spam content in email text and provide a
  SHAP-based, token-level explanation of the decision.
- **Users:** individuals checking suspicious emails; a demonstration of
  explainable phishing detection.
- **Out of scope:** not a replacement for enterprise email security; not a
  guarantee of safety; not designed for languages other than English or for
  detecting malware attachments (text-only).

## Training data
- Original corpus (~5,700 emails, 2000s-era spam/ham) **plus** ~700 modern
  emails collected from a real inbox (Google Takeout), locally parsed and
  anonymized (addresses, numbers, names redacted), and hand-labeled
  (683 legitimate / 13 spam / 14 skipped). Combined pool ~6,400 after dedup.
- The modern data was added specifically to correct distribution shift (see
  Limitations).

## Performance
- **Held-out test set (n = 570):** accuracy 0.997, precision 0.993,
  recall 0.993, F1 0.993 (TN 432, FP 1, FN 1, TP 136).
- **Independent gold set** of 11 hand-collected modern security emails that the
  v1 model false-flagged (81.8% FP): retrained model produces **0/11 false
  positives**, scoring them 0.001–0.004 (was 0.88–0.9997).
- **Held-out modern emails (n = 137):** 0 false positives.

## Limitations
- The gold set is small (11 emails); the correct claim is "0 false positives on
  the emails that previously failed," not a globally-zero rate.
- Modern-spam sample is small (13); the evaluation primarily validates the
  legitimate-mail false-positive fix, not modern-spam recall.
- English-only; text-only (no attachment/header/link analysis in this model).
- A novel sender style absent from training could still be misclassified.

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
