# PhishSentry — Email Model Evaluation Report

**Model under test:** RoBERTa-base sequence classifier, fine-tuned for phishing/spam email detection (the text-analysis model behind PhishSentry's email scan and `/api/predict/text`).

> **Current production model: v4.1 (see the "Update - v4.1" section below).**
> This report is written as a full history. The evaluations proceed
> v1 -> v2 -> v3 -> v4.1; **only v4.1 is the deployed model.** Earlier sections
> are retained because the failures they document are what drove each subsequent
> retrain.
>
> **Two corrections to earlier claims in this document.** First, the v3
> validation gate (7 phishing / 12 legitimate) was too small to support the
> conclusions drawn from it -- 0/12 has a 95% upper bound near 24%. Re-measuring
> at n=150 per category exposed two failures it could not detect. Second, the
> measurement harness itself required three corrections before it produced
> trustworthy numbers; those are documented in the v4.1 section rather than
> quietly fixed.
>
> If you only read one section, read **Update - v4.1**. Current claims are
> reproducible via `eval_email_models.py`.
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

*Reproducibility (email model): the v1 held-out split and v1/v2 gold-set numbers are produced by `evaluate_model.py` against the weights that were deployed at the time. The **v3** out-of-distribution gate and v2 regression probe are reproduced by `evaluate_model_v3.py` against the current production weights and preprocessing. The v3 in-distribution held-out metrics come from the v3 training run (`phishsentry_email_retrain_v3.ipynb`) and are not reproduced by the local script, which does not carry the 160k-email multi-corpus split.*
---

## Update — Email model retrain (v2): fixing the modern-mail false-positive rate

> ⚠️ **Superseded by v3.** The v2 model described in this section was later found
> to flag ~50% of legitimate *transactional* email as phishing - a failure its
> (security-notification-only) test set did not surface. See "Update - v3" below
> for the resolution. This section is kept as the historical record of the second
> iteration.

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

## Update - Email model retrain (v3): fixing the transactional-mail false positives that v2 introduced

> **Read this before trusting the v2 numbers above.** The v2 results (0/18 false
> positives) were measured on a gold set composed almost entirely of *security
> notifications* (sign-in alerts, passkey confirmations, device verification).
> That set did **not** contain *transactional/commercial* legitimate mail - order
> confirmations, receipts, shipping notices, account statements, subscription
> renewals. Broader testing showed that v2, while it fixed the
> security-notification problem, had **overcorrected**: it flagged roughly
> **50% of legitimate transactional email as phishing**. The v2 evaluation missed
> this because its test set did not cover that category. **v3 is the model that
> actually resolves it, and it is the current production model.**

### What v2's evaluation missed

Testing v2 on a broader legitimate set exposed a systematic failure on
transactional mail (all messages below are genuinely legitimate; any "phishing"
verdict is a false positive):

| Legitimate email (v2 model) | P(phishing) | Verdict |
|---|---:|---|
| "Your order #29481 has shipped, arrives Thursday" | 0.963 | **false positive** |
| "Thanks for your purchase! Receipt attached" | 1.000 | **false positive** |
| "Your Amazon package was delivered" | 1.000 | **false positive** |
| "Your monthly statement is now available" | 1.000 | **false positive** |
| "Your subscription renews on the 15th" | 1.000 | **false positive** |
| Personal/work mail (standup notes, 1:1 reminder) | 0.000 | correct |

**5 of 10 legitimate emails flagged (50%).** Root cause: phishing corpora are
saturated with *fake* transactional lures ("your order shipped - click to
track", "confirm your payment"), while v2's legitimate class was dominated by
older personal/business mail with little modern transactional content. The model
learned **"transactional vocabulary -> phishing."** It could not distinguish a
real order confirmation from a phishing one - it keyed on the vocabulary, not the
intent. This is the mirror image of the v1 failure: v1 called everything
legitimate on out-of-distribution mail; v2 called legitimate transactional mail
phishing.

### The v3 fix

The remedy was a proper multi-corpus retrain with an explicit anti-false-positive
design and a hard validation gate:

- **Real, diverse training data** - a multi-corpus phishing dataset (Nazario
  real-phishing, Nigerian/419 fraud, CEAS-08, SpamAssassin, Enron ham, Ling),
  ~160,000 emails, balanced ~50/50 phishing/legitimate. This replaced the single
  legacy spam corpus, giving the model *real modern phishing* to learn from
  rather than 2000s marketing spam.
- **Format normalization** - strip `Subject:` prefixes and HTML, replace URLs
  with a neutral `httpaddr` token - so the model learns content, not
  dataset-specific formatting artifacts. The production `inference_service.py`
  preprocessing was updated to match this normalization exactly, closing a
  train/serve skew.
- **Legit-transactional representation** - oversampled the hand-labeled modern
  legitimate emails x15, and added 600 curated realistic legitimate transactional
  templates (order shipped, receipt, statement, subscription, refund, booking,
  flight) - directly teaching the model that transactional language is *not*
  inherently phishing.
- **Class-weighted loss** to prevent collapse toward either class.
- **A strict validation gate:** the model is saved only if it catches **>=90% of a
  novel out-of-distribution phishing set AND passes >=90% of a diverse legitimate
  set that explicitly includes transactional mail.** This gate exists precisely
  because the v1 and v2 evaluations had each missed a failure category.

### Results (v3 - current production model)

**Out-of-distribution validation gate** (none of these appear in any training
set), reproducible locally via `evaluate_model_v3.py`:

| Set | Result |
|---|---|
| Novel phishing (credential-harvest, fraud, fake-transactional lures) | **7/7 caught (100%)** |
| Diverse legitimate - personal, work, **and transactional** (orders, receipts, statements, subscriptions, refunds, flights) | **12/12 passed (100%)**, all transactional at ~0.000 |
| v2 transactional regression probe (the exact emails that broke v2) | **5/5 passed** |

**Held-out split (in-distribution, multi-corpus test set)** - from the v3
training run (`phishsentry_email_retrain_v3.ipynb`; not reproduced by the local
script): Accuracy 0.9969, Precision 0.9965, Recall 0.9973, F1 0.9969.

**Live-UI spot check (real inbox mail, through the deployed site):** real
legitimate mail - a Volante internship offer, an FIE 2026 camera-ready notice, a
Bluestock congratulations - all correctly classified **legitimate**; an
unambiguous phishing email correctly flagged **phishing at 100%**. The specific
transactional case that broke v2 ("Amazon order shipped") returns **legitimate
(P(phishing) ~ 0.00002)** through the production endpoint.

### The honest meta-lesson

Each iteration's evaluation was only as good as its test set's coverage:

- **v1** - eval missed modern *security-notification* mail -> 81.8% FP discovered.
- **v2** - eval missed modern *transactional/commercial* mail -> 50% FP discovered.
- **v3** - added an explicit validation gate covering *both* categories, and will
  not save a model that fails either.

The value here is not that the final model is flawless - it is that each failure
mode was found, measured, and closed rather than asserted away, and that the
evaluation methodology itself was hardened in response to each miss.

### Remaining honest limitations (v3)

- The out-of-distribution validation sets are modest (single- to low-double-digit
  counts per category). They demonstrate the specific failure modes are closed,
  not that the false-positive rate is globally zero.
- v3 is deliberately **aggressive on genuinely ambiguous transactional-phishing
  lures** (e.g. "your parcel couldn't be delivered - pay a customs fee"), which it
  flags as phishing. For a security tool this is arguably the correct bias, but a
  legitimate email that closely mimics a known lure structure could occasionally
  be flagged.
- Modern-spam recall is validated primarily through the phishing corpora; the
  hand-labeled modern-spam sample remains small.

---

## Update - Email model retrain (v4.1): BEC coverage, and correcting the measurement

> **The v3 gate was too small to be trusted.** It reported 7/7 novel phishing
> caught and 12/12 diverse legitimate passed, and that was taken as evidence the
> model was sound. Re-measuring at n=150 per category exposed two failures the
> gate could not have detected: near-blindness to business email compromise, and
> a false-positive rate on legitimate security notifications that had been
> reported as fixed since v1.

### Why the previous numbers could not support their claims

The v3 gate contained 7 phishing, 12 legitimate, and a 5-email regression probe.
It was built to confirm two *specific* previously-observed failures had been
closed, and it did that correctly. It was then treated as general evidence of
model quality, which at that size it could not provide.

Two categories it never covered: **business email compromise / spear phishing**
(wire-transfer requests, fake invoices, vendor bank-detail changes, portal
credential lures), and **legitimate security notifications tested broadly**.

### Three corrections to the measurement harness

Building a trustworthy harness took three attempts. Each earlier version
produced confident numbers that were wrong, which is worth recording because the
measurement apparatus proved as easy to get wrong as the model:

| Harness | Defect | Effect |
|---|---|---|
| v3 gate (7/12/5) | Too small | CI too wide to support any rate claim |
| First expanded harness | Brands assigned to content at random | Generated incoherent mail ("ICICI Bank -- your 1:1 is at 9am") and counted the model's correct reaction as a false positive |
| Colab training gate | Scored bare template bodies, no subject line | Not comparable to the local harness; an early side-by-side using both was invalid |

The current harness (`eval_email_models.py`) pairs brands only with categories
they plausibly send, gives work mail no brand at all, varies subject lines per
category, and **scores both checkpoints on the identical generated corpus** so
differences are attributable to the model rather than to sampling.

**What it can and cannot claim.** The emails are templated. They measure whether
the model is consistent across a phrasing space and whether whole categories
fail. They are **not** a random sample of real mail, so these are coverage
results, not production estimates. A model can score well here and still fail on
phrasings the templates do not contain -- which is exactly how the v1, v2 and v3
evaluations each went wrong in turn.

### v3 versus v4.1

Same corpus, same harness, both checkpoints scored in one run. n=150 per
category (120 for marketing), 95% Wilson intervals.

| Category | v3 | 95% CI | v4.1 | 95% CI | delta |
|---|---:|---|---:|---|---:|
| Phishing -- overt | 76.7% | [69, 83] | **92.7%** | [87, 96] | +16.0 |
| Phishing -- subtle / BEC | 24.7% | [18, 32] | **82.0%** | [75, 87] | +57.3 |
| Legitimate -- transactional | 94.0% | [89, 97] | 90.7% | [85, 94] | -3.3 |
| Legitimate -- security notices | 82.0% | [75, 87] | 81.3% | [74, 87] | -0.7 |
| Legitimate -- work / personal | 100% | [98, 100] | 100% | [98, 100] | 0.0 |
| Legitimate -- marketing | 99.2% | [95, 100] | 96.7% | [92, 99] | -2.5 |
| **mean phishing recall** | **50.7%** | | **87.3%** | | **+36.6** |
| **mean legitimate passed** | **93.8%** | | **92.2%** | | **-1.6** |

**Shipping decision.** v4.1 trades 1.6 points of mean legitimate precision for
36.6 points of mean phishing recall. In this deployment -- an assistive tool that
shows a probability and an explanation rather than blocking mail -- a missed
phishing email is the costlier error, so the trade was accepted. Recorded as a
judgement call, not an unambiguous upgrade.

**What changed in training.** ~4,000 generated BEC/spear examples added to the
phishing class; ~4,000 additional legitimate security-notification phrasings and
expanded transactional templates added to the legitimate class, on top of the
existing multi-corpus dataset. Held-out in-distribution: accuracy 0.9964,
F1 0.9965.

### The finding that matters more than the comparison

**Legitimate security notifications sit at roughly 82% in *both* models.**

This is not a v4.1 regression. It is a long-standing weakness the v3 gate was too
small to detect, in the same category that caused the original v1 failure and was
declared fixed on the strength of eleven emails. Roughly one in five legitimate
security emails is flagged:

```
Two-step verification was enabled on your Coinbase account   p=0.9991
A new device was added to your HDFC Bank account             p=0.9687
Your Dropbox two-factor authentication code is 48213         p=0.6780
You recently requested a password reset...                   p=0.9777
```

These are among the emails users most need to trust. A tool that flags password
resets and 2FA codes trains people to dismiss its warnings, eroding the value of
the warnings that are correct. **This is the most important open problem in the
email model** and v4.1 does not resolve it.

Likely cause: genuine security notifications and credential-harvest phishing
share nearly all their surface vocabulary -- *verify*, *sign-in*, *device*,
*password*, *expires*. What separates them is whether the recipient initiated the
action, which the message text does not reveal.

### The BEC ceiling

BEC detection improved 24.7% -> 82.0%. Two caveats bound what that means.

**The augmentation is synthetic.** Real BEC corpora are scarce, which is why the
training set lacked them. Training used template pool A; evaluation uses
structurally disjoint pool B, verified programmatically to share no template.
Passing therefore demonstrates generalisation across BEC *phrasing structures*,
not validated detection of real-world BEC. Confirming the latter needs real
samples and remains open.

**There is a structural limit.** A BEC email is linguistically almost identical
to a legitimate internal request -- *"Can you process this payment today?"* from a
real CFO and from an impersonator are the same sentence. The discriminating
evidence is sender identity, thread history, and organisational context, none of
which a text-only classifier observes. This mirrors the ceiling documented below
for the URL model: past a point, the signal is not in the input.

### Open limitations (email model)

- Security-notification false positives at ~18%, in both v3 and v4.1 -- the most
  significant unresolved problem.
- BEC validation rests on synthetic augmentation; no real-world BEC measurement
  exists.
- All figures come from templated mail: coverage results, not production rates.
- Residual harness artifacts remain (some brand/content pairings still read
  awkwardly), so legitimate-category figures are mildly pessimistic for both
  models.
- No real users. Every figure is from generated evaluation sets.

### Reproducibility (email model)

```
eval_email_models.py                  v3 vs v4.1, corrected harness (reference)
phishsentry_email_retrain_v4_1.ipynb  the v4.1 training run and its gate
v4_1_gate_results.json                Colab gate output -- NOTE: scored
                                      subject-less text, so NOT comparable to
                                      eval_email_models.py figures
evaluate_model_v3_expanded.py         superseded; random brand pairing
evaluate_model_v3.py                  the original 7/12/5 gate, kept as history
```

---

## URL model (XGBoost) evaluation

> **Read this before the numbers below.** The held-out recall of 0.963 reported
> in this section is an **in-distribution** result on the Vrbancic dataset split.
> Measured against live, currently-active phishing, recall is **38.3%**. Four
> measured iterations attempted to close that gap; the final one produced a
> **74.3% false-positive rate on genuine brand domains** -- it flagged the real
> `paypal.com/login` -- which is disqualifying for a user-facing tool.
>
> **The browser extension therefore ships a deterministic rule-based
> lookalike-domain detector, not this model.** It measures better on every axis
> (comparison at the end of this section). The in-distribution results below
> remain accurate as in-distribution results, and are retained for that reason.

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

### Measured against live phishing: 38.3% recall

60 URLs were sampled (seeded, reproducible) from the OpenPhish community feed --
verified currently-active phishing -- and scored through the exact production
path. Reproducible via `measure_url_recall.py`.

| Metric | Result |
|---|---:|
| URLs scored | 60 |
| Caught (p >= 0.5) | 23 |
| **Live recall** | **38.3%** |
| Median calibrated score | 0.2131 |

**Not a degraded-lookup artifact.** Recall by how many of the five
network-dependent features resolved: 0 missing -> 8/16 (50%); 1 -> 0/5; 2 ->
13/35 (37%); 3 -> 2/3. With every live lookup succeeding, recall was still 50%.

### Diagnosis: blindness to hosting-platform abuse

The misses clustered on phishing hosted on **legitimate platforms**: ~15 on
`*.pages.dev`, ~8 on `*.blogspot.com`, and others on `*.godaddysites.com`,
`*.weeblysite.com`, `*.vercel.app`. What the model *did* catch were dedicated
malicious domains (`dogespin.net`, `wordksl.top`, `roblox.com.am`, `.cfd`
domains) -- the 2020-era profile that dominates the training corpus.

For a phishing page at `3rf3x34x.pages.dev`, every feature reports benign: the
registered domain is Cloudflare's (years old, valid WHOIS, legitimate ASN,
normal TTL), the URL is short, no directory depth. A legitimate
`myportfolio.pages.dev` is near-identical across all 20 features. The only
discriminating signal is that one subdomain is a random string and the other is
a word -- and no original feature measures that.

### Four measured iterations

A v2 feature set was built (37 features: subdomain entropy, digit ratio,
consonant runs, dictionary word-likeness, hosting-platform flag, brand
impersonation, action tokens, suspicious TLDs) with zero network calls, trained
on ~16,000 URLs from public feeds. Train/test split **by registered domain**,
not randomly -- feed entries share domains heavily, and a random URL split would
place the same domain on both sides and inflate the score.

| Iteration | Live recall | Deep-link FP | Platform FP | Genuine-brand FP |
|---|---:|---:|---:|---:|
| v1 -- 20 network features | 38.3% | 37.5% | -- | -- |
| v2 -- new features, apex-domain negatives | 65.0% | **100%** | 8.3% | -- |
| v2 -- plus deep-URL negatives | 63.7% | 25.0% | 8.3% | -- |
| v2 -- plus dictionary features, threshold 0.35 | 50.0% | 25.0% | 8.3% | **74.3%** |

**Iteration 2** learned *"URL has a path -> phishing"* -- `qty_slash_url` was the
top feature by gain (76.1, roughly double the next). The negative class came from
Majestic Million, which yields bare apex domains, so every legitimate example was
path-free while the phishing feed was full of deep paths. Fixed by harvesting
5,000 legitimate deep URLs (98% with a real path) from Wayback CDX, Hacker News
and Wikipedia.

**Iteration 3** removed that shortcut and recall *held* at 63.7%, confirming the
gain was real. But platform abuse was still missed: the phonotactic
word-likeness heuristic passed pronounceable nonsense
(`sp3ct-drenix-biz8-solvek-tranu.pages.dev` has vowels in the right places).

**Iteration 4** added dictionary-based subdomain features, which XGBoost assigned
near-zero gain -- correctly, because usernames are also non-dictionary strings:

```
3rf3x34x.pages.dev                 phishing    non-dictionary subdomain
johnsmith.github.io                legitimate  non-dictionary subdomain
khukucuonlogun.godaddysites.com    phishing    non-dictionary subdomain
eshwarkole1641-esh.github.io       legitimate  non-dictionary subdomain
```

It also exposed the disqualifying failure. Top features were
`action_token_in_path` and `brand_token_present`, so the model learned
**"brand + /login -> phishing"** -- and nothing in the negative set contradicted
it, because genuine brand sign-in pages were never present as legitimate
examples. On 280 genuine brand URLs: `paypal.com/login` p=0.963,
`paypal.com/signin` p=0.963, `paypal.com/account` p=0.929 -- **208 of 280
flagged (74.3%)**.

### Why this is a ceiling, not a tuning problem

Training positives were checked for platform representation: the top ten
platform domains alone (`duckdns.org` 204, `pages.dev` 173,
`000webhostapp.com` 116, `webflow.io` 105, `vercel.app` 90, and others) account
for **16.6%** of positives, with more in the tail. The model had thousands of
platform-abuse examples and still missed them on live data.

The discriminating evidence is not in the URL. Whether `3rf3x34x.pages.dev` hosts
a Roblox credential form or a portfolio is determined by **page content**, which
URL-only classification cannot observe. Closing that gap requires fetching and
analysing the page -- a materially different system with its own privacy and
latency consequences.

**Sampling caveat.** Live-recall figures come from n=60-80 samples, roughly +/-11
percentage points. The 65.0% and 63.7% readings are not distinguishable from each
other. The differences carrying signal are the large ones -- 38% versus ~60% --
and the false-positive rates, measured on 280 URLs.

### What shipped: deterministic rules

The extension's actual claim is narrower than "phishing detection": *does this
domain imitate a known brand?* That is a deterministic string question with an
exact answer.

`lookalike_detector.py` implements ordered rules -- brand-domain allowlist (R0),
official org pages on code hosts (R0-org), punycode/homograph folding (R1),
character substitution (R2), brand token in a non-brand domain (R3), edit
distance 1-2 (R4), brand in an unrelated subdomain (R5), brand plus suspicious
TLD (R6).

The decisive property is **R0**: genuine brand domains are allowlisted *before
any heuristic executes*, so the false-positive rate on them is **structurally
zero rather than statistically small**. A probabilistic model cannot offer that
guarantee.

Measured on the same four probe sets (`eval_lookalike_detector.py`):

| Metric | Rules | Model (best iteration) |
|---|---:|---:|
| Synthetic typosquat recall (n=600) | **98.0%** | 90.3% |
| Real brand-impersonation recall, live feed | **98.6%** | 77.1% |
| Genuine brand domains kept clean (n=280) | **100.0%** | 25.7% |
| Ordinary legitimate browsing kept clean | **100.0%** | 68.8% |

Secondary benefits: verdicts are explainable per-rule rather than as an opaque
probability; no training data, no drift, no retraining pipeline; and the
deployment artifact is a 1 MB zip rather than a ~1 GB container image.

### Honest limitations of the rules approach

- **The brand and platform lists require maintenance.** A brand absent from
  `BRAND_DOMAINS` is undetectable, and a *legitimate* brand domain missing from
  it would be a false positive. That list is the safety-critical component.
- **Fused short brand names are partially missed.** Brands under four characters
  match on token boundaries, or on a token prefix when corroborated by a
  credential-harvest word (`dhlaccount.com` is caught, `dhlevripremium.com` is
  not). Bare three-letter substring matching would fire "ups" inside
  `startups.com`, so the miss is accepted deliberately.
- **Heavily mangled short domains are missed** (`fedx.com`, `dlh.com`) -- too
  short to separate from legitimate initialisms.
- **Brand names in subdomains on shared hosting are genuinely ambiguous.**
  `netflix-clone-x.vercel.app` may be a student project or an impersonation page.
  These receive a lower-confidence caution verdict with wording that says so.
- **The scope limit is real and stated in the store listing:** it does not detect
  phishing hosted on legitimate platforms under an unrelated subdomain.

### The parallel with the email model

Structurally the same failure as the email model's v1 -> v3 arc: an
in-distribution score that looked excellent while concealing a generalization
gap, found only by testing against data the training set did not represent.

The difference is the conclusion. For email, better data and a stricter gate
improved the model. For URLs, four measured iterations established that the
discriminating signal is not present in the input at all -- so the honest outcome
was to narrow the claim to what could be done well, and publish the measurements
that justify the narrowing.

*Reproducibility: `measure_url_recall.py` (live recall, v1 model),
`collect_url_dataset_v2.py` + `train_url_model_v2.py` (v2 iterations),
`measure_lookalike_recall.py` (model on the narrow task),
`eval_lookalike_detector.py` (rules on the same probe sets). Live-feed figures
vary between runs as the feeds update.*
