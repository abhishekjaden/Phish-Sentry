#!/usr/bin/env python3
"""
PhishSentry — RoBERTa text-model evaluation harness
===================================================

Runs TWO evaluations against the *exact* production model + preprocessing:

  1. HELD-OUT TEST SET  — the rows of emails.csv the model never saw during
     training/validation (reconstructed with the SAME split the training
     notebook used: test_size=0.1, random_state=42, stratify=y). This gives
     honest in-distribution precision/recall/F1/accuracy.

  2. REAL LEGIT-SECURITY EMAILS — a hand-curated set of genuinely safe but
     "scary-sounding" modern emails (AWS, Google, GitHub, Microsoft, LinkedIn,
     Pearson, IEEE/CMT, etc.). Every one is truly legitimate (label = 0), so any
     "phishing" verdict here is a FALSE POSITIVE. This measures the model's
     real-world weakness on the kind of mail it was NOT trained on.

Run it from your project folder in the venv that already has torch/transformers:

    cd "C:\\Users\\ABHISHEK\\Downloads\\Phishing attack detection updated"
    python evaluate_model.py

It loads the model the same way inference_service.py does:
    models/roberta_tokenizer , models/roberta_base_local , models/roberta_phishing_model.pth

Nothing here calls the network or the live site — it's a pure local eval.
"""

import os
import re
import sys
import numpy as np

# ---------------------------------------------------------------------------
# Config — paths match inference_service.py
# ---------------------------------------------------------------------------
MODEL_DIR        = "models"
TOKENIZER_PATH   = os.path.join(MODEL_DIR, "roberta_tokenizer")
BASE_MODEL_PATH  = os.path.join(MODEL_DIR, "roberta_base_local")
WEIGHTS_PATH     = os.path.join(MODEL_DIR, "roberta_phishing_model.pth")
CSV_PATH         = "emails.csv"
MAX_LENGTH       = 256          # same as production
THRESHOLD        = 0.5          # production rule: probabilities[1] > 0.5 => phishing


# ---------------------------------------------------------------------------
# EXACT production preprocessing (copied from inference_service.py)
# ---------------------------------------------------------------------------
def preprocess_email_text(text):
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^\w\s.!?,-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ---------------------------------------------------------------------------
# Hand-curated REAL legit-security emails (all label = 0 / legitimate)
# Buckets:
#   gold       = unambiguously safe (sender domain matches brand). A phishing
#                verdict here is a clean false positive.
#   borderline = legit but genuinely suspicious-looking (urgency + form links,
#                gmail sender, etc.) — a flag here is not cleanly "wrong".
#   easy       = warm/marketing legit mail with little security language —
#                a control to confirm the model isn't flagging *everything*.
# Text is trimmed to the meaningful opening (the model only reads ~256 tokens).
# ---------------------------------------------------------------------------
LEGIT_EMAILS = [
    # ---------------- GOLD (unambiguous) ----------------
    ("google_account_deletion", "gold",
     "Google. Sign in to your Google Account. Your Google Account has not been "
     "used within a 2-year period. If you want to keep your Google Account, sign "
     "in to your Google Account before the deletion date. To protect user privacy "
     "and account data, Google will delete Google Accounts that are not used. "
     "Sign in. You received this email to let you know about important changes to "
     "your Google Account and services."),

    ("aws_iam_identity_center_invite", "gold",
     "Hello, Your administrator for AWS Account has invited you to AWS IAM Identity "
     "Center. Accepting this invitation activates your user account in IAM Identity "
     "Center so that you can access assigned AWS resources. Choose the link below to "
     "accept this invitation. Accept invitation. This invitation will expire in 7 days. "
     "Accessing the AWS access portal. After you've accepted the invitation, you can "
     "sign in to the AWS access portal using the information below. Your username."),

    ("pearson_vue_exam_confirmation", "gold",
     "Action required and exam time confirmed. Please do not reply to this message, "
     "as this inbox is not monitored. Dear AWS Candidate, We're confirming that your "
     "exam has been scheduled. Your action is required before exam day. Before the "
     "exam, your responsibilities include reading all exam policies and FAQs, as well "
     "as ensuring your computer, internet connection, and the room meet all the "
     "requirements. Run the system test prior to your exam using the link in the "
     "Appointment Details section of this email."),

    ("github_review_signin", "gold",
     "Hey! Your GitHub account was successfully signed in to but we did not recognize "
     "the location of the sign in. You can review this sign in attempt by visiting the "
     "link. If you recently signed in to your account, you do not need to take any "
     "further action. If you did not sign in to your account, your password may be "
     "compromised. Visit the settings to create a new, strong password for your GitHub "
     "account. To see this and other security events for your account, visit the security log."),

    ("github_thirdparty_app_added", "gold",
     "Hey! A third-party GitHub Application (Sentry) with the following permissions: "
     "View your email addresses. Was recently authorized to access your account. Visit "
     "the link for more information. To see this and other security events for your "
     "account, visit your security log. If you run into problems, please contact support."),

    ("github_oauth_app_added", "gold",
     "Hey! A first-party GitHub OAuth application (Git Credential Manager) with gist, "
     "repo, and workflow scopes was recently authorized to access your account. Visit "
     "the link for more information. To see this and other security events for your "
     "account, visit your security log. If you run into problems, please contact support."),

    ("microsoft_passkey_added", "gold",
     "Microsoft account. Security info was added. The following security info was "
     "recently added to your Microsoft account: Passkey. If this was you, then you can "
     "safely ignore this email. If this wasn't you, a malicious user has access to your "
     "account. Please review your recent activity and we'll help you secure your account. "
     "Review recent activity. Thanks, The Microsoft account team."),

    ("google_new_signin_windows", "gold",
     "Google. Security alert. A new sign-in on Windows. We noticed a new sign-in to your "
     "Google Account on a Windows device. If this was you, you don't need to do anything. "
     "If not, we'll help you secure your account. Check activity. You can also see security "
     "activity at your account notifications page."),

    ("linkedin_verify_new_device", "gold",
     "Hi, We noticed you recently tried to sign in to your LinkedIn account from a new "
     "device. You can finish signing in to your LinkedIn account by following the "
     "instructions that we sent to your LinkedIn App. If you're having trouble signing in, "
     "please visit the LinkedIn Help Center. When and where it happened: a new device, "
     "iOS, approximate location Chennai. Didn't do this? Be sure to change your password "
     "right away."),

    ("ieee_pdf_express_account", "gold",
     "IEEE PDF eXpress: Account Confirmation. Dear, An account has been created for you in "
     "IEEE PDF eXpress. Here is your login information: Conference ID, Email address, "
     "Password. Please type the password that you set when you created this account. If you "
     "do not remember this, please use the Forgot your password link available on the main "
     "page to get the Password Reset link. Please click on the URL given below for activation. "
     "Click for account activation. Keep this email for future reference."),

    ("microsoft_cmt_major_revision", "gold",
     "Dear Author, The first round of review process has been completed for your paper and "
     "it requires Major Revision. But your paper will be reconsidered for the conference for "
     "possible publication in IEEE Xplore digital library, if you modify your paper according "
     "to the reviewer comments. You are requested to access the detailed reviewer comments "
     "through your Microsoft CMT profile. Please ensure that the revised manuscript strictly "
     "follows the instructions. Do not create a new submission. Submit the revised manuscript "
     "using the Edit Submission option."),

    # ---------------- BORDERLINE (legit but suspicious-looking) ----------------
    ("ediglobe_onboarding_form", "borderline",
     "Quick Action Required: Complete Your Onboarding Form. Dear Abhishek, Greetings from "
     "EdiGlobe! Thank you for choosing EdiGlobe for your learning journey. To ensure a "
     "seamless start, we kindly request you to complete the mandatory Onboarding Form within "
     "the next 24 to 48 hours. Onboarding Form Link. For additional course details, feel free "
     "to explore our website. LMS dashboard access will be provided within the first week. If "
     "you encounter any issues, please contact our support team."),

    ("ieee_ict_quiz_credentials", "borderline",
     "Dear all, Thanks for registering for the IEEE CS Madras Mega ICT Quiz. We are pleased "
     "to inform the Online Prelims Test is scheduled. This mail contains login credentials, "
     "test guidelines and instructions. Login Credentials: Test Portal address. Login UserID: "
     "Your email id as given at the registration. Login Password: Your 10-digit mobile no. "
     "Please login to the portal with your UserID and Password. The portal will open for login."),

    # ---------------- EASY (warm/marketing legit controls) ----------------
    ("aws_cloudtrail_welcome", "easy",
     "Welcome to AWS CloudTrail. Visibility into your AWS account activity. Monitor your AWS "
     "activity in the cloud by getting a history of API calls for your account with AWS "
     "CloudTrail. View documentation. Explore CloudTrail tutorials. If you're new to AWS "
     "CloudTrail, these tutorials can help you learn how to use its features. Sign in to "
     "CloudTrail."),

    ("volante_onboarding", "easy",
     "Hi All, Welcome to Volante's One Team! The agenda of the session is to guide you through "
     "the joining process and required formalities to ensure smooth onboarding and "
     "documentation. Do join the video call by clicking on the below link. Microsoft Teams. "
     "Join the meeting now. Warm Regards, Manager, People Partner."),

    ("microsoft_passkeys_marketing", "easy",
     "Make signing in easier and safer with passkeys. Faster, easier, more secure. Passkeys "
     "are a more convenient and secure alternative to passwords. They are easy to use, unique "
     "to each site and resistant to phishing, eliminating the need for complex passwords. "
     "Enhance your security with passkeys in Windows. Sign-in with Hello. Windows Hello is a "
     "faster, more personal, more secure way to get instant access to your Windows devices."),

    ("linkedin_application_sent", "easy",
     "Your application was sent to Wiingy. Generative AI Engineer Intern. Wiingy, Greater "
     "Bengaluru Area. Applied. This email was intended for you. You are receiving LinkedIn "
     "notification emails."),

    ("fie_paper_withdrawn", "easy",
     "Dear Mr. Abhishek, your paper has been withdrawn from FIE 2026. You can find all your "
     "papers for this conference using the EDAS login. Regards, FIE 2026 Conference Chairs, "
     "Technical Program Committee Chairs for the IEEE Frontiers in Education Conference."),
]


def load_model():
    """Load tokenizer + model exactly as inference_service.py does."""
    import torch
    from transformers import RobertaTokenizer, RobertaForSequenceClassification

    device = torch.device("cpu")  # eval on CPU is fine for this volume
    print(f"[load] device = {device}")

    if not os.path.exists(WEIGHTS_PATH):
        sys.exit(f"ERROR: weights not found at {WEIGHTS_PATH}. Run from project root.")

    # Tokenizer: prefer the saved one; fall back to roberta-base if folder missing.
    if os.path.isdir(TOKENIZER_PATH):
        tok = RobertaTokenizer.from_pretrained(TOKENIZER_PATH)
    else:
        print("[load] roberta_tokenizer/ not found; using roberta-base tokenizer")
        tok = RobertaTokenizer.from_pretrained("roberta-base")

    # Base architecture: prefer saved base; fall back to roberta-base.
    base = BASE_MODEL_PATH if os.path.isdir(BASE_MODEL_PATH) else "roberta-base"
    model = RobertaForSequenceClassification.from_pretrained(base, num_labels=2)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("[load] model loaded OK")
    return tok, model, device


def predict_batch(texts, tok, model, device, batch_size=16):
    """Return phishing probabilities (prob of class 1) for a list of raw texts."""
    import torch
    probs = []
    for i in range(0, len(texts), batch_size):
        chunk = [preprocess_email_text(t) for t in texts[i:i + batch_size]]
        enc = tok(chunk, truncation=True, padding="max_length",
                  max_length=MAX_LENGTH, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            p = torch.softmax(logits, dim=1).cpu().numpy()[:, 1]
        probs.extend(p.tolist())
        print(f"  scored {min(i + batch_size, len(texts))}/{len(texts)}", end="\r")
    print()
    return np.array(probs)


def reconstruct_heldout():
    """Rebuild the EXACT held-out test set the notebook used (never trained on)."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(CSV_PATH)
    df = df.drop_duplicates()
    X = df["text"].values
    y = df["spam"].values
    # First split = the one that carves out X_test (the true held-out set).
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    return X_test, y_test


def metrics(y_true, y_pred):
    from sklearn.metrics import (precision_score, recall_score, f1_score,
                                 accuracy_score, confusion_matrix)
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "cm":        confusion_matrix(y_true, y_pred),
    }


def bootstrap_ci(y_true, y_pred, metric_fn, n=2000, seed=42):
    """95% bootstrap confidence interval for a metric."""
    import numpy as np
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    idx = np.arange(len(y_true)); vals = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        try: vals.append(metric_fn(y_true[s], y_pred[s]))
        except Exception: pass
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return lo, hi


def main():
    print("=" * 70)
    print("PhishSentry — RoBERTa text-model evaluation")
    print("=" * 70)

    tok, model, device = load_model()

    # ---------- Eval 1: held-out test set ----------
    print("\n[1] HELD-OUT TEST SET (rows never seen in training)")
    X_test, y_test = reconstruct_heldout()
    print(f"    held-out size: {len(X_test)}  "
          f"(legit={int((y_test==0).sum())}, spam={int((y_test==1).sum())})")
    p_test = predict_batch(list(X_test), tok, model, device)
    yhat_test = (p_test > THRESHOLD).astype(int)
    m = metrics(y_test, yhat_test)
    print(f"    accuracy : {m['accuracy']:.4f}")
    from sklearn.metrics import precision_score, recall_score
    p_lo, p_hi = bootstrap_ci(y_test, yhat_test, lambda a,b: precision_score(a,b,zero_division=0))
    r_lo, r_hi = bootstrap_ci(y_test, yhat_test, lambda a,b: recall_score(a,b,zero_division=0))
    print(f"    precision: {m['precision']:.4f}   95% CI [{p_lo:.4f}, {p_hi:.4f}]")
    print(f"    recall   : {m['recall']:.4f}   95% CI [{r_lo:.4f}, {r_hi:.4f}]")
    print(f"    f1       : {m['f1']:.4f}")
    tn, fp, fn, tp = m["cm"].ravel()
    print(f"    confusion: TN={tn} FP={fp} FN={fn} TP={tp}")

    # ---------- Eval 2: real legit-security emails ----------
    print("\n[2] REAL LEGIT-SECURITY EMAILS (all truly legitimate; flags = false positives)")
    names   = [e[0] for e in LEGIT_EMAILS]
    buckets = [e[1] for e in LEGIT_EMAILS]
    texts   = [e[2] for e in LEGIT_EMAILS]
    p_legit = predict_batch(texts, tok, model, device)
    yhat_legit = (p_legit > THRESHOLD).astype(int)

    print(f"\n    {'email':<34}{'bucket':<12}{'phish_prob':>11}  verdict")
    print("    " + "-" * 66)
    for n, b, pr, yh in zip(names, buckets, p_legit, yhat_legit):
        verdict = "PHISHING (FP!)" if yh == 1 else "legit ok"
        print(f"    {n:<34}{b:<12}{pr:>11.4f}  {verdict}")

    # Per-bucket false-positive rates
    print("\n    False-positive rate by bucket (lower is better):")
    for bk in ["gold", "borderline", "easy"]:
        idx = [i for i, b in enumerate(buckets) if b == bk]
        if not idx:
            continue
        fp_count = int(sum(yhat_legit[i] for i in idx))
        print(f"      {bk:<12}: {fp_count}/{len(idx)} flagged as phishing "
              f"({100*fp_count/len(idx):.1f}%)")
    total_fp = int(yhat_legit.sum())
    print(f"      {'ALL':<12}: {total_fp}/{len(LEGIT_EMAILS)} "
          f"({100*total_fp/len(LEGIT_EMAILS):.1f}%)")

    # Headline number = gold bucket FP rate (the unambiguous cases)
    gold_idx = [i for i, b in enumerate(buckets) if b == "gold"]
    gold_fp = int(sum(yhat_legit[i] for i in gold_idx))
    print("\n" + "=" * 70)
    print(f"HEADLINE: of {len(gold_idx)} UNAMBIGUOUSLY-LEGIT security emails, "
          f"the model wrongly flagged {gold_fp} as phishing "
          f"({100*gold_fp/len(gold_idx):.1f}% false-positive rate).")
    print("=" * 70)
    print("\nDone. Paste this entire output back into the chat for the eval report.")


if __name__ == "__main__":
    main()
