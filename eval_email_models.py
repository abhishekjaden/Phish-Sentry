#!/usr/bin/env python3
"""
Corrected out-of-distribution email evaluation, comparing two checkpoints.

WHY THIS REPLACES evaluate_model_v3_expanded.py
-----------------------------------------------
The previous harness assigned brands to content at random, generating emails no
real sender would write:

    Subject: ICICI Bank - quick note
    Reminder: our 1:1 is at 9:00 AM tomorrow.          -> scored p=0.9926

A bank does not send 1:1 reminders. Counting the model's reaction to that as a
false positive is measuring the harness, not the model. Same for
"Dropbox - your monthly statement" and "PayPal - your order shipped".

FIXES
  1. BRAND_PROFILES restricts each brand to categories it plausibly sends.
  2. Work/personal mail carries NO brand -- internal colleague email does not
     announce a vendor in its subject line.
  3. Subject lines are generated per category rather than from one template, so
     the model is not keying on a constant.
  4. Both checkpoints are scored in the same run, on the SAME generated corpus,
     so differences are attributable to the model rather than to sampling.

KNOWN DISCREPANCY WITH THE COLAB GATE
  The v4/v4.1 notebook gate scored bare template bodies with no subject line,
  while this harness scores full emails. The two number sets are therefore NOT
  comparable, which invalidated an earlier side-by-side. The next notebook
  revision should prepend subjects in its gate. Until then, treat THIS harness
  as the reference measurement.

STILL TRUE OF BOTH: these emails are templated. They measure consistency across
a phrasing space and detect whole-category failures. They are not a random
sample of real mail, so the rates are coverage results, not production
estimates.

Usage:
  python eval_email_models.py
  python eval_email_models.py --a models/roberta_phishing_model_v3.pth \
                              --b models/roberta_phishing_model_v4_1.pth
"""
import argparse
import html as _html
import itertools
import math
import random
import re

import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

random.seed(42)

# ---------------------------------------------------------------------------
# Which brands plausibly send which category. A brand absent from a category is
# never paired with it.
# ---------------------------------------------------------------------------
BRAND_PROFILES = {
    # retail / marketplace
    "Amazon":      {"txn", "phish_overt", "phish_subtle", "marketing"},
    "Flipkart":    {"txn", "phish_overt", "marketing"},
    # shipping
    "DHL":         {"txn", "phish_overt", "phish_subtle"},
    "FedEx":       {"txn", "phish_overt", "phish_subtle"},
    # banking / payments
    "PayPal":      {"txn", "sec", "phish_overt", "phish_subtle"},
    "HDFC Bank":   {"txn", "sec", "phish_overt"},
    "ICICI Bank":  {"txn", "sec", "phish_overt"},
    "Coinbase":    {"sec", "phish_overt", "phish_subtle"},
    "Ledger":      {"sec", "phish_overt", "phish_subtle"},
    # streaming / consumer
    "Netflix":     {"txn", "sec", "phish_overt", "marketing"},
    "Spotify":     {"txn", "sec", "marketing"},
    "Steam":       {"sec", "phish_overt"},
    # tech / SaaS
    "Microsoft":   {"sec", "phish_overt", "phish_subtle", "marketing"},
    "Google":      {"sec", "phish_overt", "phish_subtle", "marketing"},
    "Apple":       {"txn", "sec", "phish_overt", "marketing"},
    "Dropbox":     {"sec", "phish_subtle", "marketing"},
    "Zoom":        {"sec", "phish_subtle", "marketing"},
    "Adobe":       {"txn", "sec", "marketing"},
    "LinkedIn":    {"sec", "phish_subtle", "marketing"},
    "Instagram":   {"sec", "phish_overt"},
}

def brands_for(cat):
    return [b for b, cats in BRAND_PROFILES.items() if cat in cats]

SUBJECTS = {
    "phish_overt": ["Urgent: account notice", "Immediate action required",
                    "Security alert", "Your account has been limited",
                    "Final notice"],
    "phish_subtle": ["Re: our conversation", "Invoice attached",
                     "Payment request", "Document shared with you",
                     "Action needed before Friday"],
    "txn": ["Your order", "Receipt", "Delivery update", "Your statement",
            "Subscription renewal", "Payment confirmation"],
    "sec": ["Security notification", "Sign-in alert", "Verification code",
            "Account activity", "Password notice"],
    "marketing": ["This week's update", "Product news", "You're invited",
                  "What's new", "Your year in review"],
    "work": ["Standup notes", "Quick question", "Re: draft", "Tomorrow's sync",
             "Report ready for review", "Onboarding"],
}

FILL = {
    "num": lambda: str(random.randint(1000, 99999)),
    "amount": lambda: random.choice(["$45.00", "$250.00", "₹1,299", "£19.99",
                                     "$1,024.50", "₹8,450", "$4,800"]),
    "day": lambda: random.choice(["Monday", "Thursday", "Friday",
                                  "12 August", "3 September", "tomorrow"]),
    "time": lambda: random.choice(["9:00 AM", "2pm", "3:30 PM", "18:45"]),
    "city": lambda: random.choice(["Chennai", "Delhi", "London", "Singapore",
                                   "Toronto", "Berlin"]),
    "device": lambda: random.choice(["Windows device", "iPhone",
                                     "Android phone", "Mac"]),
    "name": lambda: random.choice(["Abhishek", "Priya", "Sam", "Alex", "Ravi"]),
    "slug": lambda: random.choice(["secure", "account", "verify", "billing",
                                   "support", "portal"]),
}

def fill(t, brand=""):
    out = t.replace("{brand}", brand)
    for k, gen in FILL.items():
        tokn = "{" + k + "}"
        while tokn in out:
            out = out.replace(tokn, gen(), 1)
    return out

# ---------------------------------------------------------------- templates
T = {
"phish_overt": [
 "Dear Customer, your {brand} account has been suspended. Immediate action is "
 "required. Click here to verify your identity: http://{slug}-{brand}.tk/login",
 "Dear User, we detected unusual sign-in activity on your {brand} account. You "
 "must respond within 24 hours: http://{slug}.cfd/verify",
 "Attention: your payment method for {brand} was declined. Act now to avoid "
 "service interruption: http://{brand}-billing.top/update",
 "Congratulations! You have been selected for a refund of {amount} from "
 "{brand}. Reply with your bank account number to claim.",
 "Your {brand} parcel is being held at customs. Pay the outstanding fee to "
 "release it: http://{slug}.icu/pay",
 "Your {brand} security certificate has expired. Re-validate your credentials "
 "now or access will be permanently revoked: http://{slug}.xyz/auth",
],
"phish_subtle": [
 "Hi {name}, following up on our conversation. I've shared the updated document "
 "via the {brand} portal: https://docs-{slug}-review.com/auth/signin",
 "Please find attached invoice #{num} for services rendered. Kindly process "
 "payment to the updated account details within 3 business days.",
 "{name}, I need you to process an urgent wire transfer before end of day. "
 "I'm in meetings, so email only. Details to follow.",
 "Your {brand} mailbox is almost full and you may stop receiving emails. "
 "Revalidate your credentials using the link below to continue.",
 "HR notice: please review and acknowledge the updated payroll document at "
 "https://{slug}-hr-portal.online/login before {day}.",
 "Our banking details have changed following an audit. Please use the account "
 "in the attached remittance form for invoice #{num}.",
],
"txn": [
 "Your {brand} order #{num} has shipped and should arrive {day}.",
 "Thanks for your purchase from {brand}. Your receipt for {amount} is attached.",
 "Your {brand} package was delivered on {day}.",
 "Your {brand} monthly statement is now available in your account.",
 "Your {brand} subscription renews on {day}. No action is needed.",
 "Your refund of {amount} has been processed and will appear in 5-7 days.",
 "Your {brand} payment of {amount} was successful. Reference {num}.",
 "Your {brand} invoice for {amount} has been paid in full. Thank you.",
],
"sec": [
 "We noticed a new sign-in to your {brand} account from a {device} in {city}. "
 "If this was you, no action is needed.",
 "You recently requested a password reset for your {brand} account. The link "
 "below expires in 30 minutes. If you did not request this, ignore this email.",
 "A new device was added to your {brand} account on {day}. If you don't "
 "recognise it, review your security settings.",
 "Your {brand} two-factor authentication code is {num}. It expires in 10 minutes.",
 "A transaction of {amount} was made on your card ending {num} at {city} on {day}.",
 "Your {brand} password was changed successfully on {day}.",
 "Two-step verification was enabled on your {brand} account.",
],
"marketing": [
 "This week at {brand}: three new features, and a look at what's next.",
 "The {brand} summer sale starts {day}. Up to 40% off selected items.",
 "You're invited to the {brand} webinar on {day} at {time}. Registration is free.",
 "{brand} newsletter: product updates, engineering notes, and job openings.",
 "New in {brand}: faster search, dark mode, and improved exports.",
 "Thanks for being with us this year. Here's your {brand} year in review.",
],
# work mail carries NO brand -- internal colleague email does not announce a vendor
"work": [
 "Hi team, attaching the notes from today's standup. Let me know if I missed "
 "anything.",
 "Reminder: our 1:1 is at {time} tomorrow. Happy to move it if needed.",
 "The quarterly report is ready for review. Comments welcome by {day}.",
 "Meeting moved to {day} at {time}. Calendar has been updated.",
 "Welcome to the team! Here's your onboarding checklist for week one.",
 "Could you review the attached draft before {day}? No rush.",
 "The deployment finished at {time}. All checks passed.",
 "Lunch on {day}? There's a new place near the office.",
],
}

def generate(cat, n):
    out = set()
    subs = SUBJECTS[cat]
    brands = brands_for(cat) if cat != "work" else [""]
    combos = list(itertools.product(T[cat], subs, brands))
    random.shuffle(combos)
    while len(out) < n:
        before = len(out)
        for body, subj, brand in combos:
            if len(out) >= n:
                break
            head = f"{brand} - {subj}" if brand else subj
            out.add(f"Subject: {head}\n\n{fill(body, brand)}")
        if len(out) == before:
            break
    return list(out)

# ---------------------------------------------------------------- model
def preprocess(t):
    t = _html.unescape(str(t))
    if "<" in t and ">" in t:
        t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"^\s*subject\s*:\s*", "", t, flags=re.I)
    t = re.sub(r"https?://\S+|www\.\S+", " httpaddr ", t)
    return re.sub(r"\s+", " ", t).strip()[:5000]

def load(path, tok_dir="models/roberta_tokenizer",
         base="models/roberta_base_local"):
    dev = torch.device("cpu")
    tok = RobertaTokenizer.from_pretrained(tok_dir)
    mdl = RobertaForSequenceClassification.from_pretrained(base, num_labels=2)
    mdl.load_state_dict(torch.load(path, map_location=dev))
    mdl.eval()
    return tok, mdl

def score(texts, tok, mdl, batch=32):
    ps = []
    for i in range(0, len(texts), batch):
        chunk = [preprocess(x) for x in texts[i:i + batch]]
        e = tok(chunk, truncation=True, padding=True, max_length=256,
                return_tensors="pt")
        with torch.no_grad():
            ps += torch.softmax(mdl(**e).logits, dim=1)[:, 1].tolist()
    return ps

def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)

CATS = [
    ("Phishing - overt",       "phish_overt",  True,  150),
    ("Phishing - subtle/BEC",  "phish_subtle", True,  150),
    ("Legit - transactional",  "txn",          False, 150),
    ("Legit - security notes", "sec",          False, 150),
    ("Legit - work/personal",  "work",         False, 150),
    ("Legit - marketing",      "marketing",    False, 120),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="models/roberta_phishing_model_v3.pth")
    ap.add_argument("--b", default="models/roberta_phishing_model_v4_1.pth")
    ap.add_argument("--label-a", default="v3")
    ap.add_argument("--label-b", default="v4.1")
    args = ap.parse_args()

    print("=" * 78)
    print("Email model comparison -- corrected harness")
    print("=" * 78)
    print("Brands are paired only with content they plausibly send; work mail")
    print("carries no brand. Both checkpoints score the SAME generated corpus.")
    print("Templated emails: coverage results with intervals, not production")
    print("estimates.\n")

    corpora = {key: generate(key, n) for _, key, _, n in CATS}
    print("sample of each category:")
    for _, key, _, _ in CATS:
        print(f"  [{key}] {corpora[key][0].replace(chr(10),' | ')[:92]}")
    print()

    tok, mdl_a = load(args.a)
    tok_b, mdl_b = load(args.b)

    rows = []
    for name, key, expect, n in CATS:
        texts = corpora[key]
        pa = score(texts, tok, mdl_a)
        pb = score(texts, tok_b, mdl_b)
        ka = sum(1 for p in pa if (p > 0.5) == expect)
        kb = sum(1 for p in pb if (p > 0.5) == expect)
        ra, rb = ka / len(texts), kb / len(texts)
        la, ha = wilson(ka, len(texts))
        lb, hb = wilson(kb, len(texts))
        rows.append((name, expect, ra, la, ha, rb, lb, hb, len(texts),
                     texts, pa, pb))

    w = 24
    print(f"{'category':<{w}}{args.label_a:>9}{'95% CI':>16}"
          f"{args.label_b:>9}{'95% CI':>16}{'delta':>9}")
    print("-" * 84)
    for (name, expect, ra, la, ha, rb, lb, hb, n, *_ ) in rows:
        d = rb - ra
        arrow = "+" if d > 0.001 else ("-" if d < -0.001 else " ")
        print(f"{name:<{w}}{ra:>8.1%} [{la:>4.0%},{ha:>4.0%}]"
              f"{rb:>9.1%} [{lb:>4.0%},{hb:>4.0%}]{arrow}{abs(d):>7.1%}")

    print("\nmisses under the newer model:")
    for (name, expect, ra, la, ha, rb, lb, hb, n, texts, pa, pb) in rows:
        wrong = [(t, p) for t, p in zip(texts, pb) if (p > 0.5) != expect]
        if not wrong:
            continue
        kind = "missed phishing" if expect else "false positives"
        print(f"\n  {name} -- {len(wrong)} {kind}")
        for t, p in wrong[:4]:
            print(f"    p={p:.4f}  {t.replace(chr(10),' | ')[:78]}")

    ph = [r for r in rows if r[1]]
    lg = [r for r in rows if not r[1]]
    print("\n" + "=" * 78)
    print(f"  phishing recall   {args.label_a}: "
          f"{sum(r[2] for r in ph)/len(ph):.1%}   "
          f"{args.label_b}: {sum(r[5] for r in ph)/len(ph):.1%}")
    print(f"  legitimate passed {args.label_a}: "
          f"{sum(r[2] for r in lg)/len(lg):.1%}   "
          f"{args.label_b}: {sum(r[5] for r in lg)/len(lg):.1%}")
    print("\n  A model that gains phishing recall while losing legitimate")
    print("  precision is a TRADEOFF, not an upgrade. Decide which error costs")
    print("  more in the deployment context and record the reasoning.")


if __name__ == "__main__":
    main()
