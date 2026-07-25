#!/usr/bin/env python3
"""
Expanded out-of-distribution evaluation for the v3 email model.

WHY: the existing gate is 7 phishing / 12 legitimate / 5 regression. That is
enough to demonstrate specific failure modes are closed; it cannot support a
false-positive RATE claim. 0/12 has a 95% upper bound near 24%.

WHAT THIS IS: several hundred emails generated combinatorially across brands,
pretexts, phrasings, and call-to-action styles, in six categories. Every
category reports a Wilson score interval.

WHAT THIS IS NOT -- read before quoting any number:
  These emails are TEMPLATED. They test whether the model is CONSISTENT across
  a wide space of surface variation, and whether whole categories fail. They are
  NOT a random sample of real-world mail, so the rates here are not unbiased
  estimates of production performance. A model could score perfectly here and
  still fail on real mail whose phrasing the templates do not cover -- which is
  precisely how the v1 and v2 evaluations went wrong.

  Treat these as coverage tests with intervals, not as a population rate.

Usage:
  python evaluate_model_v3_expanded.py
  python evaluate_model_v3_expanded.py --per-category 120
"""
import argparse
import html as _html
import itertools
import math
import random
import re
import sys

import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

random.seed(42)

# ---------------------------------------------------------------------------
# generation vocabulary
# ---------------------------------------------------------------------------
BRANDS = ["PayPal", "Amazon", "Netflix", "Apple", "Microsoft", "Google",
          "DHL", "FedEx", "HDFC Bank", "ICICI Bank", "Instagram", "LinkedIn",
          "Dropbox", "Spotify", "Coinbase", "Ledger", "Steam", "Zoom"]

# --- phishing -------------------------------------------------------------
PHISH_PRETEXT = [
    "your account has been suspended",
    "we detected unusual sign-in activity",
    "your payment method was declined",
    "your account will be permanently closed",
    "suspicious activity was found on your account",
    "your subscription could not be renewed",
    "your parcel is being held at customs",
    "your mailbox storage is full",
    "an unauthorised transaction was attempted",
    "your security certificate has expired",
]
PHISH_URGENCY = [
    "Immediate action is required.",
    "You must respond within 24 hours.",
    "Failure to act will result in permanent loss of access.",
    "This is your final notice.",
    "Act now to avoid service interruption.",
    "",
]
PHISH_CTA = [
    "Click here to verify your identity: http://{slug}.tk/login",
    "Confirm your details now: http://{slug}-secure.cfd/verify",
    "Re-validate your credentials: https://{slug}.account-verify.xyz/auth",
    "Update your payment information: http://{slug}-billing.top/update",
    "Sign in to restore access: http://secure-{slug}.icu/signin",
    "Reply to this email with your account number and password.",
]
PHISH_OPENER = ["Dear Customer,", "Dear User,", "Dear Valued Member,",
                "Attention:", "Hello,", ""]

# --- legitimate: transactional (the category that broke v2) ---------------
LEGIT_TXN = [
    "Your order #{num} has shipped and should arrive on {day}.",
    "Thanks for your purchase. Your receipt for {amount} is attached.",
    "Your package was delivered on {day}. We hope you enjoy it.",
    "Your monthly statement is now available in your account.",
    "Your subscription renews on the {dd}th. No action is needed.",
    "Your refund of {amount} has been processed and will appear in 5-7 days.",
    "Your flight {flight} to {city} is on schedule. Boarding at {time}.",
    "Your table at {city} Bistro is confirmed for {day} at {time}.",
    "Your invoice for {amount} has been paid. Thank you.",
    "Your booking reference is {num}. Check-in opens 48 hours before departure.",
]

# --- legitimate: security notifications (the category that broke v1) ------
LEGIT_SEC = [
    "We noticed a new sign-in to your account from a {device} in {city}. "
    "If this was you, no action is needed.",
    "You recently requested a password reset. The link below expires in 30 "
    "minutes. If you did not request this, you can safely ignore this email.",
    "A new device was added to your account on {day}. If you don't recognise "
    "it, review your security settings.",
    "Your two-factor authentication code is {num}. It expires in 10 minutes.",
    "A transaction of {amount} was made on your card ending {dd} at {city} "
    "on {day}.",
    "Your password was changed successfully on {day}.",
    "We've updated our terms of service, effective {day}.",
    "Your passkey for {city} Workspace was registered on {day}.",
]

# --- legitimate: work / personal ------------------------------------------
LEGIT_WORK = [
    "Hi team, attaching the notes from today's standup. Let me know if I "
    "missed anything.",
    "Reminder: our 1:1 is at {time} tomorrow. Happy to move it if needed.",
    "The quarterly report is ready for review. Comments welcome by {day}.",
    "Meeting moved to {day} at {time}. Calendar has been updated.",
    "Welcome to the team! Here's your onboarding checklist for week one.",
    "Could you review the attached draft before {day}? No rush.",
    "The deployment finished at {time}. All checks passed.",
    "Lunch on {day}? There's a new place near the office.",
]

# --- legitimate: marketing / newsletters (never previously tested) --------
LEGIT_MARKETING = [
    "This week at {brand}: three new features, and a look at what's next.",
    "Our summer sale starts {day}. Up to 40% off selected items.",
    "You're invited to our webinar on {day} at {time}. Registration is free.",
    "{brand} newsletter: product updates, engineering notes, and job openings.",
    "New in {brand}: faster search, dark mode, and improved exports.",
    "Thanks for being with us this year. Here's your {brand} year in review.",
]

# --- phishing: subtle / spear (hardest category) -------------------------
PHISH_SUBTLE = [
    "Hi {name}, following up on our conversation. I've shared the updated "
    "document via the portal: https://docs-{slug}-review.com/auth/signin",
    "Please find attached invoice #{num} for services rendered. Kindly process "
    "payment to the updated account details within 3 business days.",
    "{name}, I need you to process an urgent wire transfer before end of day. "
    "I'm in meetings, so email only. Details to follow.",
    "Your {brand} mailbox is almost full and you may stop receiving emails. "
    "Revalidate your credentials using the link below to continue.",
    "HR notice: please review and acknowledge the updated payroll document at "
    "https://{slug}-hr-portal.online/login before {day}.",
]

FILL = {
    "num": lambda: str(random.randint(10000, 99999)),
    "amount": lambda: random.choice(["$45.00", "$250.00", "₹1,299", "£19.99",
                                     "$1,024.50", "₹8,450"]),
    "day": lambda: random.choice(["Monday", "Tuesday", "Thursday", "Friday",
                                  "12 August", "3 September"]),
    "dd": lambda: str(random.randint(10, 28)),
    "time": lambda: random.choice(["9:00 AM", "2pm", "3:30 PM", "18:45"]),
    "flight": lambda: random.choice(["AI302", "6E511", "BA138", "EK507"]),
    "city": lambda: random.choice(["Chennai", "Delhi", "London", "Singapore",
                                   "Toronto", "Berlin"]),
    "device": lambda: random.choice(["Windows device", "iPhone", "Android phone",
                                     "Mac"]),
    "name": lambda: random.choice(["Abhishek", "Priya", "Sam", "Alex", "Ravi"]),
    "slug": lambda: random.choice(["secure", "account", "verify", "billing",
                                   "support", "portal", "service"]),
    "brand": lambda: random.choice(BRANDS),
}


def fill(template, brand=None):
    out = template
    for key, gen in FILL.items():
        token = "{" + key + "}"
        while token in out:
            val = brand if (key == "brand" and brand) else gen()
            out = out.replace(token, val, 1)
    return out


def gen_phishing(n):
    out = set()
    combos = list(itertools.product(PHISH_OPENER, PHISH_PRETEXT,
                                    PHISH_URGENCY, PHISH_CTA))
    random.shuffle(combos)
    for opener, pretext, urgency, cta in combos:
        if len(out) >= n:
            break
        brand = random.choice(BRANDS)
        subject = f"{brand}: {pretext}"
        body = " ".join(x for x in [opener,
                                    f"We are writing to inform you that {pretext}.",
                                    urgency, fill(cta, brand)] if x)
        out.add(f"Subject: {subject}\n\n{body}")
    return list(out)


def gen_from(templates, n, subject_prefix):
    out = set()
    while len(out) < n:
        before = len(out)
        for t in templates:
            if len(out) >= n:
                break
            brand = random.choice(BRANDS)
            body = fill(t, brand)
            subject = f"{brand} - {subject_prefix}"
            out.add(f"Subject: {subject}\n\n{body}")
        if len(out) == before:
            break          # template space exhausted
    return list(out)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def preprocess_email_text(text):
    t = _html.unescape(str(text))
    if "<" in t and ">" in t:
        t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"^\s*subject\s*:\s*", "", t, flags=re.I)
    t = re.sub(r"https?://\S+|www\.\S+", " httpaddr ", t)
    return re.sub(r"\s+", " ", t).strip()[:5000]


def load_model():
    device = torch.device("cpu")
    tok = RobertaTokenizer.from_pretrained("models/roberta_tokenizer")
    mdl = RobertaForSequenceClassification.from_pretrained(
        "models/roberta_base_local", num_labels=2)
    mdl.load_state_dict(torch.load("models/roberta_phishing_model.pth",
                                   map_location=device))
    mdl.eval()
    return tok, mdl, device


def predict_batch(texts, tok, mdl, device, batch=32):
    probs = []
    for i in range(0, len(texts), batch):
        chunk = [preprocess_email_text(t) for t in texts[i:i + batch]]
        enc = tok(chunk, truncation=True, padding=True, max_length=256,
                  return_tensors="pt")
        with torch.no_grad():
            logits = mdl(input_ids=enc["input_ids"].to(device),
                         attention_mask=enc["attention_mask"].to(device)).logits
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        probs.extend(p.tolist())
    return probs


def wilson(k, n, z=1.96):
    """Wilson score interval -- correct near 0 and 1, unlike normal approx."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def evaluate(name, emails, expect_phishing, tok, mdl, device, show=5):
    probs = predict_batch(emails, tok, mdl, device)
    correct = sum(1 for p in probs
                  if (p > 0.5) == expect_phishing)
    n = len(emails)
    rate = correct / n if n else 0.0
    lo, hi = wilson(correct, n)
    label = "caught" if expect_phishing else "passed"
    print(f"\n--- {name} ---")
    print(f"  n = {n}")
    print(f"  {label}: {correct}/{n} = {rate:.1%}   "
          f"95% CI [{lo:.1%}, {hi:.1%}]")
    wrong = [(e, p) for e, p in zip(emails, probs)
             if (p > 0.5) != expect_phishing]
    if wrong:
        kind = "missed" if expect_phishing else "FALSE POSITIVES"
        print(f"  {len(wrong)} {kind}, e.g.:")
        for e, p in wrong[:show]:
            first = e.replace("\n", " ")[:78]
            print(f"    p={p:.4f}  {first}")
    return rate, lo, hi, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=150)
    args = ap.parse_args()
    N = args.per_category

    print("=" * 74)
    print("v3 email model -- expanded out-of-distribution evaluation")
    print("=" * 74)
    print("NOTE: these emails are TEMPLATED. They measure consistency across a")
    print("wide space of surface variation and detect whole-category failures.")
    print("They are NOT a random sample of real mail, so the rates below are")
    print("coverage results with intervals, not production estimates.")

    tok, mdl, device = load_model()

    sets = [
        ("Phishing -- overt (credential harvest, fraud, urgency)",
         gen_phishing(N), True),
        ("Phishing -- subtle (spear, BEC, portal lures)",
         gen_from(PHISH_SUBTLE, min(N, 90), "action required"), True),
        ("Legitimate -- transactional (broke v2)",
         gen_from(LEGIT_TXN, N, "your order"), False),
        ("Legitimate -- security notifications (broke v1)",
         gen_from(LEGIT_SEC, N, "security notice"), False),
        ("Legitimate -- work / personal",
         gen_from(LEGIT_WORK, N, "quick note"), False),
        ("Legitimate -- marketing / newsletter (never previously tested)",
         gen_from(LEGIT_MARKETING, min(N, 90), "this week"), False),
    ]

    results = []
    for name, emails, expect in sets:
        results.append((name, expect) + evaluate(name, emails, expect,
                                                 tok, mdl, device))

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    print(f"{'category':<52}{'rate':>8}{'95% CI':>18}")
    for name, expect, rate, lo, hi, n in results:
        print(f"{name[:50]:<52}{rate:>7.1%}  [{lo:.0%}, {hi:.0%}]  n={n}")

    fp_sets = [r for r in results if not r[1]]
    worst_fp = min(fp_sets, key=lambda r: r[2]) if fp_sets else None
    ph_sets = [r for r in results if r[1]]
    worst_ph = min(ph_sets, key=lambda r: r[2]) if ph_sets else None

    print()
    if worst_ph:
        print(f"  weakest phishing category:   {worst_ph[0][:44]} "
              f"{worst_ph[2]:.1%}")
    if worst_fp:
        print(f"  weakest legitimate category: {worst_fp[0][:44]} "
              f"{worst_fp[2]:.1%}")
    print()
    print("  A category scoring high here means the model is consistent across")
    print("  that phrasing space -- not that it generalises to phrasings the")
    print("  templates do not contain. That gap is what the v1 and v2")
    print("  evaluations missed.")


if __name__ == "__main__":
    main()
