#!/usr/bin/env python3
"""
PhishSentry — Gmail Takeout (.mbox) -> labeled-email CSV builder  (v2)
=====================================================================

v2 FIXES (v1 produced a corrupted CSV):
  * CRITICAL: every field is newline-stripped and the writer uses QUOTE_ALL,
    so bodies containing commas/quotes/newlines can no longer fracture rows
    or misalign columns.
  * Strips <style>/<script> blocks and CSS-looking junk (#outlook a{...},
    @media {...}) that leaked into the body text.
  * Removes zero-width / invisible characters (the "spacer" rows).
  * Skips rows whose REAL (post-clean) content is too short or mostly markup.
  * Stronger dedupe (normalized alphanumeric signature).

Runs ENTIRELY on your machine. Your mail is never uploaded.

USAGE:
    python build_email_dataset.py "path/to/All mail ...-002.mbox" \
        --name "Abhishek Jaden" --email abhishekjaden@gmail.com
"""

import sys
import os
import re
import csv
import argparse
import mailbox
import unicodedata
from email.header import decode_header
from collections import defaultdict
from html.parser import HTMLParser

CAPS = {
    "security_transactional": 400,
    "ordinary_legit": 250,
    "promo": 60,
}

SECURITY_SENDER_HINTS = [
    "accounts.google", "no-reply@google", "security", "no-reply@accounts",
    "amazonaws", "awsapps", "aws", "amazon.com", "amazon.in", "github",
    "microsoft", "accountprotection", "linkedin", "paypal", "stripe", "okta",
    "auth", "pearson", "ieee", "msr-cmt", "cmt3", "edas", "noreply",
    "no-reply", "donotreply", "notification", "billing", "invoice", "bank",
    "hdfc", "icici", "sbi", "netflix", "anthropic",
]
SECURITY_SUBJECT_HINTS = [
    "sign-in", "sign in", "signin", "verify", "verification", "security alert",
    "new device", "password", "reset", "two-factor", "2fa", "otp", "code",
    "confirm", "account", "suspicious", "unusual", "login", "log in",
    "invoice", "receipt", "statement", "payment", "order", "shipped",
    "delivery", "expire", "expiring", "action required", "authorize",
    "authentication", "passkey", "recovery", "deactivat", "suspend",
    "support case", "case ", "correspondence",
]
PROMO_HINTS = [
    "newsletter", "marketing", "promo", "% off", "deal", "sale", "webinar",
    "digest", "weekly update", "subscribe", "campaign", "off everything",
    "last chance", "discount",
]

INVISIBLE = dict.fromkeys(map(ord, [
    "\u200b", "\u200c", "\u200d", "\u200e", "\u200f", "\ufeff",
    "\u2060", "\u00ad", "\u034f", "\u2061", "\u2062", "\u2063",
    "\u2064", "\u115f", "\u1160", "\u3164", "\uffa0",
]), None)


class _HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self._skip = False
    def handle_starttag(self, tag, attrs):
        if tag in ("style", "script", "head"):
            self._skip = True
    def handle_endtag(self, tag):
        if tag in ("style", "script", "head"):
            self._skip = False
    def handle_data(self, d):
        if not self._skip:
            self.parts.append(d)
    def text(self):
        return " ".join(self.parts)


def strip_html(html):
    try:
        s = _HTMLStripper()
        s.feed(html)
        return s.text()
    except Exception:
        html = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.S | re.I)
        html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.S | re.I)
        return re.sub(r"<[^>]+>", " ", html)


def kill_css(text):
    text = re.sub(r"@(?:media|import|font-face)[^{]*\{[^}]*\}", " ", text, flags=re.S)
    text = re.sub(r"[#.\[][A-Za-z0-9_\-\]\[\"'=:\s\^~]+\{[^}]*\}", " ", text, flags=re.S)
    text = re.sub(r"[A-Za-z\-]+\s*:\s*[^;{}\n]{1,40};", " ", text)
    text = re.sub(r"\{[^}]*\}", " ", text)
    return text


def clean_text(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(INVISIBLE)
    text = kill_css(text)
    text = re.sub(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", "[EMAIL]", text)
    text = re.sub(r"\b\d{5,}\b", "[NUM]", text)
    text = re.sub(r"(https?://[^\s/]+)[^\s]*", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def redact_personal(text, user_name, user_email):
    if user_email:
        text = text.replace(user_email, "[EMAIL]")
    if user_name:
        for token in [user_name] + user_name.split():
            if len(token) > 2:
                text = re.sub(re.escape(token), "[NAME]", text, flags=re.IGNORECASE)
    return text


def looks_like_boilerplate(text):
    if not text:
        return True
    letters = sum(c.isalpha() for c in text)
    if letters < 60:
        return True
    punct = sum(c in "{}:;#.[]()=<>/*" for c in text)
    if punct > letters * 0.5:
        return True
    if text.count("!important") >= 2 or "ReadMsgBody" in text or "ExternalClass" in text:
        return True
    return False


def signature(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())[:120]


def decode_hdr(value):
    if not value:
        return ""
    try:
        out = []
        for txt, enc in decode_header(value):
            if isinstance(txt, bytes):
                out.append(txt.decode(enc or "utf-8", errors="replace"))
            else:
                out.append(txt)
        return "".join(out)
    except Exception:
        return str(value)


def get_body(msg):
    if msg.is_multipart():
        plain, html = "", ""
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = str(part.get("Content-Disposition") or "")
            if "attachment" in disp:
                continue
            try:
                payload = part.get_payload(decode=True)
                if payload is None:
                    continue
                charset = part.get_content_charset() or "utf-8"
                text = payload.decode(charset, errors="replace")
            except Exception:
                continue
            if ctype == "text/plain" and not plain:
                plain = text
            elif ctype == "text/html" and not html:
                html = text
        return plain or strip_html(html)
    else:
        try:
            payload = msg.get_payload(decode=True)
            charset = msg.get_content_charset() or "utf-8"
            text = payload.decode(charset, errors="replace") if payload else ""
        except Exception:
            text = msg.get_payload() or ""
        return text if msg.get_content_type() == "text/plain" else strip_html(text)


def sender_domain(from_hdr):
    m = re.search(r"@([a-zA-Z0-9.\-]+)", from_hdr or "")
    return m.group(1).lower() if m else ""


def categorize(domain, subject, body):
    hay = f"{domain} {subject}".lower()
    body_l = (body or "")[:400].lower()
    if any(h in hay for h in PROMO_HINTS) and not any(h in hay for h in SECURITY_SUBJECT_HINTS):
        return "promo"
    if any(h in domain for h in SECURITY_SENDER_HINTS) or \
       any(h in hay for h in SECURITY_SUBJECT_HINTS) or \
       any(h in body_l for h in SECURITY_SUBJECT_HINTS):
        return "security_transactional"
    return "ordinary_legit"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mbox_path")
    ap.add_argument("--name", default=None)
    ap.add_argument("--email", default=None)
    ap.add_argument("--out", default="emails_for_review.csv")
    ap.add_argument("--min-len", type=int, default=150)
    args = ap.parse_args()

    if not os.path.exists(args.mbox_path):
        sys.exit(f"ERROR: file not found: {args.mbox_path}")

    print(f"[open] {args.mbox_path}")
    mbox = mailbox.mbox(args.mbox_path)

    counts = defaultdict(int)
    kept = 0
    skipped_boiler = 0
    skipped_dupe = 0
    seen = set()
    rows = []

    for i, msg in enumerate(mbox):
        if i % 500 == 0:
            print(f"  processed {i}, kept {kept}, dropped(boiler {skipped_boiler}/dupe {skipped_dupe})...", end="\r")
        try:
            subject = decode_hdr(msg.get("Subject", ""))
            from_hdr = decode_hdr(msg.get("From", ""))
            domain = sender_domain(from_hdr)
            raw_body = get_body(msg)

            body = clean_text(raw_body)
            body = redact_personal(body, args.name, args.email)
            subj = redact_personal(clean_text(subject), args.name, args.email)

            if len(body) < args.min_len:
                continue
            if looks_like_boilerplate(body):
                skipped_boiler += 1
                continue

            sig = signature(body)
            if sig in seen:
                skipped_dupe += 1
                continue
            seen.add(sig)

            cat = categorize(domain, subj, body)
            if counts[cat] >= CAPS.get(cat, 0):
                continue

            rows.append({
                "text": body[:1500],
                "sender_domain": domain,
                "subject": subj[:200],
                "auto_category": cat,
                "verified_label": "",
            })
            counts[cat] += 1
            kept += 1
        except Exception:
            continue

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["text", "sender_domain", "subject", "auto_category", "verified_label"],
            quoting=csv.QUOTE_ALL,
        )
        w.writeheader()
        w.writerows(rows)

    print(f"\n[done] wrote {kept} clean emails to {args.out}")
    print(f"[dropped] boilerplate/markup-only: {skipped_boiler}, duplicates: {skipped_dupe}")
    print("[breakdown]")
    for c, n in counts.items():
        print(f"    {c:<26}: {n}")
    print("\nNEXT: open emails_for_review.csv, fill 'verified_label': legit / spam / skip.")
    print("Focus on the security_transactional rows (the model's failure category).")


if __name__ == "__main__":
    main()
