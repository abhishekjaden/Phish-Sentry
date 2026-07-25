#!/usr/bin/env python3
"""
Measure the URL model's RECALL on live, verified phishing URLs.

Why this exists: the extension is URL-only. Its usefulness depends entirely on
how much live phishing it actually catches -- a number that has never been
measured. Small hand-picked test sets can't answer it; a live feed can.

Source: OpenPhish community feed (https://openphish.com/feed.txt) -- a public
list of currently-active verified phishing URLs, published for exactly this
purpose. PhishTank is an alternative but now requires an API key.

SAFETY NOTE: extract_features() performs an HTTP GET against each URL to
measure time_response. That means this script fetches live phishing pages from
your IP. It only issues a plain GET -- no credentials, no form submission, no
JS execution -- which is standard practice for defender evaluation. But be aware
it is happening, and don't run it on a network where that's a problem.

Usage:
  python measure_url_recall.py              # fetch feed, sample 60
  python measure_url_recall.py 120          # sample 120
  python measure_url_recall.py 60 feed.txt  # use a local feed file
"""
import json
import os
import random
import sys
import time

import numpy as np
import requests
import xgboost as xgb

import url_resolver
from url_features import extract_features

FEED_URL = "https://openphish.com/feed.txt"
NETWORK_FEATS = [
    "time_response", "asn_ip", "time_domain_activation",
    "time_domain_expiration", "ttl_hostname",
]

_here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_here, "models", "url_model_config.json")) as f:
    CFG = json.load(f)
FEATURES = CFG["features"]
THRESHOLD = float(CFG["threshold"])
CALIB_X = np.array(CFG["calibration"]["x"], dtype=float)
CALIB_Y = np.array(CFG["calibration"]["y"], dtype=float)

booster = xgb.Booster()
booster.load_model(os.path.join(_here, "models", "url_xgb_model.json"))


def score(raw_url):
    resolved = url_resolver.resolve_final_url(raw_url)
    target = resolved["final_url"]
    feats = extract_features(target)
    dm = xgb.DMatrix(np.array([feats], dtype=float), feature_names=FEATURES)
    raw = float(booster.predict(dm)[0])
    calibrated = float(np.interp(raw, CALIB_X, CALIB_Y))
    sentinels = sum(
        1 for n in NETWORK_FEATS
        if n in FEATURES and float(feats[FEATURES.index(n)]) == -1.0
    )
    return calibrated, sentinels


def load_feed(sample_n, local_path=None):
    if local_path:
        with open(local_path, encoding="utf-8", errors="ignore") as f:
            urls = [ln.strip() for ln in f if ln.strip()]
        print(f"Loaded {len(urls)} URLs from {local_path}")
    else:
        print(f"Fetching {FEED_URL} ...")
        r = requests.get(FEED_URL, timeout=30)
        r.raise_for_status()
        urls = [ln.strip() for ln in r.text.splitlines() if ln.strip()]
        print(f"Feed returned {len(urls)} live phishing URLs")

    urls = [u for u in urls if u.lower().startswith(("http://", "https://"))]
    random.seed(42)  # reproducible sample
    if len(urls) > sample_n:
        urls = random.sample(urls, sample_n)
    return urls


def main():
    sample_n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    local_path = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        urls = load_feed(sample_n, local_path)
    except Exception as e:
        print(f"Could not load feed ({type(e).__name__}: {e}).")
        print("Download it manually to feed.txt and re-run:")
        print(f"  curl -s {FEED_URL} -o feed.txt")
        print("  python measure_url_recall.py 60 feed.txt")
        return

    print(f"Scoring {len(urls)} URLs (threshold {THRESHOLD}). This is slow -- "
          f"each one does live WHOIS/DNS/HTTP lookups.\n")

    caught = missed = errored = 0
    by_sentinel = {i: {"caught": 0, "total": 0} for i in range(6)}
    scores = []

    for i, u in enumerate(urls, 1):
        try:
            p, sent = score(u)
        except Exception as e:
            errored += 1
            print(f"  [{i:>3}/{len(urls)}] ERROR {type(e).__name__}")
            continue

        scores.append(p)
        hit = p >= THRESHOLD
        by_sentinel[sent]["total"] += 1
        if hit:
            caught += 1
            by_sentinel[sent]["caught"] += 1
        else:
            missed += 1

        tag = "CAUGHT" if hit else "MISSED"
        print(f"  [{i:>3}/{len(urls)}] {tag}  p={p:.4f}  missing={sent}/5  {u[:52]}")
        time.sleep(0.2)  # be polite to WHOIS/DNS servers

    scored = caught + missed
    print("\n" + "=" * 74)
    print("RESULTS -- URL model on live verified phishing")
    print("=" * 74)
    if scored == 0:
        print("  Nothing scored successfully.")
        return

    print(f"  scored:  {scored}   (errors: {errored})")
    print(f"  caught:  {caught}")
    print(f"  missed:  {missed}")
    print(f"  RECALL:  {caught / scored:.1%}")
    if scores:
        print(f"  median calibrated score: {sorted(scores)[len(scores) // 2]:.4f}")

    print("\n  Recall broken down by how many network features resolved:")
    print("  (missing=0 means WHOIS/DNS/ASN/timing all succeeded)")
    for s in range(6):
        t = by_sentinel[s]["total"]
        if t:
            c = by_sentinel[s]["caught"]
            print(f"    missing {s}/5:  {c}/{t} caught  ({c / t:.0%})")

    print("\n  Interpretation:")
    print("  - Recall here is on LIVE phishing, which is the number that matters")
    print("    for a browser extension. Compare it to the 0.963 held-out recall")
    print("    in EVALUATION.md -- a large gap means the held-out split was")
    print("    easier than reality, which is worth documenting.")
    print("  - If recall is high only in the missing=0 rows, the model depends")
    print("    on live lookups succeeding, and degrades when they don't.")


if __name__ == "__main__":
    main()
