#!/usr/bin/env python3
"""
PhishSentry — legit-URL feature extractor for URL-model retraining
==================================================================

Reads a list of legitimate URLs (one per line; # comments / blanks ignored),
runs each through the SAME `url_features.py extract_features()` the production
model uses, and writes a labeled CSV of the 20 features with phishing=0.

This is the URL analogue of the email dataset builder. Output combines with the
original GregaVrbancic training data (reduced to the same 20 features) to retrain
the XGBoost model with more modern legitimate examples.

USAGE (from the project root, in your venv so url_features imports work):
    python extract_legit_url_features.py legit_urls.txt

  Optional:
    --out legit_url_features.csv     output path
    --resume                         skip URLs already in the output (re-runnable)

HONEST NOTE ON SPEED: each URL triggers live WHOIS + DNS + ASN + HTTP lookups,
each hard-capped by url_features.py's timeouts. Expect ~3-8 seconds per URL, so
276 URLs ~= 15-40 minutes. Failed/timed-out lookups return -1 (matching the
training data's sentinel) — that's expected and fine, not an error.
"""

import sys
import os
import csv
import argparse
import time

# import the EXACT production extractor
try:
    from url_features import extract_features, FEATURE_ORDER
except ImportError:
    sys.exit("ERROR: run this from the project root (where url_features.py lives), "
             "with your venv active so its deps (whois, dns, tldextract, requests) import.")


def load_urls(path):
    urls = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            urls.append(line)
    return urls


def load_done(out_path):
    """For --resume: URLs already extracted."""
    if not os.path.exists(out_path):
        return set()
    done = set()
    with open(out_path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("url"):
                done.add(row["url"])
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url_list")
    ap.add_argument("--out", default="legit_url_features.csv")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    urls = load_urls(args.url_list)
    done = load_done(args.out) if args.resume else set()
    todo = [u for u in urls if u not in done]

    print(f"[urls] {len(urls)} total, {len(done)} already done, {len(todo)} to extract")
    if not todo:
        print("Nothing to do.")
        return

    # open in append mode if resuming, else fresh with header
    fresh = not (args.resume and os.path.exists(args.out))
    mode = "a" if (args.resume and os.path.exists(args.out)) else "w"
    header = list(FEATURE_ORDER) + ["phishing", "url"]

    with open(args.out, mode, newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if fresh:
            w.writerow(header)

        t_start = time.time()
        ok, failed = 0, 0
        for i, url in enumerate(todo, 1):
            t0 = time.time()
            try:
                feats = extract_features(url)          # 20 features, -1 on failed lookups
                if len(feats) != 20:
                    feats = (feats + [-1] * 20)[:20]
                # how many features came back as the -1 sentinel (lookup failures)
                n_sentinel = sum(1 for v in feats if v == -1)
                w.writerow(list(feats) + [0, url])     # phishing=0 (all legit)
                f.flush()                              # write as we go (resumable)
                ok += 1
                dt = time.time() - t0
                print(f"  [{i}/{len(todo)}] {dt:4.1f}s  sentinels={n_sentinel:2d}  {url[:60]}")
            except Exception as e:
                failed += 1
                print(f"  [{i}/{len(todo)}] FAILED  {url[:60]}  ({e})")

    total_dt = time.time() - t_start
    print(f"\n[done] extracted {ok} URLs ({failed} failed) in {total_dt/60:.1f} min")
    print(f"[out] {args.out}")
    print("\nNEXT: this CSV (20 features + phishing=0) combines with the original")
    print("GregaVrbancic training data to retrain the XGBoost URL model.")


if __name__ == "__main__":
    main()
