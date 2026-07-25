#!/usr/bin/env python3
"""
Verify the Lambda gives the SAME verdicts as the local production code path,
and measure how often the 5 network-dependent features fall back to -1.

Usage:
  # local baseline only
  python test_lambda_parity.py

  # local baseline + compare against deployed Lambda
  python test_lambda_parity.py https://xxxx.execute-api.ap-south-1.amazonaws.com/check

Run from the project root with myenv active.
"""
import json
import os
import sys

import numpy as np
import xgboost as xgb
import requests

import url_resolver
import homograph
from url_features import extract_features

TEST_URLS = [
    # legitimate, should come back safe
    "https://github.com",
    "https://www.google.com",
    "https://en.wikipedia.org/wiki/Phishing",
    "https://aws.amazon.com/lambda/",
    "https://www.anna.univ.ac.in",
    # the known-hard case from your own evaluation: long high-entropy deep link
    "https://docs.google.com/document/d/1a2B3c4D5e6F7g8H9i0JkLmNoPqRsTuVwXyZ/edit#heading=h.abc123",
    # lookalike / suspicious
    "http://paypa1-secure.com/account/verify",
    "http://secure-verify.tk/login",
    "http://appleid-support-login.com/signin",
]

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


def score_local(raw_url):
    resolved = url_resolver.resolve_final_url(raw_url)
    target = resolved["final_url"]
    feats = extract_features(target)
    dm = xgb.DMatrix(np.array([feats], dtype=float), feature_names=FEATURES)
    raw = float(booster.predict(dm)[0])
    calibrated = float(np.interp(raw, CALIB_X, CALIB_Y))
    model_is_phishing = bool(calibrated >= THRESHOLD)
    homo = homograph.analyze(target)
    is_phishing = True if homo["is_homograph"] else model_is_phishing
    sentinels = sum(
        1 for n in NETWORK_FEATS
        if n in FEATURES and float(feats[FEATURES.index(n)]) == -1.0
    )
    return {
        "is_phishing": is_phishing,
        "prediction_value": calibrated,
        "degraded_features": sentinels,
    }


def score_remote(endpoint, raw_url):
    r = requests.post(endpoint, json={"url": raw_url}, timeout=40)
    r.raise_for_status()
    return r.json()


def main():
    endpoint = sys.argv[1] if len(sys.argv) > 1 else None

    print("=" * 74)
    print("LOCAL baseline (production code path)")
    print("=" * 74)
    local = {}
    total_sent = 0
    for u in TEST_URLS:
        try:
            res = score_local(u)
            local[u] = res
            total_sent += res["degraded_features"]
            verdict = "PHISH" if res["is_phishing"] else "safe "
            print(f"  {verdict}  p={res['prediction_value']:.4f}  "
                  f"missing_net_feats={res['degraded_features']}/5  {u[:58]}")
        except Exception as e:
            print(f"  ERROR  {type(e).__name__}  {u[:58]}")

    n = max(len(local), 1)
    print(f"\n  average missing network features: {total_sent / n:.2f} of 5")
    print("  (high numbers mean WHOIS/DNS/ASN lookups are failing and the")
    print("   verdict is resting mostly on lexical structure)")

    if not endpoint:
        print("\nNo endpoint given -- skipping Lambda parity check.")
        return

    print()
    print("=" * 74)
    print("LAMBDA parity check")
    print("=" * 74)
    mismatches = 0
    remote_sent = 0
    for u in TEST_URLS:
        if u not in local:
            continue
        try:
            r = score_remote(endpoint, u)
        except Exception as e:
            print(f"  ERROR  {type(e).__name__}  {u[:58]}")
            mismatches += 1
            continue
        remote_sent += r.get("degraded_features", 0)
        same_verdict = bool(r["is_phishing"]) == bool(local[u]["is_phishing"])
        dp = abs(float(r["prediction_value"]) - local[u]["prediction_value"])
        flag = "OK  " if same_verdict else "DIFF"
        if not same_verdict:
            mismatches += 1
        print(f"  {flag}  local={local[u]['is_phishing']!s:5} "
              f"lambda={r['is_phishing']!s:5}  dp={dp:.4f}  "
              f"missing={r.get('degraded_features')}/5  {u[:40]}")

    print(f"\n  average missing network features (Lambda): "
          f"{remote_sent / n:.2f} of 5")
    print()
    if mismatches == 0:
        print("  PARITY: PASS -- Lambda verdicts match the local production path.")
    else:
        print(f"  PARITY: {mismatches} MISMATCH(ES) -- do not ship until resolved.")
        print("  Note: differences in prediction_value are expected when WHOIS/DNS")
        print("  results differ between your machine and Lambda's network. Verdict")
        print("  flips are the thing that matters.")


if __name__ == "__main__":
    main()
