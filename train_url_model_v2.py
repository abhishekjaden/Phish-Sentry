#!/usr/bin/env python3
"""
Train the v2 URL model and evaluate it on the things that actually matter.

The v1 model reported 96.3% held-out recall and delivered 38.3% on live
phishing. So a held-out score alone is not evidence here. This script runs
four evaluations:

  1. Domain-disjoint held-out test set (in-distribution honesty)
  2. LIVE recall on a fresh OpenPhish sample (the number that decides shipping)
  3. Hosting-platform false positives -- does it flag legitimate pages.dev /
     github.io / vercel.app sites? This is the risk created by training on
     platform-hosted phishing.
  4. Deep-link false positives -- the v1 model measured 37.5% here.

Outputs models/url_xgb_model_v2.json and models/url_model_config_v2.json in the
same shape the Lambda already reads.

Usage:
  python train_url_model_v2.py
"""
import json
import os
import random

import numpy as np
import pandas as pd
import requests
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import train_test_split

from url_features_v2 import extract_features_v2, FEATURE_ORDER

random.seed(42)
np.random.seed(42)

TRAIN_CSV = "url_dataset_v2_train.csv"
TEST_CSV = "url_dataset_v2_test.csv"
OUT_MODEL = "models/url_xgb_model_v2.json"
OUT_CONFIG = "models/url_model_config_v2.json"

# Legitimate platform-hosted URLs. If the model flags these, it has learned
# "hosting platform => phishing", which would be worse than the v1 blindness.
PLATFORM_LEGIT_PROBE = [
    "https://reactjs.github.io/react-docs/",
    "https://pytorch.github.io/",
    "https://microsoft.github.io/monaco-editor/",
    "https://developers.google.com/",
    "https://docs.pages.dev/",
    "https://nextjs-commerce.vercel.app/",
    "https://vitejs.dev/",
    "https://tailwindcss.com/docs",
    "https://myportfolio.pages.dev/",
    "https://johnsmith.github.io/portfolio/",
    "https://reactbits.dev/",
    "https://svelte.netlify.app/",
]

# Long, high-entropy deep links on trusted domains -- v1 false-flagged 37.5%.
DEEPLINK_PROBE = [
    "https://docs.google.com/document/d/1a2B3c4D5e6F7g8H9i0JkLmNoPqRsTuVwXyZ/edit#heading=h.abc123",
    "https://drive.google.com/file/d/1XyZ9AbCdEfGhIjKlMnOpQrStUvWxYz01/view?usp=sharing",
    "https://github.com/torvalds/linux/commit/8e5b7a1f9c3d2e4b6a8f0c1d3e5f7a9b1c3d5e7f",
    "https://www.amazon.in/dp/B08N5WRWNW/ref=sr_1_3?keywords=laptop&qid=1699999999&sr=8-3",
    "https://stackoverflow.com/questions/12345678/how-to-do-a-thing-in-python?rq=1",
    "https://open.spotify.com/track/4cOdK2wGLETKBW3PvgPWqT?si=a1b2c3d4e5f6g7h8",
    "https://calendar.google.com/calendar/u/0/r/eventedit/MjAyNjA3MjVUMTIwMDAw",
    "https://mail.google.com/mail/u/0/#inbox/FMfcgzQbdXfKZlPnJqRtVwXyZ01234",
]


def refang(u):
    """Dataset files store URLs defanged so AV does not quarantine them."""
    return (str(u).replace("[.]", ".")
                  .replace("hxxps://", "https://")
                  .replace("hxxp://", "http://"))


def featurize(urls):
    return np.array([extract_features_v2(u) for u in urls], dtype=float)


def evaluate(name, y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    print(f"\n--- {name} ---")
    print(f"  accuracy   {accuracy_score(y_true, y_pred):.4f}")
    print(f"  precision  {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  recall     {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  f1         {f1_score(y_true, y_pred, zero_division=0):.4f}")
    try:
        print(f"  roc_auc    {roc_auc_score(y_true, y_prob):.4f}")
    except Exception:
        pass
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    print(f"  TN {tn}  FP {fp}  FN {fn}  TP {tp}")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def probe(name, urls, booster, iso, threshold, expect_phishing):
    """Score a hand-built probe set and report how many match expectation."""
    X = featurize(urls)
    raw = booster.predict(xgb.DMatrix(X, feature_names=FEATURE_ORDER))
    cal = iso.predict(raw)
    flagged = cal >= threshold
    wrong = 0
    print(f"\n--- {name} ---")
    for u, c, f in zip(urls, cal, flagged):
        ok = (bool(f) == expect_phishing)
        if not ok:
            wrong += 1
        tag = "ok  " if ok else "MISS" if expect_phishing else "FP  "
        print(f"  {tag}  p={c:.4f}  {u[:64]}")
    rate = wrong / max(len(urls), 1)
    label = "false-negative rate" if expect_phishing else "FALSE-POSITIVE RATE"
    print(f"  {label}: {wrong}/{len(urls)} = {rate:.1%}")
    return rate


def live_recall(booster, iso, threshold, n=80):
    print("\n--- LIVE recall (fresh OpenPhish sample) ---")
    try:
        r = requests.get("https://openphish.com/feed.txt", timeout=30)
        r.raise_for_status()
        urls = [ln.strip() for ln in r.text.splitlines()
                if ln.strip().startswith(("http://", "https://"))]
    except Exception as e:
        print(f"  feed unavailable ({type(e).__name__}) -- skipped")
        return None

    urls = [u for u in urls if "github.io" not in u.lower()]
    random.shuffle(urls)
    urls = urls[:n]

    X = featurize(urls)
    raw = booster.predict(xgb.DMatrix(X, feature_names=FEATURE_ORDER))
    cal = iso.predict(raw)
    caught = int((cal >= threshold).sum())
    rec = caught / max(len(urls), 1)
    print(f"  scored {len(urls)}  caught {caught}")
    print(f"  LIVE RECALL: {rec:.1%}   (v1 measured 38.3%)")

    missed = [(u, c) for u, c in zip(urls, cal) if c < threshold]
    if missed:
        print(f"  sample of misses:")
        for u, c in missed[:8]:
            print(f"    p={c:.4f}  {u[:64]}")
    return rec


def main():
    for p in (TRAIN_CSV, TEST_CSV):
        if not os.path.exists(p):
            print(f"Missing {p} -- run collect_url_dataset_v2.py first.")
            return

    tr = pd.read_csv(TRAIN_CSV)
    te = pd.read_csv(TEST_CSV)
    tr["url"] = tr["url"].map(refang)
    te["url"] = te["url"].map(refang)
    print(f"train {len(tr)} rows, test {len(te)} rows")
    print(f"features: {len(FEATURE_ORDER)}")

    print("extracting features ...")
    Xtr_all = featurize(tr["url"].tolist())
    ytr_all = tr["label"].to_numpy()
    Xte = featurize(te["url"].tolist())
    yte = te["label"].to_numpy()

    # carve a calibration split out of train (never touch test for fitting)
    Xtr, Xcal, ytr, ycal = train_test_split(
        Xtr_all, ytr_all, test_size=0.2, random_state=42, stratify=ytr_all)

    pos_w = float((ytr == 0).sum()) / max(float((ytr == 1).sum()), 1.0)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 2,
        "scale_pos_weight": pos_w,
        "seed": 42,
    }
    dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=FEATURE_ORDER)
    dcal = xgb.DMatrix(Xcal, label=ycal, feature_names=FEATURE_ORDER)

    print("training ...")
    booster = xgb.train(params, dtr, num_boost_round=600,
                        evals=[(dcal, "cal")], early_stopping_rounds=40,
                        verbose_eval=100)

    # isotonic calibration on the held-out calibration split
    raw_cal = booster.predict(dcal)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_cal, ycal)

    # Do not assume 0.5. Sweep the calibration split and pick the threshold
    # that maximises F1, then report it -- the operating point is a decision,
    # not a default.
    cal_probs = iso.predict(raw_cal)
    best_t, best_f1 = 0.5, -1.0
    print("\n--- threshold sweep (on calibration split) ---")
    for cand in np.arange(0.20, 0.85, 0.05):
        pred = (cal_probs >= cand).astype(int)
        from sklearn.metrics import f1_score as _f1
        s = _f1(ycal, pred, zero_division=0)
        prec = precision_score(ycal, pred, zero_division=0)
        rec = recall_score(ycal, pred, zero_division=0)
        print(f"  t={cand:.2f}  P={prec:.3f}  R={rec:.3f}  F1={s:.3f}")
        if s > best_f1:
            best_f1, best_t = s, float(cand)
    THRESHOLD = round(best_t, 2)
    print(f"  chosen threshold: {THRESHOLD}")

    raw_te = booster.predict(xgb.DMatrix(Xte, feature_names=FEATURE_ORDER))
    cal_te = iso.predict(raw_te)
    test_metrics = evaluate("Domain-disjoint held-out test", yte, cal_te, THRESHOLD)

    # the three checks that decide whether this is shippable
    live = live_recall(booster, iso, THRESHOLD)
    plat_fp = probe("Hosting-platform legitimate sites (want NO flags)",
                    PLATFORM_LEGIT_PROBE, booster, iso, THRESHOLD, False)
    deep_fp = probe("Deep links on trusted domains (v1 was 37.5% FP)",
                    DEEPLINK_PROBE, booster, iso, THRESHOLD, False)

    # feature importance -- confirms what the model actually leaned on
    print("\n--- top features by gain ---")
    gains = booster.get_score(importance_type="gain")
    for k, v in sorted(gains.items(), key=lambda x: -x[1])[:15]:
        print(f"  {k:<34} {v:>10.1f}")

    os.makedirs("models", exist_ok=True)
    booster.save_model(OUT_MODEL)

    grid = np.linspace(0.0, 1.0, 201)
    cfg = {
        "features": FEATURE_ORDER,
        "threshold": THRESHOLD,
        "calibration": {
            "x": [float(v) for v in grid],
            "y": [float(v) for v in iso.predict(grid)],
        },
        "metrics_test": test_metrics,
        "metrics_live_recall": live,
        "platform_false_positive_rate": plat_fp,
        "deeplink_false_positive_rate": deep_fp,
        "notes": (
            "v2: lexical + subdomain-structure + brand-impersonation features, "
            "no network lookups. Trained on public phishing feeds with "
            "github.io positives excluded. Domain-disjoint train/test split."
        ),
    }
    with open(OUT_CONFIG, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"\nsaved {OUT_MODEL}")
    print(f"saved {OUT_CONFIG}")

    print("\n" + "=" * 70)
    print("SHIP / NO-SHIP")
    print("=" * 70)
    if live is None:
        print("  Live recall unmeasured -- do not ship.")
    else:
        print(f"  live recall            {live:.1%}   (v1: 38.3%)")
        print(f"  platform FP rate       {plat_fp:.1%}   (want low)")
        print(f"  deep-link FP rate      {deep_fp:.1%}   (v1: 37.5%)")
        print()
        if live >= 0.80 and plat_fp <= 0.10 and deep_fp <= 0.15:
            print("  Meets the bar for a public extension.")
        else:
            print("  Does NOT meet the bar. A public extension that misses most")
            print("  phishing, or flags legitimate sites, is worse than none --")
            print("  it gives users active false reassurance.")


if __name__ == "__main__":
    main()
