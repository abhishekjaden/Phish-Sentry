#!/usr/bin/env python3
"""
PhishSentry — URL model (XGBoost) evaluation with rigor
=======================================================

Fills the evaluation gap for the retrained URL model. Produces honest,
defensible numbers including:

  [1] Held-out test metrics on the GregaVrbancic dataset, WITH bootstrap
      95% confidence intervals (not just point estimates).
  [2] False-positive rate on 276 modern legitimate URLs (the retrain target).
  [3] DEEP-LINK CHARACTERIZATION — the known limitation, quantified: take
      legitimate domains, append realistic deep-link paths (session IDs,
      UUIDs, query strings), and measure how often the model false-flags them.
      This turns "we know deep links sometimes fail" into a measured number.
  [4] Baseline comparison — a naive heuristic (long URL / young domain) vs
      the model, so the model's value is shown in context.

Run locally (needs the trained model + deps):
    source myenv/Scripts/activate
    pip install xgboost scikit-learn pandas numpy --quiet
    python evaluate_url_model.py

Notes:
- [1] re-fetches the GregaVrbancic dataset for a held-out split (same source
  as training; we re-derive a test split with a FIXED seed so it's reproducible
  but this is in-distribution — reported honestly as such).
- [3] uses the LIVE feature extractor (url_features.py) so the deep-link URLs
  get real features — this is the honest test that matches production behavior.
"""

import json
import os
import sys
import numpy as np
import pandas as pd

MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
CONFIG = os.path.join(MODELS, "url_model_config.json")
MODEL = os.path.join(MODELS, "url_xgb_model.json")
GREGA = ("https://raw.githubusercontent.com/GregaVrbancic/"
         "Phishing-Dataset/master/dataset_full.csv")


def load_model_and_config():
    import xgboost as xgb
    with open(CONFIG) as f:
        cfg = json.load(f)
    booster = xgb.Booster()
    booster.load_model(MODEL)
    return booster, cfg


def calibrate(raw, cfg):
    """Apply the config's calibration curve, matching inference_service.py."""
    xs = np.array(cfg["calibration"]["x"], dtype=float)
    ys = np.array(cfg["calibration"]["y"], dtype=float)
    return np.interp(raw, xs, ys)


def predict(booster, cfg, X, features):
    import xgboost as xgb
    dm = xgb.DMatrix(np.asarray(X, dtype=float), feature_names=features)
    raw = booster.predict(dm)
    calib = calibrate(raw, cfg)
    thr = float(cfg.get("threshold", 0.5))
    return calib, (calib > thr).astype(int)


def bootstrap_ci(y_true, y_pred, metric_fn, n=2000, seed=42):
    """95% bootstrap CI for a metric."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    vals = []
    idx = np.arange(len(y_true))
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        try:
            vals.append(metric_fn(y_true[s], y_pred[s]))
        except Exception:
            pass
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return lo, hi


def main():
    from sklearn.metrics import (precision_score, recall_score, f1_score,
                                 accuracy_score, roc_auc_score, confusion_matrix)

    print("=" * 72)
    print("PhishSentry — URL model (XGBoost) evaluation")
    print("=" * 72)

    booster, cfg = load_model_and_config()
    FEATURES = cfg["features"]
    print(f"[load] model OK — {len(FEATURES)} features, threshold {cfg['threshold']}")

    # ------------------------------------------------------------------
    # [1] Held-out test set (in-distribution) with bootstrap CIs
    # ------------------------------------------------------------------
    print("\n[1] HELD-OUT TEST SET (GregaVrbancic, in-distribution)")
    full = pd.read_csv(GREGA)
    from sklearn.model_selection import train_test_split
    X = full[FEATURES].values.astype(float)
    y = full["phishing"].values.astype(int)
    # same split the retrain used (test_size=0.2, seed 42) -> the held-out 20%
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.2, random_state=42,
                                        stratify=y)
    calib, pred = predict(booster, cfg, X_te, FEATURES)

    acc = accuracy_score(y_te, pred)
    prec = precision_score(y_te, pred, zero_division=0)
    rec = recall_score(y_te, pred, zero_division=0)
    f1 = f1_score(y_te, pred, zero_division=0)
    auc = roc_auc_score(y_te, calib)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()

    p_lo, p_hi = bootstrap_ci(y_te, pred, lambda a, b: precision_score(a, b, zero_division=0))
    r_lo, r_hi = bootstrap_ci(y_te, pred, lambda a, b: recall_score(a, b, zero_division=0))

    print(f"    n = {len(y_te)}  (legit={int((y_te==0).sum())}, phishing={int((y_te==1).sum())})")
    print(f"    accuracy : {acc:.4f}")
    print(f"    precision: {prec:.4f}   95% CI [{p_lo:.4f}, {p_hi:.4f}]")
    print(f"    recall   : {rec:.4f}   95% CI [{r_lo:.4f}, {r_hi:.4f}]")
    print(f"    f1       : {f1:.4f}")
    print(f"    ROC-AUC  : {auc:.4f}")
    print(f"    confusion: TN={tn} FP={fp} FN={fn} TP={tp}")

    # ------------------------------------------------------------------
    # [2] Modern legitimate URLs — false-positive rate
    # ------------------------------------------------------------------
    print("\n[2] MODERN LEGITIMATE URLs (the retrain target)")
    if os.path.exists("legit_url_features.csv"):
        legit = pd.read_csv("legit_url_features.csv")
        Xl = legit[FEATURES].values.astype(float)
        _, predl = predict(booster, cfg, Xl, FEATURES)
        fp_rate = predl.mean()
        print(f"    n = {len(legit)} legit URLs")
        print(f"    false positives: {int(predl.sum())}/{len(legit)} = {fp_rate*100:.1f}%")
    else:
        print("    (legit_url_features.csv not found — skip)")

    # ------------------------------------------------------------------
    # [3] DEEP-LINK CHARACTERIZATION — quantify the known limitation
    # ------------------------------------------------------------------
    print("\n[3] DEEP-LINK FALSE-POSITIVE CHARACTERIZATION (known limitation)")
    print("    Appending realistic deep-link paths to trusted domains and")
    print("    measuring false-positive rate. This quantifies the limitation")
    print("    observed live (e.g. claude.ai/chat/<uuid> flagged as phishing).")
    try:
        from url_features import extract_features
        import uuid
        base_domains = [
            "https://claude.ai", "https://github.com", "https://google.com",
            "https://drive.google.com", "https://mail.google.com",
            "https://linkedin.com", "https://notion.so", "https://figma.com",
        ]
        deep_urls = []
        for d in base_domains:
            u = uuid.uuid4().hex
            deep_urls.append(f"{d}/chat/{u}")
            deep_urls.append(f"{d}/document/d/{u}/edit?usp=sharing")
            deep_urls.append(f"{d}/u/1/#inbox/{u}{u[:8]}")
        rows = []
        for u in deep_urls:
            try:
                feats = extract_features(u)
                if len(feats) == 20:
                    rows.append((u, feats))
            except Exception:
                pass
        if rows:
            Xd = np.array([r[1] for r in rows], dtype=float)
            calibd, predd = predict(booster, cfg, Xd, FEATURES)
            fp_deep = predd.mean()
            print(f"    n = {len(rows)} deep-link URLs on trusted domains")
            print(f"    false positives: {int(predd.sum())}/{len(rows)} = {fp_deep*100:.1f}%")
            print("    -> This is the residual limitation the extension mitigates")
            print("       with a reputation allow-list (checks domain trust, not")
            print("       just the lexical features of the full deep-link URL).")
            # show worst offenders
            order = np.argsort(-calibd)
            print("    worst-scoring deep links:")
            for i in order[:5]:
                print(f"      {calibd[i]*100:5.1f}%  {rows[i][0][:70]}")
        else:
            print("    (feature extraction failed for all — check network)")
    except ImportError:
        print("    (url_features.py not importable here — run from project root)")

    # ------------------------------------------------------------------
    # [4] Baseline comparison — naive heuristic vs the model
    # ------------------------------------------------------------------
    print("\n[4] BASELINE COMPARISON (naive heuristic vs model, on held-out set)")
    # naive: flag as phishing if URL is 'long' OR domain is 'young'
    # length_url and time_domain_activation are in FEATURES
    li = FEATURES.index("length_url")
    ti = FEATURES.index("time_domain_activation")
    len_thr = np.median(X_te[:, li])            # crude: above-median length
    young_thr = np.percentile(X_te[:, ti][X_te[:, ti] >= 0], 25)  # young domains
    naive_pred = ((X_te[:, li] > len_thr) | ((X_te[:, ti] >= 0) & (X_te[:, ti] < young_thr))).astype(int)
    n_prec = precision_score(y_te, naive_pred, zero_division=0)
    n_rec = recall_score(y_te, naive_pred, zero_division=0)
    n_f1 = f1_score(y_te, naive_pred, zero_division=0)
    print(f"    naive heuristic : precision {n_prec:.3f}  recall {n_rec:.3f}  f1 {n_f1:.3f}")
    print(f"    XGBoost model   : precision {prec:.3f}  recall {rec:.3f}  f1 {f1:.3f}")
    print(f"    -> model f1 improvement over naive: {(f1 - n_f1)*100:+.1f} points")

    print("\n" + "=" * 72)
    print("Done. These numbers (with CIs, the deep-link characterization, and the")
    print("baseline) go into EVALUATION.md as the URL-model rigor section.")
    print("=" * 72)


if __name__ == "__main__":
    main()
