#!/usr/bin/env python3
"""
Measure the NARROW claim: is this a good lookalike-domain / typosquat detector?

Every measurement so far has been recall on ALL phishing, where platform-abuse
cases (*.pages.dev, *.blogspot.com) drag the number down -- and those are
structurally undetectable from the URL alone, since "3rf3x34x.pages.dev"
(phishing) and "johnsmith.github.io" (legitimate) are indistinguishable.

If the extension only claims to detect LOOKALIKE DOMAINS, then the number that
matters is recall on lookalikes -- which has never been measured.

Four evaluations:
  1. Synthetic typosquats/lookalikes generated from major brands
  2. Real lookalikes harvested from the live feeds (brand token in a domain
     that is not the brand's)
  3. FALSE POSITIVES on the genuine brand domains -- catching paypa1.com is
     worthless if it also flags paypal.com
  4. FALSE POSITIVES on ordinary legitimate browsing

Usage:
  python measure_lookalike_recall.py
"""
import json
import os
import random

import numpy as np
import requests
import xgboost as xgb

from url_features_v2 import extract_features_v2, FEATURE_ORDER
import homograph

random.seed(42)

MODEL = "models/url_xgb_model_v2.json"
CONFIG = "models/url_model_config_v2.json"

BRANDS = [
    ("paypal", "paypal.com"), ("apple", "apple.com"),
    ("microsoft", "microsoft.com"), ("netflix", "netflix.com"),
    ("amazon", "amazon.com"), ("google", "google.com"),
    ("facebook", "facebook.com"), ("instagram", "instagram.com"),
    ("linkedin", "linkedin.com"), ("roblox", "roblox.com"),
    ("steam", "steampowered.com"), ("spotify", "spotify.com"),
    ("ledger", "ledger.com"), ("coinbase", "coinbase.com"),
    ("binance", "binance.com"), ("dhl", "dhl.com"),
    ("fedex", "fedex.com"), ("dropbox", "dropbox.com"),
    ("chase", "chase.com"), ("hdfcbank", "hdfcbank.com"),
]

LEET = {"l": "1", "o": "0", "e": "3", "s": "5", "a": "4", "i": "1", "g": "9"}
PATHS = ["/login", "/signin", "/account/verify", "/secure/update",
         "/auth/confirm", "/", "/verify-account", "/billing/update"]


def make_typosquats(brand, real_domain):
    """Generate the standard typosquat families."""
    out = set()
    tld = real_domain.split(".", 1)[1]

    # character substitution (leet)
    for ch, sub in LEET.items():
        if ch in brand:
            out.add(f"{brand.replace(ch, sub, 1)}.{tld}")

    # doubled / dropped / swapped characters
    for i in range(1, len(brand) - 1):
        out.add(f"{brand[:i]}{brand[i]}{brand[i:]}.{tld}")      # doubled
        out.add(f"{brand[:i]}{brand[i+1:]}.{tld}")              # dropped
        if i < len(brand) - 1:
            sw = list(brand)
            sw[i], sw[i + 1] = sw[i + 1], sw[i]
            out.add(f"{''.join(sw)}.{tld}")                     # swapped

    # hyphenated and suffixed variants
    for suffix in ("secure", "login", "verify", "support", "account", "help"):
        out.add(f"{brand}-{suffix}.{tld}")
        out.add(f"{suffix}-{brand}.{tld}")
        out.add(f"{brand}{suffix}.{tld}")

    # alternate / suspicious TLDs
    for alt in ("net", "org", "co", "info", "online", "site", "top", "xyz",
                "cfd", "icu", "click", "live", "shop"):
        if alt != tld:
            out.add(f"{brand}.{alt}")

    # brand in subdomain of an unrelated domain
    for host in ("secure-portal.com", "account-services.net",
                 "verify-center.org", "login-gateway.info"):
        out.add(f"{brand}.{host}")
        out.add(f"{brand}-secure.{host}")

    # brand + country-ish suffix abuse
    for cc in ("com.am", "com.co", "com.de", "co.in"):
        out.add(f"{brand}.{cc}")

    return [f"https://{d}{random.choice(PATHS)}" for d in out]


def real_brand_urls():
    """Genuine brand URLs -- flagging these would make the tool useless."""
    urls = []
    for brand, dom in BRANDS:
        for p in ("/", "/login", "/signin", "/account", "/help",
                  "/support", "/security"):
            urls.append(f"https://www.{dom}{p}")
            urls.append(f"https://{dom}{p}")
    return urls


ORDINARY_LEGIT = [
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://stackoverflow.com/questions/tagged/python",
    "https://news.ycombinator.com/item?id=39485720",
    "https://arxiv.org/abs/1706.03762",
    "https://www.bbc.co.uk/news/technology-68123456",
    "https://docs.python.org/3/library/json.html",
    "https://pypi.org/project/requests/",
    "https://www.reddit.com/r/programming/",
    "https://github.com/pytorch/pytorch/issues/12345",
    "https://www.coursera.org/learn/machine-learning",
    "https://mail.google.com/mail/u/0/#inbox",
    "https://www.irctc.co.in/nget/train-search",
    "https://www.hdfcbank.com/personal/borrow/popular-loans",
    "https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M",
    "https://www.amazon.in/gp/bestsellers",
    "https://in.linkedin.com/in/someone-real-12345",
]


def load_model():
    if not os.path.exists(MODEL):
        print(f"Missing {MODEL} -- run train_url_model_v2.py first.")
        return None, None, None
    cfg = json.load(open(CONFIG, encoding="utf-8"))
    b = xgb.Booster()
    b.load_model(MODEL)
    x = np.array(cfg["calibration"]["x"], dtype=float)
    y = np.array(cfg["calibration"]["y"], dtype=float)
    return b, (x, y), float(cfg["threshold"])


def score_all(urls, booster, calib, threshold, use_homograph=True):
    X = np.array([extract_features_v2(u) for u in urls], dtype=float)
    raw = booster.predict(xgb.DMatrix(X, feature_names=FEATURE_ORDER))
    cal = np.interp(raw, calib[0], calib[1])
    flags = cal >= threshold
    if use_homograph:
        # production overlays homograph detection on top of the model
        for i, u in enumerate(urls):
            try:
                if homograph.analyze(u)["is_homograph"]:
                    flags[i] = True
                    cal[i] = max(cal[i], 0.99)
            except Exception:
                pass
    return cal, flags


def report(name, urls, cal, flags, expect_flag, show=6):
    n = len(urls)
    hit = int(flags.sum()) if expect_flag else int((~flags).sum())
    rate = hit / max(n, 1)
    label = "RECALL" if expect_flag else "correctly not flagged"
    print(f"\n--- {name} ---")
    print(f"  n={n}   {label}: {rate:.1%}")
    wrong = [(u, c) for u, c, f in zip(urls, cal, flags)
             if bool(f) != expect_flag]
    if wrong:
        kind = "misses" if expect_flag else "FALSE POSITIVES"
        print(f"  {len(wrong)} {kind}, e.g.:")
        for u, c in wrong[:show]:
            print(f"    p={c:.3f}  {u[:66]}")
    return rate


def main():
    booster, calib, threshold = load_model()
    if booster is None:
        return
    print(f"threshold {threshold}, {len(FEATURE_ORDER)} features")

    # 1. synthetic typosquats
    squats = []
    for brand, dom in BRANDS:
        squats += make_typosquats(brand, dom)
    random.shuffle(squats)
    squats = squats[:600]
    cal, fl = score_all(squats, booster, calib, threshold)
    r_squat = report("Synthetic typosquats / lookalikes (want FLAGGED)",
                     squats, cal, fl, True)

    # 2. real lookalikes from the live feed
    real_look = []
    try:
        txt = requests.get("https://openphish.com/feed.txt", timeout=30).text
        feed = [l.strip() for l in txt.splitlines()
                if l.strip().startswith(("http://", "https://"))]
        import tldextract
        brand_names = {b for b, _ in BRANDS}
        legit_doms = {d for _, d in BRANDS}
        for u in feed:
            ext = tldextract.extract(u)
            host = f"{ext.subdomain}.{ext.registered_domain}".lower()
            if (any(b in host for b in brand_names)
                    and ext.registered_domain.lower() not in legit_doms):
                real_look.append(u)
        real_look = real_look[:150]
    except Exception as e:
        print(f"\n(feed unavailable: {type(e).__name__})")

    if real_look:
        cal, fl = score_all(real_look, booster, calib, threshold)
        r_real = report("Real brand-impersonating URLs from live feed (want FLAGGED)",
                        real_look, cal, fl, True)
    else:
        r_real = None
        print("\n--- Real brand-impersonating URLs: none found in this feed sample ---")

    # 3. genuine brand domains -- must NOT flag
    genuine = real_brand_urls()
    cal, fl = score_all(genuine, booster, calib, threshold)
    keep_brand = report("GENUINE brand domains (want NOT flagged)",
                        genuine, cal, fl, False)

    # 4. ordinary legitimate browsing
    cal, fl = score_all(ORDINARY_LEGIT, booster, calib, threshold)
    keep_ord = report("Ordinary legitimate browsing (want NOT flagged)",
                      ORDINARY_LEGIT, cal, fl, False)

    print("\n" + "=" * 70)
    print("NARROW-CLAIM VERDICT -- lookalike / typosquat detector")
    print("=" * 70)
    print(f"  typosquat recall            {r_squat:.1%}")
    if r_real is not None:
        print(f"  real brand-impersonation    {r_real:.1%}")
    print(f"  genuine brands kept clean   {keep_brand:.1%}")
    print(f"  ordinary browsing kept clean{keep_ord:.1%}")
    print()
    fp_brand = 1 - keep_brand
    fp_ord = 1 - keep_ord
    if r_squat >= 0.80 and fp_brand <= 0.05 and fp_ord <= 0.10:
        print("  Defensible as a LOOKALIKE-DOMAIN CHECKER.")
        print("  The listing must state it does not detect phishing hosted on")
        print("  legitimate platforms (Cloudflare Pages, Blogspot, Vercel).")
    else:
        print("  Does not clear the bar even on the narrow claim.")
        if fp_brand > 0.05:
            print("  Flagging genuine brand domains is the disqualifying failure --")
            print("  users would be warned away from the real paypal.com.")


if __name__ == "__main__":
    main()
