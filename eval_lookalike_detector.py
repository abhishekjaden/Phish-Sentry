#!/usr/bin/env python3
"""
Evaluate the deterministic detector on the SAME four probe sets the model was
measured on, so the comparison is apples-to-apples.

Model results for reference (threshold 0.35, 37 features):
  typosquat recall            90.3%
  real brand-impersonation    77.1%
  genuine brands kept clean   25.7%   <- disqualifying
  ordinary browsing clean     68.8%

Usage:
  python eval_lookalike_detector.py
"""
import random

import requests
import tldextract

from lookalike_detector import analyze
from measure_lookalike_recall import (BRANDS, make_typosquats,
                                      real_brand_urls, ORDINARY_LEGIT)

random.seed(42)


def run(name, urls, expect_flag, show=8):
    n = len(urls)
    results = [(u, analyze(u)) for u in urls]
    right = sum(1 for _, v in results if v["is_lookalike"] == expect_flag)
    rate = right / max(n, 1)
    label = "RECALL" if expect_flag else "correctly not flagged"
    print(f"\n--- {name} ---")
    print(f"  n={n}   {label}: {rate:.1%}")
    wrong = [(u, v) for u, v in results if v["is_lookalike"] != expect_flag]
    if wrong:
        kind = "misses" if expect_flag else "FALSE POSITIVES"
        print(f"  {len(wrong)} {kind}, e.g.:")
        for u, v in wrong[:show]:
            print(f"    {u[:66]}")
            if v["is_lookalike"]:
                print(f"      -> {v['rule']}: {v['reason'][:80]}")
    return rate


def main():
    squats = []
    for brand, dom in BRANDS:
        squats += make_typosquats(brand, dom)
    random.shuffle(squats)
    squats = squats[:600]
    r_squat = run("Synthetic typosquats / lookalikes (want FLAGGED)",
                  squats, True)

    real_look = []
    try:
        txt = requests.get("https://openphish.com/feed.txt", timeout=30).text
        feed = [l.strip() for l in txt.splitlines()
                if l.strip().startswith(("http://", "https://"))]
        brand_names = {b for b, _ in BRANDS}
        legit = {d for _, d in BRANDS}
        for u in feed:
            e = tldextract.extract(u)
            host = f"{e.subdomain}.{e.registered_domain}".lower()
            if (any(b in host for b in brand_names)
                    and e.registered_domain.lower() not in legit):
                real_look.append(u)
        real_look = real_look[:150]
    except Exception as e:
        print(f"\n(feed unavailable: {type(e).__name__})")

    r_real = None
    if real_look:
        r_real = run("Real brand-impersonating URLs from live feed (want FLAGGED)",
                     real_look, True)

    keep_brand = run("GENUINE brand domains (want NOT flagged)",
                     real_brand_urls(), False)
    keep_ord = run("Ordinary legitimate browsing (want NOT flagged)",
                   ORDINARY_LEGIT, False)

    print("\n" + "=" * 70)
    print("DETERMINISTIC DETECTOR vs MODEL")
    print("=" * 70)
    print(f"{'metric':<32}{'rules':>10}{'model':>10}")
    print(f"{'typosquat recall':<32}{r_squat:>9.1%}{0.903:>10.1%}")
    if r_real is not None:
        print(f"{'real brand-impersonation':<32}{r_real:>9.1%}{0.771:>10.1%}")
    print(f"{'genuine brands kept clean':<32}{keep_brand:>9.1%}{0.257:>10.1%}")
    print(f"{'ordinary browsing kept clean':<32}{keep_ord:>9.1%}{0.688:>10.1%}")

    print()
    fp_brand = 1 - keep_brand
    fp_ord = 1 - keep_ord
    if r_squat >= 0.85 and fp_brand == 0.0 and fp_ord <= 0.05:
        print("  PUBLISHABLE as a lookalike-domain checker.")
        print("  Zero false positives on genuine brand domains is structural:")
        print("  rule R0 allowlists them before any heuristic runs.")
        print("  The store listing MUST state that it does not detect phishing")
        print("  hosted on legitimate platforms under random subdomains.")
    else:
        if fp_brand > 0.0:
            print(f"  {fp_brand:.1%} FP on genuine brands -- a domain is missing")
            print("  from BRAND_DOMAINS. Add it; this must be exactly zero.")
        if r_squat < 0.85:
            print(f"  Typosquat recall {r_squat:.1%} is below the 85% bar.")


if __name__ == "__main__":
    main()
