#!/usr/bin/env python3
"""
Build the v2 URL training dataset.

POSITIVES (phishing)
  - Phishing.Database (mitchellkrogza) active phishing links -- large, free,
    updated frequently.
  - OpenPhish community feed -- smaller but very fresh.
  - github.io entries are EXCLUDED. The live-recall measurement surfaced feed
    entries like "satya-1205.github.io/NetflixWebsite" and
    "ralfsales.github.io/Bank-Simulation" which are indistinguishable from
    ordinary student clone projects. Training on them would teach the model
    that brand-clone portfolios are phishing, producing false positives on
    exactly the kind of project a student would host. Documented gap: real
    GitHub Pages phishing is therefore not represented in v1.

NEGATIVES (legitimate)
  - Majestic Million top domains -- broad legitimate baseline.
  - GitHub Pages sites harvested via the GitHub API, DELIBERATELY INCLUDING
    clone/learning projects (netflix-clone, spotify-clone, bank-simulation).
    Without these the model cannot learn the distinction discussed above.
  - Any local curated legit URLs you already have.

LEAKAGE CONTROL
  Train/test are split by REGISTERED DOMAIN, not by URL. Many feed entries
  share a domain; a random URL split would put the same domain on both sides
  and inflate test scores.

Usage:
  python collect_url_dataset_v2.py                 # default sizes
  python collect_url_dataset_v2.py 8000 8000       # n_pos n_neg
  GITHUB_TOKEN=ghp_xxx python collect_url_dataset_v2.py   # higher API limits
"""
import csv
import os
import random
import sys
import time

import requests
import tldextract

random.seed(42)

PHISHDB = ("https://raw.githubusercontent.com/mitchellkrogza/"
           "Phishing.Database/master/phishing-links-ACTIVE.txt")
OPENPHISH = "https://openphish.com/feed.txt"
MAJESTIC = "https://downloads.majestic.com/majestic_million.csv"

EXCLUDE_HOST_SUFFIXES = ("github.io",)   # see module docstring

OUT_TRAIN = "url_dataset_v2_train.csv"
OUT_TEST = "url_dataset_v2_test.csv"

# Clone/learning-project search terms -- these produce the legitimate
# counterexamples to the ambiguous feed entries.
GH_CLONE_QUERIES = [
    "netflix-clone", "spotify-clone", "amazon-clone", "instagram-clone",
    "whatsapp-clone", "youtube-clone", "airbnb-clone", "twitter-clone",
    "bank-simulation", "banking-system", "ecommerce-website",
    "login-page", "portfolio-website", "landing-page", "todo-app",
    "weather-app", "chat-app", "blog-template", "dashboard-ui",
    "food-delivery", "movie-app", "quiz-app",
]


def _get(url, **kw):
    kw.setdefault("timeout", 60)
    kw.setdefault("headers", {"User-Agent": "phishsentry-research/1.0"})
    r = requests.get(url, **kw)
    r.raise_for_status()
    return r


# --- defanging -------------------------------------------------------------
# Stored defanged so AV does not quarantine the dataset files. Standard
# threat-intel convention; refanged in memory at load time.
def defang(u):
    return (u.replace("https://", "hxxps://")
             .replace("http://", "hxxp://")
             .replace(".", "[.]"))


def refang(u):
    return (u.replace("[.]", ".")
             .replace("hxxps://", "https://")
             .replace("hxxp://", "http://"))


def reg_domain(u):
    try:
        return tldextract.extract(u).registered_domain.lower()
    except Exception:
        return ""


def fetch_positives(n):
    urls = set()

    for name, src in (("Phishing.Database", PHISHDB), ("OpenPhish", OPENPHISH)):
        try:
            print(f"  fetching {name} ...", end=" ", flush=True)
            text = _get(src).text
            got = [ln.strip() for ln in text.splitlines()
                   if ln.strip().lower().startswith(("http://", "https://"))]
            urls.update(got)
            print(f"{len(got)} urls")
        except Exception as e:
            print(f"FAILED ({type(e).__name__})")

    before = len(urls)
    urls = {u for u in urls
            if not any(s in u.lower() for s in EXCLUDE_HOST_SUFFIXES)}
    print(f"  excluded {before - len(urls)} github.io entries (see docstring)")

    urls = list(urls)
    random.shuffle(urls)
    return urls[:n]


def fetch_github_pages(target):
    """Harvest legitimate GitHub Pages URLs, including clone/learning projects."""
    token = os.environ.get("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github+json",
               "User-Agent": "phishsentry-research/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        print("  using GITHUB_TOKEN (5000 req/hr)")
    else:
        print("  no GITHUB_TOKEN set (60 req/hr -- set one for more data)")

    out = set()
    for q in GH_CLONE_QUERIES:
        if len(out) >= target:
            break
        try:
            r = requests.get(
                "https://api.github.com/search/repositories",
                params={"q": f"{q} in:name", "per_page": 100, "sort": "stars"},
                headers=headers, timeout=45,
            )
            if r.status_code == 403:
                print(f"  rate limited at query '{q}' -- stopping harvest")
                break
            r.raise_for_status()
            items = r.json().get("items", [])
            for it in items:
                owner = (it.get("owner") or {}).get("login")
                repo = it.get("name")
                if not owner or not repo:
                    continue
                if repo.lower() == f"{owner.lower()}.github.io":
                    out.add(f"https://{owner.lower()}.github.io/")
                else:
                    out.add(f"https://{owner.lower()}.github.io/{repo}/")
                hp = (it.get("homepage") or "").strip()
                if hp.startswith("https://") and "github.io" not in hp:
                    out.add(hp)   # often a pages.dev / vercel.app / netlify.app
            print(f"  '{q}': total {len(out)}")
            time.sleep(2 if not token else 0.5)
        except Exception as e:
            print(f"  '{q}' failed ({type(e).__name__})")
    return list(out)


def fetch_negatives(n):
    urls = set()

    gh = fetch_github_pages(target=max(600, n // 8))
    urls.update(gh)
    print(f"  GitHub Pages / homepages: {len(gh)}")

    # Deep legitimate URLs FIRST -- these are the structural counterexamples.
    # Without them the model learns "URL has a path => phishing" (measured:
    # 100% deep-link false positives on the first v2 run).
    deep = []
    if os.path.exists("legit_deep_urls.txt"):
        with open("legit_deep_urls.txt", encoding="utf-8", errors="ignore") as f:
            deep = [ln.strip() for ln in f
                    if ln.strip().startswith(("http://", "https://"))]
        urls.update(deep)
        print(f"  deep legitimate URLs: {len(deep)}")
    else:
        print("  WARNING: legit_deep_urls.txt missing -- run "
              "collect_legit_deep_urls.py first, or the model will learn "
              "'has a path => phishing'.")

    # Majestic apex domains are capped so they cannot dominate. They give
    # domain-level diversity but every one of them is path-free, which is
    # exactly the confound we are correcting.
    apex_cap = max(500, n // 4)
    try:
        print("  fetching Majestic Million ...", end=" ", flush=True)
        text = _get(MAJESTIC).text
        rows = list(csv.reader(text.splitlines()))
        header = rows[0]
        di = header.index("Domain") if "Domain" in header else 2
        doms = [r[di].strip().lower() for r in rows[1:] if len(r) > di]
        random.shuffle(doms)
        for d in doms[:apex_cap]:
            urls.add(f"https://{d}/")
        print(f"{len(doms)} available, capped at {apex_cap}")
    except Exception as e:
        print(f"FAILED ({type(e).__name__})")

    for local in ("modern_legit_urls.txt", "legit_urls.txt"):
        if os.path.exists(local):
            with open(local, encoding="utf-8", errors="ignore") as f:
                extra = [ln.strip() for ln in f
                         if ln.strip().startswith(("http://", "https://"))]
            urls.update(extra)
            print(f"  local {local}: {len(extra)}")

    urls = list(urls)
    random.shuffle(urls)
    return urls[:n]


def domain_disjoint_split(rows, test_frac=0.25):
    """Split so no registered domain appears in both train and test."""
    by_dom = {}
    for url, label in rows:
        by_dom.setdefault(reg_domain(url) or url, []).append((url, label))
    doms = list(by_dom)
    random.shuffle(doms)
    cut = int(len(doms) * (1 - test_frac))
    train = [r for d in doms[:cut] for r in by_dom[d]]
    test = [r for d in doms[cut:] for r in by_dom[d]]
    random.shuffle(train)
    random.shuffle(test)
    return train, test


def main():
    n_pos = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    n_neg = int(sys.argv[2]) if len(sys.argv) > 2 else 8000

    print("POSITIVES (phishing)")
    pos = fetch_positives(n_pos)
    print(f"  -> {len(pos)} positives\n")

    print("NEGATIVES (legitimate)")
    neg = fetch_negatives(n_neg)
    print(f"  -> {len(neg)} negatives\n")

    if not pos or not neg:
        print("Aborting: one class is empty.")
        return

    rows = [(u, 1) for u in pos] + [(u, 0) for u in neg]
    train, test = domain_disjoint_split(rows)

    for path, data in ((OUT_TRAIN, train), (OUT_TEST, test)):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["url", "label"])
            w.writerows([(defang(u), l) for u, l in data])

    def bal(d):
        p = sum(1 for _, l in d if l == 1)
        return f"{len(d)} rows ({p} phishing / {len(d) - p} legit)"

    print("=" * 66)
    print(f"  {OUT_TRAIN}: {bal(train)}")
    print(f"  {OUT_TEST}:  {bal(test)}")
    print("=" * 66)
    print("Split is domain-disjoint: no registered domain appears in both.")
    print("\nCaveats to carry into EVALUATION.md:")
    print(" - github.io excluded from positives, so GitHub Pages phishing is")
    print("   not represented in v1.")
    print(" - Negatives are thin on pages.dev / vercel.app / netlify.app; the")
    print("   platform_and_random_subdomain interaction feature is what keeps")
    print("   the model from simply learning 'platform => phishing', but this")
    print("   must be verified on the test set, not assumed.")
    print(" - Feed labels are not hand-verified; some label noise is expected.")


if __name__ == "__main__":
    main()
