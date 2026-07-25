#!/usr/bin/env python3
"""
Harvest legitimate DEEP URLs -- real paths and query strings on real domains.

WHY: the first v2 training run learned "URL has a path => phishing"
(qty_slash_url was the top feature by gain, and deep-link false positives hit
100%). The cause was the negative set: Majestic Million yields bare apex
domains like "https://example.com/", so every legitimate training example had
no path while the phishing feed was full of deep paths. The model took the
shortcut.

Fix: negatives that structurally resemble the positives in everything except
being phishing.

SOURCES (all free, no auth):
  1. Wayback CDX  -- real archived URLs for specific domains. Directly targets
                     the failing probe: deep links on trusted domains.
  2. Hacker News  -- real submitted URLs across thousands of legitimate sites.
  3. Wikipedia    -- deep article paths, some with query strings.

Output: legit_deep_urls.txt

Usage:
  python collect_legit_deep_urls.py            # ~4000 urls
  python collect_legit_deep_urls.py 8000
"""
import json
import random
import sys
import time

import requests

random.seed(42)
OUT = "legit_deep_urls.txt"
UA = {"User-Agent": "phishsentry-research/1.0"}

# Trusted domains whose deep links the model must NOT flag. These mirror the
# deep-link probe categories: docs, drives, commerce, Q&A, media, mail.
CDX_DOMAINS = [
    "docs.google.com", "drive.google.com", "calendar.google.com",
    "github.com", "gitlab.com", "bitbucket.org",
    "stackoverflow.com", "superuser.com", "serverfault.com",
    "amazon.com", "amazon.in", "flipkart.com", "ebay.com",
    "open.spotify.com", "youtube.com", "vimeo.com",
    "linkedin.com", "medium.com", "dev.to",
    "nytimes.com", "bbc.co.uk", "theguardian.com", "reuters.com",
    "arxiv.org", "sciencedirect.com", "springer.com", "ieee.org",
    "microsoft.com", "apple.com", "adobe.com", "atlassian.com",
    "dropbox.com", "notion.so", "figma.com", "slack.com",
    "reddit.com", "quora.com", "pinterest.com",
    "booking.com", "airbnb.com", "expedia.com", "irctc.co.in",
    "paypal.com", "stripe.com", "chase.com", "hdfcbank.com",
    "coursera.org", "udemy.com", "edx.org", "khanacademy.org",
    "npmjs.com", "pypi.org", "crates.io", "hub.docker.com",
]


def get_json(url, timeout=45, **kw):
    r = requests.get(url, headers=UA, timeout=timeout, **kw)
    r.raise_for_status()
    return r.json()


def from_wayback(per_domain=90):
    """Real archived deep URLs per domain. collapse=urlkey de-duplicates."""
    out = set()
    print("Wayback CDX (deep links on trusted domains)")
    for i, dom in enumerate(CDX_DOMAINS, 1):
        try:
            rows = get_json(
                "http://web.archive.org/cdx/search/cdx",
                params={
                    "url": dom,
                    "matchType": "domain",
                    "output": "json",
                    "limit": per_domain * 3,
                    "filter": "statuscode:200",
                    "collapse": "urlkey",
                    "fl": "original",
                },
                timeout=60,
            )
            urls = [r[0] for r in rows[1:]] if len(rows) > 1 else []
            # keep only ones that actually have a path -- that is the point
            deep = [u for u in urls
                    if u.startswith(("http://", "https://"))
                    and u.rstrip("/").count("/") > 2]
            random.shuffle(deep)
            out.update(deep[:per_domain])
            print(f"  [{i:>2}/{len(CDX_DOMAINS)}] {dom:<24} +{len(deep[:per_domain]):>3}  total {len(out)}")
        except Exception as e:
            print(f"  [{i:>2}/{len(CDX_DOMAINS)}] {dom:<24} failed ({type(e).__name__})")
        time.sleep(0.4)
    return out


def from_hackernews(n=900):
    """Real URLs submitted to HN -- diverse legitimate sites, real paths."""
    out = set()
    print("\nHacker News")
    ids = []
    for feed in ("topstories", "beststories", "newstories"):
        try:
            ids += get_json(f"https://hacker-news.firebaseio.com/v0/{feed}.json")
        except Exception as e:
            print(f"  {feed} failed ({type(e).__name__})")
    ids = list(dict.fromkeys(ids))
    random.shuffle(ids)
    for i, sid in enumerate(ids[:n], 1):
        try:
            item = get_json(f"https://hacker-news.firebaseio.com/v0/item/{sid}.json",
                            timeout=20)
            u = (item or {}).get("url", "")
            if u.startswith(("http://", "https://")):
                out.add(u)
        except Exception:
            pass
        if i % 150 == 0:
            print(f"  {i}/{min(n, len(ids))} items -> {len(out)} urls")
    print(f"  total {len(out)}")
    return out


def from_wikipedia(n=1200):
    """Deep article paths, plus some with query strings."""
    out = set()
    print("\nWikipedia")
    langs = ["en", "en", "en", "simple", "de", "fr"]
    while len(out) < n:
        lang = random.choice(langs)
        try:
            data = get_json(
                f"https://{lang}.wikipedia.org/w/api.php",
                params={"action": "query", "list": "random", "rnnamespace": 0,
                        "rnlimit": 100, "format": "json"},
            )
            for p in data["query"]["random"]:
                t = p["title"].replace(" ", "_")
                out.add(f"https://{lang}.wikipedia.org/wiki/{requests.utils.quote(t)}")
                if random.random() < 0.25:
                    out.add(f"https://{lang}.wikipedia.org/w/index.php?"
                            f"title={requests.utils.quote(t)}&action=history")
        except Exception as e:
            print(f"  failed ({type(e).__name__})")
            break
        print(f"  {len(out)} urls")
        time.sleep(0.3)
    return set(list(out)[:n])


def main():
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 4000

    urls = set()
    urls |= from_wayback()
    urls |= from_hackernews()
    urls |= from_wikipedia()

    urls = [u for u in urls if len(u) < 500]
    random.shuffle(urls)
    urls = urls[:target]

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(urls))

    deep = sum(1 for u in urls if u.rstrip("/").count("/") > 2)
    q = sum(1 for u in urls if "?" in u)
    print("\n" + "=" * 66)
    print(f"  wrote {len(urls)} legitimate URLs -> {OUT}")
    print(f"  with a real path:   {deep} ({deep / max(len(urls),1):.0%})")
    print(f"  with query string:  {q} ({q / max(len(urls),1):.0%})")
    print("=" * 66)
    print("These are the structural counterexamples the first v2 run lacked.")
    print("Re-run collect_url_dataset_v2.py, then retrain.")


if __name__ == "__main__":
    main()
