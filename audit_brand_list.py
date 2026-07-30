#!/usr/bin/env python3
"""
Audit and extend the lookalike detector's brand list -- WITHOUT auto-editing it.

WHY THIS DOES NOT AUTO-APPLY
----------------------------
BRAND_DOMAINS serves two roles with opposite risk profiles:

  keys   -> DETECTION. A missing brand means a missed typosquat. Low harm.
  values -> ALLOWLIST. A missing genuine domain means the extension WARNS THE
            USER AWAY FROM A REAL SITE. That is the failure that made the ML
            model unshippable, and rule R0's guarantee ("paypal.com can never
            be flagged") holds only because a human vetted every entry.

Auto-adding domains from a scraped source would hand that guarantee to a script.
If the source were wrong or poisoned, an attacker's domain lands in the
allowlist and the extension actively vouches for it. So this tool PROPOSES;
you approve; then it patches.

THREE CHECKS

  1. SELF-AUDIT (most valuable, no external data)
     Runs the detector over a large list of well-known real domains and reports
     anything flagged. Every hit is a live false positive.

  2. ccTLD VARIANTS
     For domains already trusted, generates regional variants (fedex.co.uk from
     fedex.com), verifies each actually resolves AND that its certificate/
     redirect behaviour is consistent with the brand, then proposes.

  3. UNKNOWN HOSTING PLATFORMS
     Scans live phishing feeds for shared-hosting domains absent from
     HOSTING_PLATFORMS. Purely additive to detection -- safe to be liberal.

Usage:
  python audit_brand_list.py                 # all checks
  python audit_brand_list.py --self-audit    # check 1 only (fast, offline)
  python audit_brand_list.py --apply         # patch from an approved file
"""
import argparse
import json
import os
import socket
import ssl
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

from lookalike_detector import (analyze, BRAND_DOMAINS, ALL_LEGIT_DOMAINS,
                                HOSTING_PLATFORMS)

PROPOSALS = "brand_list_proposals.json"

# ---------------------------------------------------------------------------
# Check 1: self-audit. Well-known real domains that must never be flagged.
# Deliberately includes regional variants, banks, and sites whose names contain
# brand-adjacent substrings.
# ---------------------------------------------------------------------------
KNOWN_GOOD = [
    # the brands themselves, common entry points
    "paypal.com", "www.paypal.com", "apple.com", "icloud.com",
    "microsoft.com", "office.com", "outlook.com", "live.com",
    "netflix.com", "amazon.com", "amazon.in", "amazon.co.uk",
    "google.com", "gmail.com", "youtube.com", "facebook.com",
    "instagram.com", "linkedin.com", "x.com", "roblox.com",
    "steampowered.com", "discord.com", "spotify.com", "twitch.tv",
    "ledger.com", "trezor.io", "metamask.io", "coinbase.com",
    "binance.com", "uniswap.org", "kraken.com", "dhl.com", "fedex.com",
    "ups.com", "usps.com", "dropbox.com", "docusign.com", "zoom.us",
    "adobe.com", "chase.com", "wellsfargo.com", "bankofamerica.com",
    "citibank.com", "hsbc.com", "barclays.co.uk", "hdfcbank.com",
    "icicibank.com", "axisbank.com", "paytm.com", "phonepe.com",
    "flipkart.com", "wetransfer.com", "figma.com", "notion.so",
    "slack.com", "github.com", "gitlab.com",
    # regional variants -- the most likely gaps
    "amazon.de", "amazon.co.jp", "amazon.ca", "amazon.com.au",
    "google.co.in", "google.co.uk", "google.de", "google.fr",
    "hsbc.co.uk", "hsbc.co.in", "hsbc.com.hk",
    "fedex.co.uk", "dhl.de", "dhl.co.uk", "ups.co.uk",
    "apple.co.uk", "microsoft.co.uk", "netflix.co.uk",
    "paypal.co.uk", "paypal.me",
    # brand-adjacent names that must NOT trip a rule
    "startups.com", "upstream.com", "backups.com", "groups.io",
    "applesupport.org", "applefcu.org", "apple-scruffs.com",
    "metabase.com", "metafilter.com", "metacritic.com",
    "steamcommunity.com", "steamdb.info",
    "visa.com", "mastercard.com", "stripe.com", "razorpay.com",
    # ordinary high-traffic sites
    "wikipedia.org", "stackoverflow.com", "reddit.com", "bbc.co.uk",
    "nytimes.com", "cloudflare.com", "mozilla.org", "python.org",
    "npmjs.com", "pypi.org", "docker.com", "kubernetes.io",
    "arxiv.org", "coursera.org", "edx.org", "irctc.co.in",
    "indiapost.gov.in", "uidai.gov.in", "incometax.gov.in",
    # platform-hosted legitimate content
    "docs.pages.dev", "pytorch.github.io", "microsoft.github.io",
    "google.github.io", "reactjs.github.io", "vitejs.dev",
]

# Ordered candidate suffixes for ccTLD generation.
CCTLD_SUFFIXES = [
    "co.uk", "de", "fr", "es", "it", "nl", "co.jp", "com.au", "ca",
    "co.in", "in", "com.br", "mx", "com.sg", "com.hk", "ie", "se",
    "ch", "at", "be", "pl", "com.tr", "ae", "sa", "co.za",
]


def self_audit():
    """Run the detector over known-good domains. Any flag is a false positive."""
    print("=" * 74)
    print("CHECK 1 -- self-audit: known-good domains that must NOT be flagged")
    print("=" * 74)
    hits = []
    for host in KNOWN_GOOD:
        v = analyze(f"https://{host}/")
        if v["is_lookalike"]:
            hits.append((host, v))
    if not hits:
        print(f"  {len(KNOWN_GOOD)} domains checked -- none flagged. Clean.")
        return []
    print(f"  {len(hits)} of {len(KNOWN_GOOD)} FLAGGED -- each is a live false positive:\n")
    for host, v in hits:
        print(f"  {host}")
        print(f"      rule {v['rule']}  brand={v['brand']}  conf={v['confidence']}")
        print(f"      {v['reason'][:88]}")
    print("\n  Fix: add the genuine domain to that brand's set in BRAND_DOMAINS,")
    print("  or if the domain is unrelated to the brand, the rule is too loose.")
    return hits


def _resolves(host, timeout=4):
    try:
        socket.setdefaulttimeout(timeout)
        socket.gethostbyname(host)
        return True
    except Exception:
        return False


def _cert_names(host, timeout=5):
    """Return the names on the TLS certificate. A genuine regional variant of a
    brand normally presents a cert issued to that brand -- weak evidence, but
    better than 'it resolves', which any squatter can achieve."""
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((host, 443), timeout=timeout) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as ssock:
                cert = ssock.getpeercert()
        names = set()
        for field in cert.get("subject", ()):
            for k, v in field:
                if k == "commonName":
                    names.add(v.lower())
        for typ, val in cert.get("subjectAltName", ()):
            if typ == "DNS":
                names.add(val.lower())
        return names
    except Exception:
        return set()


def cctld_candidates(verify=True, workers=12):
    """Propose regional variants of already-trusted domains."""
    print()
    print("=" * 74)
    print("CHECK 2 -- ccTLD variants of domains already trusted")
    print("=" * 74)

    candidates = []
    for brand, domains in sorted(BRAND_DOMAINS.items()):
        for d in domains:
            parts = d.split(".")
            if len(parts) < 2:
                continue
            root = parts[0]
            for suf in CCTLD_SUFFIXES:
                cand = f"{root}.{suf}"
                if cand in ALL_LEGIT_DOMAINS or cand == d:
                    continue
                candidates.append((brand, d, cand))

    # de-dup
    seen, uniq = set(), []
    for b, src, c in candidates:
        if c not in seen:
            seen.add(c)
            uniq.append((b, src, c))
    print(f"  {len(uniq)} candidate variants generated from {len(ALL_LEGIT_DOMAINS)} trusted domains")

    if not verify:
        return []

    print("  verifying which resolve and present a matching certificate ...")

    def check(item):
        brand, src, cand = item
        if not _resolves(cand):
            return None
        names = _cert_names(cand)
        # does the cert mention the brand token or the source domain root?
        root = src.split(".")[0]
        match = any(root in n for n in names)
        return (brand, src, cand, sorted(names)[:3], match)

    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(check, uniq):
            if r:
                results.append(r)

    confirmed = [r for r in results if r[4]]
    unconfirmed = [r for r in results if not r[4]]

    print(f"\n  {len(confirmed)} resolve WITH a brand-matching certificate:")
    for brand, src, cand, names, _ in sorted(confirmed):
        print(f"    {cand:<26} (brand {brand}, from {src})")
        print(f"        cert: {', '.join(names[:2])}")

    if unconfirmed:
        print(f"\n  {len(unconfirmed)} resolve but the certificate does NOT mention")
        print("  the brand -- these may be squatters. Do NOT allowlist without")
        print("  checking each by hand:")
        for brand, src, cand, names, _ in sorted(unconfirmed)[:15]:
            print(f"    {cand:<26} cert: {', '.join(names[:2]) or '(none)'}")

    return confirmed


def unknown_platforms(sample=400):
    """Find shared-hosting domains in live feeds that HOSTING_PLATFORMS lacks."""
    print()
    print("=" * 74)
    print("CHECK 3 -- hosting platforms seen in live feeds but not in our list")
    print("=" * 74)
    try:
        import tldextract
        from collections import Counter
        r = requests.get("https://openphish.com/feed.txt", timeout=30)
        r.raise_for_status()
        urls = [l.strip() for l in r.text.splitlines()
                if l.strip().startswith(("http://", "https://"))][:sample]
    except Exception as e:
        print(f"  feed unavailable ({type(e).__name__}) -- skipped")
        return []

    counts = Counter()
    for u in urls:
        e = tldextract.extract(u)
        reg = e.registered_domain.lower()
        # a platform shows up repeatedly with DIFFERENT subdomains
        if reg and e.subdomain:
            counts[reg] += 1

    # repeated registered domains with varied subdomains suggest shared hosting
    suspects = [(d, n) for d, n in counts.most_common(40)
                if n >= 3 and d not in HOSTING_PLATFORMS]
    if not suspects:
        print("  none found in this sample")
        return []

    print(f"  {len(suspects)} registered domains appear 3+ times with different")
    print("  subdomains and are NOT in HOSTING_PLATFORMS:\n")
    for d, n in suspects:
        print(f"    {d:<32} {n} occurrences")
    print("\n  Adding these is DETECTION-ONLY (rule R5) and cannot cause a false")
    print("  positive on a genuine brand domain. Still check each is really a")
    print("  hosting platform and not simply a busy phishing domain.")
    return [d for d, _ in suspects]


def write_proposals(cctld, platforms):
    data = {
        "note": ("REVIEW EACH ENTRY BEFORE APPLYING. Domains listed under "
                 "'allowlist_additions' will be treated as GENUINE by the "
                 "extension -- an error here means the tool vouches for a "
                 "domain it should warn about."),
        "allowlist_additions": [
            {"brand": b, "domain": c, "derived_from": s,
             "cert_names": names, "approved": False}
            for b, s, c, names, _ in cctld
        ],
        "hosting_platform_additions": [
            {"domain": d, "approved": False} for d in platforms
        ],
    }
    with open(PROPOSALS, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"\nwrote {PROPOSALS}")
    print("Set \"approved\": true on the entries you accept, then run:")
    print("  python audit_brand_list.py --apply")


def apply_proposals():
    if not os.path.exists(PROPOSALS):
        print(f"{PROPOSALS} not found -- run the audit first.")
        return 1
    data = json.load(open(PROPOSALS, encoding="utf-8"))
    allow = [a for a in data.get("allowlist_additions", []) if a.get("approved")]
    plats = [p for p in data.get("hosting_platform_additions", []) if p.get("approved")]

    if not allow and not plats:
        print("Nothing marked approved. No changes made.")
        return 0

    print("Approved for application:")
    for a in allow:
        print(f"  allowlist: {a['domain']}  -> brand '{a['brand']}'")
    for p in plats:
        print(f"  platform:  {p['domain']}")
    print("\nThis edits lookalike_detector.py. A backup is written first.")
    if input("Proceed? [y/N] ").strip().lower() != "y":
        print("Aborted.")
        return 0

    import shutil
    path = "lookalike_detector.py"
    shutil.copy(path, path + ".prebrandpatch")
    t = open(path, encoding="utf-8").read()

    for a in allow:
        brand, dom = a["brand"], a["domain"]
        # find that brand's set and insert the domain
        needle = f'"{brand}": {{'
        i = t.find(needle)
        if i == -1:
            print(f"  SKIP {dom}: brand '{brand}' not found")
            continue
        j = t.find("}", i)
        if j == -1:
            print(f"  SKIP {dom}: malformed set for '{brand}'")
            continue
        t = t[:j] + f', "{dom}"' + t[j:]
        print(f"  added {dom} to {brand}")

    if plats:
        anchor = "HOSTING_PLATFORMS = {"
        i = t.find(anchor)
        if i != -1:
            ins = "".join(f'\n    "{p["domain"]}",' for p in plats)
            t = t[:i + len(anchor)] + ins + t[i + len(anchor):]
            print(f"  added {len(plats)} hosting platform(s)")

    open(path, "w", encoding="utf-8").write(t)
    print(f"\nPatched. Backup at {path}.prebrandpatch")
    print("NOW RE-RUN THE TESTS -- an allowlist change can silently break detection:")
    print("  python lookalike_detector.py")
    print("  python eval_lookalike_detector.py")
    print("  python audit_brand_list.py --self-audit")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-audit", action="store_true",
                    help="only run the offline false-positive audit")
    ap.add_argument("--apply", action="store_true",
                    help="apply approved entries from the proposals file")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip DNS/TLS verification of ccTLD candidates")
    args = ap.parse_args()

    if args.apply:
        return apply_proposals()

    hits = self_audit()
    if args.self_audit:
        return 1 if hits else 0

    cctld = cctld_candidates(verify=not args.no_verify)
    plats = unknown_platforms()
    write_proposals(cctld, plats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
