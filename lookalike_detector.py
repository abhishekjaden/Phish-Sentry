"""
lookalike_detector.py -- deterministic lookalike / typosquat detection.

WHY RULES INSTEAD OF A MODEL
---------------------------
Four supervised iterations were measured on this task. Each fixed one
confounder and exposed another:

  v1 (20 network features)  live recall 38.3%  -- blind to platform abuse
  v2 first run              deep-link FP 100%  -- learned "has a path = phishing"
  v2 + deep negatives       live recall 63.7%  -- platform abuse still missed
  v2 + dictionary features  genuine-brand FP 74.3% -- learned "brand + /login
                                                     = phishing", so it flagged
                                                     the real paypal.com/login

The last one is disqualifying and structural: a classifier assigns a
probability, so "never flag the real paypal.com" can only ever be approximately
true. But the actual question -- "is this domain a near-miss of a known brand
domain, without BEING that brand domain?" -- is a deterministic string question
with an exact answer.

Rules give what the model could not:
  * genuine brand domains are allowlisted by construction -> FP rate on them
    is structurally ZERO, not statistically small
  * every verdict carries a specific human-readable reason
  * no training data, no drift, trivially reviewable by Chrome Web Store

Scope, stated honestly: this detects LOOKALIKE DOMAINS. It does not detect
phishing hosted on a legitimate platform under a random subdomain
(3rf3x34x.pages.dev), because that is not a lookalike and the URL carries no
evidence of it.
"""

import re
import urllib.parse

import tldextract

# Pin tldextract to its bundled snapshot. Left at defaults it fetches the live
# Public Suffix List on first use, which (a) makes cold starts slow and
# network-dependent in Lambda and (b) silently changes how hosts are split.
# Platform detection above is parse-independent, but pinning keeps every other
# rule deterministic across environments.
_extract = tldextract.TLDExtract(suffix_list_urls=(), fallback_to_snapshot=True)

# ---------------------------------------------------------------------------
# Brand -> the registered domains that genuinely belong to that brand.
# This mapping is the whole safety property: an exact match here short-circuits
# to SAFE before any heuristic runs.
# ---------------------------------------------------------------------------
BRAND_DOMAINS = {
    "paypal": {"paypal.com", "paypal.me", "paypalobjects.com"},
    "apple": {"apple.com", "icloud.com", "apple.co", "applecard.apple"},
    "icloud": {"apple.com", "icloud.com"},
    "microsoft": {"microsoft.com", "live.com", "msn.com", "office.com",
                  "office365.com", "outlook.com", "microsoftonline.com",
                  "sharepoint.com", "azure.com", "windows.com"},
    "outlook": {"microsoft.com", "live.com", "outlook.com"},
    "office": {"microsoft.com", "office.com", "office365.com"},
    "netflix": {"netflix.com", "nflxvideo.net", "nflximg.net"},
    "amazon": {"amazon.com", "amazon.in", "amazon.co.uk", "amazon.de",
               "amazon.co.jp", "amazon.ca", "amazon.com.au", "amazonaws.com",
               "primevideo.com", "audible.com"},
    "google": {"google.com", "google.co.in", "google.co.uk", "youtube.com",
               "gmail.com", "withgoogle.com", "googleusercontent.com",
               "goo.gl", "android.com", "chrome.com"},
    "gmail": {"google.com", "gmail.com"},
    "youtube": {"youtube.com", "youtu.be", "google.com"},
    "facebook": {"facebook.com", "fb.com", "fbcdn.net", "meta.com",
                 "messenger.com"},
    "instagram": {"instagram.com", "cdninstagram.com", "facebook.com"},
    "whatsapp": {"whatsapp.com", "wa.me", "facebook.com"},
    "linkedin": {"linkedin.com", "licdn.com", "lnkd.in"},
    "twitter": {"twitter.com", "x.com", "t.co", "twimg.com"},
    "roblox": {"roblox.com", "rbxcdn.com"},
    "steam": {"steampowered.com", "steamcommunity.com", "valvesoftware.com"},
    "discord": {"discord.com", "discordapp.com", "discord.gg"},
    "spotify": {"spotify.com", "scdn.co", "spotifycdn.com"},
    "twitch": {"twitch.tv", "ttvnw.net"},
    "ledger": {"ledger.com", "ledgerwallet.com"},
    "trezor": {"trezor.io"},
    "metamask": {"metamask.io"},
    "coinbase": {"coinbase.com", "cbhq.net"},
    "binance": {"binance.com", "binance.us"},
    "uniswap": {"uniswap.org"},
    "kraken": {"kraken.com"},
    "dhl": {"dhl.com", "dhl.de", "dpdhl.com"},
    "fedex": {"fedex.com"},
    "ups": {"ups.com"},
    "usps": {"usps.com", "usps.gov"},
    "dropbox": {"dropbox.com", "dropboxusercontent.com"},
    "docusign": {"docusign.com", "docusign.net"},
    "zoom": {"zoom.us", "zoom.com"},
    "adobe": {"adobe.com", "adobelogin.com"},
    "chase": {"chase.com", "jpmorganchase.com"},
    "wellsfargo": {"wellsfargo.com"},
    "bankofamerica": {"bankofamerica.com", "bofa.com"},
    "citibank": {"citibank.com", "citi.com"},
    "hsbc": {"hsbc.com", "hsbc.co.uk", "hsbc.co.in"},
    "barclays": {"barclays.co.uk", "barclays.com"},
    "hdfcbank": {"hdfcbank.com"},
    "icici": {"icicibank.com"},
    "sbi": {"onlinesbi.sbi", "sbi.co.in", "statebank"},
    "axis": {"axisbank.com"},
    "paytm": {"paytm.com", "paytm.in"},
    "phonepe": {"phonepe.com"},
    "flipkart": {"flipkart.com"},
    "wetransfer": {"wetransfer.com"},
    "figma": {"figma.com"},
    "notion": {"notion.so", "notion.com"},
    "slack": {"slack.com"},
    # NOTE: github.io / gitlab.io are deliberately NOT listed here. They are
    # shared user-content hosting (see HOSTING_PLATFORMS). Allowlisting them
    # would short-circuit every user page to safe, including
    # instagram-cmd.github.io.
    "github": {"github.com", "githubusercontent.com"},
    "gitlab": {"gitlab.com"},
}

ALL_LEGIT_DOMAINS = set()
for _s in BRAND_DOMAINS.values():
    ALL_LEGIT_DOMAINS |= _s

# Shared hosting platforms -- a brand name in a subdomain here is suspicious,
# but is also how legitimate clone/demo projects are named, so it gets a
# lower-confidence verdict with explicit wording.
HOSTING_PLATFORMS = {
    "pages.dev", "blogspot.com", "vercel.app", "netlify.app", "web.app",
    "firebaseapp.com", "github.io", "gitlab.io", "herokuapp.com",
    "godaddysites.com", "weeblysite.com", "weebly.com", "wixsite.com",
    "typedream.app", "edgeone.dev", "replit.app", "repl.co", "wasmer.app",
    "lovable.app", "surge.sh", "glitch.me", "onrender.com", "r2.dev",
    "workers.dev", "duckdns.org", "000webhostapp.com", "webflow.io",
    "gitbook.io", "notion.site", "carrd.co", "framer.website",
    "ngrok.io", "ngrok-free.app", "trycloudflare.com", "azurewebsites.net",
}

SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "cfd", "top", "xyz", "icu", "buzz",
    "monster", "click", "link", "rest", "surf", "bar", "beauty", "quest",
    "sbs", "cyou", "lol", "work", "date", "loan", "men", "stream",
    "download", "zip", "mov", "info", "online", "site", "live", "shop",
}

# Homoglyph / confusable folding: Cyrillic, Greek, fullwidth, plus leet.
_CONFUSABLES = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "і": "i", "ѕ": "s", "ԁ": "d", "ɡ": "g", "ν": "v", "ο": "o", "ρ": "p",
    "τ": "t", "α": "a", "ε": "e", "ι": "i", "κ": "k", "μ": "m", "ｅ": "e",
    "1": "l", "0": "o", "3": "e", "5": "s", "4": "a", "7": "t", "9": "g",
    "$": "s", "@": "a", "|": "l", "!": "i",
}


def _fold(s):
    """Fold homoglyphs and leet to canonical latin, drop separators."""
    out = []
    for ch in s.lower():
        out.append(_CONFUSABLES.get(ch, ch))
    return "".join(out).replace("-", "").replace("_", "")


# Code-hosting platforms where organisations publish official pages under a
# subdomain matching their own name (microsoft.github.io, netflix.github.io).
CODE_HOSTS = {"github.io", "gitlab.io"}

# Brands short enough that substring matching would produce false positives
# ('ups' inside 'groups', 'startups', 'backups'). Matched on token boundaries.
SHORT_BRAND_LEN = 4


def _fold_keep_separators(s):
    """Fold confusables to latin but KEEP hyphens/dots/underscores, so that
    token boundaries survive. _fold() strips them, which made short-brand
    token matching impossible ('verify-dhl' -> 'verifydhl')."""
    return "".join(_CONFUSABLES.get(ch, ch) for ch in s.lower())


def _tokens(text):
    """Split on any non-alphanumeric boundary, confusables folded."""
    import re as _re
    return {tok for tok in _re.split(r"[^a-z0-9]+",
                                     _fold_keep_separators(text)) if tok}


# Words that corroborate a short-brand substring hit. Presence of one of these
# alongside a brand prefix is what separates 'dhlaccount.com' from 'upstream.com'.
CORROBORATING_WORDS = {
    "account", "accounts", "login", "logon", "signin", "verify", "verification",
    "secure", "security", "update", "confirm", "auth", "authenticate",
    "password", "recover", "recovery", "unlock", "validate", "support",
    "help", "service", "services", "customer", "client", "portal",
    "track", "tracking", "delivery", "parcel", "package", "shipment",
    "billing", "payment", "invoice", "refund", "claim", "reward",
    "alert", "notice", "notification", "suspend", "suspended", "restore",
}


def _short_brand_hit(brand, text):
    """Short brands (< 4 chars) need TWO signals to avoid firing inside
    ordinary words like 'startups' or 'upstream':
      1. the brand at the START of a token, and
      2. a corroborating credential-harvest word nearby.
    """
    toks = _tokens(text)
    if not any(tok.startswith(brand) and tok != brand for tok in toks):
        # exact token match is handled separately and needs no corroboration
        return False
    return bool(toks & CORROBORATING_WORDS) or any(
        any(w in tok for w in CORROBORATING_WORDS) for tok in toks
    )


def _brand_matches(brand, text, extra_context=""):
    """Substring match for normal brands. Short brands match on an exact token,
    or on a token prefix when corroborated (see _short_brand_hit)."""
    folded = _fold(text)
    if len(brand) >= SHORT_BRAND_LEN:
        return brand in folded
    toks = _tokens(text)
    if brand in toks:
        return True
    # fused case: dhlaccount.com, dhlsupport.com
    return _short_brand_hit(brand, text + " " + extra_context)


def _levenshtein(a, b, cap=3):
    if a == b:
        return 0
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
        if min(prev) > cap:
            return cap + 1
    return prev[-1]


def _has_punycode(host):
    return any(lbl.startswith("xn--") for lbl in host.split("."))


def _non_ascii(host):
    return any(ord(c) > 127 for c in host)


def _resolve_platform(host):
    """Return (platform, effective_subdomain) by matching the raw host against
    HOSTING_PLATFORMS, independent of tldextract's public-suffix behaviour.

    'microsoft.github.io' -> ('github.io', 'microsoft')
    'a.b.pages.dev'       -> ('pages.dev', 'a.b')
    'example.com'         -> (None, None)

    Longest platform match wins, so 'ngrok-free.app' beats 'app' style
    prefixes if the list ever grows overlapping entries.
    """
    host = (host or "").lower().lstrip(".")
    best = None
    for plat in HOSTING_PLATFORMS:
        if host == plat:
            if best is None or len(plat) > len(best):
                best = plat
        elif host.endswith("." + plat):
            if best is None or len(plat) > len(best):
                best = plat
    if best is None:
        return None, None
    if host == best:
        return best, ""
    return best, host[: -(len(best) + 1)]


def analyze(url):
    """Return a verdict dict:
        {is_lookalike, confidence, brand, reason, rule}
    Rules are ordered; the allowlist short-circuits first."""
    raw = (url or "").strip()
    if not raw.lower().startswith(("http://", "https://")):
        raw = "http://" + raw

    parsed = urllib.parse.urlparse(raw)
    host = (parsed.hostname or "").lower()
    ext = _extract(raw)
    registered = (ext.registered_domain or "").lower()
    label = (ext.domain or "").lower()
    subdomain = (ext.subdomain or "").lower()
    tld = (ext.suffix or "").split(".")[-1]

    def verdict(flag, conf, brand, reason, rule):
        return {"is_lookalike": flag, "confidence": conf, "brand": brand,
                "reason": reason, "rule": rule}

    # Platform detection must not depend on how tldextract splits the host --
    # whether github.io is treated as a public suffix changes registered/
    # subdomain, but not this.
    platform, platform_sub = _resolve_platform(host)
    if platform is not None:
        registered = platform
        subdomain = platform_sub or ""
        label = platform.split(".")[0]

    # -- R0: genuine brand domain -> SAFE, unconditionally ------------------
    # This is the safety property. No heuristic can override it, so the real
    # paypal.com can never be flagged.
    if platform is None and registered in ALL_LEGIT_DOMAINS:
        return verdict(False, 0.0, None,
                       "Registered domain is a verified brand domain.", "R0")

    # -- R1: punycode / non-ASCII host that folds onto a brand --------------
    if _has_punycode(host) or _non_ascii(host):
        try:
            decoded = host.encode("ascii").decode("idna") if _has_punycode(host) else host
        except Exception:
            decoded = host
        folded = _fold(decoded)
        for brand in BRAND_DOMAINS:
            if brand in folded:
                return verdict(True, 0.99, brand,
                               f"Internationalised domain resolves to a "
                               f"lookalike of '{brand}'.", "R1")
        return verdict(True, 0.70, None,
                       "Internationalised/punycode domain -- verify carefully.",
                       "R1")

    folded_label = _fold(label)
    folded_registered = _fold(registered)

    # On a shared hosting platform the registered domain is the PLATFORM's own
    # (github.io, pages.dev, vercel.app). Its name legitimately appears there,
    # so the rules that inspect the registered domain (R2/R3/R4) must not fire
    # -- otherwise every *.github.io page reads as a 'github' imitation. Only
    # the subdomain carries signal on these hosts, which is R5's job.
    on_platform = platform is not None

    # -- R2: folded label IS a brand, but domain is not the brand's ---------
    # catches paypa1.com, app1e.com, g00gle.net, netf1ix.com
    for brand in BRAND_DOMAINS:
        if not on_platform and folded_label == brand:
            return verdict(True, 0.97, brand,
                           f"Domain '{label}' is a character-substituted "
                           f"imitation of '{brand}'.", "R2")

    # -- R3: brand token inside a registered domain that is not the brand's -
    # catches paypal-secure.com, paypalsupport.com, login-steam.com,
    # fedexverify.com, roblox.com.am
    for brand in BRAND_DOMAINS:
        if not on_platform and _brand_matches(brand, registered, parsed.path):
            sus = " on a low-reputation TLD" if tld in SUSPICIOUS_TLDS else ""
            conf = 0.95 if tld in SUSPICIOUS_TLDS else 0.90
            return verdict(True, conf, brand,
                           f"'{brand}' appears in the domain '{registered}', "
                           f"which is not owned by {brand}{sus}.", "R3")

    # -- R4: near-miss edit distance from a brand ---------------------------
    # catches ppaypal, payapl, aple, amazn. Guarded by length so short
    # unrelated words do not trip it.
    if len(folded_label) >= 5 and not on_platform:
        for brand in BRAND_DOMAINS:
            if len(brand) < 5:
                continue
            d = _levenshtein(folded_label, brand, cap=2)
            if 1 <= d <= 2 and abs(len(folded_label) - len(brand)) <= 2:
                return verdict(True, 0.88, brand,
                               f"Domain '{label}' is {d} character(s) from "
                               f"'{brand}' -- possible typosquat.", "R4")

    # -- R5: brand token in the subdomain of an unrelated domain ------------
    folded_sub = _fold(subdomain)
    sub_tokens = _tokens(subdomain)
    for brand in BRAND_DOMAINS:
        if _brand_matches(brand, subdomain, registered + " " + parsed.path):
            # Official organisation pages on code hosts: the subdomain is
            # EXACTLY the brand name (microsoft.github.io, netflix.github.io).
            # Brand-plus-anything (instagram-cmd, netflixcln) is not exempt.
            if platform in CODE_HOSTS and folded_sub == brand:
                return verdict(False, 0.0, None,
                               f"Subdomain matches the '{brand}' organisation "
                               f"exactly on a code-hosting platform -- "
                               f"official project page.", "R0-org")
            if on_platform:
                # Legitimate demo/clone projects are named this way too, so
                # this is surfaced as caution rather than a confident verdict.
                return verdict(True, 0.60, brand,
                               f"'{brand}' appears in a subdomain on the shared "
                               f"host '{registered}'. This may be a demo or "
                               f"clone project, but brand impersonation on free "
                               f"hosting is common -- do not enter credentials.",
                               "R5-platform")
            return verdict(True, 0.92, brand,
                           f"'{brand}' appears in the subdomain of unrelated "
                           f"domain '{registered}'.", "R5")

    # -- R6: brand token anywhere in the URL plus a suspicious TLD ----------
    whole = _fold(f"{host}{parsed.path}")
    if tld in SUSPICIOUS_TLDS:
        for brand in BRAND_DOMAINS:
            if _brand_matches(brand, whole):
                return verdict(True, 0.85, brand,
                               f"'{brand}' referenced on a low-reputation "
                               f"'.{tld}' domain.", "R6")

    return verdict(False, 0.0, None, "No lookalike pattern detected.", None)


if __name__ == "__main__":
    cases = [
        ("https://www.paypal.com/login", False),
        ("https://paypal.com/signin", False),
        ("https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M", False),
        ("https://www.hdfcbank.com/personal/borrow/popular-loans", False),
        ("https://stackoverflow.com/questions/tagged/python", False),
        ("https://docs.python.org/3/library/json.html", False),
        ("http://paypa1-secure.com/account/verify", True),
        ("https://paypalsupport.com/", True),
        ("https://login-steam.com/", True),
        ("https://fedexverify.com/", True),
        ("https://dropbox.site/", True),
        ("https://linkedin.info/", True),
        ("https://www.roblox.com.am/users/378768172502/profile", True),
        ("https://ledger--live-auth.pages.dev/", True),
        ("https://www.netflixcln.vercel.app/", True),
        # fix 1 -- short brand names
        ("https://verify-dhl.com/signin", True),
        ("https://dhl-secure.secure-portal.com/signin", True),
        ("https://dhl.account-services.net/secure/update", True),
        ("https://groups.google.com/g/some-group", False),
        ("https://www.startups.com/", False),
        ("https://backups.example.org/", False),
        # fix 2 -- github.io user pages are no longer blanket-trusted
        ("https://instagram-cmd.github.io/accounts/index.html", True),
        # fix 3 -- official org pages on code hosts stay clean
        ("https://microsoft.github.io/monaco-editor/", False),
        ("https://netflix.github.io/", False),
        ("https://reactjs.github.io/react-docs/", False),
        ("https://johnsmith.github.io/portfolio/", False),
        # bug A regressions -- platform's own name must not read as a brand
        ("https://pytorch.github.io/", False),
        ("https://docs.pages.dev/", False),
        ("https://nextjs-commerce.vercel.app/", False),
        ("https://myportfolio.pages.dev/", False),
        # bug B -- hyphen-separated short brands
        ("https://support-dhl.com/account/verify", True),
        ("https://account-dhl.com/secure/update", True),
        ("https://ups-tracking-update.com/", True),
        # parse-independence: these must hold under either tldextract split
        ("https://google.github.io/styleguide/", False),
        ("https://apple.github.io/swift-format/", False),
        ("https://paypal-verify.pages.dev/", True),
        ("https://a.b.netflix-login.pages.dev/", True),
        ("https://sub.docs.pages.dev/", False),
        # fused short brands -- newly caught
        ("https://dhlaccount.com/signin", True),
        ("https://dhlsupport.com/signin", True),
        ("https://dhlhelp.com/account/verify", True),
        ("https://dhlverify.com/signin", True),
        ("https://upstracking.com/parcel", True),
        # ordinary words containing a short brand -- must stay clean
        ("https://upstream.com/", False),
        ("https://upsell.com/pricing", False),
        ("https://startups.com/", False),
        ("https://backups.example.org/", False),
        ("https://groups.io/g/some-group", False),
        ("https://closeups-photography.com/portfolio", False),
    ]
    print(f"{'URL':<58} {'expect':<7} {'got':<7} rule")
    print("-" * 88)
    ok = 0
    for u, want in cases:
        v = analyze(u)
        got = v["is_lookalike"]
        mark = "ok " if got == want else "XX "
        ok += (got == want)
        print(f"{mark}{u[:55]:<55} {str(want):<7} {str(got):<7} {v['rule']}")
        if got:
            print(f"     -> {v['reason']}")
    print(f"\n{ok}/{len(cases)} correct")
