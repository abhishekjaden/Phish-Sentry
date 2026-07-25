"""
url_features_v2.py -- modern URL feature extractor.

WHY THIS EXISTS
---------------
The original 20-feature set (url_features.py, after Vrbancic) achieved 96.3%
recall on its own held-out split but only 38.3% on live phishing
(see EVALUATION.md). The misses clustered on phishing hosted on legitimate
platforms: *.pages.dev, *.blogspot.com, *.vercel.app, *.godaddysites.com.

Root cause: for a phishing page at "3rf3x34x.pages.dev", every original feature
reports benign -- the registered domain is Cloudflare's (old, valid WHOIS, real
ASN, normal TTL), the URL is short, no directory depth. A legitimate
"myportfolio.pages.dev" looks nearly identical. The discriminating signal is
that one subdomain is a random string and the other is a word, and NO original
feature measures that.

This module adds that signal: subdomain entropy, digit ratio, consonant runs,
word-likeness, plus brand-impersonation and hosting-platform flags.

DESIGN CHOICE: zero network calls. Every feature is computed from the URL
string. Consequences:
  + runs in ~1ms, so it is reliable in Lambda (no WHOIS timeouts)
  + no 5-features-sentinel-out degradation (the old set lost 2.2 of 5 network
    features on average during live measurement)
  - loses domain-age signal, which was genuinely predictive for the DEDICATED
    malicious domains the old model did catch. Retraining must confirm the new
    features recover that ground; do not assume it.
"""

import math
import re
import urllib.parse

import tldextract

# ---------------------------------------------------------------------------
# Known user-content hosting platforms. Phishing on these is invisible to
# registration-based features because the registered domain is the platform's.
# This list is a starting point, not exhaustive -- it needs maintenance, which
# is a real limitation to document.
# ---------------------------------------------------------------------------
HOSTING_PLATFORMS = {
    # observed in the live-feed misses
    "pages.dev", "blogspot.com", "vercel.app", "netlify.app",
    "godaddysites.com", "weeblysite.com", "typedream.app", "edgeone.dev",
    "replit.app", "wasmer.app", "lovable.app",
    # other common ones
    "github.io", "gitlab.io", "web.app", "firebaseapp.com",
    "herokuapp.com", "azurewebsites.net", "onrender.com",
    "surge.sh", "glitch.me", "repl.co", "workers.dev",
    "wixsite.com", "squarespace.com", "webflow.io", "framer.website",
    "notion.site", "carrd.co", "bubbleapps.io", "softr.app",
    "sharepoint.com", "amazonaws.com", "cloudfront.net",
    "wordpress.com", "tumblr.com", "medium.com",
    "000webhostapp.com", "infinityfreeapp.com", "epizy.com",
    "duckdns.org", "ngrok.io", "ngrok-free.app", "trycloudflare.com",
}

# Brands most commonly impersonated. A brand token appearing in the SUBDOMAIN
# or PATH while the registered domain does not belong to that brand is one of
# the strongest modern phishing signals.
BRAND_TOKENS = {
    "paypal", "apple", "icloud", "appleid", "microsoft", "office365",
    "outlook", "netflix", "amazon", "prime", "google", "gmail",
    "facebook", "instagram", "whatsapp", "meta", "linkedin",
    "roblox", "steam", "discord", "twitch", "spotify",
    "ledger", "trezor", "metamask", "coinbase", "binance", "uniswap",
    "chase", "wellsfargo", "bankofamerica", "citibank", "hsbc", "barclays",
    "dhl", "fedex", "ups", "usps", "royalmail",
    "netflix", "disney", "hulu", "adobe", "dropbox", "docusign",
    "wetransfer", "zoom", "teams", "onedrive", "sharepoint",
    "sbi", "hdfc", "icici", "axis", "paytm", "phonepe",
}

# Legitimate registered domains for those brands, so we do not flag the real
# thing. Keyed by brand token -> set of registered domains that are genuinely
# theirs.
BRAND_LEGIT_DOMAINS = {
    "paypal": {"paypal.com"},
    "apple": {"apple.com", "icloud.com"},
    "icloud": {"apple.com", "icloud.com"},
    "appleid": {"apple.com"},
    "microsoft": {"microsoft.com", "live.com", "msn.com", "office.com"},
    "office365": {"microsoft.com", "office.com"},
    "outlook": {"microsoft.com", "live.com", "outlook.com"},
    "netflix": {"netflix.com"},
    "amazon": {"amazon.com", "amazon.in", "amazon.co.uk", "aws.amazon.com"},
    "google": {"google.com", "google.co.in", "withgoogle.com"},
    "gmail": {"google.com", "gmail.com"},
    "facebook": {"facebook.com", "fb.com"},
    "instagram": {"instagram.com"},
    "whatsapp": {"whatsapp.com"},
    "linkedin": {"linkedin.com"},
    "roblox": {"roblox.com"},
    "steam": {"steampowered.com", "steamcommunity.com"},
    "discord": {"discord.com", "discordapp.com"},
    "spotify": {"spotify.com"},
    "ledger": {"ledger.com"},
    "trezor": {"trezor.io"},
    "metamask": {"metamask.io"},
    "coinbase": {"coinbase.com"},
    "binance": {"binance.com"},
    "uniswap": {"uniswap.org"},
    "dhl": {"dhl.com", "dhl.de"},
    "fedex": {"fedex.com"},
    "ups": {"ups.com"},
    "usps": {"usps.com"},
    "dropbox": {"dropbox.com"},
    "docusign": {"docusign.com", "docusign.net"},
    "zoom": {"zoom.us"},
}

# Credential-harvest vocabulary in the path.
ACTION_TOKENS = {
    "login", "signin", "sign-in", "verify", "verification", "account",
    "secure", "security", "update", "confirm", "auth", "authenticate",
    "password", "passwd", "recover", "recovery", "unlock", "validate",
    "billing", "payment", "invoice", "wallet", "seed", "restore",
}

# TLDs disproportionately used for abuse (cheap or free registration).
SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq",           # historically free
    "cfd", "top", "xyz", "icu", "buzz", "monster", "click", "link",
    "rest", "surf", "bar", "beauty", "quest", "sbs", "cyou", "lol",
    "work", "date", "loan", "men", "stream", "download",
}

VOWELS = set("aeiou")

FEATURE_ORDER = [
    # --- retained lexical features (these were doing useful work) ---
    "length_url", "qty_slash_url", "qty_dot_url", "qty_hyphen_url",
    "qty_digit_url", "qty_at_url", "qty_question_url", "qty_equal_url",
    "qty_percent_url", "qty_underscore_url",
    "domain_length", "qty_dot_domain", "qty_hyphen_domain",
    "qty_digit_domain",
    "directory_length", "qty_slash_directory", "file_length",
    "qty_params",
    # --- NEW: subdomain structure (the platform-abuse signal) ---
    "is_hosting_platform",
    "subdomain_length", "subdomain_label_count",
    "subdomain_entropy", "subdomain_digit_ratio", "subdomain_hyphen_count",
    "subdomain_max_consonant_run", "subdomain_vowel_ratio",
    "subdomain_is_wordlike",
    "subdomain_nonword_ratio",
    "subdomain_has_no_words",
    "platform_and_random_subdomain",
    # --- NEW: brand impersonation ---
    "brand_token_present",
    "brand_impersonation",
    # --- NEW: intent / TLD ---
    "action_token_in_path",
    "suspicious_tld",
    "is_https",
    "has_ip_host",
    "qty_subdomain_digits_runs",
]


# ---------------------------------------------------------------------------
# Compact word list for subdomain analysis. Not a full dictionary -- just
# enough common English plus web/tech/brand vocabulary to tell a real word
# from pronounceable nonsense. Deliberately small so it ships anywhere.
# ---------------------------------------------------------------------------
COMMON_WORDS = set("""
about access account action active add admin adventure agency air alert all
alpha analytics app apps archive area art article asset auth auto award back
bank base basic beach best beta big bio black blog blue board body book boot
box brand bright build business buy cafe calendar call camp car card care
career cart case cash center central chain chart chat check child choice city
class clean clear click client cloud club code coffee cold college color
comfort commerce common community company compare connect contact content
control cook cool copy core corner cost count country course craft create
creative credit crew cross crypto culture current custom cyber daily dance
dark dash data date deal dear deep default delta demo design desk detail dev
device digital direct discover display doc docs document dog domain done door
double download draw dream drive drop dynamic early earth east easy eat eco
edge edit edu education effect elite email energy engine enjoy enter
enterprise entry equal error escape essay event every exact example exchange
expert explore export express extra face fact fair family farm fashion fast
feature feed field file film final finance find fire first fit five fix flash
flat flex flight flow focus food foot force forest form forum forward found
frame free fresh friend front fuel full fun fund future gadget gallery game
garden gate gear general get gift girl give glass global glow gold good grand
graph great green grid ground group grow guard guide hall hand happy harbor
hard harvest have head health heart help hero high hill hire history hobby
hold home honest hope horizon host hotel hour house hub human hunt ice idea
image impact import index india info inner input insight inspire install
insurance intel inter invest island issue item jet job join journal journey
joy jump junior just keep key kid kind king kit knowledge lab lake land
language large last launch law layer lead leaf learn left legal level library
life light like line link list live load loan local lock log logic login long
look loop love lower loyal luck lunar machine made magic mail main major make
manage map mark market master match matter max media medical meet mega member
memory menu merge meta metric micro mid mind mine mini minute mirror mission
mobile mode model modern module money monitor month moon more morning motion
motor mount move movie multi music must name nano nation native natural nature
near neat need neo nest net network never new news next nice night noble node
north note nova now number object ocean offer office official online only open
option orange order origin other outdoor output over pack page paint pair
panel paper park part partner party pass past path pay peak people perfect
performance person phone photo pick picture piece pilot pink pixel place plan
plant plate play plus pocket point polar policy pool pop portal portfolio
post power practice premium press price prime print pro product profile
project promo proper protect proto public pulse pure purple push quality
quantum query quest quick quiet quiz radar radio rail rain range rapid rate
reach read ready real record red reference refresh region register relay
release remote render rent repair report request research reserve resource
rest result retail return review rich ride right ring rise river road rock
role room root round route royal rule run rush safe sale salt sample save
scale scan school science score screen script sea search season secret
section secure select self sell send senior sense serve service set setup
seven shade shadow shape share sharp shell shield ship shop short show side
sign signal silver simple single site six size sketch skill sky slate sleep
slide small smart smile snap snow social soft solar solid solution sound
source south space spark speak special speed sphere spirit split sport spot
spring square stack staff stage stand star start state station status stay
steel step stock stone stop storage store storm story stream street strong
student studio study style sub summer summit sun super supply support sure
surf swift switch system table tag take talent talk target task tax team tech
template ten term test text theme think third three time tiny title today
together token tool top total touch tour town track trade trail train
transfer travel tree trend trip true trust truth try tube turn twin type
ultra union unique unit unity universal update upload urban usage user valley
value vault vector venture verify version video view village vision visit
vital voice volt vote wave way wealth wear web welcome well west wheel white
wide wild win wind wing winter wire wise wish with wolf wonder wood word work
world write yard year yellow yes young zone zoom
""".split())

# tech / platform vocabulary that legitimately appears in subdomains
COMMON_WORDS |= set("""
api cdn dev docs git github gitlab lab npm pypi repo sdk staging prod preview
react vue angular svelte next nuxt node deno rust python java kotlin swift
django flask rails laravel spring aws azure gcp cloud docker kube helm
frontend backend fullstack ui ux css html js ts json yaml graphql rest grpc
auth oauth sso jwt saas paas iaas devops ci cd observability grafana
prometheus kafka redis postgres mysql mongo elastic sentry
""".split())


def _token_has_word(tok):
    """True if the token is, contains, or starts with a recognisable word.
    'myportfolio' -> yes (portfolio). 'drenix' -> no. 'nextjs' -> yes (next)."""
    tok = "".join(c for c in tok.lower() if c.isalpha())
    if len(tok) < 3:
        return True                      # too short to judge; do not penalise
    if tok in COMMON_WORDS:
        return True
    for w in COMMON_WORDS:
        if len(w) >= 4 and w in tok:
            return True
    return False


def _subdomain_word_stats(sub):
    """Return (nonword_ratio, has_no_words) over hyphen/digit-split tokens."""
    import re as _re
    tokens = [t for t in _re.split(r"[-_.\d]+", sub.lower()) if t]
    if not tokens:
        return 0.0, 0.0
    nonword = sum(0 if _token_has_word(t) else 1 for t in tokens)
    ratio = nonword / len(tokens)
    return ratio, (1.0 if nonword == len(tokens) else 0.0)


def _shannon_entropy(s):
    if not s:
        return 0.0
    counts = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _max_consonant_run(s):
    best = run = 0
    for ch in s.lower():
        if ch.isalpha() and ch not in VOWELS:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def _is_wordlike(s):
    """Cheap heuristic for 'looks like a word/phrase rather than random'.
    Word-like text has a reasonable vowel ratio, no very long consonant runs,
    and few digits. Deliberately simple -- no dictionary dependency, so it
    works offline and for non-English words too."""
    letters = [c for c in s.lower() if c.isalpha()]
    if len(letters) < 3:
        return 0.0
    vowel_ratio = sum(1 for c in letters if c in VOWELS) / len(letters)
    digit_ratio = sum(1 for c in s if c.isdigit()) / max(len(s), 1)
    ok = (0.20 <= vowel_ratio <= 0.60
          and _max_consonant_run(s) <= 4
          and digit_ratio < 0.25)
    return 1.0 if ok else 0.0


def _digit_runs(s):
    return len(re.findall(r"\d+", s))


_LEET = str.maketrans({
    "1": "l", "0": "o", "3": "e", "5": "s", "4": "a", "7": "t", "8": "b",
    "$": "s", "@": "a", "!": "i", "|": "l",
})


def _deleet(s):
    """Normalise common character substitutions so brand tokens still match:
    paypa1 -> paypal, g00gle -> google, app1e -> apple, netf1ix -> netflix.
    Hyphens/underscores are stripped so 'pay-pal' matches too."""
    return s.translate(_LEET).replace("-", "").replace("_", "").replace(".", "")


def extract_features_v2(url):
    """Return the feature vector in FEATURE_ORDER. Never raises; falls back to
    zeros for anything unparseable."""
    try:
        raw = (url or "").strip()
        if not raw.lower().startswith(("http://", "https://")):
            raw = "http://" + raw

        parsed = urllib.parse.urlparse(raw)
        host = (parsed.hostname or "").lower()
        path = parsed.path or ""
        query = parsed.query or ""

        ext = tldextract.extract(raw)
        subdomain = (ext.subdomain or "").lower()
        registered = (ext.registered_domain or "").lower()
        suffix = (ext.suffix or "").lower()

        # strip a leading "www" -- it carries no signal
        sub_labels = [l for l in subdomain.split(".") if l and l != "www"]
        sub_joined = ".".join(sub_labels)

        # directory vs file split, matching the original convention
        if "/" in path.strip("/"):
            directory = path.rsplit("/", 1)[0]
            filename = path.rsplit("/", 1)[1]
        else:
            directory = ""
            filename = path.strip("/")

        is_platform = 1.0 if registered in HOSTING_PLATFORMS else 0.0

        entropy = _shannon_entropy(sub_joined)
        digit_ratio = (sum(1 for c in sub_joined if c.isdigit())
                       / max(len(sub_joined), 1))
        letters = [c for c in sub_joined.lower() if c.isalpha()]
        vowel_ratio = (sum(1 for c in letters if c in VOWELS)
                       / max(len(letters), 1))
        wordlike = _is_wordlike(sub_joined)

        # THE key interaction: a hosting platform AND a random-looking
        # subdomain. Either alone is unremarkable; together it is the modern
        # phishing signature.
        nonword_ratio, has_no_words = _subdomain_word_stats(sub_joined)
        # A subdomain is "random" if it is phonotactically odd OR contains no
        # recognisable words. The second condition is what catches
        # pronounceable nonsense like "sp3ct-drenix-biz8-solvek-tranu".
        random_sub = 1.0 if (
            len(sub_joined) >= 4 and (wordlike == 0.0 or has_no_words == 1.0)
        ) else 0.0
        platform_and_random = 1.0 if (is_platform and random_sub) else 0.0

        # brand impersonation: brand token appears in subdomain or path, but
        # the registered domain is not one of that brand's real domains
        hay = f"{sub_joined} {path} {query}".lower()
        # also match against a leet-normalised copy of the WHOLE host+path so
        # character-substitution lookalikes (paypa1, g00gle) are caught
        hay_deleet = _deleet(f"{host} {path} {query}".lower())
        brand_present = 0.0
        impersonation = 0.0
        for brand in BRAND_TOKENS:
            if brand in hay or brand in hay_deleet:
                brand_present = 1.0
                legit = BRAND_LEGIT_DOMAINS.get(brand, set())
                if registered not in legit:
                    impersonation = 1.0
                    break

        action = 1.0 if any(t in path.lower() for t in ACTION_TOKENS) else 0.0
        susp_tld = 1.0 if suffix.split(".")[-1] in SUSPICIOUS_TLDS else 0.0
        has_ip = 1.0 if re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host) else 0.0

        feats = {
            "length_url": len(raw),
            "qty_slash_url": raw.count("/"),
            "qty_dot_url": raw.count("."),
            "qty_hyphen_url": raw.count("-"),
            "qty_digit_url": sum(1 for c in raw if c.isdigit()),
            "qty_at_url": raw.count("@"),
            "qty_question_url": raw.count("?"),
            "qty_equal_url": raw.count("="),
            "qty_percent_url": raw.count("%"),
            "qty_underscore_url": raw.count("_"),
            "domain_length": len(registered),
            "qty_dot_domain": registered.count("."),
            "qty_hyphen_domain": registered.count("-"),
            "qty_digit_domain": sum(1 for c in registered if c.isdigit()),
            "directory_length": len(directory),
            "qty_slash_directory": directory.count("/"),
            "file_length": len(filename),
            "qty_params": len(urllib.parse.parse_qsl(query)),
            "is_hosting_platform": is_platform,
            "subdomain_length": len(sub_joined),
            "subdomain_label_count": len(sub_labels),
            "subdomain_entropy": entropy,
            "subdomain_digit_ratio": digit_ratio,
            "subdomain_hyphen_count": sub_joined.count("-"),
            "subdomain_max_consonant_run": _max_consonant_run(sub_joined),
            "subdomain_vowel_ratio": vowel_ratio,
            "subdomain_is_wordlike": wordlike,
            "subdomain_nonword_ratio": nonword_ratio,
            "subdomain_has_no_words": has_no_words,
            "platform_and_random_subdomain": platform_and_random,
            "brand_token_present": brand_present,
            "brand_impersonation": impersonation,
            "action_token_in_path": action,
            "suspicious_tld": susp_tld,
            "is_https": 1.0 if parsed.scheme == "https" else 0.0,
            "has_ip_host": has_ip,
            "qty_subdomain_digits_runs": _digit_runs(sub_joined),
        }
        return [float(feats[k]) for k in FEATURE_ORDER]

    except Exception:
        return [0.0] * len(FEATURE_ORDER)


if __name__ == "__main__":
    # Sanity check: the pair the original feature set could not separate.
    pairs = [
        ("https://3rf3x34x.pages.dev/", "phishing (random sub on platform)"),
        ("https://myportfolio.pages.dev/", "legit (word sub on platform)"),
        ("https://satya-1205.github.io/NetflixWebsite/", "phishing (brand in path)"),
        ("https://eshwarkole1641-esh.github.io/spotify-clone/", "phishing (brand in path)"),
        ("https://reactjs.github.io/react-docs/", "legit-ish github.io"),
        ("http://paypa1-secure.com/account/verify", "phishing (lookalike)"),
        ("https://www.paypal.com/signin", "legit paypal"),
        ("https://ledger--live-auth.pages.dev/", "phishing (brand + platform)"),
        ("https://github.com", "legit"),
    ]
    key = ["is_hosting_platform", "subdomain_is_wordlike",
           "platform_and_random_subdomain", "brand_impersonation",
           "subdomain_entropy", "action_token_in_path", "suspicious_tld"]
    idx = {k: FEATURE_ORDER.index(k) for k in key}
    print(f"{'URL':<52} " + " ".join(f"{k[:9]:>9}" for k in key))
    print("-" * 52 + " " + "-" * (10 * len(key)))
    for u, label in pairs:
        f = extract_features_v2(u)
        vals = " ".join(f"{f[idx[k]]:>9.2f}" for k in key)
        print(f"{u[:52]:<52} {vals}")
        print(f"{'  -> ' + label:<52}")
