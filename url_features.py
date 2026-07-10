# url_features.py
# Extracts the 20 Vrbancic URL features in the exact order the model expects.
# Failure sentinel is -1 (matching the training data), NOT 0.
#
# Day 29 hardening: every external network lookup (WHOIS, DNS, host resolution,
# HTTP timing) runs under a HARD wall-clock timeout. A slow or hanging
# third-party server can no longer stall the response — it fast-fails to the
# -1 sentinel, exactly as a failed lookup would.
import re
import urllib.parse
import socket
import time
import concurrent.futures
from datetime import datetime

import requests
import tldextract
import whois
import dns.resolver

# Order MUST match models/url_model_config.json -> "features"
FEATURE_ORDER = [
    "qty_slash_url", "length_url", "qty_dot_domain", "domain_length",
    "qty_dot_directory", "qty_slash_directory", "qty_equal_directory",
    "qty_at_directory", "qty_and_directory", "qty_percent_directory",
    "directory_length", "qty_hyphen_file", "qty_plus_file", "qty_dollar_file",
    "file_length", "time_response", "asn_ip", "time_domain_activation",
    "time_domain_expiration", "ttl_hostname",
]

# Hard caps (seconds) for each class of external lookup.
WHOIS_TIMEOUT = 5
RESOLVE_TIMEOUT = 3
DNS_TIMEOUT = 3
HTTP_TIMEOUT = 3


def _call_timeout(fn, timeout, default=None):
    """Run a blocking fn() with a hard wall-clock timeout.

    Returns fn()'s result, or `default` on timeout/exception. The worker thread
    is not waited on after a timeout (shutdown(wait=False)), so a hung lookup
    cannot block the request; it is left to expire on its own in the background.
    """
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        fut = ex.submit(fn)
        return fut.result(timeout=timeout)
    except Exception:
        return default
    finally:
        # Don't block on a still-running (hung) lookup.
        ex.shutdown(wait=False, cancel_futures=True)


def _dns_ttl(domain, timeout=DNS_TIMEOUT):
    """Real DNS TTL of the domain's A record. -1 on failure (dataset sentinel)."""
    try:
        r = dns.resolver.Resolver()
        r.timeout = timeout
        r.lifetime = timeout
        ans = r.resolve(domain, "A")
        return int(ans.rrset.ttl)
    except Exception:
        return -1


def _asn(ip, timeout=DNS_TIMEOUT):
    """Real ASN via Team Cymru's free DNS service. -1 on failure (dataset sentinel)."""
    try:
        rev = ".".join(reversed(ip.split(".")))
        query = f"{rev}.origin.asn.cymru.com"
        r = dns.resolver.Resolver()
        r.timeout = timeout
        r.lifetime = timeout
        ans = r.resolve(query, "TXT")
        txt = str(ans[0]).strip('"')          # e.g. "15169 | 8.8.8.0/24 | US | arin | ..."
        return int(txt.split("|")[0].strip().split()[0])
    except Exception:
        return -1


def _whois_dates(domain):
    """(creation_date, expiration_date) as datetimes, or (None, None). Bounded by timeout."""
    def _lookup():
        w = whois.whois(domain)
        cd = w.creation_date
        if isinstance(cd, list):
            cd = cd[0] if cd else None
        ed = w.expiration_date
        if isinstance(ed, list):
            ed = ed[0] if ed else None
        return cd, ed
    res = _call_timeout(_lookup, WHOIS_TIMEOUT, default=(None, None))
    return res if isinstance(res, tuple) else (None, None)


def extract_features(url, timeout=HTTP_TIMEOUT):
    """Return the 20 features as a list, in FEATURE_ORDER. -1 marks unknown/failed lookups."""
    try:
        if not url.startswith(("http://", "https://")):
            url = "http://" + url

        parsed = urllib.parse.urlparse(url)
        ext = tldextract.extract(url)
        domain = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain

        # Split path into directory + file (Vrbancic semantics)
        path = parsed.path
        if "/" in path:
            directory, file = path.rsplit("/", 1)
        else:
            directory, file = "", path

        f = []
        f.append(url.count("/"))            # 1  qty_slash_url
        f.append(len(url))                  # 2  length_url
        f.append(domain.count("."))         # 3  qty_dot_domain
        f.append(len(domain))               # 4  domain_length
        f.append(directory.count("."))      # 5  qty_dot_directory
        f.append(directory.count("/"))      # 6  qty_slash_directory
        f.append(directory.count("="))      # 7  qty_equal_directory
        f.append(directory.count("@"))      # 8  qty_at_directory
        f.append(directory.count("&"))      # 9  qty_and_directory
        f.append(directory.count("%"))      # 10 qty_percent_directory
        f.append(len(directory))            # 11 directory_length
        f.append(file.count("-"))           # 12 qty_hyphen_file
        f.append(file.count("+"))           # 13 qty_plus_file
        f.append(file.count("$"))           # 14 qty_dollar_file
        f.append(len(file))                 # 15 file_length

        # 16 time_response (ms) ; -1 on failure (requests already bounded by timeout)
        try:
            t0 = time.time()
            requests.get(url, timeout=timeout, allow_redirects=False)
            f.append((time.time() - t0) * 1000.0)
        except requests.exceptions.RequestException:
            f.append(-1)

        # Resolve IP once for ASN (hard-bounded so a slow resolver can't hang)
        ip = _call_timeout(lambda: socket.gethostbyname(domain), RESOLVE_TIMEOUT, default=None)

        f.append(_asn(ip) if ip else -1)    # 17 asn_ip ; -1 on failure

        # 18/19 domain activation & expiration (days) ; -1 on failure/timeout
        cd, ed = _whois_dates(domain)
        now = datetime.utcnow()
        f.append((now - cd.replace(tzinfo=None)).days if isinstance(cd, datetime) else -1)
        f.append((ed.replace(tzinfo=None) - now).days if isinstance(ed, datetime) else -1)

        f.append(_dns_ttl(domain))          # 20 ttl_hostname ; -1 on failure

    except Exception as e:
        print(f"Error extracting features for URL {url}: {e}")
        return [-1] * 20

    # Guarantee exactly 20
    if len(f) < 20:
        f.extend([-1] * (20 - len(f)))
    return f[:20]


def get_feature_descriptions():
    return list(FEATURE_ORDER)


if __name__ == "__main__":
    # Quick timeout self-test (no external deps needed for this part).
    import time as _t
    slow = lambda: _t.sleep(10) or "done"
    t0 = _t.time()
    r = _call_timeout(slow, 1, default="TIMED_OUT")
    print(f"_call_timeout returned {r!r} after {_t.time()-t0:.2f}s (should be ~1s, TIMED_OUT)")

    fast = lambda: 42
    print("_call_timeout fast:", _call_timeout(fast, 1, default=None))

    for u in ["https://www.google.com", "http://paypal-secure-login.tk/verify/account.html"]:
        feats = extract_features(u)
        print(f"\n{u}")
        for name, val in zip(FEATURE_ORDER, feats):
            print(f"  {name:24s} {val}")
