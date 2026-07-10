# url_resolver.py
# Day 29: expand URL shorteners and follow redirect chains to the TRUE final
# destination before the model classifies it. Phishers hide behind shorteners
# (bit.ly/...) and multi-hop redirects; classifying the raw link misses the real
# target. Resolution is hard-bounded (overall budget + per-hop timeout) and
# SSRF-guarded: it never follows a hop whose host resolves to a private /
# internal / loopback address.
import time
import socket
import ipaddress
import urllib.parse
import concurrent.futures

import requests

# Known URL-shortener hosts (the common ones; not exhaustive).
SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
    "rebrand.ly", "cutt.ly", "shorturl.at", "t.ly", "rb.gy", "bit.do",
    "soo.gd", "clck.ru", "v.gd", "tiny.cc", "lnkd.in", "db.tt", "qr.ae",
    "adf.ly", "bitly.com", "shorte.st", "tr.im", "snip.ly", "shor.by",
}

MAX_HOPS = 5
PER_HOP_TIMEOUT = 3       # seconds per HTTP request
OVERALL_BUDGET = 8        # seconds for the whole resolution
UA = "Mozilla/5.0 (compatible; PhishSentry/1.0; +https://phishsentry.app)"


def _call_timeout(fn, timeout, default=None):
    """Run blocking fn() with a hard wall-clock timeout; default on timeout/error."""
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        return ex.submit(fn).result(timeout=timeout)
    except Exception:
        return default
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def is_shortener(host):
    host = (host or "").lower().lstrip(".")
    if host.startswith("www."):
        host = host[4:]
    return host in SHORTENERS


def _ip_is_public(ip_str):
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    return not (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified)


def host_is_public(host):
    """True only if `host` resolves exclusively to public/global IP addresses.
    Used as an SSRF guard before fetching any URL hop."""
    if not host:
        return False
    # IP literal in the host?
    try:
        ipaddress.ip_address(host)
        return _ip_is_public(host)
    except ValueError:
        pass
    infos = _call_timeout(lambda: socket.getaddrinfo(host, None), PER_HOP_TIMEOUT, default=None)
    if not infos:
        return False
    for info in infos:
        if not _ip_is_public(info[4][0]):
            return False
    return True


def resolve_final_url(url):
    """
    Follow redirects hop-by-hop to the final URL, with SSRF protection.

    Returns:
      {
        "original_url": str,
        "final_url": str,          # best-effort true destination
        "redirected": bool,
        "redirect_chain": [str],   # intermediate URLs (excludes the original)
        "was_shortener": bool,     # original host was a known shortener
        "blocked": bool,           # a hop pointed at a private/internal host
      }
    """
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    original = url
    orig_host = urllib.parse.urlparse(url).hostname
    was_shortener = is_shortener(orig_host)

    chain = []
    current = url
    blocked = False
    start = time.time()

    for _ in range(MAX_HOPS):
        if time.time() - start > OVERALL_BUDGET:
            break
        host = urllib.parse.urlparse(current).hostname
        if not host_is_public(host):
            blocked = blocked or (host is not None)
            break

        url_for_hop = current  # bind for the closure

        def _do():
            try:
                r = requests.head(url_for_hop, allow_redirects=False,
                                  timeout=PER_HOP_TIMEOUT, headers={"User-Agent": UA})
                # Some servers reject/mishandle HEAD; fall back to a streamed GET.
                if r.status_code in (403, 405, 501) or r.status_code >= 500:
                    r = requests.get(url_for_hop, allow_redirects=False, stream=True,
                                     timeout=PER_HOP_TIMEOUT, headers={"User-Agent": UA})
                    r.close()
                return r.status_code, r.headers.get("Location")
            except requests.exceptions.RequestException:
                return None

        res = _call_timeout(_do, PER_HOP_TIMEOUT + 1, default=None)
        if not res:
            break
        status, location = res
        if status and 300 <= status < 400 and location:
            nxt = urllib.parse.urljoin(current, location)
            if nxt == current or nxt in chain:   # loop / self-redirect guard
                break
            chain.append(nxt)
            current = nxt
            continue
        break  # not a redirect -> this is the final URL

    return {
        "original_url": original,
        "final_url": current,
        "redirected": len(chain) > 0,
        "redirect_chain": chain,
        "was_shortener": was_shortener,
        "blocked": blocked,
    }


if __name__ == "__main__":
    # Tests that don't need outbound network: shortener + SSRF logic.
    print("is_shortener tests:")
    for h, exp in [("bit.ly", True), ("www.bit.ly", True), ("github.com", False),
                   ("t.co", True), ("example.com", False)]:
        print(f"  {h:16s} -> {is_shortener(h)} (expect {exp})")

    print("\nhost_is_public (IP literals, no network):")
    for h, exp in [("8.8.8.8", True), ("127.0.0.1", False), ("10.0.0.5", False),
                   ("192.168.1.1", False), ("169.254.1.1", False), ("172.16.0.1", False)]:
        print(f"  {h:16s} -> {host_is_public(h)} (expect {exp})")
