# homograph.py
# Detects IDN/punycode hostnames and mixed-script "homograph" lookalike domains
# (e.g. "аpple.com" where the first letter is a Cyrillic а, U+0430).
#
# This is a HEURISTIC overlay that complements the trained URL model — the
# model's 20 length/character features do not capture script-confusion attacks,
# so a domain like "аpple.com" can look structurally normal yet be malicious.
#
# Design goal: high-confidence, low-false-positive. We flag a homograph only
# when a single hostname label MIXES Latin with another script (Cyrillic, Greek,
# Armenian). A wholly non-Latin label (e.g. a legitimate Russian or Greek IDN
# site) is reported as IDN but NOT flagged as a homograph, to avoid penalising
# legitimate internationalised domains.
import unicodedata
import urllib.parse

# Scripts most commonly abused to imitate Latin characters.
_CONFUSABLE_SCRIPTS = ("CYRILLIC", "GREEK", "ARMENIAN")

# Common single-character confusables mapping non-Latin -> their Latin lookalike.
# Used to detect labels that *read* as a Latin word but are built from lookalikes
# (e.g. all-Cyrillic "аррӏе" -> "apple"). Not exhaustive, but covers the
# characters overwhelmingly used in real homograph attacks.
_CONFUSABLES = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "ѕ": "s", "і": "i", "ј": "j", "ԁ": "d", "ո": "n", "м": "m", "т": "t",
    "к": "k", "ӏ": "l", "һ": "h", "ѵ": "v", "ԝ": "w", "ӡ": "3", "қ": "k",
    "ν": "v", "κ": "k", "ρ": "p", "τ": "t", "υ": "u", "ο": "o", "α": "a",
    "ι": "i", "β": "b", "η": "n", "ε": "e", "ϲ": "c", "ѡ": "w", "ԛ": "q",
}


def _skeleton_is_latin_spoof(label):
    """True if the label uses non-Latin lookalikes that, once mapped, read as a
    pure-Latin string (the all-Cyrillic 'аррӏе' -> 'apple' case)."""
    saw_confusable = False
    for ch in label:
        if ch.isascii():
            continue
        mapped = _CONFUSABLES.get(ch)
        if mapped is None:
            return False          # a non-Latin char with no Latin lookalike -> likely legit IDN
        saw_confusable = True
    return saw_confusable


def _scripts_in_label(label):
    """Return the set of scripts present among the alphabetic characters."""
    scripts = set()
    for ch in label:
        if ch.isascii():
            if ch.isalpha():
                scripts.add("LATIN")
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            scripts.add("UNKNOWN")
            continue
        matched = False
        for s in ("LATIN",) + _CONFUSABLE_SCRIPTS + ("HEBREW", "ARABIC", "HAN", "HIRAGANA", "KATAKANA"):
            if s in name:
                scripts.add(s)
                matched = True
                break
        if not matched:
            scripts.add("OTHER")
    return scripts


def _decode_punycode(host):
    """Best-effort decode of xn-- labels to their Unicode form."""
    try:
        return host.encode("ascii").decode("idna")
    except Exception:
        pass
    # Fall back to decoding individual xn-- labels.
    out = []
    for label in host.split("."):
        if label.startswith("xn--"):
            try:
                out.append(label[4:].encode("ascii").decode("punycode"))
            except Exception:
                out.append(label)
        else:
            out.append(label)
    return ".".join(out)


def analyze(url):
    """
    Returns:
      {
        "is_homograph": bool,   # strong signal: mixed-script lookalike
        "is_idn": bool,         # informational: domain uses non-ASCII / punycode
        "reason": str|None,     # human-readable explanation when flagged
        "decoded": str|None,    # Unicode form when the host was punycode
      }
    """
    result = {"is_homograph": False, "is_idn": False, "reason": None, "decoded": None}
    try:
        if not url.startswith(("http://", "https://")):
            url = "http://" + url
        host = (urllib.parse.urlparse(url).hostname or "").strip(".")
    except Exception:
        return result
    if not host:
        return result

    has_punycode = "xn--" in host
    decoded = _decode_punycode(host) if has_punycode else host
    if has_punycode and decoded != host:
        result["decoded"] = decoded

    reasons = []
    for label in decoded.split("."):
        if any(not ch.isascii() for ch in label):
            result["is_idn"] = True
        scripts = _scripts_in_label(label)
        non_latin_confusable = {s for s in scripts if s in _CONFUSABLE_SCRIPTS}
        # (A) Classic homograph: Latin mixed with a confusable script in one label.
        if "LATIN" in scripts and non_latin_confusable:
            pretty = ", ".join(s.capitalize() for s in sorted(non_latin_confusable))
            reasons.append(f"label '{label}' mixes Latin with {pretty}")
        # (B) All-lookalike spoof: non-Latin chars that map to a pure-Latin word.
        elif _skeleton_is_latin_spoof(label):
            spoofed = "".join(_CONFUSABLES.get(c, c) for c in label)
            reasons.append(f"label '{label}' imitates the Latin string '{spoofed}'")

    if reasons:
        result["is_homograph"] = True
        result["reason"] = "; ".join(reasons)
    return result


if __name__ == "__main__":
    tests = [
        "https://www.google.com",                 # plain Latin -> clean
        "https://github.com/login",               # clean
        "http://\u0430pple.com",                  # Cyrillic 'а' + Latin 'pple' -> HOMOGRAPH
        "http://p\u0430yp\u0430l.com/verify",     # Cyrillic 'а's mixed with Latin -> HOMOGRAPH
        "https://xn--80ak6aa92e.com",             # punycode of apple (all-Cyrillic) -> IDN
        "https://m\u00fcnchen.de",                # legit German IDN (Latin + umlaut) -> IDN, not homograph
        "https://\u043f\u0440\u0438\u043c\u0435\u0440.\u0440\u0444",  # all-Cyrillic legit -> IDN, not homograph
    ]
    for u in tests:
        r = analyze(u)
        print(f"{u}\n  -> {r}\n")
