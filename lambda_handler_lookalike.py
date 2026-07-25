"""
PhishSentry lookalike-domain check -- public Lambda endpoint.

Wraps lookalike_detector.analyze(). No model, no network calls, no state.

PRIVACY: this handler never logs the submitted URL or any host derived from
it. Only the verdict and the rule that fired are logged, both of which are
non-identifying. Do not add URL logging without updating the published privacy
policy -- the policy states that URLs are not stored.

SCOPE (must match the store listing): detects lookalike and typosquat domains.
Does NOT detect phishing hosted on legitimate platforms under an unrelated
subdomain -- see EVALUATION.md for the measurement behind that limitation.
"""

import json
import logging
import os

from lookalike_detector import analyze

logger = logging.getLogger()
logger.setLevel(logging.INFO)

MAX_URL_LEN = 2048

CORS_HEADERS = {
    "Content-Type": "application/json",
    # Set ALLOWED_ORIGIN to chrome-extension://<your-published-id> once the
    # extension is live. "*" is acceptable for an unauthenticated, read-only,
    # stateless endpoint but tightening it is better hygiene.
    "Access-Control-Allow-Origin": os.environ.get("ALLOWED_ORIGIN", "*"),
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
    "Cache-Control": "no-store",
}


def _resp(status, body):
    return {"statusCode": status, "headers": CORS_HEADERS,
            "body": json.dumps(body)}


def _method(event):
    return (event.get("requestContext", {}).get("http", {}).get("method")
            or event.get("httpMethod") or "POST").upper()


def lambda_handler(event, context):
    if _method(event) == "OPTIONS":
        return _resp(200, {"ok": True})

    try:
        body = event.get("body") or "{}"
        if event.get("isBase64Encoded"):
            import base64
            body = base64.b64decode(body).decode("utf-8")
        data = json.loads(body)
    except Exception:
        logger.info("bad_request reason=unparseable_body")
        return _resp(400, {"error": "Invalid JSON body"})

    url = (data.get("url") or "").strip()
    if not url:
        logger.info("bad_request reason=missing_url")
        return _resp(400, {"error": "Missing 'url'"})
    if len(url) > MAX_URL_LEN:
        logger.info("bad_request reason=url_too_long")
        return _resp(400, {"error": "URL too long"})

    try:
        v = analyze(url)
        # Aggregate-only logging: verdict and rule, never the URL or host.
        logger.info("checked is_lookalike=%s rule=%s",
                    v["is_lookalike"], v["rule"])
        return _resp(200, {
            "is_lookalike": v["is_lookalike"],
            "confidence": v["confidence"],
            "brand": v["brand"],
            "reason": v["reason"],
            "rule": v["rule"],
            "scope_note": ("Checks for lookalike and typosquat domains only. "
                           "Does not detect phishing hosted on legitimate "
                           "platforms under an unrelated subdomain."),
        })
    except Exception as e:
        # Log the exception TYPE only -- messages can contain the URL.
        logger.error("check_failed error_type=%s", type(e).__name__)
        return _resp(500, {"error": "Check failed"})
