"""
PhishSentry public URL-check Lambda.

Replicates the production /predict/url path from inference_service.py EXACTLY:
  0. SSRF-guarded redirect/shortener resolution   (url_resolver)
  1. 20-feature extraction on the FINAL url       (url_features)
  2. XGBoost raw score -> isotonic calibration -> threshold
  3. Homograph/IDN overlay                        (homograph)
  4. Fresh-domain informational overlay

PRIVACY: this handler is stateless. It NEVER logs the submitted URL, the
resolved URL, or any derived host. Only aggregate counters and error types are
logged. Do not add URL logging without updating the published privacy policy.
"""

import json
import os
import logging

import numpy as np
import xgboost as xgb

import url_resolver
import homograph
from url_features import extract_features

# --- logging: aggregate only, never URLs -------------------------------------
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_here = os.path.dirname(os.path.abspath(__file__))

# --- model + calibration loaded once per container ---------------------------
with open(os.path.join(_here, "models", "url_model_config.json")) as _f:
    URL_CFG = json.load(_f)

URL_FEATURES = URL_CFG["features"]
URL_THRESHOLD = float(URL_CFG["threshold"])
URL_CALIB_X = np.array(URL_CFG["calibration"]["x"], dtype=float)
URL_CALIB_Y = np.array(URL_CFG["calibration"]["y"], dtype=float)

url_booster = xgb.Booster()
url_booster.load_model(os.path.join(_here, "models", "url_xgb_model.json"))

FRESH_DAYS = 30
MAX_URL_LEN = 2048

CORS_HEADERS = {
    "Content-Type": "application/json",
    # Extension service workers send an Origin of chrome-extension://<id>.
    # Set ALLOWED_ORIGIN env var to that value to lock this down after you know
    # your published extension ID. "*" is fine for an unauthenticated,
    # read-only scoring endpoint.
    "Access-Control-Allow-Origin": os.environ.get("ALLOWED_ORIGIN", "*"),
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
}


def _resp(status, body):
    return {
        "statusCode": status,
        "headers": CORS_HEADERS,
        "body": json.dumps(body),
    }


def score_url(raw_url):
    """The exact production inference path. Returns the same dict shape as
    /predict/url so the extension needs no response-parsing changes."""
    resolved = url_resolver.resolve_final_url(raw_url)
    target_url = resolved["final_url"]

    feats = extract_features(target_url)
    dm = xgb.DMatrix(np.array([feats], dtype=float), feature_names=URL_FEATURES)

    raw = float(url_booster.predict(dm)[0])
    calibrated = float(np.interp(raw, URL_CALIB_X, URL_CALIB_Y))
    model_is_phishing = bool(calibrated >= URL_THRESHOLD)

    homo = homograph.analyze(target_url)

    is_phishing = model_is_phishing
    reason = None
    confidence = calibrated * 100 if model_is_phishing else (1.0 - calibrated) * 100

    if homo["is_homograph"]:
        is_phishing = True
        reason = f"Homograph/IDN lookalike domain \u2014 {homo['reason']}"
        confidence = 99.0

    try:
        age_days = float(feats[URL_FEATURES.index("time_domain_activation")])
    except Exception:
        age_days = -1.0
    is_fresh = 0 <= age_days < FRESH_DAYS

    # how many of the 5 network-dependent features fell back to the -1 sentinel
    network_feats = [
        "time_response", "asn_ip", "time_domain_activation",
        "time_domain_expiration", "ttl_hostname",
    ]
    sentinel_count = sum(
        1 for name in network_feats
        if name in URL_FEATURES and float(feats[URL_FEATURES.index(name)]) == -1.0
    )

    return {
        "is_phishing": is_phishing,
        "confidence": float(confidence),
        "prediction_value": calibrated,
        "model_is_phishing": model_is_phishing,
        "homograph": homo,
        "resolved": resolved,
        "fresh_domain": {
            "is_fresh": bool(is_fresh),
            "age_days": int(age_days) if age_days >= 0 else None,
        },
        "reason": reason,
        # transparency: how many live-lookup features were unavailable.
        # 5 means every network feature sentinel'd out and the verdict rests on
        # lexical structure alone.
        "degraded_features": sentinel_count,
    }


def lambda_handler(event, context):
    # CORS preflight
    if (event.get("requestContext", {}).get("http", {}).get("method")
            or event.get("httpMethod")) == "OPTIONS":
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
    if not url.lower().startswith(("http://", "https://")):
        url = "http://" + url

    try:
        result = score_url(url)
        # aggregate-only log line: verdict + degradation, never the URL
        logger.info(
            "scored is_phishing=%s degraded_features=%s",
            result["is_phishing"], result["degraded_features"],
        )
        return _resp(200, result)
    except Exception as e:
        # log the exception TYPE only -- messages can contain the URL
        logger.error("scoring_failed error_type=%s", type(e).__name__)
        return _resp(500, {"error": "Scoring failed"})
