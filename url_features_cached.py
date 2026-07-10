"""
Caching + circuit-breaker wrapper around the slow WHOIS-based URL feature extraction.
- Redis cache: repeat scans of the same domain return instantly.
- Circuit breaker: if WHOIS/HTTP calls keep failing, stop hammering them and
  return a fast degraded result instead of hanging.
"""
import os
import json
import hashlib
import redis
import pybreaker
from url_features import extract_features

# ---- Redis connection ----
REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
_redis_client = redis.from_url(REDIS_URL, decode_responses=True)

CACHE_TTL_SECONDS = 24 * 60 * 60  # cache URL features for 24 hours

# ---- Circuit breaker ----
# Opens after 5 consecutive failures, stays open for 60s before retrying.
url_breaker = pybreaker.CircuitBreaker(
    fail_max=5,
    reset_timeout=60,
    name='url_feature_extraction'
)


def _cache_key(url):
    """Build a stable Redis key for a URL."""
    digest = hashlib.sha256(url.encode('utf-8')).hexdigest()
    return f"url_features:{digest}"


@url_breaker
def _extract_with_breaker(url):
    """The actual slow extraction, protected by the circuit breaker."""
    return extract_features(url)


def get_url_features(url):
    """
    Return URL features, using cache first, then live extraction.
    Returns a dict: {features, source, [error]}
    """
    key = _cache_key(url)

    # 1. Try cache
    try:
        cached = _redis_client.get(key)
        if cached is not None:
            return {"features": json.loads(cached), "source": "cache"}
    except redis.RedisError as e:
        print(f"[cache] Redis read error: {e}")

    # 2. Live extraction, protected by circuit breaker
    try:
        features = _extract_with_breaker(url)
    except pybreaker.CircuitBreakerError:
        # Breaker is open — external services are failing, return degraded result
        return {"features": None, "source": "circuit_open",
                "error": "Feature service temporarily unavailable"}
    except Exception as e:
        return {"features": None, "source": "error", "error": str(e)}

    # 3. Store in cache (best-effort)
    try:
        _redis_client.setex(key, CACHE_TTL_SECONDS, json.dumps(features))
    except redis.RedisError as e:
        print(f"[cache] Redis write error: {e}")

    return {"features": features, "source": "live"}