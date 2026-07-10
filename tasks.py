"""
Celery background tasks for PhishGuard.
Slow work (WHOIS lookups, feature extraction) runs here instead of blocking web requests.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from celery import Celery

REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')

celery_app = Celery(
    'phishguard',
    broker=REDIS_URL,
    backend=REDIS_URL
)

celery_app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    result_expires=3600,          # results kept for 1 hour
    task_time_limit=60,           # hard kill a task after 60s
    task_soft_time_limit=45,      # soft warning at 45s
)


@celery_app.task(name='tasks.extract_url_features')
def extract_url_features_task(url):
    """
    Run URL feature extraction with caching + circuit breaker.
    """
    from url_features_cached import get_url_features
    try:
        result = get_url_features(url)
        if result.get('features') is not None:
            return {'status': 'success', 'features': result['features'],
                    'source': result['source'], 'url': url}
        else:
            return {'status': 'error', 'error': result.get('error'),
                    'source': result['source'], 'url': url}
    except Exception as e:
        return {'status': 'error', 'error': str(e), 'url': url}
@celery_app.task(name='tasks.compute_shap_explanation')
def compute_shap_explanation_task(text):
    """
    Call the inference service's SHAP endpoint in the background.
    Returns token-level explanation for the given text.
    """
    import os
    import requests
    inference_url = os.environ.get('INFERENCE_URL', 'http://inference:9000')
    try:
        resp = requests.post(
            f"{inference_url}/explain/text",
            json={"text": text},
            timeout=90
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": f"SHAP task failed: {str(e)}"}