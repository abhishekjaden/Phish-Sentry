"""
Structured JSON logging for PhishGuard.
Every log line is a JSON object with consistent fields, ready for
log aggregators (CloudWatch, Loki, Datadog) and the metrics layer.
"""
import logging
import sys
import structlog


def configure_logging():
    """Configure structlog to emit JSON to stdout."""
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,   # pulls in request_id, user_id
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name=None):
    return structlog.get_logger(name)