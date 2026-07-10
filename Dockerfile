# Stage 1: Build dependencies
FROM python:3.10-slim AS builder

WORKDIR /app

# Install system dependencies needed for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime image (smaller)
FROM python:3.10-slim

WORKDIR /app

# Install locale support for ONNX Runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    locales \
    && sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen \
    && locale-gen \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY app.py .
COPY database.py .
COPY gemini_helper.py .
COPY url_features.py .
COPY tasks.py .
COPY url_features_cached.py .
COPY jwt_auth.py .
COPY logging_config.py .
COPY tracing_config.py .
COPY email_helper.py .
COPY templates/ templates/

# Copy model files (excluding training data and credentials)
COPY models/roberta_phishing_model.pth models/
COPY models/roberta_tokenizer/ models/roberta_tokenizer/
COPY models/selected_features.txt models/
COPY models/selected_url_features.txt models/
COPY models/model.onnx models/
COPY models/roberta_base_local/ models/roberta_base_local/
# Create non-root user and give it ownership of /app
RUN useradd --create-home appuser && chown -R appuser:appuser /app

# Create data directory for SQLite (writable by appuser)
RUN mkdir -p /app/data && chown appuser:appuser /app/data

USER appuser

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/')" || exit 1

# Run with Gunicorn (production server) instead of Flask dev server
CMD ["gunicorn", "--bind", "0.0.0.0:8001", "--workers", "2", "--timeout", "120", "app:app"]