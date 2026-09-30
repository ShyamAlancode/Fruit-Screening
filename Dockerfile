# CFRD — Docker Container
# Reproducible inference environment: Python 3.11, sklearn 1.5.0,
# FastAPI service on port 8000.
#
# Build:  docker build -t cfrd-api .
# Run:    docker run -p 8000:8000 cfrd-api
# Test:   curl http://localhost:8000/health
# Docs:   http://localhost:8000/docs

# ── Stage 1: builder (installs deps) ──────────────────────────────────────
FROM python:3.11.8-slim AS builder

WORKDIR /build

# Install system dependencies for spectral/numpy/scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (layer caching)
COPY requirements_lock.txt .
COPY requirements.txt .

# Install pinned environment (sklearn 1.5.0 — matches model artifacts)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_lock.txt && \
    pip install --no-cache-dir fastapi>=0.110.0 "uvicorn[standard]>=0.29.0" pydantic>=2.0.0

# ── Stage 2: runtime ──────────────────────────────────────────────────────
FROM python:3.11.8-slim AS runtime

LABEL maintainer="CFRD Project"
LABEL description="CFRD Apple Quality Screening API — pre-hardware research prototype"
LABEL version="0.1.0-pre-hardware"
LABEL sklearn_version="1.5.0"

# Non-root user for security
RUN useradd --create-home --shell /bin/bash cfrd
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project source
COPY --chown=cfrd:cfrd src/ ./src/
COPY --chown=cfrd:cfrd models/ ./models/
COPY --chown=cfrd:cfrd data/pesticide_screening/ ./data/pesticide_screening/

# Switch to non-root
USER cfrd

# Python path
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV CFRD_ENV=docker

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Expose port
EXPOSE 8000

# Run FastAPI via uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]
