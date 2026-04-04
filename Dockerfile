FROM python:3.11-slim

LABEL maintainer="Dhavala Kartikeya Somayaji"
LABEL description="FALCON: Frame-Aware Localization of Crime via Observation-budget Networks"
LABEL org.opencontainers.image.title="FALCON"
LABEL org.opencontainers.image.version="1.0.0"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# Generate scenarios.json at build time (no real video files needed)
RUN python preprocess.py

# HF Spaces uses port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import requests; r = requests.get('http://localhost:7860/health'); exit(0 if r.status_code == 200 else 1)"

# Start FastAPI server on port 7860
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
