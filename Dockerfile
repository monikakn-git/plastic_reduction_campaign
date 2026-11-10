# ============================================================
# EcoPulse AI - Waste Detection Production Container
# ============================================================

FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr and writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    MLFLOW_TRACKING_URI="sqlite:////app/mlflow.db" \
    ECOPULSE_MAX_UPLOAD_BYTES=10485760

# Install OS libraries required for OpenCV, image decoding, and healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the CPU-only PyTorch build for this CPU inference service, then its
# runtime dependencies. Training, DVC, and pytest dependencies stay out of the
# serving image.
COPY requirements-api.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements-api.txt

# Copy application source code, configs, checkpoints, and reports
COPY src/ ./src/
COPY configs/ ./configs/
COPY artifacts/checkpoints/ ./artifacts/checkpoints/
COPY reports/ ./reports/

# Create a non-privileged system user for container execution
RUN useradd -m -u 1000 ecopulse && \
    chown -R ecopulse:ecopulse /app

USER ecopulse

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the EcoPulse AI FastAPI service
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

