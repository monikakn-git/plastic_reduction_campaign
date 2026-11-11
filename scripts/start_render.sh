#!/bin/sh
set -eu

model_path="${WASTE_MODEL_PATH:-/app/artifacts/checkpoints/yolo11n_best.pt}"
if [ ! -s "$model_path" ]; then
    if [ -z "${WASTE_MODEL_URL:-}" ]; then
        echo "Model checkpoint is missing at $model_path; set WASTE_MODEL_URL to a downloadable YOLO11n checkpoint." >&2
        exit 1
    fi
    mkdir -p "$(dirname "$model_path")"
    echo "Downloading YOLO11n checkpoint for Render..."
    curl --fail --location --silent --show-error "$WASTE_MODEL_URL" --output "$model_path.tmp"
    test -s "$model_path.tmp"
    if [ -n "${WASTE_MODEL_SHA256:-}" ]; then
        actual_sha256="$(sha256sum "$model_path.tmp" | cut -d ' ' -f 1)"
        if [ "$actual_sha256" != "$WASTE_MODEL_SHA256" ]; then
            echo "Checkpoint SHA-256 mismatch: expected $WASTE_MODEL_SHA256, got $actual_sha256" >&2
            rm -f "$model_path.tmp"
            exit 1
        fi
    fi
    mv "$model_path.tmp" "$model_path"
fi

exec uvicorn src.api.app:app --host 0.0.0.0 --port "${PORT:-10000}"
