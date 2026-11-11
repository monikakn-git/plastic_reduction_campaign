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
    mv "$model_path.tmp" "$model_path"
fi

exec uvicorn src.api.app:app --host 0.0.0.0 --port "${PORT:-10000}"
