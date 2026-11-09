#!/bin/sh
# Approved full comparison; never run test evaluation or promote automatically.
set -eu
cd "$(dirname "$0")/.."
export MLFLOW_DISABLE_AGENT_HINT=1
trap '.venv/bin/python scripts/summarize_phase1.py || true' EXIT
# Use the existing, verified dataset version without rebuilding upstream data.
.venv/bin/dvc repro --single-item experiments/yolo11s/dvc.yaml:train_candidate
.venv/bin/dvc repro --single-item experiments/yolo11s/dvc.yaml:compare_candidate
.venv/bin/python scripts/summarize_phase1.py
git add experiments/yolo11s/dvc.lock reports/phase1
git commit -m "Record full YOLO11s candidate and validation comparison"
