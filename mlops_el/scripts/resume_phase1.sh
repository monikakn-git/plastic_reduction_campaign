#!/bin/sh
# Recover the existing baseline, then complete its Phase 1 evaluations.
set -eu
cd "$(dirname "$0")/.."
export MLFLOW_DISABLE_AGENT_HINT=1
trap '.venv/bin/python scripts/summarize_phase1.py || true' EXIT
.venv/bin/python scripts/train.py --profile baseline \
    --resume runs/baseline-20261104-164642/weights/last.pt \
    --mlflow-run-id 2cd6e612cd25457bb4547df0d42511b8
# Record recovered outputs without restarting a fresh 50-epoch DVC run.
.venv/bin/dvc commit train --force
.venv/bin/python scripts/evaluate.py --split val
.venv/bin/python scripts/evaluate.py --split test
.venv/bin/python scripts/summarize_phase1.py
git add dvc.lock reports/phase1
git commit -m "Record resumed baseline and initial held-out evaluations"
