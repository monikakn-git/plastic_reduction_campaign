#!/bin/sh
# Complete the college's Phase 1 workflow for the configured 10,000-image dataset.
set -eu
cd "$(dirname "$0")/.."
mkdir -p .local reports/phase1
export MLFLOW_DISABLE_AGENT_HINT=1

# Optional: wait for a downloader already running in this project. A second
# acquisition pass verifies cached originals and retries remaining files.
if [ "${1:-}" = "--after-download-pid" ]; then
    download_pid="$2"
    echo "Waiting for existing acquisition process $download_pid"
    while kill -0 "$download_pid" 2>/dev/null; do sleep 10; done
fi

trap '.venv/bin/python scripts/summarize_phase1.py || true' EXIT
.venv/bin/python scripts/acquire_plitter.py --target-images 10000 --workers 8
.venv/bin/dvc add data/raw/plitterstreet
.venv/bin/dvc checkout --relink data/raw/plitterstreet.dvc
.venv/bin/dvc repro prepare explore
.venv/bin/python -m pytest -q tests > reports/phase1/data_checks.txt
.venv/bin/python scripts/summarize_phase1.py
git add data/raw/plitterstreet.dvc data/raw/.gitignore data/prepared/.gitignore dvc.lock reports/phase1
git commit -m "Record verified 10000-image pLitterStreet data and exploration" || test -z "$(git diff --cached --name-only)"

.venv/bin/python scripts/train.py --profile benchmark
.venv/bin/dvc status prepare explore
git add reports/phase1/benchmark_metrics.json
git commit -m "Record pipeline benchmark for the 10000-image dataset" || test -z "$(git diff --cached --name-only)"

.venv/bin/dvc repro train
.venv/bin/python scripts/evaluate.py --split val
.venv/bin/python scripts/evaluate.py --split test
.venv/bin/python scripts/summarize_phase1.py
git add dvc.lock reports/phase1
git commit -m "Record 10000-image baseline and held-out Phase 1 evaluation" || test -z "$(git diff --cached --name-only)"
