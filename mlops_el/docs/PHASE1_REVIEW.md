# Phase 1 review and demonstration

The technical review package is ready. Faculty dataset approval is pending. This guide reads existing evidence and does not launch training.

## Review sequence

1. Present the project problem and college Phase 1 scope.
2. Show `reports/phase1/PHASE1_FINAL_REPORT.md` and `PHASE1_CHECKLIST.md`.
3. Explain the 10,000 image rows, 60,826 annotation rows, ten classes and 7,000/1,500/1,500 split with `docs/DATA_SCHEMA.md`.
4. Open the EDA figures and discuss imbalance and small objects.
5. Show Git history and the raw/prepared/model DVC pointers.
6. Open MLflow history, distinguish engineering checks from model-quality experiments, and show the saved best model and final test report.
7. Show the PR/confusion figures and explain the cup weakness and 70% target shortfall.
8. Explain cloud resume, verified backups, and the later untrained labels-v4 corrections.
9. Close with limitations and the college's Phase 2 deliverables.

## Safe terminal commands

```sh
git log -8 --oneline
.venv/bin/dvc status
.venv/bin/dvc metrics show
.venv/bin/python -m pytest tests -q
```

`dvc status` may include changes if inputs or DVC rules change after the recorded review. Inspect a difference rather than overwriting a historical version or blindly committing it. Finder `.DS_Store` files are excluded. The normalization record preserves previous dataset/dependency references; it does not represent retraining.

### MLflow: local baseline, candidate, test and quality audit

```sh
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000`. This store contains local runs, including YOLO11n, YOLO11s, YOLO11m test evaluation and the v4 data-quality audit.

### MLflow: completed cloud training

```sh
.venv/bin/mlflow ui --backend-store-uri sqlite:///$PWD/.local/phase1-cloud-tracking/mlflow.db --host 127.0.0.1 --port 5001
```

Open `http://127.0.0.1:5001`. Run `85714466948d4e658ebb610a475dcf57` remains the original completed cloud run. A separate local view maps copied artifact paths to this Mac. `reports/phase1/cloud_tracking_local_view.json` records the change, and the original downloaded database is unchanged. The archived state lives in `.local/kaggle-completed-epoch86/train` and is also preserved in the verified private Kaggle backup. Do not mix the two databases or claim every run is in the root store.

## Reproduce evaluation only when deliberately needed

The existing report and figures already provide the final test evidence. Do not repeatedly evaluate the test set to select model settings. If an independent reviewer needs to reproduce that frozen evaluation, use a fresh report path:

```sh
.venv/bin/python scripts/evaluate.py \
  --weights models/plitterstreet-yolo11m-candidate.pt \
  --config configs/yolo11m_cloud_proposed.yaml \
  --split test --device mps --batch 1 \
  --report reports/phase1/evaluation_yolo11m_test_reproduction.json \
  --source-run-id 85714466948d4e658ebb610a475dcf57
```

Weights SHA256: `a9a39b52f2de03927ce5f387a6feb52295468e50262e436d75ee6a0a95ed1b8e`.

The filename `yolo11m_cloud_proposed.yaml` is historical. It records the settings authorized for the completed labels-v3 experiment; its old proposal status is not the current project status. The current completion record is `reports/phase1/yolo11m_completion_summary.json`.

## Dataset versions

- Original prepared data: baseline and YOLO11s experiments, DVC pipeline output.
- Orientation-v2: four training images' box geometry aligned with decoding.
- Labels-v3: inherited fixes plus two annotation refinements. The completed YOLO11m used this version.
- Labels-v4: 30 further class corrections across 28 training images. This version has no trained model and no new accuracy result.

Use DVC checkout only when intentionally restoring a recorded version. Keep a clean commit or backup before switching. A configured off-device DVC remote is absent; local DVC cache availability is therefore necessary for local checkout. Private cloud train/validation and checkpoint backups are a separate mechanism.

## Commands that can do expensive work

Do not use `dvc repro`, `scripts/run_phase1.sh`, `scripts/run_yolo11s_candidate.sh`, or cloud resume cells as presentation shortcuts. They can rebuild data, launch training or alter operational state. There is no running training task to resume for the completed experiment.

## Pending external approval

Ask the faculty to approve the selected pLitterStreet dataset and ten-class scope using `docs/DATASET_APPROVAL.md`. Do not mark this approval as complete until it is actually received. No new training or dataset replacement is planned for the Phase 1 submission.

## Experiment metadata snapshots

`reports/phase1/experiment_history.dvc` versions consistent local/cloud SQLite metadata snapshots. The live stores remain unchanged. The snapshot index lists run identities and artifact locations; artifact files themselves remain in the original local stores and verified cloud archive. Do not present these metadata snapshots as a complete off-device backup.
