# EcoPulse AI Project Audit Handover

**Prepared:** 2026-11-08  
**Repository:** `monikakn-git/plastic_reduction_campaign`  
**Checkout:** `master` at `3c0c6ee` (`origin/master`)  
**Purpose:** Review the current project without transferring datasets, model weights, environments, caches, databases, or large generated artifacts.

> This handover records the checkout as inspected. `NOT FOUND` means that the requested evidence was not available or could not be verified; it does not imply the resource never existed. No credentials or secrets are included.

## 1. Final project directory tree

Large or generated contents are intentionally omitted: dataset image/label files, model checkpoints, feature CSV data, MLflow database/run stores, YOLO runs, virtual environments, caches, and generated plots. The tree shows the source/configuration and reviewable report files.

```text
.
|-- .dockerignore
|-- .dvc/                                  [local config not reproduced]
|-- .dvcignore
|-- .gitignore
|-- .github/
|   `-- workflows/
|       `-- ci.yml
|-- Dockerfile
|-- README.md
|-- annotations.zip.dvc
|-- docker-compose.yml
|-- dvc.yaml
|-- mlflowruncmd.txt
|-- PROJECT_AUDIT_HANDOVER.md
|-- requirements.txt
|-- restore_dataset_images.py
|-- test_existing_model.py
|-- artifacts/
|   |-- checkpoints/                         [weights omitted]
|   |-- datasets/
|   |   |-- .gitignore
|   |   |-- processed.dvc
|   |   `-- processed/
|   |       |-- data.yaml
|   |       |-- images/                      [dataset omitted]
|   |       `-- labels/                      [dataset omitted]
|   |-- feature_engineering/
|   |   `-- SHA256.txt
|   `-- features/
|       |-- .gitignore
|       `-- engineered_features.csv.dvc
|-- configs/
|   |-- data.yaml
|   `-- train.yaml
|-- data/
|   `-- processed/
|       |-- annotations/
|       |   |-- annotations_test_cleaned.json
|       |   `-- annotations_train_cleaned.json
|       `-- yolo_labels/                     [dataset labels omitted]
|-- pipelines/
|   `-- training_pipeline.py
|-- reports/
|   |-- data_validation_report.json
|   |-- dataset_exploration.json
|   |-- image_mapping_report.json
|   |-- model_evaluation_report.json
|   |-- model_evaluation_report.md
|   |-- model_risk_quality_report.json        [untracked]
|   |-- model_risk_quality_report.md          [untracked]
|   |-- model_test_report.json                [untracked]
|   |-- yolo11n_best_test_evaluation.json
|   |-- yolo11n_best_test_evaluation.md
|   |-- yolov8n_best_test_evaluation.json
|   |-- yolov8n_best_test_evaluation.md
|   |-- feature_engineering/
|   |   `-- README.md
|   |-- figures/                             [generated plots omitted]
|   `-- mlflow_artifacts/                    [generated report artifacts]
|-- scripts/
|   `-- log_model_comparison.py
|-- src/
|   |-- api/app.py
|   |-- campaigns/
|   |   |-- hotspot.py
|   |   `-- planner.py
|   |-- data/
|   |   |-- explore.py
|   |   |-- image_loader.py
|   |   |-- prepare_yolo_dataset.py
|   |   |-- validate.py
|   |   |-- validate_yolo_dataset.py
|   |   `-- verify_image_mapping.py
|   |-- features/feature_engineering.py
|   |-- models/
|   |   |-- evaluate.py
|   |   |-- predict.py
|   |   |-- risk_quality_checks.py
|   |   |-- test_models.py
|   |   `-- train.py
|   |-- tracking/mlflow_utils.py
|   `-- visualization/plot_dataset.py
`-- tests/
    |-- test_api.py
    |-- test_dataset.py
    |-- test_models.py
    `-- test_tracking.py
```

`mlflow.db`, `mlruns/`, `runs/`, and actual data/weight contents exist in the working tree but are intentionally not copied into this handover.

## 2. Python source files and purpose

| File | Purpose |
|---|---|
| `restore_dataset_images.py` | Restores processed dataset images from an archive by matching image names to label files. |
| `test_existing_model.py` | Ad hoc smoke test using an existing Google Drive checkpoint and archive; depends on external `G:` paths. |
| `pipelines/training_pipeline.py` | Loads and validates training/data config and dataset paths; `main()` validates only and does not start training. Its `train()` function runs YOLO training with MLflow tracking. |
| `scripts/log_model_comparison.py` | Reads existing external YOLO run CSV/config/checkpoint files and logs the historical comparison to MLflow. |
| `src/api/app.py` | FastAPI health and image-prediction endpoints; prediction events are optionally logged to MLflow. |
| `src/campaigns/hotspot.py` | Computes/geospatially groups waste detection hotspots. |
| `src/campaigns/planner.py` | Campaign-planning functionality using detections and tracking utilities. |
| `src/data/explore.py` | Dataset exploration and summary generation. |
| `src/data/image_loader.py` | Loads/associates source images and image metadata for the dataset workflow. |
| `src/data/prepare_yolo_dataset.py` | Converts/prepares annotations and images in YOLO dataset format. |
| `src/data/validate.py` | General data/annotation validation. |
| `src/data/validate_yolo_dataset.py` | Checks YOLO dataset YAML, image/label splits, and dataset consistency. |
| `src/data/verify_image_mapping.py` | Verifies correspondence between image names, annotations, and labels. |
| `src/features/feature_engineering.py` | Extracts tabular image/object features and writes feature-engineering outputs. |
| `src/models/evaluate.py` | Validates an existing checkpoint, calculates test/validation metrics and inference latency, writes reports, and optionally logs evaluation artifacts to MLflow. |
| `src/models/predict.py` | Loads/caches a checkpoint and performs single-image prediction; also provides a CLI. |
| `src/models/risk_quality_checks.py` | Runs model risk/quality checks and robustness perturbation evaluation. |
| `src/models/test_models.py` | Standalone checkpoint/inference/edge-case test runner. |
| `src/models/train.py` | Training entry point for configurable YOLO training with MLflow logging and an optional dry-run. |
| `src/tracking/mlflow_utils.py` | Resolves the tracking URI and logs prediction metadata/metrics without uploading image contents. |
| `src/visualization/plot_dataset.py` | Generates dataset exploration plots. |
| `tests/test_api.py` | Unit tests for API home and health responses. |
| `tests/test_dataset.py` | Unit tests for dataset YAML, class names, and split directories. |
| `tests/test_models.py` | Unit tests for checkpoint availability/loading and sample-image inference. |
| `tests/test_tracking.py` | Unit tests for MLflow URI resolution and tracking setup. |

## 3. Configuration files and purpose

| File | Purpose / observed configuration |
|---|---|
| `requirements.txt` | Declares minimum Python dependency versions; the exact file is reproduced below. |
| `configs/train.yaml` | Selects `yolov8n.yaml`; smoke-test training settings (1 epoch, 320 image size, batch 4, CPU, workers 0, patience 1, seed 42); output path/name and MLflow experiment. |
| `configs/data.yaml` | Project dataset paths, 10 canonical classes, external archive references, and DVC dataset reference. |
| `artifacts/datasets/processed/data.yaml` | Ultralytics dataset root, train/val/test image directories, and the 10-class name mapping. |
| `dvc.yaml` | DVC stages for dataset validation and YOLO11n/YOLOv8n test evaluation. |
| `annotations.zip.dvc`, `artifacts/datasets/processed.dvc`, `artifacts/features/engineered_features.csv.dvc` | DVC metadata/pointers for the source archive, processed dataset, and generated feature file. |
| `.dvc/config` | DVC repository settings/remote configuration; contents are not reproduced because such files may contain sensitive remote details. |
| `.gitignore` | Excludes local environments, dataset archives/data, checkpoints/runs, MLflow data, and caches; contains exceptions for selected DVC metadata. |
| `.dockerignore` | Excludes VCS/DVC metadata, local environments, datasets, generated runs/tracking data, IDE and OS files. |
| `.dvcignore` | DVC ignore file; currently contains only the generated explanatory comments. |
| `.github/workflows/ci.yml` | GitHub Actions workflow configuration; currently untracked in this checkout. |
| `Dockerfile` | Builds the Python 3.11 slim FastAPI service image and copies application, configs, checkpoints, and reports. |
| `docker-compose.yml` | Defines MLflow and API services, their ports, local persistence mounts, and health checks; currently untracked. |
| `mlflowruncmd.txt` | Local Windows commands recorded for activating `venv` and starting the MLflow UI. |

The `.dvc/config` contents and configured remote are **NOT FOUND** in this report; no credential-bearing DVC configuration is reproduced.

## 4. Files created, modified, and deleted during the prior audit/cleanup

The repository history contains two relevant commits, but their commit messages do not establish which exact changes were part of the described audit. The following is the verifiable commit inventory, not an attribution beyond that evidence:

* `21177e9` (`changes in code`): 12 files changed; 414 insertions and 65 deletions. Modified: `.dvc/config`, `configs/data.yaml`, `configs/train.yaml`, `data/waste10/images/test/ATT22_4-2.jpg`, `pipelines/training_pipeline.py`, `src/data/prepare_yolo_dataset.py`, `src/data/validate_yolo_dataset.py`, `src/features/feature_engineering.py`, and `src/models/evaluate.py`. Added: `data/waste10/labels/train/Rangsit_2021_GX094614_134.txt`, `reports/model_evaluation_report.json`, and `reports/model_evaluation_report.md`.
* `3c0c6ee` (`added some changes`, current `HEAD`): 173 files changed (31 added, 11 modified, 132 deleted; 2,884 insertions and 637 deletions). Added: `.dockerignore`, `restore_dataset_images.py`, `tests/test_api.py`, `tests/test_dataset.py`, `tests/test_models.py`, `tests/test_tracking.py`, the four `reports/yolo{11n,v8n}_best_test_evaluation.{json,md}` reports, evaluation summary/dataset/environment/model/per-class/visualization-manifest files under `reports/mlflow_artifacts/yolo11n_best/test/`, and the YOLO evaluation plots in that same artifact tree. Modified: `Dockerfile`, `README.md`, `configs/data.yaml`, `configs/train.yaml`, `dvc.yaml`, `pipelines/training_pipeline.py`, `src/api/app.py`, `src/data/validate_yolo_dataset.py`, `src/models/evaluate.py`, and `src/tracking/mlflow_utils.py`. Deleted: 66 `.jpg` files, 65 `.txt` label files, and `data/waste10/data.yaml`; the image/label removals are under `data/` and `data/waste10/`.
* Current uncommitted state before this report: no tracked modifications or staged changes; untracked `.github/`, `docker-compose.yml`, `reports/model_risk_quality_report.json`, `reports/model_risk_quality_report.md`, and `reports/model_test_report.json`. Their original creation/modification times and authorship are **NOT FOUND**. This report is the only file created by this handover task.

### Complete committed path lists

Paths below are listed verbatim from Git history (A = added, M = modified, D = deleted).

#### `21177e9`

**A**
- `data/waste10/labels/train/Rangsit_2021_GX094614_134.txt`
- `reports/model_evaluation_report.json`
- `reports/model_evaluation_report.md`

**M**
- `.dvc/config`
- `configs/data.yaml`
- `configs/train.yaml`
- `data/waste10/images/test/ATT22_4-2.jpg`
- `pipelines/training_pipeline.py`
- `src/data/prepare_yolo_dataset.py`
- `src/data/validate_yolo_dataset.py`
- `src/features/feature_engineering.py`
- `src/models/evaluate.py`

#### `3c0c6ee`

**A**
- `.dockerignore`
- `reports/mlflow_artifacts/yolo11n_best/test/dataset/data.yaml`
- `reports/mlflow_artifacts/yolo11n_best/test/dataset/dataset_summary.json`
- `reports/mlflow_artifacts/yolo11n_best/test/environment/requirements.txt`
- `reports/mlflow_artifacts/yolo11n_best/test/environment/run_environment.json`
- `reports/mlflow_artifacts/yolo11n_best/test/evaluation/evaluation_summary.json`
- `reports/mlflow_artifacts/yolo11n_best/test/evaluation/per_class_metrics.csv`
- `reports/mlflow_artifacts/yolo11n_best/test/evaluation/per_class_metrics.json`
- `reports/mlflow_artifacts/yolo11n_best/test/model/model_metadata.json`
- `reports/mlflow_artifacts/yolo11n_best/test/visualizations/BoxF1_curve.png`
- `reports/mlflow_artifacts/yolo11n_best/test/visualizations/BoxPR_curve.png`
- `reports/mlflow_artifacts/yolo11n_best/test/visualizations/BoxP_curve.png`
- `reports/mlflow_artifacts/yolo11n_best/test/visualizations/BoxR_curve.png`
- `reports/mlflow_artifacts/yolo11n_best/test/visualizations/confusion_matrix.png`
- `reports/mlflow_artifacts/yolo11n_best/test/visualizations/confusion_matrix_normalized.png`
- `reports/mlflow_artifacts/yolo11n_best/test/visualizations/val_batch0_labels.jpg`
- `reports/mlflow_artifacts/yolo11n_best/test/visualizations/val_batch0_pred.jpg`
- `reports/mlflow_artifacts/yolo11n_best/test/visualizations/val_batch1_labels.jpg`
- `reports/mlflow_artifacts/yolo11n_best/test/visualizations/val_batch1_pred.jpg`
- `reports/mlflow_artifacts/yolo11n_best/test/visualizations/val_batch2_labels.jpg`
- `reports/mlflow_artifacts/yolo11n_best/test/visualizations/val_batch2_pred.jpg`
- `reports/mlflow_artifacts/yolo11n_best/test/visualizations/visualization_manifest.json`
- `reports/yolo11n_best_test_evaluation.json`
- `reports/yolo11n_best_test_evaluation.md`
- `reports/yolov8n_best_test_evaluation.json`
- `reports/yolov8n_best_test_evaluation.md`
- `restore_dataset_images.py`
- `tests/test_api.py`
- `tests/test_dataset.py`
- `tests/test_models.py`
- `tests/test_tracking.py`

**M**
- `Dockerfile`
- `README.md`
- `configs/data.yaml`
- `configs/train.yaml`
- `dvc.yaml`
- `pipelines/training_pipeline.py`
- `requirements.txt`
- `src/api/app.py`
- `src/data/validate_yolo_dataset.py`
- `src/models/evaluate.py`
- `src/tracking/mlflow_utils.py`

**D**
- `data/Chiangrai_2021_GH020107_308.jpg`
- `data/waste10/data.yaml`
- `data/waste10/images/test/ATT22_4-2.jpg`
- `data/waste10/images/test/Rangsit_2021_GX014614_36.jpg`
- `data/waste10/images/test/Rangsit_2021_GX044613_197.jpg`
- `data/waste10/images/test/Rangsit_2021_GX064614_169.jpg`
- `data/waste10/images/test/Rangsit_2021_GX074613_235.jpg`
- `data/waste10/images/test/Rangsit_2021_GX114614_102.jpg`
- `data/waste10/images/test/Rangsit_2021_GX144613_223.jpg`
- `data/waste10/images/test/Ubon_2022_GH010244_150.jpg`
- `data/waste10/images/train/Chiangrai_2021_GH010101_374.jpg`
- `data/waste10/images/train/Chiangrai_2021_GH010103_147.jpg`
- `data/waste10/images/train/Chiangrai_2021_GH010119_46.jpg`
- `data/waste10/images/train/Chiangrai_2021_GH020102_284.jpg`
- `data/waste10/images/train/Chiangrai_2021_GH020107_308.jpg`
- `data/waste10/images/train/Chiangrai_2021_GH020126_75.jpg`
- `data/waste10/images/train/Hanwella_2022_GX010105_156.jpg`
- `data/waste10/images/train/Hanwella_2022_GX010113_16.jpg`
- `data/waste10/images/train/Mawanella_2022_GX010144_148.jpg`
- `data/waste10/images/train/Rangsit_2021_GX014613_28.jpg`
- `data/waste10/images/train/Rangsit_2021_GX014613_47.jpg`
- `data/waste10/images/train/Rangsit_2021_GX024613_266.jpg`
- `data/waste10/images/train/Rangsit_2021_GX024614_49.jpg`
- `data/waste10/images/train/Rangsit_2021_GX054614_213.jpg`
- `data/waste10/images/train/Rangsit_2021_GX074613_168.jpg`
- `data/waste10/images/train/Rangsit_2021_GX084613_121.jpg`
- `data/waste10/images/train/Rangsit_2021_GX084613_194.jpg`
- `data/waste10/images/train/Rangsit_2021_GX084614_107.jpg`
- `data/waste10/images/train/Rangsit_2021_GX084614_310.jpg`
- `data/waste10/images/train/Rangsit_2021_GX094614_134.jpg`
- `data/waste10/images/train/Rangsit_2021_GX104613_261.jpg`
- `data/waste10/images/train/Rangsit_2021_GX104614_74.jpg`
- `data/waste10/images/train/Rangsit_2021_GX114613_191.jpg`
- `data/waste10/images/train/Rangsit_2021_GX114613_219.jpg`
- `data/waste10/images/train/Rangsit_2021_GX114613_96.jpg`
- `data/waste10/images/train/Rangsit_2021_GX114614_174.jpg`
- `data/waste10/images/train/Rangsit_2021_GX114614_18.jpg`
- `data/waste10/images/train/Rangsit_2021_GX124613_284.jpg`
- `data/waste10/images/train/Rangsit_2021_GX144613_285.jpg`
- `data/waste10/images/train/Rangsit_2021_GX144613_36.jpg`
- `data/waste10/images/train/Rangsit_2021_GX174613_189.jpg`
- `data/waste10/images/train/Rangsit_2021_GX194613_190.jpg`
- `data/waste10/images/train/Rangsit_2021_GX204613_89.jpg`
- `data/waste10/images/train/Rangsit_2021_GX214613_90.jpg`
- `data/waste10/images/train/Rangsit_2021_GX224613_17.jpg`
- `data/waste10/images/train/Rangsit_2021_GX234613_206.jpg`
- `data/waste10/images/train/Rangsit_2021_GX234613_74.jpg`
- `data/waste10/images/train/Rangsit_2021_GX244613_189.jpg`
- `data/waste10/images/train/Taladthai_2021_GX014429_163.jpg`
- `data/waste10/images/train/Taladthai_2021_GX014430_130.jpg`
- `data/waste10/images/train/Taladthai_2021_GX024429_028.jpg`
- `data/waste10/images/train/Taladthai_2021_GX054430_092.jpg`
- `data/waste10/images/train/Taladthai_2021_GX064430_084.jpg`
- `data/waste10/images/train/Taladthai_2021_GX064430_268.jpg`
- `data/waste10/images/train/Ubon_2022_GH010244_170.jpg`
- `data/waste10/images/val/Chiangrai_2021_GH010105_65.jpg`
- `data/waste10/images/val/Chiangrai_2021_GH020102_252.jpg`
- `data/waste10/images/val/Hanwella_2022_GX010105_583.jpg`
- `data/waste10/images/val/Mawanella_2022_GX010146_97.jpg`
- `data/waste10/images/val/Rangsit_2021_GX014614_113.jpg`
- `data/waste10/images/val/Rangsit_2021_GX024614_205.jpg`
- `data/waste10/images/val/Rangsit_2021_GX064613_209.jpg`
- `data/waste10/images/val/Rangsit_2021_GX134613_116.jpg`
- `data/waste10/images/val/Rangsit_2021_GX134613_315.jpg`
- `data/waste10/images/val/Rangsit_2021_GX234613_315.jpg`
- `data/waste10/images/val/Taladthai_2021_GX034430_088.jpg`
- `data/waste10/images/val/Taladthai_2021_GX064430_183.jpg`
- `data/waste10/labels/test/ATT22_4-2.txt`
- `data/waste10/labels/test/Rangsit_2021_GX014614_36.txt`
- `data/waste10/labels/test/Rangsit_2021_GX044613_197.txt`
- `data/waste10/labels/test/Rangsit_2021_GX064614_169.txt`
- `data/waste10/labels/test/Rangsit_2021_GX074613_235.txt`
- `data/waste10/labels/test/Rangsit_2021_GX114614_102.txt`
- `data/waste10/labels/test/Rangsit_2021_GX144613_223.txt`
- `data/waste10/labels/test/Ubon_2022_GH010244_150.txt`
- `data/waste10/labels/train/Chiangrai_2021_GH010101_374.txt`
- `data/waste10/labels/train/Chiangrai_2021_GH010103_147.txt`
- `data/waste10/labels/train/Chiangrai_2021_GH010119_46.txt`
- `data/waste10/labels/train/Chiangrai_2021_GH020102_284.txt`
- `data/waste10/labels/train/Chiangrai_2021_GH020107_308.txt`
- `data/waste10/labels/train/Chiangrai_2021_GH020126_75.txt`
- `data/waste10/labels/train/Hanwella_2022_GX010105_156.txt`
- `data/waste10/labels/train/Hanwella_2022_GX010113_16.txt`
- `data/waste10/labels/train/Mawanella_2022_GX010144_148.txt`
- `data/waste10/labels/train/Rangsit_2021_GX014613_28.txt`
- `data/waste10/labels/train/Rangsit_2021_GX014613_47.txt`
- `data/waste10/labels/train/Rangsit_2021_GX024613_266.txt`
- `data/waste10/labels/train/Rangsit_2021_GX024614_49.txt`
- `data/waste10/labels/train/Rangsit_2021_GX054614_213.txt`
- `data/waste10/labels/train/Rangsit_2021_GX074613_168.txt`
- `data/waste10/labels/train/Rangsit_2021_GX084613_121.txt`
- `data/waste10/labels/train/Rangsit_2021_GX084613_194.txt`
- `data/waste10/labels/train/Rangsit_2021_GX084614_107.txt`
- `data/waste10/labels/train/Rangsit_2021_GX084614_310.txt`
- `data/waste10/labels/train/Rangsit_2021_GX094614_134.txt`
- `data/waste10/labels/train/Rangsit_2021_GX104613_261.txt`
- `data/waste10/labels/train/Rangsit_2021_GX104614_74.txt`
- `data/waste10/labels/train/Rangsit_2021_GX114613_191.txt`
- `data/waste10/labels/train/Rangsit_2021_GX114613_219.txt`
- `data/waste10/labels/train/Rangsit_2021_GX114613_96.txt`
- `data/waste10/labels/train/Rangsit_2021_GX114614_174.txt`
- `data/waste10/labels/train/Rangsit_2021_GX114614_18.txt`
- `data/waste10/labels/train/Rangsit_2021_GX124613_284.txt`
- `data/waste10/labels/train/Rangsit_2021_GX144613_285.txt`
- `data/waste10/labels/train/Rangsit_2021_GX144613_36.txt`
- `data/waste10/labels/train/Rangsit_2021_GX174613_189.txt`
- `data/waste10/labels/train/Rangsit_2021_GX194613_190.txt`
- `data/waste10/labels/train/Rangsit_2021_GX204613_89.txt`
- `data/waste10/labels/train/Rangsit_2021_GX214613_90.txt`
- `data/waste10/labels/train/Rangsit_2021_GX224613_17.txt`
- `data/waste10/labels/train/Rangsit_2021_GX234613_206.txt`
- `data/waste10/labels/train/Rangsit_2021_GX234613_74.txt`
- `data/waste10/labels/train/Rangsit_2021_GX244613_189.txt`
- `data/waste10/labels/train/Taladthai_2021_GX014429_163.txt`
- `data/waste10/labels/train/Taladthai_2021_GX014430_130.txt`
- `data/waste10/labels/train/Taladthai_2021_GX024429_028.txt`
- `data/waste10/labels/train/Taladthai_2021_GX054430_092.txt`
- `data/waste10/labels/train/Taladthai_2021_GX064430_084.txt`
- `data/waste10/labels/train/Taladthai_2021_GX064430_268.txt`
- `data/waste10/labels/train/Ubon_2022_GH010244_170.txt`
- `data/waste10/labels/val/Chiangrai_2021_GH010105_65.txt`
- `data/waste10/labels/val/Chiangrai_2021_GH020102_252.txt`
- `data/waste10/labels/val/Hanwella_2022_GX010105_583.txt`
- `data/waste10/labels/val/Mawanella_2022_GX010146_97.txt`
- `data/waste10/labels/val/Rangsit_2021_GX014614_113.txt`
- `data/waste10/labels/val/Rangsit_2021_GX024614_205.txt`
- `data/waste10/labels/val/Rangsit_2021_GX064613_209.txt`
- `data/waste10/labels/val/Rangsit_2021_GX134613_116.txt`
- `data/waste10/labels/val/Rangsit_2021_GX134613_315.txt`
- `data/waste10/labels/val/Rangsit_2021_GX234613_315.txt`
- `data/waste10/labels/val/Taladthai_2021_GX034430_088.txt`
- `data/waste10/labels/val/Taladthai_2021_GX064430_183.txt`

## 5. Dataset paths, image counts, classes, and data.yaml

* Processed dataset: `artifacts/datasets/processed/`.
* Current image and label file counts observed:

| Split | Images | Label files |
|---|---:|---:|
| Train | 9,798 | 9,798 |
| Validation | 1,960 | 1,960 |
| Test | 1,306 | 1,306 |
| **Total** | **13,064** | **13,064** |

* Canonical class mapping: `0 litter`, `1 pile`, `2 face_mask`, `3 trash_bin`, `4 plastic_bag`, `5 bottle`, `6 cup`, `7 rope`, `8 sachet`, `9 straw`.
* Dataset YAML split directories are `images/train`, `images/val`, and `images/test`; its configured root is the absolute Windows path `D:/mlops/artifacts/datasets/processed`.
* DVC metadata for the processed dataset records 13,067 total files (including non-image/label files), with a recorded DVC object size of 6,510,566 bytes.
* `configs/data.yaml` also refers to `G:/My Drive/datasets/Ai_Waste_Campaign/images.zip`; that external archive path is machine-specific. The current processed dataset YAML itself is included verbatim below.

## 6. Available model checkpoints

Both checkpoint files were present during inspection:

| Model | Path | Exact size | Approx. size |
|---|---|---:|---:|
| YOLO11n | `artifacts/checkpoints/yolo11n_best.pt` | 5,506,714 bytes | 5.25 MiB |
| YOLOv8n | `artifacts/checkpoints/yolov8n_best.pt` | 6,505,834 bytes | 6.20 MiB |

These are model weights and are intentionally not copied into the handover.

## 7. Training pipeline status

* Training code is present in `src/models/train.py` and `pipelines/training_pipeline.py`.
* `configs/train.yaml` specifies `yolov8n.yaml`, one epoch, 320-pixel input, batch size 4, CPU, zero data-loader workers, and MLflow experiment `ecopulse-waste-detection`. This is a smoke-test configuration, not evidence of a production training run.
* `pipelines/training_pipeline.py` runs configuration/dataset validation from its CLI `main()`; it explicitly does not start training. Training occurs only when its `train()` function is called.
* `src/models/train.py` is the direct training script and supports `--dry-run`; actual current-checkout training completion is **NOT FOUND**.
* Existing checkpoints are present, but training logs/run provenance linking them to a specific current source revision are **NOT FOUND**.

## 8. Inference pipeline status

* Single-image prediction is implemented by `src/models/predict.py`; the FastAPI service is in `src/api/app.py`.
* Default model selection is YOLO11n; YOLOv8n can be selected by checkpoint or API `model_name`.
* Existing `reports/model_test_report.json` records overall `PASS` for loading and sample prediction on both models and several edge cases. Its exact invocation command is **NOT FOUND**.
* A live API/server check was not run for this handover.

## 9. Evaluation pipeline status

* `src/models/evaluate.py` can evaluate an existing checkpoint on `val` or `test`, benchmark inference latency, emit JSON/Markdown reports and optionally create/log detailed MLflow artifacts.
* `dvc.yaml` defines dataset validation and test-evaluation stages for both models.
* Current test evaluation reports for both checkpoints exist and report `PASSED`; see the metrics below.
* DVC stage/cache/remote state could not be checked because the `dvc` executable is not installed/available in the current shell (`dvc status` returned “The term 'dvc' is not recognized…”).

## 10. Latest available YOLOv8n and YOLO11n evaluation metrics

The available model-specific test reports evaluate 1,306 test images. The separate `reports/model_evaluation_report.md` reports substantially different metrics (mAP50 0.5657) against `D:/mlops/data/waste10/data.yaml`; it is not the same dataset path/split evidence and should not be mixed with these current processed-dataset metrics.

| Metric | YOLOv8n | YOLO11n |
|---|---:|---:|
| Precision | 0.3626 | 0.3690 |
| Recall | 0.3296 | 0.3153 |
| mAP@50 | 0.2761 | 0.2721 |
| mAP@50:95 | 0.1675 | 0.1652 |
| Mean latency | 219.72 ms | 215.07 ms |
| Median latency | 205.04 ms | 194.41 ms |
| P95 latency | 342.77 ms | 331.28 ms |
| Evaluation images | 1,306 | 1,306 |
| Quality gate | PASSED | PASSED |

These values are transcribed from `reports/yolov8n_best_test_evaluation.json` and `reports/yolo11n_best_test_evaluation.json`; they are not rerun by this handover.

## 11. MLflow tracking

* Default local tracking URI in the training/evaluation/utility code: `sqlite:///D:/mlops/mlflow.db` (constructed from the project-root database path; the environment variable `MLFLOW_TRACKING_URI` overrides it).
* Source-configured experiment names: `ecopulse-waste-detection` (training, evaluation, and inference utility default) and `ecopulse-model-comparison` (comparison script).
* Compose configuration points the API service to `http://mlflow:5000`. The MLflow service uses backend DB `/data/mlflow.db` and artifact root `/data/mlruns`; host mounts are `./mlflow.db` and `./mlruns`.
* Local MLflow artifacts are also generated under `reports/mlflow_artifacts/`. The project-root `mlruns/` directory is present.
* Experiment names/locations above are code/configuration values. A direct database query of recorded experiment rows and exact current artifact URIs is **NOT FOUND**. Database files and secrets are not included.

## 12. DVC and Git status

* Git branch: `master`, at `3c0c6ee`, tracking `origin/master`.
* At initial inspection: tracked working tree clean; no staged changes.
* Untracked before this report: `.github/`, `docker-compose.yml`, `reports/model_risk_quality_report.json`, `reports/model_risk_quality_report.md`, `reports/model_test_report.json`.
* The DVC pointer files are present for `annotations.zip`, `artifacts/datasets/processed`, and `artifacts/features/engineered_features.csv`. DVC cache/remote synchronization status is **NOT FOUND** because the DVC CLI is unavailable.
* After this report is created it will also be untracked until the user chooses to add/commit it.

## 13. Requirements and important dependency versions

`requirements.txt` provides minimum versions, not a fully locked environment. The recorded evaluation environment (`reports/mlflow_artifacts/yolo11n_best/test/environment/run_environment.json` and its `requirements.txt`, timestamped 2026-11-06) reports Python 3.14.4, Ultralytics 8.4.193, NumPy 2.5.4, MLflow 3.17.2, DVC 3.67.1, PyTorch 2.15.0, torchvision 0.29.2, FastAPI 0.162.0, OpenCV 5.0.0.93, pandas 3.1.0, scikit-learn 1.9.1, Pydantic 2.14.2, and Uvicorn 0.54.0. These are recorded run versions, not a guarantee of current shell installation.

The Dockerfile uses Python 3.11-slim, whereas the recorded evaluation environment is Python 3.14.4. This runtime difference is worth validating before deployment.

## 14. Tests executed and actual results

* `reports/model_test_report.json` records `overall_status: PASS` for both model checkpoint loading, a sample prediction, and listed edge cases. The report timestamp is `2026-11-08T16:46:33Z`. The exact command/script invocation that produced this report is **NOT FOUND**.
* The README documents `python -m unittest discover tests`, but an execution record/result for that exact command is **NOT FOUND**. This handover did not run the test suite.
* `reports/yolo11n_best_test_evaluation.json` and `reports/yolov8n_best_test_evaluation.json` record `quality_gate.decision: PASSED` on 1,306 test images; these are evaluation results, not unit-test results.
* The attempted `dvc status` command failed because `dvc` was not recognized in the shell. No other test or build command was run for this report.

## 15. Known issues and incomplete components

1. DVC status, cache availability, and remote synchronization have not been verified; the current shell cannot run `dvc`.
2. No exact test-suite execution log is available. Tests exist, but the README command is not evidence that they passed.
3. Training configuration is a one-epoch CPU smoke test; there is no verified current-source training run/provenance record.
4. Paths in `configs/data.yaml`, `artifacts/datasets/processed/data.yaml`, `scripts/log_model_comparison.py`, `test_existing_model.py`, and `mlflowruncmd.txt` include machine-specific `D:` or `G:` locations. External archive/checkpoint references may not resolve for another developer.
5. `reports/model_evaluation_report.md` contains older/incomparable metrics and points to `D:/mlops/data/waste10/data.yaml`; use the model-specific `*_best_test_evaluation` reports for the processed dataset.
6. Compose's MLflow service uses `python:3.11-slim` but its health check invokes `curl`; the image does not install curl in the compose service definition. The API dependency is `service_started`, not `service_healthy`.
7. Dependency constraints are open-ended minimums; no lock file is present in the inspected root.
8. The training pipeline CLI validates configuration only; invoking it does not train a model.

## 16. Commands to run the project

Commands below assume the repository root is the current directory and dependencies/dataset/checkpoints are present. Dataset/image/checkpoint files are not included in this handover.

### Windows PowerShell environment setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Validate the dataset

```powershell
python src/data/validate_yolo_dataset.py
```

### Validate the MLOps pipeline configuration (does not train)

```powershell
python pipelines/training_pipeline.py
```

### Train using `src/models/train.py`

```powershell
python src/models/train.py --config configs/train.yaml --data artifacts/datasets/processed/data.yaml
```

For config/data existence validation only:

```powershell
python src/models/train.py --config configs/train.yaml --data artifacts/datasets/processed/data.yaml --dry-run
```

### Run single-image inference

```powershell
python src/models/predict.py --image "artifacts/datasets/processed/images/test/ATT104_20200301_164523.jpg" --confidence 0.25 --save
python src/models/predict.py --image "artifacts/datasets/processed/images/test/ATT104_20200301_164523.jpg" --checkpoint "artifacts/checkpoints/yolov8n_best.pt" --confidence 0.25
```

### Evaluate existing checkpoints

```powershell
python src/models/evaluate.py --checkpoint artifacts/checkpoints/yolo11n_best.pt --split test
python src/models/evaluate.py --checkpoint artifacts/checkpoints/yolov8n_best.pt --split test
```

Add `--no-mlflow` to skip tracking. Evaluation writes JSON/Markdown reports under `reports/`.

### Run unit tests (documented command; prior result NOT FOUND)

```powershell
python -m unittest discover tests
```

### Start API and MLflow locally

```powershell
uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload
python -m mlflow ui --backend-store-uri "sqlite:///D:/mlops/mlflow.db" --host 127.0.0.1 --port 5000
```

API docs: `http://127.0.0.1:8000/docs`; health endpoint: `http://127.0.0.1:8000/health`. For a relocated checkout, update the absolute SQLite URI or set `MLFLOW_TRACKING_URI`.

### Start services with Docker Compose

```powershell
docker compose up --build -d
docker compose logs -f api
docker compose down
```

## Complete requested file contents

The following sections reproduce the requested files verbatim from the inspected checkout. No weights, datasets, database files, credentials, or large generated artifacts are embedded.

### `requirements.txt`
```text
# EcoPulse AI - Core Dependencies
# Tested on Python 3.14 (Windows / Linux)

# Core Deep Learning & Computer Vision
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.3.0
opencv-python>=4.8.0
pillow>=10.0.0

# Experiment Tracking & MLOps
mlflow>=2.15.0
dvc>=3.50.0

# API & Model Serving
fastapi>=0.110.0
uvicorn>=0.30.0
python-multipart>=0.0.9
pydantic>=2.7.0

# Data Processing & Visualization
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.4.0
matplotlib>=3.8.0
pyyaml>=6.0.0
requests>=2.31.0

```

### `configs/train.yaml`
```yaml
model:
  name: yolov8n.yaml

dataset:
  path: artifacts/datasets/processed/data.yaml

training:
  epochs: 1
  image_size: 320
  batch_size: 4
  device: cpu
  workers: 0
  patience: 1
  seed: 42

output:
  project: runs/detect
  name: ecopulse_waste_smoke_test
  exist_ok: true

tracking:
  enabled: true
  experiment_name: ecopulse-waste-detection
```

### `configs/data.yaml`
```yaml
dataset:
  # Actual local processed dataset
  root: "artifacts/datasets/processed"
  data_yaml: "artifacts/datasets/processed/data.yaml"
  images_dir: "artifacts/datasets/processed/images"
  labels_dir: "artifacts/datasets/processed/labels"

  # External storage references (keep for archival purposes)
  external_images_archive: "G:/My Drive/datasets/Ai_Waste_Campaign/images.zip"
  external_annotations_archive: "annotations.zip"
  annotation_prefix: "annotations/10_class_splits/"

  # DVC-tracked processed dataset
  dvc_processed_dataset: "artifacts/datasets/processed/data.yaml"

  num_classes: 10
  classes:
    0: "litter"
    1: "pile"
    2: "face_mask"
    3: "trash_bin"
    4: "plastic_bag"
    5: "bottle"
    6: "cup"
    7: "rope"
    8: "sachet"
    9: "straw"
```

### `artifacts/datasets/processed/data.yaml`
```yaml
path: D:/mlops/artifacts/datasets/processed

train: images/train
val: images/val
test: images/test

names:
  0: litter
  1: pile
  2: face_mask
  3: trash_bin
  4: plastic_bag
  5: bottle
  6: cup
  7: rope
  8: sachet
  9: straw
```

### `src/models/evaluate.py`
```python

from pathlib import Path
import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any

import mlflow
import numpy as np
import yaml
import ultralytics
from ultralytics import YOLO


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORTS_DIR = PROJECT_ROOT / "reports"
TRACKING_DB = PROJECT_ROOT / "mlflow.db"

MLFLOW_EXPERIMENT = "ecopulse-waste-detection"

DEFAULT_DATA_YAML = (
    PROJECT_ROOT
    / "artifacts"
    / "datasets"
    / "processed"
    / "data.yaml"
)


# ============================================================
# EXPECTED DATASET CLASSES
# ============================================================

EXPECTED_CLASSES = {
    0: "litter",
    1: "pile",
    2: "face_mask",
    3: "trash_bin",
    4: "plastic_bag",
    5: "bottle",
    6: "cup",
    7: "rope",
    8: "sachet",
    9: "straw",
}


# ============================================================
# DEFAULT MODEL CHECKPOINTS
# ============================================================

DEFAULT_CHECKPOINTS = {
    "yolo11n": (
        PROJECT_ROOT
        / "artifacts"
        / "checkpoints"
        / "yolo11n_best.pt"
    ),
    "yolov8n": (
        PROJECT_ROOT
        / "artifacts"
        / "checkpoints"
        / "yolov8n_best.pt"
    ),
}


# ============================================================
# PATH UTILITIES
# ============================================================

def resolve_path(value: str | Path) -> Path:
    """Resolve an absolute or project-relative path."""

    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


# ============================================================
# MLFLOW CONFIGURATION
# ============================================================

def setup_mlflow() -> None:
    """Configure the common EcoPulse AI MLflow experiment."""

    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{TRACKING_DB.as_posix()}"
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)


# ============================================================
# DATASET CONFIGURATION
# ============================================================

def normalize_class_names(names: Any) -> dict[int, str]:
    """Normalize YOLO class names into an integer-keyed mapping."""

    if isinstance(names, list):
        return {
            index: str(name)
            for index, name in enumerate(names)
        }

    if isinstance(names, dict):
        return {
            int(key): str(value)
            for key, value in names.items()
        }

    raise ValueError(
        "Dataset YAML must contain a valid names list or mapping."
    )


def load_dataset_config(
    data_yaml_path: str | Path,
) -> tuple[Path, dict]:
    """Load and validate the YOLO dataset configuration."""

    data_yaml = resolve_path(data_yaml_path)

    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_yaml}"
        )

    with data_yaml.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ValueError(
            "Dataset YAML must contain a mapping."
        )

    dataset_classes = normalize_class_names(
        config.get("names")
    )

    if dataset_classes != EXPECTED_CLASSES:
        raise ValueError(
            "Dataset class mapping does not match expected classes.\n"
            f"Expected: {EXPECTED_CLASSES}\n"
            f"Found: {dataset_classes}"
        )

    return data_yaml, config


# ============================================================
# CHECKPOINT VALIDATION
# ============================================================

def validate_checkpoint(
    model: YOLO,
    checkpoint: Path,
) -> dict[int, str]:
    """Verify checkpoint classes against project classes."""

    model_classes = normalize_class_names(model.names)

    if model_classes != EXPECTED_CLASSES:
        raise ValueError(
            f"Checkpoint class mapping mismatch: {checkpoint}\n"
            f"Expected: {EXPECTED_CLASSES}\n"
            f"Found: {model_classes}"
        )

    return model_classes


# ============================================================
# DATASET IMAGE DISCOVERY
# ============================================================

def get_split_images(
    data_yaml: Path,
    config: dict,
    split: str,
) -> list[Path]:
    """Find images belonging to the requested dataset split."""

    if split not in ("train", "val", "test"):
        raise ValueError(
            "Split must be train, val, or test."
        )

    base_path = Path(
        config.get("path", data_yaml.parent)
    )

    if not base_path.is_absolute():
        base_path = data_yaml.parent / base_path

    split_value = config.get(split)

    if not split_value:
        raise ValueError(
            f"Dataset YAML does not define '{split}'."
        )

    split_values = (
        split_value
        if isinstance(split_value, list)
        else [split_value]
    )

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
        ".tif",
        ".tiff",
    }

    images = []

    for value in split_values:
        split_path = Path(value)

        if not split_path.is_absolute():
            split_path = base_path / split_path

        if not split_path.is_dir():
            raise FileNotFoundError(
                f"Image directory not found: {split_path}"
            )

        images.extend(
            path
            for path in split_path.rglob("*")
            if path.is_file()
            and path.suffix.lower() in extensions
        )

    images = sorted(set(images))

    if not images:
        raise ValueError(
            f"No images found in the {split} split."
        )

    return images


# ============================================================
# INFERENCE LATENCY BENCHMARK
# ============================================================

def benchmark_latency(
    model: YOLO,
    sample_images: list[Path],
    warmup: int = 3,
    iterations: int = 10,
    device: str | int | None = None,
) -> dict[str, float]:
    """Measure inference latency in milliseconds."""

    valid_images = [
        path for path in sample_images
        if path.is_file()
    ]

    if not valid_images:
        raise ValueError(
            "No valid images available for latency benchmarking."
        )

    if iterations < 1:
        raise ValueError(
            "Iterations must be greater than zero."
        )

    predict_args = {
        "verbose": False,
    }

    if device is not None:
        predict_args["device"] = device

    for _ in range(warmup):
        model.predict(
            source=str(valid_images[0]),
            **predict_args,
        )

    latencies = []

    for index in range(iterations):
        image = valid_images[index % len(valid_images)]

        start = time.perf_counter()

        model.predict(
            source=str(image),
            **predict_args,
        )

        elapsed = (
            time.perf_counter() - start
        ) * 1000

        latencies.append(elapsed)

    return {
        "mean_ms": round(
            float(np.mean(latencies)), 2
        ),
        "median_ms": round(
            float(np.median(latencies)), 2
        ),
        "p95_ms": round(
            float(np.percentile(latencies, 95)), 2
        ),
    }


# ============================================================
# MARKDOWN REPORT
# ============================================================

def generate_markdown_report(
    result: dict,
    output_path: Path,
) -> None:
    """Generate a readable evaluation report."""

    metrics = result["metrics"]
    latency = result["inference_latency"]
    gate = result["quality_gate"]

    class_mapping = "\n".join(
        f"- {key}: {value}"
        for key, value in result["classes"].items()
    )

    markdown = f"""# EcoPulse AI Model Evaluation

## Model Information

- Model: {result['model_name']}
- Checkpoint: {result['checkpoint']}
- Dataset: {result['dataset']}
- Split: {result['split']}
- Evaluation images: {result['evaluation_images']}
- Number of classes: {result['num_classes']}
- Model size: {result['model_size_mb']} MB

## Performance Metrics

| Metric | Result |
|---|---:|
| Precision | {metrics['precision']:.4f} |
| Recall | {metrics['recall']:.4f} |
| mAP50 | {metrics['map50']:.4f} |
| mAP50-95 | {metrics['map50_95']:.4f} |

## Inference Latency

| Metric | Result |
|---|---:|
| Mean | {latency['mean_ms']} ms |
| Median | {latency['median_ms']} ms |
| P95 | {latency['p95_ms']} ms |

## Quality Gate

- Decision: {gate['decision']}
- Minimum mAP50: {gate['min_map50_threshold']}
- Maximum mean latency: {gate['max_latency_threshold_ms']} ms
- Passed: {gate['passed']}

## Class Mapping

{class_mapping}
"""

    output_path.write_text(
        markdown,
        encoding="utf-8",
    )


# ============================================================
# PER-CLASS METRICS
# ============================================================

def extract_per_class_metrics(
    validation_metrics: Any,
    class_names: dict[int, str],
) -> list[dict]:
    """Extract per-class metrics when available."""

    box_metrics = getattr(
        validation_metrics,
        "box",
        None,
    )

    if box_metrics is None:
        return []

    precision = getattr(box_metrics, "p", None)
    recall = getattr(box_metrics, "r", None)
    ap50 = getattr(box_metrics, "ap50", None)
    ap = getattr(box_metrics, "ap", None)
    class_ids = getattr(
        box_metrics,
        "ap_class_index",
        None,
    )

    if any(
        value is None
        for value in [precision, recall, ap50, ap, class_ids]
    ):
        return []

    per_class = []

    for index, class_id in enumerate(class_ids):
        class_id = int(class_id)

        ap_values = np.asarray(ap[index])

        per_class.append({
            "class_id": class_id,
            "class_name": class_names[class_id],
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "map50": float(ap50[index]),
            "map50_95": float(np.mean(ap_values)),
        })

    return per_class


# ============================================================
# COMPLETE ARTIFACT GENERATION
# ============================================================

def create_evaluation_artifacts(
    result: dict,
    data_yaml: Path,
    sample_images: list[Path],
    validation_metrics: Any,
    checkpoint: Path,
) -> Path:
    """Create detailed artifacts for MLflow."""

    artifact_dir = (
        REPORTS_DIR
        / "mlflow_artifacts"
        / result["model_name"]
        / result["split"]
    )

    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)

    evaluation_dir = artifact_dir / "evaluation"
    dataset_dir = artifact_dir / "dataset"
    model_dir = artifact_dir / "model"
    visual_dir = artifact_dir / "visualizations"
    environment_dir = artifact_dir / "environment"

    for directory in [
        evaluation_dir,
        dataset_dir,
        model_dir,
        visual_dir,
        environment_dir,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # Dataset summary
    dataset_summary = {
        "project": "EcoPulse AI",
        "dataset": str(data_yaml),
        "split": result["split"],
        "evaluation_images": len(sample_images),
        "num_classes": result["num_classes"],
        "classes": result["classes"],
    }

    (dataset_dir / "dataset_summary.json").write_text(
        json.dumps(dataset_summary, indent=4),
        encoding="utf-8",
    )

    shutil.copy2(
        data_yaml,
        dataset_dir / "data.yaml",
    )

    # Evaluation summary
    (evaluation_dir / "evaluation_summary.json").write_text(
        json.dumps(result, indent=4),
        encoding="utf-8",
    )

    # Per-class metrics
    per_class = extract_per_class_metrics(
        validation_metrics,
        {
            int(key): value
            for key, value in result["classes"].items()
        },
    )

    if per_class:
        csv_path = (
            evaluation_dir
            / "per_class_metrics.csv"
        )

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(per_class[0].keys()),
            )
            writer.writeheader()
            writer.writerows(per_class)

        (evaluation_dir / "per_class_metrics.json").write_text(
            json.dumps(per_class, indent=4),
            encoding="utf-8",
        )

    # Model metadata
    model_metadata = {
        "model_name": result["model_name"],
        "checkpoint": str(checkpoint),
        "model_size_mb": result["model_size_mb"],
        "num_classes": result["num_classes"],
        "classes": result["classes"],
    }

    (model_dir / "model_metadata.json").write_text(
        json.dumps(model_metadata, indent=4),
        encoding="utf-8",
    )

    # Copy YOLO-generated validation plots
    save_dir = getattr(
        validation_metrics,
        "save_dir",
        None,
    )

    copied_plots = []

    if save_dir:
        save_dir = Path(save_dir)

        if save_dir.exists():
            for file in save_dir.rglob("*"):
                if (
                    file.is_file()
                    and file.suffix.lower() in {
                        ".png",
                        ".jpg",
                        ".jpeg",
                        ".csv",
                    }
                ):
                    destination = (
                        visual_dir / file.name
                    )

                    shutil.copy2(
                        file,
                        destination,
                    )

                    copied_plots.append(file.name)

    # Record plot information
    (visual_dir / "visualization_manifest.json").write_text(
        json.dumps(
            {
                "source_directory": (
                    str(save_dir) if save_dir else None
                ),
                "copied_files": copied_plots,
            },
            indent=4,
        ),
        encoding="utf-8",
    )

    # Environment details
    environment = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "ultralytics_version": ultralytics.__version__,
        "numpy_version": np.__version__,
        "mlflow_version": mlflow.__version__,
        "evaluation_time_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    (environment_dir / "run_environment.json").write_text(
        json.dumps(environment, indent=4),
        encoding="utf-8",
    )

    # Installed dependencies
    requirements_path = (
        environment_dir / "requirements.txt"
    )

    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "freeze",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        requirements_path.write_text(
            completed.stdout,
            encoding="utf-8",
        )

    except subprocess.SubprocessError as error:
        requirements_path.write_text(
            f"Could not collect dependencies: {error}",
            encoding="utf-8",
        )

    return artifact_dir


# ============================================================
# MLFLOW EVALUATION LOGGING
# ============================================================

def log_evaluation_to_mlflow(
    result: dict,
    json_path: Path,
    markdown_path: Path,
    artifact_dir: Path,
) -> None:
    """Log complete evaluation details to EcoPulse MLflow."""

    setup_mlflow()

    with mlflow.start_run(
        run_name=(
            f"evaluation-{result['model_name']}-"
            f"{result['split']}"
        )
    ):

        mlflow.set_tags({
            "project": "EcoPulse AI",
            "stage": "model_evaluation",
            "task": "waste_object_detection",
            "framework": "Ultralytics YOLO",
            "model_name": result["model_name"],
            "dataset_split": result["split"],
            "quality_gate": result["quality_gate"]["decision"],
        })

        # Parameters
        parameters = {
            "checkpoint": result["checkpoint"],
            "dataset": result["dataset"],
            "split": result["split"],
            "num_classes": result["num_classes"],
            "evaluation_images": result["evaluation_images"],
            "model_size_mb": result["model_size_mb"],
            "min_map50_threshold": (
                result["quality_gate"]["min_map50_threshold"]
            ),
            "max_latency_threshold_ms": (
                result["quality_gate"]["max_latency_threshold_ms"]
            ),
        }

        mlflow.log_params(parameters)

        for class_id, class_name in result["classes"].items():
            mlflow.log_param(
                f"class_{class_id}",
                class_name,
            )

        # Metrics
        mlflow.log_metrics({
            "precision": result["metrics"]["precision"],
            "recall": result["metrics"]["recall"],
            "map50": result["metrics"]["map50"],
            "map50_95": result["metrics"]["map50_95"],
            "latency_mean_ms": result["inference_latency"]["mean_ms"],
            "latency_median_ms": result["inference_latency"]["median_ms"],
            "latency_p95_ms": result["inference_latency"]["p95_ms"],
            "quality_gate_passed": int(
                result["quality_gate"]["passed"]
            ),
        })

        # Existing evaluation reports
        mlflow.log_artifact(
            str(json_path),
            artifact_path="evaluation",
        )

        mlflow.log_artifact(
            str(markdown_path),
            artifact_path="evaluation",
        )

        # Complete artifact directory
        mlflow.log_artifacts(
            str(artifact_dir),
            artifact_path="ecopulse_details",
        )

        run = mlflow.active_run()

        print("\nMLflow logging completed successfully.")
        print(f"Experiment: {MLFLOW_EXPERIMENT}")
        print(f"Run ID: {run.info.run_id}")
        print(f"Artifacts: {artifact_dir}")


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    checkpoint_path: str | Path,
    data_yaml_path: str | Path = DEFAULT_DATA_YAML,
    split: str = "test",
    min_map50: float = 0.20,
    max_latency_ms: float = 500.0,
    device: str | int | None = None,
    log_mlflow: bool = True,
) -> dict[str, Any]:
    """
    Evaluate an existing YOLO checkpoint.

    This function does not retrain the model.
    """

    checkpoint = resolve_path(checkpoint_path)

    data_yaml, data_config = load_dataset_config(
        data_yaml_path
    )

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}"
        )

    print("=" * 65)
    print("ECOPULSE AI - MODEL EVALUATION")
    print("=" * 65)

    print(f"Checkpoint: {checkpoint}")
    print(f"Dataset: {data_yaml}")
    print(f"Split: {split}")

    model = YOLO(str(checkpoint))

    class_names = validate_checkpoint(
        model,
        checkpoint,
    )

    sample_images = get_split_images(
        data_yaml,
        data_config,
        split,
    )

    print(
        f"Evaluation images: {len(sample_images)}"
    )

    model_size_mb = round(
        checkpoint.stat().st_size / (1024 * 1024),
        2,
    )

    validation_args = {
        "data": str(data_yaml),
        "split": split,
        "verbose": False,
        "plots": True,
    }

    if device is not None:
        validation_args["device"] = device

    print("\nRunning model validation...")

    metrics = model.val(**validation_args)

    print(
        f"YOLO validation output: {metrics.save_dir}"
    )

    results_dict = metrics.results_dict

    precision = float(
        results_dict.get("metrics/precision(B)", 0.0)
    )

    recall = float(
        results_dict.get("metrics/recall(B)", 0.0)
    )

    map50 = float(
        results_dict.get("metrics/mAP50(B)", 0.0)
    )

    map50_95 = float(
        results_dict.get("metrics/mAP50-95(B)", 0.0)
    )

    print("\nBenchmarking inference latency...")

    latency = benchmark_latency(
        model,
        sample_images,
        device=device,
    )

    # Quality gate
    qg_passed = (
        map50 >= min_map50
        and latency["mean_ms"] <= max_latency_ms
    )

    decision = (
        "PASSED"
        if qg_passed
        else "QUALITY_GATE_WARNING"
    )

    # Evaluation result
    eval_result = {
        "project": "EcoPulse AI",
        "checkpoint": str(checkpoint),
        "model_name": checkpoint.stem,
        "dataset": str(data_yaml),
        "classes": {
            str(key): value
            for key, value in class_names.items()
        },
        "num_classes": len(class_names),
        "split": split,
        "evaluation_images": len(sample_images),
        "model_size_mb": model_size_mb,
        "inference_latency": latency,
        "metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "map50": round(map50, 4),
            "map50_95": round(map50_95, 4),
        },
        "quality_gate": {
            "decision": decision,
            "min_map50_threshold": min_map50,
            "max_latency_threshold_ms": max_latency_ms,
            "passed": qg_passed,
        },
    }

    # Reports directory
    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = REPORTS_DIR / (
        f"{checkpoint.stem}_{split}_evaluation.json"
    )

    markdown_path = REPORTS_DIR / (
        f"{checkpoint.stem}_{split}_evaluation.md"
    )

    json_path.write_text(
        json.dumps(eval_result, indent=4),
        encoding="utf-8",
    )

    generate_markdown_report(
        eval_result,
        markdown_path,
    )

    # Console summary
    print("\n" + "=" * 65)
    print("EVALUATION RESULTS")
    print("=" * 65)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"mAP50: {map50:.4f}")
    print(f"mAP50-95: {map50_95:.4f}")

    print(
        f"Mean latency: {latency['mean_ms']} ms"
    )

    print(
        f"Median latency: {latency['median_ms']} ms"
    )

    print(
        f"P95 latency: {latency['p95_ms']} ms"
    )

    print(
        f"Model size: {model_size_mb} MB"
    )

    print(
        f"Quality gate: {decision}"
    )

    print(f"\nJSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")

    # Detailed MLflow artifact logging
    if log_mlflow:

        artifact_dir = create_evaluation_artifacts(
            result=eval_result,
            data_yaml=data_yaml,
            sample_images=sample_images,
            validation_metrics=metrics,
            checkpoint=checkpoint,
        )

        log_evaluation_to_mlflow(
            eval_result,
            json_path,
            markdown_path,
            artifact_dir,
        )

    return eval_result


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description="EcoPulse AI model evaluation"
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to an existing YOLO checkpoint",
    )

    parser.add_argument(
        "--data",
        default=os.environ.get(
            "WASTE_DATA_YAML",
            str(DEFAULT_DATA_YAML),
        ),
        help="Path to the YOLO dataset YAML",
    )

    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
    )

    parser.add_argument(
        "--min-map50",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--max-latency",
        type=float,
        default=500.0,
    )

    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, e.g. cpu or 0",
    )

    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Skip MLflow logging",
    )

    args = parser.parse_args()

    result = evaluate_model(
        checkpoint_path=args.checkpoint,
        data_yaml_path=args.data,
        split=args.split,
        min_map50=args.min_map50,
        max_latency_ms=args.max_latency,
        device=args.device,
        log_mlflow=not args.no_mlflow,
    )

    print("\nFinal quality gate:")
    print(
        json.dumps(
            result["quality_gate"],
            indent=4,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### `src/models/predict.py`
```python
from pathlib import Path
import os
from typing import Any, Optional

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CHECKPOINTS = {
    "yolo11n": PROJECT_ROOT / "artifacts" / "checkpoints" / "yolo11n_best.pt",
    "yolov8n": PROJECT_ROOT / "artifacts" / "checkpoints" / "yolov8n_best.pt",
}

LOCAL_CHECKPOINT = DEFAULT_CHECKPOINTS["yolo11n"]
DRIVE_CHECKPOINT = (
    Path(r"G:\My Drive\datasets\Ai_Waste_Campaign")
    / "yolo_runs"
    / "yolo11n_waste_detection"
    / "weights"
    / "best.pt"
)

CLASS_NAMES = {
    0: "litter",
    1: "pile",
    2: "face_mask",
    3: "trash_bin",
    4: "plastic_bag",
    5: "bottle",
    6: "cup",
    7: "rope",
    8: "sachet",
    9: "straw",
}

_model_cache: dict[str, YOLO] = {}


def get_checkpoint_path(model_key: str = "yolo11n") -> Path:
    """Resolve the active model checkpoint path with fallback.
    
    Args:
        model_key: Checkpoint key ('yolo11n' or 'yolov8n') or path string.
    """
    env_path = os.environ.get("WASTE_MODEL_PATH")
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return p

    key = model_key.lower().replace("-", "").replace("_", "")
    for k, p in DEFAULT_CHECKPOINTS.items():
        if k in key and p.is_file():
            return p

    # Fallback to local default checkpoint
    if LOCAL_CHECKPOINT.is_file():
        return LOCAL_CHECKPOINT

    # Fallback to YOLOv8 if YOLO11 is absent
    if DEFAULT_CHECKPOINTS["yolov8n"].is_file():
        return DEFAULT_CHECKPOINTS["yolov8n"]

    if DRIVE_CHECKPOINT.is_file():
        return DRIVE_CHECKPOINT

    return LOCAL_CHECKPOINT


def detect_model_name(checkpoint_path: str | Path) -> str:
    """Infer model family name from checkpoint path or filename."""
    path_str = str(checkpoint_path).lower()
    if "yolo11" in path_str:
        return "YOLO11n"
    if "yolov8" in path_str:
        return "YOLOv8n"
    return Path(checkpoint_path).stem


def load_model(checkpoint_path: str | Path | None = None) -> YOLO:
    """Load and cache the YOLO model checkpoint."""
    global _model_cache

    target_path = Path(checkpoint_path) if checkpoint_path else get_checkpoint_path()
    if not target_path.is_absolute():
        target_path = PROJECT_ROOT / target_path

    target_key = str(target_path.resolve())

    if target_key in _model_cache:
        return _model_cache[target_key]

    if not target_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {target_path}. "
            f"Ensure {LOCAL_CHECKPOINT} exists or set WASTE_MODEL_PATH."
        )

    print(f"Loading trained model: {target_path}")
    model = YOLO(str(target_path))
    _model_cache[target_key] = model

    return model


def predict_image(
    image_path: str | Path,
    confidence: float = 0.25,
    checkpoint_path: str | Path | None = None,
    save: bool = False,
    save_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run waste detection on a single image file.
    
    Args:
        image_path: Path to the image file.
        confidence: Minimum confidence threshold (0.0 to 1.0).
        checkpoint_path: Optional custom model checkpoint path.
        save: Whether to save predicted visualization to disk.
        save_dir: Output directory if saving visualization.
    """
    image_path = Path(image_path)
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = load_model(checkpoint_path)
    active_path = str(Path(checkpoint_path) if checkpoint_path else get_checkpoint_path())
    model_name = detect_model_name(active_path)

    predict_kwargs: dict[str, Any] = {
        "source": str(image_path),
        "conf": confidence,
        "verbose": False,
    }

    if save:
        predict_kwargs["save"] = True
        if save_dir:
            out_p = Path(save_dir)
            out_p.mkdir(parents=True, exist_ok=True)
            predict_kwargs["project"] = str(out_p.parent)
            predict_kwargs["name"] = out_p.name
            predict_kwargs["exist_ok"] = True

    results = model.predict(**predict_kwargs)

    predictions = []
    saved_image_path = None

    for result in results:
        if save and hasattr(result, "save_dir") and result.save_dir:
            cand = Path(result.save_dir) / image_path.name
            if cand.is_file():
                saved_image_path = str(cand)

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names.get(
                class_id, CLASS_NAMES.get(class_id, f"class_{class_id}")
            )

            predictions.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(float(box.conf[0]), 4),
                "bbox": [
                    round(float(value), 2)
                    for value in box.xyxy[0].tolist()
                ],
            })

    output: dict[str, Any] = {
        "image": image_path.name,
        "image_path": str(image_path),
        "model": model_name,
        "checkpoint": active_path,
        "total_detections": len(predictions),
        "predictions": predictions,
    }

    if saved_image_path:
        output["saved_image_path"] = saved_image_path

    return output


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="EcoPulse AI Single Image Waste Detection")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold (0-1)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Custom checkpoint path")
    parser.add_argument("--save", action="store_true", help="Save annotated prediction image")

    args = parser.parse_args()

    out = predict_image(args.image, args.confidence, args.checkpoint, save=args.save)
    print(json.dumps(out, indent=2))
```

### `src/api/app.py`
```python
import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from src.models.predict import DEFAULT_CHECKPOINTS, get_checkpoint_path, load_model, predict_image
from src.tracking.mlflow_utils import log_prediction


app = FastAPI(
    title="EcoPulse AI Waste Detection API",
    description="Waste detection using trained YOLO11n and YOLOv8n models",
    version="1.0.0",
)


@app.get("/")
def home():
    return {
        "project": "EcoPulse AI",
        "message": "EcoPulse AI Waste Detection API is running",
        "available_models": ["yolo11n", "yolov8n"],
        "default_model": "yolo11n",
    }


@app.get("/health")
def health():
    try:
        active_checkpoint = get_checkpoint_path()
        model = load_model(active_checkpoint)

        return {
            "status": "healthy",
            "model_loaded": True,
            "active_checkpoint": str(active_checkpoint),
            "classes": model.names,
            "available_checkpoints": {
                k: str(v) for k, v in DEFAULT_CHECKPOINTS.items() if v.is_file()
            },
        }

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        )


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    confidence: float = 0.25,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    site_id: Optional[str] = None,
    model_name: str = Query(
        default="yolo11n",
        description="Model family to run ('yolo11n' or 'yolov8n')",
    ),
):
    # Validate confidence
    if not 0.0 <= confidence <= 1.0:
        raise HTTPException(
            status_code=400,
            detail="Confidence must be between 0 and 1.",
        )

    # Validate image extension
    allowed_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    extension = Path(file.filename or "").suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format. Allowed: .jpg, .jpeg, .png, .bmp, .webp",
        )

    # Validate GPS coordinates
    if (latitude is None) != (longitude is None):
        raise HTTPException(
            status_code=400,
            detail="Provide both latitude and longitude, or neither.",
        )

    if latitude is not None:
        if not (
            -90 <= latitude <= 90
            and -180 <= longitude <= 180
        ):
            raise HTTPException(
                status_code=400,
                detail="Invalid GPS coordinates.",
            )

    checkpoint_path = get_checkpoint_path(model_name)
    if not checkpoint_path.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"Requested model checkpoint '{model_name}' not found at {checkpoint_path}.",
        )

    temporary_path = None

    try:
        # Read uploaded image
        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty.",
            )

        # Save temporary image
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
        ) as temporary_file:
            temporary_file.write(contents)
            temporary_path = temporary_file.name

        # Run inference
        result = predict_image(
            temporary_path,
            confidence=confidence,
            checkpoint_path=checkpoint_path,
        )

        result["uploaded_filename"] = file.filename

        # Log prediction to MLflow
        log_prediction(
            result,
            latitude=latitude,
            longitude=longitude,
            site_id=site_id,
        )

        return result

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    finally:
        # Remove temporary uploaded image
        if temporary_path and os.path.exists(temporary_path):
            try:
                os.remove(temporary_path)
            except OSError:
                pass
```

### `src/models/train.py`
```python
"""
EcoPulse AI - Model Training & MLflow Tracking

Configurable training entrypoint for YOLO waste detection models.
Integrates with MLflow experiment tracking and the 10-class dataset configuration.
"""

from pathlib import Path
import os
import sys
from typing import Any

import mlflow
import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train.yaml"
DEFAULT_DATA_YAML = (
    PROJECT_ROOT / "artifacts" / "datasets" / "processed" / "data.yaml"
)


def train_model(
    config_path: str | Path | None = None,
    data_yaml_path: str | Path | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    image_size: int | None = None,
    device: str | None = None,
    experiment_name: str | None = None,
) -> dict[str, Any]:
    """Train or fine-tune YOLO waste detection model with MLflow tracking."""
    cfg_path = Path(config_path or DEFAULT_CONFIG)

    config = {}
    if cfg_path.is_file():
        with cfg_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    training_cfg = config.get("training", {})
    output_cfg = config.get("output", {})
    tracking_cfg = config.get("tracking", {})

    model_name = config.get("model", {}).get("name", "yolov8n.yaml")
    data_path = Path(
        data_yaml_path
        or os.environ.get("WASTE_DATA_YAML", str(DEFAULT_DATA_YAML))
    )

    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    epochs_val = epochs if epochs is not None else int(training_cfg.get("epochs", 1))
    batch_val = batch_size if batch_size is not None else int(training_cfg.get("batch_size", 4))
    imgsz_val = image_size if image_size is not None else int(training_cfg.get("image_size", 320))
    device_val = device or training_cfg.get("device", "cpu")

    output_dir = PROJECT_ROOT / output_cfg.get("project", "runs/detect")
    run_name = output_cfg.get("name", "ecopulse_waste_training")

    print("=" * 60)
    print("EcoPulse AI - Model Training")
    print(f"Model: {model_name}")
    print(f"Dataset: {data_path}")
    print(f"Epochs: {epochs_val} | Batch: {batch_val} | ImgSz: {imgsz_val} | Device: {device_val}")
    print("=" * 60)

    model = YOLO(model_name)

    train_args = {
        "data": str(data_path),
        "epochs": epochs_val,
        "imgsz": imgsz_val,
        "batch": batch_val,
        "device": device_val,
        "workers": int(training_cfg.get("workers", 0)),
        "seed": int(training_cfg.get("seed", 42)),
        "project": str(output_dir),
        "name": run_name,
        "exist_ok": True,
        "pretrained": False,
        "plots": True,
    }

    exp_name = experiment_name or tracking_cfg.get("experiment_name", "ecopulse-waste-detection")
    tracking_db = PROJECT_ROOT / "mlflow.db"
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{tracking_db.as_posix()}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(exp_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model_architecture": model_name,
            "epochs": epochs_val,
            "image_size": imgsz_val,
            "batch_size": batch_val,
            "device": device_val,
            "dataset_yaml": str(data_path),
        })

        results = model.train(**train_args)

        save_dir = Path(results.save_dir)
        best_pt = save_dir / "weights" / "best.pt"
        if best_pt.is_file():
            mlflow.log_artifact(str(best_pt), artifact_path="model")

        results_csv = save_dir / "results.csv"
        if results_csv.is_file():
            mlflow.log_artifact(str(results_csv), artifact_path="training_results")

        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow training run completed. Run ID: {run_id}")

    return {
        "run_id": run_id,
        "save_dir": str(results.save_dir),
        "best_checkpoint": str(best_pt) if best_pt.is_file() else None,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EcoPulse AI Training")
    parser.add_argument("--config", default=None, help="Path to train.yaml")
    parser.add_argument("--data", default=None, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate configs without training")

    args = parser.parse_args()

    if args.dry_run:
        data_p = Path(args.data or DEFAULT_DATA_YAML)
        if not data_p.is_file():
            sys.exit(f"Dataset not found: {data_p}")
        print(f"Dry run check passed for dataset: {data_p}")
    else:
        train_model(
            config_path=args.config,
            data_yaml_path=args.data,
            epochs=args.epochs,
            batch_size=args.batch,
        )
```

### `pipelines/training_pipeline.py`
```python

from pathlib import Path
import csv
import os
import sys

import mlflow
import yaml
from ultralytics import YOLO


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_CONFIG_PATH = PROJECT_ROOT / "configs" / "train.yaml"
DATA_CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"

TRACKING_DB = PROJECT_ROOT / "mlflow.db"

DEFAULT_EXPERIMENT = "ecopulse-waste-detection"

REPORTS_DIR = PROJECT_ROOT / "reports"
FEATURES_DIR = PROJECT_ROOT / "artifacts" / "features"


# ============================================================
# CONFIGURATION LOADING
# ============================================================

def load_yaml(path: Path) -> dict:
    """Load a YAML file and validate its structure."""

    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML configuration: {path}")

    return config


def load_config() -> dict:
    """Load training configuration."""

    config = load_yaml(TRAIN_CONFIG_PATH)

    for section in ("model", "training", "output"):
        if section not in config:
            raise ValueError(
                f"Missing '{section}' section in {TRAIN_CONFIG_PATH}"
            )

    return config


def load_dataset_config() -> dict:
    """Load project-level dataset configuration."""

    config = load_yaml(DATA_CONFIG_PATH)

    if "dataset" not in config:
        raise ValueError(
            f"Missing 'dataset' section in {DATA_CONFIG_PATH}"
        )

    return config["dataset"]


# ============================================================
# DATASET PATH RESOLUTION
# ============================================================

def resolve_data_yaml(config: dict | None = None) -> Path:
    """
    Resolve the actual YOLO dataset YAML.

    Priority:
    1. WASTE_DATA_YAML environment variable.
    2. configs/data.yaml dataset.data_yaml.

    The small data/waste10 dataset is never selected automatically.
    """

    env_yaml = os.environ.get("WASTE_DATA_YAML")

    if env_yaml:
        data_yaml = Path(env_yaml)

        if not data_yaml.is_absolute():
            data_yaml = PROJECT_ROOT / data_yaml

    else:
        dataset_config = load_dataset_config()

        configured_path = dataset_config.get("data_yaml")

        if not configured_path:
            raise ValueError(
                "dataset.data_yaml is missing from configs/data.yaml"
            )

        data_yaml = Path(configured_path)

        if not data_yaml.is_absolute():
            data_yaml = PROJECT_ROOT / data_yaml

    data_yaml = data_yaml.resolve()

    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_yaml}"
        )

    return data_yaml


# ============================================================
# DATASET VALIDATION
# ============================================================

def resolve_split_path(base_path: Path, split_value) -> list[Path]:
    """Resolve one or more image directories from a YOLO split entry."""

    values = split_value if isinstance(split_value, list) else [split_value]

    paths = []

    for value in values:
        path = Path(value)

        if not path.is_absolute():
            path = base_path / path

        paths.append(path.resolve())

    return paths


def validate_paths(data_yaml_path: Path) -> dict:
    """Validate dataset YAML and train/val/test image directories."""

    dataset = load_yaml(data_yaml_path)

    configured_base = dataset.get("path")

    if configured_base:
        base_path = Path(configured_base)

        if not base_path.is_absolute():
            base_path = data_yaml_path.parent / base_path
    else:
        base_path = data_yaml_path.parent

    base_path = base_path.resolve()

    for split in ("train", "val", "test"):

        if split not in dataset:
            raise ValueError(
                f"Dataset YAML is missing the '{split}' entry."
            )

        split_paths = resolve_split_path(
            base_path,
            dataset[split]
        )

        for split_path in split_paths:

            if not split_path.is_dir():
                raise FileNotFoundError(
                    f"Dataset split '{split}' not found: {split_path}"
                )

            image_count = sum(
                1
                for file in split_path.rglob("*")
                if file.is_file()
                and file.suffix.lower() in (
                    ".jpg", ".jpeg", ".png", ".bmp",
                    ".webp", ".tif", ".tiff"
                )
            )

            print(
                f"{split.upper()}: {image_count} images "
                f"({split_path})"
            )

            if image_count == 0:
                raise ValueError(
                    f"No images found in {split_path}"
                )

    names = dataset.get("names", {})

    if isinstance(names, list):
        class_names = {
            index: str(name)
            for index, name in enumerate(names)
        }
    elif isinstance(names, dict):
        class_names = {
            int(index): str(name)
            for index, name in names.items()
        }
    else:
        raise ValueError(
            "Dataset YAML must contain a valid 'names' list or mapping."
        )

    if not class_names:
        raise ValueError("No classes found in dataset YAML.")

    if len(class_names) != len(set(class_names)):
        raise ValueError("Duplicate class IDs found.")

    print(f"\nDataset YAML: {data_yaml_path}")
    print(f"Dataset root: {base_path}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {class_names}")

    return {
        "dataset": dataset,
        "base_path": base_path,
        "class_names": class_names,
    }


# ============================================================
# MLFLOW SETUP
# ============================================================

def setup_mlflow(experiment_name: str = DEFAULT_EXPERIMENT):
    """Configure the existing local MLflow tracking database."""

    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{TRACKING_DB.as_posix()}"
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_existing_artifact(path: Path, artifact_path: str):
    """Log an artifact only if it already exists."""

    if path.is_file():
        mlflow.log_artifact(
            str(path),
            artifact_path=artifact_path
        )

        print(f"MLflow artifact logged: {path}")

    else:
        print(f"Artifact not available yet: {path}")


def log_existing_directory(directory: Path, artifact_path: str):
    """Log files from an existing directory."""

    if not directory.is_dir():
        print(f"Directory not available yet: {directory}")
        return

    files = [
        file for file in directory.rglob("*")
        if file.is_file()
    ]

    for file in files:
        relative_parent = file.parent.relative_to(directory)

        destination = Path(artifact_path) / relative_parent

        mlflow.log_artifact(
            str(file),
            artifact_path=destination.as_posix()
        )

    print(
        f"Logged {len(files)} files from {directory}"
    )


# ============================================================
# DATASET TRACKING
# ============================================================

def log_dataset_information(
    data_yaml_path: Path,
    validation_info: dict,
    project_dataset_config: dict,
):
    """Log dataset metadata and configuration to MLflow."""

    dataset = validation_info["dataset"]
    class_names = validation_info["class_names"]
    base_path = validation_info["base_path"]

    mlflow.log_params({
        "dataset_path": str(base_path),
        "dataset_yaml": str(data_yaml_path),
        "dataset_num_classes": len(class_names),
        "dataset_format": "YOLO",
        "dataset_train_path": str(dataset["train"]),
        "dataset_val_path": str(dataset["val"]),
        "dataset_test_path": str(dataset["test"]),
    })

    for class_id, class_name in class_names.items():
        mlflow.log_param(
            f"class_{class_id}",
            class_name
        )

    configured_classes = project_dataset_config.get("classes", {})

    if configured_classes:
        mlflow.log_dict(
            {
                "configured_classes": configured_classes,
                "actual_classes": class_names,
            },
            "dataset/class_mapping.json"
        )

    mlflow.log_dict(
        dataset,
        "dataset/dataset_configuration.json"
    )

    mlflow.log_artifact(
        str(data_yaml_path),
        artifact_path="dataset"
    )


# ============================================================
# DATA EXPLORATION AND VALIDATION TRACKING
# ============================================================

def log_dataset_reports():
    """Log existing dataset exploration and validation reports."""

    exploration_report = (
        REPORTS_DIR / "dataset_exploration.json"
    )

    validation_report = (
        REPORTS_DIR / "data_validation_report.json"
    )

    log_existing_artifact(
        exploration_report,
        "dataset/exploration"
    )

    log_existing_artifact(
        validation_report,
        "dataset/validation"
    )


# ============================================================
# FEATURE ENGINEERING TRACKING
# ============================================================

def log_feature_engineering_reports():
    """Log existing feature-engineering CSV outputs."""

    expected_files = [
        "image_features.csv",
        "object_features.csv",
        "class_distribution.csv",
        "dataset_feature_summary.csv",
        "label_quality_issues.csv",
    ]

    for filename in expected_files:

        file_path = FEATURES_DIR / filename

        log_existing_artifact(
            file_path,
            "feature_engineering"
        )


# ============================================================
# TRAINING METRIC TRACKING
# ============================================================

def log_training_metrics(results_csv: Path):
    """Read the last training epoch metrics from Ultralytics results.csv."""

    if not results_csv.is_file():
        print("Training results.csv not found.")
        return

    with results_csv.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        rows = list(csv.DictReader(file))

    if not rows:
        return

    final_row = rows[-1]

    for key, value in final_row.items():

        if key is None:
            continue

        try:
            metric_value = float(value)
        except (ValueError, TypeError):
            continue

        if not key.strip():
            continue

        metric_name = (
            key.strip()
            .replace(" ", "_")
            .replace("/", "_")
        )

        mlflow.log_metric(
            metric_name,
            metric_value
        )

    print("Final epoch metrics logged to MLflow.")


# ============================================================
# YOLO TRAINING
# ============================================================

def train(config: dict, data_yaml_path: Path):
    """
    Execute YOLO training and track the run in MLflow.

    This function is called only when explicitly requested.
    """

    training = config["training"]
    output = config["output"]
    tracking = config.get("tracking", {})

    project_dataset_config = load_dataset_config()

    validation_info = validate_paths(data_yaml_path)

    model_name = config["model"].get(
        "name",
        "yolov8n.yaml"
    )

    print(f"\nLoading YOLO architecture: {model_name}")

    model = YOLO(model_name)

    output_dir = (
        PROJECT_ROOT /
        output.get("project", "runs/detect")
    )

    run_name = output.get(
        "name",
        "ecopulse_waste_smoke_test"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    train_args = {
        "data": str(data_yaml_path),
        "epochs": int(training.get("epochs", 1)),
        "imgsz": int(training.get("image_size", 320)),
        "batch": int(training.get("batch_size", 4)),
        "device": training.get("device", "cpu"),
        "workers": int(training.get("workers", 0)),
        "patience": int(training.get("patience", 1)),
        "seed": int(training.get("seed", 42)),
        "project": str(output_dir),
        "name": run_name,
        "exist_ok": bool(output.get("exist_ok", True)),
        "pretrained": False,
        "plots": True,
    }

    def execute_training():

        mlflow.log_params({
            "model": model_name,
            "epochs": train_args["epochs"],
            "image_size": train_args["imgsz"],
            "batch_size": train_args["batch"],
            "device": str(train_args["device"]),
            "workers": train_args["workers"],
            "patience": train_args["patience"],
            "seed": train_args["seed"],
            "pretrained": False,
            "dataset": str(data_yaml_path),
            "run_name": run_name,
        })

        mlflow.set_tags({
            "project": "EcoPulse AI",
            "task": "waste_object_detection",
            "framework": "Ultralytics YOLO",
            "dataset_type": "plastic_waste",
        })

        # Dataset information
        log_dataset_information(
            data_yaml_path,
            validation_info,
            project_dataset_config
        )

        # Existing dataset reports
        log_dataset_reports()

        # Existing feature engineering reports
        log_feature_engineering_reports()

        print("\nStarting YOLO training...")

        results = model.train(**train_args)

        save_dir = Path(results.save_dir)

        # Best checkpoint
        best_weights = (
            save_dir / "weights" / "best.pt"
        )

        if best_weights.is_file():

            mlflow.log_artifact(
                str(best_weights),
                artifact_path="model"
            )

        # Last checkpoint
        last_weights = (
            save_dir / "weights" / "last.pt"
        )

        if last_weights.is_file():

            mlflow.log_artifact(
                str(last_weights),
                artifact_path="model"
            )

        # Training results CSV
        results_csv = save_dir / "results.csv"

        if results_csv.is_file():

            mlflow.log_artifact(
                str(results_csv),
                artifact_path="training_results"
            )

            log_training_metrics(results_csv)

        # YOLO plots and other training artifacts
        log_existing_directory(
            save_dir,
            "training_outputs"
        )

        mlflow.log_dict(
            {
                "model": model_name,
                "training_arguments": train_args,
                "dataset_yaml": str(data_yaml_path),
                "best_checkpoint": (
                    str(best_weights)
                    if best_weights.is_file()
                    else None
                ),
                "training_output": str(save_dir),
            },
            "training/training_summary.json"
        )

        print("\nTraining completed.")

        return results

    if tracking.get("enabled", False):

        experiment_name = tracking.get(
            "experiment_name",
            DEFAULT_EXPERIMENT
        )

        setup_mlflow(experiment_name)

        with mlflow.start_run(run_name=run_name):

            results = execute_training()

            print(
                "\nMLflow Run ID:",
                mlflow.active_run().info.run_id
            )

    else:

        results = model.train(**train_args)

    print("\nTraining run finished.")
    print(f"Output directory: {results.save_dir}")

    print(
        "Best checkpoint:",
        Path(results.save_dir) / "weights" / "best.pt"
    )

    return results


# ============================================================
# MAIN PIPELINE
# ============================================================

def main() -> int:
    """
    Validate the pipeline configuration and dataset paths.

    IMPORTANT:
    This function does not start model training.
    """

    try:

        print("=" * 60)
        print("ECOPULSE AI - MLOPS PIPELINE")
        print("=" * 60)

        config = load_config()

        project_dataset_config = load_dataset_config()

        data_yaml_path = resolve_data_yaml(config)

        validation_info = validate_paths(
            data_yaml_path
        )

        print("\nConfiguration summary:")
        print(f"Model: {config['model'].get('name')}")
        print(f"Dataset: {data_yaml_path}")
        print(
            f"Classes: {len(validation_info['class_names'])}"
        )
        print(
            f"MLflow enabled: {config.get('tracking', {}).get('enabled', False)}"
        )

        print(
            "\nDVC dataset reference:",
            project_dataset_config.get(
                "dvc_processed_dataset",
                "Not configured"
            )
        )

        print("\nPipeline configuration validation passed.")

        print(
            "No model training was started. "
            "Call train() explicitly to train the model."
        )

        return 0

    except Exception as exc:

        print(
            f"\nPIPELINE CHECK FAILED: {exc}",
            file=sys.stderr
        )

        raise


if __name__ == "__main__":
    raise SystemExit(main())
```

### `src/tracking/mlflow_utils.py`
```python
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Optional

import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRACKING_DB = PROJECT_ROOT / "mlflow.db"
DEFAULT_EXPERIMENT = "ecopulse-waste-detection"


def get_tracking_uri() -> str:
    """Resolve MLflow tracking URI, prioritizing environment variables."""
    return os.environ.get(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{TRACKING_DB.as_posix()}"
    )


def setup_tracking(experiment_name: str = DEFAULT_EXPERIMENT) -> None:
    """Configure MLflow tracking URI and experiment."""
    tracking_uri = get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_prediction(
    result: dict[str, Any],
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    site_id: Optional[str] = None,
    experiment_name: str = DEFAULT_EXPERIMENT,
) -> Optional[str]:
    """Log an inference event without logging uploaded image contents.
    
    Returns:
        The MLflow run_id if successful, or None if logging failed or is disabled.
    """
    if os.environ.get("ECOPULSE_DISABLE_MLFLOW", "").lower() in ("1", "true", "yes"):
        return None

    try:
        setup_tracking(experiment_name)

        with mlflow.start_run(run_name="waste-inference") as run:
            mlflow.set_tag("event_type", "inference")
            mlflow.set_tag("project", "EcoPulse AI")
            mlflow.set_tag("model", result.get("model", "YOLO11n"))

            mlflow.log_param(
                "checkpoint",
                str(result.get("checkpoint", "unknown"))
            )
            mlflow.log_metric(
                "total_detections",
                result.get("total_detections", 0)
            )

            if latitude is not None:
                mlflow.log_param("latitude", latitude)

            if longitude is not None:
                mlflow.log_param("longitude", longitude)

            if site_id:
                mlflow.set_tag("site_id", site_id)

            counts: dict[str, int] = {}
            for prediction in result.get("predictions", []):
                name = prediction["class_name"]
                counts[name] = counts.get(name, 0) + 1

            for class_name, count in counts.items():
                mlflow.log_metric(f"detections_{class_name}", count)

            mlflow.log_dict(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "result": result,
                    "latitude": latitude,
                    "longitude": longitude,
                    "site_id": site_id,
                },
                "prediction.json",
            )
            return run.info.run_id
    except Exception as exc:
        print(f"Warning: Failed to log prediction to MLflow: {exc}")
        return None
```

### `Dockerfile`
```dockerfile
# ============================================================
# EcoPulse AI - Waste Detection Production Container
# ============================================================

FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr and writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    MLFLOW_TRACKING_URI="sqlite:////app/mlflow.db"

# Install OS libraries required for OpenCV, image decoding, and healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

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

```

### `docker-compose.yml`
```yaml
# =============================================================================
# EcoPulse AI — Docker Compose Orchestration
# =============================================================================
# Services:
#   1. api     – FastAPI inference server (builds from local Dockerfile)
#   2. mlflow  – MLflow tracking server (lightweight python:3.11-slim image)
#
# Usage:
#   docker compose up -d          # start all services
#   docker compose logs -f api    # tail API logs
#   docker compose down           # stop and remove containers
# =============================================================================

version: "3.8"

services:
  # ---------------------------------------------------------------------------
  # MLflow Tracking Server
  # ---------------------------------------------------------------------------
  # Provides experiment tracking, model registry, and artifact storage.
  # The backend store uses a SQLite database mounted from the host, and
  # artifacts are persisted under ./mlruns on the host.
  # ---------------------------------------------------------------------------
  mlflow:
    image: python:3.11-slim
    container_name: ecopulse-mlflow
    ports:
      - "5000:5000"
    command: >
      bash -c "pip install mlflow>=2.15.0 &&
               mlflow server
                 --host 0.0.0.0
                 --port 5000
                 --backend-store-uri sqlite:///data/mlflow.db
                 --default-artifact-root /data/mlruns"
    volumes:
      # Shared SQLite database (host ↔ container)
      - ./mlflow.db:/data/mlflow.db
      # Persistent artifact storage
      - ./mlruns:/data/mlruns
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      start_period: 40s
      retries: 5

  # ---------------------------------------------------------------------------
  # EcoPulse AI — FastAPI Inference API
  # ---------------------------------------------------------------------------
  # Builds the application image from the project root Dockerfile and exposes
  # the prediction API on port 8000. Waits for the MLflow service to start
  # before launching so that tracking calls succeed on first request.
  # ---------------------------------------------------------------------------
  api:
    build:
      context: .
      dockerfile: Dockerfile
    image: ecopulse-ai:latest
    container_name: ecopulse-api
    ports:
      - "8000:8000"
    environment:
      # Point the SDK at the in-cluster MLflow server
      MLFLOW_TRACKING_URI: "http://mlflow:5000"
      # Enable MLflow integration
      ECOPULSE_DISABLE_MLFLOW: "false"
    volumes:
      # Share the same SQLite file so the API can read experiment data
      - ./mlflow.db:/app/mlflow.db
    depends_on:
      mlflow:
        condition: service_started
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      start_period: 30s
      retries: 3
```

### `.gitignore`
```gitignore

# Python
__pycache__/
*.py[cod]
.venv/
venv/
.env

# Dataset archives and raw data
*.zip
data/raw/
data/processed/
datasets/

# Model artifacts and training runs
*.pt
*.pth
*.onnx
runs/
yolo_runs/
mlruns/
models/

# Local caches
.cache/
*.cache

# Generated reports: retain code, avoid accidental large outputs
# Uncomment if you want reports tracked only selectively
# reports/*.json
# Local MLflow database
mlflow.db
mlflow.db-shm
mlflow.db-wal

# Training outputs
runs/
mlruns/

# Model checkpoints
*.pt
*.onnx

# Python cache
__pycache__/
*.pyc
.pytest_cache/
# Allow DVC metadata for project datasets
!artifacts/datasets/
!artifacts/datasets/*.dvc
!artifacts/datasets/.gitignore
```

### `.dockerignore`
```text
# Git and DVC versioning
.git/
.gitignore
.dvc/
.dvcignore
*.dvc

# Python environments and caches
venv/
.venv/
__pycache__/
*.py[cod]
*.pyo
.pytest_cache/

# Large datasets and archives
*.zip
*.tar.gz
data/
artifacts/datasets/
artifacts/features/

# Training runs and MLflow tracking
mlruns/
mlflow.db
mlflow.db-shm
mlflow.db-wal
runs/
yolo_runs/

# IDE and OS
.vscode/
.idea/
.DS_Store
Thumbs.db

```


## Git status at handover

Final repository status after creating this report:

```text
## master...origin/master
?? .github/
?? PROJECT_AUDIT_HANDOVER.md
?? docker-compose.yml
?? reports/model_risk_quality_report.json
?? reports/model_risk_quality_report.md
?? reports/model_test_report.json
```

No tracked edits or staged changes are present.

### Remaining problems

* DVC is unavailable in the current shell, so DVC status is unverified.
* No exact unittest run evidence is available.
* The project contains machine-specific data/MLflow paths.
* The old model evaluation report uses a different dataset path and non-comparable metrics.
* The Docker Compose MLflow health check calls `curl` in an image definition that does not install it.
* Training is configured for a smoke test; successful training provenance is not established.
