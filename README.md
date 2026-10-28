# AI-Based Waste Reduction Campaign (MLOps Pipeline)

An end-to-end, production-grade MLOps system that detects street waste using computer vision (YOLO), tracks experiments with MLflow, versions datasets with DVC, serves predictions via FastAPI, and drives targeted municipal waste reduction and cleanup campaigns.

---

## 🏗️ Architecture & Implementation Mapping

| Phase | Step | Implementation File | Purpose |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **Step 1: Dataset Verification** | [`src/data/verify_dataset.py`](file:///d:/mlops/src/data/verify_dataset.py) | Validates image integrity, bounding boxes, duplicates, and generates Dataset Approval Report |
| | **Step 2: Git Repository Setup** | [`.gitignore`](file:///d:/mlops/.gitignore) | Modular repository layout; excludes large data and weights |
| | **Step 3: DVC Dataset Versioning** | [`dvc.yaml`](file:///d:/mlops/dvc.yaml) | Tracks raw/processed data and builds reproducible DAG pipeline |
| | **Step 4: EDA & YOLO Converter** | [`src/data/preprocess_coco_to_yolo.py`](file:///d:/mlops/src/data/preprocess_coco_to_yolo.py) | Converts COCO to YOLO, splits Train/Val/Test (70/15/15), outputs EDA report |
| | **Step 5: Baseline YOLO Training** | [`src/models/train_baseline.py`](file:///d:/mlops/src/models/train_baseline.py) | Fine-tunes YOLO on pLitterStreet dataset |
| | **Step 6: MLflow Experiment Tracking** | [`src/models/train_baseline.py`](file:///d:/mlops/src/models/train_baseline.py) | Logs params, metrics (mAP50, precision, recall), Git commit, and artifacts |
| | **Step 7 & 9: Model Evaluation & Gates**| [`src/models/evaluate.py`](file:///d:/mlops/src/models/evaluate.py) | Benchmarks latency (ms), model size, and enforces Quality Gate criteria |
| **Phase 2** | **Step 8: Automated Testing** | [`tests/test_pipeline.py`](file:///d:/mlops/tests/test_pipeline.py) | Pytest suite testing model, endpoints, corrupted files, and optimizer |
| | **Step 10: Model Registry** | [`src/models/model_registry.py`](file:///d:/mlops/src/models/model_registry.py) | Registers models to MLflow Model Registry and manages stage promotions |
| | **Step 11: FastAPI Inference Service** | [`src/api/main.py`](file:///d:/mlops/src/api/main.py) | Endpoints `/health`, `/ready`, `/predict`, `/metrics` |
| | **Step 12: Docker Containerization** | [`Dockerfile`](file:///d:/mlops/Dockerfile) | Production Docker container with healthchecks |
| | **Step 13: CI/CD Pipeline** | [`.github/workflows/ci_cd.yaml`](file:///d:/mlops/.github/workflows/ci_cd.yaml) | GitHub Actions workflow for linting, pytest, and Docker builds |
| | **Step 14: Monitoring** | [`src/api/main.py`](file:///d:/mlops/src/api/main.py) | Tracks latency, request counts, and class drift via `/metrics` |
| | **Step 15: Campaign Hotspot Optimizer** | [`src/campaign/hotspot_optimizer.py`](file:///d:/mlops/src/campaign/hotspot_optimizer.py) | Clusters GPS reports, computes Waste Severity Index, and optimizes budget |

---

## 🚀 Quickstart Guide

### 1. Place Your Downloaded Dataset
Extract or copy your **pLitterStreet** dataset (downloaded from Google Drive) into:
```text
d:\mlops\data\raw\pLitterStreet\
  ├── annotations.json      <- COCO annotations file
  └── images\               <- All image files (.jpg / .png)
```

### 2. Verify Dataset Integrity & Generate Approval Report (Step 1)
```powershell
python src/data/verify_dataset.py
```
*Outputs: `reports/dataset_approval_report.md` and `reports/dataset_verification.json`.*

### 3. Track Dataset with DVC (Step 3)
```powershell
dvc add data/raw/pLitterStreet
git add data/raw/pLitterStreet.dvc .gitignore
git commit -m "feat(data): track raw pLitterStreet dataset via DVC"
```

### 4. Preprocess Annotations & Generate Splits (Step 4)
```powershell
python src/data/preprocess_coco_to_yolo.py
```
*Generates YOLO annotations, train/val/test splits, `dataset.yaml`, and `reports/eda_report.md`.*

### 5. Train Baseline Model with MLflow Tracking (Step 5 & 6)
```powershell
python src/models/train_baseline.py
```
To launch the interactive MLflow experiment dashboard:
```powershell
mlflow ui
```
*(Open http://localhost:5000 in your browser to view metrics, parameters, and model artifacts).*

### 6. Evaluate Model & Run Quality Gate (Step 7 & 9)
```powershell
python src/models/evaluate.py
```
*Generates: `reports/model_evaluation_report.md`.*

### 7. Run Test Suite (Step 8)
```powershell
pytest tests/ -v
```

### 8. Run FastAPI Inference Service (Step 11 & 14)
```powershell
uvicorn src.api.main:app --reload --port 8000
```
Interactive Swagger API docs available at: **http://localhost:8000/docs**

### 9. Run Campaign Hotspot Optimizer (Step 15)
```powershell
python src/campaign/hotspot_optimizer.py
```
