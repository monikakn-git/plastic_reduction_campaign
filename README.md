# EcoPulse AI – Plastic Waste Detection & Reduction using Computer Vision and MLOps

EcoPulse AI is an end-to-end MLOps computer vision system designed to detect, categorize, and quantify plastic waste from imagery to prioritize cleanup campaigns and monitor waste reduction.

---

## 1. Project Architecture

```
EcoPulse AI /
├── artifacts/
│   ├── checkpoints/                 # Trained model weights (preserved)
│   │   ├── yolo11n_best.pt          # YOLO11n fine-tuned checkpoint (5.25 MB)
│   │   └── yolov8n_best.pt          # YOLOv8n baseline checkpoint (6.20 MB)
│   ├── datasets/
│   │   └── processed/               # DVC-tracked 10-class YOLO dataset
│   │       ├── data.yaml            # Dataset split config & 10-class mapping
│   │       ├── images/{train,val,test}  # 13,064 processed images
│   │       └── labels/{train,val,test}  # 13,064 matching YOLO annotations
│   └── features/                    # Feature engineering tabular artifacts
│       └── engineered_features.csv  # 13,064 image & object feature rows
├── configs/
│   ├── data.yaml                    # Dataset paths and canonical classes
│   └── train.yaml                   # Model training and smoke-test config
├── pipelines/
│   └── training_pipeline.py         # MLOps pipeline validator & trainer
├── reports/
│   ├── figures/                     # EDA and class distribution plots
│   ├── mlflow_artifacts/            # Exported MLflow run artifacts
│   ├── yolo11n_best_test_evaluation.json/.md
│   └── yolov8n_best_test_evaluation.json/.md
├── src/
│   ├── api/app.py                   # FastAPI prediction service
│   ├── campaigns/                   # Spatial hotspot detection & campaign planning
│   ├── data/
│   │   └── validate_yolo_dataset.py # Automated YOLO dataset validator
│   ├── features/                    # Feature engineering pipeline
│   ├── models/
│   │   ├── evaluate.py              # Model evaluation & latency benchmark
│   │   ├── predict.py               # Single-image detection entrypoint
│   │   └── train.py                 # Training script with MLflow logging
│   ├── tracking/mlflow_utils.py     # Central MLflow tracking utility
│   └── visualization/               # Dataset & evaluation plotting
├── tests/                           # Automated test suite (unittest)
│   ├── test_api.py
│   ├── test_dataset.py
│   ├── test_models.py
│   └── test_tracking.py
├── Dockerfile                       # Production container specification
├── .dockerignore                    # Docker build context exclusions
├── dvc.yaml                         # Reproducible DVC pipeline definition
├── mlflow.db                        # SQLite MLflow experiment database
└── requirements.txt                 # Project Python dependencies
```

---

## 2. Dataset Overview

- **Location:** `artifacts/datasets/processed/`
- **Total Images:** 13,064
  - **Train:** 9,798 images (9,798 labels)
  - **Val:** 1,960 images (1,960 labels)
  - **Test:** 1,306 images (1,306 labels)
- **Bounding Boxes:** 79,101 valid annotations
- **Number of Classes:** 10 canonical classes:
  - `0: litter`
  - `1: pile`
  - `2: face_mask`
  - `3: trash_bin`
  - `4: plastic_bag`
  - `5: bottle`
  - `6: cup`
  - `7: rope`
  - `8: sachet`
  - `9: straw`

---

## 3. Quick Start & Execution Commands

### A. Environment Setup
```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### B. Validate Dataset
```powershell
python src/data/validate_yolo_dataset.py
```

### C. Run Automated Tests
```powershell
python -m unittest discover tests
```

### D. Single-Image Waste Inference
Run inference with the default YOLO11n checkpoint:
```powershell
python src/models/predict.py --image "artifacts/datasets/processed/images/test/ATT104_20200301_164523.jpg" --confidence 0.25 --save
```
Run inference with YOLOv8n:
```powershell
python src/models/predict.py --image "artifacts/datasets/processed/images/test/ATT104_20200301_164523.jpg" --checkpoint "artifacts/checkpoints/yolov8n_best.pt" --confidence 0.25
```

### E. Model Evaluation (Without Retraining)
```powershell
# Evaluate YOLO11n on test split
python src/models/evaluate.py --checkpoint artifacts/checkpoints/yolo11n_best.pt --split test

# Evaluate YOLOv8n on test split
python src/models/evaluate.py --checkpoint artifacts/checkpoints/yolov8n_best.pt --split test
```

### F. Run MLOps Pipeline Dry-Run
```powershell
python pipelines/training_pipeline.py
```

### G. Launch MLflow UI
```powershell
python -m mlflow ui --backend-store-uri "sqlite:///D:/mlops/mlflow.db" --host 127.0.0.1 --port 5000
```
Open `http://127.0.0.1:5000` in your browser. The main experiment is `ecopulse-waste-detection`.

### H. Launch FastAPI Service
```powershell
uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload
```
Interactive API docs are available at `http://127.0.0.1:8000/docs`.
Open `http://127.0.0.1:8000/ui` for the EcoPulse Detection Studio. Upload an image, choose a model and confidence threshold, and inspect or download the annotated result. Recent run summaries are stored in your browser.
If the MLflow registry is unreachable, the API falls back to local checkpoints in `artifacts/checkpoints/` so image inference can continue.

---

## 4. Docker Deployment

Build the container:
```bash
docker build -t ecopulse-ai:latest .
```

Run the container:
```bash
docker run -d -p 8000:8000 --name ecopulse-api ecopulse-ai:latest
```
Check health:
```bash
curl http://localhost:8000/health
```

### Compose deployment (API + MLflow)

The API expects checkpoints in `artifacts/checkpoints/`. Start the API and tracking server with:

```bash
cp .env.example .env
# Set a unique GRAFANA_ADMIN_PASSWORD in .env before starting.
docker compose up --build -d
docker compose ps
```

The API is available at `http://localhost:8000/docs`, MLflow at `http://localhost:5000`,
Prometheus at `http://localhost:9090`, and Grafana at `http://localhost:3000` (default local login `admin` / `change-me-before-exposure`). Services bind to localhost so the metrics and MLflow UI are not exposed directly on a cloud host. Set a strong `GRAFANA_ADMIN_PASSWORD` in `.env` before using outside a local machine. Set `ECOPULSE_MAX_UPLOAD_BYTES` on the API service to change the default 10 MiB upload limit.

### Model evaluation, risk gate, and registry promotion

Evaluate the checkpoint on the test split, run the risk checks, then register/promote it only if both reports pass:

```bash
python src/models/evaluate.py --checkpoint artifacts/checkpoints/yolo11n_best.pt --split test --no-mlflow
python src/models/risk_quality_checks.py --checkpoint yolo11n
docker compose up -d mlflow
python src/models/registry.py --tracking-uri http://localhost:5000 --name yolo11n --checkpoint artifacts/checkpoints/yolo11n_best.pt --evaluation reports/yolo11n_best_test_evaluation.json --risk reports/model_risk_quality_report.json --promote
```

The registry command must use the HTTP tracking server because this project configures MLflow artifact proxying in Docker Compose. A local SQLite tracking URI cannot write to the server's `mlflow-artifacts:/` artifact URI. The registry records a PyFunc model wrapping the YOLO checkpoint, evaluation metrics, gate tags, and a checkpoint SHA-256 digest. The API resolves the MLflow `production` alias, downloads its checkpoint artifact, and verifies the digest before loading it. See the [MLflow alias workflow](https://mlflow.org/docs/latest/ml/model-registry/workflow) for registry concepts.

The registration run stores the serving model/checkpoint, evaluation JSON and Markdown, dataset configuration and summary, per-class metrics, evaluation visualizations, runtime requirements/environment, model risk report, Deepchecks JSON/HTML output when available, and a SHA-256 artifact manifest. Training runs retain the best/last checkpoints, results CSV, training arguments, dataset/config files, standard training plots, installed package snapshot, and a reproducibility manifest. Dataset contents stay in DVC; MLflow stores the dataset configuration and summary used for evaluation. For a model already registered, append the latest evidence to its current aliased run without creating another model version:

```bash
python src/models/registry.py --tracking-uri http://localhost:5000 --name yolo11n --checkpoint artifacts/checkpoints/yolo11n_best.pt --evaluation reports/yolo11n_best_test_evaluation.json --risk reports/model_risk_quality_report.json --backfill-existing
```

### Model quality, visual review, and production drift

Install the offline quality dependencies for Deepchecks and Evidently:

```bash
pip install -r requirements-quality.txt
```

Install FiftyOne and Supervision when you want an interactive visual review:

```bash
pip install -r requirements-vision-tools.txt
python src/visualization/inspect_with_fiftyone.py --checkpoint artifacts/checkpoints/yolo11n_best.pt --split test --limit 100
```

Pytest runs the automated checks with `python -m pytest`. The risk gate runs Deepchecks image/label integrity checks and writes JSON/HTML artifacts under `reports/`.

For production drift, generate a reference observation set from the held-out test images, then compare it with the privacy-safe feature summaries the API writes to `monitoring/inference_observations.csv`:

```bash
python src/monitoring/build_baseline.py --checkpoint artifacts/checkpoints/yolo11n_best.pt --limit 300
python src/monitoring/drift.py --reference reports/baseline_inference.csv --current monitoring/inference_observations.csv
```

The API also exposes Prometheus request, inference-latency, and class-detection metrics at `/metrics`; Compose provisions a Grafana dashboard for them.

### Render deployment

The Render Blueprint creates two services: the EcoPulse API/UI and a login-protected MLflow tracking server. The MLflow service stores its SQLite metadata database and artifacts on a 10 GB persistent disk; a persistent disk requires a paid Render service and is limited to one instance. Both services must be in the same Render region. The API uses Render's private network to send prediction runs to MLflow, and the MLflow dashboard is available at the MLflow service's public `onrender.com` URL.

Before applying the Blueprint, push these files to the Git branch connected to Render. In Render, create/update the Blueprint and set `WASTE_MODEL_URL` on the API service to the direct GitHub Release asset URL for `yolo11n_best.pt` (the URL ending in `/releases/download/<tag>/yolo11n_best.pt`, not the release-page URL). The Blueprint verifies this file against `WASTE_MODEL_SHA256`. If you replace the checkpoint, update both the asset URL and SHA-256 in `render.yaml`. The shared environment group generates the MLflow password; after deployment, open the MLflow service URL and sign in with username `admin` and that generated `MLFLOW_TRACKING_PASSWORD` value. Use the same credentials for any MLflow client outside the API by setting `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD`.

The detection UI is at the API service root URL. It fetches `/api` on load and shows only checkpoints actually present on the service. YOLO11n is the default Render model. Add and configure a YOLOv8n checkpoint separately before that option appears. MLflow run logging is enabled; if MLflow is temporarily unavailable, inference still returns and the API logs a warning. Configure a persistent disk for MLflow and do not delete it if you need to retain run history and artifacts.

### Jenkins CI/CD

The root `Jenkinsfile` replaces the GitHub Actions workflow. Configure a Jenkins Multibranch Pipeline for this repository on a Linux agent with Python 3.11 and Docker available. It runs lint and syntax checks, unit tests, optional dataset and model-quality checks, then builds both Docker images. On `main` or `master`, Jenkins triggers Render deploy hooks only after those stages pass. Render auto-deploy is disabled in the Blueprint so a push cannot deploy before Jenkins finishes.

Create two Render deploy hooks, one for the API service and one for `plastic-reduction-mlflow`. In Jenkins, add each hook as a **Secret text** credential with these exact IDs: `render-api-deploy-hook` and `render-mlflow-deploy-hook`. Keep the URLs out of the repository. Connect GitHub push/branch events to the Jenkins Multibranch Pipeline so pushes start the pipeline. The Jenkins agent needs permission to access Docker and enough disk space to build the API image.

---

## 5. Model Performance Summary (Test Split: 1,306 images)

| Metric | YOLO11n (`yolo11n_best.pt`) | YOLOv8n (`yolov8n_best.pt`) | Threshold / Gate |
|---|---|---|---|
| **File Size** | 5.25 MB | 6.20 MB | Edge deployable |
| **mAP50** | 0.2721 | 0.2761 | >= 0.20 (both PASS) |
| **mAP50-95** | 0.1652 | 0.1675 | Informational (YOLOv8n higher) |
| **Precision** | 0.3690 | 0.3626 | Informational |
| **Recall** | 0.3153 | 0.3296 | Informational (YOLOv8n higher) |
| **Mean Latency (CPU)** | 204.11 ms | 219.72 ms | <= 500 ms (PASS) |
| **Quality Gate** | **PASSED** | **PASSED** | Both exceed mAP50 and latency thresholds |

YOLO11n passed because its test report records mAP50 0.2721 (minimum 0.20) and mean CPU latency 204.11 ms (maximum 500 ms). The separate risk report also records passing precision, recall, p95 latency, model-size, robustness, and Deepchecks checks. This is a pass against configured thresholds; it does not establish that YOLO11n outperforms the YOLOv8n baseline. In these reports, YOLOv8n has slightly better mAP50 (0.2761), mAP50–95 (0.1675), and recall (0.3296). The robustness test covered only five images, so that result is a limited signal.

