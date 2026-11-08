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

---

## 5. Model Performance Summary (Test Split: 1,306 images)

| Metric | YOLO11n (`yolo11n_best.pt`) | YOLOv8n (`yolov8n_best.pt`) | Threshold / Gate |
|---|---|---|---|
| **File Size** | 5.25 MB | 6.20 MB | Edge deployable |
| **mAP50** | 0.2721 | 0.2761 | >= 0.20 (PASS) |
| **mAP50-95** | 0.1652 | 0.1675 | Informational |
| **Precision** | 0.3690 | 0.3626 | Informational |
| **Recall** | 0.3153 | 0.3296 | Informational |
| **Mean Latency (CPU)** | 215.07 ms | 219.72 ms | <= 500 ms (PASS) |
| **Quality Gate** | **PASSED** | **PASSED** | PASSED |

