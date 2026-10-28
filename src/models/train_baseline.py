"""
Baseline Model Training & MLflow Experiment Tracking
Fulfills Step 5 & Step 6 of the Implementation Plan:
- Trains pretrained YOLO object detection model on pLitterStreet
- Integrates MLflow experiment tracking
- Logs Git commit, dataset version, hyperparameters, and evaluation metrics
- Saves best model artifacts and performance plots
"""

import os
import time
import subprocess
from pathlib import Path
import yaml
import mlflow
from ultralytics import YOLO


def load_params(params_path="params.yaml"):
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def get_git_commit_hash():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
        return commit
    except Exception:
        return "git-unavailable"


def get_dvc_dataset_version():
    dvc_file = Path("data/raw.dvc")
    if dvc_file.exists():
        try:
            with open(dvc_file, "r") as f:
                dvc_info = yaml.safe_load(f)
                return dvc_info.get("outs", [{}])[0].get("md5", "unknown-dvc-md5")
        except Exception:
            return "dvc-tracked"
    return "unversioned-local"


def train_baseline(params_path="params.yaml", custom_epochs=None, custom_batch=None):
    params = load_params(params_path)
    
    # Paths & configs
    dataset_yaml = Path(params["dataset"]["processed_dir"]) / "dataset.yaml"
    model_type = params["train"]["model_type"]
    img_size = params["train"]["image_size"]
    batch_size = custom_batch or params["train"]["batch_size"]
    epochs = custom_epochs or params["train"]["epochs"]
    seed = params["base"]["random_seed"]
    device = params["train"].get("device", "cpu")

    git_commit = get_git_commit_hash()
    dvc_version = get_dvc_dataset_version()

    print("=" * 60)
    print("AI-BASED WASTE REDUCTION CAMPAIGN - BASELINE TRAINING")
    print(f"Model Backbone: {model_type}")
    print(f"Dataset Config: {dataset_yaml}")
    print(f"Epochs: {epochs} | Batch: {batch_size} | Image Size: {img_size}")
    print(f"Git Commit: {git_commit} | DVC Version: {dvc_version}")
    print("=" * 60)

    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"Processed dataset config not found at '{dataset_yaml}'. "
            "Please run 'src/data/preprocess_coco_to_yolo.py' first!"
        )

    # MLflow Setup
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    run_name = f"YOLOv8_Baseline_E{epochs}_B{batch_size}_{time.strftime('%Y%m%d-%H%M%S')}"

    with mlflow.start_run(run_name=run_name):
        # 1. Log Hyperparameters & Metadata
        mlflow.log_params({
            "model_type": model_type,
            "image_size": img_size,
            "batch_size": batch_size,
            "epochs": epochs,
            "optimizer": params["train"]["optimizer"],
            "learning_rate": params["train"]["lr0"],
            "random_seed": seed,
            "git_commit": git_commit,
            "dvc_dataset_version": dvc_version,
            "target_classes_count": len(params["dataset"]["classes"]),
        })

        # 2. Initialize YOLO Model
        print(f"[*] Loading pretrained weights: {model_type}")
        model = YOLO(model_type)

        # 3. Train Model
        start_time = time.time()
        results = model.train(
            data=str(dataset_yaml),
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            seed=seed,
            device=device,
            project="runs/detect",
            name=run_name,
            exist_ok=True,
            verbose=True,
        )
        training_duration = round(time.time() - start_time, 2)

        # 4. Extract Validation Metrics
        # YOLO saves best metrics to results.box
        metrics_dict = {
            "training_duration_seconds": training_duration,
            "precision": round(float(results.results_dict.get("metrics/precision(B)", 0.0)), 4),
            "recall": round(float(results.results_dict.get("metrics/recall(B)", 0.0)), 4),
            "mAP50": round(float(results.results_dict.get("metrics/mAP50(B)", 0.0)), 4),
            "mAP50_95": round(float(results.results_dict.get("metrics/mAP50-95(B)", 0.0)), 4),
        }

        print("\n" + "=" * 40)
        print("TRAINING COMPLETED - METRICS SUMMARY")
        for k, v in metrics_dict.items():
            print(f"  {k}: {v}")
            mlflow.log_metric(k, v)
        print("=" * 40)

        # 5. Log Artifacts
        save_dir = Path(f"runs/detect/{run_name}")
        best_weights = save_dir / "weights" / "best.pt"

        if best_weights.exists():
            os.makedirs("models", exist_ok=True)
            target_model_path = "models/baseline_yolov8.pt"
            import shutil
            shutil.copy2(best_weights, target_model_path)
            mlflow.log_artifact(str(best_weights), artifact_path="weights")
            print(f"[✓] Saved model artifact to: {target_model_path}")

        # Log training curves and confusion matrix if generated
        for plot_name in ["confusion_matrix.png", "results.png", "PR_curve.png", "F1_curve.png"]:
            plot_file = save_dir / plot_name
            if plot_file.exists():
                mlflow.log_artifact(str(plot_file), artifact_path="plots")

        print(f"[✓] MLflow Run ID: {mlflow.active_run().info.run_id}")
        print("[✓] View MLflow dashboard by executing: mlflow ui")

        return metrics_dict


if __name__ == "__main__":
    train_baseline()
