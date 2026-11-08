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
