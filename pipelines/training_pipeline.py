
from pathlib import Path
import os
import sys

import mlflow
import yaml
from ultralytics import YOLO

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "train.yaml"

DATA_ROOT = Path(
    os.environ.get(
        "WASTE_DATA_DIR",
        r"D:\datasets\Ai_Waste_Campaign",
    )
)

DATA_YAML = DATA_ROOT / "yolo_dataset" / "dataset.yaml"


def load_config() -> dict:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"Training configuration not found: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Training configuration must be a YAML mapping.")

    for section in ("model", "training", "output"):
        if section not in config:
            raise ValueError(f"Missing '{section}' section in {CONFIG_PATH}")

    return config


def validate_paths() -> None:
    if not DATA_YAML.is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found: {DATA_YAML}\n"
            "Run src/data/prepare_yolo_dataset.py first."
        )

    with DATA_YAML.open("r", encoding="utf-8") as file:
        dataset = yaml.safe_load(file)

    for split in ("train", "val", "test"):
        if split not in dataset:
            raise ValueError(
                f"Dataset YAML is missing the '{split}' entry."
            )

        split_path = Path(dataset["path"]) / dataset[split]
        if not split_path.is_dir():
            raise FileNotFoundError(
                f"Dataset split '{split}' not found: {split_path}"
            )

    print(f"Dataset configuration: {DATA_YAML}")
    print(f"Dataset root: {DATA_ROOT}")


def train(config: dict) -> None:
    training = config["training"]
    output = config["output"]
    tracking = config.get("tracking", {})

    model_name = config["model"].get("name", "yolov8n.yaml")

    # yolov8n.yaml creates the architecture locally without downloading
    # pretrained weights.
    print(f"Loading model architecture: {model_name}")
    model = YOLO(model_name)

    output_dir = PROJECT_ROOT / output.get("project", "runs/detect")
    run_name = output.get("name", "ecopulse_waste_smoke_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_args = {
        "data": str(DATA_YAML),
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

    if tracking.get("enabled", False):
        mlflow.set_experiment(
            tracking.get(
                "experiment_name",
                "ecopulse-waste-detection",
            )
        )

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "model": model_name,
                "epochs": train_args["epochs"],
                "image_size": train_args["imgsz"],
                "batch_size": train_args["batch"],
                "device": train_args["device"],
                "pretrained": False,
                "dataset": str(DATA_YAML),
            })

            results = model.train(**train_args)

            save_dir = Path(results.save_dir)
            best_weights = save_dir / "weights" / "best.pt"

            if best_weights.is_file():
                mlflow.log_artifact(
                    str(best_weights),
                    artifact_path="model",
                )

            results_csv = save_dir / "results.csv"
            if results_csv.is_file():
                mlflow.log_artifact(
                    str(results_csv),
                    artifact_path="training_results",
                )

            print(f"\nMLflow run ID: {mlflow.active_run().info.run_id}")

    else:
        results = model.train(**train_args)

    print("\nTraining run finished.")
    print(f"Output directory: {results.save_dir}")
    print(f"Best checkpoint: {Path(results.save_dir) / 'weights' / 'best.pt'}")


def main() -> int:
    try:
        config = load_config()
        validate_paths()
        train(config)
        return 0

    except Exception as exc:
        print(f"\nTRAINING FAILED: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
