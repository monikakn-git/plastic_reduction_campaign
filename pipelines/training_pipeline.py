from pathlib import Path
import os
import sys

import mlflow
import yaml
from ultralytics import YOLO

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "train.yaml"
DEFAULT_DATA_YAML = PROJECT_ROOT / "data" / "waste10" / "data.yaml"


def resolve_data_yaml(config: dict | None = None) -> Path:
    """Resolve the dataset YAML configuration path."""
    env_yaml = os.environ.get("WASTE_DATA_YAML")
    if env_yaml:
        p = Path(env_yaml)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p

    if config and "dataset" in config and "path" in config["dataset"]:
        p = Path(config["dataset"]["path"])
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.is_file():
            return p

    if DEFAULT_DATA_YAML.is_file():
        return DEFAULT_DATA_YAML

    data_root = Path(
        os.environ.get(
            "WASTE_DATA_DIR",
            r"D:\datasets\Ai_Waste_Campaign",
        )
    )
    return data_root / "yolo_dataset" / "dataset.yaml"


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


def validate_paths(data_yaml_path: Path) -> dict:
    if not data_yaml_path.is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_yaml_path}\n"
            "Ensure data/waste10/data.yaml exists or configure configs/train.yaml."
        )

    with data_yaml_path.open("r", encoding="utf-8") as file:
        dataset = yaml.safe_load(file)

    base_path = Path(dataset.get("path", data_yaml_path.parent))
    if not base_path.is_absolute():
        base_path = data_yaml_path.parent / base_path

    for split in ("train", "val", "test"):
        if split not in dataset:
            raise ValueError(
                f"Dataset YAML is missing the '{split}' entry."
            )

        split_path = base_path / dataset[split]
        if not split_path.is_dir():
            raise FileNotFoundError(
                f"Dataset split '{split}' not found: {split_path}"
            )

    print(f"Dataset configuration: {data_yaml_path}")
    print(f"Dataset base path: {base_path}")
    return dataset


def train(config: dict, data_yaml_path: Path) -> None:
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

    if tracking.get("enabled", False):
        tracking_db = PROJECT_ROOT / "mlflow.db"
        mlflow.set_tracking_uri(f"sqlite:///{tracking_db.as_posix()}")
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
                "dataset": str(data_yaml_path),
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
        data_yaml_path = resolve_data_yaml(config)
        validate_paths(data_yaml_path)
        # Note: per safety instructions, do not execute full training by default
        print("Pipeline path validation and configuration check passed.")
        return 0

    except Exception as exc:
        print(f"\nTRAINING PIPELINE CHECK FAILED: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
