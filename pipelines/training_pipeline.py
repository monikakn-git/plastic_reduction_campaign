
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
