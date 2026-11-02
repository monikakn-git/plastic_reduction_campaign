from pathlib import Path

import mlflow
import pandas as pd
import yaml


# ---------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------
PROJECT_DIR = Path(r"D:\mlops")
DATASET_DIR = Path(r"G:\My Drive\datasets\Ai_Waste_Campaign")
TRACKING_DB = PROJECT_DIR / "mlflow.db"

EXPERIMENT_NAME = "ecopulse-model-comparison"

# These are the existing training runs. This script only reads their
# results.csv / args.yaml files; it does NOT train or modify either model.
MODELS = [
    {
        "name": "YOLOv8n",
        "run_name": "YOLOv8n-baseline-comparison",
        "run_dir": DATASET_DIR / "yolo_runs" / "yolov8n_baseline",
        "checkpoint": (
            DATASET_DIR / "yolo_runs" / "yolov8n_baseline"
            / "weights" / "best.pt"
        ),
    },
    {
        "name": "YOLO11n",
        "run_name": "YOLO11n-waste-detection-comparison",
        "run_dir": DATASET_DIR / "yolo_runs" / "yolo11n_waste_detection",
        "checkpoint": (
            DATASET_DIR / "yolo_runs" / "yolo11n_waste_detection"
            / "weights" / "best.pt"
        ),
    },
]


def configure_mlflow():
    """Configure MLflow to use the SQLite tracking database."""
    TRACKING_DB.parent.mkdir(parents=True, exist_ok=True)

    tracking_uri = f"sqlite:///{TRACKING_DB.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"MLflow experiment: {EXPERIMENT_NAME}")


def load_training_files(model):
    """Load existing training results and configuration for one model."""
    results_file = model["run_dir"] / "results.csv"
    args_file = model["run_dir"] / "args.yaml"
    checkpoint_file = model["checkpoint"]

    for required_file in (results_file, args_file, checkpoint_file):
        if not required_file.is_file():
            raise FileNotFoundError(f"Required file not found: {required_file}")

    results = pd.read_csv(results_file)
    results.columns = [str(column).strip() for column in results.columns]

    if results.empty:
        raise ValueError(f"No training metric rows found in: {results_file}")

    with args_file.open("r", encoding="utf-8") as file:
        args = yaml.safe_load(file) or {}

    if not isinstance(args, dict):
        args = {}

    return results, args, results_file, args_file


def numeric_value(row, column):
    """Return a finite numeric value, or None when unavailable."""
    if column not in row.index:
        return None

    value = pd.to_numeric(row[column], errors="coerce")
    if pd.isna(value):
        return None

    return float(value)


def log_model_comparison(model):
    """Log one existing model's training history as an MLflow run."""
    results, args, results_file, args_file = load_training_files(model)

    with mlflow.start_run(run_name=model["run_name"]):
        mlflow.set_tag("model_family", model["name"])
        mlflow.set_tag("comparison_type", "existing_training_history")
        mlflow.set_tag("source", "Google Drive existing YOLO training run")

        mlflow.log_param("checkpoint_path", str(model["checkpoint"]))
        mlflow.log_param("source_run_directory", str(model["run_dir"]))

        # Log selected training configuration values when available.
        for key in ("model", "epochs", "imgsz", "batch", "device", "optimizer"):
            value = args.get(key)
            if value is not None and isinstance(value, (str, int, float, bool)):
                mlflow.log_param(f"train_{key}", value)

        # Log epoch-by-epoch metrics so MLflow can display training curves.
        metric_columns = [
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
            "train/box_loss",
            "train/cls_loss",
            "train/dfl_loss",
            "val/box_loss",
            "val/cls_loss",
            "val/dfl_loss",
        ]

        for row_index, row in results.iterrows():
            epoch_value = numeric_value(row, "epoch")
            step = int(epoch_value) if epoch_value is not None else int(row_index)

            epoch_metrics = {}
            for column in metric_columns:
                value = numeric_value(row, column)
                if value is not None:
                    metric_name = column.replace("/", "_").replace("(B)", "")
                    epoch_metrics[metric_name] = value

            if epoch_metrics:
                mlflow.log_metrics(epoch_metrics, step=step)

        # Log final-epoch validation metrics with easy-to-compare names.
        final_row = results.iloc[-1]
        final_metric_map = {
            "final_precision": "metrics/precision(B)",
            "final_recall": "metrics/recall(B)",
            "final_map50": "metrics/mAP50(B)",
            "final_map50_95": "metrics/mAP50-95(B)",
        }

        for mlflow_name, csv_column in final_metric_map.items():
            value = numeric_value(final_row, csv_column)
            if value is not None:
                mlflow.log_metric(mlflow_name, value)

        # Record the maximum observed value in the existing training history.
        best_metric_map = {
            "best_precision": "metrics/precision(B)",
            "best_recall": "metrics/recall(B)",
            "best_map50": "metrics/mAP50(B)",
            "best_map50_95": "metrics/mAP50-95(B)",
        }

        for mlflow_name, csv_column in best_metric_map.items():
            if csv_column in results.columns:
                values = pd.to_numeric(
                    results[csv_column], errors="coerce"
                ).dropna()

                if not values.empty:
                    mlflow.log_metric(mlflow_name, float(values.max()))

        # Ultralytics results.csv records "time" as elapsed training time in hours.
        elapsed_hours = numeric_value(final_row, "time")
        if elapsed_hours is not None:
            mlflow.log_metric("training_time_hours", elapsed_hours)
            mlflow.log_metric("training_time_minutes", elapsed_hours * 60.0)

        # Attach the original training evidence to this MLflow run.
        mlflow.log_artifact(str(results_file), artifact_path="training_source")
        mlflow.log_artifact(str(args_file), artifact_path="training_source")

    print(
        f"Logged {model['name']}: {len(results)} epochs "
        f"(checkpoint: {model['checkpoint']})"
    )


def main():
    configure_mlflow()

    for model in MODELS:
        log_model_comparison(model)

    print("\nModel comparison logging completed successfully.")
    print(f"Open MLflow and select experiment: {EXPERIMENT_NAME}")


if __name__ == "__main__":
    main()
