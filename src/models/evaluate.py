
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
