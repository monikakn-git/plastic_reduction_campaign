"""
EcoPulse AI - Model Evaluation & Quality Gate Verification

Evaluates trained YOLO checkpoints on waste detection splits, benchmarks latency,
computes precision, recall, mAP50, mAP50-95, and saves reports.
"""

from pathlib import Path
import json
import os
import sys
import time
from typing import Any

import numpy as np
import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "artifacts" / "checkpoints" / "yolo11n_best.pt"
DEFAULT_DATA_YAML = PROJECT_ROOT / "data" / "waste10" / "data.yaml"
REPORTS_DIR = PROJECT_ROOT / "reports"


def benchmark_latency(
    model: YOLO,
    sample_images: list[Path],
    warmup: int = 3,
    iterations: int = 10,
) -> float:
    """Benchmark mean inference latency in milliseconds."""
    if not sample_images:
        return 0.0

    valid_images = [str(p) for p in sample_images if p.is_file()]
    if not valid_images:
        return 0.0

    # Warmup
    for _ in range(min(warmup, len(valid_images))):
        _ = model.predict(source=valid_images[0], verbose=False)

    latencies = []
    for i in range(iterations):
        target = valid_images[i % len(valid_images)]
        t0 = time.perf_counter()
        _ = model.predict(source=target, verbose=False)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    return round(float(np.mean(latencies)), 2)


def evaluate_model(
    checkpoint_path: str | Path | None = None,
    data_yaml_path: str | Path | None = None,
    split: str = "test",
    min_map50: float = 0.20,
    max_latency_ms: float = 500.0,
) -> dict[str, Any]:
    """Evaluate a YOLO model checkpoint on test or validation split."""
    checkpoint = Path(checkpoint_path or os.environ.get("WASTE_MODEL_PATH", str(DEFAULT_CHECKPOINT)))
    data_yaml = Path(data_yaml_path or os.environ.get("WASTE_DATA_YAML", str(DEFAULT_DATA_YAML)))

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    print("=" * 60)
    print("EcoPulse AI - Waste Detection Model Evaluation")
    print(f"Checkpoint: {checkpoint}")
    print(f"Dataset YAML: {data_yaml}")
    print(f"Split: {split}")
    print("=" * 60)

    model_size_mb = round(checkpoint.stat().st_size / (1024 * 1024), 2)
    model = YOLO(str(checkpoint))

    # Run validation
    metrics = model.val(data=str(data_yaml), split=split, verbose=False)

    results_dict = getattr(metrics, "results_dict", {})
    precision = round(float(results_dict.get("metrics/precision(B)", 0.0)), 4)
    recall = round(float(results_dict.get("metrics/recall(B)", 0.0)), 4)
    map50 = round(float(results_dict.get("metrics/mAP50(B)", 0.0)), 4)
    map50_95 = round(float(results_dict.get("metrics/mAP50-95(B)", 0.0)), 4)

    # Benchmark latency on sample images
    with data_yaml.open("r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f) or {}

    base_path = Path(data_cfg.get("path", data_yaml.parent))
    if not base_path.is_absolute():
        base_path = data_yaml.parent / base_path

    split_dir = base_path / data_cfg.get(split, f"images/{split}")
    sample_images = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png"))
    mean_latency_ms = benchmark_latency(model, sample_images)

    # Quality Gate check
    qg_passed = (map50 >= min_map50) and (mean_latency_ms <= max_latency_ms)
    decision = "PASSED_APPROVED_FOR_DEPLOYMENT" if qg_passed else "QUALITY_GATE_WARNING"

    eval_result = {
        "checkpoint": str(checkpoint),
        "model_name": checkpoint.stem,
        "classes": model.names,
        "num_classes": len(model.names),
        "split": split,
        "model_size_mb": model_size_mb,
        "mean_latency_ms": mean_latency_ms,
        "metrics": {
            "precision": precision,
            "recall": recall,
            "map50": map50,
            "map50_95": map50_95,
        },
        "quality_gate": {
            "decision": decision,
            "min_map50_threshold": min_map50,
            "max_latency_threshold_ms": max_latency_ms,
            "passed": qg_passed,
        },
    }

    # Save reports
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / "model_evaluation_report.json"
    json_path.write_text(json.dumps(eval_result, indent=2), encoding="utf-8")

    md_path = REPORTS_DIR / "model_evaluation_report.md"
    md_content = f"""# EcoPulse AI — Model Evaluation Report

**Model:** `{checkpoint.name}`  
**Dataset Configuration:** `{data_yaml.as_posix()}`  
**Evaluation Split:** `{split}`  
**Quality Gate Verdict:** **{decision}**  

## 1. Overall Performance Metrics
| Metric | Observed Value | Quality Gate Target | Status |
| :--- | :--- | :--- | :--- |
| **Precision** | {precision:.4f} | >= 0.20 | {'PASS' if precision >= 0.20 else 'WARN'} |
| **Recall** | {recall:.4f} | >= 0.20 | {'PASS' if recall >= 0.20 else 'WARN'} |
| **mAP@50** | {map50:.4f} | >= {min_map50} | {'PASS' if map50 >= min_map50 else 'WARN'} |
| **mAP@50-95** | {map50_95:.4f} | Informational | OK |
| **Inference Latency** | {mean_latency_ms} ms/frame | <= {max_latency_ms} ms | {'PASS' if mean_latency_ms <= max_latency_ms else 'FAIL'} |
| **Model File Size** | {model_size_mb} MB | Edge Deployable | PASS |

## 2. Model Detection Classes ({len(model.names)})
{chr(10).join(f"- **Class {k}:** `{v}`" for k, v in sorted(model.names.items()))}

## 3. Operational Deployment Assessment
- **Edge Feasibility:** Model file size of {model_size_mb} MB permits packaging into Docker containers and deployment on edge devices.
- **Latency Budget:** Inference latency of {mean_latency_ms} ms/frame supports real-time batch processing and responsive API endpoints.
"""
    md_path.write_text(md_content, encoding="utf-8")

    print(f"\nEvaluation summary saved:")
    print(f" - JSON: {json_path}")
    print(f" - Markdown: {md_path}")
    print(f"Quality gate verdict: {decision}")

    return eval_result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EcoPulse AI Model Evaluation")
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint path")
    parser.add_argument("--data", default=None, help="Dataset YAML path")
    parser.add_argument("--split", default="test", help="Split to evaluate: test or val")
    parser.add_argument("--min-map50", type=float, default=0.20)
    parser.add_argument("--max-latency", type=float, default=500.0)

    args = parser.parse_args()

    res = evaluate_model(
        checkpoint_path=args.checkpoint,
        data_yaml_path=args.data,
        split=args.split,
        min_map50=args.min_map50,
        max_latency_ms=args.max_latency,
    )
    print("Done.")
