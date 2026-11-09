"""Validation-only error analysis and an optional controlled resolution experiment."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".local/ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".local/matplotlib"))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import numpy as np
import yaml


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, choices=[960], help="Evaluate the same weights on validation only")
    args = parser.parse_args()
    os.chdir(ROOT)
    config = yaml.safe_load((ROOT / "params.yaml").read_text())
    name = config["dataset"]["name"]
    report_dir = ROOT / "reports/phase1"
    baseline = json.loads((report_dir / "evaluation_val.json").read_text())
    training = json.loads((report_dir / "baseline_metrics.json").read_text())
    weights = ROOT / training["model"]
    if sha(weights) != training["model_sha256"]:
        raise SystemExit("Baseline weight hash mismatch")
    splits = ROOT / f"data/prepared/{name}/splits.json"
    split_ids = {row["image_id"]: row["split"] for row in json.loads(splits.read_text())}
    annotations_path = ROOT / f"data/raw/{name}/annotations.json"
    data = json.loads(annotations_path.read_text())
    images = {row["id"]: row for row in data["images"]}
    sizes = defaultdict(list)
    for row in data["annotations"]:
        split = split_ids[row["image_id"]]
        if split not in ("train", "val"):
            continue  # Test labels and predictions never guide this experiment.
        image = images[row["image_id"]]
        scale = 640 / max(image["width"], image["height"])
        _, _, w, h = row["bbox"]
        sizes[split, row["category_id"]].append((w * scale, h * scale))
    stats = []
    for category in data["categories"]:
        item = {"class": category["name"].replace(" ", "_")}
        for split in ("train", "val"):
            a = np.array(sizes[split, category["id"]])
            item[split] = {"objects": len(a),
                "median_width_at_640": float(np.median(a[:, 0])),
                "median_height_at_640": float(np.median(a[:, 1])),
                "fraction_short_side_under_8px": float((a.min(axis=1) < 8).mean()),
                "fraction_area_under_32_squared": float((a.prod(axis=1) < 32**2).mean())}
        stats.append(item)
    record = {"baseline_run_id": training["run_id"], "weights_sha256": sha(weights),
        "split_manifest_sha256": sha(splits), "annotations_sha256": sha(annotations_path),
        "analysis_splits": ["train", "val"], "baseline_validation": baseline["metrics"],
        "object_size_statistics": stats,
        "size_note": "Projected sizes after aspect-preserving longest-side resize to 640; no augmentation. Area threshold is descriptive, not official COCO size evaluation.",
        "limitations": ["Class imbalance and object size suggest hypotheses, not proven causes.",
            "The baseline test set has already been evaluated; it is excluded from improvement selection.",
            "Three saved validation preview batches are illustrative, not a representative error audit."]}
    previous_path = report_dir / "baseline_error_analysis.json"
    if not args.resolution and previous_path.exists():
        previous = json.loads(previous_path.read_text())
        identities = ("weights_sha256", "split_manifest_sha256", "annotations_sha256")
        if all(previous.get(key) == record[key] for key in identities) and "resolution_experiment" in previous:
            record["resolution_experiment"] = previous["resolution_experiment"]
    if args.resolution:
        import mlflow
        from ultralytics import YOLO, settings
        settings.update({"mlflow": False})
        import torch
        if not torch.backends.mps.is_available():
            raise SystemExit("Native MPS unavailable; refusing silent CPU fallback")
        runtime = ROOT / ".local/resolution-experiment-data.yaml"
        dataset = yaml.safe_load((ROOT / f"data/prepared/{name}/dataset.yaml").read_text())
        dataset["path"] = str(ROOT / dataset["path"])
        runtime.write_text(yaml.safe_dump(dataset))
        mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
        mlflow.set_experiment("plastic-waste-phase1")
        with mlflow.start_run(run_name=f"validation-resolution-{args.resolution}-{time.strftime('%Y%m%d-%H%M%S')}") as run:
            mlflow.set_tags({"phase": "1", "profile": "validation-resolution-experiment",
                "parent_baseline_run": training["run_id"], "weights_sha256": sha(weights),
                "split_manifest_sha256": sha(splits), "dataset_approval": "pending"})
            mlflow.log_params({"split": "val", "imgsz": args.resolution, "conf": 0.001,
                "nms_iou": 0.7, "batch": 2, "device": "mps", "retraining": False})
            results = YOLO(str(weights)).val(data=str(runtime), split="val", imgsz=args.resolution,
                device="mps", batch=2, workers=0, conf=0.001, iou=0.7, plots=True,
                project=str(ROOT / "runs"), name=f"validation-resolution-{args.resolution}-{run.info.run_id[:8]}")
            metrics = {key: float(value) for key, value in results.results_dict.items()}
            mlflow.log_metrics({k.replace("(", "_").replace(")", ""): v for k, v in metrics.items()})
            record["resolution_experiment"] = {"run_id": run.info.run_id, "imgsz": args.resolution,
                "metrics": metrics, "per_class": results.summary(), "speed_ms_per_image": results.speed,
                "note": "Same trained weights; only inference resolution changed. Timing is descriptive, not a controlled speed benchmark."}
            for path in Path(results.save_dir).glob("*.png"):
                mlflow.log_artifact(str(path), artifact_path="evaluation")
            # Keep original baseline evaluation reports untouched.
            output = report_dir / "baseline_error_analysis.json"
            output.write_text(json.dumps(record, indent=2, default=lambda x: x.item()))
            mlflow.log_artifact(str(output))
            mlflow.log_artifact(__file__, artifact_path="provenance")
    else:
        (report_dir / "baseline_error_analysis.json").write_text(json.dumps(record, indent=2))
    lines = ["# Baseline validation error analysis", "",
        "The original baseline, data splits, and test results are preserved. No retraining has been performed.", "",
        "## Evidence", "",
        "| Class | Train objects | Val AP@50 | Val recall | Median val width × height at 640 | Val short side <8px |",
        "| --- | ---: | ---: | ---: | --- | ---: |"]
    per_class = {row["Class"]: row for row in baseline["per_class"]}
    for item in stats:
        v = item["val"]
        m = per_class[item["class"]]
        lines.append(f"| {item['class']} | {item['train']['objects']} | {m['mAP50']:.1%} | {m['Box-R']:.1%} | {v['median_width_at_640']:.1f} × {v['median_height_at_640']:.1f} px | {v['fraction_short_side_under_8px']:.1%} |")
    lines += ["", "Litter dominates the training annotations. Cups and bottles have especially weak validation AP. Object size and imbalance are plausible contributors; this analysis does not establish causation.",
        "", "Validation prediction previews show crowded scenes and small objects. These few examples are illustrative, not a representative audit."]
    if "resolution_experiment" in record:
        experiment = record["resolution_experiment"]
        lines += ["", "## Controlled resolution comparison", "",
            "Same weights, validation split, batch size, confidence floor and NMS IoU. Only image resolution changes. No test-set experiment.", "",
            "| Resolution | mAP@50 | mAP@50–95 |", "| --- | ---: | ---: |"]
        for resolution, metrics in [(640, baseline["metrics"]), (experiment["imgsz"], experiment["metrics"])]:
            lines.append(f"| {resolution} | {metrics['metrics/mAP50(B)']:.2%} | {metrics['metrics/mAP50-95(B)']:.2%} |")
        delta = experiment["metrics"]["metrics/mAP50(B)"] - baseline["metrics"]["metrics/mAP50(B)"]
        lines += ["", f"Validation AP@50 change: {delta * 100:+.2f} percentage points. This measures an inference setting, not new learning.", "",
            "| Class | AP@50 at 640 | AP@50 at 960 | Change (percentage points) |",
            "| --- | ---: | ---: | ---: |"]
        for row in experiment["per_class"]:
            old = per_class[row["Class"]]["mAP50"]
            lines.append(f"| {row['Class']} | {old:.2%} | {row['mAP50']:.2%} | {(row['mAP50'] - old) * 100:+.2f} |")
        lines += ["", "### Decision", ""]
        if delta > 0:
            lines += ["Higher resolution improves overall validation AP@50 in this experiment. Treat 960 as a candidate inference setting, subject to latency/memory checks and per-class tradeoffs. A separate higher-resolution fine-tuning experiment is justified, but not yet started or guaranteed to improve results."]
        else:
            lines += ["Higher inference resolution does not improve overall validation AP@50 here. Keep 640 as the reference; do not adopt 960 solely because it is larger.",
                "", "The next proposed training comparison is YOLO11s versus the YOLO11n baseline, keeping the same 640 input size, data splits, 50-epoch budget, seed and other explicit settings. This tests whether extra model capacity helps; improvement is not guaranteed. Run a short GPU/memory benchmark first, record its engineering-only status, and preserve baseline weights. No candidate training has started."]
    lines += ["", "## Next steps", "",
        "1. Review a systematic validation sample covering misses, false positives, crowded scenes and weak classes before changing annotations.",
        "2. Preserve all 10,000 originals, baseline weights, and current data versions. Train only on training data.",
        "3. If training changes are justified, run one separately named candidate with its own configuration, MLflow run and weights; select using validation performance and resource costs.",
        "4. Do not repeatedly evaluate the existing test set during tuning. Since its baseline results have already been viewed, document that exposure; obtain a fresh independent holdout for stronger final performance claims if needed.",
        "5. Faculty approval remains pending. These public-street images do not establish plastic material identity or campus campaign outcomes.", ""]
    (report_dir / "baseline_error_analysis.md").write_text("\n".join(lines))
    if "resolution_experiment" in record:
        from mlflow import MlflowClient
        client = MlflowClient(tracking_uri=f"sqlite:///{ROOT / 'mlflow.db'}")
        run_id = record["resolution_experiment"]["run_id"]
        for path in [report_dir / "baseline_error_analysis.json", report_dir / "baseline_error_analysis.md", Path(__file__)]:
            client.log_artifact(run_id, str(path), artifact_path="analysis")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
