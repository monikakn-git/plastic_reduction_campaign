"""Evaluate selected weights explicitly; test results must not be used for tuning."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".local/ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".local/matplotlib"))
import mlflow
import numpy as np
import yaml
from ultralytics import YOLO, settings


def json_safe(value):
    """Convert NumPy metric scalars/arrays without changing their values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f'Unsupported report value: {type(value).__name__}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--report", help="Explicit report path; existing reports are preserved")
    parser.add_argument("--source-run-id", help="Training run that produced the frozen weights")
    args = parser.parse_args()
    os.chdir(ROOT)
    config = yaml.safe_load((ROOT / args.config).read_text())
    params = config['train']
    dataset_name = config['dataset']['name']
    args.weights = args.weights or f'models/{dataset_name}-baseline.pt'
    prepared = config['dataset'].get('prepared_path', f'data/prepared/{dataset_name}')
    data = yaml.safe_load((ROOT / prepared / "dataset.yaml").read_text())
    data["path"] = str(ROOT / data["path"])
    runtime = ROOT / ".local/evaluation-data.yaml"
    runtime.parent.mkdir(exist_ok=True)
    runtime.write_text(yaml.safe_dump(data))
    settings.update({"mlflow": False})
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("plastic-waste-phase1")
    weights = ROOT / args.weights
    if not weights.exists():
        raise SystemExit(f"Weights not found: {weights}")
    report = ROOT / (args.report or f"reports/phase1/evaluation_{args.split}.json")
    if args.report and report.exists():
        raise SystemExit(f"Report already exists: {report}")
    with weights.open('rb') as stream:
        weights_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
    split_images = len(list((Path(data['path']) / data[args.split]).glob('*.*')))
    with mlflow.start_run(run_name=f"evaluation-{args.split}-{time.strftime('%Y%m%d-%H%M%S')}") as run:
        mlflow.log_params({"weights": args.weights, "split": args.split,
                           "weights_sha256": weights_sha256, "prepared_path": prepared,
                           "source_run_id": args.source_run_id or "unspecified",
                           "imgsz": params["imgsz"], "batch": args.batch,
                           "conf": 0.001, "nms_iou": 0.7, "augment": False})
        results = YOLO(str(weights)).val(data=str(runtime), split=args.split,
            device=args.device, imgsz=params["imgsz"], batch=args.batch, workers=0,
            conf=0.001, iou=0.7, plots=True, project=str(ROOT / "runs"),
            name=f"evaluation-{args.split}-{run.info.run_id[:8]}")
        metrics = {k: float(v) for k, v in results.results_dict.items()}
        mlflow.log_metrics({k.replace("(", "_").replace(")", ""): v for k, v in metrics.items()})
        record = {"split": args.split, "run_id": run.info.run_id, "dataset": dataset_name,
                  "dataset_images": config['dataset']['target_images'], "metrics": metrics,
                  "evaluated_images": split_images, "weights": args.weights,
                  "weights_sha256": weights_sha256, "source_run_id": args.source_run_id,
                  "prepared_path": prepared, "imgsz": params['imgsz'],
                  "device": args.device, "batch": args.batch,
                  "per_class": results.summary(), "speed_ms_per_image": results.speed,
                  "caveat": "Initial evaluation; public-dataset results do not establish campus performance."}
        report.write_text(json.dumps(record, indent=2, default=json_safe))
        mlflow.log_artifact(str(report))
        for path in Path(results.save_dir).glob("*.png"):
            mlflow.log_artifact(str(path), artifact_path="evaluation")
        print(json.dumps(record, indent=2, default=json_safe))


if __name__ == "__main__":
    main()
