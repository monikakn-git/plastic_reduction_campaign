"""Track a real YOLO training run; benchmark runs are explicitly not final baselines."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".local/ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".local/matplotlib"))
import mlflow
import torch
import yaml
from ultralytics import YOLO, settings


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numeric_metrics(values):
    result = {}
    for key, value in values.items():
        try:
            number = float(value)
            if number == number and abs(number) != float("inf"):
                result[re.sub(r"[^a-zA-Z0-9_./ -]", "_", key)] = number
        except (TypeError, ValueError):
            pass
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["benchmark", "baseline"], default="baseline")
    parser.add_argument("--device", choices=["mps", "cpu", "0"])
    parser.add_argument("--run-name")
    parser.add_argument("--resume", type=Path, help="Resume an interrupted local checkpoint")
    parser.add_argument("--mlflow-run-id", help="Original MLflow run for the resumed checkpoint")
    args = parser.parse_args()
    if bool(args.resume) != bool(args.mlflow_run_id):
        parser.error("--resume and --mlflow-run-id must be supplied together")
    os.chdir(ROOT)
    config = yaml.safe_load((ROOT / "params.yaml").read_text())
    dataset_name = config['dataset']['name']
    params = config['train']
    if args.device:
        params["device"] = args.device
    if params["device"] == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable in this process. Run natively with GPU access, or explicitly choose --device cpu.")
    if shutil.disk_usage(ROOT).free < 10 * 1024**3:
        raise SystemExit("Less than 10 GiB free. Training not started; preserve OS headroom.")
    if args.profile == "benchmark":
        params.update(epochs=1, fraction=0.05, imgsz=320, plots=False, close_mosaic=0)
    runtime = ROOT / ".local"
    runtime.mkdir(exist_ok=True)
    dataset = yaml.safe_load((ROOT / f"data/prepared/{dataset_name}/dataset.yaml").read_text())
    dataset["path"] = str(ROOT / dataset["path"])
    runtime_yaml = runtime / "dataset-runtime.yaml"
    runtime_yaml.write_text(yaml.safe_dump(dataset))
    # Explicit logging avoids duplicate automatic tracking runs and third-party trackers.
    settings.update({key: False for key in ("mlflow", "wandb", "clearml", "comet", "neptune", "tensorboard")
                     if key in settings})
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("plastic-waste-phase1")
    run_name = args.run_name or f"{args.profile}-{time.strftime('%Y%m%d-%H%M%S')}"
    resume_record = None
    if args.resume:
        original = mlflow.get_run(args.mlflow_run_id)
        for key, expected in {
            "profile": args.profile,
            "dataset_annotations_sha256": digest(ROOT / f"data/raw/{dataset_name}/annotations.json"),
            "split_manifest_sha256": digest(ROOT / f"data/prepared/{dataset_name}/splits.json"),
        }.items():
            if original.data.tags.get(key) != expected:
                raise SystemExit(f"Resume refused: original run {key} differs from current inputs")
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved.get("optimizer") is None or saved.get("epoch", -1) < 0:
            raise SystemExit("Checkpoint has no resumable optimizer/epoch state")
        for key, value in params.items():
            if key != "model" and saved["train_args"].get(key) != value:
                raise SystemExit(f"Resume refused: training parameter {key} changed")
        if Path(saved["train_args"]["save_dir"]).resolve() != args.resume.resolve().parents[1]:
            raise SystemExit("Checkpoint directory differs from its original training run")
        run_name = original.data.tags["mlflow.runName"]
        resume_record = {"checkpoint": str(args.resume), "checkpoint_sha256": digest(args.resume),
                         "completed_epochs": saved["epoch"] + 1,
                         "resumed_at": time.strftime('%Y-%m-%dT%H:%M:%S%z')}
        del saved
    with mlflow.start_run(run_id=args.mlflow_run_id, run_name=None if args.resume else run_name) as run:
        try:
            code_version = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        except subprocess.CalledProcessError:
            code_version, dirty = "uncommitted", True
        if args.resume:
            original_tags = mlflow.get_run(run.info.run_id).data.tags
            code_version = original_tags["git_commit"]
            dirty = original_tags["git_dirty"] == "True"
            mlflow.set_tag("resume_git_commit", subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
            mlflow.log_dict(resume_record, "recovery/resume.json")
        else:
            mlflow.set_tags({"phase": "1", "profile": args.profile,
            "dataset_approval": "pending", "git_commit": code_version, "git_dirty": str(dirty),
            "dataset": dataset_name, "dataset_images": str(config['dataset']['target_images']),
            "dataset_annotations_sha256": digest(ROOT / f"data/raw/{dataset_name}/annotations.json"),
            "split_manifest_sha256": digest(ROOT / f"data/prepared/{dataset_name}/splits.json"),
            "not_for_final_performance_claims": str(args.profile == "benchmark")})
        mlflow.log_params(params)
        provenance_path = "recovery/provenance" if args.resume else "provenance"
        for path in ["params.yaml", "dvc.yaml", "dvc.lock", config['prepare']['classes'], "requirements-lock.txt",
                     "reports/phase1/dataset_summary.json", f"data/raw/{dataset_name}/source.json"]:
            mlflow.log_artifact(str(ROOT / path), artifact_path=provenance_path)
        for path in (ROOT / "data").glob("**/*.dvc"):
            mlflow.log_artifact(str(path), artifact_path=provenance_path)
        model = YOLO(str(args.resume) if args.resume else params["model"])
        checkpoint = Path(model.ckpt_path)
        if checkpoint.is_file() and not args.resume:
            mlflow.set_tag("initial_weights_sha256", digest(checkpoint))
        def record_epoch(trainer):
            mlflow.log_metrics(numeric_metrics(trainer.metrics), step=int(trainer.epoch))
        model.add_callback("on_fit_epoch_end", record_epoch)
        started = time.monotonic()
        if args.resume:
            results = model.train(resume=True, device=params["device"])
        else:
            results = model.train(data=str(runtime_yaml), project=str(ROOT / "runs"),
                                  name=run_name, exist_ok=False, save_period=-1,
                                  **{k: v for k, v in params.items() if k != "model"})
        elapsed = time.monotonic() - started
        metrics = numeric_metrics(results.results_dict)
        metrics["training_seconds"] = elapsed
        mlflow.log_metrics(metrics)
        destination = ROOT / "models" / f"{dataset_name}-{args.profile}.pt"
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(model.trainer.best, destination)
        mlflow.log_artifact(str(destination), artifact_path="weights")
        for path in Path(model.trainer.save_dir).iterdir():
            if path.suffix in (".csv", ".png", ".jpg", ".yaml"):
                mlflow.log_artifact(str(path), artifact_path="training")
        record = {"run_id": run.info.run_id, "profile": args.profile, 'dataset': dataset_name,
            'dataset_images': config['dataset']['target_images'],
            "metrics": metrics, "git_commit": code_version, "git_dirty": dirty,
            "model": str(destination.relative_to(ROOT)), "model_sha256": digest(destination),
            "dataset_approval": "pending", "evaluation_split": "validation",
            "note": "Pipeline check only, not a final baseline" if args.profile == "benchmark" else "Initial baseline; test evaluation is separate"}
        if resume_record:
            record["recovery"] = resume_record
            record["timing_note"] = "training_seconds measures the resumed session; results.csv time restarts at recovery"
        (ROOT / f"reports/phase1/{args.profile}_metrics.json").write_text(json.dumps(record, indent=2))
        print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
