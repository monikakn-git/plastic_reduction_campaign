"""Train the approved YOLO11s comparison without overwriting the baseline."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".local/ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".local/matplotlib"))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow
import torch
import yaml
from ultralytics import YOLO, settings
from train import digest, numeric_metrics


def verify_inputs():
    plan = json.loads((ROOT/"reports/phase1/yolo11s_candidate_plan.json").read_text())
    for name, expected in plan["protected_sha256"].items():
        if digest(ROOT/name) != expected:
            raise RuntimeError(f"Protected baseline/data input changed: {name}")
    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if bool(args.resume) != bool(args.run_id):
        parser.error("Supply both --resume and --run-id")
    os.chdir(ROOT)
    plan = verify_inputs()
    config_path = ROOT/"configs/yolo11s_candidate.yaml"
    config = yaml.safe_load(config_path.read_text())
    baseline = yaml.safe_load((ROOT/"params.yaml").read_text())
    params = dict(config["train"])
    if config["dataset"] != baseline["dataset"] or any(params.get(k) != v for k,v in baseline["train"].items() if k != "model") or set(params) != set(baseline["train"]):
        raise SystemExit("Candidate must differ from baseline only in model initialization")
    if params["model"] != "models/yolo11s.pt":
        raise SystemExit("Only the approved YOLO11s candidate is allowed")
    if not torch.backends.mps.is_available():
        raise SystemExit("Native MPS unavailable; no silent CPU fallback")
    if shutil.disk_usage(ROOT).free < 10*1024**3:
        raise SystemExit("Less than 10 GiB free; preserve system headroom")
    dataset = yaml.safe_load((ROOT/"data/prepared/plitterstreet/dataset.yaml").read_text())
    dataset["path"] = str(ROOT/dataset["path"])
    dataset.pop("test", None)  # This workflow performs training and validation only.
    runtime = ROOT/".local/yolo11s-candidate-dataset.yaml"
    runtime.write_text(yaml.safe_dump(dataset))
    settings.update({k:False for k in ("mlflow","wandb","clearml","comet","neptune","tensorboard") if k in settings})
    mlflow.set_tracking_uri(f"sqlite:///{ROOT/'mlflow.db'}")
    mlflow.set_experiment("plastic-waste-phase1")
    run_name = f"yolo11s-candidate-{time.strftime('%Y%m%d-%H%M%S')}"
    if args.resume:
        previous = mlflow.get_run(args.run_id)
        if previous.data.tags.get("candidate_config_sha256") != digest(config_path):
            raise SystemExit("Resume configuration does not match original candidate")
        if previous.data.tags.get("profile") != "yolo11s-candidate":
            raise SystemExit("Not a candidate training run")
        ckpt = torch.load(args.resume,map_location="cpu",weights_only=False)
        if ckpt.get("optimizer") is None or ckpt.get("epoch",-1)<0:
            raise SystemExit("Checkpoint has no resumable training state")
        for key,value in params.items():
            if key != "model" and ckpt["train_args"].get(key) != value:
                raise SystemExit(f"Resume setting mismatch: {key}")
        if Path(ckpt["train_args"]["save_dir"]).resolve() != args.resume.resolve().parents[1]:
            raise SystemExit("Checkpoint directory differs from original run")
        resume_epoch = ckpt["epoch"]+1
        del ckpt
        run_name = previous.data.tags["mlflow.runName"]
    with mlflow.start_run(run_id=args.run_id,run_name=None if args.resume else run_name) as run:
        commit = subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
        if not args.resume:
            mlflow.set_tags({"phase":"1","profile":"yolo11s-candidate","dataset_approval":"pending",
                "baseline_run_id":plan["baseline_run_id"],"candidate_config_sha256":digest(config_path),
                "split_manifest_sha256":digest(ROOT/"data/prepared/plitterstreet/splits.json"),
                "dataset_annotations_sha256":digest(ROOT/"data/raw/plitterstreet/annotations.json"),
                "initial_weights_sha256":digest(ROOT/params["model"]),"git_commit":commit,
                "git_dirty":str(bool(subprocess.check_output(["git","status","--porcelain"],text=True).strip()))})
            mlflow.log_params(params)
        else:
            mlflow.log_dict({"checkpoint":str(args.resume),"sha256":digest(args.resume),
                "completed_epochs":resume_epoch,"resume_git_commit":commit},f"recovery/resume-{time.strftime('%Y%m%d-%H%M%S')}.json")
        for name in ["configs/yolo11s_candidate.yaml","experiments/yolo11s/dvc.yaml","requirements-lock.txt",
                "reports/phase1/yolo11s_candidate_plan.json","reports/phase1/yolo11s_pretrained.json",
                "models/yolo11s.pt.dvc","scripts/train_candidate.py"]:
            mlflow.log_artifact(str(ROOT/name),artifact_path="recovery/provenance" if args.resume else "provenance")
        active = {"run_id":run.info.run_id,"run_directory":f"runs/{run_name}","profile":"yolo11s-candidate",
            "config":str(config_path.relative_to(ROOT)),"started_at":time.strftime('%Y-%m-%dT%H:%M:%S%z')}
        (ROOT/".local/yolo11s-candidate-active.json").write_text(json.dumps(active,indent=2))
        print(json.dumps(active),flush=True)
        model = YOLO(str(args.resume) if args.resume else params["model"])
        def record_epoch(trainer):
            mlflow.log_metrics(numeric_metrics(trainer.metrics),step=int(trainer.epoch))
        model.add_callback("on_fit_epoch_end",record_epoch)
        started=time.monotonic()
        if args.resume:
            results = model.train(resume=True,device=params["device"])
        else:
            results = model.train(data=str(runtime),project=str(ROOT/"runs"),name=run_name,
                exist_ok=False,save_period=-1,**{k:v for k,v in params.items() if k != "model"})
        metrics=numeric_metrics(results.results_dict)
        metrics["training_session_seconds"]=time.monotonic()-started
        verify_inputs()
        destination=ROOT/"models/plitterstreet-yolo11s-candidate.pt"
        if destination.exists():
            raise RuntimeError("Candidate destination exists; refuse to overwrite a previous experiment")
        shutil.copy2(model.trainer.best,destination)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(destination),artifact_path="weights")
        for p in Path(model.trainer.save_dir).iterdir():
            if p.suffix in (".csv",".png",".jpg",".yaml"):
                mlflow.log_artifact(str(p),artifact_path="training")
        record={**active,"status":"completed","dataset":"plitterstreet","training_images":7000,
            "validation_images":1500,"model":str(destination.relative_to(ROOT)),"model_sha256":digest(destination),
            "metrics":metrics,"baseline_preserved":True,"test_evaluated":False,
            "dataset_approval":"pending","note":"Candidate experiment; no automatic model promotion. Session timing excludes earlier sessions if resumed."}
        report=ROOT/"reports/phase1/yolo11s_candidate_training.json"
        report.write_text(json.dumps(record,indent=2))
        mlflow.log_artifact(str(report))
        print(json.dumps(record,indent=2),flush=True)


if __name__ == "__main__":
    main()
