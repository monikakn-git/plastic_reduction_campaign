"""Short, isolated engineering benchmark; never a candidate quality evaluation."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import shutil
import statistics
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


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    os.chdir(ROOT)
    if not torch.backends.mps.is_available():
        raise SystemExit("Native MPS is required; no silent CPU fallback.")
    if shutil.disk_usage(ROOT).free < 10 * 1024**3:
        raise SystemExit("Less than 10 GiB free; benchmark not started.")
    config = yaml.safe_load((ROOT / "params.yaml").read_text())
    baseline = json.loads((ROOT / "reports/phase1/baseline_metrics.json").read_text())
    baseline_path = ROOT / baseline["model"]
    if sha(baseline_path) != baseline["model_sha256"]:
        raise SystemExit("Baseline weight hash differs from its record")
    splits_path = ROOT / "data/prepared/plitterstreet/splits.json"
    annotations = ROOT / "data/raw/plitterstreet/annotations.json"
    identities = {str(p.relative_to(ROOT)): sha(p) for p in (baseline_path, splits_path, annotations, ROOT/"params.yaml")}
    split_data = json.loads(splits_path.read_text())
    train = sorted([s for s in split_data if s["split"] == "train"], key=lambda s: s["image_id"])
    random.Random(42).shuffle(train)
    train_sample, smoke_sample = train[:256], train[256:288]
    runtime = ROOT / ".local/yolo11s-benchmark"
    runtime.mkdir(parents=True, exist_ok=True)
    prepared = ROOT / "data/prepared/plitterstreet"
    for name, sample in (("train", train_sample), ("smoke", smoke_sample)):
        (runtime/f"{name}.txt").write_text("\n".join(str(prepared/"images/train"/s["file_name"]) for s in sample)+"\n")
    data = yaml.safe_load((prepared/"dataset.yaml").read_text())
    data = {"path": str(prepared), "train": str(runtime/"train.txt"), "val": str(runtime/"smoke.txt"), "names": data["names"]}
    (runtime/"dataset.yaml").write_text(yaml.safe_dump(data))
    manifest = {"seed":42,"source_split":"train", "training_images":train_sample,
        "smoke_validation_images":smoke_sample,
        "note":"Both subsets come only from the original training split. Smoke metrics are not independent quality evidence. Official validation/test sets are unused."}
    manifest_path = ROOT / "reports/phase1/yolo11s_benchmark_sample.json"
    manifest_path.write_text(json.dumps(manifest,indent=2))
    weights = ROOT/"models/yolo11s.pt"
    if not weights.exists():
        raise SystemExit("Download official pretrained YOLO11s weights first")
    weights_hash = sha(weights)
    params = dict(config["train"])
    params.update(model=str(weights), epochs=1, imgsz=640, batch=2, plots=False, close_mosaic=0)
    settings.update({key: False for key in ("mlflow", "wandb", "clearml", "comet", "neptune", "tensorboard") if key in settings})
    mlflow.set_tracking_uri(f"sqlite:///{ROOT/'mlflow.db'}")
    mlflow.set_experiment("plastic-waste-phase1")
    run_name = f"yolo11s-engineering-benchmark-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    measurements = []
    previous = None
    training_start = None
    training_end = None
    def epoch_start(trainer):
        nonlocal previous, training_start
        torch.mps.synchronize()
        previous = training_start = time.perf_counter()
    def batch_end(trainer):
        nonlocal previous
        torch.mps.synchronize()
        now = time.perf_counter()
        measurements.append({"batch":len(measurements)+1, "seconds":now-previous,
            "mps_tensor_bytes":torch.mps.current_allocated_memory(),
            "mps_driver_bytes":torch.mps.driver_allocated_memory(),
            "process_peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        previous = now
        (runtime/"progress.json").write_text(json.dumps(measurements[-1]))
    def epoch_end(trainer):
        nonlocal training_end
        torch.mps.synchronize()
        training_end = time.perf_counter()
    report_path = ROOT/"reports/phase1/yolo11s_benchmark.json"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({"phase":"1", "profile":"engineering-benchmark-yolo11s",
            "not_for_final_performance_claims":"True", "dataset_approval":"pending",
            "baseline_run_id":baseline["run_id"], "pretrained_weights_sha256":weights_hash,
            "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
            "git_dirty":str(bool(subprocess.check_output(["git","status","--porcelain"],text=True).strip()))})
        mlflow.log_params(params)
        for p in [Path(__file__), manifest_path, ROOT/"params.yaml", ROOT/"requirements-lock.txt", ROOT/"reports/phase1/yolo11s_pretrained.json"]:
            mlflow.log_artifact(str(p),artifact_path="provenance")
        mlflow.log_dict(identities,"provenance/preserved_input_hashes.json")
        model = YOLO(str(weights))
        model.add_callback("on_train_epoch_start",epoch_start)
        model.add_callback("on_train_batch_end",batch_end)
        model.add_callback("on_train_epoch_end",epoch_end)
        started = time.perf_counter()
        try:
            model.train(data=str(runtime/"dataset.yaml"), project=str(ROOT/"runs"),
                name=run_name, exist_ok=False, **{k:v for k,v in params.items() if k != "model"})
        except Exception as error:
            failure={"status":"failed","run_id":run.info.run_id,"error":repr(error),"completed_batches":len(measurements),"measurements":measurements}
            report_path.write_text(json.dumps(failure,indent=2))
            mlflow.log_artifact(str(report_path))
            raise
        elapsed = time.perf_counter()-started
        steady = [x["seconds"] for x in measurements[10:]]
        ordered = sorted(steady)
        median = statistics.median(steady)
        p90 = ordered[min(len(ordered)-1,int(len(ordered)*.9))]
        metrics={"benchmark_total_seconds":elapsed,
            "training_epoch_seconds":training_end-training_start,
            "measured_batches":len(measurements), "warmup_batches_excluded":10,
            "steady_batch_median_seconds":median, "steady_batch_p90_seconds":p90,
            "steady_images_per_second":2/statistics.mean(steady),
            "peak_mps_tensor_GiB":max(x["mps_tensor_bytes"] for x in measurements)/1024**3,
            "peak_mps_driver_GiB":max(x["mps_driver_bytes"] for x in measurements)/1024**3,
            "peak_process_rss_GiB":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3,
            "full_epoch_projection_minutes":3500*statistics.mean(steady)/60,
            "fifty_epoch_training_only_projection_hours":3500*50*statistics.mean(steady)/3600}
        unchanged = all(sha(ROOT/path)==value for path,value in identities.items())
        if not unchanged:
            raise RuntimeError("An input changed during benchmark; investigate before accepting results")
        record={"status":"completed","run_id":run.info.run_id,"profile":"engineering-only",
            "model":"YOLO11s","training_images":256,"smoke_images":32,"imgsz":640,"batch":2,
            "dataset_approval":"pending","device":"mps","torch_version":torch.__version__,
            "pretrained_weights_sha256":weights_hash,"metrics":metrics,
            "inputs_unchanged":unchanged,"preserved_input_hashes":identities,
            "measurements":measurements,"run_directory":str(Path(model.trainer.save_dir).relative_to(ROOT)),
            "limitations":["One epoch on a small sample does not establish detection quality or long-run memory stability.",
                "Timing includes explicit GPU synchronization; first ten batches excluded from steady statistics, but optimizer warmup can still influence results.",
                "Runtime projection excludes full validation, checkpointing and startup; thermal throttling and other apps can extend runtime.",
                "MPS and process resident memory share unified memory; these figures must not be added together.",
                "MPS peak fields are maxima observed at batch-end callbacks, not guaranteed transient allocation peaks; process RSS is the OS high-water mark.",
                "No full retraining approved or started. Baseline weights, dataset, splits and baseline configuration unchanged."]}
        report_path.write_text(json.dumps(record,indent=2))
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(report_path))
        for p in Path(model.trainer.save_dir).glob("*.csv"):
            mlflow.log_artifact(str(p),artifact_path="engineering-only")
        text = ["# YOLO11s short engineering benchmark", "",
            "One epoch on 256 training images at 640 pixels, batch 2, native MPS. A disjoint 32-image subset of the original training split exercises validation code; these smoke scores are not quality evidence. Official validation and test splits were not used.", "",
            "| Measurement | Result |", "| --- | ---: |"]
        for key,value in metrics.items():text.append(f"| {key} | {value:.3f} |")
        text += ["", "## Interpretation", "",
            "The short benchmark completed without a reported out-of-memory exception. This supports feasibility for a controlled experiment, but does not prove full-run stability or improvement over the baseline.", "",
            "The full-run projection is training-only and approximate. Budget extra time for validation, checkpointing, data loading variability and MacBook Air thermal throttling. A full run remains subject to separate user approval.", "",
            "CPU process RSS and MPS figures overlap on unified memory; do not sum them. MPS peak fields are maxima observed at batch end, which can miss transient peaks. Process RSS is the OS high-water mark. See the JSON report for batch-level measurements and limitations.", "",
            "Baseline weights, annotations, split manifest and baseline params.yaml hashes were verified unchanged. No candidate model has been selected.", ""]
        markdown=ROOT/"reports/phase1/yolo11s_benchmark.md"
        markdown.write_text("\n".join(text))
        mlflow.log_artifact(str(markdown))
        print(json.dumps({"status":"completed","run_id":run.info.run_id,"metrics":metrics,"inputs_unchanged":unchanged},indent=2),flush=True)


if __name__ == "__main__":
    main()
