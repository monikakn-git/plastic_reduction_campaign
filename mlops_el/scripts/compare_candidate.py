"""Evaluate the candidate on validation only; preserve all baseline evidence."""
import json
from pathlib import Path
import time

from train_candidate import ROOT, verify_inputs
from train import digest
from evaluate import json_safe
import mlflow
import torch
import yaml
from ultralytics import YOLO, settings


def main():
    verify_inputs()
    if not torch.backends.mps.is_available():
        raise SystemExit("Native MPS unavailable")
    out=ROOT/"reports/phase1"
    candidate=json.loads((out/"yolo11s_candidate_training.json").read_text())
    baseline=json.loads((out/"evaluation_val.json").read_text())
    weights=ROOT/candidate["model"]
    if digest(weights) != candidate["model_sha256"]:
        raise SystemExit("Candidate weight hash mismatch")
    data=yaml.safe_load((ROOT/"data/prepared/plitterstreet/dataset.yaml").read_text())
    data["path"]=str(ROOT/data["path"])
    data.pop("test",None)
    runtime=ROOT/".local/yolo11s-candidate-validation.yaml"
    runtime.write_text(yaml.safe_dump(data))
    settings.update({"mlflow":False})
    mlflow.set_tracking_uri(f"sqlite:///{ROOT/'mlflow.db'}")
    mlflow.set_experiment("plastic-waste-phase1")
    with mlflow.start_run(run_name=f"yolo11s-validation-{time.strftime('%Y%m%d-%H%M%S')}") as run:
        mlflow.set_tags({"phase":"1","profile":"candidate-validation","training_run_id":candidate["run_id"],
            "weights_sha256":digest(weights),"dataset_approval":"pending"})
        mlflow.log_params({"split":"val","images":1500,"imgsz":640,"batch":2,"conf":.001,"nms_iou":.7})
        results=YOLO(str(weights)).val(data=str(runtime),split="val",device="mps",imgsz=640,
            batch=2,workers=0,conf=.001,iou=.7,plots=True,project=str(ROOT/"runs"),
            name=f"yolo11s-validation-{run.info.run_id[:8]}")
        metrics={k:float(v) for k,v in results.results_dict.items()}
        mlflow.log_metrics({k.replace("(","_").replace(")","_"):v for k,v in metrics.items()})
        record={"split":"val","run_id":run.info.run_id,"training_run_id":candidate["run_id"],
            "weights_sha256":digest(weights),"metrics":metrics,"per_class":results.summary(),
            "speed_ms_per_image":results.speed,"baseline_validation_run_id":baseline["run_id"],
            "test_evaluated":False,"automatic_promotion":False,"dataset_approval":"pending",
            "caveat":"Validation comparison only. Ambiguous labels remain; public-street results do not establish campus performance."}
        verify_inputs()
        report=out/"yolo11s_candidate_validation.json"
        report.write_text(json.dumps(record,indent=2,default=json_safe))
        lines=["# YOLO11n baseline versus YOLO11s candidate", "",
            "Same original 7,000/1,500 train/validation images and explicit training settings; only model initialization/capacity changes. No new test evaluation. No automatic model replacement.", "",
            "| Metric | YOLO11n | YOLO11s | Change (percentage points) |", "| --- | ---: | ---: | ---: |"]
        for key in ["metrics/precision(B)","metrics/recall(B)","metrics/mAP50(B)","metrics/mAP50-95(B)"]:
            a,b=baseline["metrics"][key],metrics[key]
            lines.append(f"| {key} | {a:.2%} | {b:.2%} | {(b-a)*100:+.2f} |")
        lines += ["", "| Class | Baseline AP@50 | Candidate AP@50 | Change (percentage points) |", "| --- | ---: | ---: | ---: |"]
        old={x["Class"]:x for x in baseline["per_class"]}
        for row in record["per_class"]:
            a,b=old[row["Class"]]["mAP50"],row["mAP50"]
            lines.append(f"| {row['Class']} | {a:.2%} | {b:.2%} | {(b-a)*100:+.2f} |")
        lines += ["", "## Decision requires review", "",
            "Review both aggregate AP metrics and bottle/cup/plastic-bag performance before selecting a model. Consider detection errors and resource costs; higher aggregate AP alone does not resolve material ambiguity or unreliable classes. The baseline remains preserved.", "",
            "Ambiguous source labels flagged during annotation review were not changed. The baseline test results have previously been viewed; this experiment uses validation only for comparison. No campus or campaign effectiveness claim is established.", ""]
        comparison=out/"yolo11s_validation_comparison.md"
        comparison.write_text("\n".join(lines))
        for p in [report,comparison,Path(__file__),ROOT/"configs/yolo11s_candidate.yaml"]:
            mlflow.log_artifact(str(p))
        for p in Path(results.save_dir).glob("*.png"):
            mlflow.log_artifact(str(p),artifact_path="evaluation")
        print("\n".join(lines),flush=True)


if __name__ == "__main__":
    main()
