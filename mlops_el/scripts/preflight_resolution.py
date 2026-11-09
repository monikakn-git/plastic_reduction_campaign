"""One standard-protocol validation trial at 960, with the saved small model."""
import json
import os
from pathlib import Path
import time

from train_candidate import ROOT,verify_inputs
from train import digest
from evaluate import json_safe
import mlflow
import torch
from ultralytics import YOLO,settings


def main():
    os.chdir(ROOT);os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT','1')
    verify_inputs()
    if not torch.backends.mps.is_available():raise SystemExit('Native MPS required')
    out=ROOT/'reports/phase1/next_training_preflight';out.mkdir(exist_ok=True)
    report=out/'resolution960.json'
    if report.exists():raise SystemExit('Completed resolution check exists; preserve it')
    trained=json.loads((ROOT/'reports/phase1/yolo11s_candidate_training.json').read_text())
    baseline=json.loads((ROOT/'reports/phase1/yolo11s_candidate_validation.json').read_text())
    weights=ROOT/trained['model']
    if digest(weights)!=trained['model_sha256']:raise SystemExit('Saved weight hash mismatch')
    settings.update({'mlflow':False})
    mlflow.set_tracking_uri(f"sqlite:///{ROOT/'mlflow.db'}");mlflow.set_experiment('plastic-waste-phase1')
    with mlflow.start_run(run_name='yolo11s-preflight-960-'+time.strftime('%Y%m%d-%H%M%S')) as run:
        mlflow.set_tags({'phase':'1','profile':'next-training-resolution-preflight','dataset_approval':'pending','training_run_id':trained['run_id'],'weights_sha256':digest(weights)})
        mlflow.log_params({'imgsz':960,'batch':2,'rect':True,'conf':.001,'iou':.7,'split':'val','multi_label':True,'retraining':False})
        result=YOLO(str(weights)).val(data=str(ROOT/'.local/yolo11s-candidate-validation.yaml'),split='val',device='mps',imgsz=960,batch=2,workers=0,conf=.001,iou=.7,rect=True,plots=True,project=str(ROOT/'runs'),name=f'preflight-960-{run.info.run_id[:8]}')
        metrics={k:float(v) for k,v in result.results_dict.items()}
        record={'run_id':run.info.run_id,'images':1500,'weights_sha256':digest(weights),'split':'val','imgsz':960,
            'metrics':metrics,'baseline640_metrics':baseline['metrics'],'per_class':result.summary(),
            'speed_ms_per_image':result.speed,'note':'Same saved model and standard validation protocol; only resolution changed. This does not test training at higher resolution or a larger model.'}
        verify_inputs();report.write_text(json.dumps(record,indent=2,default=json_safe))
        mlflow.log_metrics({k.replace('(','_').replace(')','_'):v for k,v in metrics.items()})
        for p in [report,Path(__file__)]:mlflow.log_artifact(str(p))
        for p in Path(result.save_dir).glob('*.png'):mlflow.log_artifact(str(p),artifact_path='validation')
        print(json.dumps(record,indent=2,default=json_safe),flush=True)


if __name__=='__main__':main()
