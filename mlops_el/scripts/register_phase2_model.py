"""Package, register and export a demo candidate without changing Phase 1 weights."""
import argparse
from importlib.metadata import version as package_version
import base64
import json
import os
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'.local/ultralytics'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.local/matplotlib'))
import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema,ColSpec
from service.model import MODEL_SHA256,CLASSES,file_sha
from service.quality import assess
from service.registry_model import WasteDetectorModel

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--release-dir',default='releases/yolo11m-demo-v3');args=parser.parse_args()
    os.chdir(ROOT);out=ROOT/'reports/phase2';out.mkdir(exist_ok=True)
    policy=json.loads((ROOT/'configs/phase2/release_policy.json').read_text());evaluation=json.loads((ROOT/policy['evaluation']).read_text());weights=ROOT/policy['weights']
    quality=assess(policy,evaluation,file_sha(weights));(out/'quality_gates.json').write_text(json.dumps(quality,indent=2)+'\n')
    if not quality['integrity_passed']:raise SystemExit('Artifact integrity check failed; do not register or serve')
    if args.prepare_only:print(json.dumps(quality,indent=2));return
    bundle=ROOT/args.release_dir
    if bundle.exists():raise SystemExit('Release bundle exists; inspect its registry identity rather than duplicate registration')
    mlflow.set_tracking_uri(f'sqlite:///{ROOT / "mlflow.db"}');mlflow.set_experiment('plastic-waste-phase2')
    with mlflow.start_run(run_name='package-yolo11m-demo') as run:
        mlflow.set_tags({'source_training_run_id':policy['source_training_run_id'],'source_test_run_id':policy['source_test_run_id'],'trained_dataset':'labels-v3','allowed_scope':'local_demo','production_approved':'false'})
        mlflow.log_params({'weights_sha256':MODEL_SHA256,'imgsz':960,'confidence_threshold':.25,'nms_iou':.7})
        mlflow.log_artifact(str(out/'quality_gates.json'),artifact_path='quality')
        info=mlflow.pyfunc.log_model(name='detector',python_model=WasteDetectorModel(),artifacts={'weights':str(weights)},code_paths=[str(ROOT/'service')],signature=ModelSignature(inputs=Schema([ColSpec('string','image_base64')]),outputs=Schema([ColSpec('string')])),pip_requirements=[f'{p}=={package_version(p)}' for p in ('mlflow','ultralytics','torch','torchvision','Pillow','numpy')],metadata={'weights_sha256':MODEL_SHA256,'trained_dataset':'labels-v3','production_approved':False})
        version=mlflow.register_model(info.model_uri,policy['registry_name']);client=mlflow.MlflowClient()
        client.set_model_version_tag(policy['registry_name'],version.version,'production_approved','false')
        client.set_model_version_tag(policy['registry_name'],version.version,'source_training_run_id',policy['source_training_run_id'])
        client.set_registered_model_alias(policy['registry_name'],policy['alias'],version.version)
        bundle.mkdir(parents=True);shutil.copy2(weights,bundle/'weights.pt')
        release={'registry_name':policy['registry_name'],'registry_version':str(version.version),'alias':policy['alias'],'model_uri':f'models:/{policy["registry_name"]}/{version.version}','packaging_run_id':run.info.run_id,'source_training_run_id':policy['source_training_run_id'],'source_test_run_id':policy['source_test_run_id'],'weights_sha256':MODEL_SHA256,'trained_dataset':'labels-v3','classes':CLASSES,'production_approved':False,'allowed_scope':'local demonstration with human review','test_map50':evaluation['metrics']['metrics/mAP50(B)'],'test_map50_95':evaluation['metrics']['metrics/mAP50-95(B)']}
        (bundle/'release.json').write_text(json.dumps(release,indent=2)+'\n');(out/'registered_model.json').write_text(json.dumps(release,indent=2)+'\n');mlflow.log_artifact(str(bundle/'release.json'),artifact_path='release')
    print(json.dumps(release,indent=2))

if __name__=='__main__':main()
