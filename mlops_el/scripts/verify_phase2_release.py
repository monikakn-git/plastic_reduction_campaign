"""Standard-library release check for deployment hosts, without unpickling weights."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def verify(bundle):
    bundle=Path(bundle);policy=json.loads((ROOT/'configs/phase2/release_policy.json').read_text());evaluation=json.loads((ROOT/policy['evaluation']).read_text());release=json.loads((bundle/'release.json').read_text())
    with (bundle/'weights.pt').open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if digest!=evaluation['weights_sha256'] or digest!=release['weights_sha256']:raise ValueError('Release checkpoint integrity failed')
    if release['source_training_run_id']!=policy['source_training_run_id'] or release['trained_dataset']!='labels-v3':raise ValueError('Release training provenance differs')
    if release['registry_name']!=policy['registry_name'] or release['production_approved'] is not False:raise ValueError('Only the demo registry release may deploy here')
    return {'status':'verified','model_sha256':digest,'registry_version':release['registry_version'],'production_approved':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('bundle',nargs='?',default=str(ROOT/'releases/yolo11m-demo-v3'));args=parser.parse_args();print(json.dumps(verify(args.bundle),indent=2))
