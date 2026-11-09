"""Create deterministic, checksummed regular-file train/val archives; exclude test data."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import shutil
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT/'cloud'))
from runner import sha
import yaml


class Reader:
    def __init__(self,stream):self.stream=stream;self.hash=hashlib.sha256()
    def read(self,size=-1):
        data=self.stream.read(size);self.hash.update(data);return data


def add_bytes(tar,name,payload):
    info=tarfile.TarInfo(name);info.size=len(payload);info.mode=0o644;info.mtime=0
    tar.addfile(info,io.BytesIO(payload))
    return {'path':name,'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}


def build(scope):
    source=ROOT/'data/prepared/plitterstreet-labels-v3'
    records=json.loads((source/'splits.json').read_text())
    train=sorted([r for r in records if r['split']=='train'],key=lambda r:r['image_id'])
    val=sorted([r for r in records if r['split']=='val'],key=lambda r:r['image_id'])
    random.Random(42).shuffle(train)
    if scope=='benchmark':
        important={372,449,528,651,5986,14677}
        train=sorted(train,key=lambda r:r['image_id'] not in important)
        selected_train,selected_val=train[:256],train[256:288]
    else:selected_train,selected_val=train,val
    source_files=[]
    for split,rows in [('train',selected_train),('val',selected_val)]:
        for row in rows:
            original_split=row['split'];flat='__'.join(Path(row['file_name']).parts)
            for kind,filename in [('images',flat),('labels',str(Path(flat).with_suffix('.txt')))]:
                src=source/kind/original_split/filename
                source_files.append((f'dataset/{kind}/{split}/{filename}',src))
    extras=[('models/yolo11m.pt',ROOT/'models/yolo11m.pt'),('runner.py',ROOT/'cloud/runner.py'),
        ('config.yaml',ROOT/'configs/yolo11m_cloud_proposed.yaml'),
        ('requirements-cloud.txt',ROOT/'cloud/requirements-cloud.txt')]
    for name in ['data/prepared/plitterstreet-labels-v3.dvc','data/raw/plitterstreet.dvc',
        'configs/label_corrections_v3.json','reports/phase1/yolo11m_pretrained.json',
        'reports/phase1/label_refinement/summary.json']:
        extras.append(('provenance/'+Path(name).name,ROOT/name))
    source_files+=extras
    for _,src in source_files:
        if not src.is_file():raise FileNotFoundError(src)
    expected=sum(src.stat().st_size for _,src in source_files)
    if shutil.disk_usage(ROOT).free < expected+10*1024**3:raise RuntimeError('Insufficient headroom for archive plus 10 GiB')
    out=ROOT/'exports';out.mkdir(exist_ok=True)
    archive=out/f'mlops-{scope}-labels-v3.tar'
    if archive.exists():raise RuntimeError(f'Completed export exists: {archive}')
    temporary=archive.with_suffix('.tar.part')
    files=[]
    with tarfile.open(temporary,'w',format=tarfile.PAX_FORMAT) as tar:
        for index,(name,src) in enumerate(sorted(source_files)):
            info=tarfile.TarInfo(name);info.size=src.stat().st_size;info.mode=0o644;info.mtime=0
            with src.open('rb') as stream:
                reader=Reader(stream);tar.addfile(info,reader)
            files.append({'path':name,'bytes':info.size,'sha256':reader.hash.hexdigest()})
            if (index+1)%2000==0:print(f'{scope}: archived {index+1}/{len(source_files)} files',flush=True)
        config=yaml.safe_load((source/'dataset.yaml').read_text())
        dataset={'path':'dataset','train':'images/train','val':'images/val','names':config['names']}
        files.append(add_bytes(tar,'dataset/dataset.yaml',yaml.safe_dump(dataset).encode()))
        manifest={'scope':scope,'source_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'dataset_version':'plitterstreet-labels-v3','source_dvc_sha256':sha(ROOT/'data/prepared/plitterstreet-labels-v3.dvc'),
            'train_images':len(selected_train),'val_images':len(selected_val),'test_images':0,
            'benchmark_smoke_is_original_training_data':scope=='benchmark',
            'selection_seed':42,'train_image_ids':[r['image_id'] for r in selected_train],
            'val_image_ids':[r['image_id'] for r in selected_val],
            'original_split_of_val': 'train' if scope=='benchmark' else 'val',
            'note':'Benchmark smoke metrics are not independent quality evidence.' if scope=='benchmark' else 'Full original train/validation split. Test images excluded.',
            'files':files}
        add_bytes(tar,'manifest.json',json.dumps(manifest,sort_keys=True).encode())
    temporary.replace(archive)
    record={'scope':scope,'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),
        'train_images':len(selected_train),'val_images':len(selected_val),'test_images':0,
        'source_git_commit':manifest['source_git_commit'],'source_dvc_sha256':manifest['source_dvc_sha256']}
    report=ROOT/f'reports/phase1/cloud_export_{scope}.json';report.write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2),flush=True)
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--scope',choices=['benchmark','trainval'],required=True)
    build(parser.parse_args().scope)
