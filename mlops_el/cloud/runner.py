"""Free-CUDA benchmark/training helpers. Full training is explicitly gated."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import sqlite3
import subprocess
import tarfile
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def safe_name(name):
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or not p.parts or '\\' in name:
        raise ValueError('Unsafe archive path')
    return p


def unpack(archive,destination,expected_sha):
    archive,destination=Path(archive),Path(destination)
    if sha(archive)!=expected_sha:raise ValueError('Archive checksum mismatch')
    if destination.exists():raise ValueError('Extraction destination exists; use a fresh directory')
    with tarfile.open(archive,'r:') as tar:
        members=tar.getmembers();names=[m.name for m in members]
        if len(names)!=len(set(names)):raise ValueError('Duplicate archive entries')
        for m in members:
            safe_name(m.name)
            if not m.isfile():raise ValueError('Only regular files are accepted; links are rejected')
        manifest=json.load(tar.extractfile('manifest.json'))
        expected={r['path']:r for r in manifest['files']}
        if len(expected)!=len(manifest['files']):raise ValueError('Duplicate manifest paths')
        if set(names)!=set(expected)|{'manifest.json'}:raise ValueError('Archive members differ from manifest')
        destination.mkdir(parents=True)
        for m in members:
            if m.name=='manifest.json':continue
            row=expected[m.name]
            if row['bytes']!=m.size:raise ValueError('Archive size mismatch')
            target=destination.joinpath(*safe_name(m.name).parts);target.parent.mkdir(parents=True,exist_ok=True)
            h=hashlib.sha256()
            with tar.extractfile(m) as src,target.open('wb') as dst:
                for block in iter(lambda:src.read(1024*1024),b''):
                    h.update(block);dst.write(block)
            if h.hexdigest()!=row['sha256']:raise ValueError(f'File checksum mismatch: {m.name}')
        (destination/'manifest.json').write_text(json.dumps(manifest,indent=2))
    verify_bundle(destination)
    return manifest


def verify_bundle(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    for row in manifest['files']:
        path=root.joinpath(*safe_name(row['path']).parts)
        if not path.is_file() or path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:
            raise ValueError(f'Bundle verification failed: {row["path"]}')
    if any('/test/' in r['path'] for r in manifest['files']):raise ValueError('Test data must not be exported')
    expected={'benchmark':(256,32),'trainval':(7000,1500)}[manifest['scope']]
    if (manifest['train_images'],manifest['val_images'])!=expected:raise ValueError('Unexpected dataset size')
    counts=Counter('/'.join(r['path'].split('/')[:3]) for r in manifest['files'])
    for split,count in zip(('train','val'),expected):
        if counts[f'dataset/images/{split}']!=count or counts[f'dataset/labels/{split}']!=count:
            raise ValueError('Actual image/label counts differ from declared scope')
    return manifest


def atomic_copy(source,destination):
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name(destination.name+'.part')
    shutil.copyfile(source,temporary)
    if sha(temporary)!=sha(source):raise IOError('Persistent copy verification failed')
    temporary.replace(destination)


def save_checkpoint_generation(trainer,persistent,metadata):
    """Keep two complete checkpoint generations; publish pointer only after verified copies."""
    persistent=Path(persistent);base=persistent/'checkpoints';base.mkdir(parents=True,exist_ok=True)
    name=f'epoch-{int(trainer.epoch)+1:04d}'
    generation=base/name;generation.mkdir(exist_ok=True)
    files={}
    for key in ('last','best'):
        source=Path(getattr(trainer,key))
        if source.exists():
            target=generation/f'{key}.pt';atomic_copy(source,target);files[key]=sha(target)
    if 'last' not in files:raise IOError('No resumable checkpoint was saved')
    record={**metadata,'epoch':int(trainer.epoch)+1,'files':files,'generation':name}
    temporary=generation/'state.json.part';temporary.write_text(json.dumps(record,indent=2));temporary.replace(generation/'state.json')
    pointer=base/'latest.json.part';pointer.write_text(json.dumps(record,indent=2));pointer.replace(base/'latest.json')
    # Only generated project checkpoints under this mode directory are eligible for rotation.
    complete=sorted([p for p in base.glob('epoch-*') if (p/'state.json').is_file()])
    for old in complete[:-2]:shutil.rmtree(old)
    return record


def verify_latest(persistent):
    base=Path(persistent)/'checkpoints'
    state=json.loads((base/'latest.json').read_text())
    generation=base/state['generation']
    for key,expected in state['files'].items():
        if sha(generation/f'{key}.pt')!=expected:raise ValueError('Saved checkpoint failed recovery checksum')
    return generation/'last.pt',state


def snapshot_tracking(database,artifact_dir,persistent):
    persistent=Path(persistent);persistent.mkdir(parents=True,exist_ok=True)
    if Path(database).exists():
        temporary=persistent/'mlflow.db.part';temporary.unlink(missing_ok=True)
        with sqlite3.connect(database) as src,sqlite3.connect(temporary) as dst:src.backup(dst)
        temporary.replace(persistent/'mlflow.db')
    if Path(artifact_dir).exists():
        shutil.copytree(artifact_dir,persistent/'mlartifacts',dirs_exist_ok=True)


def run(root,persistent,mode='benchmark',approve_full=False,resume=False,recovery_backend=None):
    root,persistent=Path(root).resolve(),Path(persistent).resolve()
    manifest=verify_bundle(root)
    expected_scope='benchmark' if mode=='benchmark' else 'trainval'
    if manifest['scope']!=expected_scope:raise ValueError('Wrong bundle scope for requested run')
    if mode=='train' and not approve_full:raise ValueError('Full training approval is required; benchmark only is currently authorized')
    # The mounted path must be outside ephemeral /content. A mount-level check follows below.
    if recovery_backend is None:
        if not str(persistent).startswith('/content/drive/MyDrive/'):
            raise ValueError('Use a project directory under mounted Google Drive MyDrive')
        if not os.path.ismount('/content/drive'):
            raise ValueError('Google Drive is not mounted; persistent recovery cannot be promised')
    else:
        # A provider adapter must prove remote persistence before training; local disk is insufficient.
        recovery_backend.validate(persistent, mode, resume)
    import yaml
    import torch
    import mlflow
    from ultralytics import YOLO,settings
    if not torch.cuda.is_available():raise RuntimeError('No CUDA GPU allocated. Stop; never fall back to Mac/CPU or a paid runtime automatically.')
    persistent=persistent/mode;persistent.mkdir(parents=True,exist_ok=True)
    write_probe=persistent/'persistence-probe.txt';write_probe.write_text('mlops-checkpoint-probe')
    if write_probe.read_text()!='mlops-checkpoint-probe':raise IOError('Persistent storage check failed')
    work=Path('/content/mlops_work')/mode;work.mkdir(parents=True,exist_ok=True)
    database=work/'mlflow.db';artifact_dir=work/'mlartifacts'
    if resume and not database.exists() and (persistent/'mlflow.db').exists():
        shutil.copy2(persistent/'mlflow.db',database)
        if (persistent/'mlartifacts').exists():shutil.copytree(persistent/'mlartifacts',artifact_dir,dirs_exist_ok=True)
    mlflow.set_tracking_uri(f'sqlite:///{database}')
    experiment=mlflow.get_experiment_by_name('plastic-waste-phase1-cloud')
    if experiment is None:mlflow.create_experiment('plastic-waste-phase1-cloud',artifact_location=artifact_dir.as_uri())
    mlflow.set_experiment('plastic-waste-phase1-cloud')
    config=yaml.safe_load((root/'config.yaml').read_text());params=dict(config['train'])
    params['model']=str(root/'models/yolo11m.pt');params['device']='0'
    if mode=='benchmark':params.update(epochs=2,close_mosaic=0,plots=False)
    data=yaml.safe_load((root/'dataset/dataset.yaml').read_text());data['path']=str(root/'dataset')
    runtime=work/'dataset.yaml';runtime.write_text(yaml.safe_dump(data))
    settings.update({key:False for key in ('mlflow','wandb','clearml','comet','neptune','tensorboard') if key in settings})
    os.environ['MLFLOW_DISABLE_AGENT_HINT']='1'
    freeze=subprocess.check_output([os.sys.executable,'-m','pip','freeze'],text=True)
    (work/'requirements-cloud-resolved.txt').write_text(freeze)
    gpu={'name':torch.cuda.get_device_name(0),'vram_GiB':torch.cuda.get_device_properties(0).total_memory/1024**3,
        'torch':torch.__version__,'cuda':torch.version.cuda,'python':platform.python_version()}
    (work/'gpu.json').write_text(json.dumps(gpu,indent=2))
    checkpoint=None;run_id=None
    fingerprint=sha(root/'manifest.json')
    if resume:
        checkpoint,state=verify_latest(persistent)
        if state['mode']!=mode or state['bundle_fingerprint']!=fingerprint:raise ValueError('Recovery bundle/mode mismatch')
        run_id=state['run_id']
        restored=work/'runs'/run_id;restored.mkdir(parents=True,exist_ok=True)
        if (persistent/'training').exists():shutil.copytree(persistent/'training',restored,dirs_exist_ok=True)
        for key in state['files']:
            atomic_copy(checkpoint.parent/f'{key}.pt',restored/'weights'/f'{key}.pt')
    with mlflow.start_run(run_id=run_id,run_name=None if resume else f'yolo11m-{mode}-{time.strftime("%Y%m%d-%H%M%S")}') as experiment_run:
        run_id=experiment_run.info.run_id
        mlflow.set_tags({'phase':'1','profile':f'cloud-{mode}','dataset_approval':'pending','bundle_fingerprint':fingerprint,
            'source_git_commit':manifest['source_git_commit'],'not_for_quality_claims':str(mode=='benchmark'),'gpu':gpu['name']})
        if not resume:mlflow.log_params(params)
        for path in [root/'manifest.json',root/'config.yaml',work/'requirements-cloud-resolved.txt',work/'gpu.json']:
            mlflow.log_artifact(str(path),artifact_path='provenance')
        torch.cuda.reset_peak_memory_stats()
        model=YOLO(str(checkpoint) if checkpoint else params['model'])
        metadata={'mode':mode,'run_id':run_id,'bundle_fingerprint':fingerprint}
        saved=[];times=[];start=[None]
        def batch_start(trainer):
            torch.cuda.synchronize();start[0]=time.perf_counter()
        def batch_end(trainer):
            torch.cuda.synchronize();times.append(time.perf_counter()-start[0])
        def on_save(trainer):
            state=save_checkpoint_generation(trainer,persistent,metadata);saved.append(state)
            for name in ('results.csv','args.yaml'):
                p=Path(trainer.save_dir)/name
                if p.exists():atomic_copy(p,persistent/'training'/name)
            mlflow.log_metrics({k.replace('(','_').replace(')','_'):float(v) for k,v in trainer.metrics.items()},step=int(trainer.epoch))
            snapshot_tracking(database,artifact_dir,persistent)
            if recovery_backend is not None:
                recovery_backend.publish(persistent)
        model.add_callback('on_train_batch_start',batch_start);model.add_callback('on_train_batch_end',batch_end)
        model.add_callback('on_model_save',on_save)
        started=time.perf_counter()
        try:
            if resume:result=model.train(resume=True,device='0')
            else:result=model.train(data=str(runtime),project=str(work/'runs'),name=run_id,exist_ok=False,save_period=-1,**{k:v for k,v in params.items() if k!='model'})
            last,state=verify_latest(persistent)
            # Verify the recovery payload actually contains epoch + optimizer state, without loading CUDA tensors.
            payload=torch.load(last,map_location='cpu',weights_only=False)
            recoverable=payload.get('optimizer') is not None and payload.get('epoch',-1)>=0
            del payload
            record={**metadata,'status':'completed','gpu':gpu,'elapsed_seconds':time.perf_counter()-started,
                'peak_cuda_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3,
                'peak_cuda_reserved_GiB':torch.cuda.max_memory_reserved()/1024**3,
                'mean_batch_compute_seconds_after10':sum(times[10:])/max(1,len(times[10:])),
                'persistent_checkpoint_verified':True,'optimizer_state_verified':recoverable,
                'saved_generations':len(saved),'dataset_images':manifest['train_images'],
                'quality_claim':False if mode=='benchmark' else 'Validation only; separate final evaluation required',
                'note':'Batch timing excludes loader time; use epoch results.csv for full-run ETA. No 70% performance claim.'}
            if mode=='benchmark' and not recoverable:raise RuntimeError('Benchmark recovery payload lacks optimizer state')
            output=work/f'{mode}_report.json';output.write_text(json.dumps(record,indent=2))
            atomic_copy(output,persistent/output.name);mlflow.log_artifact(str(output))
            if mode=='train':
                destination=work/'yolo11m-waste-best.pt';shutil.copy2(model.trainer.best,destination)
                atomic_copy(destination,persistent/destination.name);mlflow.log_artifact(str(destination),artifact_path='weights')
            for p in Path(model.trainer.save_dir).iterdir():
                if p.suffix in ('.csv','.yaml','.png','.jpg'):
                    mlflow.log_artifact(str(p),artifact_path='training')
                    atomic_copy(p,persistent/'training'/p.name)
            print(json.dumps(record,indent=2))
        finally:
            # SQLite snapshots use its backup API rather than copying a live WAL database.
            snapshot_tracking(database,artifact_dir,persistent)
    snapshot_tracking(database,artifact_dir,persistent)
    if recovery_backend is not None:
        recovery_backend.publish(persistent)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['benchmark','train']);parser.add_argument('--root',required=True);parser.add_argument('--persistent',required=True)
    parser.add_argument('--approve-full',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();run(args.root,args.persistent,args.mode,args.approve_full,args.resume)
