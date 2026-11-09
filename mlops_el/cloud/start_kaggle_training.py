"""Resume the same experiment only after private remote backup round-trip succeeds."""
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch
import runner
from kaggle_backup import KaggleBackup
from verify_migration import RUN_ID, EXPECTED_VERSIONS, verify_versions, verify_recovery


def link_existing_path(alias, target):
    alias, target = Path(alias), Path(target)
    if alias.exists() or alias.is_symlink():
        if alias.resolve() != target.resolve():
            raise RuntimeError(f'Existing path conflicts with original MLflow/checkpoint path: {alias}')
        return
    alias.parent.mkdir(parents=True, exist_ok=True)
    alias.symlink_to(target, target_is_directory=True)


def main(initialize_backup=False):
    assert torch.cuda.is_available(), 'Free CUDA GPU required; no CPU fallback.'
    verify_versions({name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS})
    root = Path('/kaggle/temp/mlops_trainval_bundle')
    manifest = runner.verify_bundle(root)
    assert manifest['scope'] == 'trainval'
    report = json.loads(Path('/kaggle/working/mlops_migration/preflight.json').read_text())
    assert report['optimizer_state_verified'] and report['run_id'] == RUN_ID
    from kaggle_secrets import UserSecretsClient
    os.environ['KAGGLE_API_TOKEN'] = UserSecretsClient().get_secret('MLOPS_EL_KAGGLE_API_TOKEN')
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    backend = KaggleBackup(api)
    project = Path('/kaggle/working/MLOPS_EL_cloud')
    train = project / 'train'
    if initialize_backup:
        if train.exists():
            raise FileExistsError('Initial migration attempt already exists; inspect before retrying.')
        project.mkdir(parents=True, exist_ok=True)
        shutil.copytree('/kaggle/working/mlops_migration/train', train)
    else:
        # Restore the authoritative remote version, never silently reuse an older input snapshot.
        imported = Path('/kaggle/temp') / ('mlops-resume-' + str(time.time_ns()))
        backend.restore(imported)
        if train.exists():
            local_last, local_state = runner.verify_latest(train)
            remote_last, remote_state = runner.verify_latest(imported / 'train')
            if local_state['epoch'] > remote_state['epoch']:
                raise RuntimeError('Local progress is newer than remote: preserve and inspect before resuming.')
            train.rename(project / ('train-preserved-' + str(time.time_ns())))
        else:
            project.mkdir(parents=True, exist_ok=True)
        shutil.copytree(imported / 'train', train)
    state = verify_recovery(train, root / 'manifest.json')
    if (train / 'train_report.json').exists():
        if json.loads((train / 'train_report.json').read_text()).get('status') == 'completed':
            print('Existing experiment completed; no training started.', flush=True)
            return
    last, pointer = runner.verify_latest(train)
    payload = torch.load(last, map_location='cpu', weights_only=False)
    assert payload.get('optimizer') is not None and payload['epoch'] + 1 == state['epoch']
    args = payload['train_args']
    assert (args['imgsz'], args['batch'], args['epochs'], args['patience']) == (960, 4, 100, 20)
    del payload
    work = Path('/kaggle/working/mlops_work')
    work.mkdir(parents=True, exist_ok=True)
    link_existing_path('/content/mlops_work', work)
    link_existing_path('/content/mlops_trainval_bundle', root)
    session = project / 'sessions' / str(time.time_ns())
    session.mkdir(parents=True)
    session_record = {'run_id': RUN_ID, 'provider': 'kaggle', 'resume_epoch': state['epoch'],
                      'colab_on_hold': True, 'settings': {'imgsz': 960, 'batch': 4, 'epochs': 100, 'patience': 20},
                      'versions': {name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS},
                      'runner_sha256': runner.sha(Path(runner.__file__)), 'backup_sha256': runner.sha(Path(sys.modules['kaggle_backup'].__file__)),
                      'gpu': torch.cuda.get_device_name(0)}
    (session / 'session.json').write_text(json.dumps(session_record, indent=2))
    (session / 'requirements-resolved.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    provenance = train / 'training' / ('kaggle-session-' + session.name)
    shutil.copytree(session, provenance)
    for module in (Path(__file__), Path(runner.__file__), Path(sys.modules['kaggle_backup'].__file__)):
        shutil.copyfile(module, provenance / module.name)
    # Each later on_model_save callback repeats full upload/download/hash verification.
    backend.publish(train, create=initialize_backup)
    print(f'Resuming SAME MLflow run {RUN_ID} after verified epoch {state["epoch"]}; Colab held stopped.', flush=True)
    runner.run(root, project, mode='train', approve_full=True, resume=True, recovery_backend=backend)


if __name__ == '__main__':
    main(initialize_backup='--initialize-backup' in sys.argv)
