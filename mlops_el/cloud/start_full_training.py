"""Start/resume the user-authorized full experiment after persistent input verification."""
import json
import gc
import importlib.metadata
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

import mlops_cloud_runner as runner

PROJECT = Path('/content/drive/MyDrive/MLOPS_EL_cloud')
ARCHIVE = PROJECT / 'mlops-trainval-labels-v3.tar'
ARCHIVE_BYTES = 6700103680
ARCHIVE_SHA256 = '37c7e50b954a996519fc8d4f52715fc93c733b313374ab4f2b91af9725064098'
ROOT = Path('/content/mlops_trainval_bundle')


def main():
    drill_path = PROJECT / 'recovery-drill/recovery_drill_report.json'
    assert drill_path.is_file(), 'Complete checkpoint recovery verification first.'
    drill = json.loads(drill_path.read_text())
    assert drill['status'] == 'passed' and drill['same_mlflow_run']
    completed = PROJECT / 'train/train_report.json'
    if completed.exists() and json.loads(completed.read_text()).get('status') == 'completed':
        print('This full experiment is already complete. Review it before starting another run.')
        return
    deadline = time.monotonic() + 7200
    while not ARCHIVE.is_file() or ARCHIVE.stat().st_size != ARCHIVE_BYTES:
        if time.monotonic() > deadline:
            raise TimeoutError('Full archive upload is incomplete; finish the Drive upload and rerun.')
        print('Waiting for complete persistent dataset upload:', ARCHIVE, flush=True)
        time.sleep(30)
    if not ROOT.exists():
        local_archive = Path('/content/mlops-trainval-labels-v3.tar')
        assert shutil.disk_usage('/content').free > 2 * ARCHIVE_BYTES + 5 * 1024**3, 'Insufficient runtime disk headroom'
        print('Copying archive from Drive to local runtime disk.', flush=True)
        runner.atomic_copy(ARCHIVE, local_archive)
        print('Checking archive and every payload file before training.', flush=True)
        runner.unpack(local_archive, ROOT, ARCHIVE_SHA256)
    manifest = runner.verify_bundle(ROOT)
    assert manifest['scope'] == 'trainval'
    assert manifest['train_images'] == 7000 and manifest['val_images'] == 1500
    # Refuse to overwrite incomplete evidence from a start that never produced a recovery checkpoint.
    checkpoint = PROJECT / 'train/checkpoints/latest.json'
    mode_dir = PROJECT / 'train'
    resume = checkpoint.is_file()
    if mode_dir.exists() and not resume:
        raise RuntimeError('An earlier attempt exists without a checkpoint. Inspect it before retrying.')
    # Preserve each session's environment independently of the runner's latest MLflow snapshot.
    import torch
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'torchvision', 'ultralytics', 'mlflow')}
    sessions = PROJECT / 'train-session-records'
    previous = sorted(sessions.glob('*/session.json')) if sessions.exists() else []
    if resume and previous:
        initial = json.loads(previous[0].read_text())['versions']
        assert versions == initial, f'Cloud dependency versions changed: {versions}; restore {initial} before resuming.'
    session = sessions / (time.strftime('%Y%m%d-%H%M%S', time.gmtime()) + '-' + uuid.uuid4().hex[:8])
    session.mkdir(parents=True)
    record = {'versions': versions, 'gpu': torch.cuda.get_device_name(0), 'resume': resume,
              'archive_sha256': ARCHIVE_SHA256, 'runner_sha256': runner.sha(Path(runner.__file__)),
              'bundle_fingerprint': runner.sha(ROOT / 'manifest.json')}
    (session / 'session.json').write_text(json.dumps(record, indent=2))
    (session / 'requirements-resolved.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    runner.atomic_copy(ROOT / 'config.yaml', session / 'config.yaml')
    runner.atomic_copy(ROOT / 'manifest.json', session / 'manifest.json')
    if resume:
        runner.atomic_copy(checkpoint, session / 'resume-from.json')
    approval = {'user_authorized_on': '2026-11-06', 'model': 'YOLO11m',
                'epochs_max': 100, 'patience': 20, 'imgsz': 960, 'batch': 4,
                'free_gpu_only': True, 'archive_sha256': ARCHIVE_SHA256,
                'runner_sha256': runner.sha(Path(runner.__file__)),
                'dataset_approval': 'faculty approval pending',
                'target_validation_map50': 0.70, 'target_is_not_a_guarantee': True}
    (PROJECT / 'full_training_authorization.json').write_text(json.dumps(approval, indent=2))
    print('Starting full experiment:', 'RESUME' if resume else 'NEW', '7000 train / 1500 validation; test excluded.', flush=True)
    gc.collect()
    torch.cuda.empty_cache()
    runner.run(ROOT, PROJECT, mode='train', approve_full=True, resume=resume)


if __name__ == '__main__':
    main()
