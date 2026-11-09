"""Verify the migrated experiment on Kaggle without starting training."""
import importlib.metadata
import json
from pathlib import Path
import shutil
import stat
import zipfile
from concurrent.futures import ThreadPoolExecutor

import torch
from runner import sha, unpack, verify_bundle
from verify_migration import EXPECTED_VERSIONS, verify_recovery, verify_versions

ARCHIVE_SHA = '37c7e50b954a996519fc8d4f52715fc93c733b313374ab4f2b91af9725064098'
RECOVERY_SHA = '8bfdef12a7918ccb452a590815fcb2804faa333626d608b75aecc849e4356949'


def buffered_copy(source, destination):
    # Avoid sendfile on Kaggle's input filesystem; use bounded buffered reads.
    with open(source, 'rb') as src, open(destination, 'wb') as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    return str(destination)


def input_root(inputs, slug):
    candidates = [inputs / 'datasets/aayushbmlops26' / slug, inputs / slug]
    found = [p for p in candidates if p.is_dir()]
    if len(found) != 1:
        raise ValueError(f'Expected one attached input at its known mount path: {slug}')
    return found[0]


def main():
    assert torch.cuda.is_available(), 'A free CUDA GPU is required.'
    actual = {name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS}
    verify_versions(actual)
    inputs = Path('/kaggle/input')
    data_input = input_root(inputs, 'mlops-el-labels-v3-trainval-private')
    recovery_input = input_root(inputs, 'mlops-el-epoch30-recovery-private')
    work = Path('/kaggle/working/mlops_migration')
    work.mkdir(parents=True, exist_ok=True)
    root = Path('/kaggle/temp/mlops_trainval_bundle')
    print('Staging full train/validation input from known mount path.', flush=True)
    archive = data_input / 'mlops-trainval-labels-v3.tar'
    if archive.is_file():
        if not root.exists():
            unpack(archive, root, ARCHIVE_SHA)
    else:
        source = data_input
        assert (source / 'dataset/images/train').is_dir(), 'Complete input layout is required.'
        manifest = json.loads((source / 'manifest.json').read_text())
        canonical = json.dumps(manifest, indent=2)
        root.mkdir(parents=True, exist_ok=True)
        target_manifest = root / 'manifest.json'
        if target_manifest.exists() and target_manifest.read_text() != canonical:
            raise ValueError('An existing staging directory belongs to a different bundle.')
        from runner import safe_name
        def copy_payload(row):
            relative = safe_name(row['path'])
            src, dst = source.joinpath(*relative.parts), root.joinpath(*relative.parts)
            if not dst.is_file() or dst.stat().st_size != row['bytes']:
                dst.parent.mkdir(parents=True, exist_ok=True)
                buffered_copy(src, dst)
        with ThreadPoolExecutor(max_workers=8) as pool:
            for count, _ in enumerate(pool.map(copy_payload, manifest['files']), 1):
                if count % 1000 == 0:
                    print(f'Staged {count}/{len(manifest["files"])} payload files', flush=True)
        target_manifest.write_text(canonical)
    print('Verifying all payload sizes, hashes and split counts.', flush=True)
    verify_bundle(root)
    train = work / 'train'
    source_train = recovery_input / 'train'
    if source_train.is_dir():
        print('Copying immutable recovery input with buffered I/O.', flush=True)
        shutil.copytree(source_train, train, dirs_exist_ok=True, copy_function=buffered_copy)
    elif not train.exists():
        recovery_zip = recovery_input / 'train-20261106T105552Z-1-001.zip'
        if recovery_zip.is_file():
            assert sha(recovery_zip) == RECOVERY_SHA, 'Recovery archive checksum mismatch.'
            with zipfile.ZipFile(recovery_zip) as z:
                for item in z.infolist():
                    path = Path(item.filename)
                    assert not path.is_absolute() and '..' not in path.parts
                    assert not stat.S_ISLNK(item.external_attr >> 16), 'Archive links rejected.'
                z.extractall(work)
        else:
            source_train = recovery_input / 'train'
            assert (source_train / 'checkpoints/latest.json').is_file(), 'Complete recovery input is required.'
            shutil.copytree(source_train, train, copy_function=buffered_copy)
    print('Verifying epoch-30 checkpoint, MLflow history and optimizer payload.', flush=True)
    record = verify_recovery(train, root / 'manifest.json')
    last = train / 'checkpoints' / f'epoch-{record["epoch"]:04d}' / 'last.pt'
    payload = torch.load(last, map_location='cpu', weights_only=False)
    assert payload.get('optimizer') is not None, 'Optimizer state missing; cannot resume same training.'
    assert payload['epoch'] + 1 == record['epoch'], 'Checkpoint epoch differs from saved metadata.'
    args = payload['train_args']
    assert (args['imgsz'], args['batch'], args['epochs'], args['patience']) == (960, 4, 100, 20)
    record.update(status='preflight_passed_training_not_started', versions=actual,
                  optimizer_state_verified=True, checkpoint_internal_epoch=payload['epoch'],
                  dataset_counts={'train': 7000, 'validation': 1500, 'test': 0},
                  gpu=torch.cuda.get_device_name(0),
                  durable_backup_verified=False,
                  next_gate='Verify durable private backups and same-run restoration before any training.')
    del payload
    (work / 'preflight.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
