"""Read-only checks before moving the existing cloud experiment between providers."""
import argparse
import json
from pathlib import Path
import sqlite3
from runner import sha

RUN_ID = '85714466948d4e658ebb610a475dcf57'
EXPECTED_VERSIONS = {'torch': '2.11.0+cu130', 'torchvision': '0.26.0+cu130',
                     'ultralytics': '8.4.201', 'mlflow': '3.17.2'}


def verify_recovery(train_directory, manifest_path, expected_run=RUN_ID, minimum_epoch=30):
    train = Path(train_directory)
    pointer = json.loads((train / 'checkpoints/latest.json').read_text())
    epoch = pointer['epoch']
    if type(epoch) is not int or epoch < minimum_epoch:
        raise ValueError('Recovery would roll back completed epochs')
    if pointer['run_id'] != expected_run or pointer['mode'] != 'train':
        raise ValueError('Wrong experiment or recovery mode')
    if pointer['generation'] != f'epoch-{epoch:04d}':
        raise ValueError('Unsafe or inconsistent checkpoint generation')
    if pointer['bundle_fingerprint'] != sha(Path(manifest_path)):
        raise ValueError('Dataset bundle fingerprint changed')
    if set(pointer['files']) != {'last', 'best'}:
        raise ValueError('Both last and best weights are required')
    generation = train / 'checkpoints' / pointer['generation']
    if json.loads((generation / 'state.json').read_text()) != pointer:
        raise ValueError('Checkpoint generation metadata differs from latest pointer')
    for key, digest in pointer['files'].items():
        if sha(generation / f'{key}.pt') != digest:
            raise ValueError('Checkpoint hash mismatch')
    database = train / 'mlflow.db'
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as db:
        if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('Tracking database integrity check failed')
        row = db.execute('SELECT run_uuid, artifact_uri FROM runs WHERE run_uuid = ?', (expected_run,)).fetchone()
        if row is None:
            raise ValueError('Same MLflow run is missing from recovery database')
        artifact_uri = row[1]
    for filename in ('args.yaml', 'results.csv'):
        if not (train / 'training' / filename).is_file():
            raise ValueError('Training history/configuration is missing')
    if not (train / 'mlartifacts').is_dir():
        raise ValueError('Tracking artifacts are missing')
    return {'run_id': expected_run, 'epoch': epoch, 'checkpoint_sha256': pointer['files'],
            'bundle_fingerprint': pointer['bundle_fingerprint'], 'sqlite_integrity': 'ok',
            'original_artifact_uri': artifact_uri, 'weights_and_tracking_verified': True,
            'optimizer_state_verified': False,
            'note': 'Load checkpoint with the pinned Torch environment and verify optimizer/epoch before training. No training performed.'}


def verify_versions(actual):
    if actual != EXPECTED_VERSIONS:
        raise ValueError(f'Exact training dependencies required: {EXPECTED_VERSIONS}; found {actual}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-directory', required=True)
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    print(json.dumps(verify_recovery(args.train_directory, args.manifest), indent=2))
