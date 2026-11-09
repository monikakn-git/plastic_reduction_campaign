import json
from pathlib import Path
import sqlite3
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'cloud'))
from runner import sha
from verify_migration import verify_recovery, verify_versions, EXPECTED_VERSIONS, RUN_ID


@pytest.fixture
def recovery(tmp_path):
    train = tmp_path / 'train'
    generation = train / 'checkpoints/epoch-0030'
    generation.mkdir(parents=True)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{"fixture": true}')
    for name in ('last', 'best'):
        (generation / f'{name}.pt').write_bytes(b'fixture checkpoint')
    pointer = {'epoch': 30, 'generation': 'epoch-0030', 'run_id': RUN_ID, 'mode': 'train',
               'bundle_fingerprint': sha(manifest),
               'files': {name: sha(generation / f'{name}.pt') for name in ('last', 'best')}}
    (train / 'checkpoints/latest.json').write_text(json.dumps(pointer))
    (generation / 'state.json').write_text(json.dumps(pointer))
    (train / 'training').mkdir()
    for name in ('args.yaml', 'results.csv'):
        (train / 'training' / name).write_text('fixture')
    (train / 'mlartifacts').mkdir()
    with sqlite3.connect(train / 'mlflow.db') as db:
        db.execute('CREATE TABLE runs (run_uuid TEXT, artifact_uri TEXT)')
        db.execute('INSERT INTO runs VALUES (?,?)', (RUN_ID, 'file:///content/mlops_work/train/mlartifacts/run'))
    return train, manifest


def test_original_identity_and_hashes_are_required(recovery):
    train, manifest = recovery
    assert verify_recovery(train, manifest)['epoch'] == 30
    (train / 'checkpoints/epoch-0030/last.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='hash mismatch'):
        verify_recovery(train, manifest)


def test_wrong_run_and_generation_are_rejected(recovery):
    train, manifest = recovery
    path = train / 'checkpoints/latest.json'
    pointer = json.loads(path.read_text())
    pointer['run_id'] = 'another-run'
    path.write_text(json.dumps(pointer))
    with pytest.raises(ValueError, match='Wrong experiment'):
        verify_recovery(train, manifest)
    pointer.update(run_id=RUN_ID, generation='../escape')
    path.write_text(json.dumps(pointer))
    with pytest.raises(ValueError, match='generation'):
        verify_recovery(train, manifest)


def test_missing_tracking_run_is_rejected(recovery):
    train, manifest = recovery
    with sqlite3.connect(train / 'mlflow.db') as db:
        db.execute('DELETE FROM runs')
    with pytest.raises(ValueError, match='MLflow run is missing'):
        verify_recovery(train, manifest)


def test_dependencies_must_match_exactly():
    verify_versions(EXPECTED_VERSIONS.copy())
    with pytest.raises(ValueError, match='Exact training dependencies'):
        verify_versions({**EXPECTED_VERSIONS, 'torch': '2.15.0'})
