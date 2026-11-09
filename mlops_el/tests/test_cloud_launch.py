"""Protect long cloud runs from incomplete inputs and accidental restarts."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture
def launch(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('mlops_cloud_runner', root / 'cloud/runner.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setitem(sys.modules, 'mlops_cloud_runner', runner)
    spec = importlib.util.spec_from_file_location('cloud_launch_under_test', root / 'cloud/start_full_training.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PROJECT = tmp_path
    module.ARCHIVE = tmp_path / 'archive.tar'
    module.ROOT = tmp_path / 'bundle'
    return module


def approved_drill(module):
    path = module.PROJECT / 'recovery-drill/recovery_drill_report.json'
    path.parent.mkdir()
    path.write_text(json.dumps({'status': 'passed', 'same_mlflow_run': True}))


def test_requires_successful_recovery_drill(launch):
    with pytest.raises(AssertionError, match='recovery verification'):
        launch.main()


def test_completed_experiment_is_not_restarted(launch, capsys):
    approved_drill(launch)
    path = launch.PROJECT / 'train/train_report.json'
    path.parent.mkdir()
    path.write_text(json.dumps({'status': 'completed'}))
    launch.main()
    assert 'already complete' in capsys.readouterr().out


def test_partial_upload_times_out_before_training(launch, monkeypatch):
    approved_drill(launch)
    launch.ARCHIVE.write_bytes(b'incomplete')
    moments = iter([0, 7201])
    monkeypatch.setattr(launch.time, 'monotonic', lambda: next(moments))
    with pytest.raises(TimeoutError, match='incomplete'):
        launch.main()
    assert not (launch.PROJECT / 'train').exists()


def test_attempt_without_checkpoint_is_not_overwritten(launch, monkeypatch):
    approved_drill(launch)
    launch.ARCHIVE.write_bytes(b'')
    launch.ARCHIVE_BYTES = 0
    launch.ROOT.mkdir()
    (launch.PROJECT / 'train').mkdir()
    monkeypatch.setattr(launch.runner, 'verify_bundle', lambda root: {'scope': 'trainval', 'train_images': 7000, 'val_images': 1500})
    with pytest.raises(RuntimeError, match='without a checkpoint'):
        launch.main()
