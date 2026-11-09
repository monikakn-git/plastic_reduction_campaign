from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'cloud'))
from kaggle_backup import KaggleBackup
from runner import save_checkpoint_generation
from verify_migration import RUN_ID


class FakeAPI:
    def __init__(self, remote, private=True, corrupt=False):
        self.remote, self.private, self.corrupt = remote, private, corrupt

    def dataset_view(self, slug):
        return SimpleNamespace(to_dict=lambda: {'is_private': self.private})

    def dataset_create_new(self, folder, **kwargs):
        assert kwargs['public'] is False
        self.remote.mkdir()
        for name in ('receipt.json', 'recovery.bin'):
            shutil.copyfile(Path(folder) / name, self.remote / name)

    def dataset_status(self, slug):
        return 'ready'

    def dataset_download_file(self, slug, name, path, **kwargs):
        target = Path(path) / name
        shutil.copyfile(self.remote / name, target)
        if self.corrupt and name == 'recovery.bin':
            target.write_bytes(b'corrupted remote payload')


def test_unconfirmed_privacy_is_rejected(tmp_path):
    backend = object.__new__(KaggleBackup)
    backend.api = FakeAPI(tmp_path / 'remote', private=False)
    with pytest.raises(RuntimeError, match='not confirmed private'):
        backend.is_private()


def test_default_runner_still_requires_mounted_drive(tmp_path, monkeypatch):
    import runner
    monkeypatch.setattr(runner, 'verify_bundle', lambda root: {'scope': 'trainval'})
    with pytest.raises(ValueError, match='mounted Google Drive'):
        runner.run(tmp_path, tmp_path, mode='train', approve_full=True, resume=True)


def test_remote_validation_must_pass_before_training(tmp_path, monkeypatch):
    import runner
    monkeypatch.setattr(runner, 'verify_bundle', lambda root: {'scope': 'trainval'})
    def reject(project, mode, resume):
        assert mode == 'train' and resume
        raise RuntimeError('Remote recovery not verified')
    backend = SimpleNamespace(validate=reject)
    with pytest.raises(RuntimeError, match='Remote recovery not verified'):
        runner.run(tmp_path, tmp_path, mode='train', approve_full=True, resume=True, recovery_backend=backend)


@pytest.mark.parametrize('corrupt', [False, True])
def test_uploaded_backup_must_pass_full_download_hash(tmp_path, monkeypatch, corrupt):
    import kaggle_backup
    # Keep every test artifact in pytest's sandbox rather than a Kaggle path.
    original_path = Path
    monkeypatch.setattr(kaggle_backup, 'Path', lambda p: tmp_path / 'verify' if str(p) == '/kaggle/temp/mlops_backup_verify' else original_path(p))
    persistent = tmp_path / 'train'
    weights = tmp_path / 'weights'
    weights.mkdir()
    for name in ('last', 'best'):
        (weights / f'{name}.pt').write_bytes(b'fixture optimizer checkpoint')
    save_checkpoint_generation(SimpleNamespace(epoch=29, last=weights / 'last.pt', best=weights / 'best.pt'),
                               persistent, {'run_id': RUN_ID, 'mode': 'train'})
    (persistent / 'mlflow.db').write_bytes(b'fixture tracking snapshot')
    (persistent / 'training').mkdir()
    (persistent / 'mlartifacts').mkdir()
    backend = object.__new__(KaggleBackup)
    backend.api = FakeAPI(tmp_path / 'remote', corrupt=corrupt)
    backend.stage = tmp_path / 'stage'
    backend.stage.mkdir()
    backend.verified_epoch = None
    if corrupt:
        with pytest.raises(RuntimeError, match='checksum mismatch'):
            backend.publish(persistent, create=True)
        assert backend.verified_epoch is None
    else:
        receipt = backend.publish(persistent, create=True)
        assert receipt['epoch'] == backend.verified_epoch == 30
