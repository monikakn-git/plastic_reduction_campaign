"""Private remote checkpoint versions; local Kaggle files alone are not durable."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import tarfile
import time
import uuid
import zipfile

from runner import sha, verify_latest
from verify_migration import RUN_ID


class KaggleBackup:
    slug = 'aayushbmlops26/mlops-el-yolo11m-checkpoints-private'

    def __init__(self, api):
        self.api = api
        self.verified_epoch = None
        self.stage = Path('/kaggle/temp/mlops_backup_upload')
        self.stage.mkdir(parents=True, exist_ok=True)

    def is_private(self):
        if hasattr(self.api, 'dataset_view'):
            view = self.api.dataset_view(self.slug)
            data = view.to_dict() if hasattr(view, 'to_dict') else vars(view)
            private = data.get('is_private', data.get('isPrivate'))
        else:
            # The authenticated owned-dataset listing exposes the privacy bit
            # even while a newly created dataset's detail endpoint is pending.
            matches = [item for item in (self.api.dataset_list(mine=True) or [])
                       if getattr(item, 'ref', None) == self.slug]
            private = getattr(matches[0], 'is_private', None) if len(matches) == 1 else None
        if private is not True:
            raise RuntimeError('Remote checkpoint dataset is not confirmed private')

    def download_file(self, name, destination):
        destination = Path(destination)
        target = destination / name
        # Kaggle 2.0.2 wraps large downloads in ZIP and has no unzip argument.
        target.unlink(missing_ok=True)
        (destination / (name + '.zip')).unlink(missing_ok=True)
        with contextlib.redirect_stdout(io.StringIO()):
            self.api.dataset_download_file(self.slug, name, str(destination), force=True, quiet=True)
        if not target.is_file():
            wrapped = destination / (name + '.zip')
            with zipfile.ZipFile(wrapped) as z:
                members = z.infolist()
                if len(members) != 1 or Path(members[0].filename).name != name or members[0].is_dir():
                    raise ValueError('Unexpected wrapped download members')
                with z.open(members[0]) as src, target.open('wb') as dst:
                    shutil.copyfileobj(src, dst, 1024 * 1024)
            wrapped.unlink()
        return target

    def publish(self, persistent, create=False):
        persistent = Path(persistent)
        _, state = verify_latest(persistent)
        if state['run_id'] != RUN_ID or state['mode'] != 'train':
            raise ValueError('Refusing backup for another experiment')
        if not create:
            self.is_private()
        # Only the latest complete generation goes in each remote version.
        # Earlier remote versions preserve earlier generations; no remote deletion occurs.
        archive = self.stage / 'recovery.bin'
        generation = 'checkpoints/' + state['generation']
        with tarfile.open(archive, 'w:') as tar:
            for relative in ('checkpoints/latest.json', generation, 'mlflow.db', 'mlartifacts', 'training'):
                path = persistent / relative
                if not path.exists():
                    raise FileNotFoundError(f'Missing recovery artifact: {relative}')
                tar.add(path, arcname='train/' + relative, recursive=True)
            for optional in ('train_report.json', 'yolo11m-waste-best.pt'):
                if (persistent / optional).is_file():
                    tar.add(persistent / optional, arcname='train/' + optional)
        receipt = {'run_id': RUN_ID, 'epoch': state['epoch'], 'provider': 'kaggle',
                   'recovery_sha256': sha(archive), 'recovery_bytes': archive.stat().st_size,
                   'checkpoint_files': state['files'], 'nonce': uuid.uuid4().hex}
        (self.stage / 'receipt.json').write_text(json.dumps(receipt, indent=2))
        (self.stage / 'dataset-metadata.json').write_text(json.dumps({
            'id': self.slug, 'title': 'MLOPS EL YOLO11m checkpoints private',
            'licenses': [{'name': 'other'}]}, indent=2))
        if create:
            response = self.api.dataset_create_new(str(self.stage), public=False, quiet=True, convert_to_csv=False)
        else:
            response = self.api.dataset_create_version(str(self.stage), f'Same run epoch {state["epoch"]}',
                                                       quiet=True, convert_to_csv=False, delete_old_versions=False)
        if getattr(response, 'error', None):
            raise RuntimeError('Kaggle rejected the checkpoint version; inspect remote dataset before retrying.')
        download = Path('/kaggle/temp/mlops_backup_verify')
        download.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            try:
                self.is_private()
                if self.api.dataset_status(self.slug).lower() != 'ready':
                    time.sleep(10)
                    continue
                # Suppress SDK URL/license chatter; never log credentials or raw error responses.
                self.download_file('receipt.json', download)
                actual = json.loads((download / 'receipt.json').read_text())
                if actual != receipt:
                    time.sleep(10)
                    continue
                self.download_file('recovery.bin', download)
                restored = download / 'recovery.bin'
                if restored.stat().st_size != receipt['recovery_bytes'] or sha(restored) != receipt['recovery_sha256']:
                    raise RuntimeError('Remote recovery archive checksum mismatch')
                self.verified_epoch = state['epoch']
                (persistent / 'kaggle-remote-receipt.json').write_text(json.dumps(receipt, indent=2))
                # Remove disposable verification downloads after a successful hash check.
                shutil.rmtree(download)
                print(f'PRIVATE REMOTE BACKUP VERIFIED: epoch {self.verified_epoch}', flush=True)
                return receipt
            except (FileNotFoundError, json.JSONDecodeError):
                time.sleep(10)
        raise TimeoutError('Remote checkpoint version was not verified; stop training and inspect recovery.')

    def validate(self, project, mode, resume):
        project = Path(project).resolve()
        if str(project) != '/kaggle/working/MLOPS_EL_cloud' or mode != 'train' or not resume:
            raise ValueError('Kaggle adapter permits only the existing resumed full experiment')
        _, state = verify_latest(project / mode)
        if self.verified_epoch != state['epoch']:
            raise RuntimeError('Current recovery generation has not passed remote round-trip verification')
        self.is_private()

    def restore(self, destination):
        """Fetch a complete remote generation into a new, empty recovery directory."""
        destination = Path(destination)
        if destination.exists():
            raise FileExistsError('Recovery destination must be fresh; preserve existing attempts')
        self.is_private()
        download = Path('/kaggle/temp/mlops_remote_restore')
        download.mkdir(parents=True, exist_ok=True)
        for name in ('receipt.json', 'recovery.bin'):
            self.download_file(name, download)
        receipt = json.loads((download / 'receipt.json').read_text())
        archive = download / 'recovery.bin'
        if receipt['run_id'] != RUN_ID or sha(archive) != receipt['recovery_sha256']:
            raise ValueError('Remote recovery identity/hash mismatch')
        with tarfile.open(archive, 'r:') as tar:
            from runner import safe_name
            for member in tar.getmembers():
                path = safe_name(member.name)
                if path.parts[0] != 'train' or not (member.isfile() or member.isdir()):
                    raise ValueError('Unsafe recovery archive member')
            destination.mkdir(parents=True)
            tar.extractall(destination, filter='data')
        verify_latest(destination / 'train')
        shutil.rmtree(download)
        return receipt
