"""Bounded MPS feasibility test; does not resume or alter the cloud experiment."""
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import time
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.local/ultralytics'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.local/matplotlib'))
import mlflow
import torch
import yaml
from ultralytics import YOLO, settings


def main():
    if not torch.backends.mps.is_available():
        raise SystemExit('Apple MPS required; CPU fallback is not used.')
    weights = ROOT / 'models/yolo11m.pt'
    weight_sha = hashlib.sha256(weights.read_bytes()).hexdigest()
    assert weight_sha == 'd5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95'
    source = ROOT / '.local/cloud_export_roundtrip/dataset'
    runtime = ROOT / '.local/yolo11m-mac-benchmark'
    runtime.mkdir(parents=True, exist_ok=True)
    identities = {}
    for split, count in [('train', 32), ('val', 8)]:
        paths = sorted((source / 'images' / split).iterdir())[:count]
        assert len(paths) == count
        (runtime / f'{split}.txt').write_text('\n'.join(map(str, paths)) + '\n')
        for path in paths:
            label = source / 'labels' / split / (path.stem + '.txt')
            for item in (path, label):
                identities[str(item.relative_to(ROOT))] = hashlib.sha256(item.read_bytes()).hexdigest()
    data = yaml.safe_load((source / 'dataset.yaml').read_text())
    data.update(path=str(source), train=str(runtime / 'train.txt'), val=str(runtime / 'val.txt'))
    data.pop('test', None)
    (runtime / 'dataset.yaml').write_text(yaml.safe_dump(data))
    settings.update({k: False for k in ('mlflow', 'wandb', 'clearml', 'comet', 'neptune', 'tensorboard') if k in settings})
    config = yaml.safe_load((ROOT / 'configs/yolo11m_cloud_proposed.yaml').read_text())['train']
    config.pop('model')
    config.update(epochs=1, device='mps', workers=0, amp=False, plots=False, close_mosaic=0)
    name = 'yolo11m-mac-feasibility-' + datetime.now().strftime('%Y%m%d-%H%M%S')
    report_path = ROOT / 'reports/phase1/yolo11m_mac_benchmark.json'
    record = {'status': 'running', 'scope': 'engineering feasibility only', 'training_images': 32,
              'smoke_images': 8, 'model': 'YOLO11m', 'imgsz': 960, 'batch': 4,
              'source': 'labels-v3 cloud benchmark export; both subsets originated in training split',
              'cloud_run_unchanged': '85714466948d4e658ebb610a475dcf57',
              'torch': torch.__version__, 'pretrained_sha256': weight_sha,
              'mps_recommended_max_bytes': torch.mps.recommended_max_memory(),
              'settings': config, 'input_sha256': identities, 'batches': [],
              'limitations': ['Small sample does not establish accuracy or sustained thermal performance.',
                              'MPS uses FP32 here; cloud CUDA uses AMP. Backend performance differs.',
                              'Sampled MPS allocations can miss transient peaks; RSS and MPS memory overlap.',
                              'Pretrained architecture test does not verify cloud checkpoint/optimizer resume compatibility.']}
    def persist():
        report_path.write_text(json.dumps(record, indent=2))
    previous = None
    def epoch_start(trainer):
        nonlocal previous
        torch.mps.synchronize()
        previous = time.perf_counter()
    def batch_end(trainer):
        nonlocal previous
        torch.mps.synchronize()
        now = time.perf_counter()
        record['batches'].append({'batch': len(record['batches']) + 1, 'seconds': now - previous,
            'mps_tensor_bytes': torch.mps.current_allocated_memory(),
            'mps_driver_bytes': torch.mps.driver_allocated_memory(),
            'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        previous = now
        persist()
    mlflow.set_tracking_uri(f'sqlite:///{ROOT / "mlflow.db"}')
    mlflow.set_experiment('plastic-waste-phase1')
    with mlflow.start_run(run_name=name) as run:
        record['run_id'] = run.info.run_id
        mlflow.set_tags({'phase': '1', 'profile': 'engineering-only-mac-feasibility',
                         'dataset_approval': 'pending', 'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()})
        mlflow.log_params(config)
        mlflow.log_artifact(__file__, artifact_path='provenance')
        persist()
        started = time.perf_counter()
        try:
            model = YOLO(str(weights))
            model.add_callback('on_train_epoch_start', epoch_start)
            model.add_callback('on_train_batch_end', batch_end)
            model.train(data=str(runtime / 'dataset.yaml'), project=str(ROOT / 'runs'), name=name, **config)
            record['status'] = 'completed'
            steady = [b['seconds'] for b in record['batches'][2:]]
            record['steady_batch_seconds'] = statistics.median(steady)
            record['training_only_full_epoch_minutes'] = 1750 * statistics.mean(steady) / 60
            record['run_directory'] = str(model.trainer.save_dir)
        except Exception as error:
            record.update(status='failed', error=repr(error))
            raise
        finally:
            record['elapsed_seconds'] = time.perf_counter() - started
            record['inputs_unchanged'] = all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == digest for p, digest in identities.items())
            record['peak_process_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            persist()
            mlflow.log_artifact(str(report_path), artifact_path='engineering-only')
            print(json.dumps({k: v for k, v in record.items() if k not in ('input_sha256', 'settings', 'batches')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
