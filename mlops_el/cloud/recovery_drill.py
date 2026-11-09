"""Engineering-only stop/resume drill, isolated from full training and benchmark records."""
import json
from pathlib import Path
import subprocess
import sys


def main():
    project = Path('/content/drive/MyDrive/MLOPS_EL_cloud')
    persistent = project / 'recovery-drill'
    work = Path('/content/mlops_work/benchmark')
    preserved = work.with_name('benchmark-before-recovery-drill')
    first_session = work.with_name('recovery-drill-session1')
    second_session = work.with_name('recovery-drill-session2')
    for path in (persistent, preserved, first_session, second_session):
        if path.exists():
            raise FileExistsError(f'Drill already exists; review it rather than overwriting: {path}')
    if work.exists():
        work.rename(preserved)
    common = """
import sys
sys.path.insert(0, '/content')
import mlops_cloud_runner as runner
ROOT='/content/mlops_benchmark_bundle'
PERSISTENT='/content/drive/MyDrive/MLOPS_EL_cloud/recovery-drill'
"""
    first = common + """
original_save=runner.save_checkpoint_generation
def stop_after_saved_epoch(trainer, persistent, metadata):
    state=original_save(trainer, persistent, metadata)
    trainer.stop=True
    return state
runner.save_checkpoint_generation=stop_after_saved_epoch
runner.run(ROOT, PERSISTENT, mode='benchmark')
"""
    second = common + "runner.run(ROOT, PERSISTENT, mode='benchmark', resume=True)\n"
    try:
        # Separate interpreters close all SQLite connections and remove in-memory model state.
        subprocess.run([sys.executable, '-c', first], check=True)
        state_path = persistent / 'benchmark/checkpoints/latest.json'
        before = json.loads(state_path.read_text())
        assert before['epoch'] == 1, 'The controlled stop must preserve epoch 1'
        work.rename(first_session)
        subprocess.run([sys.executable, '-c', second], check=True)
        after = json.loads(state_path.read_text())
        report = json.loads((persistent / 'benchmark/benchmark_report.json').read_text())
        assert after['epoch'] == 2, 'Training must resume and finish epoch 2'
        assert before['run_id'] == after['run_id'], 'MLflow run continuity must be preserved'
        assert before['bundle_fingerprint'] == after['bundle_fingerprint']
        assert report['persistent_checkpoint_verified'] and report['optimizer_state_verified']
        record = {'status': 'passed', 'method': 'controlled epoch stop, fresh Python process and empty local work directory',
                  'before_epoch': before['epoch'], 'after_epoch': after['epoch'],
                  'run_id': after['run_id'], 'bundle_fingerprint': after['bundle_fingerprint'],
                  'same_mlflow_run': True, 'optimizer_state_verified': True,
                  'not_a_provider_disconnect_test': True, 'quality_claim': False}
        (persistent / 'recovery_drill_report.json').write_text(json.dumps(record, indent=2))
        print('RECOVERY_DRILL_RESULT', json.dumps(record), flush=True)
    finally:
        if work.exists():
            work.rename(second_session)
        if preserved.exists():
            preserved.rename(work)


if __name__ == '__main__':
    main()
