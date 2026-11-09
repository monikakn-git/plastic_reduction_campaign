"""Snapshot experiment metadata consistently; preserve live operational stores."""
import json,sqlite3
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'reports/phase1/experiment_history';out.mkdir(exist_ok=True)
    sources={'local':ROOT/'mlflow.db','cloud':ROOT/'.local/phase1-cloud-tracking/mlflow.db'};records=[]
    for name,source in sources.items():
        target=out/f'{name}_mlflow.db'
        if target.exists():raise SystemExit('Refuse to overwrite an experiment snapshot')
        with sqlite3.connect(source.as_uri()+'?mode=ro',uri=True) as src,sqlite3.connect(target) as dst:
            src.backup(dst)
            assert dst.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
            runs=[{'run_id':r[0],'status':r[1],'experiment_id':r[2],'artifact_uri':r[3]} for r in dst.execute('select run_uuid,status,experiment_id,artifact_uri from runs')]
        records.append({'store':name,'source':str(source),'snapshot':str(target.relative_to(ROOT)),'runs':runs})
    record={'purpose':'Consistent Phase 1 experiment metadata snapshots','stores':records,'artifact_files':'Artifacts remain in the live local stores and original verified cloud recovery archive. These SQLite snapshots alone do not contain every artifact file.','off_device_dvc_remote_configured':False}
    (out/'index.json').write_text(json.dumps(record,indent=2)+'\n')
    print('Snapshotted local and cloud experiment metadata without modifying operational databases.')

if __name__=='__main__':main()
