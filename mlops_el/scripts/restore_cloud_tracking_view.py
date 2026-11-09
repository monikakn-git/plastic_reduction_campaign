"""Create a local MLflow view of the completed cloud archive without changing it."""
from pathlib import Path
import json,shutil,sqlite3

ROOT=Path(__file__).resolve().parents[1]
RUN='85714466948d4e658ebb610a475dcf57'

def main():
    source=ROOT/'.local/kaggle-completed-epoch86/train'; target=ROOT/'.local/phase1-cloud-tracking'
    if target.exists():raise SystemExit('Local view already exists; inspect it rather than overwrite')
    target.mkdir()
    with sqlite3.connect((source/'mlflow.db').resolve().as_uri()+'?mode=ro',uri=True) as src,sqlite3.connect(target/'mlflow.db') as dst:src.backup(dst)
    shutil.copytree(source/'mlartifacts',target/'mlartifacts')
    old=f'file:///content/mlops_work/train/mlartifacts/{RUN}/artifacts'; new=(target/'mlartifacts'/RUN/'artifacts').as_uri()
    with sqlite3.connect(target/'mlflow.db') as db:
        row=db.execute('select artifact_uri,status from runs where run_uuid=?',(RUN,)).fetchone()
        if row!=(old,'FINISHED'):raise ValueError('Unexpected archived run identity or status')
        db.execute('update runs set artifact_uri=? where run_uuid=?',(new,RUN))
        for ident,location in db.execute('select experiment_id,artifact_location from experiments').fetchall():
            if location and location.startswith('file:///content/mlops_work/train/mlartifacts'):
                db.execute('update experiments set artifact_location=? where experiment_id=?',((target/'mlartifacts').as_uri(),ident))
        assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    record={'run_id':RUN,'archived_database':str(source/'mlflow.db'),'local_view_database':str(target/'mlflow.db'),'old_artifact_uri':old,'local_artifact_uri':new,'original_archive_unchanged':True,'run_id_preserved':True,'status':'FINISHED','migration':'Only artifact URI paths in a copied database changed. Artifacts copied from archived payload; original files remain unchanged.'}
    (ROOT/'reports/phase1/cloud_tracking_local_view.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
