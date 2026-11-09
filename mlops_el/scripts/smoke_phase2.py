"""Real-weight inference smoke: registry round-trip equals API prediction; no quality tuning."""
import base64,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'.local/ultralytics'));os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.local/matplotlib'))
import mlflow,pandas as pd
from fastapi.testclient import TestClient
from service.app import create_app
from service.model import MODEL_SHA256,file_sha,CLASSES

def main():
    release=json.loads((ROOT/'reports/phase2/registered_model.json').read_text())
    splits=json.loads((ROOT/'data/prepared/plitterstreet-labels-v3/splits.json').read_text());sample=next(r for r in splits if r['split']=='train')
    image=ROOT/'data/prepared/plitterstreet-labels-v3/images/train'/sample['file_name'];raw=image.read_bytes()
    mlflow.set_tracking_uri(f'sqlite:///{ROOT / "mlflow.db"}')
    loaded=mlflow.pyfunc.load_model(release['model_uri']);started=time.perf_counter()
    expected=json.loads(loaded.predict(pd.DataFrame({'image_base64':[base64.b64encode(raw).decode()]}))[0]);registry_ms=(time.perf_counter()-started)*1000
    del loaded
    with TestClient(create_app()) as client:
        assert client.get('/ready').status_code==200
        started=time.perf_counter();response=client.post('/predict',content=raw,headers={'Content-Type':'image/jpeg'});api_ms=(time.perf_counter()-started)*1000
        assert response.status_code==200,response.text
        actual=response.json();assert {k:actual[k] for k in expected}==expected
        assert actual['model']['sha256']==MODEL_SHA256
        for d in actual['detections']:
            assert d['label']==CLASSES[d['class_id']] and .25<=d['confidence']<=1
            x1,y1,x2,y2=d['bbox_xyxy'];assert 0<=x1<=x2<=actual['width'] and 0<=y1<=y2<=actual['height']
    report={'status':'passed','test_type':'inference_only_training_image_smoke','training_performed':False,'test_set_used':False,'sample_image_id':sample['image_id'],'sample_sha256':file_sha(image),'registered_model':release['model_uri'],'model_sha256':MODEL_SHA256,'registry_api_predictions_identical':True,'detections':len(actual['detections']),'width':actual['width'],'height':actual['height'],'registry_single_request_ms':registry_ms,'api_single_request_ms':api_ms,'device':'cpu','latency_claim':'One engineering request per path, including cold inference; not a throughput or accuracy benchmark.'}
    (ROOT/'reports/phase2/smoke_test.json').write_text(json.dumps(report,indent=2)+'\n');(ROOT/'reports/phase2/example_prediction.json').write_text(json.dumps(actual,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
