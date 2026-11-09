"""Verify the native localhost API using one training image; save HTTP evidence."""
from pathlib import Path
import json,time,urllib.request
ROOT=Path(__file__).resolve().parents[1]

def main():
    ready=json.load(urllib.request.urlopen('http://127.0.0.1:8010/ready',timeout=10))
    splits=json.loads((ROOT/'data/prepared/plitterstreet-labels-v3/splits.json').read_text());row=next(x for x in splits if x['split']=='train');path=ROOT/'data/prepared/plitterstreet-labels-v3/images/train'/row['file_name']
    request=urllib.request.Request('http://127.0.0.1:8010/predict',data=path.read_bytes(),headers={'Content-Type':'image/jpeg'},method='POST');started=time.perf_counter()
    with urllib.request.urlopen(request,timeout=30) as response:
        result=json.load(response);status=response.status
    assert ready['model_sha256']==result['model']['sha256'] and ready['status']=='ready'
    report={'status':'passed','address':'http://127.0.0.1:8010','bind_scope':'localhost_only','deployment_type':'native_uvicorn_not_docker','ready':ready,'prediction_http_status':status,'sample_split':'train','sample_image_id':row['image_id'],'detections':len(result['detections']),'elapsed_ms':(time.perf_counter()-started)*1000,'model':result['model']}
    (ROOT/'reports/phase2/live_http_smoke.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
