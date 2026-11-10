"""Local-first FastAPI service. Requests never select a model path or remote URL."""
import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
import time
import uuid
from fastapi import FastAPI,HTTPException,Request
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool
from service.model import Detector,InvalidImage,ModelBusy,decode_image,MAX_BYTES,MODEL_SHA256,SOURCE_RUN_ID,CLASSES,IMGSZ,CONFIDENCE,NMS_IOU

log=logging.getLogger('mlops.inference')
log.setLevel(logging.INFO)
if not log.handlers:
    handler=logging.StreamHandler();handler.setFormatter(logging.Formatter('%(message)s'));log.addHandler(handler)
log.propagate=False

def load_release(bundle):
    release=json.loads((Path(bundle)/'release.json').read_text())
    if release['weights_sha256']!=MODEL_SHA256 or release['source_training_run_id']!=SOURCE_RUN_ID:raise ValueError('Release identity mismatch')
    if release['trained_dataset']!='labels-v3' or release['classes']!=CLASSES:raise ValueError('Release data/class schema mismatch')
    if release.get('production_approved') is not False:raise ValueError('This candidate supports demonstration only')
    return release

def create_app(detector_factory=None,release_override=None):
    @asynccontextmanager
    async def lifespan(app):
        bundle=Path(os.environ.get('MODEL_BUNDLE','releases/yolo11m-demo-v3'))
        app.state.release=release_override if release_override is not None else load_release(bundle)
        app.state.detector=detector_factory() if detector_factory else Detector(bundle/'weights.pt',os.environ.get('INFERENCE_DEVICE','cpu'))
        app.state.slot=asyncio.Semaphore(1)
        yield
        app.state.detector=None

    app=FastAPI(title='Litter detection demo',version='0.1.0',lifespan=lifespan)

    @app.get('/health')
    async def health():return {'status':'ok'}

    @app.get('/ready')
    async def ready(request:Request):
        if getattr(request.app.state,'detector',None) is None:raise HTTPException(503,'Model is not ready')
        return {'status':'ready','model_sha256':MODEL_SHA256,'production_approved':False}

    @app.get('/model')
    async def model(request:Request):
        return {**request.app.state.release,'inference':{'imgsz':IMGSZ,'confidence_threshold':CONFIDENCE,'nms_iou':NMS_IOU,'max_detections':300},'human_review_required':True}

    @app.post('/predict')
    async def predict(request:Request):
        request_id=uuid.uuid4().hex
        if request.headers.get('content-type','').split(';')[0].strip().lower() not in ('image/jpeg','image/png'):
            raise HTTPException(415,'Send raw JPEG or PNG bytes with the matching Content-Type')
        length=request.headers.get('content-length')
        if length:
            try:declared=int(length)
            except ValueError:raise HTTPException(400,'Invalid Content-Length')
            if declared<0:raise HTTPException(400,'Invalid Content-Length')
            if declared>MAX_BYTES:raise HTTPException(413,'Image exceeds 8 MiB')
        if request.app.state.slot.locked():raise HTTPException(503,'Service is busy; retry shortly',headers={'Retry-After':'1'})
        await request.app.state.slot.acquire()
        started=time.perf_counter()
        try:
            body=bytearray()
            async with asyncio.timeout(15):
                async for chunk in request.stream():
                    if len(body)+len(chunk)>MAX_BYTES:raise HTTPException(413,'Image exceeds 8 MiB')
                    body.extend(chunk)
            image=await run_in_threadpool(decode_image,bytes(body))
            result=await run_in_threadpool(request.app.state.detector.predict,image)
            release=request.app.state.release
            output={**result,'request_id':request_id,'model':{'sha256':MODEL_SHA256,'registry_name':release['registry_name'],'registry_version':release['registry_version'],'source_training_run_id':SOURCE_RUN_ID,'trained_dataset':'labels-v3'},'confidence_threshold':CONFIDENCE,'human_review_required':True,'production_approved':False}
            log.info(json.dumps({'event':'inference_completed','request_id':request_id,'duration_ms':round((time.perf_counter()-started)*1000,1),'detection_count':len(result['detections']),'model_sha256':MODEL_SHA256}))
            return output
        except TimeoutError as exc:raise HTTPException(408,'Image upload timed out') from exc
        except InvalidImage as exc:raise HTTPException(422,str(exc)) from exc
        except ModelBusy as exc:raise HTTPException(503,'Service is busy; retry shortly',headers={'Retry-After':'1'}) from exc
        except HTTPException:raise
        except Exception:
            log.error(json.dumps({'event':'inference_failed','request_id':request_id}))
            raise HTTPException(500,'Inference failed; consult the service operator')
        finally:request.app.state.slot.release()

    @app.get('/',response_class=HTMLResponse,include_in_schema=False)
    async def demo():
        return Path(__file__).with_name('demo.html').read_text(encoding='utf-8')
    return app

app=create_app()
