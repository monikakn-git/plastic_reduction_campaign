"""API boundary tests use a deterministic model stub; real model smoke is separate."""
import io
import json
from pathlib import Path
from fastapi.testclient import TestClient
from PIL import Image
from service.app import create_app,load_release
from service.model import InvalidImage,decode_image,MODEL_SHA256,SOURCE_RUN_ID,CLASSES,MAX_BYTES
import pytest

class StubDetector:
    calls=0
    def predict(self,image):
        self.calls+=1
        return {'width':image.width,'height':image.height,'detections':[]}

def image_bytes(fmt='JPEG',size=(32,24)):
    f=io.BytesIO();Image.new('RGB',size).save(f,format=fmt);return f.getvalue()

def release():return {'registry_name':'test-demo','registry_version':'1','production_approved':False}

def test_valid_image_contract_and_model_identity():
    model=StubDetector()
    with TestClient(create_app(lambda:model,release())) as client:
        assert client.get('/health').status_code==200
        assert client.get('/ready').json()['model_sha256']==MODEL_SHA256
        result=client.post('/predict',content=image_bytes(),headers={'Content-Type':'image/jpeg'})
        assert result.status_code==200
        data=result.json();assert (data['width'],data['height'])==(32,24)
        assert data['model']['sha256']==MODEL_SHA256 and data['model']['trained_dataset']=='labels-v3'
        assert data['human_review_required'] and not data['production_approved']
        assert len(data['request_id'])==32 and model.calls==1
        assert 'confidence_threshold' in data

@pytest.mark.parametrize('content,content_type,status',[(b'not an image','image/jpeg',422),(b'','image/png',422),(b'hello','application/json',415),(b'x'*(MAX_BYTES+1),'image/jpeg',413)])
def test_bad_uploads_never_reach_model(content,content_type,status):
    model=StubDetector()
    with TestClient(create_app(lambda:model,release())) as client:
        assert client.post('/predict',content=content,headers={'Content-Type':content_type}).status_code==status
        assert model.calls==0


def test_pixel_and_format_limits():
    with pytest.raises(InvalidImage):decode_image(image_bytes('GIF'))
    with pytest.raises(InvalidImage):decode_image(image_bytes(size=(8193,16)))
    with pytest.raises(InvalidImage):decode_image(image_bytes(size=(1,1)))


def test_exif_rotation_matches_output_dimensions():
    f=io.BytesIO();image=Image.new('RGB',(40,20));exif=image.getexif();exif[274]=6;image.save(f,format='JPEG',exif=exif)
    assert decode_image(f.getvalue()).size==(20,40)


def test_unexpected_inference_errors_do_not_leak_paths_or_details():
    class Broken:
        def predict(self,image):raise RuntimeError('/secret/path/private_key')
    with TestClient(create_app(Broken,release())) as client:
        response=client.post('/predict',content=image_bytes(),headers={'Content-Type':'image/jpeg'})
        assert response.status_code==500 and '/secret' not in response.text


def test_release_rejects_changed_training_identity(tmp_path):
    record={'weights_sha256':MODEL_SHA256,'source_training_run_id':SOURCE_RUN_ID,'trained_dataset':'labels-v4','classes':CLASSES,'production_approved':False}
    (tmp_path/'release.json').write_text(json.dumps(record))
    with pytest.raises(ValueError,match='schema'):load_release(tmp_path)


def test_stream_limit_applies_without_content_length():
    model=StubDetector()
    def chunks():
        yield b'x'*(MAX_BYTES//2)
        yield b'y'*(MAX_BYTES//2+1)
    with TestClient(create_app(lambda:model,release())) as client:
        response=client.post('/predict',content=chunks(),headers={'Content-Type':'image/jpeg'})
        assert response.status_code==413 and model.calls==0


def test_concurrent_inference_is_rejected_instead_of_queued():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    entered=Event();finish=Event()
    class Blocking(StubDetector):
        def predict(self,image):
            entered.set();assert finish.wait(5);return super().predict(image)
    with TestClient(create_app(Blocking,release())) as client,ThreadPoolExecutor(max_workers=1) as pool:
        first=pool.submit(client.post,'/predict',content=image_bytes(),headers={'Content-Type':'image/jpeg'})
        try:
            assert entered.wait(3)
            second=client.post('/predict',content=image_bytes(),headers={'Content-Type':'image/jpeg'})
            assert second.status_code==503 and second.headers['Retry-After']=='1'
        finally:finish.set()
        assert first.result(timeout=5).status_code==200


def test_invalid_jpeg_header_does_not_call_optional_image_plugins(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Global image plugins must not run')
    monkeypatch.setattr(Image,'open',forbidden)
    with pytest.raises(InvalidImage):decode_image(b'\xff\xd8\xffbroken')
