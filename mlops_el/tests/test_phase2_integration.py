"""Opt-in real model test. Requires the private DVC release and training image."""
import json,os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from service.app import create_app
from service.model import CLASSES,MODEL_SHA256
ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.skipif(os.environ.get('RUN_MODEL_INTEGRATION')!='1',reason='Set RUN_MODEL_INTEGRATION=1 with the private release/data available')
def test_actual_checkpoint_inference_contract():
    split=json.loads((ROOT/'data/prepared/plitterstreet-labels-v3/splits.json').read_text());sample=next(r for r in split if r['split']=='train');image=ROOT/'data/prepared/plitterstreet-labels-v3/images/train'/sample['file_name']
    with TestClient(create_app()) as client:
        response=client.post('/predict',content=image.read_bytes(),headers={'Content-Type':'image/jpeg'})
        assert response.status_code==200
        result=response.json();assert result['model']['sha256']==MODEL_SHA256
        assert len(result['detections'])>0
        for row in result['detections']:
            assert row['label']==CLASSES[row['class_id']] and .25<=row['confidence']<=1
            a,b,c,d=row['bbox_xyxy'];assert 0<=a<=c<=result['width'] and 0<=b<=d<=result['height']
