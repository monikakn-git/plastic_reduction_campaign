"""A valid artifact must still fail unmet production-quality gates."""
import copy,json
from pathlib import Path
from service.quality import assess
from service.model import MODEL_SHA256,Detector
import pytest
ROOT=Path(__file__).resolve().parents[1]

def test_current_candidate_is_demo_only_and_cup_gate_is_unmet():
    p=json.loads((ROOT/'configs/phase2/release_policy.json').read_text());e=json.loads((ROOT/p['evaluation']).read_text());r=assess(p,e,MODEL_SHA256)
    assert r['integrity_passed'] and r['demo_eligible']
    assert not r['production_approved']
    assert r['failed_classes']==['cup']
    assert not r['production_checks']['faculty_dataset_approval']


def test_wrong_weights_or_source_cannot_pass_integrity():
    p=json.loads((ROOT/'configs/phase2/release_policy.json').read_text());e=json.loads((ROOT/p['evaluation']).read_text())
    assert not assess(p,e,'0'*64)['demo_eligible']
    e['source_run_id']='unrelated-experiment'
    assert not assess(p,e,MODEL_SHA256)['integrity_passed']


def test_bad_checkpoint_rejected_before_deserialization(tmp_path):
    p=tmp_path/'bad.pt';p.write_bytes(b'not a model')
    with pytest.raises(ValueError,match='checksum'):Detector(p)
