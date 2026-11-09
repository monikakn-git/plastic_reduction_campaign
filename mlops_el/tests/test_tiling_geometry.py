"""Geometry/metric checks needed before interpreting tiled predictions."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import torch
from experiment_tiling import starts,merge,update
from ultralytics.utils.metrics import DetMetrics


def test_windows_cover_boundaries_without_duplicates():
    assert starts(640)==[0]
    positions=starts(4032)
    assert positions[0]==0 and positions[-1]+960==4032
    assert positions==sorted(set(positions))
    assert all(b-a<=960 for a,b in zip(positions,positions[1:]))


def test_merge_suppresses_same_class_duplicates_but_preserves_other_classes():
    p=torch.tensor([[10,10,30,30,.9,0],[10,10,30,30,.8,0],[10,10,30,30,.7,1]])
    result=merge([p])
    assert len(result)==2 and result[:,5].tolist()==[0,1]
    assert len(merge([]))==0


def test_matching_claims_target_only_once_and_handles_empty_predictions():
    metric=DetMetrics(names={0:'object'})
    pred=torch.tensor([[10,10,30,30,.9,0],[10,10,30,30,.8,0]])
    gt=torch.tensor([[0,10,10,30,30]])
    update(metric,pred,gt,'sample')
    assert metric.stats['tp'][0].sum()==10
    assert metric.stats['tp'][0][0].all() and not metric.stats['tp'][0][1].any()
    update(metric,torch.empty((0,6)),gt,'empty')
    assert metric.stats['tp'][1].shape==(0,10)
