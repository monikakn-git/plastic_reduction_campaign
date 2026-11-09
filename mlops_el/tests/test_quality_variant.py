"""Prevent dataset cleanup from modifying holdouts, geometry or unreviewed parent rows."""
import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('prepare_quality_variant',ROOT/'scripts/prepare_quality_variant.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_patch_rejects_holdouts_geometry_changes_and_stale_rows(tmp_path):
    f=tmp_path/'labels/train/a.txt';f.parent.mkdir(parents=True);f.write_text('6 0.5 0.5 0.1 0.1\n')
    patch={'label_file':'labels/train/a.txt','row':0,'before_line':'6 0.5 0.5 0.1 0.1','after_line':'0 0.5 0.5 0.1 0.1'}
    assert m.validate_patch(patch,tmp_path)==(Path('labels/train/a.txt'),0)
    for override in ({'label_file':'labels/test/a.txt'},{'label_file':'labels/train/../../test/a.txt'},{'after_line':'0 0.6 0.5 0.1 0.1'},{'before_line':'5 0.5 0.5 0.1 0.1'},{'after_line':'10 0.5 0.5 0.1 0.1'}):
        with pytest.raises(ValueError):m.validate_patch({**patch,**override},tmp_path)


def test_v4_retains_holdouts_all_instances_and_declared_changes_only():
    config=json.loads((ROOT/'configs/label_corrections_v4.json').read_text()); source=ROOT/config['source_prepared'];target=ROOT/config['output_prepared']
    expected={p['label_file'] for p in config['patches']};changed=set()
    assert m.labels_fingerprint(target)==m.labels_fingerprint(source)==config['holdout_labels_sha256']
    assert (source/'splits.json').read_bytes()==(target/'splits.json').read_bytes()
    for p in (source/'labels').rglob('*.txt'):
        relative=p.relative_to(source);a=p.read_text().splitlines();b=(target/relative).read_text().splitlines()
        assert len(a)==len(b)
        if a!=b:changed.add(str(relative))
        for old,new in zip(a,b):assert old.split()[1:]==new.split()[1:]
    assert changed==expected
    assert len(config['patches'])==30


def test_structural_audit_rejects_malformed_nonfinite_and_out_of_bounds_labels():
    spec=importlib.util.spec_from_file_location('audit_training_quality',ROOT/'scripts/audit_training_quality.py')
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    assert audit.parse_label('6 0.5 0.5 0.1 0.1')[0]==6
    for line in ('6 .5 .5 .1','6 .5 nan .1 .1','6 .5 .5 -0.1 .1','10 .5 .5 .1 .1','6 .99 .5 .1 .1'):
        with pytest.raises(ValueError):audit.parse_label(line)
