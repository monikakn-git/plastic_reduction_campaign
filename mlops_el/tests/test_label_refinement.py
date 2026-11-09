"""Verify a derived annotation patch cannot change holdouts or lose instances."""
from collections import Counter
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def test_refinement_keeps_holdouts_and_instance_counts_intact():
    config=json.loads((ROOT/'configs/label_corrections_v3.json').read_text())
    source=ROOT/config['source_prepared'];target=ROOT/config['output_prepared']
    before,after=Counter(),Counter();changed=[]
    for label in (source/'labels').rglob('*.txt'):
        relative=label.relative_to(source);other=target/relative
        old,new=label.read_text(),other.read_text()
        if relative.parts[1] in ('val','test'):assert old==new
        if old!=new:changed.append(relative)
        for line in old.splitlines():
            if line.strip():before[int(line.split()[0])]+=1
        for line in new.splitlines():
            if line.strip():after[int(line.split()[0])]+=1
    assert len(changed)==2 and all(p.parts[1]=='train' for p in changed)
    assert sum(before.values())==sum(after.values())==60826
    assert after[5]-before[5]==1 and after[0]-before[0]==-1
    assert all(after[i]==before[i] for i in range(10) if i not in (0,5))
    assert (source/'splits.json').read_bytes()==(target/'splits.json').read_bytes()
