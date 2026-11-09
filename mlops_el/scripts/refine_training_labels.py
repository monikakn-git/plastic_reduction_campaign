"""Apply an explicit small training-only patch to a separate derived dataset."""
from collections import Counter
import json
import os
from pathlib import Path
import shutil

from train_candidate import ROOT,verify_inputs
from train import digest
from prepare_data import convert_box
import yaml


def main():
    verify_inputs()
    config_path=ROOT/'configs/label_corrections_v3.json'
    config=json.loads(config_path.read_text())
    source=ROOT/config['source_prepared'];target=ROOT/config['output_prepared']
    if target.exists():raise SystemExit('Completed variant exists; refuse overwrite')
    data=json.loads((ROOT/'data/raw/plitterstreet/annotations.json').read_text())
    images={i['id']:i for i in data['images']}
    annotations={(a['image_id'],a['id']):a for a in data['annotations']}
    splits={s['image_id']:s for s in json.loads((source/'splits.json').read_text())}
    names=json.loads((ROOT/'configs/plitter_classes.json').read_text())
    mapping={c['id']:next(i for i,n in enumerate(names) if c['name'] in names[n]) for c in data['categories']}
    patches={};class_delta=Counter()
    for patch in config['corrections']:
        key=(patch['image_id'],patch['annotation_id']);original=annotations[key]
        assert splits[patch['image_id']]['split']=='train'
        assert original['bbox']==patch['source_bbox'] and original['category_id']==patch['source_category_id']
        i=images[patch['image_id']]
        convert_box(patch['new_bbox'],i['width'],i['height'])
        patches[key]=patch
        class_delta[mapping[patch['new_category_id']]]+=1
        class_delta[mapping[patch['source_category_id']]]-=1
    shutil.copytree(source,target,copy_function=os.link,ignore=shutil.ignore_patterns('*.cache','.DS_Store'))
    affected=[]
    for image_id in sorted({k[0] for k in patches}):
        image=images[image_id];s=splits[image_id]
        relative=Path('labels/train')/Path('__'.join(Path(s['file_name']).parts)).with_suffix('.txt')
        original_text=(source/relative).read_text();rows=[line.split() for line in original_text.splitlines() if line.strip()]
        for key,patch in patches.items():
            if key[0]!=image_id:continue
            expected=(mapping[patch['source_category_id']],*convert_box(patch['source_bbox'],image['width'],image['height']))
            indices=[j for j,row in enumerate(rows) if int(row[0])==expected[0] and max(abs(float(v)-x) for v,x in zip(row[1:],expected[1:]))<1e-8]
            if len(indices)!=1:raise RuntimeError('Patch must identify exactly one prepared source row')
            new=(mapping[patch['new_category_id']],*convert_box(patch['new_bbox'],image['width'],image['height']))
            rows[indices[0]]=[str(new[0]),*(f'{x:.9f}' for x in new[1:])]
        file=target/relative;file.unlink();file.write_text('\n'.join(' '.join(row) for row in rows)+'\n')
        assert (source/relative).read_text()==original_text
        affected.append(str(relative))
    cfg=yaml.safe_load((source/'dataset.yaml').read_text());cfg['path']=config['output_prepared']
    file=target/'dataset.yaml';file.unlink();file.write_text(yaml.safe_dump(cfg))
    (target/'label_corrections.json').write_text(json.dumps(config,indent=2))
    differences=[];image_count=0;val_test_checked=0
    for p in (source/'labels').rglob('*.txt'):
        other=target/p.relative_to(source)
        if digest(p)!=digest(other):differences.append(str(p.relative_to(source)))
        if p.relative_to(source).parts[1] in ('val','test'):
            assert digest(p)==digest(other);val_test_checked+=1
    assert sorted(differences)==sorted(affected)
    for p in (source/'images').rglob('*'):
        if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png'):
            other=target/p.relative_to(source)
            assert os.path.samefile(p,other);image_count+=1
    assert image_count==10000 and val_test_checked==3000
    verify_inputs()
    out=ROOT/'reports/phase1/label_refinement';out.mkdir(exist_ok=True)
    report={'status':'derived_training_label_variant_created_not_trained',
        'target_metric':'70% validation mAP@50 aspiration, no score assumed',
        'variant_path':config['output_prepared'],'parent_path':config['source_prepared'],
        'correction_manifest_sha256':digest(config_path),'annotations_modified':len(patches),
        'training_label_files_modified':len(affected),'box_changes':2,'class_changes':1,
        'training_class_count_delta':{list(names)[k]:v for k,v in class_delta.items() if v},
        'image_count':image_count,'validation_test_label_files_verified_identical':val_test_checked,
        'original_data_and_models_preserved':True,'new_annotations_added':0,'annotations_removed':0,
        'different_label_files':differences,'held_cases':config['held_cases'],
        'limitation':'Small, visually supported AI-assisted patch; not a comprehensive expert relabeling or evidence of a large accuracy gain.'}
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
