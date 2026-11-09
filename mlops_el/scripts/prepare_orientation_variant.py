"""Create a separate derived variant aligning encoded boxes with EXIF-aware decoding."""
import json
import os
from pathlib import Path
import shutil

from train_candidate import ROOT,verify_inputs
from train import digest
import yaml


def rotate180_box(box,width,height):
    x,y,w,h=box
    return [width-x-w,height-y-h,w,h]


def main():
    verify_inputs()
    source=ROOT/'data/prepared/plitterstreet'
    target=ROOT/'data/prepared/plitterstreet-orientation-v2'
    if target.exists():raise SystemExit('Variant exists; refuse to overwrite it')
    data=json.loads((ROOT/'data/raw/plitterstreet/annotations.json').read_text())
    splits={r['image_id']:r for r in json.loads((source/'splits.json').read_text())}
    proposals=json.loads((ROOT/'reports/phase1/next_training_preflight/orientation_proposal.json').read_text())
    identities={(r['image_id'],r['annotation_id']) for r in proposals['corrections']}
    aliases=json.loads((ROOT/'configs/plitter_classes.json').read_text())
    names=list(aliases);mapping={c['id']:next(i for i,n in enumerate(names) if c['name'] in aliases[n]) for c in data['categories']}
    images={i['id']:i for i in data['images']}
    # Hardlinks avoid copying gigabytes of existing immutable images/labels.
    shutil.copytree(source,target,copy_function=os.link,ignore=shutil.ignore_patterns('*.cache'))
    changes=[]
    for image_id in sorted({x[0] for x in identities}):
        s=splits[image_id];assert s['split']=='train'
        path=target/'labels/train'/Path('__'.join(Path(s['file_name']).parts)).with_suffix('.txt')
        original_sha=digest(path);rows=[];im=images[image_id]
        for a in data['annotations']:
            if a['image_id']!=image_id:continue
            if (image_id,a['id']) not in identities:raise RuntimeError('Every box in a rotated image must follow one convention')
            x,y,w,h=rotate180_box(a['bbox'],im['width'],im['height'])
            assert 0<=x<=im['width']-w and 0<=y<=im['height']-h
            norm=[(x+w/2)/im['width'],(y+h/2)/im['height'],w/im['width'],h/im['height']]
            rows.append(' '.join([str(mapping[a['category_id']]),*(f'{v:.9f}' for v in norm)]))
            changes.append({'image_id':image_id,'annotation_id':a['id'],'original_bbox':a['bbox'],'derived_bbox':[x,y,w,h]})
        path.unlink() # Never write through a hardlink to the original prepared label/cache.
        path.write_text('\n'.join(rows)+'\n')
    config=yaml.safe_load((source/'dataset.yaml').read_text())
    config['path']='data/prepared/plitterstreet-orientation-v2'
    dataset_file=target/'dataset.yaml';dataset_file.unlink();dataset_file.write_text(yaml.safe_dump(config))
    record={'status':'separate_preprocessing_variant_created_not_trained','variant_path':str(target.relative_to(ROOT)),
        'source_path':str(source.relative_to(ROOT)),'changed_training_images':4,'changed_annotations':len(changes),
        'class_changes':0,'split_changes':0,'original_raw_and_baseline_unchanged':True,
        'convention':'Source boxes in encoded-pixel coordinates; OpenCV/Ultralytics decodes EXIF orientation 3 as 180 degrees. Derived boxes transformed to that decoder orientation.',
        'changes':changes}
    (target/'orientation_changes.json').write_text(json.dumps(record,indent=2))
    verify_inputs()
    # Check every original prepared label stayed untouched and only four derived labels differ.
    differences=[]
    for label in (source/'labels').rglob('*.txt'):
        other=target/label.relative_to(source)
        if digest(label)!=digest(other):differences.append(str(label.relative_to(source)))
    assert len(differences)==4
    record['different_label_files']=differences
    (ROOT/'reports/phase1/next_training_preflight/orientation_variant.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
