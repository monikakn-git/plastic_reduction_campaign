"""Read-only corpus checks and learning-curve evidence before a new training proposal."""
from collections import Counter,defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path

from train_candidate import ROOT,verify_inputs
from prepare_data import convert_box
from PIL import Image
import numpy as np

OUT=ROOT/'reports/phase1/next_training_preflight'


def filehash(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    verify_inputs();OUT.mkdir(parents=True,exist_ok=True)
    raw=ROOT/'data/raw/plitterstreet';prepared=ROOT/'data/prepared/plitterstreet'
    data=json.loads((raw/'annotations.json').read_text());splits=json.loads((prepared/'splits.json').read_text())
    by_id={x['id']:x for x in data['images']};assign={x['image_id']:x for x in splits}
    aliases=json.loads((ROOT/'configs/plitter_classes.json').read_text());names=list(aliases)
    mapping={c['id']:next(i for i,n in enumerate(names) if c['name'] in aliases[n]) for c in data['categories']}
    annotations=defaultdict(list)
    for a in data['annotations']:annotations[a['image_id']].append(a)
    issues=[];checked=0;duplicate_rows=[];orientations=Counter();groups=defaultdict(set);hash_splits=defaultdict(set);sizes=defaultdict(list)
    protected_hashes={};class_counts=defaultdict(Counter)
    for index,s in enumerate(splits):
        group=s['group_id'];groups[group].add(s['split']);hash_splits[s['sha256']].add(s['split'])
        image=by_id[s['image_id']];path=raw/'images'/s['file_name']
        if not path.exists() or filehash(path)!=s['sha256']:
            issues.append({'image_id':image['id'],'issue':'raw_bytes_missing_or_hash_mismatch'})
        if s['split']=='test':continue # Only integrity metadata checked for test; no label/visual audit or scoring.
        with Image.open(path) as im:
            if im.size!=(image['width'],image['height']):issues.append({'image_id':image['id'],'issue':'encoded_dimensions_mismatch'})
            orientation=im.getexif().get(274,1);orientations[str(orientation)]+=1
            if orientation not in (None,0,1):issues.append({'image_id':image['id'],'issue':'exif_orientation_requires_review','orientation':orientation})
        flat='__'.join(Path(s['file_name']).parts)
        label=prepared/'labels'/s['split']/Path(flat).with_suffix('.txt')
        if not label.exists():issues.append({'image_id':image['id'],'issue':'missing_prepared_label'});continue
        expected=[]
        for a in annotations[image['id']]:
            c=mapping[a['category_id']];box=convert_box(a['bbox'],image['width'],image['height'])
            expected.append((c,*[round(v,9) for v in box]));class_counts[s['split']][names[c]]+=1
            sizes[s['split'],names[c]].append(min(a['bbox'][2:])*640/max(image['width'],image['height']))
        try:
            actual=[(int(parts[0]),*[round(float(v),9) for v in parts[1:]]) for parts in [row.split() for row in label.read_text().splitlines() if row.strip()]]
            if Counter(expected)!=Counter(actual):issues.append({'image_id':image['id'],'issue':'source_to_prepared_label_mismatch'})
            duplicates=sum(n-1 for n in Counter(actual).values() if n>1)
            if duplicates:duplicate_rows.append({'image_id':image['id'],'extra_identical_rows':duplicates})
        except Exception as e:issues.append({'image_id':image['id'],'issue':'invalid_label','error':str(e)})
        checked+=1
        if (index+1)%1000==0:print(f'Checked integrity metadata for {index+1}/{len(splits)} images',flush=True)
    for group,parts in groups.items():
        if len(parts)>1:issues.append({'issue':'inferred_group_crosses_split','group':group,'splits':sorted(parts)})
    for h,parts in hash_splits.items():
        if len(parts)>1:issues.append({'issue':'exact_image_hash_crosses_split','sha256':h,'splits':sorted(parts)})
    curves={}
    for name,record_path,default_dir in [('nano','baseline_metrics.json','runs/baseline-20261104-164642'),('small','yolo11s_candidate_training.json',None)]:
        record=json.loads((ROOT/'reports/phase1'/record_path).read_text());folder=record.get('run_directory',default_dir)
        rows=list(csv.DictReader((ROOT/folder/'results.csv').open()))
        best=max(rows,key=lambda r:float(r['metrics/mAP50-95(B)']))
        curves[name]={'epochs_completed':len(rows),'best_epoch_by_AP50_95':int(best['epoch']),
            'best_logged_AP50':float(best['metrics/mAP50(B)']),'best_logged_AP50_95':float(best['metrics/mAP50-95(B)']),
            'epoch40_AP50':float(rows[39]['metrics/mAP50(B)']), 'epoch50_AP50':float(rows[49]['metrics/mAP50(B)']),
            'train_cls_epoch40':float(rows[39]['train/cls_loss']),'train_cls_epoch50':float(rows[49]['train/cls_loss']),
            'val_cls_epoch40':float(rows[39]['val/cls_loss']),'val_cls_epoch50':float(rows[49]['val/cls_loss'])}
    report={'raw_images_hashed':len(splits),'train_val_label_files_checked':checked,
        'issue_count':len(issues),'issues':issues,'duplicate_label_rows':duplicate_rows,'exif_orientations_train_val':dict(orientations),
        'orientation_note':'Tag 0 is undefined and treated as no rotation by this decoder; it is not classified as an error. Tag 3 triggers a 180-degree rotation and needs coordinate review.',
        'class_counts':{k:dict(v) for k,v in class_counts.items()},'learning_curves':curves,
        'object_size_short_side_at_640':{f'{s}/{c}':{'median':float(np.median(v)),'under8_fraction':float((np.array(v)<8).mean())} for (s,c),v in sizes.items()},
        'limitations':['Mechanical consistency is not annotation correctness; unresolved visual flags remain.',
            'Group separation is inferred from filenames, not a guarantee of absence of cross-recording near-duplicates.',
            'No test labels inspected for tuning, no test scoring, no model or dataset edits.']}
    verify_inputs();(OUT/'corpus_checks.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k in ['raw_images_hashed','train_val_label_files_checked','issue_count','duplicate_label_rows','exif_orientations_train_val','learning_curves']},indent=2))


if __name__=='__main__':main()
