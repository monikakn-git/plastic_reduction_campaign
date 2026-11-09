"""Apply reviewed class-only training patches with immutable-parent and holdout gates."""
from collections import Counter
import argparse, hashlib, json, os, shutil
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def labels_fingerprint(source, splits=('val','test')):
    h=hashlib.sha256()
    for split in splits:
        for p in sorted((source/'labels'/split).glob('*.txt')):
            h.update(str(p.relative_to(source)).encode());h.update(p.read_bytes())
    return h.hexdigest()

def validate_patch(patch, source):
    relative=Path(patch['label_file'])
    if relative.is_absolute() or '..' in relative.parts or relative.parts[:2]!=('labels','train') or relative.suffix!='.txt':
        raise ValueError('Only training label files may be patched')
    before=patch['before_line'].split(); after=patch['after_line'].split()
    if len(before)!=5 or len(after)!=5 or before[1:]!=after[1:]:raise ValueError('This variant permits class-only changes')
    for row in (before,after):
        if str(int(row[0]))!=row[0] or not 0<=int(row[0])<10:raise ValueError('Invalid class ID')
    if before==after:raise ValueError('Empty correction')
    row=patch['row']
    if type(row) is not int or row<0:raise ValueError('Invalid row index')
    lines=(source/relative).read_text().splitlines()
    if row>=len(lines) or lines[row]!=patch['before_line']:raise ValueError('Parent annotation differs from reviewed row')
    return relative,row

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',default='configs/label_corrections_v4.json');args=parser.parse_args()
    config=json.loads((ROOT/args.manifest).read_text());source=ROOT/config['source_prepared'];target=ROOT/config['output_prepared']
    if target.exists():raise SystemExit('Refuse to overwrite a dataset version')
    if labels_fingerprint(source)!=config['holdout_labels_sha256'] or sha(source/'splits.json')!=config['splits_sha256']:raise ValueError('Holdouts or splits changed')
    assignments={r['image_id']:r for r in json.loads((source/'splits.json').read_text())}
    grouped={};keys=set()
    for patch in config['patches']:
        relative,row=validate_patch(patch,source)
        if assignments[patch['image_id']]['split']!='train' or Path(assignments[patch['image_id']]['file_name']).with_suffix('.txt').name!=relative.name:raise ValueError('Wrong source image identity')
        if (str(relative),row) in keys:raise ValueError('Duplicate patch')
        keys.add((str(relative),row));grouped.setdefault(relative,[]).append(patch)
    before={str(p.relative_to(source)):sha(p) for p in (source/'labels').rglob('*.txt')}
    shutil.copytree(source,target,copy_function=os.link,ignore=shutil.ignore_patterns('*.cache','.DS_Store'))
    delta=Counter()
    for relative,patches in grouped.items():
        lines=(source/relative).read_text().splitlines()
        for patch in patches:
            lines[patch['row']]=patch['after_line'];delta[int(patch['before_line'].split()[0])]-=1;delta[int(patch['after_line'].split()[0])]+=1
        (target/relative).unlink();(target/relative).write_text('\n'.join(lines)+'\n')
    cfg=yaml.safe_load((source/'dataset.yaml').read_text());cfg['path']=config['output_prepared']
    (target/'dataset.yaml').unlink();(target/'dataset.yaml').write_text(yaml.safe_dump(cfg))
    (target/'training_quality_changes.json').write_text(json.dumps(config,indent=2)+'\n')
    changed=[]
    for relative,digest in before.items():
        if sha(source/relative)!=digest:raise ValueError('Parent labels were modified')
        if sha(target/relative)!=digest:changed.append(relative)
        if len((source/relative).read_text().splitlines())!=len((target/relative).read_text().splitlines()):raise ValueError('Instance count changed')
    if set(changed)!={str(p) for p in grouped}:raise ValueError('Unexpected label change')
    if labels_fingerprint(target)!=config['holdout_labels_sha256']:raise ValueError('Holdout labels changed')
    if sha(target/'splits.json')!=config['splits_sha256']:raise ValueError('Split membership changed')
    images=[p for p in (source/'images').rglob('*') if p.is_file()]
    if len(images)!=10000 or any(not os.path.samefile(p,target/p.relative_to(source)) for p in images):raise ValueError('Images not preserved')
    report={'status':'created_and_verified_not_trained','source':config['source_prepared'],'variant':config['output_prepared'],'patch_manifest_sha256':sha(ROOT/args.manifest),'class_corrections':len(keys),'changed_training_label_files':len(changed),'class_count_delta':dict(delta),'images_preserved':len(images),'holdout_label_files_verified_identical':3000,'all_parent_labels_verified_unchanged':len(before),'instances_added':0,'instances_removed':0,'box_changes':0,'split_changes':0,'holdout_labels_sha256':config['holdout_labels_sha256'],'corrected_files':sorted(changed),'training_started':False,'score_improvement_measured':False}
    (ROOT/'reports/phase1/training_quality_v4/variant_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
