"""Reproducible, training-only structural audit and class/size-stratified visual queue."""
import argparse, hashlib, json
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT=Path(__file__).resolve().parents[1]
NAMES=list(json.loads((ROOT/'configs/plitter_classes.json').read_text()))

def digest(p):
    with p.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()

def parse_label(line):
    r=list(map(float,line.split()))
    if len(r)!=5 or not np.isfinite(r).all():raise ValueError('Expected five finite values')
    c,x,y,w,h=r
    if c!=int(c) or not 0<=c<10 or min(w,h)<=0 or min(x-w/2,y-h/2)<-1e-7 or max(x+w/2,y+h/2)>1+1e-7:raise ValueError('Invalid class or bounding box')
    return r

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',default='reports/phase1/training_quality_v4'); parser.add_argument('--remaining-cups',help='Exclude previously reviewed keys from this manifest and review every other training cup'); args=parser.parse_args()
    source=ROOT/'data/prepared/plitterstreet-labels-v3'; out=ROOT/args.output
    if out.exists(): raise SystemExit('Refuse to overwrite an existing audit')
    out.mkdir(parents=True)
    splits=json.loads((source/'splits.json').read_text()); lookup={Path(r['file_name']).with_suffix('.txt').name:r for r in splits if r['split']=='train'}
    pools={i:[] for i in range(10)}; invalid=[]; duplicate=[]; overlap=[]; counts=Counter(); protected=hashlib.sha256()
    for split in ('val','test'):
        for p in sorted((source/'labels'/split).glob('*.txt')):
            protected.update(str(p.relative_to(source)).encode()); protected.update(p.read_bytes())
    for p in sorted((source/'labels/train').glob('*.txt')):
        rows=[]; seen=set()
        for row,line in enumerate(p.read_text().splitlines()):
            if not line.strip():continue
            try:r=parse_label(line)
            except (ValueError,OverflowError):invalid.append([p.name,row,line]);continue
            c,x,y,w,h=r
            if tuple(r) in seen:duplicate.append([p.name,row,line])
            seen.add(tuple(r));rows.append(r);counts[int(c)]+=1
            s=lookup[p.name]
            pools[int(c)].append({'key':f'{s["image_id"]}:{row}','image_id':s['image_id'],'file_name':s['file_name'],'label_file':str(p.relative_to(source)),'row':row,'class_id':int(c),'class':NAMES[int(c)],'box':r[1:],'area':w*h,'group_id':s['group_id']})
        if len(rows)>1:
            a=np.array(rows); b=np.column_stack((a[:,1]-a[:,3]/2,a[:,2]-a[:,4]/2,a[:,1]+a[:,3]/2,a[:,2]+a[:,4]/2)); area=a[:,3]*a[:,4]
            inter=np.maximum(0,np.minimum(b[:,None,2:],b[None,:,2:])-np.maximum(b[:,None,:2],b[None,:,:2])).prod(2); iou=inter/(area[:,None]+area[None,:]-inter+1e-20)
            for i,j in zip(*np.where(np.triu(iou>.95,1))):overlap.append({'label_file':str(p.relative_to(source)),'rows':[int(i),int(j)],'classes':[NAMES[int(a[i,0])],NAMES[int(a[j,0])]],'iou':float(iou[i,j])})
    selected=[]
    for c in range(10):
        pool=sorted(pools[c],key=lambda r:(r['area'],r['key'])); n=180 if c==6 else 100 if c==5 else 10
        # Evenly spaced quantiles cover small through large targets without choosing by model confidence.
        indices=np.linspace(0,len(pool)-1,min(n,len(pool)),dtype=int)
        selected.extend(pool[i] for i in indices)
    for flag in overlap:
        for pool in pools.values():
            for r in pool:
                if r['label_file']==flag['label_file'] and r['row'] in flag['rows'] and r['key'] not in {s['key'] for s in selected}:selected.append(r)
    if args.remaining_cups:
        previous=json.loads((ROOT/args.remaining_cups).read_text()); keys={r['key'] for r in previous['samples']}
        selected=[r for r in sorted(pools[6],key=lambda r:(r['area'],r['key'])) if r['key'] not in keys]
    for i,r in enumerate(selected):r['review_id']=i+1
    manifest={'source':str(source.relative_to(ROOT)),'scope':'Training only; holdout labels unchanged; no model predictions or test examples used for selection.','selection':('All remaining training cups not present in '+args.remaining_cups if args.remaining_cups else '180 cup, 100 bottle, 10 each other class; evenly spaced area quantiles, plus cross-class overlaps'),'samples':selected,'holdout_labels_sha256':protected.hexdigest(),'splits_sha256':digest(source/'splits.json'),'train_class_counts':{NAMES[k]:v for k,v in counts.items()},'invalid_rows':invalid,'exact_duplicate_rows':duplicate,'high_overlap_pairs':overlap,'labels_scanned':len(list((source/'labels/train').glob('*.txt')))}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for start in range(0,len(selected),20):
        sheet=Image.new('RGB',(1600,1200),'white');draw=ImageDraw.Draw(sheet)
        for j,r in enumerate(selected[start:start+20]):
            image=ImageOps.exif_transpose(Image.open(source/'images/train'/r['file_name'])).convert('RGB'); W,H=image.size
            x,y,w,h=r['box']; box=((x-w/2)*W,(y-h/2)*H,(x+w/2)*W,(y+h/2)*H)
            margin=max(20,max(w*W,h*H)*.40); cropbox=(max(0,int(box[0]-margin)),max(0,int(box[1]-margin)),min(W,int(box[2]+margin+1)),min(H,int(box[3]+margin+1)))
            crop=image.crop(cropbox); cd=ImageDraw.Draw(crop);cd.rectangle((box[0]-cropbox[0],box[1]-cropbox[1],box[2]-cropbox[0],box[3]-cropbox[1]),outline='red',width=2)
            scale=min(392/crop.width,208/crop.height,4); crop=crop.resize((max(1,round(crop.width*scale)),max(1,round(crop.height*scale)))); col=j%4;row=j//4;px=col*400;py=row*240
            sheet.paste(crop,(px+(400-crop.width)//2,py+26+(208-crop.height)//2))
            draw.text((px+5,py+4),f'{r["review_id"]} {r["class"]} im{r["image_id"]} row{r["row"]} {round(w*W)}x{round(h*H)}',fill='black')
        sheet.save(out/f'sheet_{start//20+1:02d}.jpg',quality=92)
    print(json.dumps({k:v for k,v in manifest.items() if k!='samples'},indent=2)); print('Visual targets:',len(selected))

if __name__=='__main__':main()
