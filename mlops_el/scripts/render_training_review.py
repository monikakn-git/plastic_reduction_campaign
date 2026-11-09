"""Render retained visual-review selections; does not change annotations."""
from pathlib import Path
import json
from PIL import Image,ImageOps,ImageDraw
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--audit',default='reports/phase1/training_quality_v4');args=parser.parse_args()
root=Path(__file__).resolve().parents[1];out=root/args.audit;m=json.loads((out/'manifest.json').read_text());src=root/m['source'];ids=json.loads((out/'detail_selection.json').read_text())['review_ids']
items=[r for r in m['samples'] if r['review_id'] in ids]

for start in range(0,len(items),6):
 sheet=Image.new('RGB',(1500,1350),'white'); d=ImageDraw.Draw(sheet)
 for i,r in enumerate(items[start:start+6]):
  im=ImageOps.exif_transpose(Image.open(src/'images/train'/r['file_name'])).convert('RGB'); W,H=im.size; x,y,w,h=r['box']; b=((x-w/2)*W,(y-h/2)*H,(x+w/2)*W,(y+h/2)*H)
  raw=im.copy(); ImageDraw.Draw(im).rectangle(b,outline='red',width=max(2,W//500)); im.thumbnail((350,410)); px=i%2*750;py=i//2*450;sheet.paste(im,(px,py+30))
  margin=max(12,max(w*W,h*H)*.15); cb=(max(0,int(b[0]-margin)),max(0,int(b[1]-margin)),min(W,int(b[2]+margin+1)),min(H,int(b[3]+margin+1)))
  crop=raw.crop(cb); scale=min(390/crop.width,400/crop.height,8);crop=crop.resize((max(1,round(crop.width*scale)),max(1,round(crop.height*scale)))); sheet.paste(crop,(px+355,py+30));d.text((px+4,py+8),f'{r["review_id"]} {r["class"]} image{r["image_id"]} row{r["row"]}',fill='black')
 sheet.save(out/f'detail_{start//6+1:02d}.jpg',quality=94)
