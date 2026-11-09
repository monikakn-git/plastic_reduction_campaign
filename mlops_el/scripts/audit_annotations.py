"""Prepare a reproducible 100-target audit without editing dataset artifacts."""
from collections import Counter, defaultdict
import io
import json
from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFont
from review_annotations import ROOT, digest, panel

OUT = ROOT/"reports/phase1/annotation_audit_100"


def main():
    raw = ROOT/"data/raw/plitterstreet"
    data = json.loads((raw/"annotations.json").read_text())
    splits = ROOT/"data/prepared/plitterstreet/splits.json"
    assignment = {x["image_id"]:x for x in json.loads(splits.read_text())}
    images = {x["id"]:x for x in data["images"]}
    all_boxes = defaultdict(list)
    for a in data["annotations"]:
        all_boxes[a["image_id"]].append(a)
    rng = random.Random(2026)
    selected, used_images = [], set()
    checks = []
    for category,label in ((6,"bottle"),(7,"cup")):
        for split in ("train","val"):
            def size(a):
                i=images[a["image_id"]]
                return min(a["bbox"][2:])*640/max(i["width"],i["height"])
            pool=sorted([a for a in data["annotations"] if a["category_id"]==category and assignment[a["image_id"]]["split"]==split],key=lambda a:(size(a),a["image_id"],a["id"]))
            group_use=Counter()
            for band_index,(band,count) in enumerate((("small",8),("medium",8),("large",9))):
                candidates=pool[len(pool)*band_index//3:len(pool)*(band_index+1)//3]
                rng.shuffle(candidates)
                for index in range(count):
                    available=[a for a in candidates if a["image_id"] not in used_images]
                    if not available:
                        raise RuntimeError(f"Insufficient distinct images: {label}/{split}/{band}")
                    a=min(available,key=lambda a:group_use[assignment[a["image_id"]]["group_id"]])
                    i=images[a["image_id"]];group=assignment[a["image_id"]]["group_id"]
                    used_images.add(a["image_id"]);group_use[group]+=1
                    x,y,w,h=a["bbox"]
                    expected=[(x+w/2)/i["width"],(y+h/2)/i["height"],w/i["width"],h/i["height"]]
                    label_path=ROOT/f"data/prepared/plitterstreet/labels/{split}"/Path(i["file_name"]).with_suffix('.txt')
                    entries=[line.split() for line in label_path.read_text().splitlines() if line.strip()]
                    match=any(int(row[0])==category-1 and max(abs(float(v)-e) for v,e in zip(row[1:],expected))<1e-6 for row in entries)
                    valid=w>0 and h>0 and x>=0 and y>=0 and x+w<=i["width"]+1e-6 and y+h<=i["height"]+1e-6
                    checks.append({"image_id":i["id"],"annotation_id":a["id"],"box_in_bounds":valid,"prepared_label_matches":match})
                    if not match or not valid:
                        raise RuntimeError("Sample label conversion or bounds failed")
                    selected.append({"sample_id":f"{label}-{split}-{band}-{index+1:02d}","class":label,"split":split,
                        "size_stratum":band,"image_id":i["id"],"annotation_id":a["id"],"file_name":i["file_name"],
                        "group_id":group,"bbox":a["bbox"],"source_size":[i["width"],i["height"]],"short_side_at_640":size(a),
                        "image_sha256":digest(raw/"images"/i["file_name"]),"prepared_label_sha256":digest(label_path),
                        "image_annotation_count":len(all_boxes[i["id"]])})
    OUT.mkdir(parents=True,exist_ok=True)
    try:font=ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf",16)
    except OSError:font=ImageFont.load_default()
    for label in ("bottle","cup"):
        for split in ("train","val"):
            subset=[x for x in selected if x["class"]==label and x["split"]==split]
            for start in range(0,25,5):
                sheet=Image.new("RGB",(1200,1750),"#eeeeee")
                for row,item in enumerate(subset[start:start+5]):
                    content=(raw/"images"/item["file_name"]).read_bytes()
                    if content[:2]==b'\xff\xd8' and not content.endswith(b'\xff\xd9'):content+=b'\xff\xd9'
                    with Image.open(io.BytesIO(content)) as source:image=source.convert('RGB')
                    assert image.size==tuple(item["source_size"])
                    x,y,w,h=item["bbox"];cx,cy=x+w/2,y+h/2;side=max(80,max(w,h)*2.5)
                    bounds=(max(0,int(cx-side/2)),max(0,int(cy-side/2)),min(image.width,int(cx+side/2+1)),min(image.height,int(cy+side/2+1)))
                    crop=image.crop(bounds);cropbox=(x-bounds[0],y-bounds[1],x+w-bounds[0],y+h-bounds[1])
                    factor=640/max(image.size)
                    small=image.resize((round(image.width*factor),round(image.height*factor)))
                    sb=(int(bounds[0]*factor),int(bounds[1]*factor),max(int(bounds[0]*factor)+1,round(bounds[2]*factor)),max(int(bounds[1]*factor)+1,round(bounds[3]*factor)))
                    smallcrop=small.crop(sb);smallbox=(x*factor-sb[0],y*factor-sb[1],(x+w)*factor-sb[0],(y+h)*factor-sb[1])
                    context=image.copy();draw=ImageDraw.Draw(context)
                    for other in all_boxes[item["image_id"]]:
                        ox,oy,ow,oh=other['bbox'];draw.rectangle((ox,oy,ox+ow,oy+oh),outline='#888888',width=max(1,round(max(image.size)/800)))
                    for col,(view,box,title) in enumerate([(context,(x,y,x+w,y+h),"Context: target red, other labels grey"),(crop,cropbox,"Original target detail (magnified)"),(smallcrop,smallbox,"640 input detail (magnified)")]):
                        tile=Image.new('RGB',(400,310),'white')
                        if col:
                            mult=min(360/view.width,260/view.height)
                            view=view.resize((max(1,round(view.width*mult)),max(1,round(view.height*mult))))
                            box=tuple(v*mult for v in box)
                        panel(view,box,tile,title,font)
                        sheet.paste(tile,(col*400,row*350+35))
                    ImageDraw.Draw(sheet).text((7,row*350+8),f"{item['sample_id']} | img {item['image_id']} ann {item['annotation_id']} | short side {item['short_side_at_640']:.1f}px",fill='black',font=font)
                    item['contact_sheet']=f"{label}_{split}_{start//5+1:02d}.jpg"
                sheet.save(OUT/f"{label}_{split}_{start//5+1:02d}.jpg",quality=90)
    # Magnified all-label context for the two scenes reviewed in context_findings.json.
    category_names={x['id']:x['name'] for x in data['categories']}
    for image_id in (14677,5720):
        item=next(x for x in selected if x['image_id']==image_id)
        with Image.open(raw/'images'/item['file_name']) as source:image=source.convert('RGB')
        x,y,w,h=item['bbox'];pad=max(w,h)*2
        bounds=(max(0,int(x-pad)),max(0,int(y-pad)),min(image.width,int(x+w+pad)),min(image.height,int(y+h+pad)))
        crop=image.crop(bounds);ratio=min(2,1100/crop.width)
        crop=crop.resize((round(crop.width*ratio),round(crop.height*ratio)))
        draw=ImageDraw.Draw(crop)
        for other in all_boxes[image_id]:
            bx,by,bw,bh=other['bbox']
            if bx+bw<bounds[0] or bx>bounds[2] or by+bh<bounds[1] or by>bounds[3]:continue
            box=((bx-bounds[0])*ratio,(by-bounds[1])*ratio,(bx+bw-bounds[0])*ratio,(by+bh-bounds[1])*ratio)
            color='red' if other['id']==item['annotation_id'] else '#00aaff'
            draw.rectangle(box,outline=color,width=2)
            draw.text((max(0,box[0]),max(0,box[1]-20)),f"{category_names[other['category_id']]} {other['id']}",font=font,fill=color,stroke_width=1,stroke_fill='white')
        crop.save(OUT/f'context_detail_{image_id}.jpg',quality=92)
    protected=["data/raw/plitterstreet/annotations.json","data/prepared/plitterstreet/splits.json","params.yaml",
        "models/plitterstreet-baseline.pt","models/plitterstreet-yolo11s-candidate.pt",
        "reports/phase1/evaluation_val.json","reports/phase1/evaluation_test.json","reports/phase1/yolo11s_candidate_validation.json"]
    manifest={"seed":2026,"sample_count":100,"unique_images":len(used_images),"selection":"25 targets per class/split: 8 small, 8 medium and 9 large by within-pool short-side tertiles. Seeded shuffle, prefer least-used inferred source group; globally distinct images. Group repetition is necessary in validation (especially cups: 14 available groups).",
        "scope":"Training and validation only; no model predictions used for sampling. Not a probability sample for dataset-wide error estimation.",
        "protected_sha256":{p:digest(ROOT/p) for p in protected},"samples":selected}
    (OUT/"sample_manifest.json").write_text(json.dumps(manifest,indent=2))
    (OUT/"technical_checks.json").write_text(json.dumps({"sampled_targets":len(checks),"all_boxes_in_bounds":all(x['box_in_bounds'] for x in checks),"all_prepared_labels_match_source":all(x['prepared_label_matches'] for x in checks),"checks":checks},indent=2))
    print(f"Prepared 100 distinct-image targets in 20 sheets: {OUT}")


if __name__=='__main__':main()
