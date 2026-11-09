"""Create a deterministic size-stratified bottle/cup annotation review sample.

Creates inspection artifacts only. Does not edit labels, images, splits or models.
"""
import hashlib
import io
import json
from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/phase1/annotation_review"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def panel(image, box, target, title, font):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, outline="#ff3333", width=max(1, round(max(image.size)/600)))
    image.thumbnail((390, 280))
    target.paste(image, (5 + (390-image.width)//2, 28 + (280-image.height)//2))
    ImageDraw.Draw(target).text((7, 5), title, fill="black", font=font)


def main():
    raw = ROOT / "data/raw/plitterstreet"
    annotations = raw / "annotations.json"
    splits = ROOT / "data/prepared/plitterstreet/splits.json"
    data = json.loads(annotations.read_text())
    images = {x["id"]: x for x in data["images"]}
    assignment = {x["image_id"]: x for x in json.loads(splits.read_text())}
    names = {x["id"]: x["name"] for x in data["categories"]}
    rng = random.Random(42)
    selected = []
    OUT.mkdir(parents=True, exist_ok=True)
    for label in ("bottle", "cup"):
        for split in ("train", "val"):
            pool = [a for a in data["annotations"] if names[a["category_id"]] == label and assignment[a["image_id"]]["split"] == split]
            def size(a):
                i = images[a["image_id"]]
                return min(a["bbox"][2:]) * 640 / max(i["width"], i["height"])
            pool.sort(key=lambda a: (size(a), a["image_id"], a["id"]))
            used_images, used_groups = set(), set()
            for index, band in enumerate(("small", "medium", "large")):
                candidates = pool[len(pool)*index//3:len(pool)*(index+1)//3]
                rng.shuffle(candidates)
                chosen = 0
                for a in candidates:
                    group = assignment[a["image_id"]]["group_id"]
                    if a["image_id"] in used_images or group in used_groups:
                        continue
                    used_images.add(a["image_id"])
                    used_groups.add(group)
                    i = images[a["image_id"]]
                    key = f"{label}-{split}-{band}-{chosen+1}"
                    selected.append({"sample_id": key, "class": label, "split": split,
                        "size_stratum": band, "image_id": a["image_id"], "annotation_id": a["id"],
                        "file_name": i["file_name"], "group_id": group, "bbox": a["bbox"],
                        "source_size": [i["width"], i["height"]],
                        "short_side_at_640": size(a), "image_sha256": digest(raw/"images"/i["file_name"])})
                    chosen += 1
                    if chosen == 2:
                        break
                if chosen != 2:
                    raise RuntimeError(f"Insufficient independent examples for {label}/{split}/{band}")
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    for label in ("bottle", "cup"):
        for split in ("train", "val"):
            sheet = Image.new("RGB", (1200, 2100), "#eeeeee")
            subset = [x for x in selected if x["class"] == label and x["split"] == split]
            for row, item in enumerate(subset):
                # Marker-only JPEG repair in memory, matching prepared-data policy.
                content = (raw/"images"/item["file_name"]).read_bytes()
                if content[:2] == b'\xff\xd8' and not content.endswith(b'\xff\xd9'):
                    content += b'\xff\xd9'
                with Image.open(io.BytesIO(content)) as source:
                    image = source.convert("RGB")
                x,y,w,h = item["bbox"]
                if image.size != tuple(item["source_size"]):
                    raise ValueError(f"Image dimension mismatch: {item['sample_id']}")
                cx, cy = x+w/2, y+h/2
                side = max(w,h)*2.5
                side = max(side, 80)
                bounds = (max(0, int(cx-side/2)), max(0,int(cy-side/2)), min(image.width,int(cx+side/2+1)), min(image.height,int(cy+side/2+1)))
                crop = image.crop(bounds)
                crop_box = (x-bounds[0],y-bounds[1],x+w-bounds[0],y+h-bounds[1])
                factor = 640/max(image.size)
                scaled = image.resize((round(image.width*factor),round(image.height*factor)))
                small_bounds = (int(bounds[0]*factor),int(bounds[1]*factor),max(int(bounds[0]*factor)+1,round(bounds[2]*factor)),max(int(bounds[1]*factor)+1,round(bounds[3]*factor)))
                scaled_crop = scaled.crop(small_bounds)
                scaled_box = (x*factor-small_bounds[0],y*factor-small_bounds[1],(x+w)*factor-small_bounds[0],(y+h)*factor-small_bounds[1])
                for col, (view, box, title) in enumerate([
                    (image, (x,y,x+w,y+h), "Full image: target box in red"),
                    (crop, crop_box, "Original detail: enlarged crop"),
                    (scaled_crop, scaled_box, "640 input detail: enlarged crop")]):
                    tile = Image.new("RGB",(400,310),"white")
                    if col:
                        multiplier = min(360/view.width,260/view.height)
                        view = view.resize((max(1,round(view.width*multiplier)),max(1,round(view.height*multiplier))))
                        box = tuple(v*multiplier for v in box)
                    panel(view, box, tile, title, font)
                    sheet.paste(tile,(col*400,row*350+35))
                ImageDraw.Draw(sheet).text((8,row*350+8),f"{item['sample_id']} | image {item['image_id']}, ann {item['annotation_id']} | short side at 640: {item['short_side_at_640']:.1f}px",fill="black",font=font)
            sheet.save(OUT/f"{label}_{split}.jpg",quality=92)
    manifest = {"seed":42,"selection":"Two examples per class/split/size-tertile, distinct source groups within each class/split. Tertiles based on projected short side at 640; seeded shuffle within each tertile.",
        "classes":["bottle","cup"],"splits":["train","val"],"sample_count":len(selected),
        "annotation_sha256":digest(annotations),"split_manifest_sha256":digest(splits),
        "limitations":"Stratified inspection sample, not an estimate of dataset-wide label error rates. Detail views are magnified; no test images reviewed. Only selected target box drawn, not all annotations.",
        "samples":selected}
    (OUT/"sample_manifest.json").write_text(json.dumps(manifest,indent=2))
    print(f"Created {len(selected)} review examples across four contact sheets in {OUT}")


if __name__ == "__main__":
    main()
