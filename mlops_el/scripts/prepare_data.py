"""Validate COCO labels and prepare grouped YOLO splits without copying image bytes."""
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import random

import yaml

ROOT = Path(__file__).resolve().parents[1]
ACTIVE = yaml.safe_load((ROOT / "params.yaml").read_text())["dataset"]
RAW = ROOT / "data/raw" / ACTIVE["name"]
OUT = ROOT / "data/prepared" / ACTIVE["name"]
REPORT = ROOT / "reports/phase1"


def convert_box(box, width, height):
    if width <= 0 or height <= 0 or len(box) != 4 or not all(math.isfinite(v) for v in box):
        raise ValueError("Invalid image dimensions or bounding box")
    x, y, w, h = box
    if w <= 0 or h <= 0:
        raise ValueError("Bounding box must have positive area")
    left, top = max(0, x), max(0, y)
    right, bottom = min(width, x + w), min(height, y + h)
    if right <= left or bottom <= top:
        raise ValueError("Bounding box outside image")
    return ((left + right) / (2 * width), (top + bottom) / (2 * height),
            (right - left) / width, (bottom - top) / height)


def prepare_image(source, target, expected_sha256):
    """Preserve raw files; repair JPEG EOF marker without pixel re-encoding."""
    with source.open('rb') as stream:
        stream.seek(-2, 2)
        repair = stream.read() != b'\xff\xd9'
    expected = expected_sha256
    repaired_bytes = None
    if repair:
        repaired_bytes = source.read_bytes() + b'\xff\xd9'
        expected = hashlib.sha256(repaired_bytes).hexdigest()
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() == expected:
            return repair
        # Replace only derived output, never alter linked originals in place.
        target.unlink()
    if repair:
        target.write_bytes(repaired_bytes)
    else:
        os.link(source, target)
    return repair


def group_splits(images, manifest, seed, train_fraction, val_fraction):
    """Keep source batches and exact duplicates together; batches are conservative proxies."""
    image_groups = {i['id']: str(i.get('group_id', Path(i['file_name']).parts[0])) for i in images}
    parents = {g:g for g in image_groups.values()}
    def find(x):
        while parents[x] != x:
            x = parents[x]
        return x
    seen = {}
    for item in manifest:
        batch = image_groups[item['image_id']]
        if item["sha256"] in seen:
            parents[find(batch)] = find(seen[item["sha256"]])
        seen[item["sha256"]] = batch
    groups = sorted({find(x) for x in parents})
    if len(groups) < 3:
        raise ValueError("At least three independent groups required")
    random.Random(seed).shuffle(groups)
    sizes = collections.Counter(find(g) for g in image_groups.values())
    groups.sort(key=lambda g:-sizes[g])
    fractions = {'train':train_fraction,'val':val_fraction,'test':1-train_fraction-val_fraction}
    if min(fractions.values()) <= 0:
        raise ValueError('All split fractions must be positive')
    targets = {s:f*len(images) for s,f in fractions.items()}
    assigned = {s:0 for s in fractions}
    assignment = {}
    for group in groups:
        split = min(fractions, key=lambda s:(assigned[s]+sizes[group])/targets[s])
        assignment[group] = split
        assigned[split] += sizes[group]
    return {i['id']:assignment[find(image_groups[i['id']])] for i in images}


def main():
    data = json.loads((RAW / "annotations.json").read_text())
    manifest = json.loads((RAW / "manifest.json").read_text())
    acquisition = json.loads((REPORT / "acquisition.json").read_text())
    if acquisition["status"] != "complete" or len(manifest) != len(data["images"]):
        raise SystemExit("Complete and verify acquisition before preparing splits.")
    if len(data['images']) != ACTIVE['target_images']:
        raise SystemExit('Image count does not match the project target')
    params = yaml.safe_load((ROOT / "params.yaml").read_text())["prepare"]
    classes = json.loads((ROOT / params["classes"]).read_text())
    names = list(classes)
    source_to_target = {source: idx for idx, sources in enumerate(classes.values()) for source in sources}
    categories = {c["id"]: c["name"] for c in data["categories"]}
    unknown = set(source_to_target) - set(categories.values())
    if unknown:
        raise ValueError(f"Unknown class mappings: {unknown}")
    mapping = {cid: source_to_target[name] for cid, name in categories.items() if name in source_to_target}
    images = {i["id"]: i for i in data["images"]}
    if len(images) != len(data["images"]):
        raise ValueError("Duplicate image IDs")
    labels = collections.defaultdict(list)
    original_counts = collections.Counter()
    clipped = []
    annotation_ids = set()
    for ann in data["annotations"]:
        identity = (ann["image_id"], ann["id"])
        if identity in annotation_ids:
            raise ValueError("Duplicate annotation ID within an image")
        annotation_ids.add(identity)
        image = images[ann["image_id"]]
        original_counts[categories[ann["category_id"]]] += 1
        box = convert_box(ann["bbox"], image["width"], image["height"])
        x, y, w, h = ann["bbox"]
        if x < 0 or y < 0 or x + w > image["width"] or y + h > image["height"]:
            clipped.append(ann["id"])
        if ann["category_id"] in mapping:
            labels[image["id"]].append((mapping[ann["category_id"]], *box))
    splits = group_splits(data["images"], manifest, params["seed"],
                          params["train_fraction"], params["val_fraction"])
    counts = {s: collections.Counter() for s in ("train", "val", "test")}
    split_records = []
    jpeg_marker_repairs = []
    hashes = {i["image_id"]: i["sha256"] for i in manifest}
    for image in data["images"]:
        split = splits[image["id"]]
        source = RAW / "images" / image["file_name"]
        flat = "__".join(Path(image["file_name"]).parts)
        target = OUT / "images" / split / flat
        target.parent.mkdir(parents=True, exist_ok=True)
        if prepare_image(source, target, hashes[image["id"]]):
            jpeg_marker_repairs.append(image["file_name"])
        label = OUT / "labels" / split / Path(flat).with_suffix(".txt")
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("".join(" ".join([str(row[0]), *(f"{v:.9f}" for v in row[1:])]) + "\n"
                                 for row in labels[image["id"]]))
        for row in labels[image["id"]]:
            counts[split][names[row[0]]] += 1
        split_records.append({"image_id": image["id"], "file_name": image["file_name"],
                              "split": split, "sha256": hashes[image["id"]],
                              "group_id": image.get('group_id',Path(image['file_name']).parts[0])})
    for split, counter in counts.items():
        missing = set(names) - set(counter)
        if missing:
            raise ValueError(f"Classes missing from {split}: {missing}; review split strategy")
    OUT.mkdir(parents=True, exist_ok=True)
    # Relative path is resolved by the training wrapper; no machine-specific paths are versioned.
    (OUT / "dataset.yaml").write_text(yaml.safe_dump({"path": f"data/prepared/{ACTIVE['name']}",
        "train": "images/train", "val": "images/val", "test": "images/test",
        "names": dict(enumerate(names))}, sort_keys=False))
    (OUT / "splits.json").write_text(json.dumps(split_records, indent=2))
    summary = {"dataset": ACTIVE['name'], "images": len(images), "source_annotations": len(data["annotations"]),
        "target_annotations": sum(sum(v.values()) for v in counts.values()),
        "classes": names, "original_category_counts": dict(original_counts.most_common()),
        "split_image_counts": dict(collections.Counter(splits.values())),
        "split_target_counts": {s: dict(c) for s, c in counts.items()},
        "negative_images": sum(not labels[i] for i in images),
        "clipped_annotation_ids": clipped,
        "exact_duplicate_images": len(manifest) - len(set(hashes.values())),
        "source_annotation_id_collisions": len(data["annotations"]) - len({a["id"] for a in data["annotations"]}),
        "annotation_identity": "Composite (image_id, annotation_id) identity; source ID collisions counted separately; no annotations discarded",
        "prepared_jpeg_marker_repairs": jpeg_marker_repairs,
        "jpeg_repair_method": "Append JPEG end marker in derived file only; no resizing or pixel re-encoding; raw hash unchanged",
        "split_strategy": "Whole source-video groups kept together; exact duplicates merged; size-balanced seeded assignment",
        "split_limitation": "Groups inferred from source dataset and video filename prefixes. Same scenes across different recordings may require additional review.",
        "original_image_bytes_preserved": True}
    (REPORT / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
