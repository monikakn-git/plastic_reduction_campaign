"""
COCO to YOLO Converter, Splitter & EDA Generator
Fulfills Step 4 of the Implementation Plan:
- Converts COCO bounding boxes to normalized YOLO format [class_id, x_center, y_center, width, height]
- Splits dataset into Train, Val, and Test subsets (70/15/15)
- Prevents data leakage
- Generates dataset.yaml for YOLO training
- Generates EDA Report with bounding box statistics
"""

import os
import json
import shutil
import random
from pathlib import Path
import yaml
from PIL import Image
import pandas as pd


def load_params(params_path="params.yaml"):
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def map_category_to_target(cat_name, target_classes):
    """
    Maps various category string names to the 4 standard pLitterStreet classes:
    0: plastic
    1: waste_pile
    2: face_mask
    3: trash_bin
    """
    cat_lower = str(cat_name).lower().strip()
    if any(k in cat_lower for k in ["plastic", "bottle", "cup", "bag", "sachet", "wrapper", "straw", "litter"]):
        return 0  # plastic / general litter
    elif any(k in cat_lower for k in ["pile", "dump", "heap"]):
        return 1  # waste_pile
    elif any(k in cat_lower for k in ["mask", "face mask", "ppe"]):
        return 2  # face_mask
    elif any(k in cat_lower for k in ["bin", "trash bin", "dustbin", "can", "dumpster"]):
        return 3  # trash_bin
    return 0  # default fallback to plastic/litter


def coco_to_yolo_bbox(bbox, img_w, img_h):
    """Converts [x_min, y_min, width, height] to normalized [x_c, y_c, w, h]."""
    x, y, w, h = bbox
    # Clip coordinates within image
    x = max(0, x)
    y = max(0, y)
    w = min(w, img_w - x)
    h = min(h, img_h - y)

    x_c = (x + w / 2.0) / img_w
    y_c = (y + h / 2.0) / img_h
    w_norm = w / img_w
    h_norm = h / img_h

    # Bound to [0, 1]
    x_c = max(0.0, min(1.0, x_c))
    y_c = max(0.0, min(1.0, y_c))
    w_norm = max(0.0, min(1.0, w_norm))
    h_norm = max(0.0, min(1.0, h_norm))

    return round(x_c, 6), round(y_c, 6), round(w_norm, 6), round(h_norm, 6)


def preprocess_and_split(params_path="params.yaml", custom_raw_dir=None):
    params = load_params(params_path)
    raw_dir = Path(custom_raw_dir or params["dataset"]["raw_dir"])
    processed_dir = Path(params["dataset"]["processed_dir"])
    seed = params["base"]["random_seed"]
    random.seed(seed)

    print(f"[*] Preprocessing dataset from: {raw_dir}")
    print(f"[*] Target processed directory: {processed_dir}")

    annotation_file = raw_dir / "annotations.json"
    images_dir = raw_dir / "images"
    if not images_dir.exists():
        images_dir = raw_dir

    if not annotation_file.exists():
        print(f"[!] Warning: Annotations file not found at {annotation_file}. Skipping split.")
        return False

    with open(annotation_file, "r") as f:
        coco = json.load(f)

    # Categories
    cat_map = {c["id"]: c["name"] for c in coco.get("categories", [])}
    target_classes = params["dataset"]["classes"]

    # Image dict
    img_dict = {img["id"]: img for img in coco.get("images", [])}

    # Group annotations by image
    img_annotations = {img_id: [] for img_id in img_dict}
    bbox_stats = []

    for ann in coco.get("annotations", []):
        img_id = ann.get("image_id")
        if img_id not in img_dict:
            continue
        cat_name = cat_map.get(ann.get("category_id"), "plastic")
        target_cls_id = map_category_to_target(cat_name, target_classes)

        img_info = img_dict[img_id]
        iw, ih = img_info.get("width", 640), img_info.get("height", 640)
        bbox = ann.get("bbox", [])
        if len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:
            yolo_box = coco_to_yolo_bbox(bbox, iw, ih)
            img_annotations[img_id].append((target_cls_id, yolo_box))
            # Record bbox area for EDA
            area_px = bbox[2] * bbox[3]
            bbox_stats.append({
                "category": target_classes[target_cls_id],
                "area_px": area_px,
                "aspect_ratio": bbox[2] / max(1, bbox[3]),
                "scale": "small" if area_px < 32**2 else ("medium" if area_px < 96**2 else "large")
            })

    # Prepare Split
    all_img_ids = list(img_dict.keys())
    random.shuffle(all_img_ids)

    n_total = len(all_img_ids)
    n_train = int(n_total * params["dataset"]["train_split"])
    n_val = int(n_total * params["dataset"]["val_split"])

    train_ids = set(all_img_ids[:n_train])
    val_ids = set(all_img_ids[n_train:n_train + n_val])
    test_ids = set(all_img_ids[n_train + n_val:])

    splits = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids
    }

    # Setup target subdirectories
    for split_name in ["train", "val", "test"]:
        os.makedirs(processed_dir / "images" / split_name, exist_ok=True)
        os.makedirs(processed_dir / "labels" / split_name, exist_ok=True)

    copied_images = 0
    written_labels = 0

    print("[*] Writing YOLO format images and labels...")
    for split_name, id_set in splits.items():
        for img_id in id_set:
            img_info = img_dict[img_id]
            file_name = img_info.get("file_name", "")
            src_img_path = images_dir / file_name

            # Handle if file_name has subpaths
            base_name = Path(file_name).name
            dst_img_path = processed_dir / "images" / split_name / base_name
            dst_label_path = processed_dir / "labels" / split_name / f"{Path(base_name).stem}.txt"

            if src_img_path.exists():
                shutil.copy2(src_img_path, dst_img_path)
                copied_images += 1

            # Write label file
            with open(dst_label_path, "w") as lf:
                for cls_id, (xc, yc, w, h) in img_annotations.get(img_id, []):
                    lf.write(f"{cls_id} {xc} {yc} {w} {h}\n")
                    written_labels += 1

    # Create dataset.yaml
    dataset_yaml_content = {
        "path": str(processed_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {int(k): v for k, v in target_classes.items()}
    }

    yaml_path = processed_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml_content, f, sort_keys=False)

    print(f"[✓] Created dataset configuration: {yaml_path}")
    print(f"[✓] Processed: {copied_images} images copied, {written_labels} annotations written across train/val/test.")

    # Generate EDA Report
    generate_eda_report(splits, bbox_stats, params, "reports/eda_report.md")
    return True


def generate_eda_report(splits, bbox_stats, params, report_path):
    df_bbox = pd.DataFrame(bbox_stats)
    report_md = f"""# Exploratory Data Analysis (EDA) Report
**Project:** AI-Based Waste Reduction Campaign  
**Dataset:** {params['dataset']['name']} (YOLO Processed Format)  

---

## 1. Dataset Partitioning (Preventing Data Leakage)
| Subset | Images Count | Proportion | Purpose |
| :--- | :--- | :--- | :--- |
| **Train** | {len(splits['train'])} | {params['dataset']['train_split']*100:.0f}% | Model feature learning and parameter optimization |
| **Validation** | {len(splits['val'])} | {params['dataset']['val_split']*100:.0f}% | Hyperparameter tuning and validation metrics (mAP50) |
| **Test** | {len(splits['test'])} | {params['dataset']['test_split']*100:.0f}% | Final unbiased model benchmark and latency evaluation |

---

## 2. Bounding Box Scale Analysis (COCO Metric Standards)
"""
    if not df_bbox.empty and "scale" in df_bbox:
        scale_counts = df_bbox["scale"].value_counts().to_dict()
        total = len(df_bbox)
        report_md += """| Object Scale Size | Pixel Area Range | Count | Percentage |
| :--- | :--- | :--- | :--- |
"""
        for sc, name, r in [("small", "Small Objects", "< 32x32 px"), ("medium", "Medium Objects", "32x32 to 96x96 px"), ("large", "Large Objects", "> 96x96 px")]:
            c = scale_counts.get(sc, 0)
            report_md += f"| {name} | {r} | {c} | {(c/total)*100:.1f}% |\n"
    else:
        report_md += "*No bounding box statistics recorded yet.*"

    report_md += """
---

## 3. Data Augmentation Strategies Configured for Training
To enhance generalization under complex outdoor environmental conditions, the training pipeline implements:
* **Mosaic & MixUp:** Combines 4 street scenes into one to handle dense litter clusters.
* **Photometric Distortions:** HSV color space shifts (Hue: 0.015, Saturation: 0.7, Value: 0.4) for varied sunlight.
* **Affine Transformations:** Random horizontal flips (p=0.5), scale jitter (+/- 50%), and perspective adjustments.
"""

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"[✓] EDA Report saved to: {report_path}")


if __name__ == "__main__":
    preprocess_and_split()
