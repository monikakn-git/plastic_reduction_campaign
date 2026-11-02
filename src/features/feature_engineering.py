"""
EcoPulse AI - Feature Engineering for 10-class YOLO waste dataset.

This script reads existing YOLO-format labels and creates analysis-ready
features and quality reports. It does not modify images/labels or train a model.

Expected dataset structure:
Ai_Waste_Campaign/
    processed/
        data.yaml
        labels/
            train/
            val/
            test/

Outputs are written to:
D:/mlops/artifacts/feature_engineering/
"""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "PyYAML is required. Install it in your active venv with: pip install pyyaml"
    ) from exc


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
DATASET_ROOT = Path(r"G:\My Drive\datasets\Ai_Waste_Campaign")
DATA_YAML = DATASET_ROOT / "processed" / "data.yaml"
LABELS_ROOT = DATASET_ROOT / "processed" / "labels"
OUTPUT_DIR = Path(r"D:\mlops\artifacts\feature_engineering")

SPLITS = ("train", "val", "test")

# Fallback class names in case data.yaml does not contain a names field.
FALLBACK_CLASS_NAMES = {
    0: "litter",
    1: "pile",
    2: "face_mask",
    3: "trash_bin",
    4: "plastic_bag",
    5: "bottle",
    6: "cup",
    7: "rope",
    8: "sachet",
    9: "straw",
}


def load_class_names() -> dict[int, str]:
    """Load class names from data.yaml, falling back to the known 10 classes."""
    if not DATA_YAML.is_file():
        print(f"WARNING: {DATA_YAML} not found; using fallback class names.")
        return FALLBACK_CLASS_NAMES.copy()

    with DATA_YAML.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    names: Any = config.get("names", FALLBACK_CLASS_NAMES)

    if isinstance(names, list):
        class_names = {index: str(name) for index, name in enumerate(names)}
    elif isinstance(names, dict):
        class_names = {int(index): str(name) for index, name in names.items()}
    else:
        class_names = FALLBACK_CLASS_NAMES.copy()

    return class_names


def parse_yolo_label_file(
    label_path: Path,
    class_names: dict[int, str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Parse YOLO rows: class_id x_center y_center width height.
    Coordinates are normalized to the range 0..1.
    """
    objects: list[dict[str, Any]] = []
    errors: list[str] = []

    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        return objects, [f"Could not read file: {exc}"]

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            errors.append(
                f"Line {line_number}: expected 5 values, found {len(parts)}"
            )
            continue

        try:
            class_value = float(parts[0])
            coordinates = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(f"Line {line_number}: non-numeric value")
            continue

        if not class_value.is_integer():
            errors.append(f"Line {line_number}: class ID must be an integer")
            continue

        class_id = int(class_value)
        x_center, y_center, width, height = coordinates

        if class_id not in class_names:
            errors.append(f"Line {line_number}: unknown class ID {class_id}")
            continue

        if not all(math.isfinite(value) for value in coordinates):
            errors.append(f"Line {line_number}: non-finite coordinate")
            continue

        if not (
            0 <= x_center <= 1
            and 0 <= y_center <= 1
            and 0 < width <= 1
            and 0 < height <= 1
        ):
            errors.append(f"Line {line_number}: coordinates outside valid range")
            continue

        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2

        if x_min < 0 or y_min < 0 or x_max > 1 or y_max > 1:
            errors.append(f"Line {line_number}: bounding box extends outside image")
            continue

        area = width * height
        if area < 0.01:
            size_category = "small"
        elif area < 0.10:
            size_category = "medium"
        else:
            size_category = "large"

        objects.append(
            {
                "class_id": class_id,
                "class_name": class_names[class_id],
                "x_center": x_center,
                "y_center": y_center,
                "bbox_width": width,
                "bbox_height": height,
                "bbox_area": area,
                "aspect_ratio": width / height,
                "size_category": size_category,
            }
        )

    return objects, errors


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write a list of dictionaries to CSV, including headers for empty reports."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_split(
    split: str,
    class_names: dict[int, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    labels_dir = LABELS_ROOT / split
    if not labels_dir.is_dir():
        raise FileNotFoundError(
            f"Label directory for split '{split}' not found: {labels_dir}"
        )

    label_files = sorted(labels_dir.glob("*.txt"))
    if not label_files:
        raise FileNotFoundError(f"No .txt label files found in: {labels_dir}")

    image_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []

    for label_path in label_files:
        objects, errors = parse_yolo_label_file(label_path, class_names)
        counts = Counter(obj["class_name"] for obj in objects)

        image_feature: dict[str, Any] = {
            "split": split,
            "label_file": label_path.name,
            "object_count": len(objects),
            "unique_class_count": len(counts),
            "mean_bbox_area": (
                sum(obj["bbox_area"] for obj in objects) / len(objects)
                if objects else 0.0
            ),
            "mean_bbox_width": (
                sum(obj["bbox_width"] for obj in objects) / len(objects)
                if objects else 0.0
            ),
            "mean_bbox_height": (
                sum(obj["bbox_height"] for obj in objects) / len(objects)
                if objects else 0.0
            ),
            "small_object_count": sum(
                obj["size_category"] == "small" for obj in objects
            ),
            "medium_object_count": sum(
                obj["size_category"] == "medium" for obj in objects
            ),
            "large_object_count": sum(
                obj["size_category"] == "large" for obj in objects
            ),
        }

        for class_id, class_name in sorted(class_names.items()):
            image_feature[f"class_{class_id}_{class_name}_count"] = counts.get(
                class_name, 0
            )

        image_rows.append(image_feature)

        for object_index, obj in enumerate(objects, start=1):
            object_rows.append(
                {
                    "split": split,
                    "label_file": label_path.name,
                    "object_index": object_index,
                    **obj,
                }
            )

        if errors:
            quality_rows.append(
                {
                    "split": split,
                    "label_file": label_path.name,
                    "error_count": len(errors),
                    "errors": " | ".join(errors),
                }
            )

    return image_rows, object_rows, quality_rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    class_names = load_class_names()

    print("EcoPulse AI - 10-class feature extraction")
    print(f"Dataset: {DATASET_ROOT}")
    print(f"Classes found: {len(class_names)}")
    print("Reading existing YOLO labels. No training will be performed.\n")

    all_image_rows: list[dict[str, Any]] = []
    all_object_rows: list[dict[str, Any]] = []
    all_quality_rows: list[dict[str, Any]] = []

    for split in SPLITS:
        image_rows, object_rows, quality_rows = process_split(split, class_names)
        all_image_rows.extend(image_rows)
        all_object_rows.extend(object_rows)
        all_quality_rows.extend(quality_rows)

        print(
            f"{split}: {len(image_rows)} label files, "
            f"{len(object_rows)} valid objects, "
            f"{len(quality_rows)} files with label issues"
        )

    image_fields = [
        "split", "label_file", "object_count", "unique_class_count",
        "mean_bbox_area", "mean_bbox_width", "mean_bbox_height",
        "small_object_count", "medium_object_count", "large_object_count",
    ]
    for class_id, class_name in sorted(class_names.items()):
        image_fields.append(f"class_{class_id}_{class_name}_count")

    object_fields = [
        "split", "label_file", "object_index", "class_id", "class_name",
        "x_center", "y_center", "bbox_width", "bbox_height", "bbox_area",
        "aspect_ratio", "size_category",
    ]
    quality_fields = ["split", "label_file", "error_count", "errors"]

    write_csv(OUTPUT_DIR / "image_features.csv", all_image_rows, image_fields)
    write_csv(OUTPUT_DIR / "object_features.csv", all_object_rows, object_fields)
    write_csv(OUTPUT_DIR / "label_quality_issues.csv", all_quality_rows, quality_fields)

    # Class counts per split.
    class_counter: dict[str, Counter] = defaultdict(Counter)
    for obj in all_object_rows:
        class_counter[obj["split"]][obj["class_name"]] += 1

    class_summary_rows = []
    for split in SPLITS:
        for class_id, class_name in sorted(class_names.items()):
            class_summary_rows.append(
                {
                    "split": split,
                    "class_id": class_id,
                    "class_name": class_name,
                    "object_count": class_counter[split].get(class_name, 0),
                }
            )

    write_csv(
        OUTPUT_DIR / "class_distribution.csv",
        class_summary_rows,
        ["split", "class_id", "class_name", "object_count"],
    )

    summary_rows = []
    for split in SPLITS:
        split_images = [row for row in all_image_rows if row["split"] == split]
        split_objects = [row for row in all_object_rows if row["split"] == split]
        summary_rows.append(
            {
                "split": split,
                "label_files": len(split_images),
                "valid_objects": len(split_objects),
                "files_with_label_issues": sum(
                    row["split"] == split for row in all_quality_rows
                ),
                "average_objects_per_label_file": (
                    len(split_objects) / len(split_images) if split_images else 0
                ),
            }
        )

    write_csv(
        OUTPUT_DIR / "dataset_feature_summary.csv",
        summary_rows,
        [
            "split", "label_files", "valid_objects",
            "files_with_label_issues", "average_objects_per_label_file",
        ],
    )

    print("\nFeature extraction completed.")
    print(f"Output folder: {OUTPUT_DIR}")
    print("Generated files:")
    print("  - image_features.csv")
    print("  - object_features.csv")
    print("  - class_distribution.csv")
    print("  - dataset_feature_summary.csv")
    print("  - label_quality_issues.csv")
    print("\nNote: label files are analyzed as-is; source data was not changed.")


if __name__ == "__main__":
    main()
