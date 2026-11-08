from pathlib import Path
import argparse
import json
import os
import sys
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_YOLO_ROOT = PROJECT_ROOT / "artifacts" / "datasets" / "processed"
YOLO_ROOT = Path(
    os.environ.get(
        "WASTE_YOLO_DIR",
        os.environ.get("WASTE_DATA_DIR", str(DEFAULT_YOLO_ROOT)),
    )
)

SPLITS = ("train", "val", "test")
VALID_CLASSES = set(range(10))  # 10 waste detection classes (0 to 9)

EXPECTED_CLASS_MAPPING = {
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


def validate_data_yaml(dataset_dir: Path) -> dict:
    """Validate data.yaml existence and class mappings."""
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Missing data.yaml at {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    names = cfg.get("names", {})
    if isinstance(names, list):
        names_dict = {i: str(n) for i, n in enumerate(names)}
    elif isinstance(names, dict):
        names_dict = {int(k): str(v) for k, v in names.items()}
    else:
        raise ValueError(f"Invalid 'names' field in {yaml_path}")

    if names_dict != EXPECTED_CLASS_MAPPING:
        print(f"Warning: data.yaml names differ from expected 10 canonical classes.")
        print(f"Expected: {EXPECTED_CLASS_MAPPING}")
        print(f"Found: {names_dict}")

    return {
        "yaml_path": str(yaml_path),
        "names": names_dict,
        "raw_config": cfg,
    }


def validate_split(split: str, root_dir: Path | None = None) -> dict:
    dataset_dir = root_dir or YOLO_ROOT
    images_dir = dataset_dir / "images" / split
    labels_dir = dataset_dir / "labels" / split

    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing image or label directory for {split} at {dataset_dir}")

    images = {
        path.stem: path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
        }
    }
    labels = {
        path.stem: path for path in labels_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".txt"
    }

    errors = []
    empty_labels = 0
    box_count = 0
    class_counts = {i: 0 for i in VALID_CLASSES}

    missing_labels = set(images.keys()) - set(labels.keys())
    orphan_labels = set(labels.keys()) - set(images.keys())

    for stem in missing_labels:
        errors.append(f"Missing label file for image: {images[stem].name}")

    for stem in orphan_labels:
        errors.append(f"Orphan label file without image: {labels[stem].name}")

    for stem, label in labels.items():
        content = label.read_text(encoding="utf-8").strip()
        if not content:
            empty_labels += 1
            continue

        for line_number, line in enumerate(content.splitlines(), start=1):
            fields = line.split()

            if len(fields) != 5:
                errors.append(f"{label.name}:{line_number}: expected 5 values")
                continue

            try:
                class_id = int(fields[0])
                x, y, width, height = map(float, fields[1:])
            except ValueError:
                errors.append(f"{label.name}:{line_number}: non-numeric values")
                continue

            if class_id not in VALID_CLASSES:
                errors.append(f"{label.name}:{line_number}: invalid class {class_id}")
                continue

            if not all(0 <= value <= 1 for value in (x, y, width, height)):
                errors.append(f"{label.name}:{line_number}: values outside [0, 1]")
                continue

            if width <= 0 or height <= 0:
                errors.append(f"{label.name}:{line_number}: non-positive box size")
                continue

            box_count += 1
            class_counts[class_id] += 1

    summary = {
        "split": split,
        "images_count": len(images),
        "labels_count": len(labels),
        "empty_labels": empty_labels,
        "valid_boxes": box_count,
        "class_counts": class_counts,
        "errors_count": len(errors),
        "errors": errors[:25],
    }

    print(f"\n{split.upper()}")
    print(f"Images found: {len(images):,}")
    print(f"Empty label files (background): {empty_labels:,}")
    print(f"Valid boxes counted: {box_count:,}")
    print(f"Errors: {len(errors):,}")

    for error in errors[:10]:
        print(" -", error)

    if errors:
        raise RuntimeError(
            f"{split} validation failed with {len(errors)} error(s)."
        )

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate EcoPulse YOLO Dataset")
    parser.add_argument(
        "--root",
        default=str(YOLO_ROOT),
        help="Path to YOLO dataset root (containing data.yaml, images, labels)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    print(f"Validating dataset at: {root.resolve()}")
    print(f"Canonical classes: {EXPECTED_CLASS_MAPPING}")

    yaml_info = validate_data_yaml(root)
    print(f"data.yaml verified: {yaml_info['yaml_path']}")

    results = {}
    for split in SPLITS:
        results[split] = validate_split(split, root_dir=root)

    total_images = sum(r["images_count"] for r in results.values())
    total_boxes = sum(r["valid_boxes"] for r in results.values())
    print("\n" + "=" * 50)
    print(f"ALL SPLITS VALIDATED SUCCESSFULLY")
    print(f"Total Images: {total_images:,}")
    print(f"Total Bounding Boxes: {total_boxes:,}")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
