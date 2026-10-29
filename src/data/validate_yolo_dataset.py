
from pathlib import Path
import os

DATA_ROOT = Path(
    os.environ.get("WASTE_DATA_DIR", r"D:\datasets\Ai_Waste_Campaign")
)
YOLO_ROOT = DATA_ROOT / "yolo_dataset"
SPLITS = ("train", "val", "test")
VALID_CLASSES = {0, 1, 2, 3}


def validate_split(split):
    images_dir = YOLO_ROOT / "images" / split
    labels_dir = YOLO_ROOT / "labels" / split

    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing image or label directory for {split}")

    images = [
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".bmp", ".webp"
        }
    ]

    errors = []
    empty_labels = 0
    box_count = 0

    for image in images:
        label = labels_dir / f"{image.stem}.txt"

        if not label.is_file():
            errors.append(f"Missing label file: {label.name}")
            continue

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

    print(f"\n{split.upper()}")
    print(f"Images found: {len(images):,}")
    print(f"Empty label files: {empty_labels:,}")
    print(f"Valid boxes counted: {box_count:,}")
    print(f"Errors: {len(errors):,}")

    for error in errors[:20]:
        print(" -", error)

    if errors:
        raise RuntimeError(
            f"{split} validation failed with {len(errors)} error(s)."
        )


def main():
    for split in SPLITS:
        validate_split(split)

    print("\nDataset validation passed.")


if __name__ == "__main__":
    main()
