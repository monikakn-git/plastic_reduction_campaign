
from pathlib import Path
import json
import math
import os
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path(
    os.environ.get("WASTE_DATA_DIR", r"D:\datasets\Ai_Waste_Campaign")
)

ANNOTATIONS_ZIP = DATA_ROOT / "annotations.zip"
ANNOTATION_DIR = "annotations/4_class_splits"

IMAGES_DIR = DATA_ROOT / "extracted" / "images"
YOLO_DIR = DATA_ROOT / "yolo_dataset"

SPLITS = ("train", "val", "test")

# Original COCO category IDs -> YOLO's zero-based class IDs.
CATEGORY_TO_YOLO = {
    1: 0,  # Plastic
    2: 1,  # Pile
    3: 2,  # Face mask
    4: 3,  # Trash bin
}


def load_split(split: str) -> dict:
    member = f"{ANNOTATION_DIR}/annotations_{split}.json"

    with zipfile.ZipFile(ANNOTATIONS_ZIP, "r") as archive:
        with archive.open(member) as file:
            return json.load(file)


def valid_bbox(annotation: dict, image: dict) -> bool:
    bbox = annotation.get("bbox", [])

    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False

    try:
        x, y, width, height = map(float, bbox)
        image_width = float(image["width"])
        image_height = float(image["height"])
    except (KeyError, TypeError, ValueError):
        return False

    values = (x, y, width, height, image_width, image_height)

    if not all(math.isfinite(v) for v in values):
        return False

    return (
        image_width > 0
        and image_height > 0
        and width > 0
        and height > 0
        and x >= 0
        and y >= 0
        and x + width <= image_width
        and y + height <= image_height
    )


def convert_split(split: str) -> None:
    data = load_split(split)
    images = data["images"]
    annotations = data["annotations"]

    image_by_id = {image["id"]: image for image in images}
    labels_by_image = {image["id"]: [] for image in images}

    category_ids = {category["id"] for category in data["categories"]}
    if not category_ids.issubset(CATEGORY_TO_YOLO):
        raise ValueError(f"Unexpected category IDs in {split}: {category_ids}")

    rejected = 0

    for annotation in annotations:
        image = image_by_id.get(annotation.get("image_id"))
        category_id = annotation.get("category_id")

        if (
            image is None
            or category_id not in CATEGORY_TO_YOLO
            or not valid_bbox(annotation, image)
        ):
            rejected += 1
            continue

        x, y, width, height = map(float, annotation["bbox"])
        image_width = float(image["width"])
        image_height = float(image["height"])

        x_center = (x + width / 2) / image_width
        y_center = (y + height / 2) / image_height
        norm_width = width / image_width
        norm_height = height / image_height

        class_id = CATEGORY_TO_YOLO[category_id]

        labels_by_image[image["id"]].append(
            f"{class_id} {x_center:.6f} {y_center:.6f} "
            f"{norm_width:.6f} {norm_height:.6f}"
        )

    labels_dir = YOLO_DIR / "labels" / split
    labels_dir.mkdir(parents=True, exist_ok=True)

    missing_images = 0
    created = 0

    for image in images:
        filename = Path(image["file_name"]).name
        image_path = IMAGES_DIR / filename

        if not image_path.is_file():
            missing_images += 1
            continue

        # YOLO matches labels to images by filename stem.
        label_path = labels_dir / f"{Path(filename).stem}.txt"

        label_path.write_text(
            "\n".join(labels_by_image[image["id"]]),
            encoding="utf-8",
        )
        created += 1

    print(f"\n{split.upper()}")
    print(f"Images listed: {len(images):,}")
    print(f"Label files created: {created:,}")
    print(f"Missing images: {missing_images:,}")
    print(f"Rejected annotations: {rejected:,}")

    if missing_images:
        raise RuntimeError(
            f"{missing_images} images are missing from {IMAGES_DIR}. "
            "Check the extraction path before training."
        )


def write_dataset_yaml() -> None:
    # Use absolute paths so the YAML works regardless of the current directory.
    yaml_path = YOLO_DIR / "dataset.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    content = f"""path: {YOLO_DIR.resolve().as_posix()}
train: {IMAGES_DIR.resolve().as_posix()}
val: {IMAGES_DIR.resolve().as_posix()}
test: {IMAGES_DIR.resolve().as_posix()}

names:
  0: Plastic
  1: Pile
  2: Face mask
  3: Trash bin
"""

    # This shared-image layout is temporary; Ultralytics needs distinct
    # split image paths for a reliable train/val/test configuration.
    # The YAML is written here for inspection, not yet for training.
    yaml_path.write_text(content, encoding="utf-8")
    print(f"\nDataset YAML written to: {yaml_path}")


def main() -> None:
    if not ANNOTATIONS_ZIP.is_file():
        raise FileNotFoundError(f"Annotations ZIP not found: {ANNOTATIONS_ZIP}")

    if not IMAGES_DIR.is_dir():
        raise FileNotFoundError(f"Images directory not found: {IMAGES_DIR}")

    for split in SPLITS:
        convert_split(split)

    print(f"\nLabels saved under: {YOLO_DIR / 'labels'}")


if __name__ == "__main__":
    main()
