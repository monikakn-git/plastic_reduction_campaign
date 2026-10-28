
from pathlib import Path
import json
import math
import zipfile

DATA_ROOT = Path(r"G:\My Drive\datasets\Ai_Waste_Campaign")
ANNOTATIONS_ZIP = DATA_ROOT / "annotations.zip"
ANNOTATION_DIR = "annotations/4_class_splits"

OUTPUT_DIR = Path("data/processed/yolo_labels")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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
    except (TypeError, ValueError):
        return False

    image_width = image.get("width")
    image_height = image.get("height")

    if not all(math.isfinite(v) for v in (x, y, width, height)):
        return False

    if not isinstance(image_width, (int, float)):
        return False
    if not isinstance(image_height, (int, float)):
        return False

    return (
        width > 0
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

    # COCO IDs 1-4 map to YOLO class IDs 0-3.
    category_to_yolo = {
        category["id"]: index
        for index, category in enumerate(data["categories"])
    }

    rejected = 0

    for annotation in annotations:
        image = image_by_id.get(annotation.get("image_id"))

        if image is None or not valid_bbox(annotation, image):
            rejected += 1
            continue

        x, y, width, height = map(float, annotation["bbox"])
        image_width = float(image["width"])
        image_height = float(image["height"])

        x_center = (x + width / 2) / image_width
        y_center = (y + height / 2) / image_height
        norm_width = width / image_width
        norm_height = height / image_height

        class_id = category_to_yolo.get(annotation["category_id"])
        if class_id is None:
            rejected += 1
            continue

        labels_by_image[image["id"]].append(
            f"{class_id} {x_center:.6f} {y_center:.6f} "
            f"{norm_width:.6f} {norm_height:.6f}"
        )

    split_dir = OUTPUT_DIR / split
    split_dir.mkdir(parents=True, exist_ok=True)

    for image in images:
        # Image IDs avoid filename collisions in the output labels.
        label_path = split_dir / f"{image['id']}.txt"
        label_path.write_text(
            "\n".join(labels_by_image[image["id"]]),
            encoding="utf-8",
        )

    print(f"\n{split.upper()}")
    print(f"Images: {len(images):,}")
    print(f"Rejected annotations: {rejected}")
    print(f"Label files created: {len(images):,}")


def main():
    for split in ("train", "val", "test"):
        convert_split(split)

    print(f"\nYOLO labels saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
