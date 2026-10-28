
from pathlib import Path
import json
import math
import zipfile

# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_ROOT = Path(
    r"G:\My Drive\datasets\Ai_Waste_Campaign"
)

ANNOTATIONS_ZIP = DATA_ROOT / "annotations.zip"
SPLIT_DIRECTORY = "annotations/4_class_splits"

EXPECTED_IMAGES = {
    "train": 12111,
    "val": 1514,
    "test": 1513,
}

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load annotation JSON directly from ZIP
# --------------------------------------------------

def load_annotations(split: str) -> dict:
    """Load a COCO-style annotation split without extracting the ZIP."""

    if split not in EXPECTED_IMAGES:
        raise ValueError(f"Unsupported split: {split}")

    if not ANNOTATIONS_ZIP.is_file():
        raise FileNotFoundError(
            f"Annotation archive not found: {ANNOTATIONS_ZIP}"
        )

    member = (
        f"{SPLIT_DIRECTORY}/annotations_{split}.json"
    )

    with zipfile.ZipFile(ANNOTATIONS_ZIP, "r") as archive:
        with archive.open(member) as file:
            return json.load(file)


# --------------------------------------------------
# Validate individual bounding boxes
# --------------------------------------------------

def validate_bbox(annotation: dict, image_by_id: dict) -> list[str]:
    """Return validation errors for a single annotation."""

    errors = []
    bbox = annotation.get("bbox")
    image = image_by_id.get(annotation.get("image_id"))

    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return ["bbox must contain four values"]

    try:
        x, y, width, height = map(float, bbox)
    except (TypeError, ValueError):
        return ["bbox contains non-numeric values"]

    if not all(math.isfinite(v) for v in (x, y, width, height)):
        return ["bbox contains non-finite values"]

    if width <= 0 or height <= 0:
        errors.append("width and height must be positive")

    if image is not None:
        image_width = image.get("width")
        image_height = image.get("height")

        if (
            isinstance(image_width, (int, float))
            and isinstance(image_height, (int, float))
            and width > 0
            and height > 0
        ):
            if x < 0 or y < 0:
                errors.append("bbox starts outside image bounds")

            if x + width > image_width or y + height > image_height:
                errors.append("bbox extends beyond image bounds")

    return errors


# --------------------------------------------------
# Validate one dataset split
# --------------------------------------------------

def validate_split(split: str) -> dict:
    data = load_annotations(split)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    image_by_id = {
        image["id"]: image
        for image in images
        if "id" in image
    }

    category_ids = {
        category["id"]
        for category in categories
        if "id" in category
    }

    image_ids = [image["id"] for image in images if "id" in image]
    duplicate_image_ids = len(image_ids) - len(set(image_ids))

    annotation_ids = [
        ann["id"] for ann in annotations if "id" in ann
    ]
    duplicate_annotation_ids = (
        len(annotation_ids) - len(set(annotation_ids))
    )

    missing_images = []
    unknown_categories = []
    invalid_boxes = []
    invalid_dimensions = []

    class_counts = {
        category["id"]: {
            "name": category.get("name", "Unknown"),
            "count": 0,
        }
        for category in categories
        if "id" in category
    }

    for image in images:
        width = image.get("width")
        height = image.get("height")

        if (
            not isinstance(width, (int, float))
            or not isinstance(height, (int, float))
            or not math.isfinite(width)
            or not math.isfinite(height)
            or width <= 0
            or height <= 0
        ):
            invalid_dimensions.append({
                "image_id": image.get("id"),
                "file_name": image.get("file_name"),
                "width": width,
                "height": height,
            })

    for annotation in annotations:
        image_id = annotation.get("image_id")
        category_id = annotation.get("category_id")

        if image_id not in image_by_id:
            missing_images.append({
                "annotation_id": annotation.get("id"),
                "image_id": image_id,
            })

        if category_id not in category_ids:
            unknown_categories.append({
                "annotation_id": annotation.get("id"),
                "category_id": category_id,
            })
        elif category_id in class_counts:
            class_counts[category_id]["count"] += 1

        errors = validate_bbox(annotation, image_by_id)

        if errors:
            invalid_boxes.append({
                "annotation_id": annotation.get("id"),
                "image_id": image_id,
                "bbox": annotation.get("bbox"),
                "errors": errors,
            })

    result = {
        "split": split,
        "image_count": len(images),
        "annotation_count": len(annotations),
        "category_count": len(categories),
        "expected_image_count": EXPECTED_IMAGES[split],
        "image_count_matches": len(images) == EXPECTED_IMAGES[split],
        "duplicate_image_ids": duplicate_image_ids,
        "duplicate_annotation_ids": duplicate_annotation_ids,
        "missing_image_references": missing_images,
        "unknown_category_references": unknown_categories,
        "invalid_dimensions": invalid_dimensions,
        "invalid_boxes": invalid_boxes,
        "class_counts": class_counts,
    }

    print(f"\n{'=' * 55}")
    print(f"{split.upper()} DATASET VALIDATION")
    print(f"{'=' * 55}")
    print(f"Images: {len(images):,}")
    print(f"Annotations: {len(annotations):,}")
    print(f"Categories: {len(categories)}")
    print(f"Expected image count matches: {result['image_count_matches']}")
    print(f"Duplicate image IDs: {duplicate_image_ids}")
    print(f"Duplicate annotation IDs: {duplicate_annotation_ids}")
    print(f"Missing image references: {len(missing_images)}")
    print(f"Unknown category references: {len(unknown_categories)}")
    print(f"Invalid image dimensions: {len(invalid_dimensions)}")
    print(f"Invalid bounding boxes: {len(invalid_boxes)}")

    print("\nClass distribution:")
    for category in categories:
        category_id = category.get("id")
        if category_id in class_counts:
            print(
                f"  {class_counts[category_id]['name']}: "
                f"{class_counts[category_id]['count']:,}"
            )

    return result


# --------------------------------------------------
# Main execution and JSON report
# --------------------------------------------------

def main():
    results = []

    for split in ("train", "val", "test"):
        results.append(validate_split(split))

    report_path = REPORT_DIR / "data_validation_report.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    total_images = sum(item["image_count"] for item in results)
    total_annotations = sum(
        item["annotation_count"] for item in results
    )
    total_invalid_boxes = sum(
        len(item["invalid_boxes"]) for item in results
    )

    print(f"\n{'=' * 55}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 55}")
    print(f"Total images: {total_images:,}")
    print(f"Total annotations: {total_annotations:,}")
    print(f"Total invalid bounding boxes: {total_invalid_boxes}")
    print(f"Report saved to: {report_path.resolve()}")


if __name__ == "__main__":
    main()
