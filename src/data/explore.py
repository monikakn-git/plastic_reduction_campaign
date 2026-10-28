
from pathlib import Path
import json
import zipfile
from collections import Counter

DATA_ROOT = Path(r"G:\My Drive\datasets\Ai_Waste_Campaign")
ANNOTATIONS_ZIP = DATA_ROOT / "annotations.zip"
SPLIT_DIR = "annotations/4_class_splits"
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_split(split: str) -> dict:
    member = f"{SPLIT_DIR}/annotations_{split}.json"

    with zipfile.ZipFile(ANNOTATIONS_ZIP, "r") as archive:
        with archive.open(member) as file:
            return json.load(file)


def analyze_split(split: str) -> dict:
    data = load_split(split)
    images = data["images"]
    annotations = data["annotations"]
    categories = {
        category["id"]: category["name"]
        for category in data["categories"]
    }

    image_ids = {image["id"] for image in images}
    annotations_per_image = Counter(
        annotation["image_id"] for annotation in annotations
    )
    class_counts = Counter(
        annotation["category_id"] for annotation in annotations
    )

    widths = [image["width"] for image in images]
    heights = [image["height"] for image in images]

    # Count images by their annotated status.
    annotated_status = Counter(
        image.get("annotated", "unknown") for image in images
    )

    # Images without annotations are not necessarily true negative examples.
    zero_annotation_images = sum(
        image_id not in annotations_per_image for image_id in image_ids
    )

    report = {
        "split": split,
        "image_count": len(images),
        "annotation_count": len(annotations),
        "average_annotations_per_image": (
            round(len(annotations) / len(images), 3) if images else 0
        ),
        "zero_annotation_images": zero_annotation_images,
        "annotated_status": dict(annotated_status),
        "image_dimensions": {
            "min_width": min(widths, default=0),
            "max_width": max(widths, default=0),
            "min_height": min(heights, default=0),
            "max_height": max(heights, default=0),
        },
        "class_distribution": {
            categories.get(class_id, f"Unknown ({class_id})"): count
            for class_id, count in sorted(class_counts.items())
        },
    }

    print(f"\n--- {split.upper()} ---")
    print(f"Images: {report['image_count']:,}")
    print(f"Annotations: {report['annotation_count']:,}")
    print(
        "Average annotations/image:",
        report["average_annotations_per_image"],
    )
    print("Images with zero annotations:", zero_annotation_images)
    print("Annotated status:", dict(annotated_status))
    print("Dimensions:", report["image_dimensions"])
    print("Class distribution:", report["class_distribution"])

    return report


def main():
    reports = [
        analyze_split(split)
        for split in ("train", "val", "test")
    ]

    output_path = REPORT_DIR / "dataset_exploration.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(reports, file, indent=2, ensure_ascii=False)

    print(f"\nExploration report saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
