from pathlib import Path
import json
import os
import shutil
import zipfile

DATA_ROOT = Path(
    os.environ.get("WASTE_DATA_DIR", r"D:\datasets\Ai_Waste_Campaign")
)

ANNOTATIONS_ZIP = Path(
    os.environ.get("WASTE_ANNOTATIONS_ZIP", str(DATA_ROOT / "annotations.zip"))
)
IMAGES_DIR = Path(
    os.environ.get("WASTE_IMAGES_DIR", str(DATA_ROOT / "extracted" / "images"))
)
YOLO_ROOT = Path(
    os.environ.get("WASTE_YOLO_DIR", str(DATA_ROOT / "yolo_dataset"))
)
ANNOTATION_DIR = os.environ.get("ANNOTATION_DIR", "annotations/10_class_splits")

SPLITS = ("train", "val", "test")

CANONICAL_10_CLASSES = {
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


def load_split(archive: zipfile.ZipFile, split: str) -> dict:
    member = f"{ANNOTATION_DIR}/annotations_{split}.json"
    if member not in archive.namelist():
        # Fallback to 4-class if 10-class is not in archive
        fallback = f"annotations/4_class_splits/annotations_{split}.json"
        if fallback in archive.namelist():
            member = fallback
        else:
            raise FileNotFoundError(f"Neither {member} nor {fallback} found in archive.")

    with archive.open(member) as file:
        return json.load(file)


def create_image_link(source: Path, destination: Path) -> None:
    """Use a hard link to avoid copying the large image files."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        if destination.samefile(source):
            return
        raise FileExistsError(f"Destination already exists: {destination}")

    try:
        os.link(source, destination)
    except OSError:
        # Copy only if hard links are unavailable.
        shutil.copy2(source, destination)


def main() -> None:
    if not ANNOTATIONS_ZIP.is_file():
        raise FileNotFoundError(f"Missing annotation archive: {ANNOTATIONS_ZIP}")

    if not IMAGES_DIR.is_dir():
        raise FileNotFoundError(f"Missing extracted images: {IMAGES_DIR}")

    YOLO_ROOT.mkdir(parents=True, exist_ok=True)

    counts = {}
    seen_files = set()
    category_names = {}

    with zipfile.ZipFile(ANNOTATIONS_ZIP, "r") as archive:
        for split in SPLITS:
            data = load_split(archive, split)
            images = data["images"]
            annotations = data["annotations"]

            image_by_id = {image["id"]: image for image in images}
            labels_by_image = {image["id"]: [] for image in images}

            sorted_cats = sorted(data["categories"], key=lambda item: item["id"])
            categories = {
                category["id"]: index
                for index, category in enumerate(sorted_cats)
            }
            for cat in sorted_cats:
                category_names[categories[cat["id"]]] = cat["name"]

            for annotation in annotations:
                image_id = annotation.get("image_id")
                image = image_by_id.get(image_id)
                category_id = annotation.get("category_id")
                bbox = annotation.get("bbox", [])

                if image is None or category_id not in categories:
                    continue
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue

                try:
                    x, y, width, height = map(float, bbox)
                    image_width = float(image["width"])
                    image_height = float(image["height"])
                except (ValueError, TypeError, KeyError):
                    continue

                if (
                    image_width <= 0 or image_height <= 0
                    or width <= 0 or height <= 0
                    or x < 0 or y < 0
                    or x + width > image_width
                    or y + height > image_height
                ):
                    continue

                labels_by_image[image_id].append(
                    f"{categories[category_id]} "
                    f"{(x + width / 2) / image_width:.6f} "
                    f"{(y + height / 2) / image_height:.6f} "
                    f"{width / image_width:.6f} "
                    f"{height / image_height:.6f}"
                )

            image_split_dir = YOLO_ROOT / "images" / split
            label_split_dir = YOLO_ROOT / "labels" / split
            image_split_dir.mkdir(parents=True, exist_ok=True)
            label_split_dir.mkdir(parents=True, exist_ok=True)

            for image in images:
                filename = Path(image["file_name"]).name
                source = IMAGES_DIR / filename
                key = filename.casefold()

                if key in seen_files:
                    raise ValueError(
                        f"Duplicate image filename across splits: {filename}"
                    )
                seen_files.add(key)

                if not source.is_file():
                    raise FileNotFoundError(f"Image not found: {source}")

                create_image_link(source, image_split_dir / filename)

                label_path = label_split_dir / f"{Path(filename).stem}.txt"
                label_path.write_text(
                    "\n".join(labels_by_image[image["id"]]),
                    encoding="utf-8",
                )

            counts[split] = {
                "images": len(images),
                "annotations": len(annotations),
            }

            print(
                f"{split}: {len(images):,} images, "
                f"{len(annotations):,} annotations"
            )

    names_to_use = category_names if category_names else CANONICAL_10_CLASSES
    names_block = "\n".join(f"  {idx}: {name}" for idx, name in sorted(names_to_use.items()))

    yaml_path = YOLO_ROOT / "dataset.yaml"
    yaml_path.write_text(
        f"path: {YOLO_ROOT.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        f"names:\n{names_block}\n",
        encoding="utf-8",
    )

    report_path = YOLO_ROOT / "preparation_report.json"
    report_path.write_text(json.dumps(counts, indent=2), encoding="utf-8")

    print(f"\nDataset YAML: {yaml_path}")
    print(f"Preparation report: {report_path}")
    print("Dataset preparation complete.")


if __name__ == "__main__":
    main()
