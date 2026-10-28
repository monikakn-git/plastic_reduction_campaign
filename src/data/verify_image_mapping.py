
from pathlib import Path
import json
import zipfile

DATA_ROOT = Path(r"G:\My Drive\datasets\Ai_Waste_Campaign")
IMAGES_ZIP = DATA_ROOT / "images.zip"
ANNOTATIONS_ZIP = DATA_ROOT / "annotations.zip"

SPLITS = ("train", "val", "test")
ANNOTATION_DIR = "annotations/4_class_splits"


def load_split(archive, split):
    member = f"{ANNOTATION_DIR}/annotations_{split}.json"
    with archive.open(member) as file:
        return json.load(file)


def main():
    if not IMAGES_ZIP.is_file() or not ANNOTATIONS_ZIP.is_file():
        raise FileNotFoundError("Check your Google Drive archive paths.")

    # Index archive members by basename. Preserve duplicates for detection.
    with zipfile.ZipFile(IMAGES_ZIP) as images_archive, \
         zipfile.ZipFile(ANNOTATIONS_ZIP) as annotations_archive:

        image_index = {}

        for member in images_archive.namelist():
            if member.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                image_index.setdefault(
                    Path(member).name.lower(), []
                ).append(member)

        total_images = 0
        total_matched = 0
        missing = []
        ambiguous = []

        for split in SPLITS:
            data = load_split(annotations_archive, split)
            split_matched = 0

            for image in data.get("images", []):
                total_images += 1
                filename = Path(
                    image.get("file_name") or image.get("path", "")
                ).name.lower()

                candidates = image_index.get(filename, [])

                if len(candidates) == 1:
                    split_matched += 1
                    total_matched += 1
                elif len(candidates) == 0:
                    missing.append({
                        "split": split,
                        "image_id": image.get("id"),
                        "filename": filename,
                    })
                else:
                    ambiguous.append({
                        "split": split,
                        "image_id": image.get("id"),
                        "filename": filename,
                        "matches": candidates,
                    })

            print(
                f"{split.upper()}: {len(data['images']):,} images, "
                f"{split_matched:,} uniquely matched"
            )

        print("\nFULL DATASET MAPPING")
        print(f"Total images: {total_images:,}")
        print(f"Unique matches: {total_matched:,}")
        print(f"Missing: {len(missing):,}")
        print(f"Ambiguous: {len(ambiguous):,}")

        report = {
            "total_images": total_images,
            "unique_matches": total_matched,
            "missing": missing,
            "ambiguous": ambiguous,
        }

        output = Path("reports/image_mapping_report.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        print(f"Report saved to: {output.resolve()}")


if __name__ == "__main__":
    main()
