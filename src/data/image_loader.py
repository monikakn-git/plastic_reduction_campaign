
from pathlib import Path
import json
import zipfile
from functools import lru_cache

DATA_ROOT = Path(r"G:\My Drive\datasets\Ai_Waste_Campaign")
IMAGES_ZIP = DATA_ROOT / "images.zip"
ANNOTATIONS_ZIP = DATA_ROOT / "annotations.zip"
ANNOTATION_DIR = "annotations/4_class_splits"


def load_split(split: str) -> dict:
    """Load COCO annotations directly from the annotation ZIP."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split}")

    member = f"{ANNOTATION_DIR}/annotations_{split}.json"

    with zipfile.ZipFile(ANNOTATIONS_ZIP, "r") as archive:
        with archive.open(member) as file:
            return json.load(file)


def build_image_index() -> dict[str, str]:
    """Index image ZIP members by filename, without extracting images."""
    if not IMAGES_ZIP.is_file():
        raise FileNotFoundError(f"Image archive not found: {IMAGES_ZIP}")

    index = {}

    with zipfile.ZipFile(IMAGES_ZIP, "r") as archive:
        for member in archive.namelist():
            if member.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                filename = Path(member).name.lower()
                # Preserve the first match; duplicate basenames are handled below.
                index.setdefault(filename, []).append(member)

    # Retain only unique basename matches to avoid silently choosing the wrong image.
    return {
        filename: members[0]
        for filename, members in index.items()
        if len(members) == 1
    }


def resolve_image_member(image: dict, image_index: dict[str, str]) -> str | None:
    """Resolve an annotation image to a unique member in images.zip."""
    filename = image.get("file_name")

    if not filename:
        path = image.get("path", "")
        filename = Path(path).name

    if not filename:
        return None

    return image_index.get(Path(filename).name.lower())


@lru_cache(maxsize=32)
def _read_image_bytes_cached(member: str) -> bytes:
    """Read one image from the ZIP; cache a small number of accessed images."""
    with zipfile.ZipFile(IMAGES_ZIP, "r") as archive:
        return archive.read(member)


def read_image_bytes(member: str) -> bytes:
    """Read image bytes without extracting the image to disk."""
    return _read_image_bytes_cached(member)


def smoke_test(split: str = "train", limit: int = 20) -> None:
    """Check image mapping on a small sample before training."""
    data = load_split(split)
    image_index = build_image_index()

    images = data.get("images", [])
    matched = []
    missing = []

    for image in images[:limit]:
        member = resolve_image_member(image, image_index)

        if member:
            matched.append((image.get("id"), member))
        else:
            missing.append({
                "image_id": image.get("id"),
                "file_name": image.get("file_name"),
                "path": image.get("path"),
            })

    print(f"Split: {split}")
    print(f"Images checked: {min(limit, len(images))}")
    print(f"Matched: {len(matched)}")
    print(f"Missing/ambiguous: {len(missing)}")

    for image_id, member in matched[:5]:
        print(f"  {image_id}: {member}")

    for item in missing[:5]:
        print(f"  UNMATCHED: {item}")

    if missing:
        print(
            "\nSome filenames did not resolve uniquely. "
            "Inspect the paths before training."
        )
    else:
        print("\nSmoke test passed for this sample.")


if __name__ == "__main__":
    smoke_test()
