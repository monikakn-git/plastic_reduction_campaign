
from pathlib import Path
import zipfile
import shutil
import time

ARCHIVE = Path(r"D:\mlops\images.zip")
DATASET = Path(r"D:\mlops\artifacts\datasets\processed")

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

print("EcoPulse AI - Resume Dataset Extraction")
print("Reading archive index...", flush=True)

start = time.time()
extracted = 0
skipped = 0

with zipfile.ZipFile(ARCHIVE) as z:
    archive_images = {}

    for name in z.namelist():
        path = Path(name)
        if path.suffix.lower() in EXTENSIONS:
            archive_images[path.stem] = name

    print(f"Archive images: {len(archive_images)}", flush=True)

    for split in ("train", "val", "test"):
        label_dir = DATASET / "labels" / split
        image_dir = DATASET / "images" / split
        image_dir.mkdir(parents=True, exist_ok=True)

        labels = list(label_dir.glob("*.txt"))
        split_extracted = 0
        split_skipped = 0

        print(f"\nStarting {split}: {len(labels)} labels", flush=True)

        for index, label in enumerate(labels, start=1):
            stem = label.stem
            candidate = stem.split("_", 1)[1] if "_" in stem else stem

            archive_name = archive_images.get(candidate)

            if archive_name is None:
                raise FileNotFoundError(
                    f"No matching image for {label.name}"
                )

            destination = image_dir / (
                candidate + Path(archive_name).suffix.lower()
            )

            if destination.exists():
                skipped += 1
                split_skipped += 1
            else:
                with z.open(archive_name) as source:
                    with destination.open("wb") as target:
                        shutil.copyfileobj(source, target)

                extracted += 1
                split_extracted += 1

            if index % 100 == 0 or index == len(labels):
                print(
                    f"{split}: {index}/{len(labels)} processed | "
                    f"{split_extracted} new | "
                    f"{split_skipped} already present",
                    flush=True
                )

        print(
            f"Finished {split}: "
            f"{split_extracted} extracted, "
            f"{split_skipped} skipped",
            flush=True
        )

print("\nExtraction finished.")
print(f"New images: {extracted}")
print(f"Existing images skipped: {skipped}")
print(f"Elapsed minutes: {(time.time() - start) / 60:.1f}")
