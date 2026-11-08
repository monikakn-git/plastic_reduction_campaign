"""Review a labeled test subset and YOLO predictions in FiftyOne."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import fiftyone as fo
import numpy as np
import supervision as sv
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.evaluate import EXPECTED_CLASSES

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _labels(path: Path, width: int, height: int) -> list[fo.Detection]:
    output = []
    label_path = path
    if not label_path.is_file():
        return output
    for row in label_path.read_text(encoding="utf-8").splitlines():
        values = row.split()
        if len(values) != 5:
            continue
        class_id, xc, yc, box_width, box_height = map(float, values)
        class_id = int(class_id)
        if class_id not in EXPECTED_CLASSES:
            continue
        output.append(fo.Detection(
            label=EXPECTED_CLASSES[class_id],
            bounding_box=[xc - box_width / 2, yc - box_height / 2, box_width, box_height],
        ))
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect ground truth and YOLO predictions with FiftyOne")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "artifacts" / "datasets" / "processed")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    image_dir = args.dataset_root / "images" / args.split
    label_dir = args.dataset_root / "labels" / args.split
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)[:args.limit]
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")

    model = YOLO(str(args.checkpoint))
    dataset_name = args.name or f"ecopulse-{args.split}-{args.checkpoint.stem}-{datetime.now(timezone.utc):%Y%m%d%H%M%S}"
    if fo.dataset_exists(dataset_name):
        raise ValueError(f"FiftyOne dataset already exists: {dataset_name}; provide a new --name")
    dataset = fo.Dataset(dataset_name)
    preview_dir = PROJECT_ROOT / "reports" / "fiftyone_previews" / dataset_name
    preview_dir.mkdir(parents=True, exist_ok=True)
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    samples = []

    for image_path in images:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            image_array = np.array(image)
        result = model.predict(str(image_path), verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        labels = [
            f"{EXPECTED_CLASSES.get(int(class_id), class_id)} {float(confidence):.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]
        annotated = label_annotator.annotate(
            scene=box_annotator.annotate(scene=image_array.copy(), detections=detections),
            detections=detections,
            labels=labels,
        )
        preview = preview_dir / f"{image_path.stem}_prediction.jpg"
        Image.fromarray(annotated).save(preview, quality=90)

        sample = fo.Sample(filepath=str(image_path))
        sample["ground_truth"] = fo.Detections(detections=_labels(label_dir / f"{image_path.stem}.txt", width, height))
        predicted = []
        for class_id, confidence, (x1, y1, x2, y2) in zip(detections.class_id, detections.confidence, detections.xyxy):
            predicted.append(fo.Detection(
                label=EXPECTED_CLASSES.get(int(class_id), f"class_{class_id}"),
                confidence=float(confidence),
                bounding_box=[float(x1) / width, float(y1) / height, float(x2 - x1) / width, float(y2 - y1) / height],
            ))
        sample["predictions"] = fo.Detections(detections=predicted)
        sample["annotated_preview"] = str(preview)
        samples.append(sample)

    dataset.add_samples(samples)
    session = fo.launch_app(dataset)
    print(f"FiftyOne dataset: {dataset.name} ({len(dataset)} samples)")
    print("Close the FiftyOne App session when review is complete.")
    session.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
