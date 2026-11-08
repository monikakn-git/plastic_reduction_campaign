"""Create a reference inference-observation CSV from the held-out YOLO split."""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predict import predict_image
from src.monitoring.observations import FIELDS, summarize


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a drift baseline from held-out images")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--images", type=Path, default=PROJECT_ROOT / "artifacts" / "datasets" / "processed" / "images" / "test")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--model-name", choices=("yolo11n", "yolov8n"), default=None)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "baseline_inference.csv")
    args = parser.parse_args()
    images = sorted(p for p in args.images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    if not images:
        raise FileNotFoundError(f"No images found in {args.images}")
    model_name = args.model_name or ("yolo11n" if "yolo11" in args.checkpoint.name.lower() else "yolov8n")
    random.Random(42).shuffle(images)
    records = []
    for image_path in images[:args.limit]:
        with Image.open(image_path) as image:
            width, height = image.size
        result = predict_image(image_path, checkpoint_path=args.checkpoint)
        records.append(summarize(result, model_name, width, height))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} baseline observations: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
