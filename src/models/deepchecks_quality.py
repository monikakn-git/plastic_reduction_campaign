"""Run Deepchecks vision integrity checks against a YOLO object-detection split."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.evaluate import EXPECTED_CLASSES


def prepare_deepchecks_compatibility() -> None:
    """Bridge legacy Deepchecks aliases removed by current NumPy/scikit-learn."""
    import numpy as np

    # Deepchecks 0.19 references np.Inf, removed in NumPy 2.0.
    if not hasattr(np, "Inf"):
        np.Inf = np.inf

    # Deepchecks 0.19 asks sklearn for the old max_error scorer name. The
    # equivalent current scorer is neg_max_error (higher is better).
    try:
        from sklearn.metrics._scorer import _SCORERS
        if "max_error" not in _SCORERS and "neg_max_error" in _SCORERS:
            _SCORERS["max_error"] = _SCORERS["neg_max_error"]
    except ImportError:
        pass


def _batch_loader(images: list[Path], labels_dir: Path, batch_size: int):
    from deepchecks.vision.vision_data import BatchOutputFormat

    for start in range(0, len(images), batch_size):
        batch_images = []
        batch_labels = []
        for path in images[start : start + batch_size]:
            with Image.open(path) as img:
                image = img.convert("RGB")
                width, height = image.size
                batch_images.append(image.copy())
            rows = []
            label_path = labels_dir / f"{path.stem}.txt"
            if label_path.exists():
                for line in label_path.read_text(encoding="utf-8").splitlines():
                    values = line.split()
                    if len(values) != 5:
                        continue
                    class_id, x_center, y_center, box_width, box_height = map(float, values)
                    rows.append((
                        int(class_id),
                        (x_center - box_width / 2) * width,
                        (y_center - box_height / 2) * height,
                        box_width * width,
                        box_height * height,
                    ))
            batch_labels.append(rows)
        yield BatchOutputFormat(images=batch_images, labels=batch_labels)


def run_deepchecks_quality(images_dir: Path, labels_dir: Path, output_json: Path, output_html: Path, limit: int = 200) -> dict[str, Any]:
    prepare_deepchecks_compatibility()
    from deepchecks.vision import VisionData
    from deepchecks.vision.suites import data_integrity

    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    if not images:
        raise FileNotFoundError(f"No images found in {images_dir}")
    random.Random(42).shuffle(images)
    images = images[:limit]
    dataset = VisionData(
        batch_loader=_batch_loader(images, labels_dir, batch_size=8),
        task_type="object_detection",
        label_map=EXPECTED_CLASSES,
        dataset_name="held-out test",
        reshuffle_data=False,
    )
    result = data_integrity().run(train_dataset=dataset, run_single_dataset="Train", max_samples=len(images), with_display=False)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(result.to_json(), encoding="utf-8")
    result.save_as_html(str(output_html), as_widget=False)
    return {
        "passed": bool(result.passed()),
        "samples": len(images),
        "json_report": str(output_json),
        "html_report": str(output_html),
        "checks": len(result.results),
    }
