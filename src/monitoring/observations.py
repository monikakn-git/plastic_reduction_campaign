"""Privacy-conscious inference summaries used by offline drift reports."""
from __future__ import annotations

import csv
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OBSERVATION_PATH = Path(os.environ.get("ECOPULSE_OBSERVATIONS_PATH", PROJECT_ROOT / "monitoring" / "inference_observations.csv"))
_LOCK = threading.Lock()
FIELDS = [
    "timestamp_utc", "model", "model_version", "image_width", "image_height",
    "total_detections", "mean_confidence", "mean_box_area_ratio",
] + [f"detections_{i}" for i in range(10)]


def summarize(result: dict[str, Any], model: str, width: int, height: int, version: int | None = None) -> dict[str, Any]:
    detections = result.get("predictions", [])
    confidences = [float(d["confidence"]) for d in detections]
    areas = []
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        areas.append(max(0.0, x2 - x1) * max(0.0, y2 - y1) / max(1, width * height))
    record: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "model_version": version or 0,
        "image_width": width,
        "image_height": height,
        "total_detections": len(detections),
        "mean_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        "mean_box_area_ratio": sum(areas) / len(areas) if areas else 0.0,
    }
    record.update({f"detections_{i}": 0 for i in range(10)})
    for detection in detections:
        class_id = int(detection["class_id"])
        if 0 <= class_id < 10:
            record[f"detections_{class_id}"] += 1
    return record


def log_observation(result: dict[str, Any], model: str, width: int, height: int, registry_info: dict[str, Any] | None) -> None:
    record = summarize(result, model, width, height, registry_info.get("version") if registry_info else None)
    OBSERVATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        needs_header = not OBSERVATION_PATH.exists() or OBSERVATION_PATH.stat().st_size == 0
        with OBSERVATION_PATH.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            if needs_header:
                writer.writeheader()
            writer.writerow(record)
