"""Generate Evidently drift reports from reference and production observation CSVs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_COLUMNS = ["image_width", "image_height", "total_detections", "mean_confidence", "mean_box_area_ratio"] + [
    f"detections_{i}" for i in range(10)
]


def generate_report(reference_path: Path, current_path: Path, output_html: Path, output_json: Path) -> dict:
    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)
    required = {"model", *FEATURE_COLUMNS}
    for name, frame in (("reference", reference), ("current", current)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} observations are missing columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"{name} observation file has no rows")
    # Keep only stable, privacy-safe summary fields. Timestamps and identifiers
    # are excluded so the report measures output/input-feature distribution shift.
    reference = reference[["model", *FEATURE_COLUMNS]]
    current = current[["model", *FEATURE_COLUMNS]]
    snapshot = Report([DataDriftPreset()]).run(current_data=current, reference_data=reference)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(output_html))
    payload = json.loads(snapshot.json())
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare production inference summaries with a reference window")
    parser.add_argument("--reference", type=Path, required=True, help="Baseline/reference observation CSV")
    parser.add_argument("--current", type=Path, required=True, help="Recent production observation CSV")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "production_drift.html")
    parser.add_argument("--json", type=Path, default=PROJECT_ROOT / "reports" / "production_drift.json")
    args = parser.parse_args()
    generate_report(args.reference, args.current, args.output, args.json)
    print(f"Evidently drift report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
