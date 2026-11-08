"""
EcoPulse AI - Standalone Model Testing Suite
=============================================
Independent verification and stress testing of YOLO checkpoints:
- Loading verification
- Output schema and coordinate validation
- Latency & throughput (FPS) benchmarking
- Robustness on edge cases (blank images, extreme aspect ratios, corrupt inputs)
- Structured JSON test report generation
"""

from pathlib import Path
import argparse
import json
import os
import sys
import time
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINTS = {
    "yolo11n": PROJECT_ROOT / "artifacts" / "checkpoints" / "yolo11n_best.pt",
    "yolov8n": PROJECT_ROOT / "artifacts" / "checkpoints" / "yolov8n_best.pt",
}
TEST_IMAGES_DIR = PROJECT_ROOT / "artifacts" / "datasets" / "processed" / "images" / "test"
REPORTS_DIR = PROJECT_ROOT / "reports"

CANONICAL_CLASSES = {
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


def test_checkpoint_loading(model_path: Path) -> dict[str, Any]:
    """Verify that a checkpoint loads cleanly and exposes valid metadata."""
    from ultralytics import YOLO

    if not model_path.is_file():
        return {
            "name": model_path.name,
            "status": "FAIL",
            "error": f"Checkpoint file not found: {model_path}",
        }

    size_mb = round(model_path.stat().st_size / (1024 * 1024), 2)
    start = time.perf_counter()
    try:
        model = YOLO(str(model_path))
        load_time_ms = round((time.perf_counter() - start) * 1000, 2)
        names = model.names
        is_10_class = len(names) == 10

        return {
            "name": model_path.name,
            "path": str(model_path),
            "size_mb": size_mb,
            "load_time_ms": load_time_ms,
            "class_count": len(names),
            "classes": names,
            "has_10_classes": is_10_class,
            "status": "PASS" if is_10_class else "WARNING",
        }
    except Exception as exc:
        return {
            "name": model_path.name,
            "path": str(model_path),
            "size_mb": size_mb,
            "status": "FAIL",
            "error": str(exc),
        }


def test_inference_correctness(model_path: Path, sample_image: Path) -> dict[str, Any]:
    """Verify inference output format, coordinate bounds, and confidence values."""
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    res = model.predict(source=str(sample_image), conf=0.25, verbose=False)[0]

    detections = []
    coord_errors = []
    conf_errors = []
    cls_errors = []

    for box in res.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        coords = [round(float(v), 2) for v in box.xyxy[0].tolist()]
        x1, y1, x2, y2 = coords

        if not (0 <= conf <= 1.0):
            conf_errors.append(f"Confidence out of range [0, 1]: {conf}")
        if cls_id not in CANONICAL_CLASSES:
            cls_errors.append(f"Unknown class id: {cls_id}")
        if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0:
            coord_errors.append(f"Invalid bounding box geometry: {coords}")

        detections.append({
            "class_id": cls_id,
            "class_name": CANONICAL_CLASSES.get(cls_id, f"class_{cls_id}"),
            "confidence": round(conf, 4),
            "bbox": coords,
        })

    all_valid = not (coord_errors or conf_errors or cls_errors)
    return {
        "sample_image": sample_image.name,
        "total_detections": len(detections),
        "detections": detections,
        "valid_coordinates": len(coord_errors) == 0,
        "valid_confidences": len(conf_errors) == 0,
        "valid_classes": len(cls_errors) == 0,
        "errors": coord_errors + conf_errors + cls_errors,
        "status": "PASS" if all_valid else "FAIL",
    }


def benchmark_model_speed(
    model_path: Path, sample_image: Path, iterations: int = 10, warmup: int = 3
) -> dict[str, Any]:
    """Benchmark inference latency and FPS."""
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    img_str = str(sample_image)

    for _ in range(warmup):
        model.predict(source=img_str, verbose=False)

    latencies_ms = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        model.predict(source=img_str, verbose=False)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    mean_ms = round(float(np.mean(latencies_ms)), 2)
    median_ms = round(float(np.median(latencies_ms)), 2)
    p95_ms = round(float(np.percentile(latencies_ms, 95)), 2)
    fps = round(1000.0 / mean_ms, 2) if mean_ms > 0 else 0

    return {
        "iterations": iterations,
        "mean_latency_ms": mean_ms,
        "median_latency_ms": median_ms,
        "p95_latency_ms": p95_ms,
        "fps": fps,
        "status": "PASS" if mean_ms <= 500.0 else "WARNING",
    }


def test_edge_cases(model_path: Path, temp_dir: Path) -> dict[str, Any]:
    """Test model behavior on unusual inputs (solid black image, extreme aspect ratio)."""
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    temp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Solid black image
    black_img_path = temp_dir / "edge_black.jpg"
    Image.fromarray(np.zeros((300, 300, 3), dtype=np.uint8)).save(black_img_path)

    # 2. Extreme aspect ratio (1000 x 50)
    panoramic_path = temp_dir / "edge_panoramic.jpg"
    Image.fromarray(np.ones((50, 1000, 3), dtype=np.uint8) * 128).save(panoramic_path)

    results = {}
    for name, path in [("solid_black", black_img_path), ("extreme_aspect", panoramic_path)]:
        try:
            preds = model.predict(source=str(path), verbose=False)[0]
            results[name] = {
                "handled": True,
                "detections": len(preds.boxes),
                "status": "PASS",
            }
        except Exception as exc:
            results[name] = {
                "handled": False,
                "error": str(exc),
                "status": "FAIL",
            }

    # 3. Non-existent file error handling
    missing_path = temp_dir / "does_not_exist.jpg"
    try:
        model.predict(source=str(missing_path), verbose=False)
        results["missing_file_handling"] = {"status": "FAIL", "message": "Expected error not raised"}
    except Exception:
        results["missing_file_handling"] = {"status": "PASS", "handled": True}

    return results


def run_full_model_tests(
    checkpoint_key: str = "all",
    iterations: int = 10,
    output_report: Path | None = None,
) -> dict[str, Any]:
    """Execute complete model testing suite across selected checkpoints."""
    targets: dict[str, Path] = {}
    if checkpoint_key == "all":
        targets = {k: v for k, v in DEFAULT_CHECKPOINTS.items() if v.is_file()}
    elif checkpoint_key in DEFAULT_CHECKPOINTS:
        targets = {checkpoint_key: DEFAULT_CHECKPOINTS[checkpoint_key]}
    else:
        p = Path(checkpoint_key)
        targets = {p.stem: p}

    test_imgs = list(TEST_IMAGES_DIR.glob("*.jpg"))
    if not test_imgs:
        raise FileNotFoundError(f"No test images found in {TEST_IMAGES_DIR}")
    sample_img = test_imgs[0]

    temp_edge_dir = PROJECT_ROOT / "reports" / "temp_edge_tests"

    print("=" * 65)
    print("ECOPULSE AI - MODEL TESTING SUITE")
    print("=" * 65)
    print(f"Sample test image: {sample_img.name}")
    print(f"Target checkpoints: {list(targets.keys())}")

    suite_report: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sample_image": str(sample_img),
        "checkpoints_tested": {},
        "overall_status": "PASS",
    }

    for name, path in targets.items():
        print(f"\n--- Testing Checkpoint: {name} ({path.name}) ---")

        # 1. Loading
        load_res = test_checkpoint_loading(path)
        print(f"  [1/4] Loading: {load_res['status']} ({load_res.get('size_mb', 0)} MB, {load_res.get('load_time_ms', 0)} ms)")

        # 2. Correctness
        correct_res = test_inference_correctness(path, sample_img)
        print(f"  [2/4] Inference Correctness: {correct_res['status']} ({correct_res['total_detections']} detections)")

        # 3. Speed
        speed_res = benchmark_model_speed(path, sample_img, iterations=iterations)
        print(f"  [3/4] Speed Benchmark: {speed_res['status']} ({speed_res['mean_latency_ms']} ms, {speed_res['fps']} FPS)")

        # 4. Edge cases
        edge_res = test_edge_cases(path, temp_edge_dir)
        edge_pass = all(v.get("status") == "PASS" for v in edge_res.values())
        print(f"  [4/4] Edge-Case Robustness: {'PASS' if edge_pass else 'FAIL'}")

        chk_status = "PASS" if (
            load_res["status"] == "PASS"
            and correct_res["status"] == "PASS"
            and edge_pass
        ) else "FAIL"

        if chk_status == "FAIL":
            suite_report["overall_status"] = "FAIL"

        suite_report["checkpoints_tested"][name] = {
            "checkpoint": str(path),
            "loading": load_res,
            "correctness": correct_res,
            "speed": speed_res,
            "edge_cases": edge_res,
            "status": chk_status,
        }

    # Cleanup temp edge dir
    if temp_edge_dir.exists():
        for f in temp_edge_dir.iterdir():
            f.unlink()
        temp_edge_dir.rmdir()

    report_path = output_report or (REPORTS_DIR / "model_test_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(suite_report, indent=2), encoding="utf-8")
    print("\n" + "=" * 65)
    print(f"Model Testing Complete. Overall Verdict: {suite_report['overall_status']}")
    print(f"Report saved to: {report_path}")
    print("=" * 65)

    return suite_report


def main() -> int:
    parser = argparse.ArgumentParser(description="EcoPulse AI Model Testing Suite")
    parser.add_argument(
        "--checkpoint",
        default="all",
        help="Checkpoint key ('yolo11n', 'yolov8n', 'all') or path to .pt file",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Benchmark iterations for speed testing",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path for saving JSON test report",
    )
    args = parser.parse_args()

    out_p = Path(args.output) if args.output else None
    report = run_full_model_tests(
        checkpoint_key=args.checkpoint,
        iterations=args.iterations,
        output_report=out_p,
    )
    return 0 if report["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
