from pathlib import Path
import os
from typing import Any, Optional

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CHECKPOINTS = {
    "yolo11n": PROJECT_ROOT / "artifacts" / "checkpoints" / "yolo11n_best.pt",
    "yolov8n": PROJECT_ROOT / "artifacts" / "checkpoints" / "yolov8n_best.pt",
}

LOCAL_CHECKPOINT = DEFAULT_CHECKPOINTS["yolo11n"]
DRIVE_CHECKPOINT = (
    Path(r"G:\My Drive\datasets\Ai_Waste_Campaign")
    / "yolo_runs"
    / "yolo11n_waste_detection"
    / "weights"
    / "best.pt"
)

CLASS_NAMES = {
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

_model_cache: dict[str, YOLO] = {}


def get_checkpoint_path(model_key: str = "yolo11n") -> Path:
    """Resolve the active model checkpoint path with fallback.

    Args:
        model_key: Checkpoint key ('yolo11n' or 'yolov8n') or path string.
    """
    env_path = os.environ.get("WASTE_MODEL_PATH")
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return p

    key = model_key.lower().replace("-", "").replace("_", "")
    for k, p in DEFAULT_CHECKPOINTS.items():
        if k in key and p.is_file():
            return p

    # Fallback to local default checkpoint
    if LOCAL_CHECKPOINT.is_file():
        return LOCAL_CHECKPOINT

    # Fallback to YOLOv8 if YOLO11 is absent
    if DEFAULT_CHECKPOINTS["yolov8n"].is_file():
        return DEFAULT_CHECKPOINTS["yolov8n"]

    if DRIVE_CHECKPOINT.is_file():
        return DRIVE_CHECKPOINT

    return LOCAL_CHECKPOINT


def detect_model_name(checkpoint_path: str | Path) -> str:
    """Infer model family name from checkpoint path or filename."""
    path_str = str(checkpoint_path).lower()
    if "yolo11" in path_str:
        return "YOLO11n"
    if "yolov8" in path_str:
        return "YOLOv8n"
    return Path(checkpoint_path).stem


def load_model(checkpoint_path: str | Path | None = None) -> YOLO:
    """Load and cache the YOLO model checkpoint."""
    global _model_cache

    target_path = Path(checkpoint_path) if checkpoint_path else get_checkpoint_path()
    if not target_path.is_absolute():
        target_path = PROJECT_ROOT / target_path

    target_key = str(target_path.resolve())

    if target_key in _model_cache:
        return _model_cache[target_key]

    if not target_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {target_path}. "
            f"Ensure {LOCAL_CHECKPOINT} exists or set WASTE_MODEL_PATH."
        )

    print(f"Loading trained model: {target_path}")
    model = YOLO(str(target_path))
    _model_cache[target_key] = model

    return model


def predict_image(
    image_path: str | Path,
    confidence: float = 0.25,
    checkpoint_path: str | Path | None = None,
    save: bool = False,
    save_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run waste detection on a single image file.

    Args:
        image_path: Path to the image file.
        confidence: Minimum confidence threshold (0.0 to 1.0).
        checkpoint_path: Optional custom model checkpoint path.
        save: Whether to save predicted visualization to disk.
        save_dir: Output directory if saving visualization.
    """
    image_path = Path(image_path)
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = load_model(checkpoint_path)
    active_path = str(Path(checkpoint_path) if checkpoint_path else get_checkpoint_path())
    model_name = detect_model_name(active_path)

    predict_kwargs: dict[str, Any] = {
        "source": str(image_path),
        "conf": confidence,
        "verbose": False,
    }

    if save:
        predict_kwargs["save"] = True
        if save_dir:
            out_p = Path(save_dir)
            out_p.mkdir(parents=True, exist_ok=True)
            predict_kwargs["project"] = str(out_p.parent)
            predict_kwargs["name"] = out_p.name
            predict_kwargs["exist_ok"] = True

    results = model.predict(**predict_kwargs)

    predictions = []
    saved_image_path = None

    for result in results:
        if save and hasattr(result, "save_dir") and result.save_dir:
            cand = Path(result.save_dir) / image_path.name
            if cand.is_file():
                saved_image_path = str(cand)

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names.get(
                class_id, CLASS_NAMES.get(class_id, f"class_{class_id}")
            )

            predictions.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(float(box.conf[0]), 4),
                "bbox": [
                    round(float(value), 2)
                    for value in box.xyxy[0].tolist()
                ],
            })

    output: dict[str, Any] = {
        "image": image_path.name,
        "image_path": str(image_path),
        "model": model_name,
        "checkpoint": active_path,
        "total_detections": len(predictions),
        "predictions": predictions,
    }

    if saved_image_path:
        output["saved_image_path"] = saved_image_path

    return output


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="EcoPulse AI Single Image Waste Detection")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold (0-1)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Custom checkpoint path")
    parser.add_argument("--save", action="store_true", help="Save annotated prediction image")

    args = parser.parse_args()

    out = predict_image(args.image, args.confidence, args.checkpoint, save=args.save)
    print(json.dumps(out, indent=2))
