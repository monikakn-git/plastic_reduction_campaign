import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from src.models.predict import DEFAULT_CHECKPOINTS, get_checkpoint_path, load_model, predict_image
from src.tracking.mlflow_utils import log_prediction


app = FastAPI(
    title="EcoPulse AI Waste Detection API",
    description="Waste detection using trained YOLO11n and YOLOv8n models",
    version="1.0.0",
)


@app.get("/")
def home():
    return {
        "project": "EcoPulse AI",
        "message": "EcoPulse AI Waste Detection API is running",
        "available_models": ["yolo11n", "yolov8n"],
        "default_model": "yolo11n",
    }


@app.get("/health")
def health():
    try:
        active_checkpoint = get_checkpoint_path()
        model = load_model(active_checkpoint)

        return {
            "status": "healthy",
            "model_loaded": True,
            "active_checkpoint": str(active_checkpoint),
            "classes": model.names,
            "available_checkpoints": {
                k: str(v) for k, v in DEFAULT_CHECKPOINTS.items() if v.is_file()
            },
        }

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        )


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    confidence: float = 0.25,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    site_id: Optional[str] = None,
    model_name: str = Query(
        default="yolo11n",
        description="Model family to run ('yolo11n' or 'yolov8n')",
    ),
):
    # Validate confidence
    if not 0.0 <= confidence <= 1.0:
        raise HTTPException(
            status_code=400,
            detail="Confidence must be between 0 and 1.",
        )

    # Validate image extension
    allowed_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    extension = Path(file.filename or "").suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format. Allowed: .jpg, .jpeg, .png, .bmp, .webp",
        )

    # Validate GPS coordinates
    if (latitude is None) != (longitude is None):
        raise HTTPException(
            status_code=400,
            detail="Provide both latitude and longitude, or neither.",
        )

    if latitude is not None:
        if not (
            -90 <= latitude <= 90
            and -180 <= longitude <= 180
        ):
            raise HTTPException(
                status_code=400,
                detail="Invalid GPS coordinates.",
            )

    checkpoint_path = get_checkpoint_path(model_name)
    if not checkpoint_path.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"Requested model checkpoint '{model_name}' not found at {checkpoint_path}.",
        )

    temporary_path = None

    try:
        # Read uploaded image
        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty.",
            )

        # Save temporary image
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
        ) as temporary_file:
            temporary_file.write(contents)
            temporary_path = temporary_file.name

        # Run inference
        result = predict_image(
            temporary_path,
            confidence=confidence,
            checkpoint_path=checkpoint_path,
        )

        result["uploaded_filename"] = file.filename

        # Log prediction to MLflow
        log_prediction(
            result,
            latitude=latitude,
            longitude=longitude,
            site_id=site_id,
        )

        return result

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    finally:
        # Remove temporary uploaded image
        if temporary_path and os.path.exists(temporary_path):
            try:
                os.remove(temporary_path)
            except OSError:
                pass
