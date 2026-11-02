
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from src.models.predict import load_model, predict_image
from src.tracking.mlflow_utils import log_prediction


app = FastAPI(
    title="EcoPulse AI Waste Detection API",
    description="Waste detection using the existing trained YOLO11n model",
    version="1.0.0",
)


@app.get("/")
def home():
    return {
        "project": "EcoPulse AI",
        "message": "Waste Detection API is running",
        "model": "YOLO11n",
    }


@app.get("/health")
def health():
    try:
        model = load_model()

        return {
            "status": "healthy",
            "model_loaded": True,
            "classes": model.names,
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
    latitude: float | None = None,
    longitude: float | None = None,
    site_id: str | None = None,
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
            detail="Unsupported image format.",
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

        # Run existing trained YOLO11n model
        result = predict_image(
            temporary_path,
            confidence=confidence,
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
            os.remove(temporary_path)
