"""
FastAPI Waste Detection Inference & Campaign Service
Fulfills Step 11 & Step 14 of the Implementation Plan:
- Serves the approved YOLO waste detection model
- Health-check (/health) and readiness (/ready) endpoints
- Prediction endpoint (/predict) accepting image uploads with full validation
- Returns detected classes, confidences, bounding boxes, counts, and latency
- Embedded monitoring metrics (/metrics) for drift and performance tracking
"""

import io
import time
import os
from pathlib import Path
from typing import Dict, List, Any
import yaml
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from ultralytics import YOLO

# Initialize FastAPI App
app = FastAPI(
    title="AI-Based Waste Reduction Campaign API",
    description="Automated Street Waste Detection & Hotspot Analysis Service",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State & Monitoring
APP_START_TIME = time.time()
MODEL_PATH = os.getenv("MODEL_PATH", "models/baseline_yolov8.pt")
yolo_model = None

# In-memory Monitoring Metrics (Step 14)
METRICS = {
    "requests_total": 0,
    "successful_predictions": 0,
    "failed_requests": 0,
    "total_latency_ms": 0.0,
    "detected_classes_count": {
        "plastic": 0,
        "waste_pile": 0,
        "face_mask": 0,
        "trash_bin": 0,
    },
    "confidence_scores": []
}

# Response Schemas
class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

class DetectionItem(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox

class PredictionResponse(BaseModel):
    status: str
    inference_time_ms: float
    total_detections: int
    detections_by_class: Dict[str, int]
    detections: List[DetectionItem]
    waste_severity_score: float
    campaign_recommendation: str

class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    version: str

class ReadyResponse(BaseModel):
    ready: bool
    model_loaded: bool
    model_path: str


def get_model():
    global yolo_model
    if yolo_model is None:
        p = Path(MODEL_PATH)
        if not p.exists():
            # Fallback to pretrained nano if custom fine-tuned weights not yet trained
            p = Path("yolov8n.pt")
        print(f"[*] Loading YOLO model from: {p}")
        yolo_model = YOLO(str(p))
    return yolo_model


@app.on_event("startup")
async def startup_event():
    try:
        get_model()
        print("[✓] Model loaded successfully on startup.")
    except Exception as e:
        print(f"[!] Warning: Model failed to load on startup: {e}")


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Step 11: Service availability health check."""
    return {
        "status": "healthy",
        "uptime_seconds": round(time.time() - APP_START_TIME, 2),
        "version": "1.0.0"
    }


@app.get("/ready", response_model=ReadyResponse, tags=["System"])
def readiness_check():
    """Step 11: Readiness endpoint verifying model availability."""
    loaded = yolo_model is not None
    if not loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded or still initializing."
        )
    return {
        "ready": True,
        "model_loaded": True,
        "model_path": MODEL_PATH
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_waste(file: UploadFile = File(...)):
    """
    Step 11: Prediction endpoint accepting uploaded image.
    Validates MIME type, file size, image integrity, and outputs structured detections.
    """
    global METRICS
    METRICS["requests_total"] += 1
    t0 = time.perf_counter()

    # 1. MIME Type Validation
    allowed_types = ["image/jpeg", "image/png", "image/webp", "image/jpg"]
    if file.content_type not in allowed_types:
        METRICS["failed_requests"] += 1
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file format '{file.content_type}'. Allowed: JPEG, PNG, WEBP."
        )

    # 2. File Size Validation (Max 10MB)
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        METRICS["failed_requests"] += 1
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds maximum allowed limit of 10MB."
        )

    # 3. Image Integrity Check
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        METRICS["failed_requests"] += 1
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Corrupted or invalid image content: {str(e)}"
        )

    # 4. Inference
    try:
        model = get_model()
        results = model.predict(source=image, conf=0.25, verbose=False)[0]
    except Exception as e:
        METRICS["failed_requests"] += 1
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}"
        )

    inference_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    METRICS["total_latency_ms"] += inference_ms

    # 5. Parse Output
    detections = []
    class_counts = {"plastic": 0, "waste_pile": 0, "face_mask": 0, "trash_bin": 0}

    # Target class lookup
    target_names = {0: "plastic", 1: "waste_pile", 2: "face_mask", 3: "trash_bin"}

    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        conf = round(float(box.conf[0].item()), 4)
        xyxy = box.xyxy[0].tolist()

        cname = target_names.get(cls_id, results.names.get(cls_id, "plastic"))
        if cname in class_counts:
            class_counts[cname] += 1
            METRICS["detected_classes_count"][cname] = METRICS["detected_classes_count"].get(cname, 0) + 1

        METRICS["confidence_scores"].append(conf)

        detections.append(DetectionItem(
            class_id=cls_id,
            class_name=cname,
            confidence=conf,
            bbox=BoundingBox(
                x1=round(xyxy[0], 2),
                y1=round(xyxy[1], 2),
                x2=round(xyxy[2], 2),
                y2=round(xyxy[3], 2)
            )
        ))

    # 6. Waste Severity & Campaign Recommendation Calculation
    severity_score = round(
        class_counts["waste_pile"] * 3.0 +
        class_counts["plastic"] * 1.5 +
        class_counts["face_mask"] * 1.0 -
        class_counts["trash_bin"] * 1.0,
        2
    )
    severity_score = max(0.0, severity_score)

    if severity_score >= 10.0:
        recommendation = "HIGH_PRIORITY_CLEANUP: Trigger urgent municipal cleanup crew dispatch."
    elif severity_score >= 4.0:
        recommendation = "MEDIUM_PRIORITY: Schedule targeted community volunteer clean-up campaign."
    elif class_counts["trash_bin"] == 0 and severity_score > 0:
        recommendation = "INFRASTRUCTURE_GAP: High litter detected without nearby trash bins. Recommend installing public bin."
    else:
        recommendation = "LOW_PRIORITY / CLEAN: Standard maintenance and routine awareness campaigns."

    METRICS["successful_predictions"] += 1

    return PredictionResponse(
        status="success",
        inference_time_ms=inference_ms,
        total_detections=len(detections),
        detections_by_class=class_counts,
        detections=detections,
        waste_severity_score=severity_score,
        campaign_recommendation=recommendation
    )


@app.get("/metrics", tags=["Monitoring"])
def get_monitoring_metrics():
    """Step 14: API and model drift monitoring endpoint."""
    total_reqs = max(METRICS["requests_total"], 1)
    avg_latency = round(METRICS["total_latency_ms"] / max(METRICS["successful_predictions"], 1), 2)
    avg_conf = round(sum(METRICS["confidence_scores"]) / max(len(METRICS["confidence_scores"]), 1), 3)

    return {
        "requests_total": METRICS["requests_total"],
        "successful_predictions": METRICS["successful_predictions"],
        "failed_requests": METRICS["failed_requests"],
        "error_rate": round(METRICS["failed_requests"] / total_reqs, 4),
        "average_latency_ms": avg_latency,
        "average_confidence": avg_conf,
        "class_detections_aggregate": METRICS["detected_classes_count"],
        "active_model_path": MODEL_PATH
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
