
from datetime import datetime, timezone
import json
import mlflow


def setup_tracking(experiment_name="ecopulse-waste-detection"):
    mlflow.set_experiment(experiment_name)


def log_prediction(result, latitude=None, longitude=None, site_id=None):
    """Log an inference event without logging uploaded image contents."""

    setup_tracking()

    with mlflow.start_run(run_name="waste-inference"):
        mlflow.set_tag("event_type", "inference")
        mlflow.set_tag("model", result.get("model", "YOLO11n"))

        mlflow.log_param(
            "checkpoint",
            result.get("checkpoint", "unknown")
        )
        mlflow.log_metric(
            "total_detections",
            result.get("total_detections", 0)
        )

        if latitude is not None:
            mlflow.log_param("latitude", latitude)

        if longitude is not None:
            mlflow.log_param("longitude", longitude)

        if site_id:
            mlflow.set_tag("site_id", site_id)

        counts = {}
        for prediction in result.get("predictions", []):
            name = prediction["class_name"]
            counts[name] = counts.get(name, 0) + 1

        for class_name, count in counts.items():
            mlflow.log_metric(f"detections_{class_name}", count)

        mlflow.log_dict(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "result": result,
                "latitude": latitude,
                "longitude": longitude,
                "site_id": site_id,
            },
            "prediction.json",
        )
