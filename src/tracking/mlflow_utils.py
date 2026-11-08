from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Optional

import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRACKING_DB = PROJECT_ROOT / "mlflow.db"
DEFAULT_EXPERIMENT = "ecopulse-waste-detection"


def get_tracking_uri() -> str:
    """Resolve MLflow tracking URI, prioritizing environment variables."""
    return os.environ.get(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{TRACKING_DB.as_posix()}"
    )


def setup_tracking(experiment_name: str = DEFAULT_EXPERIMENT) -> None:
    """Configure MLflow tracking URI and experiment."""
    tracking_uri = get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_prediction(
    result: dict[str, Any],
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    site_id: Optional[str] = None,
    experiment_name: str = DEFAULT_EXPERIMENT,
) -> Optional[str]:
    """Log an inference event without logging uploaded image contents.
    
    Returns:
        The MLflow run_id if successful, or None if logging failed or is disabled.
    """
    if os.environ.get("ECOPULSE_DISABLE_MLFLOW", "").lower() in ("1", "true", "yes"):
        return None

    try:
        setup_tracking(experiment_name)

        with mlflow.start_run(run_name="waste-inference") as run:
            mlflow.set_tag("event_type", "inference")
            mlflow.set_tag("project", "EcoPulse AI")
            mlflow.set_tag("model", result.get("model", "YOLO11n"))

            mlflow.log_param(
                "checkpoint",
                str(result.get("checkpoint", "unknown"))
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

            counts: dict[str, int] = {}
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
            return run.info.run_id
    except Exception as exc:
        print(f"Warning: Failed to log prediction to MLflow: {exc}")
        return None
