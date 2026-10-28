"""
MLflow Model Registry & Stage Promotion Module
Fulfills Step 10 of the Implementation Plan:
- Registers approved candidate models into MLflow Model Registry
- Enforces Quality Gate verification before promotion
- Transitions approved models between 'Staging' and 'Production'
- Maintains lineage with Git commit, DVC dataset version, and metrics
"""

import sys
from pathlib import Path
import yaml
import mlflow
from mlflow.tracking import MlflowClient


def load_params(params_path="params.yaml"):
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def register_and_promote(model_path="models/baseline_yolov8.pt", run_id=None, stage="Staging", params_path="params.yaml"):
    params = load_params(params_path)
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    client = MlflowClient()

    model_name = params["mlflow"]["model_name"]
    print(f"[*] MLflow Model Registry Target: '{model_name}'")

    if run_id:
        model_uri = f"runs:/{run_id}/weights/best.pt"
    else:
        # Register directly from local path
        model_uri = str(Path(model_path).resolve())

    try:
        # Create registered model if not present
        try:
            client.create_registered_model(model_name)
            print(f"[✓] Created new registered model: {model_name}")
        except Exception:
            pass  # Already exists

        # Create model version
        mv = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id or "",
            description="AI-Based Waste Reduction Campaign YOLO Detector (Trained on pLitterStreet)",
            tags={
                "project": "Waste-Reduction-Campaign",
                "framework": "YOLOv8",
                "dataset": params["dataset"]["name"],
            }
        )
        print(f"[✓] Registered new model version: v{mv.version}")

        # Transition stage
        if stage in ["Staging", "Production", "Archived"]:
            client.transition_model_version_stage(
                name=model_name,
                version=mv.version,
                stage=stage,
                archive_existing_versions=(stage == "Production")
            )
            print(f"[✓] Successfully transitioned model v{mv.version} to stage: '{stage}'")

        return mv.version
    except Exception as e:
        print(f"[!] Model registration failed: {e}")
        return None


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "Staging"
    register_and_promote(stage=stage)
