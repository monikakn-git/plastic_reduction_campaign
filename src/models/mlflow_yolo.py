"""MLflow PyFunc adapter for versioning YOLO checkpoints in the model registry."""
from __future__ import annotations

import json
from typing import Any

import mlflow.pyfunc


class YOLOInferenceModel(mlflow.pyfunc.PythonModel):
    """Accept a pandas DataFrame with image_path and optional confidence columns."""

    def load_context(self, context: Any) -> None:
        from ultralytics import YOLO

        self.model = YOLO(context.artifacts["checkpoint"])

    def predict(self, context, model_input, params=None):
        import pandas as pd

        if "image_path" not in model_input.columns:
            raise ValueError("Input DataFrame must include an image_path column")
        rows = []
        for _, row in model_input.iterrows():
            confidence = float(row.get("confidence", 0.25))
            result = self.model.predict(str(row["image_path"]), conf=confidence, verbose=False)[0]
            detections = []
            for box in result.boxes:
                class_id = int(box.cls[0])
                detections.append({
                    "class_id": class_id,
                    "class_name": self.model.names.get(class_id, f"class_{class_id}"),
                    "confidence": float(box.conf[0]),
                    "bbox": [float(v) for v in box.xyxy[0].tolist()],
                })
            rows.append({"image_path": str(row["image_path"]), "predictions": json.dumps(detections)})
        return pd.DataFrame(rows)
