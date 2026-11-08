import csv

from src.monitoring import observations


def test_summarize_outputs_stable_drift_features():
    result = {
        "predictions": [
            {"class_id": 5, "confidence": 0.8, "bbox": [10, 20, 30, 60]},
            {"class_id": 5, "confidence": 0.6, "bbox": [0, 0, 10, 10]},
        ]
    }
    summary = observations.summarize(result, "yolo11n", 100, 100, version=3)

    assert summary["total_detections"] == 2
    assert summary["detections_5"] == 2
    assert summary["mean_confidence"] == 0.7
    assert summary["model_version"] == 3


def test_log_observation_persists_only_summary(monkeypatch, tmp_path):
    output = tmp_path / "observations.csv"
    monkeypatch.setattr(observations, "OBSERVATION_PATH", output)
    observations.log_observation(
        {"predictions": []}, "yolo11n", 640, 480, {"version": 2}
    )

    with output.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["image_width"] == "640"
    assert "uploaded_filename" not in rows[0]


def test_api_exposes_prometheus_scrape_endpoint():
    from src.api.app import app

    paths = {route.path for route in app.routes}
    assert "/metrics" in paths
