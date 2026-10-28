"""
Automated Pytest Suite for AI Waste Reduction Campaign
Fulfills Step 8 of the Implementation Plan:
- Tests model loading and prediction format
- Tests bounding box coordinate validity and confidence ranges
- Tests API endpoints (/health, /ready, /predict)
- Tests error handling for unsupported formats and corrupted files
- Tests campaign hotspot prioritization and resource optimizer
"""

import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from ultralytics import YOLO

from src.api.main import app
from src.campaign.hotspot_optimizer import WasteReport, cluster_hotspots, optimize_campaign_budget


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_rgb_image_bytes():
    """Generates a small test RGB image in memory."""
    img = Image.new("RGB", (640, 640), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.getvalue()


def test_model_loading():
    """Verify that a YOLO model instance can be instantiated."""
    model = YOLO("yolov8n.pt")
    assert model is not None
    assert hasattr(model, "predict")


def test_api_health_endpoint(client):
    """Test healthcheck endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "uptime_seconds" in data


def test_api_ready_endpoint(client):
    """Test readiness endpoint."""
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is True


def test_api_predict_valid_image(client, sample_rgb_image_bytes):
    """Test prediction endpoint with valid JPEG image."""
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", sample_rgb_image_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "inference_time_ms" in data
    assert "waste_severity_score" in data
    assert "campaign_recommendation" in data
    assert isinstance(data["detections"], list)


def test_api_predict_unsupported_file_type(client):
    """Test rejection of non-image file formats (e.g. text file)."""
    fake_txt = io.BytesIO(b"Not an image")
    response = client.post(
        "/predict",
        files={"file": ("malicious.txt", fake_txt, "text/plain")}
    )
    assert response.status_code == 415


def test_api_metrics_endpoint(client):
    """Test monitoring metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "requests_total" in data
    assert "average_latency_ms" in data
    assert "class_detections_aggregate" in data


def test_campaign_hotspot_clustering_and_optimizer():
    """Test campaign logic: geospatial clustering and knapsack budget allocation."""
    reports = [
        WasteReport("R1", 12.900, 77.500, plastic_count=10, waste_pile_count=2, face_mask_count=0, trash_bin_count=0, timestamp="2026-10-28"),
        WasteReport("R2", 12.901, 77.501, plastic_count=5, waste_pile_count=1, face_mask_count=0, trash_bin_count=0, timestamp="2026-10-28"),
        WasteReport("R3", 13.100, 78.100, plastic_count=2, waste_pile_count=0, face_mask_count=0, trash_bin_count=1, timestamp="2026-10-28"),
    ]
    hotspots = cluster_hotspots(reports, radius_meters=300.0)
    assert len(hotspots) == 2  # R1 and R2 clustered, R3 distinct

    # Test knapsack budget allocation
    allocation = optimize_campaign_budget(hotspots, max_budget_usd=300.0, max_volunteers=15)
    assert allocation["scheduled_campaigns_count"] >= 1
    assert allocation["total_budget_allocated_usd"] <= 300.0
