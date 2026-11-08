import json
import sys

from src.models import risk_quality_checks


def test_failed_risk_assessment_is_reported_without_crashing(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, "argv", [
        "risk_quality_checks.py",
        "--checkpoint", "yolo11n",
        "--output-json", str(tmp_path / "risk.json"),
        "--output-md", str(tmp_path / "risk.md"),
    ])
    monkeypatch.setattr(
        risk_quality_checks,
        "evaluate_model_risk",
        lambda **kwargs: {
            "model_key": "yolo11n",
            "status": "FAILED",
            "reason": "Deepchecks is not installed",
        },
    )

    exit_code = risk_quality_checks.main()

    assert exit_code == 1
    assert "Deepchecks is not installed" in capsys.readouterr().out
    report = json.loads((tmp_path / "risk.json").read_text(encoding="utf-8"))
    assert report["yolo11n"]["status"] == "FAILED"
    assert "Deepchecks is not installed" in (tmp_path / "risk.md").read_text(encoding="utf-8")
