import json

from src.models.registry import register, resolve_model


def test_mlflow_registry_registers_only_passed_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"test checkpoint bytes")
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(json.dumps({
        "checkpoint": str(checkpoint),
        "quality_gate": {"passed": True},
        "metrics": {"map50": 0.42},
    }), encoding="utf-8")
    risk = tmp_path / "risk.json"
    risk.write_text(json.dumps({
        "testmodel": {"checkpoint": str(checkpoint), "passed": True, "verdict": "APPROVED_FOR_PRODUCTION"}
    }), encoding="utf-8")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'registry.db').as_posix()}")

    version = register(checkpoint, evaluation, risk, "testmodel", promote=True)
    selected = resolve_model("testmodel")

    assert version == 1
    assert selected["version"] == version
    assert selected["path"].read_bytes() == checkpoint.read_bytes()


def test_registry_blocks_failed_quality_gate(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"test checkpoint bytes")
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(json.dumps({
        "checkpoint": str(checkpoint),
        "quality_gate": {"passed": False},
    }), encoding="utf-8")
    risk = tmp_path / "risk.json"
    risk.write_text("{}", encoding="utf-8")

    try:
        register(checkpoint, evaluation, risk, "testmodel")
    except ValueError as exc:
        assert "quality gate" in str(exc)
    else:
        raise AssertionError("Registry accepted a failed evaluation gate")
