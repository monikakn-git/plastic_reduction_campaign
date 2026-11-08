"""MLflow Model Registry integration with gated alias promotion."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from src.models.mlflow_yolo import YOLOInferenceModel

from src.tracking.mlflow_utils import get_tracking_uri

DEFAULT_ALIAS = "production"
_CACHE_SECONDS = max(0, int(os.environ.get("ECOPULSE_REGISTRY_CACHE_SECONDS", "30")))
_CACHE: dict[tuple[str, str], tuple[float, dict[str, Any] | None]] = {}
_CACHE_LOCK = threading.Lock()


def _client(tracking_uri: str | None = None) -> MlflowClient:
    uri = tracking_uri or get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    return MlflowClient(tracking_uri=uri)


def resolve_model(model_name: str, alias: str = DEFAULT_ALIAS) -> dict[str, Any] | None:
    """Resolve an MLflow model alias to a downloaded, digest-checked checkpoint."""
    cache_key = (model_name, alias)
    now = time.monotonic()
    with _CACHE_LOCK:
        cached = _CACHE.get(cache_key)
        if cached and now - cached[0] < _CACHE_SECONDS:
            return cached[1]

    client = _client()
    try:
        version = client.get_model_version_by_alias(model_name, alias)
    except MlflowException as exc:
        if exc.error_code == "RESOURCE_DOES_NOT_EXIST" or "alias" in str(exc).lower() and "not found" in str(exc).lower():
            with _CACHE_LOCK:
                _CACHE[cache_key] = (now, None)
            return None
        raise

    source = version.source
    artifact_dir = Path(mlflow.artifacts.download_artifacts(artifact_uri=source))
    checkpoint_name = version.tags.get("checkpoint_name")
    if not checkpoint_name:
        raise ValueError(f"MLflow model {model_name}@{alias} has no checkpoint_name tag")
    checkpoint_artifact_path = version.tags.get("checkpoint_artifact_path", f"artifacts/{checkpoint_name}")
    checkpoint = artifact_dir / checkpoint_artifact_path
    if not checkpoint.is_file():
        available = [str(path.relative_to(artifact_dir)) for path in artifact_dir.rglob("*") if path.is_file()]
        raise FileNotFoundError(f"Checkpoint artifact missing from registered model: {checkpoint}; artifacts={available}")
    expected_digest = version.tags.get("checkpoint_sha256")
    actual_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if not expected_digest or expected_digest != actual_digest:
        raise ValueError(f"Registered checkpoint digest mismatch for {model_name}@{alias}")
    resolved = {
        "name": model_name,
        "version": int(version.version),
        "alias": alias,
        "path": checkpoint.resolve(),
        "sha256": actual_digest,
        "source": source,
    }
    with _CACHE_LOCK:
        _CACHE[cache_key] = (time.monotonic(), resolved)
    return resolved


def _check_reports(checkpoint: Path, evaluation_report: Path, risk_report: Path, model_name: str) -> tuple[dict, dict]:
    evaluation = json.loads(evaluation_report.read_text(encoding="utf-8"))
    if Path(evaluation.get("checkpoint", "")).name != checkpoint.name:
        raise ValueError("Evaluation report does not describe the supplied checkpoint")
    if not evaluation.get("quality_gate", {}).get("passed"):
        raise ValueError("Evaluation quality gate did not pass")
    risk = json.loads(risk_report.read_text(encoding="utf-8"))
    entry = risk.get(model_name) or next(
        (item for item in risk.values() if Path(item.get("checkpoint", "")).name == checkpoint.name), None
    )
    if not entry or Path(entry.get("checkpoint", "")).name != checkpoint.name or not entry.get("passed"):
        raise ValueError("A passing model risk report for this checkpoint is required")
    return evaluation, entry


def register(
    checkpoint: Path,
    evaluation_report: Path,
    risk_report: Path,
    model_name: str,
    promote: bool = False,
    alias: str = DEFAULT_ALIAS,
    tracking_uri: str | None = None,
) -> int:
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    evaluation, risk = _check_reports(checkpoint, evaluation_report, risk_report, model_name)
    client = _client(tracking_uri)

    try:
        client.get_registered_model(model_name)
    except MlflowException as exc:
        if exc.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
        client.create_registered_model(model_name)

    try:
        run_context = mlflow.start_run(run_name=f"register-{model_name}")
        with run_context as run:
            checkpoint_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=YOLOInferenceModel(),
                artifacts={"checkpoint": str(checkpoint)},
                pip_requirements=["mlflow>=2.15.0", "ultralytics>=8.3.0", "pandas>=2.0.0"],
            )
            mlflow.log_artifact(str(evaluation_report), artifact_path="governance")
            mlflow.log_artifact(str(risk_report), artifact_path="governance")
            for key, value in evaluation.get("metrics", {}).items():
                try:
                    mlflow.log_metric(f"evaluation_{key}", float(value))
                except (TypeError, ValueError):
                    continue
            mlflow.set_tags({
                "checkpoint_name": checkpoint.name,
                "checkpoint_sha256": checkpoint_digest,
                "checkpoint_artifact_path": f"artifacts/{checkpoint.name}",
                "evaluation_gate": "PASSED",
                "risk_verdict": str(risk.get("verdict", "PASSED")),
                "evaluation_report": evaluation_report.name,
                "risk_report": risk_report.name,
            })
            source = f"runs:/{run.info.run_id}/model"
            model_version = client.create_model_version(
                name=model_name,
                source=source,
                run_id=run.info.run_id,
                tags={
                    "checkpoint_name": checkpoint.name,
                    "checkpoint_sha256": checkpoint_digest,
                    "checkpoint_artifact_path": f"artifacts/{checkpoint.name}",
                    "evaluation_gate": "PASSED",
                    "risk_verdict": str(risk.get("verdict", "PASSED")),
                },
            )
    except MlflowException as exc:
        message = str(exc)
        if "mlflow-artifacts:/" in message and "HTTP" in message:
            raise RuntimeError(
                "This experiment uses MLflow's artifact proxy, but the configured tracking URI is local. "
                "Start the project MLflow server with `docker compose up -d mlflow` and rerun this command "
                "with `--tracking-uri http://localhost:5000`."
            ) from exc
        raise

    if promote:
        client.set_registered_model_alias(model_name, alias, model_version.version)
    return int(model_version.version)


def main() -> int:
    parser = argparse.ArgumentParser(description="Register/promote a checkpoint after quality and risk gates pass")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--risk", type=Path, required=True)
    parser.add_argument("--name", required=True, help="MLflow registered model name, e.g. yolo11n")
    parser.add_argument("--alias", default=DEFAULT_ALIAS)
    parser.add_argument(
        "--tracking-uri",
        help="MLflow tracking server URI; use http://localhost:5000 for the Docker Compose registry",
    )
    parser.add_argument("--promote", action="store_true", help="Point the MLflow model alias to the new version")
    args = parser.parse_args()
    version = register(
        args.checkpoint, args.evaluation, args.risk, args.name, args.promote, args.alias, args.tracking_uri
    )
    print(f"Registered {args.name} version {version}" + (f" as {args.alias}" if args.promote else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
