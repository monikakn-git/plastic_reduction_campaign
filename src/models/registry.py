"""MLflow Model Registry integration with gated alias promotion."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
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

from src.tracking.mlflow_utils import DEFAULT_EXPERIMENT, get_tracking_uri

DEFAULT_ALIAS = "production"
_CACHE_SECONDS = max(0, int(os.environ.get("ECOPULSE_REGISTRY_CACHE_SECONDS", "30")))
_CACHE: dict[tuple[str, str], tuple[float, dict[str, Any] | None]] = {}
_CACHE_LOCK = threading.Lock()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluation_artifact_dir(
    evaluation_report: Path,
    checkpoint: Path,
    evaluation: dict[str, Any],
    explicit_dir: Path | None,
) -> Path | None:
    """Find the detailed evaluation bundle created by evaluate.py."""
    if explicit_dir:
        resolved = explicit_dir.resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"Evaluation artifact directory not found: {resolved}")
        return resolved

    split = str(evaluation.get("split", "test"))
    candidates = (
        evaluation_report.parent / "mlflow_artifacts" / checkpoint.stem / split,
        PROJECT_ROOT / "reports" / "mlflow_artifacts" / checkpoint.stem / split,
    )
    return next((candidate.resolve() for candidate in candidates if candidate.is_dir()), None)


def _quality_artifacts(risk_report: Path, checkpoint: Path) -> list[Path]:
    """Collect available risk-gate summaries and detailed Deepchecks output."""
    candidates = [
        risk_report.with_suffix(".md"),
        risk_report.parent / f"deepchecks_{checkpoint.stem.replace('_best', '')}_quality.json",
        risk_report.parent / f"deepchecks_{checkpoint.stem.replace('_best', '')}_quality.html",
    ]
    return [path.resolve() for path in candidates if path.is_file()]


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
    expected_digest = version.tags.get("checkpoint_sha256")
    checkpoint_artifact_path = version.tags.get("checkpoint_artifact_path", f"artifacts/{checkpoint_name}")
    tagged_checkpoint = artifact_dir / checkpoint_artifact_path
    candidates = [tagged_checkpoint] if tagged_checkpoint.is_file() else []
    candidates.extend(
        path for path in artifact_dir.rglob("*")
        if path.is_file() and path not in candidates
        and (path.name == checkpoint_name or path.name == "checkpoint")
    )
    checkpoint = next((path for path in candidates if expected_digest and _sha256(path) == expected_digest), None)
    if checkpoint is None:
        available = [str(path.relative_to(artifact_dir)) for path in artifact_dir.rglob("*") if path.is_file()]
        if not expected_digest:
            raise ValueError(f"Registered checkpoint {model_name}@{alias} has no SHA-256 tag")
        raise ValueError(
            f"Registered checkpoint digest mismatch or artifact missing for {model_name}@{alias}; "
            f"expected SHA-256 {expected_digest}; artifacts={available}"
        )
    actual_digest = expected_digest
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


def _log_registry_evidence(
    checkpoint: Path,
    evaluation_report: Path,
    risk_report: Path,
    model_name: str,
    evaluation: dict[str, Any],
    risk: dict[str, Any],
    evaluation_bundle: Path | None,
    quality_artifacts: list[Path],
) -> str:
    """Log the complete reproducibility and governance bundle to the active run."""
    checkpoint_digest = _sha256(checkpoint)
    mlflow.log_artifact(str(evaluation_report), artifact_path="governance/evaluation")
    evaluation_markdown = evaluation_report.with_suffix(".md")
    if evaluation_markdown.is_file():
        mlflow.log_artifact(str(evaluation_markdown), artifact_path="governance/evaluation")
    mlflow.log_artifact(str(risk_report), artifact_path="governance/risk")
    if evaluation_bundle:
        mlflow.log_artifacts(str(evaluation_bundle), artifact_path="evaluation_details")
    else:
        print(
            "Warning: no detailed evaluation artifact directory found; "
            "only the evaluation summary will be logged."
        )
    for artifact in quality_artifacts:
        mlflow.log_artifact(str(artifact), artifact_path="governance/quality_checks")

    artifact_files = [checkpoint, evaluation_report, risk_report, *quality_artifacts]
    if evaluation_markdown.is_file():
        artifact_files.append(evaluation_markdown)
    if evaluation_bundle:
        artifact_files.extend(path for path in evaluation_bundle.rglob("*") if path.is_file())
    artifact_manifest = [
        {
            "file": str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else path.name,
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(set(artifact_files), key=lambda item: str(item))
    ]
    mlflow.log_dict(
        {
            "model_name": model_name,
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": checkpoint_digest,
            "evaluation_gate": "PASSED",
            "risk_verdict": str(risk.get("verdict", "PASSED")),
            "metrics": evaluation.get("metrics", {}),
            "classes": evaluation.get("classes", {}),
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "mlflow": mlflow.__version__,
            },
            "artifacts": artifact_manifest,
        },
        "registry/model_manifest.json",
    )
    for key, value in evaluation.get("metrics", {}).items():
        try:
            mlflow.log_metric(f"evaluation_{key}", float(value))
        except (TypeError, ValueError):
            continue
    mlflow.set_tags({
        "checkpoint_name": checkpoint.name,
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_artifact_path": "artifacts/checkpoint",
        "evaluation_gate": "PASSED",
        "risk_verdict": str(risk.get("verdict", "PASSED")),
        "evaluation_report": evaluation_report.name,
        "risk_report": risk_report.name,
        "artifacts_complete": "true",
    })
    return checkpoint_digest


def backfill_registered_artifacts(
    checkpoint: Path,
    evaluation_report: Path,
    risk_report: Path,
    model_name: str,
    alias: str = DEFAULT_ALIAS,
    tracking_uri: str | None = None,
    evaluation_artifacts: Path | None = None,
) -> int:
    """Add missing governance/evaluation artifacts to the currently aliased model run."""
    checkpoint = checkpoint.resolve()
    evaluation_report = evaluation_report.resolve()
    risk_report = risk_report.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    evaluation, risk = _check_reports(checkpoint, evaluation_report, risk_report, model_name)
    evaluation_bundle = _evaluation_artifact_dir(evaluation_report, checkpoint, evaluation, evaluation_artifacts)
    quality_artifacts = _quality_artifacts(risk_report, checkpoint)
    client = _client(tracking_uri)
    version = client.get_model_version_by_alias(model_name, alias)
    checkpoint_digest = _sha256(checkpoint)
    if version.tags.get("checkpoint_sha256") != checkpoint_digest:
        raise ValueError(
            f"The {model_name}@{alias} alias points to a different checkpoint; "
            "refusing to attach these reports to the wrong model version."
        )

    with mlflow.start_run(run_id=version.run_id):
        _log_registry_evidence(
            checkpoint, evaluation_report, risk_report, model_name,
            evaluation, risk, evaluation_bundle, quality_artifacts,
        )
    client.set_model_version_tag(model_name, version.version, "artifacts_complete", "true")
    client.set_model_version_tag(model_name, version.version, "checkpoint_artifact_path", "artifacts/checkpoint")
    return int(version.version)


def register(
    checkpoint: Path,
    evaluation_report: Path,
    risk_report: Path,
    model_name: str,
    promote: bool = False,
    alias: str = DEFAULT_ALIAS,
    tracking_uri: str | None = None,
    evaluation_artifacts: Path | None = None,
) -> int:
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    evaluation, risk = _check_reports(checkpoint, evaluation_report, risk_report, model_name)
    evaluation_bundle = _evaluation_artifact_dir(
        evaluation_report.resolve(), checkpoint, evaluation, evaluation_artifacts
    )
    quality_artifacts = _quality_artifacts(risk_report.resolve(), checkpoint)
    client = _client(tracking_uri)
    mlflow.set_experiment(DEFAULT_EXPERIMENT)

    try:
        client.get_registered_model(model_name)
    except MlflowException as exc:
        if exc.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
        client.create_registered_model(model_name)

    try:
        run_context = mlflow.start_run(run_name=f"register-{model_name}")
        with run_context as run:
            # The inference API only resolves checkpoints; keep the heavier PyFunc
            # wrapper imports out of its startup path.
            from src.models.mlflow_yolo import YOLOInferenceModel

            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=YOLOInferenceModel(),
                artifacts={"checkpoint": str(checkpoint)},
                pip_requirements=["mlflow>=2.15.0", "ultralytics>=8.3.0", "pandas>=2.0.0"],
            )
            checkpoint_digest = _log_registry_evidence(
                checkpoint, evaluation_report.resolve(), risk_report.resolve(), model_name,
                evaluation, risk, evaluation_bundle, quality_artifacts,
            )
            source = f"runs:/{run.info.run_id}/model"
            model_version = client.create_model_version(
                name=model_name,
                source=source,
                run_id=run.info.run_id,
                tags={
                    "checkpoint_name": checkpoint.name,
                    "checkpoint_sha256": checkpoint_digest,
                    "checkpoint_artifact_path": "artifacts/checkpoint",
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
    parser.add_argument(
        "--evaluation-artifacts",
        type=Path,
        help="Detailed evaluation artifact directory; auto-discovered under reports/mlflow_artifacts when omitted",
    )
    parser.add_argument(
        "--backfill-existing",
        action="store_true",
        help="Add reports and detailed evaluation files to the currently aliased model run without creating a new version",
    )
    parser.add_argument("--promote", action="store_true", help="Point the MLflow model alias to the new version")
    args = parser.parse_args()
    if args.backfill_existing:
        version = backfill_registered_artifacts(
            args.checkpoint, args.evaluation, args.risk, args.name,
            args.alias, args.tracking_uri, args.evaluation_artifacts,
        )
        print(f"Added complete artifacts to {args.name} version {version} ({args.alias})")
        return 0
    version = register(
        args.checkpoint, args.evaluation, args.risk, args.name, args.promote, args.alias,
        args.tracking_uri, args.evaluation_artifacts,
    )
    print(f"Registered {args.name} version {version}" + (f" as {args.alias}" if args.promote else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
