# Phase 2 final review: productionization and local deployment

Completed on 9 November 2026 for a local demonstration. The existing YOLO11m model is now packaged, tested, registered, served and delivered through a verified GitHub pipeline. No retraining was performed.

| College requirement | Implementation | Execution evidence |
|---|---|---|
| PyTest model/service testing | API contracts, invalid inputs, concurrency, source identity, deployment gate and real inference | `pytest_results.txt`: 58 passed; GitHub CI: 21 passed |
| Model-risk and quality checks | Trusted checkpoint SHA256, fixed schema, frozen metrics, explicit failed production gates and human review | `quality_gates.json`, `configs/phase2/release_policy.json` |
| MLflow registry | `plastic-waste-yolo11m` version 3, `demo` alias, source experiment lineage | `registered_model.json`; local MLflow registry |
| FastAPI inference | Bounded JPEG/PNG upload, model/readiness endpoints, predictions and browser demonstration | `example_prediction.json`, `docker_deployment.json` |
| Docker | Digest-pinned base, ARM64 CPU image, non-root user, read-only model/root, localhost binding, memory/pid limits | `Dockerfile`, `compose.yaml`, `docker_build.log`, `container_requirements.txt` |
| GitHub Actions CI | Owner/main/private-only tests, commit-labelled image build, import and dependency checks | Successful run 42597980578; `github/ci_run.json`, `github/ci_success.log` |
| GitHub Actions delivery | Manual dispatch, latest completed successful CI for exact commit, verified model, same-image promotion and readiness | Successful run 42598091030; `github/cd_run.json`, `github/cd_success.log` |
| Deployment | Local service, followed by real-image inference and malformed-image rejection | http://127.0.0.1:8011; `docker_deployment.json` |

## Reproducibility and lineage

Private repository: https://github.com/Aayushbhandari17/MLOPS_EL . Successful CI: https://github.com/Aayushbhandari17/MLOPS_EL/actions/runs/42597980578 . Successful delivery: https://github.com/Aayushbhandari17/MLOPS_EL/actions/runs/42598091030 . Deployed code is commit `fbe5d9003acd370f5beabbde31a9937e90468fd0`; later documentation/evidence commits do not change that deployed code.

Model weights SHA256 is `a9a39b52f2de03927ce5f387a6feb52295468e50262e436d75ee6a0a95ed1b8e`. Training run is `85714466948d4e658ebb610a475dcf57`; final test run is `4b26901d393b4fc89f0a716818788fa6`. The trained dataset remains labels-v3. Registry version increments recorded service packaging changes, not new training. The exact CI source commit is embedded in the Docker image label and matched to both successful workflow runs. The private weights are mounted read-only and never included in the image build context. The release is DVC-versioned; DVC status remained clean. The DVC remote is still unconfigured.

## Evaluation and operating limits

Frozen test mAP50 is 37.12%, mAP50–95 22.74%, precision 46.13% and recall 39.65%. These are Phase 1 evaluation results, not additional Phase 2 test tuning. Engineering smoke requests use a training image. The application uses input size 960, confidence threshold 0.25 and NMS IoU 0.7. Its confidence threshold is an operating filter, not accuracy, and is not production-calibrated. Missed detections remain possible, including indoor/coloured objects; no detection does not establish the absence of waste. Object classes do not establish plastic material.

Prospective project production gates and faculty dataset approval remain unmet; `production_approved=false` is enforced. These quality thresholds were chosen as engineering policy, not supplied as college accuracy requirements. Human review is required. Location collection, hotspot maps and campaign recommendations are not implemented yet.

## Security and delivery controls

Images are decoded in memory with explicit JPEG/PNG decoders, byte/pixel/dimension limits and EXIF handling. The API accepts no remote image URL or user-selected model path. Upload timing and concurrent inference are bounded; invalid uploads are rejected. Image bytes, filenames and locations are not logged. The service binds to localhost. The Docker image runs under user 10001, drops capabilities and has a read-only root and read-only model mount.

The CI/CD runner was temporary and explicitly approved after automatic approval review required authorization for remote workflow execution on the Mac. Both workflows restrict execution to the repository owner, private repository and main branch. Tokens were not committed or printed. The runner was stopped, unregistered, its credentials removed and its installation deleted after execution; `github/runner_cleanup.json` records verification. No always-on runner or login service remains. Future workflow runs require a new approved online runner. GitHub-hosted compute and artifact uploads were not used, and no spending was authorized or initiated.

GitHub Free private repositories do not provide the originally proposed protected deployment environment. Manual owner-only dispatch and an exact-commit CI gate are the demonstrated release controls. This is continuous delivery with an explicit release action. It is not unattended production deployment or independently reviewed approval.

## Failure and recovery evidence

The first CI run was queued before its Docker-path variable existed and failed to locate the project context. A subsequent run with the configured variables passed. The first delivery run correctly stopped at its CI lookup because Python's default CA path was missing. Explicit use of the existing macOS trusted CA bundle fixed it; TLS verification was retained. Those failed logs and successful retries are preserved under `github/`. The old running application was preserved until a verified image could be promoted.

The local full suite passed 58 tests with one existing Starlette TestClient/httpx deprecation warning. CI passed 21 boundary, quality and deployment-gate tests; private image integration is verified separately on the local host. The deployment target demonstrated is ARM64 Linux through Colima on this Mac. Public exposure, AMD64 builds, hash-locking of all transitive dependencies and independent security certification are not claimed.

## Handoff

Use `docs/PHASE2.md` for run/stop commands. Keep the model bundle and project Docker context available. The application works while Colima and its container run. Saved logs and JSON receipts support the college review even after GitHub log retention expires.

Phase 3 remains: monitoring, drift/degradation experiment, feedback and retraining/redeployment evidence, model card/governance, final report and planned location/campaign features. Faculty approval remains an external outstanding item.
