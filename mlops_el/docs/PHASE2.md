# Phase 2: productionization and deployment

## Current scope

The unchanged Phase 1 YOLO11m checkpoint is packaged for a local demonstration. No retraining or new test-set tuning is involved. Test mAP50 remains 37.12%, mAP50–95 22.74%, precision 46.13% and recall 39.65%.

Completed: FastAPI inference service, input/security checks, PyTest tests including real inference, MLflow registry version 3 (`plastic-waste-yolo11m`, alias `demo`), verified release manifest, registry/API prediction comparison and native localhost HTTP smoke test. Docker was installed through Colima and the ARM64 CPU image was built and deployed on 9 November 2026. Actual container health, HTTP inference, malformed-upload rejection and dependency consistency checks passed. GitHub Actions CI and manual local delivery both executed successfully; links and evidence appear below. Faculty dataset approval remains pending.

## Local demonstration

Local deployment runs on this Mac. Public deployment would run on an external server and provide an internet-accessible address; its provider, access controls and resource limits require a separate deployment choice.

The current Docker demonstration uses http://127.0.0.1:8011 ; interactive API documentation is at http://127.0.0.1:8011/docs . The former native process on port 8010 has been stopped. The service is available while the Docker VM and container are running. For a separate native development session, use:

```sh
MODEL_BUNDLE=releases/yolo11m-demo-v3 INFERENCE_DEVICE=cpu .venv/bin/uvicorn service.app:app --host 127.0.0.1 --port 8010 --workers 1 --no-access-log
```

Send raw JPEG/PNG bytes, not a multipart form:

```sh
curl -H 'Content-Type: image/jpeg' --data-binary @your-photo.jpg http://127.0.0.1:8011/predict
```

`/health` checks the process, `/ready` checks model readiness, and `/model` exposes release provenance. The web page supports image upload and displays bounding boxes. Requests use fixed 960 input size, confidence 0.25 and NMS IoU 0.7. Confidence 0.25 is a demo operating default, not an optimized production cutoff; it differs from the low confidence filter used to compute evaluation mAP.

## Quality and security limits

The service verifies the trusted checkpoint SHA256 before loading it. It accepts JPEG/PNG only, up to 8 MiB and 20 million pixels, with dimension and decoder checks. It processes one inference at a time on CPU, rejects excess concurrent work, limits upload time and returns generic internal errors. Images are decoded in memory and are not saved. Logs include request IDs, timings, detection counts and model hash; they omit image bytes, filenames and locations. The service accepts no remote image URLs and binds to localhost. Public exposure would require authentication, rate limiting, transport security and a provider-specific review.

Outputs require human review. Visual classes do not establish plastic material, geographic representativeness or campaign impact. Cup detection is particularly weak. Prospective engineering gates require mAP50 >= 0.50, recall >= 0.50, every class AP50 >= 0.25 and faculty dataset approval. These are project release criteria, not college-mandated accuracy thresholds. The candidate fails production gates and is explicitly marked `production_approved=false`; only the `demo` alias is assigned.

## Reproducibility and evidence

- `reports/phase2/registered_model.json`: registry identity and source runs.
- `reports/phase2/quality_gates.json`: integrity and production gate results.
- `reports/phase2/smoke_test.json`: actual registry/API agreement.
- `reports/phase2/live_http_smoke.json`: earlier native localhost HTTP request.
- `reports/phase2/docker_deployment.json`: actual container, isolation settings and HTTP inference evidence.
- `reports/phase2/container_requirements.txt`: exact installed container packages.
- `reports/phase2/docker_build.log`: successful image build output.
- `releases/yolo11m-demo-v3.dvc`: release weights and manifest, stored in the existing local DVC cache.

The source training run is `85714466948d4e658ebb610a475dcf57`; the packaging run is separate. Phase 1 metrics, archived tracking and weights are preserved. The DVC remote remains unconfigured; local versioning is not an off-device backup. The test image used for service smoke checks comes from the training split and provides engineering verification, not another accuracy evaluation.

Run checks:

```sh
RUN_MODEL_INTEGRATION=1 .venv/bin/python -m pytest -q
.venv/bin/python scripts/verify_phase2_release.py releases/yolo11m-demo-v3
```

The integration test requires the local private release and training sample. GitHub CI runs the API/quality tests without private images or weights. Native dependency versions are recorded in the existing environment and MLflow package. Docker pins direct service packages and CPU Torch versions, and the Python base image is digest-pinned. The ARM64 Linux build and actual inference passed; resolved transitive versions are recorded in `container_requirements.txt`. Transitive packages are recorded for audit but are not yet a hash-locked installation input. CI also uses the native ARM64 Mac runner and verifies ARM64 Linux images; AMD64 builds remain outside the demonstrated deployment target.

## Docker operation

Docker Engine runs in the dedicated `mlops-el` Colima profile: 2 CPUs, 3 GiB VM RAM, two 8 GiB maximum sparse disks. These disk capacities are ceilings, not immediately allocated storage. The container is limited to 2 GiB and used approximately 740 MiB in a single smoke observation. Runtime load varies.

Project Docker configuration is kept in `.local/docker`; it does not replace the user's default Docker configuration. `scripts/docker_local.sh` selects this project context when that configuration exists. The model directory is the only host directory mounted into this VM and is read-only. The container also mounts it read-only. No datasets or credentials enter the build context.

From the repository directory:

```sh
# Restart the VM after shutdown (not automatically started at login):
DOCKER_CONFIG="$PWD/.local/docker" colima start mlops-el --activate=false
# Start the already-built service:
sh scripts/docker_local.sh compose up -d --wait
# Verify actual model inference:
.venv/bin/python scripts/check_docker_phase2.py --port 8011
# Stop the container and free VM memory when finished:
sh scripts/docker_local.sh compose down
colima stop mlops-el
```

`sh scripts/deploy_local.sh` verifies the private release, builds and deploys it. Default port is 8011; override `API_PORT` if occupied. Never kill an unrelated application to free a port. Restarting the existing VM does not need another installer download. Recreating a deleted VM requires the official Colima image; the initial normal release URL returned 504, so the same official asset was retrieved through GitHub's API and checked against its published SHA256 before use. Temporary decompressed installer data was removed. Rebuildable Docker build cache was pruned after verification; trained weights, tracking and data were preserved.

## Verified GitHub CI and controlled delivery

Private repository: https://github.com/Aayushbhandari17/MLOPS_EL . Both workflows executed successfully on 9 November 2026 using a user-approved temporary ARM64 Mac runner.

- Successful CI: https://github.com/Aayushbhandari17/MLOPS_EL/actions/runs/42597980578
- Successful delivery: https://github.com/Aayushbhandari17/MLOPS_EL/actions/runs/42598091030
- Deployed source commit: `fbe5d9003acd370f5beabbde31a9937e90468fd0`.
- CI passed 21 API, quality and deployment-gate tests, built the commit-labelled Docker image, verified service import and ran `pip check`.
- CD verified a successful CI result for that exact commit, verified the private model release, checked the image revision label, promoted the tested image without rebuilding and waited for readiness.
- A real training-image HTTP request after CD verified actual inference. The complete local suite, including real-model integration, passed 58 tests with one existing TestClient deprecation warning.

CI runs on owner-triggered main-branch pushes and manual dispatch, excluding documentation/evidence-only pushes. CD is manually dispatched from `main`; this is continuous delivery with an explicit release action, not unattended production deployment. Both workflows refuse other actors, other branches and public repositories. Checkout is SHA-pinned and does not persist its token. Workflow token permissions are limited to the needed read access. No weights or datasets are uploaded to GitHub Actions build contexts.

GitHub Free private repositories do not provide the planned protected deployment environment. The demonstrated control is therefore owner-only manual dispatch plus successful CI for the exact source commit. It is not a claim of reviewer approval or branch-protection enforcement. The runner used this existing Mac, with no hosted compute or artifact-storage uploads and no spending.

The first CI attempt could not find Docker because its configuration variable was absent when that run was queued. A new run after configuration passed. Initial CD refused to continue when this Mac's Python lacked its default HTTPS CA bundle. The workflows now explicitly use the existing macOS `/etc/ssl/cert.pem` bundle; certificate verification remains enabled. Failed and successful logs are retained under `reports/phase2/github`.

After successful delivery, the temporary runner was stopped, unregistered, its credentials removed and its local installation deleted, as approved by the user. No runner remains registered and no runner login service was installed. The Docker application remains running. **Future workflow execution requires a newly approved, registered online runner bearing `mlops-el-demo`; queued jobs cannot run while that runner is absent.** Keep the repository private and the owner/main restrictions in place.

For another supervised demonstration, install the official ARM64 runner, verify its published checksum, register it temporarily to this private repository and start it. Existing repository variables `MODEL_RELEASE_DIR` and `PROJECT_DOCKER_CONFIG` select the private model and this project's isolated Docker context. Dispatch CI for the desired main commit; after success, dispatch CD for that same commit. Remove the runner afterward if temporary access is again selected. Runner registration grants repository code execution as the Mac user and requires explicit authorization; do not put registration tokens into source files or logs.

Phase 2 technical deliverables are complete for the local demonstration. Faculty dataset approval is still pending and model-quality gates still prevent production approval. Public hosting, location/campaign analysis, drift monitoring, feedback/retraining demonstrations and governance remain outside this completed Phase 2 scope.

Official references: [self-hosted runner setup](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/add-runners), [Actions billing and usage](https://docs.github.com/en/actions/concepts/billing-and-usage).
