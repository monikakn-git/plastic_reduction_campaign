# Phase 2 status

Technical deliverables complete for the local demonstration on 9 November 2026. The model remains demo-only and faculty dataset approval is pending.

- Full local suite: **58 passed**, one existing TestClient deprecation warning.
- GitHub CI: **21 tests passed**, Docker build, import and dependency checks succeeded.
- GitHub CD: exact-commit CI gate, model integrity, image promotion and readiness succeeded.
- Post-deployment check: real-image HTTP inference and invalid-image rejection passed at http://127.0.0.1:8011.
- MLflow registry: `plastic-waste-yolo11m`, version 3, alias `demo`; trained weights and Phase 1 scores are unchanged.
- DVC status: data and pipelines up to date.
- Temporary runner: stopped, unregistered, credentials and local installation removed. Future CI/CD runs need a newly approved online runner; no permanent Mac workflow access remains.

[CI run](https://github.com/Aayushbhandari17/MLOPS_EL/actions/runs/42597980578) · [CD run](https://github.com/Aayushbhandari17/MLOPS_EL/actions/runs/42598091030)

See `PHASE2_FINAL_REPORT.md`, `phase2_final_summary.json`, `github/`, `docker_deployment.json` and `pytest_results.txt` for evidence. No public deployment, new training, production approval or Phase 3 completion is claimed.
