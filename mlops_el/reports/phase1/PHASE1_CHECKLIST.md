# Phase 1 college checklist

Technical evidence is ready for review. Faculty dataset approval remains pending.

| College requirement | Status | Evidence |
|---|---|---|
| Dataset acquisition / approval | Acquisition complete; approval pending | acquisition.json, docs/DATASET_APPROVAL.md, raw DVC pointer |
| Data exploration | Complete | exploration.md, distributions, image/object-size plots |
| Feature engineering | Complete | class map, prepare_data.py, DATA_SCHEMA.md, preprocessing records |
| Baseline models | Complete | YOLO11n baseline, YOLO11s candidate, YOLO11m final candidate |
| Git repository | Complete locally | Commit history and tracked code/config/reports; no remote publication claimed |
| DVC dataset versioning | Complete locally | dvc.lock, raw/v2/v3/v4 pointers, model pointer; remote not configured |
| MLflow experiment tracking | Complete | Local baseline/evaluation store and restored cloud training view |
| Initial model evaluation | Complete | Separate validation/test reports, per-class results, confusion/PR curves |

Additional evidence: 36 passing tests; same-run checkpoint recovery between Colab and Kaggle; verified final private cloud backup; separate untrained labels-v4 audit.

Phase 2 and Phase 3 remain pending. Faculty approval must not be presented as obtained. The corrected labels-v4 dataset has not produced the reported model scores.
