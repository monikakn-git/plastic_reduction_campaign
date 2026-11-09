# Phase 1 current status

Technical review package ready as of 7 November 2026. Faculty dataset approval remains pending. No training is running.

| Model | Training data | Input | Epochs | Val mAP50 | Val mAP50–95 | Test mAP50 |
|---|---|---:|---|---:|---:|---:|
| YOLO11n | Original prepared | 640 | 50 | 26.8% | 15.9% | 26.2% |
| YOLO11s | Original prepared | 640 | 50 | 32.2% | 19.5% | Not evaluated |
| YOLO11m | labels-v3 | 960 | 86 (best 66) | 39.4% | 24.7% | 37.1% |

Final YOLO11m test: precision 46.1%, recall 39.6%, mAP50 37.1%, mAP50–95 22.7%. Labels-v3 produced this model. Labels-v4 contains 30 later class corrections and is untrained.

See [final report](PHASE1_FINAL_REPORT.md), [college checklist](PHASE1_CHECKLIST.md), [review guide](../../docs/PHASE1_REVIEW.md), and phase1_final_summary.json. Local DVC verification is clean. Off-device DVC remote and faculty approval remain pending. Private cloud checkpoints are independently verified. Phase 2 has not started.
