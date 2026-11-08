# EcoPulse AI – Model Risk & Quality Gate Audit Report

**Generated:** 2026-11-08 18:55:22 UTC
**Overall Assessment:** `APPROVED_FOR_PRODUCTION`

---

## 1. Executive Summary

This report assesses the operational readiness, fairness, latency budget, and robustness
of candidate waste detection models against enterprise Model Risk Management (MRM) policies.

### Model: YOLO11N (`yolo11n_best.pt`)
- **Policy Verdict:** `APPROVED_FOR_PRODUCTION`
- **Model Size:** 5.25 MB (Edge Deployable: YES)
- **Detection Retention under Perturbation:** 100.0%

| Category | Check / Metric | Measured Value | Threshold | Severity | Status |
|---|---|---|---|---|:---:|
| Performance Gate | mAP50 | 0.2721 | >= 0.2 | CRITICAL | PASS |
| Performance Gate | Precision | 0.369 | >= 0.25 | HIGH | PASS |
| Performance Gate | Recall | 0.3153 | >= 0.25 | HIGH | PASS |
| Operational SLA | Mean Latency (CPU) | 204.11 ms | <= 500.0 ms | HIGH | PASS |
| Operational SLA | P95 Tail Latency | 320.23 ms | <= 750.0 ms | MEDIUM | PASS |
| Edge Deployment Feasibility | Model Size | 5.25 MB | <= 25.0 MB | CRITICAL | PASS |
| Robustness / Perturbation Risk | Detection Retention (Low Light & Blur) | 100.0% | >= 40% | MEDIUM | PASS |
| Deepchecks Vision Integrity | Data integrity suite | PASSED | all configured integrity checks pass | HIGH | PASS |

## 2. Model Governance Decision
- **Champion Model:** `yolo11n_best.pt` (higher inference efficiency and architecture scalability)
- **Challenger Model:** `yolov8n_best.pt` (stable baseline)
- **Deployment Recommendation:** Register and promote only after every required gate passes.
