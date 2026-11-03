# EcoPulse AI — Model Evaluation Report

**Model:** `yolo11n_best.pt`  
**Dataset Configuration:** `D:/mlops/data/waste10/data.yaml`  
**Evaluation Split:** `test`  
**Quality Gate Verdict:** **PASSED_APPROVED_FOR_DEPLOYMENT**  

## 1. Overall Performance Metrics
| Metric | Observed Value | Quality Gate Target | Status |
| :--- | :--- | :--- | :--- |
| **Precision** | 0.6436 | >= 0.20 | PASS |
| **Recall** | 0.6051 | >= 0.20 | PASS |
| **mAP@50** | 0.5657 | >= 0.2 | PASS |
| **mAP@50-95** | 0.4231 | Informational | OK |
| **Inference Latency** | 146.26 ms/frame | <= 500.0 ms | PASS |
| **Model File Size** | 5.25 MB | Edge Deployable | PASS |

## 2. Model Detection Classes (10)
- **Class 0:** `litter`
- **Class 1:** `pile`
- **Class 2:** `face_mask`
- **Class 3:** `trash_bin`
- **Class 4:** `plastic_bag`
- **Class 5:** `bottle`
- **Class 6:** `cup`
- **Class 7:** `rope`
- **Class 8:** `sachet`
- **Class 9:** `straw`

## 3. Operational Deployment Assessment
- **Edge Feasibility:** Model file size of 5.25 MB permits packaging into Docker containers and deployment on edge devices.
- **Latency Budget:** Inference latency of 146.26 ms/frame supports real-time batch processing and responsive API endpoints.
