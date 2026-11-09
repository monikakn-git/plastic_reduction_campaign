# YOLO11n baseline versus YOLO11s candidate

Same original 7,000/1,500 train/validation images and explicit training settings; only model initialization/capacity changes. No new test evaluation. No automatic model replacement.

| Metric | YOLO11n | YOLO11s | Change (percentage points) |
| --- | ---: | ---: | ---: |
| metrics/precision(B) | 43.36% | 39.55% | -3.81 |
| metrics/recall(B) | 31.13% | 32.47% | +1.34 |
| metrics/mAP50(B) | 26.78% | 32.24% | +5.46 |
| metrics/mAP50-95(B) | 15.91% | 19.51% | +3.61 |

| Class | Baseline AP@50 | Candidate AP@50 | Change (percentage points) |
| --- | ---: | ---: | ---: |
| litter | 37.67% | 41.78% | +4.12 |
| pile | 38.87% | 40.25% | +1.38 |
| face_mask | 31.59% | 43.42% | +11.82 |
| trash_bin | 54.24% | 53.13% | -1.11 |
| plastic_bag | 30.92% | 33.53% | +2.61 |
| bottle | 6.29% | 14.74% | +8.45 |
| cup | 3.84% | 5.09% | +1.25 |
| rope | 23.25% | 31.40% | +8.15 |
| sachet | 20.62% | 26.99% | +6.38 |
| straw | 20.48% | 32.07% | +11.59 |

## Decision requires review

Review both aggregate AP metrics and bottle/cup/plastic-bag performance before selecting a model. Consider detection errors and resource costs; higher aggregate AP alone does not resolve material ambiguity or unreliable classes. The baseline remains preserved.

Ambiguous source labels flagged during annotation review were not changed. The baseline test results have previously been viewed; this experiment uses validation only for comparison. No campus or campaign effectiveness claim is established.
