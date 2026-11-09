# Baseline validation error analysis

The original baseline, data splits, and test results are preserved. No retraining has been performed.

## Evidence

| Class | Train objects | Val AP@50 | Val recall | Median val width × height at 640 | Val short side <8px |
| --- | ---: | ---: | ---: | --- | ---: |
| litter | 29639 | 37.7% | 37.8% | 15.0 × 10.0 px | 32.6% |
| pile | 761 | 38.9% | 52.6% | 232.2 × 129.2 px | 4.4% |
| face_mask | 466 | 31.6% | 47.0% | 29.6 × 17.5 px | 4.5% |
| trash_bin | 429 | 54.2% | 55.9% | 62.5 × 87.0 px | 1.1% |
| plastic_bag | 4285 | 30.9% | 31.9% | 24.0 × 17.0 px | 6.4% |
| bottle | 910 | 6.3% | 3.5% | 17.5 × 11.0 px | 28.0% |
| cup | 488 | 3.8% | 0.0% | 17.5 × 15.1 px | 11.3% |
| rope | 667 | 23.3% | 23.7% | 45.7 × 24.5 px | 1.4% |
| sachet | 2023 | 20.6% | 35.5% | 18.6 × 12.2 px | 20.9% |
| straw | 1097 | 20.5% | 23.5% | 22.5 × 12.2 px | 32.9% |

Litter dominates the training annotations. Cups and bottles have especially weak validation AP. Object size and imbalance are plausible contributors; this analysis does not establish causation.

Validation prediction previews show crowded scenes and small objects. These few examples are illustrative, not a representative audit.

## Controlled resolution comparison

Same weights, validation split, batch size, confidence floor and NMS IoU. Only image resolution changes. No test-set experiment.

| Resolution | mAP@50 | mAP@50–95 |
| --- | ---: | ---: |
| 640 | 26.78% | 15.91% |
| 960 | 26.56% | 15.15% |

Validation AP@50 change: -0.21 percentage points. This measures an inference setting, not new learning.

| Class | AP@50 at 640 | AP@50 at 960 | Change (percentage points) |
| --- | ---: | ---: | ---: |
| litter | 37.67% | 37.91% | +0.25 |
| pile | 38.87% | 24.47% | -14.39 |
| face_mask | 31.59% | 32.21% | +0.62 |
| trash_bin | 54.24% | 48.56% | -5.69 |
| plastic_bag | 30.92% | 31.12% | +0.20 |
| bottle | 6.29% | 9.50% | +3.21 |
| cup | 3.84% | 3.67% | -0.17 |
| rope | 23.25% | 29.28% | +6.03 |
| sachet | 20.62% | 23.50% | +2.88 |
| straw | 20.48% | 25.40% | +4.92 |

### Decision

Higher inference resolution does not improve overall validation AP@50 here. Keep 640 as the reference; do not adopt 960 solely because it is larger.

The next proposed training comparison is YOLO11s versus the YOLO11n baseline, keeping the same 640 input size, data splits, 50-epoch budget, seed and other explicit settings. This tests whether extra model capacity helps; improvement is not guaranteed. Run a short GPU/memory benchmark first, record its engineering-only status, and preserve baseline weights. No candidate training has started.

## Next steps

1. Review a systematic validation sample covering misses, false positives, crowded scenes and weak classes before changing annotations.
2. Preserve all 10,000 originals, baseline weights, and current data versions. Train only on training data.
3. If training changes are justified, run one separately named candidate with its own configuration, MLflow run and weights; select using validation performance and resource costs.
4. Do not repeatedly evaluate the existing test set during tuning. Since its baseline results have already been viewed, document that exposure; obtain a fresh independent holdout for stronger final performance claims if needed.
5. Faculty approval remains pending. These public-street images do not establish plastic material identity or campus campaign outcomes.
