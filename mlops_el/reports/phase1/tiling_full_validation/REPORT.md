# Full-validation tiling comparison: YOLO11s

1500-image paired full-validation comparison. Same model weights; no training or label changes.

| Mode | mAP@50 | mAP@50–95 | Mean inference + processing time/image |
| --- | ---: | ---: | ---: |
| whole | 28.86% | 17.66% | 0.02 s |
| tiled | 27.29% | 16.88% | 0.10 s |

| Class | Objects | Whole AP@50 | Tiled AP@50 |
| --- | ---: | ---: | ---: |
| litter | 7093 | 40.67% | 38.90% |
| pile | 137 | 38.06% | 20.75% |
| face_mask | 66 | 39.31% | 31.30% |
| trash_bin | 93 | 53.59% | 46.38% |
| plastic_bag | 1117 | 24.31% | 27.46% |
| bottle | 371 | 8.16% | 13.87% |
| cup | 115 | 0.10% | 0.85% |
| rope | 139 | 30.69% | 32.19% |
| sachet | 330 | 22.99% | 26.00% |
| straw | 170 | 30.68% | 35.23% |

## Interpretation

Compare the two methods on this same full validation split and matching inference/evaluation settings. The earlier 32.2% used a different validation loader and multi-label NMS. This paired comparison uses single-label prediction postprocessing, so the scores are retained as separate protocols. No automatic inference-setting change or model promotion is performed, and the test set has not been evaluated again.

## Method and limitations

Overlapping 960-native-pixel windows, 20% overlap, each resized/letterboxed to 640. Whole-image predictions are included. Global class-aware NMS at IoU 0.7 removes duplicates, with at most 300 final predictions per image. Ultralytics matching and AP routines evaluate original full-image boxes at IoU 0.50–0.95.

Conceptual reference: [Ultralytics sliced-inference guide](https://docs.ultralytics.com/guides/sahi-tiled-inference/). This experiment implements tiling with existing installed dependencies; it does not install or claim to run SAHI.

- Uses original unmodified labels, including unresolved ambiguity.
- Custom overlapping tiling plus class-aware NMS, not the SAHI library. Includes whole-image predictions to retain large-object context.
- Whole and tile inputs are square-letterboxed at 640; this comparison is not numerically interchangeable with the earlier validation loader. Compare the paired methods here.
- Timing includes inference and merge, excludes image decoding, and is not a controlled hardware benchmark.
- Validation data has been used for method selection; this is not independent test performance.
- No confidence intervals or campus generalization claims. Labels remain ambiguous in some reviewed cases.

## Evaluator verification

The standard Ultralytics validator at square padding and batch 1 scored 31.91% AP@50 with its default multi-label NMS. With only NMS changed to single-label, it scored 28.78%, close to this experiment's 28.86% whole-image result (difference 0.07 percentage points). This confirms that default prediction and standard validation do not use identical postprocessing. It does not imply that the saved model deteriorated. The original standard-validation score of 32.24% remains preserved. See the two library-reference JSON files and `scripts/verify_tiling_evaluator.py`.

## Decision

Do not adopt this fixed tiling setup. It lowered overall AP@50 by 1.56 percentage points and AP@50–95 by 0.78 points, despite improving six class AP@50 scores. Predictions also took longer. The pilot improvement did not carry over to the full validation split. Bottle and cup performance remains weak under this prediction protocol. These results reject this specific configuration; they do not establish that every possible tiling approach would fail. No inference-setting change, label correction, new training, test evaluation or model promotion was performed.

Reproduce the approved experiment with `.venv/bin/python scripts/experiment_tiling.py --full-validation --output-dir reports/phase1/tiling_validation_repeat` in a fresh output location (the command refuses to overwrite completed results). The existing results contain exact sample identities, paired predictions, PR curves and verification records.
