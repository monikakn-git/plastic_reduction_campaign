# Paired tiling pilot: YOLO11s

100-image paired validation pilot; not the full validation set. Same model weights; no training or label changes.

| Mode | mAP@50 | mAP@50–95 | Mean inference + processing time/image |
| --- | ---: | ---: | ---: |
| whole | 28.29% | 18.92% | 0.02 s |
| tiled | 32.42% | 21.75% | 0.09 s |

| Class | Objects | Whole AP@50 | Tiled AP@50 |
| --- | ---: | ---: | ---: |
| litter | 586 | 42.95% | 42.54% |
| pile | 4 | 30.55% | 29.48% |
| face_mask | 4 | 39.25% | 35.25% |
| trash_bin | 8 | 58.92% | 57.09% |
| plastic_bag | 70 | 18.37% | 26.40% |
| bottle | 32 | 11.50% | 17.52% |
| cup | 6 | 0.00% | 23.30% |
| rope | 12 | 37.21% | 42.84% |
| sachet | 24 | 39.67% | 43.77% |
| straw | 10 | 4.48% | 6.00% |

## Interpretation

Compare only the two columns on this same subset. Do not compare either score directly with the earlier 32.2% on 1,500 validation images. Rare-class support is especially low. No automatic inference-setting change or model promotion is performed. A full-validation comparison would require separate approval.

## Method and limitations

Overlapping 960-native-pixel windows, 20% overlap, each resized/letterboxed to 640. Whole-image predictions are included. Global class-aware NMS at IoU 0.7 removes duplicates, with at most 300 final predictions per image. Ultralytics matching and AP routines evaluate original full-image boxes at IoU 0.50–0.95.

Conceptual reference: [Ultralytics sliced-inference guide](https://docs.ultralytics.com/guides/sahi-tiled-inference/). This experiment implements tiling with existing installed dependencies; it does not install or claim to run SAHI.

- Several classes have only 4–8 ground-truth objects in this subset; class scores are highly uncertain.
- No confidence intervals or proof of full-dataset improvement; exploratory pilot only.
- Uses original unmodified labels, including unresolved ambiguity.
- Custom overlapping tiling plus class-aware NMS, not the SAHI library. Includes whole-image predictions to retain large-object context.
- Whole and tile inputs are square-letterboxed at 640; this pilot is not numerically interchangeable with the earlier full-validation loader.
- Timing includes inference and merge, excludes image decoding, and is not a controlled hardware benchmark.
