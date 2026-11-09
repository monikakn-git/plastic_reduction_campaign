# Bottle and cup annotation audit: 100 targets

Completed an AI-assisted visual inspection of 100 labelled targets in 100 distinct images: 25 bottles and 25 cups from each of training and validation. No test images or model predictions were used. No labels, images, splits, preprocessing settings or model weights were changed.

## What was verified

- All 100 sampled boxes have positive dimensions and lie within their images.
- All 100 sampled source labels match their prepared YOLO class and normalized coordinates within 1e-6. This checks conversion, not semantic truth.
- Hashes of sampled images and label files, full annotation JSON, splits, baseline/candidate weights, baseline configuration and protected evaluation reports remained unchanged.

## Visual findings

| Sample | Label plausible | Cannot confidently judge | Suspected class mismatch |
| --- | ---: | ---: | ---: |
| bottle / train | 14 | 9 | 2 |
| bottle / val | 10 | 13 | 2 |
| cup / train | 10 | 14 | 1 |
| cup / val | 10 | 10 | 5 |
| **Total** | **44** | **46** | **10** |

**These are reviewer judgments, not confirmed error counts.** Uncertain cases are frequently tiny, motion-blurred, crushed, dirty or obscured. The size-stratified, group-balanced sample is not representative enough to estimate a dataset-wide error rate or prove why a model scored poorly.

Ten targets have competing visual interpretations (for example carton-like items labelled cup, or flat packet/disc-like items labelled bottle). Four additional box/instance flags were recorded: one possible truncated boundary and three possible multi-object boxes. These require human adjudication; none was automatically corrected.

### Specific follow-up evidence

- `bottle-train-large-01`: the target box appears to exclude part of a blue cap/neck. Review tightness at source resolution.
- `bottle-train-large-09`, image 14677: several visible bottles share one bottle box; another nearby bottle-like item carries the residual `litter` label. This is a class-specificity/grouping consistency flag, not an unlabelled-object discovery. See `context_detail_14677.jpg` and `context_findings.json`.
- `cup-val-small-04` and `cup-val-medium-03`: carton/packet-like shapes carrying cup labels. These are suspected mismatches, pending a human decision.
- `cup-val-large-04` and `cup-val-large-06`: crushed can/container-like objects with cup labels. Ambiguous shape does not establish material.
- Image 5720: both visible bottles already have individual boxes; an extra context check prevented incorrectly calling the neighbouring bottle unlabelled.

**Missing annotations:** no missing-object correction was established. All source boxes were shown in context previews, with magnified all-label context checks in two selected scenes. This is not exhaustive object-by-object completeness verification; small previews and unknown discarded/functional status limit that conclusion.

## Class definitions and source evidence

The authors provide broad annotation guidelines and acknowledge interpretation difficulty. They do not resolve the specific cup/lid/carton boundary cases in the documentation reviewed. Source links, relevant section and findings are recorded in [source_definitions.json](source_definitions.json).
The source JSON residual class is named `litter`; the paper uses `other plastic`. No taxonomy remapping was applied during this audit. Neither a source class label nor visual inspection proves that every object is plastic.

## Sampling and limits

Seed 2026; 8 small, 8 medium and 9 large targets per class/split, based on within-pool projected short-side tertiles at 640. Seeded ordering and a least-used inferred-group preference increase scene coverage. All images are distinct. Validation groups necessarily repeat: cup validation has only 14 available inferred groups. Grouping is a filename proxy, not verified geographic independence.
This audit contains 5 targets from the earlier 24-target review; it is not 100 additional independent targets. Original, magnified detail and resized-detail views were inspected. Contact sheets contain five examples each, with red target boxes and grey boxes for other source annotations. The model-input views illustrate aspect-preserving resizing, not every training augmentation.

## Recommendation before another training run

Review the 14 high-priority entries in `adjudication_queue.json` with a human reviewer and a written bottle/cup inclusion policy. The queue contains ten suspected class cases and four box/instance cases; it is not an approved correction file. Keep the other ambiguous examples available for discussion. Do not infer replacement classes or delete hard examples simply to raise validation scores.
If corrections are approved later, create a versioned correction manifest over source image/annotation IDs and rebuild a new derived DVC version, retaining the originals and current baseline. Validation corrections must follow the same independent policy; scores across changed labels need explicit qualification. Do not modify the existing test labels or use test predictions for tuning.
The audit found no conversion defect in the sampled labels, so it does not justify rewriting the folder structure or blindly changing preprocessing. No new training or automatic model promotion was performed. Faculty approval remains pending.

## Reproduction and records

Run `.venv/bin/python scripts/audit_annotations.py` to regenerate the fixed sample/contact sheets, then `.venv/bin/python scripts/summarize_annotation_audit.py` to validate and summarize the retained visual judgments. Visual judgments require inspection; they are not generated by the preparation script.

- `sample_manifest.json`: selection, exact IDs, coordinates, groups and protected hashes.
- `visual_findings.json` and `review_table.csv`: all 100 visual observations.
- `technical_checks.json`: source-to-prepared label checks.
- `adjudication_queue.json`: pending human decisions.
- `context_findings.json` and two context-detail images: additional label-coverage/grouping evidence.
- `summary.json`: actual counts and preservation status.

## Contact sheets

| Class / split | Five targets per page |
| --- | --- |
| bottle / train | [1](bottle_train_01.jpg) · [2](bottle_train_02.jpg) · [3](bottle_train_03.jpg) · [4](bottle_train_04.jpg) · [5](bottle_train_05.jpg) |
| bottle / val | [1](bottle_val_01.jpg) · [2](bottle_val_02.jpg) · [3](bottle_val_03.jpg) · [4](bottle_val_04.jpg) · [5](bottle_val_05.jpg) |
| cup / train | [1](cup_train_01.jpg) · [2](cup_train_02.jpg) · [3](cup_train_03.jpg) · [4](cup_train_04.jpg) · [5](cup_train_05.jpg) |
| cup / val | [1](cup_val_01.jpg) · [2](cup_val_02.jpg) · [3](cup_val_03.jpg) · [4](cup_val_04.jpg) · [5](cup_val_05.jpg) |
