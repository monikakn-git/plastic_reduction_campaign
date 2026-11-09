# Step 1: bottle and cup annotation inspection

## Scope and method

Reviewed 24 annotated objects in 24 images: six bottles and six cups from training, and six of each from validation. For each class/split, examples were selected with seed 42 from three size strata, with two objects per stratum and distinct inferred source groups within that class/split. There are 22 distinct inferred groups overall; groups shared across classes remain in the same split. No test images or predictions were inspected.

The four contact sheets show full context, a magnified original crop, and a magnified crop after longest-side resizing to 640. Red boxes indicate the selected source annotation, not all annotations in the image. The 640 view illustrates resizing; it does not reproduce every training augmentation. Small, medium and large refer to within-class/split size tertiles, not standard object-size categories.

Selection, image identities, boxes and source hashes are recorded in `sample_manifest.json`. Individual visual observations are in `visual_findings.json`. Run `python scripts/review_annotations.py` using the project environment to regenerate the sample and sheets.

## Findings

- **Box placement:** the sampled boxes generally cover a visible target item. This limited inspection did not reveal a consistent gross coordinate shift. It cannot certify all labels, precise box tightness, annotation completeness, or the rest of the dataset.
- **Small and degraded targets:** selected target short sides range from 5.5 to 43.5 pixels at 640 resolution. Several items are already blurred or crushed in the original image; loss of detail is not caused solely by model resizing. Enlarging a crop does not recover missing information.
- **Class ambiguity:** 11 sampled labels look plausible, while 13 could not be confidently confirmed. These are subjective inspection categories, not confirmed errors or a dataset-wide error rate. Several cup examples look like lids, cans or other containers; some bottle examples are flattened packaging or cylindrical objects with no clear neck. A class-definition check and expert adjudication are required before changing labels.
- **Uneven training support:** existing preparation records contain 910 bottle and 488 cup objects, compared with 29,639 generic litter objects in training. This imbalance and difficult appearances are plausible contributors to weak results; this inspection does not prove a causal explanation.
- **Material limits:** object appearance and the source bottle/cup label do not establish that an item is plastic. Predictions should not be described as verified material identification.

Examples needing attention include `bottle-val-medium-1` (flattened rectangular object), `bottle-val-medium-2` (can-like crushed container), `cup-val-medium-2` (shallow round container/lid), and `cup-val-large-1` / `cup-val-large-2` (crushed container-like objects). The manifest maps every review ID to its original image and annotation.

The authors' [repository](https://github.com/gicait/pLitter) describes the dataset as COCO-format street litter annotations and identifies its annotation tool. The README reviewed here does not settle these specific cup-versus-container ambiguities. The visual flags above are our inspection judgments, not statements from the dataset authors.

## Decision and next step

Keep all source labels and all 10,000 images unchanged. Do not automatically remove uncertain examples or relabel validation data based on model performance. If corrections are later approved, document class definitions, adjudicate consistently, and create a new version rather than overwriting the baseline data.

Step 2 would be a short YOLO11s GPU/memory benchmark to determine whether a controlled capacity comparison is practical on this Mac. It is an engineering check, not a promise that a larger model will fix ambiguous labels. A full training run would require separate approval after that benchmark, and any final quality claim must retain these data limitations.

Only Step 1 was performed. No retraining, dataset modification, additional test evaluation, or model selection occurred.

## Review artifacts

- [Bottle training examples](bottle_train.jpg)
- [Bottle validation examples](bottle_val.jpg)
- [Cup training examples](cup_train.jpg)
- [Cup validation examples](cup_val.jpg)
- [Sample manifest](sample_manifest.json)
- [Per-example findings](visual_findings.json)
