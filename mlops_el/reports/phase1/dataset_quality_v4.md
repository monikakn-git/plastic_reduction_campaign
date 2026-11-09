# Training dataset quality improvements — labels-v4

A separate derived dataset was created from labels-v3. **30 class corrections in 28 training images** were applied following visual review. No new model has been trained on v4, and no performance improvement is measured or promised. The completed YOLO11m model and its 37.1% test mAP50 remain unchanged.

## Scope and evidence

- Structurally scanned all 7,000 training label files: no malformed/out-of-bounds boxes or exact duplicate rows. One cross-class box pair with IoU 0.966 was flagged and retained pending adjudication.
- Visually screened **670 distinct training annotations in 595 images**: every one of the 488 training cup annotations, 100 bottle annotations, ten samples from each other class, and two overlapping annotations. Contact sheets span object-size quantiles; this is not a full manual review of every object in every image.
- Rechecked 79 selected annotations in full-image context and enlarged detail. Images are resized for review without generative enhancement. AI judgments are not expert ground truth.
- Applied 25 footwear/sole-to-litter corrections, three identifiable beverage-can-to-litter corrections, one face-mask-to-trash-bin correction, and one litter-to-pile correction. The patch manifest records original/replacement YOLO rows and image identity for every edit.
- **148 annotations** are in the unresolved review queue. These include ambiguous vessels, lids, crushed objects, grouped instances and very large boxes. They were not deleted or automatically relabelled. Contact-sheet “no clear error” assessments do not establish exhaustive annotation correctness or completeness.

## Preservation and verification

All 10,000 images are retained without copying their pixels, through hardlinks to immutable parent images. All 60,826 annotation instances, coordinates, class IDs/taxonomy, and 7,000/1,500/1,500 split membership are retained, with only the declared class assignments changed. All 10,000 parent label files were checked unchanged; all 3,000 validation/test label files are byte-identical to v3. Earlier EXIF and JPEG fixes are inherited.

Training class count changes: litter +27, pile +1, face_mask −1, trash_bin +1, bottle −4, cup −24. Other classes are unchanged. This leaves 464 cup and 907 bottle instances versus 29,665 residual litter instances. Strong imbalance and limited independent scenes remain; correcting a small number of labels is not a basis for forecasting a large mAP gain.

Four focused tests pass, including unsafe patch/holdout rejection, stale-parent rejection, malformed labels, and complete dataset comparison of declared changes, preserved coordinates, instance counts and unchanged holdouts. DVC records the new dataset separately. The original training/evaluation records are preserved.

## Interpretation and next development

Use the written [annotation policy](../../docs/ANNOTATION_POLICY.md). Cups and bottles remain object identities; neither labels nor images establish plastic material certainty. Clear errors were corrected conservatively, while uncertain cans, flattened printed cups and lids were retained for adjudication. Holdout source-label ambiguities remain unchanged and may limit interpretation of subsequent results.

Before a long training run, resolve the highest-confidence remaining box problems, extend the review to additional training bottle/residual labels, and seek diverse correctly labelled examples of the rare classes. Do not simply duplicate existing frames or delete hard examples. A new or fine-tuned model using v4 must be a separate MLflow run; do not resume or overwrite the completed labels-v3 run. Use validation for development and comparisons. The previously evaluated test set is a historical assessment; do not repeatedly select models against its score. A new independent holdout would be needed for a strong confirmatory assessment after further development.

## Reproduction

1. Run `scripts/audit_training_quality.py` into a fresh output directory to rebuild the structural scan and quantile contact sheets. Use `--remaining-cups reports/phase1/training_quality_v4/manifest.json` with a different `--output` directory to rebuild the remaining-cup sample.
2. `scripts/render_training_review.py --audit <audit-directory>` rebuilds the context sheets from the retained `detail_selection.json`.
3. Visual judgments are retained in `training_quality_v4/visual_decisions.json`; the script does not pretend to generate those decisions.
4. `scripts/prepare_quality_variant.py --manifest configs/label_corrections_v4.json` rebuilds the separate version, refusing existing output or changed source rows.
5. Run `pytest tests/test_quality_variant.py tests/test_label_refinement.py -q`.

Records: `training_quality_v4/manifest.json`, `remaining_cups/manifest.json`, `visual_decisions.json`, `variant_verification.json`, `configs/label_corrections_v4.json`, and `data/prepared/plitterstreet-labels-v4.dvc`.
