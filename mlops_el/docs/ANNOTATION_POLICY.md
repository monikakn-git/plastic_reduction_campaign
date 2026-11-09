# Annotation policy for the training-quality-v4 variant

This is a project review policy over the existing ten classes, not a claim of expert ground truth or a replacement for the dataset authors' guidance. Preserve the original dataset, splits, models and evaluation reports. Corrections use training pixels and source annotations only; prediction confidence and test errors are not evidence for a new label.

- **Bottle:** identifiable bottle body and neck/cap. Crushing alone does not justify changing its class. A generic cylindrical object without distinguishing cues remains unresolved.
- **Cup:** identifiable open drinking vessel or a visibly recognisable deformed cup. A visible footwear sole or a sealed beverage can with a pull-tab is not a cup. Ambiguous tubs, lids, jars, crushed printed containers and cartons stay unchanged pending adjudication.
- **Residual litter:** retain the existing `litter` ID for identifiable discarded objects outside the named categories, including footwear and beverage cans. Do not rename it “plastic” or claim that visual review establishes material composition.
- **Pile:** an inseparable accumulation of multiple discarded objects. Do not convert an individual object to pile merely because nearby objects exist. Coextensive pile/litter annotations need context review; this variant leaves the unresolved overlapping pair unchanged.
- **Face mask:** recognisable mask body, folds and/or ear loops. An obvious full-size waste barrel is instead `trash_bin`.
- **Trash bin:** waste receptacle. Existing source coverage includes functional bins; preserve that scope. Grouped bins and partially covered lids are flagged rather than automatically split or reboxed.
- **Plastic bag, rope, sachet, straw:** retain existing IDs and source scope. Difficult distinctions such as rope versus branch or straw versus wire require visible cues; do not infer a replacement class from thinness alone.
- **Boxes:** include the visible object extent and avoid unrelated objects. Occluded boundaries, inseparable groups and apparently erroneous very large boxes stay in a review queue until an exact defensible replacement can be determined.
- **Uncertainty:** keep the label and record the issue. Do not delete blurry or small targets to make evaluation easier. Do not automatically use model predictions as labels.

All v4 edits are class changes only, with original row, replacement row, image ID, reason and visual evidence retained. No validation/test label is corrected, no image or instance is removed, and the existing class taxonomy and split membership remain fixed. The inherited v3 orientation and JPEG repairs remain intact.

Evaluation after retraining must use validation for development. The previous 37.1% test mAP50 is a frozen historical result, not a target to tune against. Since that test set has already been evaluated, repeated model selection against it would contaminate its role; a genuinely new independent holdout is needed for strong confirmatory claims after further development. Remaining source-label ambiguity in unchanged holdouts can also limit interpretation. Improvements are hypotheses until a separate, reproducible training experiment measures them.
