# Preflight for one additional training experiment

Status: checks complete; cloud training configuration is a proposal, not an executed run. No full training or spending has started. Original datasets and baseline outputs are unchanged; a separate derived preprocessing variant has been prepared.

## Actual checks and trials

- Recomputed SHA-256 for all 10,000 original images against the split/acquisition identities. No byte mismatch or missing source image was found.
- Compared every one of the 8,500 training/validation label files to the original annotations and class mapping. No conversion mismatch or identical duplicate label row was found. Positive box dimensions and coordinate conversion are checked by the existing preparation routine.
- Confirmed that inferred source groups and exact-image hashes do not cross splits. Cross-recording near-duplicates are not ruled out by this filename/hash check.
- Checked encoded image dimensions and EXIF orientation headers in training/validation. Encoded dimensions match metadata. There are 37 undefined orientation tags (0), which do not request a rotation in the current decoder, and four training images with tag 3 (180-degree rotation).
- Inspected those four rotation cases using encoded pixels and the actual OpenCV decoder. In image 528, the small litter annotation lands on the wrong side after decoding. Five annotation coordinate transforms for four training images were prepared in a separate `plitterstreet-orientation-v2` derived version; the original annotations and prepared baseline remain unchanged. Details are in `orientation_variant.json` and `orientation_variant_review.jpg`. Broad pile boxes in the remaining images make visual alignment less decisive. Keep the original files and baseline versions.
- Reviewed the learning curves. YOLO11s' best logged AP@50–95 was at epoch 48. AP@50 increased modestly from 31.15% at epoch 40 to 31.97% at epoch 50, while training classification loss fell and validation classification loss changed from 1.777 to 1.801. This supports considering a longer capped run with early stopping, but does not show that more epochs alone will reach 50%.
- Evaluated the saved YOLO11s weights at 960 using the same standard validation settings as its earlier 640 evaluation: **30.53% AP@50 / 17.41% AP@50–95**, versus **32.24% / 19.51%** at 640. Bottle AP@50 increased to 19.02%, but overall quality fell. This was prediction-time evaluation, not training at 960.
- Prior evidence remains: the full fixed tiling comparison reduced aggregate performance, and the label audit contains unresolved human interpretation flags. Neither is silently accepted as a model improvement or a correction.

## Suggested next experiment

**Conditional recommendation: YOLO11m on a cloud CUDA GPU, training at 960 for at most 100 epochs, with early stopping patience 20.** Start a short engineering benchmark on the actual cloud GPU before approving the long run. If memory or runtime is unsuitable, change the proposal explicitly; do not silently reduce the dataset or resolution.

The [official model family documentation](https://docs.ultralytics.com/models/yolo11/) lists the medium model as a larger capacity option than small. Its general benchmark score is not a prediction of performance on this waste dataset. Training at 960 may let the model adapt to finer object detail, but the failed prediction-only resolution test is a reason for caution, not evidence that higher-resolution training will work.

`configs/yolo11m_cloud_proposed.yaml` records the proposal. It starts from the same pretrained family, keeps all ten classes and the full 7,000/1,500 train/validation split, preserves existing augmentation/optimizer settings, and proposes CUDA, AMP and batch 4. Batch 4 is provisional until measured on the allocated GPU. Device, batch, precision, model capacity, resolution and epoch budget differ from the baseline: this is a combined improvement candidate, not a clean attribution of gains to one factor.

Do not promise 50% mAP@50. More capacity, more epochs and a cloud GPU do not resolve ambiguous labels. An independently reviewed dataset improvement would be needed for stronger semantic label assurance. Do not fix only the validation examples to improve scores or remove hard classes/images.

## Changes to resolve before freezing a cloud run

1. The separate derived version has been created and versioned with DVC. Its convention is decoder-aligned coordinates for the four flagged training images; review that recorded variant before freezing the cloud inputs. Raw originals remain unchanged. The proposed 180-degree box transform is `(W-x-w, H-y-h, w, h)` when boxes refer to encoded pixels but decoding rotates them. Verify the adopted convention visually and with geometry tests.
2. Keep unresolved class relabelling proposals unapplied. A user's uncertainty is not approval; the earlier review decisions are recorded as unsure.
3. Use a fixed standard-validation protocol for comparison: document rectangular padding, multi-label NMS, input resolution, confidence floor, IoU and max detections. Prediction-mode and validator scores are not interchangeable.
4. Freeze dataset DVC identities, exact cloud config, code commit and selected initial weight hash. Keep the current YOLO11n and YOLO11s artifacts.

## Cloud recommendation and safeguards

Cloud CUDA training is reasonable for the next experiment; it avoids another prolonged local training job. Choose the provider only after considering the available GPU and account/budget preference. No provider account or available GPU was inspected in this preflight, and no cloud runtime has been started.

- A GPU with 16 GB or more VRAM is a sensible initial planning target, subject to the benchmark. Profile actual batch memory and time; do not estimate a completion time from the Mac run.
- Both [Kaggle notebooks](https://www.kaggle.com/docs/notebooks) and [Colab](https://research.google.com/colaboratory/intl/en-GB/faq.html) support GPU workflows. Colab resources and session limits vary; its free runtime can be at most 12 hours, and availability is not guaranteed.
- Stage training images on the runtime's local disk. Use persistent storage for checkpoints and experiment records. Keep a verified persistent copy of last/best checkpoints after each completed epoch, written atomically where supported, so a provider session ending does not erase the whole experiment. Do not keep 100 redundant full checkpoint copies.
- The existing Mac lock includes its Torch environment. On CUDA, pin compatible application packages, retain a functioning CUDA Torch build, run dependency checks, and record the exact resolved cloud environment/GPU/driver versions. Cross-device reproducibility has numerical limits.
- Preserve MLflow records and DVC provenance. Stage only the train/validation subset for cloud computation; the 1,500 test images can remain local. This does not change the project's 10,000-image dataset or split.
- A short GPU/optimizer/checkpoint benchmark is necessary. It is engineering evidence, not a third full model experiment or a quality claim. Cloud notebook access, dataset transfer and persistent storage setup must be concrete before launch.

## Limits of preflight

These are targeted checks and inference trials, not every possible experiment. Learning-rate, augmentation, longer training and model-size effects cannot be settled without actual training. No YOLO11m quality result, CUDA throughput, cloud bill or 50% score has been assumed. The next full run remains pending cloud account/runtime selection, a short CUDA benchmark and final review. Faculty dataset approval remains pending separately.
