# YOLO11s short engineering benchmark

One epoch on 256 training images at 640 pixels, batch 2, native MPS. A disjoint 32-image subset of the original training split exercises validation code; these smoke scores are not quality evidence. Official validation and test splits were not used.

| Measurement | Result |
| --- | ---: |
| benchmark_total_seconds | 36.278 |
| training_epoch_seconds | 29.493 |
| measured_batches | 128.000 |
| warmup_batches_excluded | 10.000 |
| steady_batch_median_seconds | 0.211 |
| steady_batch_p90_seconds | 0.229 |
| steady_images_per_second | 9.332 |
| peak_mps_tensor_GiB | 0.198 |
| peak_mps_driver_GiB | 1.299 |
| peak_process_rss_GiB | 0.751 |
| full_epoch_projection_minutes | 12.501 |
| fifty_epoch_training_only_projection_hours | 10.418 |

## Interpretation

The short benchmark completed without a reported out-of-memory exception. This supports feasibility for a controlled experiment, but does not prove full-run stability or improvement over the baseline.

The full-run projection is training-only and approximate. Budget extra time for validation, checkpointing, data loading variability and MacBook Air thermal throttling. A full run remains subject to separate user approval.

CPU process RSS and MPS figures overlap on unified memory; do not sum them. See the JSON report for batch-level measurements and limitations.

Baseline weights, annotations, split manifest and baseline params.yaml hashes were verified unchanged. No candidate model has been selected.

Measurement clarification: MPS peak fields are maxima observed at batch-end callbacks, not guaranteed transient allocation peaks; process RSS is the OS high-water mark.

Scheduling estimate: allow approximately 12–16 hours for a full 50-epoch local run, rather than treating the 10.4-hour training-only extrapolation as a completion promise. Early stopping may shorten it.
