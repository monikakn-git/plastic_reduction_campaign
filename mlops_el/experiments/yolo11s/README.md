# Approved YOLO11s comparison

This separate candidate preserves the YOLO11n baseline. `configs/yolo11s_candidate.yaml` matches the baseline's explicit training settings except for the pretrained model path: YOLO11s replaces YOLO11n. It uses the existing 7,000 training images, 640 input size, batch 2, seed 42, up to 50 epochs and patience 12 on native MPS.

`reports/phase1/yolo11s_candidate_plan.json` records approval scope and hashes of protected baseline evidence, labels and splits. The training and comparison scripts verify those hashes. Only validation is used for comparison. No test evaluation or automatic model promotion is authorized by this workflow.

## Execute

With the project's exact environment and current DVC dataset checked out:

```sh
/usr/bin/caffeinate -i /bin/sh scripts/run_yolo11s_candidate.sh
```

The workflow executes the two candidate stages individually so upstream preparation does not rebuild the existing dataset. DVC records dependency hashes, candidate weights, and the validation reports in `experiments/yolo11s/dvc.lock`. MLflow records separate training and validation runs.

Training outputs: `models/plitterstreet-yolo11s-candidate.pt` and `reports/phase1/yolo11s_candidate_training.json`. Comparison outputs: `reports/phase1/yolo11s_candidate_validation.json` and `reports/phase1/yolo11s_validation_comparison.md`. Baseline outputs are not overwritten. Generated candidate results exist only after their actual stages complete.

Checkpoint `runs/<candidate-run-name>/weights/last.pt` is saved each completed epoch. In case of interruption, inspect `.local/yolo11s-candidate-active.json` for the original run identity. Resume explicitly rather than starting over:

```sh
.venv/bin/python scripts/train_candidate.py --resume runs/<candidate-run-name>/weights/last.pt --run-id <original-mlflow-run-id>
.venv/bin/dvc commit experiments/yolo11s/dvc.yaml:train_candidate --force
.venv/bin/dvc repro --single-item experiments/yolo11s/dvc.yaml:compare_candidate
```

Do not launch a duplicate training process. The short engineering benchmark estimated about 10.4 hours of training-only work; 12–16 hours is an approximate local scheduling allowance, with hardware temperature, application load, validation and early stopping affecting elapsed time. This is not a guarantee of improvement or completion time. Source label ambiguity remains documented in the annotation review.
