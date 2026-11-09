# Phase 1 — Development and Reproducibility

This is the implementation guide for the college's Phase 1 rubric only. The API, registry, containers, deployment, drift monitoring, and campaign application belong to later phases.

## Current review package

Phase 1 technical evidence is ready as of 7 November 2026. Faculty dataset approval remains pending. The selected evaluated YOLO11m model has 37.1% test mAP50 and 22.7% test mAP50–95. Its labels-v3 experiment completed at epoch 86, with the best checkpoint at epoch 66. The later labels-v4 correction dataset is untrained.

Use [PHASE1_REVIEW.md](PHASE1_REVIEW.md) for safe demonstration commands and the [final report](../reports/phase1/PHASE1_FINAL_REPORT.md) for actual outcomes. The commands below are implementation/reproduction instructions and can launch expensive training. They are not required just to inspect completed results.

## Dataset and approval

We use a reproducible **10,000-image original-resolution subset of pLitterStreet**, selected from the official 13,064-image, 10-class release at https://zenodo.org/records/8288500. This is 10,000 total images across training, validation, and test; augmented copies do not count.

Source checksums, selection seed, expected storage, and archive provenance are recorded in `data/raw/plitterstreet/source.json`. Each selected image is checked against its ZIP CRC32 and recorded with SHA-256, dimensions, and source-video grouping. The full image ZIP is not retained. Only member-level integrity is claimed for selective extraction.

All selected source images are preserved without resizing or recompression. **Faculty approval is pending.** The official record declares CC BY 4.0; attribution is required. See [DATASET_APPROVAL.md](DATASET_APPROVAL.md).

## Environment

The complete verified sequence can be run with `sh scripts/run_phase1.sh`. It refuses incomplete acquisition, versions the actual data, generates reports, runs checks, and then trains/evaluates. Full-data training on the 8 GB M3 Air is a long-running task.

Use Python 3.12 and a dedicated environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --no-cache-dir -r requirements-lock.txt
.venv/bin/python -m pip check
```

`requirements.in` records intended direct dependencies; `requirements-lock.txt` records exact resolved package versions. The lock is generated on Apple Silicon; other platforms may require compatible platform-specific wheels. Training on another accelerator may not reproduce results bit for bit.

## Acquire the complete dataset

```sh
.venv/bin/python scripts/acquire_plitter.py --target-images 10000 --workers 8 --min-free-gb 10
```

This is resumable. It checks original image dimensions after accounting for EXIF rotation. It stops short of declaring completion if any image is unavailable, corrupt, or inconsistent. It never silently drops failed images. A 10 GiB free-space floor preserves room for macOS and temporary files; reaching it means pause and free space, not reduce quality.

The original image URLs may change or become unavailable. Preserve the manifest and use DVC backup storage when available. The historical ZIP archive is not assumed to contain byte-identical data to the pinned current annotation snapshot.

## Data versioning and preparation

Git and DVC are initialised once. Source images are managed through DVC, not committed as binaries to Git:

```sh
.venv/bin/dvc add data/raw/plitterstreet
.venv/bin/dvc repro prepare explore
git add .gitignore .dvc .dvcignore data/raw/plitterstreet.dvc data/raw/.gitignore dvc.yaml dvc.lock params.yaml configs scripts tests docs reports requirements.in requirements-lock.txt README.md
git commit -m "Prepare versioned 10000-image pLitterStreet data and Phase 1 reproducible pipeline"
```

Prepared image paths are hard links to the original image bytes on the same filesystem. DVC uses `hardlink,copy` caching to avoid redundant image copies when supported. **Do not modify linked images in place.** Replace or regenerate files through the pipeline when changing data. The original photographs are not resized by preparation.

For source JPEGs without the expected final end-of-image marker, preparation creates a separate derived file and appends the marker without resizing or recompressing pixels. These files are listed in the dataset summary. Raw originals and hashes remain unchanged. Auto-generated label caches are excluded through `.dvcignore`.

A configured off-device DVC remote is not yet available. Local cache versioning is not an off-device backup. Do not claim successful `dvc push` or remote recovery until actual storage has been configured and verified.

The ten source classes are retained: litter, pile, face mask, trash bin, plastic bag, bottle, cup, rope, sachet, and straw. Mappings are in `configs/plitter_classes.json`. Bottle and cup labels do not certify plastic material.

Splits keep inferred source-video groups together and merge groups linked by identical hashes. Fractions target 70/15/15 by image count and may be approximate because whole groups are assigned. Dataset IDs and filename stems are grouping proxies, not a guarantee of scene independence across recordings. Review near-duplicates and source annotations before treating evaluation as final.

## Exploration and feature engineering evidence

Generated reports are stored in `reports/phase1/`:

- `acquisition.json`: completeness, failures, and verified downloaded bytes.
- `dataset_summary.json`: category counts, split counts, negatives, duplicates, and clipped boxes.
- `exploration.md`: findings and limitations.
- Distribution plots, object-size plots, and annotated original samples.

Feature engineering includes the class mapping and validated conversion from COCO bounding boxes to normalised YOLO coordinates. The detector backbone learns visual features through transfer learning. Input resizing and normalisation follow the model implementation; augmentation is applied only to training images and their boxes. No augmented dataset copies are saved.

## Pipeline check and baseline training

Run the small engineering check first:

```sh
.venv/bin/python -m pytest -q tests
.venv/bin/python scripts/train.py --profile benchmark
```

The benchmark deliberately uses one epoch, reduced training fraction, and smaller model input size. It validates the integration and estimates speed. **Its results are not the baseline and must not be presented as final model performance.** All originals remain in the dataset.

The baseline uses all training images, 640-pixel model inputs, and up to 50 epochs with early stopping:

```sh
.venv/bin/dvc repro train
```

The initial model is a pretrained YOLO11 nano detector fine-tuned for our mapped classes. Batch size 2 and zero extra loader workers accommodate 8 GB unified memory without changing the dataset. Training uses native Apple MPS; the wrapper fails clearly if GPU access is unavailable rather than silently switching device. An explicit CPU run is possible with `--device cpu`.

No full-length experiment or baseline result is claimed until training actually completes. Experiment selection is based on validation results; hold the test set back until the model/configuration is selected.

## MLflow evidence

```sh
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open http://127.0.0.1:5000 locally. Training logs parameters, epoch metrics, model weights, evaluation figures, dependency versions, data hashes, and Git revision/dirty state. Benchmark runs are labelled explicitly. The database and artifacts are local operational files excluded from Git; back them up as a unit before moving machines.

The model registry is a Phase 2 deliverable. Phase 1 tracks model artifacts without claiming registration or deployment.

## Initial evaluation

```sh
.venv/bin/python scripts/evaluate.py --split val
```

After selecting a configuration, run the held-out evaluation once for the report:

```sh
.venv/bin/python scripts/evaluate.py --split test
```

Report precision, recall, mAP@0.5, mAP@0.5:0.95, per-class results, confusion matrix, and documented inference timings. Inspect false positives and missed/small/occluded objects. Public-dataset scores do not establish campus performance, plastic mass, recyclability, geographic hotspots, or campaign effectiveness.

## Review records

Keep the Git commit, DVC data reference, MLflow run ID, selected model hash, evaluation report, figures, and limitations together. Screenshots support the presentation; the original machine-readable records provide the actual evidence.


## Model outputs

Current training writes dataset-specific filenames, including `models/plitterstreet-baseline.pt`.
