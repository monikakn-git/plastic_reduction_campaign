# AI-Assisted Plastic Waste Monitoring and Targeted Reduction Campaigns

> **Project status (7 November 2026):** Phase 1 technical evidence is ready for review. YOLO11n, YOLO11s and YOLO11m experiments are complete. The final YOLO11m test scores are **37.1% mAP50, 22.7% mAP50–95, 46.1% precision and 39.6% recall**. Faculty dataset approval remains pending. Labels-v4 is a later, untrained correction variant. See the [final Phase 1 report](reports/phase1/PHASE1_FINAL_REPORT.md), [college checklist](reports/phase1/PHASE1_CHECKLIST.md), and [review guide](docs/PHASE1_REVIEW.md).

## Overview

This project aims to help a college campus identify visible plastic litter, understand where it occurs, and plan targeted waste reduction campaigns. Users submit photographs with location and audit details. A trained object detection model identifies supported waste categories, and the system aggregates reviewed observations into a dashboard for campaign organisers.

The project combines computer vision with MLOps: dataset versioning, reproducible training, experiment tracking, model evaluation, deployment, monitoring, and controlled model updates.

The initial deployment scope is a campus with defined zones such as the canteen, hostel surroundings, academic blocks, and common areas. Expansion to neighbourhoods or larger regions is a future possibility, subject to representative data and validation.

## Problem Statement

Plastic waste reduction campaigns often rely on general awareness messages without a consistent process for identifying local waste patterns or measuring changes after an intervention. Organisers may therefore lack evidence about which locations need attention, which visible waste categories are most common, and whether observed conditions improve over time.

This project proposes an AI-assisted system that detects selected plastic waste categories in submitted images, associates observations with user-provided locations and audit records, and supports targeted campaign planning. A reproducible MLOps pipeline will support model development, evaluation, deployment, and improvement using verified feedback.

## Objectives

- Train and evaluate an object detection model for selected plastic waste categories.
- Collect observations with location, timestamp, and audit context.
- Identify recurring patterns and priority locations using comparable observations.
- Recommend practical campaigns based on the observed waste patterns.
- Compare standardised audit results before and after campaigns.
- Maintain traceability from dataset and code versions to training runs and deployed models.
- Support human review of uncertain or incorrect predictions.

## Expected Outcome After Completion

The completed project should provide a working prototype in which a user can:

1. Upload a campus waste photograph and select its location and audit session.
2. View detected items with bounding boxes, labels, and confidence scores.
3. Review or correct predictions before they contribute to verified audit results.
4. Explore waste observations by location, category, and time period.
5. See priority locations based on documented audit measures.
6. Receive campaign suggestions with a clear explanation of the underlying observations.
7. Record a campaign's location, action, dates, and intended outcome.
8. Compare equivalent audit periods before and after the campaign.

The technical deliverable should also allow a project member to retrieve a dataset version, reproduce a documented training run within stated hardware and software limits, inspect its evaluation, and identify the model version used for a prediction.

Success means delivering and evaluating this workflow. It does **not** require claiming that the system will always detect every item or guarantee a reduction in plastic waste.

## Intended Users

| User | Intended use |
| --- | --- |
| Students and volunteers | Submit images during organised audits and review observations |
| Campus sustainability teams | Identify priority areas and plan campaigns |
| Campaign coordinators | Record interventions and compare audit outcomes |
| Project administrators | Manage datasets, models, evaluation, and monitoring |

## System Workflow

```text
Image + selected location + audit details
                    |
                    v
          Input validation
                    |
                    v
       Waste object detection model
                    |
                    v
       Predictions and human review
                    |
                    v
       Stored observations and audit records
                    |
                    v
        Location and time-based analysis
                    |
                    v
      Explained campaign recommendations
                    |
                    v
         Campaign implementation record
                    |
                    v
      Comparable follow-up waste audits
```

## Core Features

### 1. Plastic Waste Detection

A pretrained YOLO object detection model will be fine-tuned on an approved, annotated waste dataset. Object detection supports locating multiple visible items in one image rather than assigning only one label to the entire image.

Each prediction will include a bounding box, class label, and confidence score. These scores help prioritise review; they are not guarantees of correctness.

Candidate categories include plastic bottles, bags, cups, and wrappers. **The final class list will depend on available annotations, material ambiguity, class balance, and faculty approval.** An object that resembles a plastic item cannot always be confirmed as plastic from its appearance alone.

The exact YOLO version and model size will be selected after checking the dataset, hardware, licence requirements, and deployment constraints. Training will use transfer learning rather than training a detector from scratch.

### 2. Location-Based Analysis

Users will select a campus zone for each observation. The application will combine detection results with location and time metadata to show category distributions, trends, and areas requiring attention.

Location is supplied by the user or an approved metadata source; the detection model does not infer geographic coordinates from an image.

Raw submission counts will be kept separate from standardised audit measures. More uploaded photographs do not necessarily mean more waste.

### 3. Targeted Campaign Recommendations

The first recommendation module will use documented rules based on verified audit patterns. A separate recommendation model is not required for the initial system.

| Observed pattern | Example campaign suggestion |
| --- | --- |
| Repeated bottle litter near a canteen | Promote reusable bottles and available refill points |
| Frequent bag litter near campus shops | Run a reusable bag awareness campaign |
| Repeated disposable cup litter near beverage stalls | Coordinate a reusable cup initiative with vendors |
| Frequent wrapper litter around common areas | Combine disposal reminders with a review of bin availability |

These are suggestions for organisers to review. They do not establish the cause of litter or prove that a particular intervention will work.

AI-generated campaign messages may be added as an optional extension. The trained waste detector remains the main AI component.

### 4. Campaign Tracking and Evaluation

Each campaign record will include its target location, target waste category, action, dates, and linked baseline and follow-up audits.

Comparisons should use consistent audit areas, routes, collection methods, and observation periods. Measures may include visible items per standardised audit or items per surveyed area. If physical waste weight is needed, it must be measured separately.

For equivalent audits:

```text
Observed reduction (%) = ((Before - After) / Before) × 100
```

This calculation is valid only when the baseline is greater than zero. A negative result indicates an increase. Changes will be reported as observed differences, since weather, footfall, cleaning schedules, and other factors can affect results. A comparison zone can strengthen evaluation when feasible.

### 5. Review and Feedback

Reviewers should be able to correct classes and bounding boxes, identify missed objects, and flag unusable images. Corrections will be stored separately from original predictions.

Reviewed examples may enter a later dataset version after validation. User corrections will not automatically replace the deployed model or change the held-out evaluation set.

## Dataset Plan

**Current Phase 1 target:** exactly 10,000 original pLitterStreet images, selected reproducibly from its official 13,064-image 10-class release. Source-resolution images and bounding boxes are retained; approximately 70/15/15 splits keep inferred source-video groups together. All ten original categories are preserved. Generic bottle/cup labels do not establish plastic material. See [dataset approval record](docs/DATASET_APPROVAL.md).

The dataset must contain images with object-level annotations suitable for detection. A folder containing only image-level category labels is insufficient for this training task.

Dataset selection will consider:

- Source, licence, permitted uses, and faculty approval.
- Availability and quality of bounding boxes and category labels.
- Coverage of relevant waste types and realistic backgrounds.
- Image quality, class imbalance, duplicates, and annotation errors.
- Similarity to intended campus deployment conditions.

Public data may be supplemented with appropriately collected campus images. Public images without verified location or audit metadata will not be presented as evidence about campus hotspots.

Training, validation, and test data will be separated before augmentation. Related images from the same scene, capture session, or video should remain in the same split to reduce leakage. The test set will be reserved for final evaluation rather than repeated tuning.

## Preprocessing and Feature Engineering

For the detector, visual features will be learned by the neural network. The Phase 1 feature engineering discussion will explain this alongside the supporting data preparation steps:

- Validate image files, class identifiers, and bounding box coordinates.
- Resize inputs using the selected model's preprocessing conventions.
- Apply appropriate training-only augmentation with corresponding box transformations.
- Inspect the effect of small objects, lighting, occlusion, and background variation.
- Document class mapping and any class-merging decisions.

Later analytical features may include counts by category, category proportions, items per standardised audit, and changes between comparable periods. These require audit context; they cannot be interpreted reliably from arbitrary photo counts alone.

## MLOps Design

| Component | Intended purpose |
| --- | --- |
| Git | Version code, configuration, documentation, and pipeline definitions |
| DVC | Version datasets, annotations, and split manifests using configured storage |
| MLflow | Track training parameters, metrics, plots, and model artifacts |
| Reproducible environment | Record dependencies, seeds, hardware, and training settings |
| Model release records | Link deployed weights to training runs and dataset versions |
| Automated checks | Validate inputs and verify essential pipeline and API behaviour |
| Deployment packaging | Provide a consistent inference environment |
| Monitoring | Track service health, input changes, and reviewed prediction quality |

Exact numerical reproducibility can vary across hardware and accelerator operations. These limits will be documented.

### Planned Model Lifecycle

```text
Approved data -> validation -> DVC version -> training
     -> MLflow evaluation -> candidate review -> deployment
     -> monitoring and verified feedback -> new dataset version
     -> candidate retraining and evaluation
```

A retrained candidate will replace the deployed model only after evaluation against defined release criteria. The previous version should remain available for rollback.

## Evaluation Plan

### Model Evaluation

- Precision and recall for supported categories.
- Mean average precision, including mAP@0.5 and mAP@0.5:0.95.
- Per-class results and analysis of missed and incorrect detections.
- Count error on an annotated audit-image subset where appropriate.
- Inference latency measured on documented hardware.
- Separate campus-image evaluation when suitable labelled images are available.

Confidence and matching thresholds will be documented. Performance targets will be set after dataset exploration and baseline evaluation; no accuracy value is claimed in advance.

### System and Campaign Evaluation

- Ability to complete the upload, review, analysis, and campaign workflow.
- Traceability of predictions to model versions.
- Reproducibility of a documented training run.
- Completeness and comparability of audit records.
- Observed changes in standardised audit measures after campaigns.

Live model accuracy requires reviewed ground-truth examples. Prediction confidence or input drift alone will not be reported as proof of accuracy.

## Development Phases

### Phase 1 — Development and Reproducibility

This phase follows the provided course requirements:

1. Acquire a suitable detection dataset and obtain approval.
2. Explore images, labels, category balance, and data quality.
3. Define class mapping, splits, preprocessing, and augmentation.
4. Train a small pretrained YOLO detector as the baseline.
5. Maintain source code and configuration in Git.
6. Version data and splits with DVC.
7. Track experiments and artifacts with MLflow.
8. Evaluate the initial model and document errors and limitations.

**Expected deliverables:** dataset description and approval record, exploration report, reproducible training configuration, DVC metadata, MLflow runs, baseline model artifact, evaluation report, and example detection outputs.

### Phase 2 — Productionization and Deployment

The model is registered in MLflow and served through a tested FastAPI application in a local ARM64 Docker container. Private GitHub Actions workflows test the API and quality gates, build a commit-labelled image and support owner-triggered deployment after successful CI for that exact commit. See [Phase 2 operations](docs/PHASE2.md) and [Phase 2 status](reports/phase2/STATUS.md) for execution evidence and remaining conditions. Location-tagged observations and campaign analysis are planned application features for Phase 3; they are not implemented in this release.

### Phase 3 — Monitoring, Feedback and Governance

Follow the college requirements: Evidently drift monitoring, Python logging, Prometheus/Grafana or Power BI, simulated production data, deliberate degradation, retraining triggers, model comparison and governance/model cards. Integrate explained regional priorities and campaign tracking with reviewed observations. Campaign effectiveness requires comparable real audits. Phase 3 has not started.

## Proposed Technology Stack

| Layer | Candidate technology |
| --- | --- |
| Model training and inference | Python, PyTorch, Ultralytics YOLO |
| Data exploration | Pandas, Matplotlib, image inspection tools |
| Data and experiment management | Git, DVC, MLflow |
| Backend | FastAPI |
| Prototype dashboard | Streamlit |
| Observation storage | SQLite initially; PostgreSQL if deployment needs justify it |
| Deployment packaging | Docker |

The implemented ML dependencies are pinned in the Mac lockfile and separate CUDA/cloud environment records. Service and deployment choices belong to Phase 2.

## Proposed Repository Structure

The following application layout is a future reference. Current Phase 1 code lives in `scripts/`, `cloud/`, `configs/`, `tests/` and `notebooks/`.

```text
MLOPS_EL/
├── README.md
├── configs/              # Dataset, training, evaluation settings
├── data/                 # Data managed through DVC
├── notebooks/            # Exploration and analysis
├── src/
│   ├── data/             # Validation and split preparation
│   ├── training/         # Model training and experiment logging
│   ├── evaluation/       # Metrics and error analysis
│   ├── inference/        # Prediction service
│   ├── analytics/        # Location and audit analysis
│   └── campaigns/        # Recommendation rules and campaign records
├── app/                  # User dashboard
├── tests/                # Essential pipeline and service checks
├── reports/              # Evaluation reports and figures
├── dvc.yaml              # Reproducible pipeline definition
├── params.yaml           # Versioned parameters
└── pyproject.toml        # Dependencies and project configuration
```

## Setup and Execution Status

Phase 1 acquisition, data preparation, exploration, training, and evaluation commands are implemented in `scripts/`. Follow [the Phase 1 guide](docs/PHASE1.md) for environment setup, DVC versioning, training, and MLflow inspection. Actual completion and metrics must be checked against the generated reports and experiment records.

YOLO11m completed the same experiment across Colab and Kaggle, stopping at epoch 86 with its best checkpoint from epoch 66. The model used labels-v3 and 960-pixel inputs. All 1,500 test images were evaluated after model selection. See [cloud training and recovery](docs/CLOUD_TRAINING.md), [final test results](reports/phase1/evaluation_yolo11m_test.json), and [safe review commands](docs/PHASE1_REVIEW.md). No training is running or planned for Phase 1 finalization.

The full application and later-phase components are not implemented yet. Off-device DVC storage and faculty dataset approval remain pending.

Large datasets and model binaries should use suitable artifact storage rather than being committed directly to Git. Credentials must remain outside the repository.

## Scope and Limitations

- Detection covers only supported classes and visible objects.
- Overlapping, small, transparent, dirty, or hidden items may be missed.
- Photos cannot reliably determine plastic weight, polymer type, or recyclability.
- Repeated or overlapping images can double-count litter; audit procedures and review must address this.
- Public-dataset performance does not establish performance on campus scenes.
- Location rankings require comparable sampling, not simply totals from voluntary submissions.
- Campaign recommendations support human decisions and do not guarantee outcomes.
- Collecting personal identity is unnecessary for waste analysis. Image collection should minimise identifiable people and follow campus permissions.
- This project supports measurement and planning; physical waste collection and campaign execution require people and campus coordination.

## Final Demonstration

The intended final demonstration will show a sample audit image producing reviewed detections, an observation appearing in its campus zone, a campaign suggestion explained by the recorded pattern, and a comparison with a follow-up audit. Any synthetic demonstration records will be explicitly labelled.

The demonstration will also show the dataset version, training run, evaluation artifacts, and model version behind the predictions, followed by an example of reviewed feedback entering a controlled model improvement cycle.

## Reference Documentation

- [Ultralytics object detection](https://docs.ultralytics.com/tasks/detect/)
- [Ultralytics detection dataset format](https://docs.ultralytics.com/datasets/detect/)
- [DVC documentation](https://dvc.org/doc)
- [MLflow documentation](https://mlflow.org/docs/latest/)

Dataset attribution, model licence details, and the project's own licence will be added after the actual sources and implementation choices are confirmed.
