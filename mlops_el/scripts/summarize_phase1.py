"""Consolidate completed Phase 1 evidence without launching training or changing data."""
import csv,hashlib,json,platform,sqlite3,subprocess
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/phase1'

def read(name):return json.loads((OUT/name).read_text())
def pct(n):return f'{n*100:.1f}%'
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    data=read('dataset_summary.json');baseline=read('baseline_metrics.json');nv=read('evaluation_val.json');nt=read('evaluation_test.json');st=read('yolo11s_candidate_training.json');sv=read('yolo11s_candidate_validation.json');mt=read('yolo11m_completion_summary.json');test=read('evaluation_yolo11m_test.json');quality=read('training_quality_v4/variant_verification.json');tracking=read('cloud_tracking_local_view.json')
    if sha(ROOT/test['weights'])!=test['weights_sha256']:raise ValueError('Selected weights changed after test evaluation')
    with sqlite3.connect(Path(tracking['local_view_database']).as_uri()+'?mode=ro',uri=True) as db:
        assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        assert db.execute('select status from runs where run_uuid=?',(mt['run_id'],)).fetchone()[0]=='FINISHED'
    tests=(ROOT/'.local/phase1-final-tests.log').read_text().strip().splitlines()[-1]
    dvc=json.loads((ROOT/'.local/phase1-dvc-status.json').read_text())
    if dvc:raise ValueError('Resolve unexpected DVC changes before finalizing')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    final=test['metrics'];model_rows=[['YOLO11n','Original prepared',640,'50',nv['metrics']['metrics/mAP50(B)'],nv['metrics']['metrics/mAP50-95(B)'],nt['metrics']['metrics/mAP50(B)']],['YOLO11s','Original prepared',640,'50',sv['metrics']['metrics/mAP50(B)'],sv['metrics']['metrics/mAP50-95(B)'],None],['YOLO11m','labels-v3',960,'86 (best 66)',mt['final_best_validation']['map50_displayed'],mt['final_best_validation']['map50_95_displayed'],final['metrics/mAP50(B)']]]
    pointers={str(p.relative_to(ROOT)):yaml.safe_load(p.read_text())['outs'][0] for p in [ROOT/'data/raw/plitterstreet.dvc',ROOT/'data/prepared/plitterstreet-orientation-v2.dvc',ROOT/'data/prepared/plitterstreet-labels-v3.dvc',ROOT/'data/prepared/plitterstreet-labels-v4.dvc',ROOT/'models/plitterstreet-yolo11m-candidate.pt.dvc',ROOT/'reports/phase1/training_quality_v4.dvc',ROOT/'reports/phase1/experiment_history.dvc']}
    summary={'phase':'1 Development and Reproducibility','as_of':'2026-11-07','technical_review_package':'ready','faculty_dataset_approval':'pending_external_action','dataset':{'images':data['images'],'annotations':data['target_annotations'],'split_images':data['split_image_counts'],'classes':data['classes'],'negative_images':data['negative_images']},'selected_evaluated_model':{'path':test['weights'],'sha256':test['weights_sha256'],'training_run_id':mt['run_id'],'test_run_id':test['run_id'],'trained_dataset':'labels-v3','completed_epochs':86,'best_epoch':66,'test_metrics':final},'experiments':model_rows,'new_dataset_variant':{'path':quality['variant'],'class_corrections':30,'training_started':False,'model_score':None},'qa':{'tests':tests,'dvc_status':dvc,'selected_model_hash_verified':True,'cloud_mlflow_integrity':'ok','cloud_mlflow_status':'FINISHED'},'dvc_pointers':pointers,'git_evidence_parent':commit,'limitations':['70% target not achieved','Faculty approval pending','No configured off-device DVC remote; verified private train/validation checkpoint backups are separate','Near-duplicate/scene independence not exhaustively verified','Further tuning requires validation; previously evaluated test set must not become a selection target','No campus performance, material certainty or campaign-impact claim'],'phase2_started':False}
    (OUT/'phase1_final_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    table='| Model | Training data | Input | Epochs | Val mAP50 | Val mAP50–95 | Test mAP50 |\n|---|---|---:|---|---:|---:|---:|\n'
    for name,version,size,epochs,v,v95,t in model_rows:table+=f'| {name} | {version} | {size} | {epochs} | {pct(v)} | {pct(v95)} | {pct(t) if t is not None else "Not evaluated"} |\n'
    classes='| Class | Test objects | Precision | Recall | AP50 | AP50–95 |\n|---|---:|---:|---:|---:|---:|\n'
    for r in test['per_class']:classes+=f'| {r["Class"]} | {r["Instances"]} | {pct(r["Box-P"])} | {pct(r["Box-R"])} | {pct(r["mAP50"])} | {pct(r["mAP50-95"])} |\n'
    report=f'''# Phase 1 final report

AI-assisted plastic waste reduction campaign\nReview date: 7 November 2026

## Problem and Phase 1 outcome

Develop a reproducible object detector for litter audit photographs as the foundation for a later system that records reviewed observations by location and recommends campaigns. The detector uses transfer learning with Ultralytics YOLO11. It learns from images and bounding boxes rather than a single image-level class.

The development and reproducibility work is ready for technical review. Faculty dataset approval remains pending and cannot be asserted by code. The chosen evaluated candidate is YOLO11m trained on labels-v3. Its test mAP50 is **{pct(final['metrics/mAP50(B)'])}**. The 70% aspiration was not achieved. We preserve the baseline and unsuccessful/engineering trials as experiment evidence.

## Dataset acquisition and schema

Official source: [pLitterStreet on Zenodo](https://zenodo.org/records/8288500), [authors' repository](https://github.com/gicait/pLitter), [paper](https://arxiv.org/abs/2401.14719). The source record declares CC BY 4.0. See `docs/DATASET_APPROVAL.md` for the pending approval record and attribution.

A reproducibly selected set of **10,000 distinct original images** contains **60,826 labelled objects**, including 155 negative images with no selected objects. No resized or augmented copies count toward 10,000. Acquisition records preserve source provenance, member-level CRC and image hashes. Selective acquisition does not establish a checksum for an entire unretained ZIP.

| Split | Images | Purpose |
|---|---:|---|
| Train | 7,000 | Fit weights and develop labels |
| Validation | 1,500 | Select experiments and best checkpoint |
| Test | 1,500 | Final assessment after model selection |

The original image metadata array has 10,000 rows and 16 fields. The annotation array has 60,826 rows and 9 fields. A nested bounding box stores x, y, width and height. Each prepared YOLO object row has **5 values: class_id, x_center, y_center, width, height**, with coordinates normalized to image dimensions. An image has height × width × 3 pixel channels. These metadata tables are not a conventional fixed-column feature matrix. See `docs/DATA_SCHEMA.md`.

Ten classes: {', '.join(data['classes'])}. Bottle and cup identify object type, not polymer/material certainty. The dataset contains no verified campus locations or campaign outcomes.

## Exploration and feature engineering

The generated EDA records category imbalance, image dimensions, object-size distributions, split composition, negatives and annotated samples. Litter dominates the source labels with 44,764 objects; cup has 724 and trash bin 607. Small, deformed, occluded and transparent objects require per-class assessment. Zero exact duplicate images and zero clipped boxes were recorded. Filename/video grouping is a proxy, not proof of independence between all scenes.

Feature engineering and preparation consist of explicit class mapping to IDs 0–9, validated COCO-to-YOLO coordinate conversion, grouped split assignment and image decode checks. The pretrained backbone learns visual features. Letterboxing, numerical normalization and augmentation operate at model input time. Training augmentations preserve corresponding boxes. Validation/test inference uses no test-time augmentation.

64 derived JPEG files received an end marker without recompression; raw images remain intact. Orientation-v2 corrects four training images' coordinate interpretation. Labels-v3 adds two visually supported annotation refinements. Original-resolution images remain available. The baseline and YOLO11s use 640-pixel inputs; YOLO11m uses nominal 960-pixel input sizing.

Evidence: `exploration.md`, `dataset_summary.json`, `source_class_distribution.png`, `target_split_distribution.png`, `image_and_object_sizes.png`, `annotated_source_samples.jpg`.

## Experimentation and model evaluation

{table}
All values above are actual recorded results. YOLO11m validation values are rounded from its final best-model validation output. The m experiment changed model capacity, resolution, data variant, training schedule and accelerator, so this is an experiment comparison rather than a controlled attribution of gains to model size alone. YOLO11s has no final test result.

YOLO11m used batch 4, AdamW with initial learning rate 0.001, seed 42, maximum 100 epochs and patience 20. It completed epoch 30 in Colab and continued the same run from epoch 31 in Kaggle. Early stopping ended training at epoch 86 after 20 epochs without improvement in the monitored validation fitness. The best checkpoint came from epoch 66. The final best-model validation mAP50 was 39.4% and mAP50–95 was 24.7%.

### Final held-out YOLO11m test result

| Metric | Score |
|---|---:|
| mAP50 | {pct(final['metrics/mAP50(B)'])} |
| mAP50–95 | {pct(final['metrics/mAP50-95(B)'])} |
| Precision | {pct(final['metrics/precision(B)'])} |
| Recall | {pct(final['metrics/recall(B)'])} |

The evaluation used all 1,500 test images and 10,430 labelled objects, nominal imgsz 960, batch 1 on Apple MPS, confidence filter 0.001 and NMS IoU 0.7. It used the saved model without training or test-time augmentation. Reported precision/recall follow the library's evaluation operating-point summary; they are not a fixed deployment confidence policy. AP integrates precision/recall across confidence thresholds. mAP50 uses IoU 0.50, while mAP50–95 averages the ten IoU thresholds 0.50 through 0.95. Detection metrics are not a single classification accuracy.

{classes}
Trash bin and litter have stronger AP50 than cups. Cup AP50 remains 11.5%, and bottle AP50 is 30.9%. Scores alone do not establish that labels caused the errors. The validation-to-test mAP50 gap is about 2.3 percentage points. Relative to the baseline's 26.2% test mAP50, the final candidate gains about 10.9 percentage points. This dataset result does not establish readiness for unsupervised campus decisions.

The test set was evaluated after candidate selection. After seeing these results, further development must use training/validation evidence. Repeated selection against this test score would weaken independence. A new independent holdout would support stronger confirmation after subsequent tuning.

## Data-quality audit and the untrained v4 variant

A later training-only audit structurally checked 7,000 label files and visually screened 670 targets in 595 images, including all 488 training cup labels. It applied **30 class corrections across 28 training images** in labels-v4, retained all images/instances/coordinates, and verified all 3,000 holdout labels unchanged. It queued 148 uncertain annotations for adjudication.

Labels-v4 has **no trained model and no evaluation score**. The selected model's result belongs to labels-v3. These AI-assisted corrections are not comprehensive expert relabelling or evidence of a score increase. See `dataset_quality_v4.md` and `docs/ANNOTATION_POLICY.md`.

## Version management, tracking and recovery

Git records code, configuration, reports and DVC pointers. DVC manages raw data, prepared variants, models and audit evidence with content hashes. The final verification reports a clean DVC workspace. The local cache is version storage, not an off-device backup, and a DVC remote is not configured.

Private Kaggle checkpoint dataset versions preserve the cloud experiment independently. Each epoch saved checkpoints, CSV/configuration and a consistent MLflow snapshot, then uploaded a private version and downloaded it to verify SHA256 before continuing. The final remote backup and its locally downloaded 500,275,200-byte payload passed integrity checks. This cloud backup contains train/validation material and experiment records, not a complete off-device copy of every project/test file.

DVC also versions consistent local and cloud MLflow metadata snapshots under `experiment_history/`. Those SQLite snapshots contain experiment metadata, while artifact files remain in their original stores and verified cloud archive.

The cloud archive retains best/last checkpoint hashes, optimizer recovery state and the original MLflow database. A separate local database view changes only copied artifact paths so MLflow can open them on this Mac. The original archive remains unchanged. Exact CUDA dependencies are in cloud setup records; Mac inference uses its separately recorded environment. Cross-device bit-for-bit training equality is not claimed.

| Record | Identity |
|---|---|
| Baseline training | `{baseline['run_id']}` |
| YOLO11s training | `{st['run_id']}` |
| YOLO11m training | `{mt['run_id']}` |
| YOLO11m test evaluation | `{test['run_id']}` |
| Evaluated model SHA256 | `{test['weights_sha256']}` |

## QA and reproducibility limits

Final check: **{tests}**. Tests cover data contracts, selected geometry/conversion behavior, acquisition safety, label patch guards, evaluation export and cloud migration/backup safety. Automated checks do not replace expert semantic annotation review. Full training was not rerun just to finalize this report; historical recovery, hashes, DVC references and tracking records provide the reproducibility evidence.

Use `docs/PHASE1_REVIEW.md` for safe review commands. Do not run `dvc repro` or the full training shell scripts during an ordinary presentation because they can trigger expensive work. Separate Mac and CUDA environment records accompany the experiments.

## Risks and limitations

Faculty dataset approval is pending. Source-label inconsistency, rare classes and small objects remain. Near-duplicate scene review is incomplete. Detection cannot certify plastic type, weight or recyclability. Human review is necessary before campaign decisions. No campus deployment, geographic hotspot model, campaign-effect measurement, API, registry, container, CI/CD or drift-monitoring result is claimed in Phase 1. Source/runtime licensing review must be completed before distribution in later phases. The 70% target was aspirational and remains unmet.

## College rubric and later phases

Phase 1 covers acquisition, exploration, feature engineering, baseline development, Git, DVC, MLflow and initial evaluation. The technical evidence is ready; faculty approval is the remaining external approval item. See `PHASE1_CHECKLIST.md` for artifact mapping.

Phase 2 follows the college specification: PyTest model/service checks, model-risk/quality gates, MLflow registry, FastAPI, Docker, GitHub Actions and deployment. Phase 3 adds Evidently, logging, monitoring dashboards, simulated production data, deliberate drift/degradation, feedback/retraining evidence, model cards and governance. Location observations and campaign planning remain application work to integrate with those phases, using reviewed location metadata and transparent rules until suitable supervised data exists.
'''
    assert sum(r['Instances'] for r in test['per_class'])==10430
    (OUT/'PHASE1_FINAL_REPORT.md').write_text(report)
    checklist='''# Phase 1 college checklist

Technical evidence is ready for review. Faculty dataset approval remains pending.

| College requirement | Status | Evidence |
|---|---|---|
| Dataset acquisition / approval | Acquisition complete; approval pending | acquisition.json, docs/DATASET_APPROVAL.md, raw DVC pointer |
| Data exploration | Complete | exploration.md, distributions, image/object-size plots |
| Feature engineering | Complete | class map, prepare_data.py, DATA_SCHEMA.md, preprocessing records |
| Baseline models | Complete | YOLO11n baseline, YOLO11s candidate, YOLO11m final candidate |
| Git repository | Complete locally | Commit history and tracked code/config/reports; no remote publication claimed |
| DVC dataset versioning | Complete locally | dvc.lock, raw/v2/v3/v4 pointers, model pointer; remote not configured |
| MLflow experiment tracking | Complete | Local baseline/evaluation store and restored cloud training view |
| Initial model evaluation | Complete | Separate validation/test reports, per-class results, confusion/PR curves |

Additional evidence: 36 passing tests; same-run checkpoint recovery between Colab and Kaggle; verified final private cloud backup; separate untrained labels-v4 audit.

Phase 2 and Phase 3 remain pending. Faculty approval must not be presented as obtained. The corrected labels-v4 dataset has not produced the reported model scores.
'''
    (OUT/'PHASE1_CHECKLIST.md').write_text(checklist)
    (OUT/'STATUS.md').write_text('# Phase 1 current status\n\nTechnical review package ready as of 7 November 2026. Faculty dataset approval remains pending. No training is running.\n\n'+table+'\nFinal YOLO11m test: precision 46.1%, recall 39.6%, mAP50 37.1%, mAP50–95 22.7%. Labels-v3 produced this model. Labels-v4 contains 30 later class corrections and is untrained.\n\nSee [final report](PHASE1_FINAL_REPORT.md), [college checklist](PHASE1_CHECKLIST.md), [review guide](../../docs/PHASE1_REVIEW.md), and phase1_final_summary.json. Local DVC verification is clean. Off-device DVC remote and faculty approval remain pending. Private cloud checkpoints are independently verified. Phase 2 has not started.\n')
    print('Phase 1 report, checklist, status and machine-readable summary generated.')

if __name__=='__main__':main()
