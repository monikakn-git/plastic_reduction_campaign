# Phase 1 free GPU experiment

The YOLO11m labels-v3 experiment is complete. Early stopping ended training at epoch 86, preserving the best checkpoint from epoch 66. Final validation mAP50 is 39.4%; final test mAP50 is 37.1%. The 70% aspiration was not achieved. The baseline and YOLO11s records remain preserved.

Completion evidence and safe review commands are in `reports/phase1/PHASE1_FINAL_REPORT.md` and `docs/PHASE1_REVIEW.md`. No training is running. The continuation instructions below are historical recovery documentation, not instructions to restart this completed experiment.

## Current status

### Kaggle migration (historical execution record)

The user selected free Kaggle while Colab is quota-blocked. **Hold Colab
stopped during Kaggle setup and training; never run both providers on this
experiment.** Older Colab reconnect instructions below apply only after an
explicit decision to switch back.

Kaggle notebook:
https://www.kaggle.com/code/aayushbmlops26/mlops-el-yolo11m-checkpoint-continuation/edit

Registration and phone verification are complete. A real environment probe
confirmed T4 GPUs, Python 3.13.16, Torch 2.11.0+cu130 and torchvision
0.26.0+cu130. The pinned Ultralytics 8.4.201 and MLflow 3.17.2 installation
also passed. Both input datasets are private. The complete recovery ZIP was
received and its epoch-30 last/best hashes, dataset fingerprint, SQLite
integrity and same MLflow run identity passed local verification. See
`reports/phase1/kaggle_recovery_verification.json`.

The user explicitly approved internet access and a Kaggle API token stored
only in this notebook's Secrets. The initial token accidentally appeared in
tool output; it was expired with explicit approval and replaced. The active
replacement is in `MLOPS_EL_KAGGLE_API_TOKEN`. No token belongs in project
files, notebook source, Git or experiment logs.

`cloud/kaggle_preflight.py` checks every dataset payload and loads the saved
epoch/optimizer under the exact dependencies before any training.
`cloud/start_kaggle_training.py` resumes only the existing run and uses
`cloud/kaggle_backup.py` for private remote checkpoint versions. The backend
uploads the latest complete generation, tracking database/artifacts and
training history, then downloads the remote payload and verifies its SHA.
Training is gated on a successful initial round trip, and later backup
failures raise rather than silently allowing training to continue. Previous
remote versions are retained; only disposable local verification downloads
are removed. This uses Kaggle storage beyond ephemeral runtime disk.

The adapter preserves `/content/mlops_work` as a path alias to Kaggle's
working directory so original MLflow artifact URIs and checkpoint arguments
remain valid. It does not fake a Drive mount or disable the default Drive
persistence checks. Original Colab behavior remains the default backend.

The first Kaggle launch requires `main(initialize_backup=True)` after the
preflight passes. This creates the private backup dataset and refuses to
overwrite an existing migration attempt. Later sessions use
`main(initialize_backup=False)` to restore the remote version and compare
its identity/epoch. A completed experiment is not restarted. A partial
initialization or a failed remote verification requires inspection before
retrying; do not blindly rerun initialization.

**Kaggle training resumed from epoch 31 on 6 November 2026.** The full-data
preflight passed, including all payload hashes, the exact package versions
and loading epoch-30 optimizer state. A private backup was uploaded,
downloaded and verified by SHA before the runner resumed MLflow run
`85714466948d4e658ebb610a475dcf57`. Active epoch-31 batches were observed.
The first new epoch's backup remains pending until that epoch finishes.

Evidence: `reports/phase1/kaggle_preflight.json`,
`kaggle_resume_output.txt`, `kaggle_training_resumed.jpg`, and the current
`cloud_training_status.json`. The standalone three-cell recovery notebook is
`notebooks/MLOPS_EL_Kaggle_Resume.ipynb`; it restores the existing private
backup and does not initialize another experiment. Sixteen relevant local
tests pass; actual cloud round-trip and active training are separate evidence.

During preparation, the first data-copy attempt was interrupted after it
stalled in `sendfile`; buffered copying and direct mount paths completed.
The first backup dataset detail check returned 403 after creation; the
authenticated owned-dataset listing confirmed privacy, processing completed,
and the existing private version was recovered rather than reinitialized.
The installed Kaggle SDK is 2.0.2; the adapter accommodates its metadata API
and ZIP-wrapped large downloads. Failed setup attempts are preserved.

- Notebook prepared and opened in Colab; a Tesla T4 with 14.56 GiB VRAM was
  allocated and CUDA availability verified.
- Project dependencies installed. Colab's missing `jedi` dependency was
  repaired with version 0.19.2; `pip check` reports no broken requirements.
- Google Drive reports 4.99 TB available (11.86 GB of 5 TB used) when checked
  on 6 November 2026. This is ample for the project; no storage purchase is
  needed. Allow approximately 10 GB for the full archive and experiment
  records; the short benchmark is expected to use under 1 GB.
- The user authorized the Drive connection conditional on sufficient space.
  Drive mounted successfully, the archive passed verification, and the
  two-epoch benchmark completed successfully.
- The user authorized full training on 6 November 2026. The full archive is
  backed up in the private project Drive folder and passed checksum/payload
  verification. The first T4 session completed 30 epochs of MLflow run
  `85714466948d4e658ebb610a475dcf57`. Both dataset scans reported zero corrupt
  files. The session ended during epoch 31 validation; reconnecting a free
  GPU was refused due to Colab usage limits. The epoch 29/30 generations and
  epoch 30 last/best weights plus state metadata were observed in Drive.
  Automatic continuation is waiting for normal free GPU availability.
  See `reports/phase1/cloud_training_status.json`. Dataset faculty approval
  remains pending.

Notebook: `notebooks/MLOPS_EL_Free_GPU_Benchmark.ipynb`

Colab copy (access controlled by the user's Google account):
https://colab.research.google.com/drive/YOUR_NOTEBOOK_ID

## Exports and verification

| Archive in `exports/` | Images | Bytes | Purpose |
| --- | --- | ---: | --- |
| `mlops-benchmark-labels-v3.tar` | 256 train + 32 other training images | 284508160 | Two-epoch engineering benchmark |
| `mlops-trainval-labels-v3.tar` | 7000 train + 1500 validation | 6700103680 | Proposed full experiment |

Neither archive contains test images or labels. The benchmark's evaluation
images come from the training partition; its metrics cannot establish model
quality. It checks CUDA execution, speed, memory and checkpoint persistence.

Archive SHA256 values, image counts, DVC pointer hash and source Git revision
are recorded in `reports/phase1/cloud_export_benchmark.json` and
`reports/phase1/cloud_export_trainval.json`. Files are checksummed individually
in the embedded manifests. The small archive was extracted and all 586
payload files verified locally. The full archive's SHA and members were
checked without creating another full dataset copy.

Archives are ignored by Git and are not separately added to DVC: the source
data and pretrained weights already have DVC references. This avoids another
unnecessary archive copy. No DVC remote is configured.

## Execution and recovery

1. Select a free GPU runtime; stop if free GPU capacity is unavailable.
2. Execute the dependency cell, preserving Colab's CUDA Torch/torchvision
   versions. The runner records the actual resolved environment.
3. Initialize the embedded runner and authorize the Drive mount. Google
   grants notebook code access to read and modify Drive files; this requires
   the user's explicit authorization.
4. Upload the small archive when the notebook asks, or through Colab's Files
   sidebar into `/content`. The inline widget's chooser did not work in the
   controlled browser; the sidebar did. The verification cell waits up to
   30 minutes for the expected size, then checks SHA and every payload file
   while safely extracting regular files. An incomplete or corrupt upload
   fails before training.
5. Run the two-epoch YOLO11m benchmark at 960 pixels and batch 4.
6. Inspect `MyDrive/MLOPS_EL_cloud/benchmark/benchmark_report.json`, memory,
   timing, saved checkpoint hashes and optimizer state before proposing a
   full run. Successful loading is not evidence of a completed resume test.

Training reads images from the runtime's local disk, not directly from Drive.
The runner copies and verifies checkpoint generations to Drive after model
saves, retains two complete generations, and snapshots MLflow's SQLite
database through its backup API with the artifact directory. A disconnect
before the first successful save can still lose the current epoch's work.

The full-run cell now uses the authorized controller in
`cloud/start_full_training.py`. Settings are in
`configs/yolo11m_cloud_proposed.yaml`: all 7000 training images, 1500
validation images, up to 100 epochs, patience 20, 960 pixels, batch 4 subject
to the benchmark. Never use test data for parameter selection. Free runtimes
can disconnect or be unavailable; no paid upgrade or provider-limit bypass
is authorized.

## Validation of preparation

All 24 local tests passed. These include archive path/link rejection,
checksum and count checks, and checkpoint generation integrity. All eight
original notebook code cells passed Python syntax checks. After authorization,
all nine code cells (including the recovery drill and launch controller) passed
syntax checks; launch safeguards also cover incomplete uploads, completed
experiments, missing recovery evidence and attempts without a checkpoint.
The real CUDA benchmark also passed; its results are below.

## Measured benchmark result

MLflow run: `e8c17f3e33584b57ad732ffd92618dae`.
Machine-readable evidence: `reports/phase1/cloud_benchmark/`.
The recovered `mlops-benchmark-evidence.zip` includes the MLflow database,
artifacts, environment freeze, reports, CSV and checkpoint metadata, but
excludes weight files. Actual weight generations remain in Drive.

- T4, Torch 2.11.0+cu130, CUDA 13.0, Python 3.13.16.
- 76.33 seconds inside the training wrapper (not upload/setup time).
- Peak allocated CUDA memory 4.48 GiB; peak reserved 5.94 GiB of 14.56 GiB.
- Two persistent checkpoint generations verified by SHA; saved optimizer
  and epoch state successfully loaded. No actual interruption/resume test
  has yet been performed.
- Measured project Drive usage: 806713504 bytes (about 807 MB).
- CSV cumulative times: 20.72 seconds after epoch 1 and 48.50 after epoch 2.
  Scaling this small sample to 7000 images suggests roughly 10–15 minutes
  per epoch including full validation and checkpoint overhead, or about
  17–25 hours for 100 epochs. This is a rough projection: full-data object
  density, early stopping and free-runtime interruptions can change it.

The tiny benchmark is a feasibility check, not a quality experiment. Its
scores do not replace the existing 32.24% YOLO11s validation result. Full
training has now been authorized. A free Colab run may need multiple sessions;
confirmed storage capacity does not establish GPU session availability.

## Recovery drill and full run

The isolated recovery drill passed: a two-epoch engineering run was stopped
after epoch 1, the Python process exited, and its entire local work directory
was moved aside. A fresh interpreter restored the MLflow database, artifacts,
CSV/configuration and weights from Drive and continued at epoch 2 with the
same MLflow run ID `e7d3e7558015491e9cbf813ee1b922a5`. Optimizer state and the
resulting persistent checkpoint passed verification. Evidence is in
`reports/phase1/recovery_drill_report.json`. This simulates losing local state;
it is not a test of an actual provider-initiated disconnect.

Full training is up to 100 epochs with early stopping (patience 20), on the
original 7000/1500 train/validation split of labels-v3. Allow roughly 17–25
GPU hours if all 100 epochs run. Plan for 2–4 free sessions, not a guaranteed
count; availability, session length, data density, early stopping and setup
time can change the calendar duration. Re-estimate after full-data epochs.

Backups include the complete immutable input archive, two complete checkpoint
generations with SHA256 verification, optimizer/epoch state, best weights,
training CSV/configuration, and SQLite/MLflow artifact snapshots after saves.
The controller additionally preserves each session's environment freeze,
configuration, bundle manifest, GPU and source hashes under
`MyDrive/MLOPS_EL_cloud/train-session-records`. Resume refuses mismatched key
package versions. A mid-epoch interruption can lose work since the last
successfully persisted checkpoint; no checkpoint system guarantees zero loss.

For a new free GPU session, open the same saved notebook, run the CUDA and
dependency setup cells (1, 2, 3) and mount Drive (5). Skip already completed
benchmark and recovery-drill cells. Run the final full-training cell. It copies
the persistent archive to local runtime disk, verifies it, restores the latest
checkpoint and MLflow state, and resumes automatically. No dataset reupload is
needed once the Drive archive is complete. Completed experiments are not
silently restarted; attempts without a first checkpoint require inspection.

Keep the notebook and Drive upload tabs open while the initial upload is in
progress. Future reconnections may require the user's Google authorization or
a manual click to obtain a free GPU; these cannot be guaranteed unattended.

Launch evidence: `reports/phase1/full_training_launch.json`,
`full_training_launch_output.txt`, and `colab_full_training_started.jpg`.
Epoch 1 was actively processing when recorded. The first full-run checkpoint
is pending completion of epoch 1; the recovery procedure was verified in the
separate drill. There is no final candidate quality result yet.

## Authorized continuation

On 6 November 2026 the user explicitly authorized continuing this same full
experiment across free sessions without asking again. The active thread
heartbeat `continue-mlops-cloud-training` checks every 30 minutes, leaves a
running job alone, and attempts recovery through the verified controller when
a session ends. It reports meaningful changes or required user action and
stops after completion evidence is secured. Authentication, verification
prompts and free GPU availability can still prevent unattended recovery.
This authorization does not include paid compute, provider-limit bypasses,
new experiments or changes to the approved training settings.
