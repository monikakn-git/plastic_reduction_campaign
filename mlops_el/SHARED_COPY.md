# Shared Phase 1–2 snapshot

This folder contains a public-safe copy of the MLOPS_EL project at source commit `61b92ae89fa5a593d02b0032fd4e430ebbee3585` (see `SHARED_COPY_MANIFEST.json` for the full authoritative commit). Work from this folder when using its scripts; the existing implementation at the parent repository root is separate.

Included: source code, tests, configuration, dataset/model DVC pointers, experiment summaries, evaluation charts, review material, MLflow registry manifests, Docker files and recorded CI/CD evidence. Model test mAP50 is 37.12%; it is a demonstration candidate with explicit limitations, not production-approved. Dataset faculty approval remains pending. The original project's 58-test result and CI/CD evidence are historical execution evidence; this shared copy has not independently reproduced training or deployment.

Not included: the full 10,000-image dataset, trained weights, local MLflow databases/artifacts, DVC cache, secrets, local tools or source repository history. Account UI screenshots and a benchmark ZIP containing a local database were omitted. Personal absolute project paths and private Drive notebook/folder links were replaced with placeholders. Dataset source/attribution remains in the project documentation. Dataset sample annotations and charts are evidence, not the full dataset.

The DVC pointers alone do not fetch data: the original DVC remote was unconfigured. Approved data/model artifacts must be supplied separately before full training or real-model inference can run. The release requires the exact weights SHA256 recorded in its manifest; do not replace it with arbitrary weights or claim the recorded metrics for another model.

The historical cloud helpers contain original account-specific Kaggle dataset defaults. They provide no access to those private resources; configure your own authorized inputs and backup resources before reuse. Private GitHub run links in historical reports require access to their originating repository, although copied text logs provide review evidence.

The copied `.github/workflows/` files are nested under this folder and are references, not active repository-level workflows. The original temporary runner was removed. No runner registration, secrets, private-repository permissions or access to the author's Mac is transferred. This copy can be edited independently; it cannot change the original private repository. Its author must separately approve any future data, model or credential sharing.

Do not run the scripts blindly: adapt paths, dependencies and cloud configuration. Running this copy requires choosing and configuring your own environment. Copying code does not start an application or training run.

Start with [README.md](README.md), [Phase 2 final report](reports/phase2/PHASE2_FINAL_REPORT.md) and [operations](docs/PHASE2.md). Planned Phase 3 monitoring, drift/retraining governance and location/campaign features are not yet completed here.

## Visual refresh

The existing `mlops-phase2` branch now also includes the original project's visual refresh from commit `f330a59`: the separate `service/demo.html`, matching FastAPI root page loader, Docker build-context allowlist and static preview/check records. Model weights, prediction behavior and thresholds are unchanged. The 21 checks recorded with the visual refresh passed in the original workspace; this shared copy has not rerun real-model inference or Docker deployment. Rebuild a properly configured local image to use the new interface. The repository root implementation and master branch remain separate.
