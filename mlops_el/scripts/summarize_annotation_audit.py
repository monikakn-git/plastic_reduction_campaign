"""Validate review records and summarize the completed visual annotation audit."""
from collections import Counter
import csv
import json
import os
from pathlib import Path

from review_annotations import ROOT, digest

OUT = ROOT/"reports/phase1/annotation_audit_100"


def main():
    manifest=json.loads((OUT/"sample_manifest.json").read_text())
    findings=json.loads((OUT/"visual_findings.json").read_text())
    technical=json.loads((OUT/"technical_checks.json").read_text())
    rows=manifest["samples"]
    observations={x["sample_id"]:x for x in findings["findings"]}
    if len(rows)!=100 or len(observations)!=100 or {x['sample_id'] for x in rows}!=set(observations):
        raise RuntimeError("Audit sample/review identity mismatch")
    if len({x['image_id'] for x in rows})!=100:
        raise RuntimeError("Review sample must contain 100 distinct images")
    assignment={x['image_id']:x for x in json.loads((ROOT/'data/prepared/plitterstreet/splits.json').read_text())}
    for p,expected in manifest['protected_sha256'].items():
        if digest(ROOT/p)!=expected:raise RuntimeError(f"Protected input changed: {p}")
    for item in rows:
        if item['split'] not in ('train','val') or assignment[item['image_id']]['split']!=item['split']:
            raise RuntimeError("Unexpected split in audit sample")
        image=ROOT/'data/raw/plitterstreet/images'/item['file_name']
        label=ROOT/f"data/prepared/plitterstreet/labels/{item['split']}"/Path(item['file_name']).with_suffix('.txt')
        if digest(image)!=item['image_sha256'] or digest(label)!=item['prepared_label_sha256']:
            raise RuntimeError(f"Sample image or prepared label changed: {item['sample_id']}")
    old=json.loads((ROOT/'reports/phase1/annotation_review/sample_manifest.json').read_text())
    overlap=len({(r['image_id'],r['annotation_id']) for r in rows}&{(r['image_id'],r['annotation_id']) for r in old['samples']})
    counts=Counter(x['class_assessment'] for x in observations.values())
    boxflags=[x for x in observations.values() if x['box_assessment']!='broadly_aligned_no_specific_issue_identified']
    summary={'audited_objects':100,'distinct_images':100,'counts':dict(counts),'box_flags':len(boxflags),
        'all_sampled_boxes_in_bounds':technical['all_boxes_in_bounds'],
        'all_sampled_prepared_labels_match':technical['all_prepared_labels_match_source'],
        'protected_inputs_unchanged':True,'overlap_with_previous_24_target_review':overlap,
        'confirmed_label_corrections':0,'test_images_reviewed':0,'dataset_modified':False,
        'interpretation':'Visual assessments are provisional, not confirmed errors or dataset-wide error rates.'}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    queue=[]
    for r in rows:
        o=observations[r['sample_id']]
        if o['class_assessment']=='suspected_class_mismatch' or o in boxflags:
            queue.append({**r,**o,'human_decision':'pending','approved_new_class':None,'approved_new_bbox':None})
    (OUT/'adjudication_queue.json').write_text(json.dumps({'scope':'Review queue only; no corrections applied. Validation cases must be adjudicated independently of prediction scores.','items':queue},indent=2))
    with (OUT/'review_table.csv').open('w',newline='') as f:
        columns=['sample_id','class','split','image_id','annotation_id','file_name','class_assessment','box_assessment','note']
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for r in rows:writer.writerow({k:({**r,**observations[r['sample_id']]}[k]) for k in columns})
    text=['# Bottle and cup annotation audit: 100 targets','',
        'Completed an AI-assisted visual inspection of 100 labelled targets in 100 distinct images: 25 bottles and 25 cups from each of training and validation. No test images or model predictions were used. No labels, images, splits, preprocessing settings or model weights were changed.','',
        '## What was verified','',
        '- All 100 sampled boxes have positive dimensions and lie within their images.',
        '- All 100 sampled source labels match their prepared YOLO class and normalized coordinates within 1e-6. This checks conversion, not semantic truth.',
        '- Hashes of sampled images and label files, full annotation JSON, splits, baseline/candidate weights, baseline configuration and protected evaluation reports remained unchanged.','',
        '## Visual findings','',
        '| Sample | Label plausible | Cannot confidently judge | Suspected class mismatch |',
        '| --- | ---: | ---: | ---: |']
    for label in ('bottle','cup'):
        for split in ('train','val'):
            group=[r for r in rows if r['class']==label and r['split']==split]
            c=Counter(observations[r['sample_id']]['class_assessment'] for r in group)
            text.append(f"| {label} / {split} | {c['plausible']} | {c['uncertain']} | {c['suspected_class_mismatch']} |")
    text += [f"| **Total** | **{counts['plausible']}** | **{counts['uncertain']}** | **{counts['suspected_class_mismatch']}** |",'',
        '**These are reviewer judgments, not confirmed error counts.** Uncertain cases are frequently tiny, motion-blurred, crushed, dirty or obscured. The size-stratified, group-balanced sample is not representative enough to estimate a dataset-wide error rate or prove why a model scored poorly.','',
        'Ten targets have competing visual interpretations (for example carton-like items labelled cup, or flat packet/disc-like items labelled bottle). Four additional box/instance flags were recorded: one possible truncated boundary and three possible multi-object boxes. These require human adjudication; none was automatically corrected.','',
        '### Specific follow-up evidence','',
        '- `bottle-train-large-01`: the target box appears to exclude part of a blue cap/neck. Review tightness at source resolution.',
        '- `bottle-train-large-09`, image 14677: several visible bottles share one bottle box; another nearby bottle-like item carries the residual `litter` label. This is a class-specificity/grouping consistency flag, not an unlabelled-object discovery. See `context_detail_14677.jpg` and `context_findings.json`.',
        '- `cup-val-small-04` and `cup-val-medium-03`: carton/packet-like shapes carrying cup labels. These are suspected mismatches, pending a human decision.',
        '- `cup-val-large-04` and `cup-val-large-06`: crushed can/container-like objects with cup labels. Ambiguous shape does not establish material.',
        '- Image 5720: both visible bottles already have individual boxes; an extra context check prevented incorrectly calling the neighbouring bottle unlabelled.','',
        '**Missing annotations:** no missing-object correction was established. All source boxes were shown in context previews, with magnified all-label context checks in two selected scenes. This is not exhaustive object-by-object completeness verification; small previews and unknown discarded/functional status limit that conclusion.','',
        '## Class definitions and source evidence','',
        'The authors provide broad annotation guidelines and acknowledge interpretation difficulty. They do not resolve the specific cup/lid/carton boundary cases in the documentation reviewed. Source links, relevant section and findings are recorded in [source_definitions.json](source_definitions.json).',
        'The source JSON residual class is named `litter`; the paper uses `other plastic`. No taxonomy remapping was applied during this audit. Neither a source class label nor visual inspection proves that every object is plastic.','',
        '## Sampling and limits','',
        'Seed 2026; 8 small, 8 medium and 9 large targets per class/split, based on within-pool projected short-side tertiles at 640. Seeded ordering and a least-used inferred-group preference increase scene coverage. All images are distinct. Validation groups necessarily repeat: cup validation has only 14 available inferred groups. Grouping is a filename proxy, not verified geographic independence.',
        f'This audit contains {overlap} targets from the earlier 24-target review; it is not 100 additional independent targets. Original, magnified detail and resized-detail views were inspected. Contact sheets contain five examples each, with red target boxes and grey boxes for other source annotations. The model-input views illustrate aspect-preserving resizing, not every training augmentation.','',
        '## Recommendation before another training run','',
        'Review the 14 high-priority entries in `adjudication_queue.json` with a human reviewer and a written bottle/cup inclusion policy. The queue contains ten suspected class cases and four box/instance cases; it is not an approved correction file. Keep the other ambiguous examples available for discussion. Do not infer replacement classes or delete hard examples simply to raise validation scores.',
        'If corrections are approved later, create a versioned correction manifest over source image/annotation IDs and rebuild a new derived DVC version, retaining the originals and current baseline. Validation corrections must follow the same independent policy; scores across changed labels need explicit qualification. Do not modify the existing test labels or use test predictions for tuning.',
        'The audit found no conversion defect in the sampled labels, so it does not justify rewriting the folder structure or blindly changing preprocessing. No new training or automatic model promotion was performed. Faculty approval remains pending.','',
        '## Reproduction and records','',
        'Run `.venv/bin/python scripts/audit_annotations.py` to regenerate the fixed sample/contact sheets, then `.venv/bin/python scripts/summarize_annotation_audit.py` to validate and summarize the retained visual judgments. Visual judgments require inspection; they are not generated by the preparation script.','',
        '- `sample_manifest.json`: selection, exact IDs, coordinates, groups and protected hashes.',
        '- `visual_findings.json` and `review_table.csv`: all 100 visual observations.',
        '- `technical_checks.json`: source-to-prepared label checks.',
        '- `adjudication_queue.json`: pending human decisions.',
        '- `context_findings.json` and two context-detail images: additional label-coverage/grouping evidence.',
        '- `summary.json`: actual counts and preservation status.','']
    text += ['## Contact sheets','', '| Class / split | Five targets per page |','| --- | --- |']
    for label in ('bottle','cup'):
        for split in ('train','val'):
            links=' · '.join(f'[{i}]({label}_{split}_{i:02d}.jpg)' for i in range(1,6))
            text.append(f'| {label} / {split} | {links} |')
    text.append('')
    (OUT/'REPORT.md').write_text('\n'.join(text))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
