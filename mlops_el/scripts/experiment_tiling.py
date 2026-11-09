"""Paired validation-only pilot of overlapping native-image tiles, without retraining."""
from collections import Counter, defaultdict
import argparse
import gzip
import io
import json
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace

from train_candidate import ROOT, verify_inputs
from train import digest
from evaluate import json_safe
import mlflow
import numpy as np
from PIL import Image
import torch
from torchvision.ops import batched_nms
from ultralytics import YOLO, settings
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils.metrics import DetMetrics, box_iou

OUT=ROOT/'reports/phase1/tiling_pilot'


def starts(length, tile=960, overlap=.2):
    if length<=tile:return [0]
    stride=max(1,round(tile*(1-overlap)))
    return sorted(set(list(range(0,length-tile+1,stride))+[length-tile]))


def merge(predictions, iou=.7, maximum=300):
    if not predictions:return torch.empty((0,6))
    p=torch.cat(predictions)
    if not len(p):return p
    return p[batched_nms(p[:,:4],p[:,4],p[:,5],iou)[:maximum]]


def update(metric,pred,gt,name):
    iouv=torch.linspace(.5,.95,10)
    if len(gt) and len(pred):
        tp=BaseValidator.match_predictions(SimpleNamespace(iouv=iouv),pred[:,5],gt[:,0],box_iou(gt[:,1:5],pred[:,:4])).numpy()
    else:tp=np.zeros((len(pred),10),dtype=bool)
    metric.update_stats({'tp':tp,'conf':pred[:,4].numpy(),'pred_cls':pred[:,5].numpy(),
        'target_cls':gt[:,0].numpy(),'target_img':np.unique(gt[:,0].numpy()),'im_name':name})


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--full-validation',action='store_true',help='Use all validation images, preserving pilot reports')
    parser.add_argument('--output-dir',type=Path,help='Fresh directory for an explicitly requested repeat; completed results are never overwritten')
    args=parser.parse_args()
    full=args.full_validation
    if full:OUT=ROOT/'reports/phase1/tiling_full_validation'
    if args.output_dir:OUT=(ROOT/args.output_dir).resolve()
    if (OUT/'results.json').exists():raise SystemExit('Completed results already exist; refusing to overwrite this experiment')
    os.chdir(ROOT)
    os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT','1')
    verify_inputs()
    if not torch.backends.mps.is_available():raise SystemExit('Native MPS required')
    OUT.mkdir(parents=True,exist_ok=True)
    training=json.loads((ROOT/'reports/phase1/yolo11s_candidate_training.json').read_text())
    weights=ROOT/training['model']
    if digest(weights)!=training['model_sha256']:raise SystemExit('Weight hash mismatch')
    splits=ROOT/'data/prepared/plitterstreet/splits.json'
    sample=sorted([s for s in json.loads(splits.read_text()) if s['split']=='val'],key=lambda s:s['image_id'])
    random.Random(42).shuffle(sample)
    if not full:sample=sample[:100]
    if full and len(sample)!=1500:raise SystemExit('Expected the approved 1,500-image validation split')
    count=len(sample)
    scope=f'{count}-image paired full-validation comparison' if full else '100-image paired validation pilot; not the full validation set'
    data=json.loads((ROOT/'data/raw/plitterstreet/annotations.json').read_text())
    objects=defaultdict(list)
    for a in data['annotations']:objects[a['image_id']].append(a)
    names={x['id']-1:x['name'].replace(' ','_') for x in data['categories']}
    counts=Counter(names[a['category_id']-1] for s in sample for a in objects[s['image_id']])
    manifest={'seed':42,'split':'val','images':sample,'object_counts':dict(counts),
        'model_sha256':digest(weights),'split_manifest_sha256':digest(splits),
        'selection':'All validation images in seeded order; no test images.' if full else 'First 100 from a seeded shuffle of validation image IDs, selected before predictions. No test images.'}
    (OUT/'sample.json').write_text(json.dumps(manifest,indent=2))
    metrics={mode:DetMetrics(names=names) for mode in ('whole','tiled')}
    settings.update({'mlflow':False})
    mlflow.set_tracking_uri(f"sqlite:///{ROOT/'mlflow.db'}");mlflow.set_experiment('plastic-waste-phase1')
    with mlflow.start_run(run_name=('yolo11s-tiling-full-' if full else 'yolo11s-tiling-pilot-')+time.strftime('%Y%m%d-%H%M%S')) as run:
        mlflow.set_tags({'phase':'1','profile':'validation-tiling-full' if full else 'validation-tiling-pilot','dataset_approval':'pending',
            'training_run_id':training['run_id'],'weights_sha256':digest(weights),
            'note':'Full paired validation; no test evaluation or automatic promotion.' if full else 'Exploratory subset; never compare its scores directly with full-validation scores.'})
        mlflow.log_params({'sample_images':count,'seed':42,'tile_size_native':960,'overlap':.2,
            'imgsz':640,'conf':.001,'local_nms_iou':.7,'global_nms_iou':.7,
            'max_det_per_image':300,'rect':False,'prediction_multi_label':False,'tile_batch':2,'whole_pass_included':True,'retraining':False})
        for p in [Path(__file__),OUT/'sample.json',ROOT/'requirements-lock.txt']:
            mlflow.log_artifact(str(p),artifact_path='provenance')
        model=YOLO(str(weights))
        options=dict(device='mps',imgsz=640,conf=.001,iou=.7,max_det=300,rect=False,verbose=False,save=False)
        model.predict(np.zeros((640,640,3),dtype=np.uint8),**options)
        records=[]
        for index,s in enumerate(sample):
            content=(ROOT/'data/raw/plitterstreet/images'/s['file_name']).read_bytes()
            if content[:2]==b'\xff\xd8' and not content.endswith(b'\xff\xd9'):content+=b'\xff\xd9'
            with Image.open(io.BytesIO(content)) as im:rgb=np.array(im.convert('RGB'))
            bgr=rgb[:,:,::-1].copy();height,width=bgr.shape[:2]
            torch.mps.synchronize();t=time.perf_counter()
            whole=model.predict(bgr,**options)[0].boxes.data.cpu()[:,:6]
            whole_seconds=time.perf_counter()-t
            t=time.perf_counter();pieces=[whole.clone()];tile_count=0
            coordinates=[(x,y) for y in starts(height) for x in starts(width)]
            for offset in range(0,len(coordinates),2):
                positions=coordinates[offset:offset+2]
                tiles=[bgr[y:min(y+960,height),x:min(x+960,width)] for x,y in positions]
                results=model.predict(tiles,batch=2,**options)
                for result,(x,y) in zip(results,positions):
                    p=result.boxes.data.cpu()[:,:6].clone()
                    p[:,[0,2]]+=x;p[:,[1,3]]+=y
                    pieces.append(p);tile_count+=1
            tiled=merge(pieces)
            extra_seconds=time.perf_counter()-t
            gt=[]
            for a in objects[s['image_id']]:
                x,y,w,h=a['bbox'];gt.append([a['category_id']-1,x,y,x+w,y+h])
            gt=torch.tensor(gt,dtype=torch.float32).reshape(-1,5)
            update(metrics['whole'],whole,gt,s['file_name']);update(metrics['tiled'],tiled,gt,s['file_name'])
            records.append({'image_id':s['image_id'],'tile_count':tile_count,'whole_seconds':whole_seconds,
                'tiled_seconds_including_whole_pass':whole_seconds+extra_seconds,
                'whole_predictions':whole.tolist(),'tiled_predictions':tiled.tolist()})
            progress={'completed_images':index+1,'total':count,'run_id':run.info.run_id,'latest_tile_count':tile_count}
            (ROOT/('.local/tiling-full-progress.json' if full else '.local/tiling-pilot-progress.json')).write_text(json.dumps(progress))
            if (index+1)%10==0:print(json.dumps(progress),flush=True)
        report={'run_id':run.info.run_id,'scope':scope,'image_count':count,
            'model_sha256':digest(weights),'object_counts':dict(counts),'retraining':False,'test_used':False,'results':{}}
        report['prediction_protocol']={'multi_label':False,'rect':False,'imgsz':640,
            'standard_validator_difference':'The installed default validator uses multi_label=True; prediction uses False. Historical validation scores use a different postprocessing protocol.'}
        for mode,metric in metrics.items():
            directory=OUT/mode;directory.mkdir(exist_ok=True)
            metric.process(save_dir=directory,plot=True)
            result={k:float(v) for k,v in metric.results_dict.items()}
            report['results'][mode]={'metrics':result,'per_class':metric.summary()}
            mlflow.log_metrics({f'{mode}_{k}'.replace('(','_').replace(')','_'):v for k,v in result.items()})
        report['timing']={'whole_mean_seconds':float(np.mean([r['whole_seconds'] for r in records])),
            'tiled_mean_seconds':float(np.mean([r['tiled_seconds_including_whole_pass'] for r in records]))}
        report['limitations']=['Several classes have only 4–8 ground-truth objects in this subset; class scores are highly uncertain.',
            'No confidence intervals or proof of full-dataset improvement; exploratory pilot only.',
            'Uses original unmodified labels, including unresolved ambiguity.',
            'Custom overlapping tiling plus class-aware NMS, not the SAHI library. Includes whole-image predictions to retain large-object context.',
            'Whole and tile inputs are square-letterboxed at 640; this pilot is not numerically interchangeable with the earlier full-validation loader.',
            'Prediction keeps one class per box; the installed standard validator uses multi-label NMS. Do not mix these scores with standard-validation scores.',
            'Timing includes inference and merge, excludes image decoding, and is not a controlled hardware benchmark.']
        if full:
            report['limitations']=report['limitations'][2:]
            report['limitations'][2]='Whole and tile inputs are square-letterboxed at 640; this comparison is not numerically interchangeable with the earlier validation loader. Compare the paired methods here.'
            report['limitations'].extend(['Validation data has been used for method selection; this is not independent test performance.',
                'No confidence intervals or campus generalization claims. Labels remain ambiguous in some reviewed cases.'])
        verify_inputs()
        (OUT/'results.json').write_text(json.dumps(report,indent=2,default=json_safe))
        if full:
            with gzip.open(OUT/'predictions.json.gz','wt',encoding='utf-8') as f:json.dump(records,f,separators=(',',':'))
        else:(OUT/'predictions.json').write_text(json.dumps(records,separators=(',',':')))
        text=['# '+('Full-validation tiling comparison: YOLO11s' if full else 'Paired tiling pilot: YOLO11s'),'',report['scope']+'. Same model weights; no training or label changes.','',
            '| Mode | mAP@50 | mAP@50–95 | Mean inference + processing time/image |','| --- | ---: | ---: | ---: |']
        for mode in ('whole','tiled'):
            m=report['results'][mode]['metrics'];sec=report['timing'][f'{mode}_mean_seconds']
            text.append(f"| {mode} | {m['metrics/mAP50(B)']:.2%} | {m['metrics/mAP50-95(B)']:.2%} | {sec:.2f} s |")
        text += ['','| Class | Objects | Whole AP@50 | Tiled AP@50 |','| --- | ---: | ---: | ---: |']
        old={r['Class']:r for r in report['results']['whole']['per_class']}
        for r in report['results']['tiled']['per_class']:
            text.append(f"| {r['Class']} | {counts[r['Class']]} | {old[r['Class']]['mAP50']:.2%} | {r['mAP50']:.2%} |")
        interpretation='Compare the two methods on this same full validation split and matching inference/evaluation settings. The earlier 32.2% used a different validation loader and multi-label NMS, whereas this prediction comparison uses single-label NMS. It is retained separately. No automatic inference-setting change or model promotion is performed, and the test set has not been evaluated again.' if full else 'Compare only the two columns on this same subset. Do not compare either score directly with the earlier 32.2% on 1,500 validation images. Rare-class support is especially low. No automatic inference-setting change or model promotion is performed. A full-validation comparison would require separate approval.'
        text += ['','## Interpretation','',interpretation,'',
            '## Method and limitations','',
            'Overlapping 960-native-pixel windows, 20% overlap, each resized/letterboxed to 640. Whole-image predictions are included. Global class-aware NMS at IoU 0.7 removes duplicates, with at most 300 final predictions per image. Ultralytics matching and AP routines evaluate original full-image boxes at IoU 0.50–0.95.','',
            'Conceptual reference: [Ultralytics sliced-inference guide](https://docs.ultralytics.com/guides/sahi-tiled-inference/). This experiment implements tiling with existing installed dependencies; it does not install or claim to run SAHI.','']
        text += ['- '+x for x in report['limitations']]+['']
        (OUT/'REPORT.md').write_text('\n'.join(text))
        for p in OUT.rglob('*'):
            if p.is_file():mlflow.log_artifact(str(p),artifact_path='pilot')
        print('\n'.join(text),flush=True)


if __name__=='__main__':main()
