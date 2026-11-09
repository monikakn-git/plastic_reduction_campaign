"""Check paired whole-image AP using the standard validator with single-label NMS."""
import json
from train_candidate import ROOT,verify_inputs
from ultralytics import YOLO,settings
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import nms


class SingleLabelValidator(DetectionValidator):
    def postprocess(self,preds):
        outputs=nms.non_max_suppression(preds,self.args.conf,self.args.iou,nc=self.nc,
            multi_label=False,agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,end2end=self.end2end,rotated=False)
        return [{'bboxes':x[:,:4],'conf':x[:,4],'cls':x[:,5],'extra':x[:,6:]} for x in outputs]


def main():
    verify_inputs();settings.update({'mlflow':False})
    results=YOLO(str(ROOT/'models/plitterstreet-yolo11s-candidate.pt')).val(
        validator=SingleLabelValidator,data=str(ROOT/'.local/yolo11s-candidate-validation.yaml'),
        split='val',device='mps',imgsz=640,batch=1,workers=0,conf=.001,iou=.7,
        rect=False,plots=False,project=str(ROOT/'runs'),name='tiling-single-label-reference',exist_ok=False)
    record={'method':'Standard Ultralytics validator with single-label NMS, square padding and batch 1; same full validation split.',
        'metrics':{k:float(v) for k,v in results.results_dict.items()},
        'note':'Only NMS multi_label differs from the default validator. Metric computation and dataset loading are standard.'}
    p=ROOT/'reports/phase1/tiling_full_validation/single_label_library_reference.json'
    p.write_text(json.dumps(record,indent=2));verify_inputs();print(p.read_text())


if __name__=='__main__':main()
