"""Trusted model loading and bounded image-only inference. No training or downloads."""
import hashlib
import io
import threading
import warnings
from pathlib import Path
from PIL import Image,ImageOps,UnidentifiedImageError,JpegImagePlugin,PngImagePlugin

MODEL_SHA256='a9a39b52f2de03927ce5f387a6feb52295468e50262e436d75ee6a0a95ed1b8e'
SOURCE_RUN_ID='85714466948d4e658ebb610a475dcf57'
CLASSES=['litter','pile','face_mask','trash_bin','plastic_bag','bottle','cup','rope','sachet','straw']
MAX_BYTES=8*1024*1024
MAX_PIXELS=20_000_000
IMGSZ=960
CONFIDENCE=.25
NMS_IOU=.7

class InvalidImage(ValueError):pass
class ModelBusy(RuntimeError):pass

def file_sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def decode_image(content):
    if not content or len(content)>MAX_BYTES:raise InvalidImage('Image must contain between 1 byte and 8 MiB')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error',Image.DecompressionBombWarning)
            if content.startswith(b'\xff\xd8\xff'):decoder=JpegImagePlugin.JpegImageFile
            elif content.startswith(b'\x89PNG\r\n\x1a\n'):decoder=PngImagePlugin.PngImageFile
            else:raise InvalidImage('Only JPEG and PNG images are supported')
            # Direct decoders avoid optional global Image.open plugins installed by dependencies.
            with decoder(io.BytesIO(content)) as image:
                if image.format not in ('JPEG','PNG'):raise InvalidImage('Only JPEG and PNG images are supported')
                w,h=image.size
                if w<16 or h<16 or max(w,h)>8192 or w*h>MAX_PIXELS:raise InvalidImage('Image dimensions exceed supported limits')
                if getattr(image,'n_frames',1)!=1:raise InvalidImage('Animated images are not supported')
                image.load()
                return ImageOps.exif_transpose(image).convert('RGB')
    except InvalidImage:raise
    except (UnidentifiedImageError,OSError,ValueError,SyntaxError,Image.DecompressionBombWarning,Image.DecompressionBombError) as exc:
        raise InvalidImage('Invalid or unsupported image') from exc

class Detector:
    def __init__(self,weights,device='cpu'):
        if device not in ('cpu','mps'):raise ValueError('Supported devices: cpu or mps')
        if file_sha(weights)!=MODEL_SHA256:raise ValueError('Model checksum does not match the frozen evaluated candidate')
        import torch
        from ultralytics import YOLO,settings
        torch.set_num_threads(2)
        settings.update({'mlflow':False})
        self.model=YOLO(str(Path(weights).resolve()))
        actual=[self.model.names[i] for i in range(len(self.model.names))]
        if actual!=CLASSES:raise ValueError('Model class schema differs from the evaluated candidate')
        self.device=device;self.lock=threading.Lock()

    def predict(self,image):
        if not self.lock.acquire(blocking=False):raise ModelBusy('One inference is already running')
        try:
            result=self.model.predict(source=image,imgsz=IMGSZ,conf=CONFIDENCE,iou=NMS_IOU,device=self.device,verbose=False,save=False,augment=False,max_det=300)[0]
            width,height=image.size;detections=[]
            for box in result.boxes:
                cls=int(box.cls.item());score=float(box.conf.item());coords=box.xyxy[0].detach().cpu().tolist()
                coords=[min(float(width if i%2==0 else height),max(0.,float(v))) for i,v in enumerate(coords)]
                detections.append({'class_id':cls,'label':CLASSES[cls],'confidence':round(score,6),'bbox_xyxy':[round(v,3) for v in coords]})
            return {'width':width,'height':height,'detections':detections}
        finally:self.lock.release()
