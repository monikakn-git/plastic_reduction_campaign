"""MLflow Python-function packaging for the same inference implementation."""
import base64
import binascii
import json
import mlflow.pyfunc
from service.model import Detector,decode_image,MAX_BYTES

class WasteDetectorModel(mlflow.pyfunc.PythonModel):
    def load_context(self,context):self.detector=Detector(context.artifacts['weights'],device='cpu')

    def predict(self,context,model_input,params=None):
        if list(model_input.columns)!=['image_base64'] or len(model_input)!=1:raise ValueError('Provide exactly one image_base64 row')
        value=model_input.iloc[0]['image_base64']
        if not isinstance(value,str) or len(value)>4*((MAX_BYTES+2)//3):raise ValueError('Image input exceeds the supported size')
        try:payload=base64.b64decode(value,validate=True)
        except (binascii.Error,ValueError) as exc:raise ValueError('Invalid base64 image') from exc
        return [json.dumps(self.detector.predict(decode_image(payload)),allow_nan=False)]
