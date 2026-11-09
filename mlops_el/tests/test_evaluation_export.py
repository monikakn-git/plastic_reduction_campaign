"""Regression check for NumPy-valued per-class evaluation records."""
import importlib.util
import json
from pathlib import Path
import numpy as np


def test_per_class_metrics_export_as_json():
    spec=importlib.util.spec_from_file_location('evaluate',Path(__file__).parents[1]/'scripts/evaluate.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report={'per_class':[{'instances':np.int64(17),'recall':np.float32(.5)}],
            'counts':np.array([1,2,3],dtype=np.int64)}
    restored=json.loads(json.dumps(report,default=module.json_safe))
    assert restored=={'per_class':[{'instances':17,'recall':.5}],'counts':[1,2,3]}
