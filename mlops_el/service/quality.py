"""Separate artifact-integrity checks from unmet prospective production-quality gates."""
from service.model import MODEL_SHA256,SOURCE_RUN_ID,CLASSES

def assess(policy,evaluation,actual_sha):
    integrity={
        'trusted_weights_sha256':actual_sha==MODEL_SHA256==evaluation.get('weights_sha256'),
        'source_training_run':policy['source_training_run_id']==SOURCE_RUN_ID==evaluation.get('source_run_id'),
        'source_test_run':policy['source_test_run_id']==evaluation.get('run_id'),
        'frozen_test_split':evaluation.get('split')=='test' and evaluation.get('evaluated_images')==1500,
        'trained_dataset':policy['trained_dataset']=='labels-v3' and evaluation.get('prepared_path')=='data/prepared/plitterstreet-labels-v3',
        'complete_class_schema':[r['Class'] for r in evaluation.get('per_class',[])]==CLASSES,
    }
    metrics=evaluation['metrics']
    gates={'map50':metrics['metrics/mAP50(B)']>=policy['production_minimum_map50'],
           'recall':metrics['metrics/recall(B)']>=policy['production_minimum_recall'],
           'all_class_ap50':all(r['mAP50']>=policy['production_minimum_class_ap50'] for r in evaluation['per_class']),
           'faculty_dataset_approval':policy['faculty_dataset_approval']=='approved'}
    return {'integrity_checks':integrity,'integrity_passed':all(integrity.values()),'demo_eligible':all(integrity.values()),
            'production_checks':gates,'production_approved':all(integrity.values()) and all(gates.values()),
            'failed_classes':[r['Class'] for r in evaluation['per_class'] if r['mAP50']<policy['production_minimum_class_ap50']],
            'measured_test_metrics':metrics,'thresholds':{k:v for k,v in policy.items() if k.startswith('production_minimum')},
            'scope':'Demo only while production gates remain unmet; reviewed predictions cannot establish plastic material or campaign impact.'}
