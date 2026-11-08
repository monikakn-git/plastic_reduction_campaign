# EcoPulse AI Model Evaluation

## Model Information

- Model: yolov8n_best
- Checkpoint: D:\mlops\artifacts\checkpoints\yolov8n_best.pt
- Dataset: D:\mlops\artifacts\datasets\processed\data.yaml
- Split: test
- Evaluation images: 1306
- Number of classes: 10
- Model size: 6.2 MB

## Performance Metrics

| Metric | Result |
|---|---:|
| Precision | 0.3626 |
| Recall | 0.3296 |
| mAP50 | 0.2761 |
| mAP50-95 | 0.1675 |

## Inference Latency

| Metric | Result |
|---|---:|
| Mean | 219.72 ms |
| Median | 205.04 ms |
| P95 | 342.77 ms |

## Quality Gate

- Decision: PASSED
- Minimum mAP50: 0.2
- Maximum mean latency: 500.0 ms
- Passed: True

## Class Mapping

- 0: litter
- 1: pile
- 2: face_mask
- 3: trash_bin
- 4: plastic_bag
- 5: bottle
- 6: cup
- 7: rope
- 8: sachet
- 9: straw
