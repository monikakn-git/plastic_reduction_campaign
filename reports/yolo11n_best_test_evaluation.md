# EcoPulse AI Model Evaluation

## Model Information

- Model: yolo11n_best
- Checkpoint: D:\mlops\artifacts\checkpoints\yolo11n_best.pt
- Dataset: D:\mlops\artifacts\datasets\processed\data.yaml
- Split: test
- Evaluation images: 1306
- Number of classes: 10
- Model size: 5.25 MB

## Performance Metrics

| Metric | Result |
|---|---:|
| Precision | 0.3690 |
| Recall | 0.3153 |
| mAP50 | 0.2721 |
| mAP50-95 | 0.1652 |

## Inference Latency

| Metric | Result |
|---|---:|
| Mean | 215.07 ms |
| Median | 194.41 ms |
| P95 | 331.28 ms |

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
