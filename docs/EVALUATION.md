# Evaluation Protocols

This document describes the evaluation metrics and protocols used in IndoorCD.

## Overview

We use three evaluation levels:

1. **Point-Level**: Per-point classification accuracy
2. **Box-Level**: Bounding box detection metrics (IoU-based)
3. **Scene-Level**: Binary scene classification

---

## 1. Point-Level Evaluation

### Classes
- **NoChange (0)**: Points present in both scans
- **Add (1)**: Points only in comparison scan
- **Remove (2)**: Points only in reference scan

### Metrics

**Accuracy**
```
Accuracy = (TP_nc + TP_add + TP_rem) / Total_Points
```

**Per-Class Precision, Recall, F1**
```
Precision_c = TP_c / (TP_c + FP_c)
Recall_c = TP_c / (TP_c + FN_c)
F1_c = 2 * (Precision_c * Recall_c) / (Precision_c + Recall_c)
```

**Macro F1**
```
Macro_F1 = (F1_nochange + F1_add + F1_remove) / 3
```

### Usage

```python
from src.evaluation.metrics import PointLevelEvaluator

evaluator = PointLevelEvaluator()

# pred_labels, gt_labels: arrays of class indices (0, 1, 2)
metrics = evaluator.evaluate(pred_labels, gt_labels)

print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"Add F1: {metrics['f1']['Add']:.3f}")
print(f"Remove F1: {metrics['f1']['Remove']:.3f}")
```

---

## 2. Box-Level Evaluation

### IoU Calculation

```
IoU(pred, gt) = Volume(pred ∩ gt) / Volume(pred ∪ gt)
```

For axis-aligned bounding boxes:
```python
def compute_iou(box1, box2):
    # Intersection
    inter_min = np.maximum(box1.min, box2.min)
    inter_max = np.minimum(box1.max, box2.max)
    inter_dims = np.maximum(inter_max - inter_min, 0)
    inter_vol = np.prod(inter_dims)
    
    # Union
    vol1 = np.prod(box1.max - box1.min)
    vol2 = np.prod(box2.max - box2.min)
    union_vol = vol1 + vol2 - inter_vol
    
    return inter_vol / union_vol if union_vol > 0 else 0
```

### IoU Threshold Selection

We use **IoU = 0.25** as the default threshold.

**Mathematical Justification**:
For objects smaller than half the room volume, IoU ≥ 0.25 ensures meaningful overlap while being achievable given LiDAR noise and annotation uncertainty.

```
IoU = 0.25 corresponds to:
- Predicted box overlapping ~40% of ground truth
- Or ground truth overlapping ~40% of predicted box
```

### Matching Strategy

1. Compute IoU matrix between predictions and ground truths
2. Use Hungarian algorithm for optimal assignment
3. Filter matches below IoU threshold
4. Compute TP, FP, FN

```python
from src.evaluation.metrics import BoxLevelEvaluator

evaluator = BoxLevelEvaluator(iou_threshold=0.25)

# pred_boxes: List[DetectedBox]
# gt_boxes: List[GroundTruthBox]
metrics = evaluator.evaluate(pred_boxes, gt_boxes)

print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
print(f"F1: {metrics['f1']:.3f}")
print(f"Mean IoU: {metrics['mean_iou']:.3f}")
```

### Per-Class Metrics

Separate evaluation for Add and Remove classes:

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Add | P_add | R_add | F1_add |
| Remove | P_rem | R_rem | F1_rem |
| **Macro** | (P_add + P_rem)/2 | (R_add + R_rem)/2 | (F1_add + F1_rem)/2 |

---

## 3. Scene-Level Evaluation

Binary classification: Does the scene contain any changes?

### Metrics

**Accuracy**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Change Detection Rate (Recall)**
```
CDR = TP / (TP + FN)
```

**False Alarm Rate**
```
FAR = FP / (FP + TN)
```

### Usage

```python
from src.evaluation.metrics import SceneLevelEvaluator

evaluator = SceneLevelEvaluator()

# has_change_pred: bool
# has_change_gt: bool
metrics = evaluator.evaluate(predictions, ground_truths)

print(f"Scene Accuracy: {metrics['accuracy']:.3f}")
print(f"Change Detection Rate: {metrics['cdr']:.3f}")
```

---

## Benchmark Protocol

### Standard Evaluation

```bash
python run_benchmark.py \
    --data_path ./Dataset \
    --output_dir ./results \
    --iou_threshold 0.25 \
    --seed 42
```

### Per-Category Evaluation

Evaluate separately for each change type (Add, Remove, Move, Composite):

```bash
python run_benchmark.py \
    --data_path ./Dataset \
    --per_category
```

### Cross-Validation

For more robust evaluation:

```python
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
    # Train and evaluate
    ...
```

---

## Result Format

### JSON Output

```json
{
  "timestamp": "20250108_120000",
  "iou_threshold": 0.25,
  "methods": {
    "multi_stage": {
      "metrics": {
        "accuracy": 0.957,
        "macro_precision": 0.312,
        "macro_recall": 0.267,
        "macro_f1": 0.273,
        "precision": {"Add": 0.298, "Remove": 0.326},
        "recall": {"Add": 0.178, "Remove": 0.356},
        "f1": {"Add": 0.207, "Remove": 0.288}
      },
      "num_scenes": 123,
      "avg_time": 0.82
    }
  }
}
```

### LaTeX Tables

```python
benchmark.generate_latex_tables()
```

Outputs publication-ready LaTeX:

```latex
\begin{table}[t]
\centering
\caption{Box-Level Evaluation Results}
\begin{tabular}{lcccc}
\toprule
Method & Precision & Recall & F1 & mIoU \\
\midrule
Multi-Stage & 0.312 & 0.267 & \textbf{0.273} & 0.451 \\
DGCNN & 0.198 & 0.145 & 0.167 & 0.382 \\
...
\bottomrule
\end{tabular}
\end{table}
```

---

## Ablation Studies

### Component Analysis

| Configuration | F1 | Δ |
|--------------|-----|---|
| Full Model | 0.273 | - |
| w/o Box Budget | 0.214 | -5.9% |
| w/o Box Merging | 0.220 | -5.3% |
| w/o Boundary Filter | 0.245 | -2.8% |
| w/o Proximity Filter | 0.251 | -2.2% |

### Parameter Sensitivity

Run grid search:

```python
from src.evaluation.param_search import grid_search

param_grid = {
    'distance_threshold': [0.04, 0.05, 0.06, 0.07],
    'eps': [0.01, 0.015, 0.02],
    'box_budget_per_class': [3, 5, 7, 10]
}

best_params, results = grid_search(
    detector_class=MultiStageDetector,
    param_grid=param_grid,
    dataset=val_dataset,
    metric='macro_f1'
)
```

---

## Statistical Significance

### Paired t-test

```python
from scipy import stats

# Compare two methods across scenes
t_stat, p_value = stats.ttest_rel(method1_f1_scores, method2_f1_scores)

if p_value < 0.05:
    print("Significant difference (p < 0.05)")
```

### Confidence Intervals

```python
import numpy as np

f1_scores = [...]  # F1 for each scene
mean = np.mean(f1_scores)
std = np.std(f1_scores)
n = len(f1_scores)
ci_95 = 1.96 * std / np.sqrt(n)

print(f"F1 = {mean:.3f} ± {ci_95:.3f}")
```
