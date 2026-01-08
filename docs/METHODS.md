# Change Detection Methods

This document describes all methods implemented in the IndoorCD benchmark.

## Overview

| Method | Type | Key Idea |
|--------|------|----------|
| Multi-Stage | Classical | Multi-stage geometric refinement |
| Distance-Based | Classical | Nearest neighbor distance |
| Octree-Based | Classical | Voxel occupancy comparison |
| ICP-Based | Classical | ICP alignment + residuals |
| RANSAC-Based | Classical | Plane removal + clustering |
| Region Growing | Classical | Region-based segmentation |
| M3C2 | Classical | Multi-scale comparison |
| PointNet++ | Deep Learning | Hierarchical feature learning |
| DGCNN | Deep Learning | Dynamic graph convolution |

---

## 1. Multi-Stage (Proposed)

Our proposed method uses a four-stage pipeline:

### Stage 1: Coarse Detection
```python
# KD-tree based distance computation
tree_ref = KDTree(reference_points)
distances, _ = tree_ref.query(comparison_points)
add_mask = distances > distance_threshold  # Default: 0.06m
```

### Stage 2: Refinement
- **Boundary analysis**: Remove edge artifacts
- **Proximity filtering**: Merge nearby detections
- **Noise removal**: Filter isolated points

### Stage 3: Clustering (DBSCAN)
```python
clustering = DBSCAN(eps=0.015, min_samples=5)
labels = clustering.fit_predict(candidate_points)
```

### Stage 4: Box Fitting
- Compute oriented bounding boxes
- Apply box budget constraint (default: 5 per class)
- Merge overlapping boxes (IoU > 0.3)

### Optimized Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `distance_threshold` | 0.06 | Coarse detection threshold (m) |
| `proximity_threshold` | 0.03 | Proximity merge threshold (m) |
| `boundary_threshold` | 0.10 | Boundary filter threshold (m) |
| `eps` | 0.015 | DBSCAN epsilon |
| `min_samples` | 5 | DBSCAN min points |
| `box_budget_per_class` | 5 | Max boxes per class |

### Usage

```python
from src.methods.classical import get_method

detector = get_method('multi_stage',
    distance_threshold=0.06,
    proximity_threshold=0.03,
    eps=0.015,
    box_budget_per_class=5
)

boxes = detector.detect(ref_points, comp_points)
for box in boxes:
    print(f"{box.label}: {box.min_bound} -> {box.max_bound}")
```

---

## 2. Distance-Based

Simple nearest neighbor distance thresholding.

### Algorithm
1. Build KD-tree on reference points
2. Query distances for all comparison points
3. Threshold to find added points
4. Repeat inversely for removed points
5. Cluster and fit boxes

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `distance_threshold` | 0.05 | Detection threshold (m) |
| `cluster_eps` | 0.05 | Clustering epsilon |
| `min_cluster_points` | 50 | Minimum cluster size |

---

## 3. Octree-Based

Voxel-based occupancy comparison.

### Algorithm
1. Voxelize both point clouds
2. Compare voxel occupancy
3. Identify added/removed voxels
4. Cluster connected voxels
5. Fit bounding boxes

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `voxel_size` | 0.05 | Voxel size (m) |
| `min_points_per_voxel` | 3 | Occupancy threshold |
| `cluster_eps` | 0.08 | Clustering epsilon |

---

## 4. ICP-Based

ICP alignment followed by residual analysis.

### Algorithm
1. Run ICP to align point clouds
2. Compute point-to-point residuals
3. Threshold high-residual points
4. Cluster and classify changes

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `icp_threshold` | 0.1 | ICP convergence threshold |
| `max_iterations` | 50 | Max ICP iterations |
| `distance_threshold` | 0.05 | Residual threshold |

---

## 5. Deep Learning Methods

### PointNet++

Hierarchical point set learning with set abstraction layers.

```python
# Requires torch and torch_geometric
from src.methods.deep_learning import PointNetPlusPlusDetector

model = PointNetPlusPlusDetector(
    num_classes=3,
    use_normal=False
)
model.load_state_dict(torch.load('checkpoints/pointnetpp.pth'))
```

### DGCNN

Dynamic graph CNN with EdgeConv operations.

```python
from src.methods.deep_learning import DGCNNDetector

model = DGCNNDetector(
    num_classes=3,
    k=20  # k-nearest neighbors
)
```

---

## Adding Custom Methods

Implement the `BaseChangeDetector` interface:

```python
from src.methods.classical import BaseChangeDetector, DetectedBox

class MyDetector(BaseChangeDetector):
    name = "my_method"
    
    def __init__(self, my_param=0.1):
        self.my_param = my_param
    
    def detect(self, reference, comparison):
        """
        Args:
            reference: np.ndarray (N, 3) - Reference point cloud
            comparison: np.ndarray (M, 3) - Comparison point cloud
        
        Returns:
            List[DetectedBox] - Detected change boxes
        """
        boxes = []
        
        # Your detection logic here
        # ...
        
        boxes.append(DetectedBox(
            min_bound=np.array([0, 0, 0]),
            max_bound=np.array([1, 1, 1]),
            label="Add",
            confidence=0.95,
            num_points=100
        ))
        
        return boxes
```

Register your method:

```python
# In src/methods/classical.py
METHOD_REGISTRY['my_method'] = MyDetector
```

---

## Performance Comparison

| Method | Accuracy | Add F1 | Remove F1 | Time/pair |
|--------|----------|--------|-----------|-----------|
| Multi-Stage | **95.7%** | **20.7%** | **28.8%** | 0.8s |
| DGCNN | 86.2% | 3.1% | 9.4% | 1.2s |
| PointNet++ | 85.7% | 4.0% | 8.7% | 1.0s |
| Distance-Based | 70.2% | 8.6% | 18.3% | 0.3s |
| ICP-Based | 71.5% | 9.0% | 19.3% | 2.1s |

*Evaluated on 123 test pairs with IoU threshold = 0.25*
