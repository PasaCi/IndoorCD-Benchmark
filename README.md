# IndoorCD: A Benchmark for 3D Point Cloud Change Detection in Indoor Environments

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/Paper-IEEE%20Access-green.svg)](https://doi.org/10.1109/ACCESS.2025.XXXXXXX)

Official implementation of **"IndoorCD: A Benchmark Dataset and Methods for 3D Point Cloud Change Detection in Indoor Environments"** (IEEE Access 2025).

---

## 📋 Overview

IndoorCD is a comprehensive benchmark dataset for evaluating 3D point cloud change detection methods in indoor environments. The dataset consists of **1,018 scene pairs** collected from **217 rooms** using iPhone LiDAR technology, featuring four types of changes: **Add**, **Remove**, **Move**, and **Composite**.

### Key Features

- 🏠 **Real-world indoor scenes** captured with consumer-grade LiDAR
- 📦 **3D bounding box annotations** for all changed objects
- 🔄 **Multiple change types**: Add, Remove, Move, Composite
- 📊 **Comprehensive evaluation protocols**: Point-level, Box-level, Scene-level
- 🧪 **9 baseline methods** including classical and deep learning approaches

---

## 📊 Benchmark Results

### Main Results (Point-Level Evaluation)

| Method | Type | Accuracy | Add F1 | Remove F1 | Macro F1 |
|--------|------|----------|--------|-----------|----------|
| **Multi-Stage (Ours)** | Classical | **95.7%** | **20.7%** | **28.8%** | **27.3%** |
| RANSAC-Based | Classical | 78.5% | 10.4% | 20.2% | 16.7% |
| ICP-Based | Classical | 71.5% | 9.0% | 19.3% | 15.4% |
| Distance-Based | Classical | 70.2% | 8.6% | 18.3% | 14.6% |
| DGCNN | Deep Learning | 86.2% | 3.1% | 9.4% | 16.7% |
| PointNet++ | Deep Learning | 85.7% | 4.0% | 8.7% | 16.5% |

### Key Findings

1. **Classical > Deep Learning**: Our geometric Multi-Stage method outperforms all learning-based approaches
2. **Remove detection is harder**: All methods show lower performance on Remove vs Add
3. **Object size matters**: Best detection accuracy (87.7%) for objects in 10-50L volume range

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/PasaCi/IndoorCD-Benchmark.git
cd IndoorCD-Benchmark

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Download Dataset

Download the IndoorCD dataset from (link-coming-soon) (Available after paper acceptance)

```bash
# Expected directory structure
Dataset/
├── Data/
│   ├── 001/
│   │   ├── 001-1.pcd    # Reference scan
│   │   └── 001-2.pcd    # Comparison scan
│   ├── 002/
│   └── ...
└── Label/
    ├── 001-2.json       # Bounding box annotations
    ├── 002-2.json
    └── ...
```

### Run Benchmark

```bash
# Run all methods on test set
python run_benchmark.py --data_path ./Dataset --output_dir ./results

# Run specific method
python run_benchmark.py --data_path ./Dataset --method multi_stage

# Run with custom parameters
python run_benchmark.py --data_path ./Dataset --method multi_stage \
    --iou_threshold 0.25 --seed 42

# Per-category evaluation
python run_benchmark.py --data_path ./Dataset --per_category
```

---

## 📁 Repository Structure

```
IndoorCD-Benchmark/
├── README.md                 # This file
├── LICENSE                   # MIT License
├── requirements.txt          # Python dependencies
├── setup.py                  # Package installation
├── run_benchmark.py          # Main benchmark script
├── config.yaml               # Default configuration
│
├── src/
│   ├── __init__.py
│   ├── methods/
│   │   ├── __init__.py
│   │   └── classical.py      # All detection methods
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py        # Evaluation metrics
│   └── utils/
│       ├── __init__.py
│       └── data_loader.py    # Dataset loading utilities
│
├── configs/
│   ├── default.yaml          # Default parameters
│   └── optimized.yaml        # Optimized parameters
│
├── scripts/
│   ├── download_dataset.py   # Dataset download helper
│   ├── visualize_results.py  # Visualization tools
│   └── export_results.py     # Export to various formats
│
├── examples/
│   ├── quick_start.py        # Basic usage example
│   ├── custom_method.py      # Adding new methods
│   └── jupyter_demo.ipynb    # Interactive demo
│
└── docs/
    ├── DATASET.md            # Dataset documentation
    ├── METHODS.md            # Method descriptions
    └── EVALUATION.md         # Evaluation protocols
```

---

## 🔧 Methods

### 1. Multi-Stage (Proposed)

Our proposed method uses a multi-stage geometric approach:

1. **Coarse Detection**: KD-tree based point-to-point distance filtering
2. **Refinement**: Boundary analysis and proximity filtering
3. **Clustering**: DBSCAN-based spatial clustering
4. **Box Fitting**: Oriented bounding box generation with budget constraints

```python
from src.methods.classical import get_method

detector = get_method('multi_stage',
    distance_threshold=0.06,
    proximity_threshold=0.03,
    boundary_threshold=0.10,
    eps=0.015,
    min_samples=5,
    box_budget_per_class=5
)

boxes = detector.detect(reference_points, comparison_points)
```

### 2. Baseline Methods

| Method | Description |
|--------|-------------|
| `distance` | Nearest neighbor distance thresholding |
| `octree` | Voxel occupancy comparison |
| `icp` | ICP alignment + residual analysis |
| `ransac` | RANSAC plane removal + clustering |
| `region_growing` | Region-based segmentation |
| `m3c2` | Multi-scale model-to-model comparison |

---

## 📈 Evaluation Protocols

### Point-Level Evaluation

Evaluates per-point classification accuracy:
- **NoChange**: Points present in both scans
- **Add**: Points only in comparison scan
- **Remove**: Points only in reference scan

### Box-Level Evaluation (IoU = 0.25)

Matches predicted and ground truth bounding boxes:

```
IoU(pred, gt) = Volume(pred ∩ gt) / Volume(pred ∪ gt)
```

### Scene-Level Classification

Binary classification: Does the scene contain any changes?

---

## 📝 Configuration

### Default Parameters

```yaml
# config.yaml
dataset:
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  seed: 42

evaluation:
  iou_threshold: 0.25
  
multi_stage:
  distance_threshold: 0.06
  proximity_threshold: 0.03
  boundary_threshold: 0.10
  eps: 0.015
  min_samples: 5
  tolerance_factor: 1.05
  roi_scale: 0.96
  roi_coverage_thresh: 0.5
  box_budget_per_class: 5
```

---

## 📚 Citation

If you use this dataset or code in your research, please cite:

```bibtex
@article{author2025indoorcd,
  title={IndoorCD: A Benchmark Dataset and Methods for 3D Point Cloud Change Detection in Indoor Environments},
  author={Ciceklidag, Pasa and others},
  journal={IEEE Access},
  year={2025},
  volume={XX},
  pages={XXXXX-XXXXX},
  doi={10.1109/ACCESS.2025.XXXXXXX}
}
```


---

## 🙏 Acknowledgments

- Dataset collected using iPhone LiDAR (3D Scanner App)
- Point cloud processing: [Open3D](http://www.open3d.org/)
- Deep learning baselines adapted from [pytorch_geometric](https://pytorch-geometric.readthedocs.io/)

---

## 📧 Contact

For questions or issues, please open a GitHub issue or contact:
- **Email**: pasa.ciceklidag@research.uwa.edu.au pasaciceklidag@gmail.com
- **Project Page**:???

---

## 🔄 Updates

