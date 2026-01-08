# IndoorCD Dataset Documentation

## Overview

IndoorCD is a benchmark dataset for 3D point cloud change detection in indoor environments. The dataset was collected using iPhone LiDAR technology (3D Scanner App) across various indoor spaces.

## Dataset Statistics

| Statistic | Value |
|-----------|-------|
| Total scene pairs | 1,018 |
| Total rooms | 217 |
| Total changes | 1,695 |
| Added objects | 856 |
| Removed objects | 839 |
| Mean points/scan | ~112,000 |
| Mean boxes/room | 7.8 |

## Change Types

### 1. Add (A)
Objects present only in the comparison scan (new objects added to the scene).

**Examples**: Books placed on shelf, chair moved into room, laptop on desk

### 2. Remove (R)  
Objects present only in the reference scan (objects removed from scene).

**Examples**: Box removed from floor, plant taken away, monitor removed

### 3. Move (M)
Objects that changed position between scans. Annotated as both Add and Remove.

**Examples**: Chair repositioned, table moved to corner

### 4. Composite (C)
Scenes with multiple types of changes (combinations of A, R, M).

## Data Format

### Point Cloud Files (.pcd)

Standard PCD format with XYZ coordinates:

```
# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH 50000
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS 50000
DATA ascii
-1.234 0.567 0.890
...
```

### Label Files (.json)

Ground truth bounding box annotations:

```json
{
  "filename": "001-2",
  "objects": [
    {
      "label": "Add",
      "min": [-0.5, 0.0, 0.2],
      "max": [0.3, 0.8, 0.9],
      "object_type": "book",
      "notes": "optional description"
    },
    {
      "label": "Remove",
      "min": [1.0, 0.5, 0.0],
      "max": [1.5, 1.2, 0.6]
    }
  ]
}
```

## Directory Structure

```
Dataset/
├── Data/
│   ├── 001/                    # Room folder
│   │   ├── 001-1.pcd          # Reference scan
│   │   └── 001-2.pcd          # Comparison scan
│   ├── 002/
│   │   ├── 002-1.pcd
│   │   ├── 002-2.pcd
│   │   └── 002-3.pcd          # Some rooms have multiple pairs
│   └── ...
└── Label/
    ├── 001-2.json             # Labels for pair (001-1, 001-2)
    ├── 002-2.json
    └── ...
```

## Train/Val/Test Split

We provide a fixed split for reproducibility:

| Split | Rooms | Pairs | Ratio |
|-------|-------|-------|-------|
| Train | 152 | 712 | 70% |
| Val | 32 | 153 | 15% |
| Test | 33 | 153 | 15% |

**Important**: Split is done by room to prevent data leakage between sets.

## Object Volume Distribution

| Volume Range | Count | % |
|--------------|-------|---|
| < 1L | 245 | 14.5% |
| 1-5L | 389 | 22.9% |
| 5-10L | 312 | 18.4% |
| 10-50L | 421 | 24.8% |
| 50-100L | 187 | 11.0% |
| 100-500L | 108 | 6.4% |
| > 500L | 33 | 1.9% |

## Collection Protocol

1. **Reference scan**: Capture initial state of room
2. **Modification**: Add, remove, or move objects
3. **Comparison scan**: Capture modified state
4. **Annotation**: Mark bounding boxes for all changes

### Quality Control
- Minimum 50cm distance from walls
- Consistent lighting conditions
- < 30 second scan duration
- Registration check: < 5cm alignment error

## Download

- **Google Drive**: [Link](https://drive.google.com/xxx)
- **Zenodo**: [DOI](https://zenodo.org/xxx)
- **Size**: ~15 GB (compressed)

## License

The dataset is released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Citation

```bibtex
@article{author2025indoorcd,
  title={IndoorCD: A Benchmark Dataset and Methods for 3D Point Cloud Change Detection},
  author={Author, First and Author, Second},
  journal={IEEE Access},
  year={2025}
}
```
