#!/usr/bin/env python3
"""
IndoorCD Quick Start Example
============================
Basic usage of the IndoorCD benchmark.
"""

import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from methods.classical import get_method, DetectedBox
from utils.data_loader import IndoorCDDataset
from evaluation.metrics import BenchmarkEvaluator


def example_basic_detection():
    """Example 1: Basic change detection on synthetic data."""
    print("=" * 60)
    print("Example 1: Basic Change Detection")
    print("=" * 60)
    
    # Create synthetic point clouds
    np.random.seed(42)
    
    # Reference: Random points in a cube
    reference = np.random.rand(1000, 3) * 2 - 1  # [-1, 1]^3
    
    # Comparison: Same + added object
    comparison = reference.copy()
    # Add a new object (small cluster)
    added_object = np.random.rand(50, 3) * 0.3 + np.array([0.5, 0.5, 0.5])
    comparison = np.vstack([comparison, added_object])
    
    # Initialize detector
    detector = get_method('multi_stage',
        distance_threshold=0.1,
        eps=0.05,
        min_samples=10,
        box_budget_per_class=3
    )
    
    # Run detection
    boxes = detector.detect(reference, comparison)
    
    print(f"\nDetected {len(boxes)} change(s):")
    for i, box in enumerate(boxes):
        volume = np.prod(box.max_bound - box.min_bound)
        print(f"  Box {i+1}: {box.label}")
        print(f"    Min: {box.min_bound}")
        print(f"    Max: {box.max_bound}")
        print(f"    Volume: {volume:.4f} m³")
        print(f"    Points: {box.num_points}")


def example_dataset_loading():
    """Example 2: Load and process IndoorCD dataset."""
    print("\n" + "=" * 60)
    print("Example 2: Dataset Loading")
    print("=" * 60)
    
    # Check if dataset exists
    data_path = Path("./Dataset")
    if not data_path.exists():
        print(f"\nDataset not found at {data_path}")
        print("Please download the dataset first.")
        print("See README.md for download links.")
        return
    
    # Load dataset
    dataset = IndoorCDDataset(str(data_path))
    stats = dataset.get_statistics()
    
    print(f"\nDataset Statistics:")
    print(f"  Total pairs: {stats['total_pairs']}")
    print(f"  Total rooms: {stats['total_rooms']}")
    print(f"  Pairs with labels: {stats['pairs_with_labels']}")
    
    # Load a sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample pair: {sample['pair_id']}")
        print(f"  Reference points: {len(sample['reference'])}")
        print(f"  Comparison points: {len(sample['comparison'])}")
        if sample['labels']:
            print(f"  Ground truth boxes: {len(sample['labels'].get('objects', []))}")


def example_evaluation():
    """Example 3: Evaluate detection results."""
    print("\n" + "=" * 60)
    print("Example 3: Evaluation")
    print("=" * 60)
    
    # Create sample predictions and ground truth
    predictions = {
        'objects': [
            {'label': 'Add', 'min': [0.0, 0.0, 0.0], 'max': [0.5, 0.5, 0.5]},
            {'label': 'Remove', 'min': [1.0, 1.0, 1.0], 'max': [1.5, 1.5, 1.5]},
        ]
    }
    
    ground_truth = {
        'objects': [
            {'label': 'Add', 'min': [0.1, 0.1, 0.1], 'max': [0.6, 0.6, 0.6]},  # Overlaps with pred 1
            {'label': 'Add', 'min': [2.0, 2.0, 2.0], 'max': [2.5, 2.5, 2.5]},  # Missed
        ]
    }
    
    # Evaluate
    evaluator = BenchmarkEvaluator(iou_threshold=0.25)
    metrics = evaluator.evaluate_method('test', [predictions], [ground_truth])
    
    print(f"\nEvaluation Results:")
    print(f"  Macro Precision: {metrics.macro_precision:.3f}")
    print(f"  Macro Recall: {metrics.macro_recall:.3f}")
    print(f"  Macro F1: {metrics.macro_f1:.3f}")
    print(f"\n  Per-class:")
    for cls in ['Add', 'Remove']:
        if cls in metrics.f1:
            print(f"    {cls}: P={metrics.precision[cls]:.3f}, "
                  f"R={metrics.recall[cls]:.3f}, F1={metrics.f1[cls]:.3f}")


def example_all_methods():
    """Example 4: Compare all methods."""
    print("\n" + "=" * 60)
    print("Example 4: Compare Methods")
    print("=" * 60)
    
    # Synthetic data
    np.random.seed(42)
    reference = np.random.rand(500, 3)
    added = np.random.rand(30, 3) * 0.2 + 0.7
    comparison = np.vstack([reference, added])
    
    methods = ['distance', 'octree', 'icp', 'multi_stage']
    
    print("\nRunning all methods...")
    for method_name in methods:
        try:
            detector = get_method(method_name)
            boxes = detector.detect(reference, comparison)
            add_count = sum(1 for b in boxes if b.label == 'Add')
            rem_count = sum(1 for b in boxes if b.label == 'Remove')
            print(f"  {method_name}: {len(boxes)} boxes (Add: {add_count}, Remove: {rem_count})")
        except Exception as e:
            print(f"  {method_name}: Error - {e}")


if __name__ == "__main__":
    print("IndoorCD Quick Start Examples")
    print("=" * 60)
    
    example_basic_detection()
    example_dataset_loading()
    example_evaluation()
    example_all_methods()
    
    print("\n" + "=" * 60)
    print("Done! See docs/ for more detailed documentation.")
    print("=" * 60)
