#!/usr/bin/env python3
"""
IndoorCD Benchmark Runner
=========================
Main script for running change detection experiments on IndoorCD dataset.

Usage:
    # Run all methods on test set
    python run_benchmark.py --data_path ./Dataset --output_dir ./results
    
    # Run specific method
    python run_benchmark.py --data_path ./Dataset --method multi_stage
    
    # Create and save dataset split
    python run_benchmark.py --data_path ./Dataset --create_split --split_path ./split.json
    
    # Run with existing split
    python run_benchmark.py --data_path ./Dataset --split_path ./split.json
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils.data_loader import (
    IndoorCDDataset, 
    create_fixed_split, 
    load_split,
    get_split_by_change_type
)
from evaluation.metrics import (
    BenchmarkEvaluator, 
    DetectionMetrics,
    parse_ground_truth
)
from methods.classical import get_method, BaseChangeDetector, DetectedBox


class IndoorCDBenchmark:
    """
    Main benchmark class for IndoorCD dataset evaluation.
    """
    
    # Default method configurations
    DEFAULT_METHODS = {
        'distance': {
            'distance_threshold': 0.05,
            'cluster_eps': 0.05,
            'min_cluster_points': 50,
        },
        'octree': {
            'voxel_size': 0.05,
            'min_points_per_voxel': 3,
            'cluster_eps': 0.08,
            'min_cluster_points': 30,
        },
        'icp': {
            'distance_threshold': 0.05,
            'icp_threshold': 0.1,
            'max_iterations': 50,
            'cluster_eps': 0.05,
            'min_cluster_points': 50,
        },
        'multi_stage': {
            'distance_threshold': 0.06,
            'proximity_threshold': 0.03,
            'boundary_threshold': 0.10,
            'eps': 0.015,
            'min_samples': 5,
            'tolerance_factor': 1.05,
            'roi_scale': 0.96,
            'roi_coverage_thresh': 0.5,
            'box_budget_per_class': 5,
        },
    }
    
    def __init__(
        self,
        data_path: str,
        output_dir: str = "./results",
        iou_threshold: float = 0.25,
        seed: int = 42
    ):
        """
        Initialize benchmark.
        
        Args:
            data_path: Path to dataset root
            output_dir: Directory for results
            iou_threshold: IoU threshold for evaluation
            seed: Random seed for reproducibility
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.iou_threshold = iou_threshold
        self.seed = seed
        
        self.dataset = None
        self.train_idx = None
        self.val_idx = None
        self.test_idx = None
        
        self.evaluator = BenchmarkEvaluator(iou_threshold=iou_threshold)
        self.results = {}
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def load_dataset(self) -> IndoorCDDataset:
        """Load dataset."""
        print(f"Loading dataset from {self.data_path}...")
        self.dataset = IndoorCDDataset(str(self.data_path))
        
        stats = self.dataset.get_statistics()
        print(f"Dataset loaded:")
        print(f"  Total pairs: {stats['total_pairs']}")
        print(f"  Total rooms: {stats['total_rooms']}")
        print(f"  With labels: {stats['pairs_with_labels']}")
        print(f"  By type: {stats['pairs_by_change_type']}")
        
        return self.dataset
    
    def create_split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        save_path: Optional[str] = None
    ) -> None:
        """Create train/val/test split."""
        if self.dataset is None:
            self.load_dataset()
        
        if save_path is None:
            save_path = str(self.output_dir / "dataset_split.json")
        
        self.train_idx, self.val_idx, self.test_idx = create_fixed_split(
            self.dataset,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=self.seed,
            by_room=True,
            save_path=save_path
        )
        
        print(f"Split created (seed={self.seed}):")
        print(f"  Train: {len(self.train_idx)} pairs")
        print(f"  Val: {len(self.val_idx)} pairs")
        print(f"  Test: {len(self.test_idx)} pairs")
    
    def load_split(self, split_path: str) -> None:
        """Load existing split."""
        if self.dataset is None:
            self.load_dataset()
        
        self.train_idx, self.val_idx, self.test_idx = load_split(
            split_path, self.dataset
        )
        
        print(f"Split loaded from {split_path}:")
        print(f"  Train: {len(self.train_idx)} pairs")
        print(f"  Val: {len(self.val_idx)} pairs")
        print(f"  Test: {len(self.test_idx)} pairs")
    
    def run_method(
        self,
        method_name: str,
        indices: List[int],
        params: Optional[Dict] = None,
        save_predictions: bool = True,
        verbose: bool = True
    ) -> Dict:
        """
        Run a single method on specified indices.
        
        Args:
            method_name: Name of the method
            indices: Dataset indices to process
            params: Method parameters (uses defaults if None)
            save_predictions: Whether to save prediction JSONs
            verbose: Print progress
        
        Returns:
            Results dictionary
        """
        if self.dataset is None:
            self.load_dataset()
        
        # Get method parameters
        if params is None:
            params = self.DEFAULT_METHODS.get(method_name, {})
        
        # Initialize method
        detector = get_method(method_name, **params)
        
        if verbose:
            print(f"\nRunning {detector.name}...")
            print(f"  Params: {params}")
            print(f"  Processing {len(indices)} scene pairs...")
        
        # Prepare output directory for predictions
        pred_dir = None
        if save_predictions:
            pred_dir = self.output_dir / f"predictions_{method_name}_{self.timestamp}"
            pred_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each scene pair
        all_predictions = []
        all_ground_truths = []
        times = []
        errors = []
        
        for i, idx in enumerate(indices):
            if verbose and (i + 1) % 20 == 0:
                print(f"    {i + 1}/{len(indices)}...")
            
            try:
                # Load data
                data = self.dataset[idx]
                
                if data['reference'] is None or data['comparison'] is None:
                    continue
                
                # Run detection
                start_time = time.time()
                boxes = detector.detect(data['reference'], data['comparison'])
                elapsed = time.time() - start_time
                times.append(elapsed)
                
                # Format predictions
                pred_dict = {
                    'filename': data['pair_id'],
                    'objects': [box.to_dict() for box in boxes]
                }
                all_predictions.append(pred_dict)
                
                # Get ground truth
                all_ground_truths.append(data['labels'])
                
                # Save prediction
                if pred_dir:
                    pred_file = pred_dir / f"{data['pair_id']}.json"
                    with open(pred_file, 'w') as f:
                        json.dump(pred_dict, f, indent=2)
                
            except Exception as e:
                errors.append((idx, str(e)))
                if verbose:
                    print(f"    Error at {idx}: {e}")
        
        # Evaluate
        metrics = self.evaluator.evaluate_method(
            method_name,
            all_predictions,
            all_ground_truths
        )
        
        # Compile results
        results = {
            'method': detector.name,
            'params': params,
            'num_scenes': len(all_predictions),
            'num_errors': len(errors),
            'avg_time_per_scene': np.mean(times) if times else 0,
            'total_time': sum(times),
            'metrics': metrics.to_dict(),
            'predictions_dir': str(pred_dir) if pred_dir else None,
        }
        
        if verbose:
            print(f"  Completed in {results['total_time']:.1f}s")
            print(f"  {metrics}")
        
        self.results[method_name] = results
        return results
    
    def run_all_methods(
        self,
        indices: Optional[List[int]] = None,
        methods: Optional[List[str]] = None,
        save_predictions: bool = True
    ) -> Dict:
        """
        Run all methods on specified indices.
        
        Args:
            indices: Dataset indices (uses test set if None)
            methods: List of method names (uses all if None)
            save_predictions: Whether to save predictions
        
        Returns:
            All results
        """
        if indices is None:
            if self.test_idx is None:
                self.create_split()
            indices = self.test_idx
        
        if methods is None:
            methods = list(self.DEFAULT_METHODS.keys())
        
        print(f"\n{'='*60}")
        print(f"Running benchmark on {len(indices)} scene pairs")
        print(f"Methods: {methods}")
        print(f"{'='*60}")
        
        for method_name in methods:
            try:
                self.run_method(
                    method_name,
                    indices,
                    save_predictions=save_predictions
                )
            except Exception as e:
                print(f"Error running {method_name}: {e}")
                import traceback
                traceback.print_exc()
        
        return self.results
    
    def run_per_category_evaluation(
        self,
        indices: Optional[List[int]] = None,
        methods: Optional[List[str]] = None
    ) -> Dict:
        """
        Run evaluation broken down by change type category.
        
        Args:
            indices: Dataset indices
            methods: Method names
        
        Returns:
            Per-category results
        """
        if indices is None:
            if self.test_idx is None:
                self.create_split()
            indices = self.test_idx
        
        if methods is None:
            methods = list(self.DEFAULT_METHODS.keys())
        
        # Group indices by change type
        by_type = get_split_by_change_type(self.dataset, indices)
        
        print(f"\n{'='*60}")
        print("Per-Category Evaluation")
        print(f"{'='*60}")
        
        category_results = {}
        
        for change_type, type_indices in by_type.items():
            print(f"\n--- {change_type} ({len(type_indices)} pairs) ---")
            
            category_results[change_type] = {}
            
            for method_name in methods:
                params = self.DEFAULT_METHODS.get(method_name, {})
                detector = get_method(method_name, **params)
                
                all_preds = []
                all_gts = []
                
                for idx in type_indices:
                    try:
                        data = self.dataset[idx]
                        if data['reference'] is None or data['comparison'] is None:
                            continue
                        
                        boxes = detector.detect(data['reference'], data['comparison'])
                        pred_dict = {'objects': [b.to_dict() for b in boxes]}
                        
                        all_preds.append(pred_dict)
                        all_gts.append(data['labels'])
                    except:
                        continue
                
                # Evaluate for this category
                cat_evaluator = BenchmarkEvaluator(iou_threshold=self.iou_threshold)
                metrics = cat_evaluator.evaluate_method(method_name, all_preds, all_gts)
                
                category_results[change_type][method_name] = metrics.to_dict()
                
                print(f"  {method_name}: F1={metrics.macro_f1:.3f}")
        
        return category_results
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """Save all results to JSON."""
        if filename is None:
            filename = f"results_{self.timestamp}.json"
        
        output_path = self.output_dir / filename
        
        # Prepare results for JSON
        results_json = {
            'timestamp': self.timestamp,
            'dataset_path': str(self.data_path),
            'iou_threshold': self.iou_threshold,
            'seed': self.seed,
            'methods': self.results
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_json, f, indent=2, default=str)
        
        print(f"Results saved to {output_path}")
        return str(output_path)
    
    def generate_report(self) -> str:
        """Generate text report."""
        lines = [
            "=" * 70,
            "IndoorCD Benchmark Results",
            "=" * 70,
            f"Timestamp: {self.timestamp}",
            f"Dataset: {self.data_path}",
            f"IoU Threshold: {self.iou_threshold}",
            "",
        ]
        
        if not self.results:
            lines.append("No results available.")
            return "\n".join(lines)
        
        # Overall comparison table
        lines.append("Overall Results:")
        lines.append("-" * 70)
        lines.append(f"{'Method':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'mIoU':>10}")
        lines.append("-" * 70)
        
        # Sort by F1
        sorted_methods = sorted(
            self.results.keys(),
            key=lambda m: self.results[m]['metrics']['macro_f1'],
            reverse=True
        )
        
        for method in sorted_methods:
            m = self.results[method]['metrics']
            lines.append(
                f"{method:<20} "
                f"{m['macro_precision']:>10.4f} "
                f"{m['macro_recall']:>10.4f} "
                f"{m['macro_f1']:>10.4f} "
                f"{m['mean_iou_matched']:>10.4f}"
            )
        
        lines.append("-" * 70)
        
        # Per-class breakdown
        lines.append("\nPer-Class Performance:")
        lines.append("-" * 70)
        
        for method in sorted_methods:
            m = self.results[method]['metrics']
            lines.append(f"\n{method}:")
            
            for cls in ['Add', 'Remove']:
                if cls in m['precision']:
                    lines.append(
                        f"  {cls}: P={m['precision'][cls]:.3f}, "
                        f"R={m['recall'][cls]:.3f}, F1={m['f1'][cls]:.3f}"
                    )
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)
    
    def generate_latex_tables(self) -> str:
        """Generate LaTeX tables for paper."""
        return self.evaluator.generate_latex_table()


def main():
    parser = argparse.ArgumentParser(
        description='IndoorCD Benchmark Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full benchmark
  python run_benchmark.py --data_path ./Dataset

  # Run specific method
  python run_benchmark.py --data_path ./Dataset --method multi_stage

  # Create and save split first
  python run_benchmark.py --data_path ./Dataset --create_split
  
  # Run with saved split
  python run_benchmark.py --data_path ./Dataset --split_path ./split.json
        """
    )
    
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to dataset root directory')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for results')
    parser.add_argument('--method', type=str, default=None,
                        help='Run specific method only')
    parser.add_argument('--iou_threshold', type=float, default=0.25,
                        help='IoU threshold for evaluation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--create_split', action='store_true',
                        help='Create and save dataset split')
    parser.add_argument('--split_path', type=str, default=None,
                        help='Path to existing split JSON')
    parser.add_argument('--per_category', action='store_true',
                        help='Run per-category evaluation')
    parser.add_argument('--no_save_predictions', action='store_true',
                        help='Do not save prediction JSONs')
    
    args = parser.parse_args()
    
    # Initialize benchmark
    benchmark = IndoorCDBenchmark(
        data_path=args.data_path,
        output_dir=args.output_dir,
        iou_threshold=args.iou_threshold,
        seed=args.seed
    )
    
    # Load dataset
    benchmark.load_dataset()
    
    if len(benchmark.dataset) == 0:
        print("\nNo data found. Please check your data path.")
        print(f"Expected structure:")
        print(f"  {args.data_path}/")
        print(f"  ├── Data/")
        print(f"  │   ├── 001/")
        print(f"  │   │   ├── 001-1.pcd")
        print(f"  │   │   └── 001-2.pcd")
        print(f"  └── Label/")
        print(f"      └── 001-2.json")
        return
    
    # Handle split
    if args.split_path:
        benchmark.load_split(args.split_path)
    else:
        split_path = str(Path(args.output_dir) / "dataset_split.json")
        benchmark.create_split(save_path=split_path)
    
    # Run benchmark
    if args.method:
        benchmark.run_method(
            args.method,
            benchmark.test_idx,
            save_predictions=not args.no_save_predictions
        )
    else:
        benchmark.run_all_methods(
            benchmark.test_idx,
            save_predictions=not args.no_save_predictions
        )
    
    # Per-category evaluation if requested
    if args.per_category:
        cat_results = benchmark.run_per_category_evaluation(benchmark.test_idx)
        
        # Save category results
        cat_path = Path(args.output_dir) / f"per_category_{benchmark.timestamp}.json"
        with open(cat_path, 'w') as f:
            json.dump(cat_results, f, indent=2, default=str)
    
    # Generate and print report
    report = benchmark.generate_report()
    print("\n" + report)
    
    # Save results
    benchmark.save_results()
    
    # Save report
    report_path = Path(args.output_dir) / f"report_{benchmark.timestamp}.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    # Generate LaTeX
    latex = benchmark.generate_latex_tables()
    if latex:
        latex_path = Path(args.output_dir) / f"tables_{benchmark.timestamp}.tex"
        with open(latex_path, 'w') as f:
            f.write(latex)
        print(f"LaTeX tables saved to {latex_path}")


if __name__ == "__main__":
    main()
