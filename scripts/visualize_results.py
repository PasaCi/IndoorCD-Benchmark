#!/usr/bin/env python3
"""
IndoorCD Visualization Tools
============================
Visualize detection results and dataset statistics.
"""

import argparse
import json
from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Some visualizations unavailable.")

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Warning: Open3D not installed. 3D visualizations unavailable.")


def visualize_detection_3d(
    reference: np.ndarray,
    comparison: np.ndarray,
    boxes: list,
    title: str = "Change Detection Result"
):
    """
    Visualize point clouds and detection boxes in 3D.
    
    Args:
        reference: Reference point cloud (N, 3)
        comparison: Comparison point cloud (M, 3)
        boxes: List of detection dictionaries with 'min', 'max', 'label'
        title: Window title
    """
    if not HAS_OPEN3D:
        print("Open3D required for 3D visualization")
        return
    
    geometries = []
    
    # Reference points (blue)
    pcd_ref = o3d.geometry.PointCloud()
    pcd_ref.points = o3d.utility.Vector3dVector(reference[:, :3])
    pcd_ref.paint_uniform_color([0.3, 0.3, 0.8])  # Blue
    geometries.append(pcd_ref)
    
    # Comparison points (green)
    pcd_comp = o3d.geometry.PointCloud()
    pcd_comp.points = o3d.utility.Vector3dVector(comparison[:, :3])
    pcd_comp.paint_uniform_color([0.3, 0.8, 0.3])  # Green
    geometries.append(pcd_comp)
    
    # Detection boxes
    colors = {
        'Add': [1.0, 0.0, 0.0],      # Red
        'Remove': [0.0, 0.0, 1.0],   # Blue
    }
    
    for box in boxes:
        min_bound = np.array(box['min'])
        max_bound = np.array(box['max'])
        label = box.get('label', 'Unknown')
        
        # Create line set for box
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=min_bound,
            max_bound=max_bound
        )
        bbox.color = colors.get(label, [0.5, 0.5, 0.5])
        geometries.append(bbox)
    
    # Visualize
    o3d.visualization.draw_geometries(
        geometries,
        window_name=title,
        width=1280,
        height=720
    )


def plot_benchmark_results(results_path: str, output_path: str = None):
    """
    Plot benchmark comparison chart.
    
    Args:
        results_path: Path to results JSON
        output_path: Path to save figure (optional)
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib required for plotting")
        return
    
    with open(results_path) as f:
        results = json.load(f)
    
    methods = []
    f1_add = []
    f1_remove = []
    
    for method, data in results.get('methods', {}).items():
        metrics = data.get('metrics', {})
        methods.append(method)
        f1_add.append(metrics.get('f1', {}).get('Add', 0))
        f1_remove.append(metrics.get('f1', {}).get('Remove', 0))
    
    # Sort by Add F1
    sorted_idx = np.argsort(f1_add)[::-1]
    methods = [methods[i] for i in sorted_idx]
    f1_add = [f1_add[i] for i in sorted_idx]
    f1_remove = [f1_remove[i] for i in sorted_idx]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, f1_add, width, label='Add F1', color='#2ecc71')
    bars2 = ax.bar(x + width/2, f1_remove, width, label='Remove F1', color='#e74c3c')
    
    ax.set_ylabel('F1 Score')
    ax.set_title('IndoorCD Benchmark Results')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, max(max(f1_add), max(f1_remove)) * 1.2)
    
    # Add value labels
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
    else:
        plt.show()


def plot_category_breakdown(results_path: str, output_path: str = None):
    """Plot per-category performance breakdown."""
    if not HAS_MATPLOTLIB:
        print("matplotlib required for plotting")
        return
    
    with open(results_path) as f:
        results = json.load(f)
    
    # This assumes per-category results format
    # Adjust based on actual result structure
    categories = ['Add', 'Remove', 'Move', 'Composite']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, cat in enumerate(categories):
        ax = axes[i]
        ax.set_title(f'{cat} Category')
        ax.set_xlabel('Method')
        ax.set_ylabel('F1 Score')
        # Add data plotting here
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
    else:
        plt.show()


def plot_volume_analysis(output_path: str = None):
    """Plot detection accuracy by object volume."""
    if not HAS_MATPLOTLIB:
        print("matplotlib required for plotting")
        return
    
    # Sample data - replace with actual results
    volumes = ['<1L', '1-5L', '5-10L', '10-50L', '50-100L', '100-500L', '>500L']
    accuracy = [65.5, 72.3, 80.1, 87.7, 82.4, 75.8, 68.2]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(volumes)))
    bars = ax.bar(volumes, accuracy, color=colors)
    
    ax.set_xlabel('Object Volume')
    ax.set_ylabel('Detection Accuracy (%)')
    ax.set_title('Detection Accuracy by Object Volume')
    ax.set_ylim(0, 100)
    
    # Add value labels
    for bar, acc in zip(bars, accuracy):
        ax.annotate(f'{acc:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, acc),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='IndoorCD Visualization Tools')
    
    subparsers = parser.add_subparsers(dest='command', help='Visualization command')
    
    # Benchmark results plot
    bench_parser = subparsers.add_parser('benchmark', help='Plot benchmark results')
    bench_parser.add_argument('results', type=str, help='Path to results JSON')
    bench_parser.add_argument('--output', '-o', type=str, help='Output path')
    
    # Volume analysis
    vol_parser = subparsers.add_parser('volume', help='Plot volume analysis')
    vol_parser.add_argument('--output', '-o', type=str, help='Output path')
    
    args = parser.parse_args()
    
    if args.command == 'benchmark':
        plot_benchmark_results(args.results, args.output)
    elif args.command == 'volume':
        plot_volume_analysis(args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
