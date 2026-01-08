#!/usr/bin/env python3
"""
IndoorCD Custom Method Example
==============================
Shows how to implement and register a custom change detection method.
"""

import numpy as np
from pathlib import Path
import sys
from typing import List
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from methods.classical import BaseChangeDetector, DetectedBox, get_method
from sklearn.cluster import DBSCAN


class SimpleVoxelDetector(BaseChangeDetector):
    """
    Example custom method: Simple voxel-based change detection.
    
    This is a minimal implementation to demonstrate how to create
    and register custom detection methods.
    """
    
    name = "simple_voxel"
    
    def __init__(
        self,
        voxel_size: float = 0.05,
        occupancy_threshold: int = 3,
        cluster_eps: float = 0.1,
        min_cluster_points: int = 20
    ):
        """
        Initialize the detector.
        
        Args:
            voxel_size: Size of voxels in meters
            occupancy_threshold: Min points to consider voxel occupied
            cluster_eps: DBSCAN epsilon for clustering
            min_cluster_points: Minimum points per cluster
        """
        self.voxel_size = voxel_size
        self.occupancy_threshold = occupancy_threshold
        self.cluster_eps = cluster_eps
        self.min_cluster_points = min_cluster_points
    
    def _voxelize(self, points: np.ndarray) -> set:
        """Convert points to voxel indices."""
        voxel_indices = np.floor(points / self.voxel_size).astype(int)
        
        # Count points per voxel
        voxel_counts = {}
        for idx in map(tuple, voxel_indices):
            voxel_counts[idx] = voxel_counts.get(idx, 0) + 1
        
        # Return occupied voxels
        return {k for k, v in voxel_counts.items() 
                if v >= self.occupancy_threshold}
    
    def _voxels_to_points(self, voxels: set) -> np.ndarray:
        """Convert voxel indices back to center points."""
        if not voxels:
            return np.array([]).reshape(0, 3)
        
        centers = np.array(list(voxels)) * self.voxel_size
        centers += self.voxel_size / 2  # Center of voxel
        return centers
    
    def _cluster_and_fit(self, points: np.ndarray, label: str) -> List[DetectedBox]:
        """Cluster points and fit bounding boxes."""
        if len(points) < self.min_cluster_points:
            return []
        
        # Cluster
        clustering = DBSCAN(
            eps=self.cluster_eps,
            min_samples=self.min_cluster_points
        )
        labels = clustering.fit_predict(points)
        
        boxes = []
        for cluster_id in set(labels):
            if cluster_id == -1:  # Noise
                continue
            
            mask = labels == cluster_id
            cluster_points = points[mask]
            
            if len(cluster_points) < self.min_cluster_points:
                continue
            
            # Fit AABB
            min_bound = cluster_points.min(axis=0)
            max_bound = cluster_points.max(axis=0)
            
            # Add small padding
            padding = self.voxel_size * 0.5
            min_bound -= padding
            max_bound += padding
            
            boxes.append(DetectedBox(
                min_bound=min_bound,
                max_bound=max_bound,
                label=label,
                confidence=1.0,
                num_points=len(cluster_points)
            ))
        
        return boxes
    
    def detect(
        self, 
        reference: np.ndarray, 
        comparison: np.ndarray
    ) -> List[DetectedBox]:
        """
        Detect changes between two point clouds.
        
        Args:
            reference: (N, 3) reference point cloud
            comparison: (M, 3) comparison point cloud
        
        Returns:
            List of detected bounding boxes
        """
        # Ensure 3D points
        reference = np.asarray(reference)[:, :3]
        comparison = np.asarray(comparison)[:, :3]
        
        # Voxelize both point clouds
        ref_voxels = self._voxelize(reference)
        comp_voxels = self._voxelize(comparison)
        
        # Find differences
        added_voxels = comp_voxels - ref_voxels
        removed_voxels = ref_voxels - comp_voxels
        
        # Convert to points
        added_points = self._voxels_to_points(added_voxels)
        removed_points = self._voxels_to_points(removed_voxels)
        
        # Cluster and fit boxes
        boxes = []
        boxes.extend(self._cluster_and_fit(added_points, "Add"))
        boxes.extend(self._cluster_and_fit(removed_points, "Remove"))
        
        return boxes


def register_custom_method():
    """Register the custom method with the framework."""
    # Import the registry
    from methods.classical import METHOD_REGISTRY
    
    # Register our custom method
    METHOD_REGISTRY['simple_voxel'] = SimpleVoxelDetector
    
    print("Custom method 'simple_voxel' registered!")


def test_custom_method():
    """Test the custom method."""
    print("=" * 60)
    print("Testing Custom Method: SimpleVoxelDetector")
    print("=" * 60)
    
    # Create test data
    np.random.seed(42)
    
    # Reference: Random room
    reference = np.random.rand(1000, 3) * 3  # 3x3x3 room
    
    # Comparison: Same room + new object
    comparison = reference.copy()
    new_object = np.random.rand(100, 3) * 0.4 + np.array([2.0, 2.0, 2.0])
    comparison = np.vstack([comparison, new_object])
    
    # Test with direct instantiation
    detector = SimpleVoxelDetector(
        voxel_size=0.05,
        occupancy_threshold=3,
        cluster_eps=0.1,
        min_cluster_points=10
    )
    
    boxes = detector.detect(reference, comparison)
    
    print(f"\nDetected {len(boxes)} changes:")
    for i, box in enumerate(boxes):
        volume = np.prod(box.max_bound - box.min_bound)
        print(f"\n  Box {i+1}:")
        print(f"    Label: {box.label}")
        print(f"    Min bound: [{box.min_bound[0]:.2f}, {box.min_bound[1]:.2f}, {box.min_bound[2]:.2f}]")
        print(f"    Max bound: [{box.max_bound[0]:.2f}, {box.max_bound[1]:.2f}, {box.max_bound[2]:.2f}]")
        print(f"    Volume: {volume:.4f} m³")
        print(f"    Points: {box.num_points}")


def compare_with_baseline():
    """Compare custom method with built-in methods."""
    print("\n" + "=" * 60)
    print("Comparison with Baseline Methods")
    print("=" * 60)
    
    # Create test data
    np.random.seed(42)
    reference = np.random.rand(500, 3) * 2
    
    # Add object to comparison
    comparison = reference.copy()
    new_obj = np.random.rand(50, 3) * 0.3 + np.array([1.5, 1.5, 1.5])
    comparison = np.vstack([comparison, new_obj])
    
    # Remove some points (simulate removed object)
    mask = ~((reference[:, 0] < 0.3) & (reference[:, 1] < 0.3))
    reference_with_obj = np.vstack([
        reference[mask],
        np.random.rand(40, 3) * 0.25  # Object to be "removed"
    ])
    
    methods = {
        'simple_voxel': SimpleVoxelDetector(voxel_size=0.05),
        'distance': get_method('distance', distance_threshold=0.05),
        'multi_stage': get_method('multi_stage'),
    }
    
    print("\nResults:")
    print("-" * 50)
    print(f"{'Method':<20} {'Add':>8} {'Remove':>8} {'Total':>8}")
    print("-" * 50)
    
    for name, detector in methods.items():
        try:
            boxes = detector.detect(reference_with_obj, comparison)
            add_count = sum(1 for b in boxes if b.label == 'Add')
            rem_count = sum(1 for b in boxes if b.label == 'Remove')
            print(f"{name:<20} {add_count:>8} {rem_count:>8} {len(boxes):>8}")
        except Exception as e:
            print(f"{name:<20} Error: {e}")


if __name__ == "__main__":
    print("IndoorCD Custom Method Example")
    print("=" * 60)
    
    # Test the custom method
    test_custom_method()
    
    # Compare with baselines
    compare_with_baseline()
    
    # Show how to register
    print("\n" + "=" * 60)
    print("To register your custom method globally:")
    print("=" * 60)
    print("""
# In your code:
from methods.classical import METHOD_REGISTRY
METHOD_REGISTRY['my_method'] = MyMethodClass

# Then use it like any other method:
detector = get_method('my_method', param1=value1)
    """)
