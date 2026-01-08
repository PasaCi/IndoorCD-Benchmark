"""
IndoorCD Change Detection Methods
=================================
Classical and geometric methods for 3D point cloud change detection.
Outputs bounding boxes for detected changes.

Methods:
1. Distance-Based: Simple nearest neighbor distance thresholding
2. Octree-Based: Voxel occupancy comparison  
3. ICP-Based: ICP alignment + residual analysis
4. Multi-Stage (Ours): Proposed multi-stage geometric approach
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from dataclasses import dataclass

try:
    import open3d as o3d
except ImportError:
    o3d = None
    print("Warning: Open3D not installed.")


# ==================== Constants ====================
DELTA_MIN = 0.02  # Minimum padding (meters)
PADDING_SCALE = 0.75


# ==================== Helper Functions ====================

def median_nn_spacing(points: np.ndarray) -> float:
    """Compute median nearest neighbor spacing for adaptive parameters."""
    if len(points) < 2:
        return DELTA_MIN
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    kdt = o3d.geometry.KDTreeFlann(pcd)
    
    dists = []
    for i in range(len(points)):
        k, idx, sqr = kdt.search_knn_vector_3d(pcd.points[i], 2)
        if k == 2:
            dists.append(float(np.sqrt(sqr[1])))
    
    return float(np.median(dists)) if dists else DELTA_MIN


def points_in_aabb_mask(points: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    """Check which points are inside an AABB."""
    return np.all((points >= mins) & (points <= maxs), axis=1)


def aabb_iou(mn1, mx1, mn2, mx2) -> float:
    """Compute IoU between two AABBs."""
    inter_min = np.maximum(mn1, mn2)
    inter_max = np.minimum(mx1, mx2)
    inter = np.maximum(inter_max - inter_min, 0.0)
    inter_v = float(inter[0] * inter[1] * inter[2])
    v1 = float(np.prod(np.maximum(mx1 - mn1, 0.0)))
    v2 = float(np.prod(np.maximum(mx2 - mn2, 0.0)))
    u = v1 + v2 - inter_v
    return (inter_v / u) if u > 0 else 0.0


def dilate_aabb(mins, maxs, scale):
    """Dilate/shrink an AABB by scale factor."""
    mins = np.asarray(mins, dtype=float)
    maxs = np.asarray(maxs, dtype=float)
    center = (mins + maxs) / 2.0
    half = (maxs - mins) / 2.0
    new_half = half * float(scale)
    return center - new_half, center + new_half


@dataclass
class DetectedBox:
    """Represents a detected bounding box."""
    min_bound: np.ndarray
    max_bound: np.ndarray
    label: str  # "Add" or "Remove"
    confidence: float = 1.0
    num_points: int = 0
    
    @property
    def vertices(self) -> np.ndarray:
        """Get 8 corner vertices."""
        mn, mx = self.min_bound, self.max_bound
        return np.array([
            [mn[0], mn[1], mn[2]],
            [mx[0], mn[1], mn[2]],
            [mx[0], mx[1], mn[2]],
            [mn[0], mx[1], mn[2]],
            [mn[0], mn[1], mx[2]],
            [mx[0], mn[1], mx[2]],
            [mx[0], mx[1], mx[2]],
            [mn[0], mx[1], mx[2]],
        ])
    
    def to_dict(self) -> Dict:
        """Convert to output dictionary format."""
        return {
            'name': self.label,
            'vertices': self.vertices.tolist(),
            'confidence': self.confidence,
        }
    
    @property
    def volume(self) -> float:
        dims = np.maximum(self.max_bound - self.min_bound, 0)
        return float(dims[0] * dims[1] * dims[2])
    
    @property
    def diagonal(self) -> float:
        return float(np.linalg.norm(self.max_bound - self.min_bound))


# ==================== Base Class ====================

class BaseChangeDetector(ABC):
    """Abstract base class for change detection methods."""
    
    def __init__(self, **kwargs):
        self.params = kwargs
        self.name = "BaseDetector"
    
    @abstractmethod
    def detect(
        self,
        reference: np.ndarray,
        comparison: np.ndarray
    ) -> List[DetectedBox]:
        """
        Detect changes between reference and comparison point clouds.
        
        Args:
            reference: Reference point cloud (N, 3+) - before state
            comparison: Comparison point cloud (M, 3+) - after state
        
        Returns:
            List of DetectedBox objects
        """
        pass
    
    def get_params(self) -> Dict:
        return self.params.copy()
    
    def get_output_dict(self, boxes: List[DetectedBox], filename: str = "") -> Dict:
        """Format output as expected by evaluation."""
        return {
            'filename': filename,
            'objects': [box.to_dict() for box in boxes]
        }


# ==================== Distance-Based Method ====================

class DistanceBasedDetector(BaseChangeDetector):
    """
    Simple distance-based change detection.
    Points beyond threshold from other cloud are clustered into change regions.
    """
    
    def __init__(
        self,
        distance_threshold: float = 0.05,
        cluster_eps: float = 0.05,
        min_cluster_points: int = 50,
        **kwargs
    ):
        super().__init__(
            distance_threshold=distance_threshold,
            cluster_eps=cluster_eps,
            min_cluster_points=min_cluster_points,
            **kwargs
        )
        self.distance_threshold = distance_threshold
        self.cluster_eps = cluster_eps
        self.min_cluster_points = min_cluster_points
        self.name = "Distance-Based"
    
    def detect(
        self,
        reference: np.ndarray,
        comparison: np.ndarray
    ) -> List[DetectedBox]:
        ref_xyz = reference[:, :3]
        comp_xyz = comparison[:, :3]
        
        # Build KD-trees
        ref_tree = cKDTree(ref_xyz)
        comp_tree = cKDTree(comp_xyz)
        
        # Find removed points (in reference but far from comparison)
        dist_ref_to_comp, _ = comp_tree.query(ref_xyz, k=1)
        removed_mask = dist_ref_to_comp > self.distance_threshold
        removed_points = ref_xyz[removed_mask]
        
        # Find added points (in comparison but far from reference)
        dist_comp_to_ref, _ = ref_tree.query(comp_xyz, k=1)
        added_mask = dist_comp_to_ref > self.distance_threshold
        added_points = comp_xyz[added_mask]
        
        # Cluster and create boxes
        boxes = []
        boxes.extend(self._cluster_to_boxes(removed_points, "Remove"))
        boxes.extend(self._cluster_to_boxes(added_points, "Add"))
        
        return boxes
    
    def _cluster_to_boxes(self, points: np.ndarray, label: str) -> List[DetectedBox]:
        """Cluster points and create bounding boxes."""
        if len(points) < self.min_cluster_points:
            return []
        
        clustering = DBSCAN(
            eps=self.cluster_eps,
            min_samples=self.min_cluster_points // 2
        ).fit(points)
        
        boxes = []
        for cluster_id in set(clustering.labels_):
            if cluster_id == -1:
                continue
            
            cluster_mask = clustering.labels_ == cluster_id
            cluster_points = points[cluster_mask]
            
            if len(cluster_points) < self.min_cluster_points:
                continue
            
            min_bound = cluster_points.min(axis=0)
            max_bound = cluster_points.max(axis=0)
            
            # Add padding
            delta = DELTA_MIN
            min_bound -= delta
            max_bound += delta
            
            boxes.append(DetectedBox(
                min_bound=min_bound,
                max_bound=max_bound,
                label=label,
                num_points=len(cluster_points)
            ))
        
        return boxes


# ==================== Octree-Based Method ====================

class OctreeBasedDetector(BaseChangeDetector):
    """
    Octree-based change detection.
    Compares voxel occupancy between point clouds.
    """
    
    def __init__(
        self,
        voxel_size: float = 0.05,
        min_points_per_voxel: int = 3,
        cluster_eps: float = 0.08,
        min_cluster_points: int = 30,
        **kwargs
    ):
        super().__init__(
            voxel_size=voxel_size,
            min_points_per_voxel=min_points_per_voxel,
            cluster_eps=cluster_eps,
            min_cluster_points=min_cluster_points,
            **kwargs
        )
        self.voxel_size = voxel_size
        self.min_points = min_points_per_voxel
        self.cluster_eps = cluster_eps
        self.min_cluster_points = min_cluster_points
        self.name = "Octree-Based"
    
    def _points_to_occupied_voxels(self, points: np.ndarray) -> set:
        """Convert points to set of occupied voxel indices."""
        voxel_indices = np.floor(points[:, :3] / self.voxel_size).astype(int)
        
        voxel_counts = {}
        for idx in map(tuple, voxel_indices):
            voxel_counts[idx] = voxel_counts.get(idx, 0) + 1
        
        return {k for k, v in voxel_counts.items() if v >= self.min_points}
    
    def _voxels_to_points(self, voxels: set) -> np.ndarray:
        """Convert voxel indices to center points."""
        if not voxels:
            return np.array([]).reshape(0, 3)
        
        centers = np.array(list(voxels)) * self.voxel_size + self.voxel_size / 2
        return centers
    
    def detect(
        self,
        reference: np.ndarray,
        comparison: np.ndarray
    ) -> List[DetectedBox]:
        ref_voxels = self._points_to_occupied_voxels(reference)
        comp_voxels = self._points_to_occupied_voxels(comparison)
        
        # Find changed voxels
        removed_voxels = ref_voxels - comp_voxels
        added_voxels = comp_voxels - ref_voxels
        
        # Convert to points and cluster
        removed_points = self._voxels_to_points(removed_voxels)
        added_points = self._voxels_to_points(added_voxels)
        
        boxes = []
        boxes.extend(self._cluster_to_boxes(removed_points, "Remove"))
        boxes.extend(self._cluster_to_boxes(added_points, "Add"))
        
        return boxes
    
    def _cluster_to_boxes(self, points: np.ndarray, label: str) -> List[DetectedBox]:
        """Cluster voxel centers and create bounding boxes."""
        if len(points) < 3:
            return []
        
        clustering = DBSCAN(
            eps=self.cluster_eps,
            min_samples=3
        ).fit(points)
        
        boxes = []
        for cluster_id in set(clustering.labels_):
            if cluster_id == -1:
                continue
            
            cluster_mask = clustering.labels_ == cluster_id
            cluster_points = points[cluster_mask]
            
            if len(cluster_points) < self.min_cluster_points // self.min_points:
                continue
            
            # Expand to cover voxel extent
            min_bound = cluster_points.min(axis=0) - self.voxel_size / 2
            max_bound = cluster_points.max(axis=0) + self.voxel_size / 2
            
            boxes.append(DetectedBox(
                min_bound=min_bound,
                max_bound=max_bound,
                label=label,
                num_points=len(cluster_points) * self.min_points
            ))
        
        return boxes


# ==================== ICP-Based Method ====================

class ICPBasedDetector(BaseChangeDetector):
    """
    ICP-based change detection.
    First refines alignment with ICP, then detects residuals.
    """
    
    def __init__(
        self,
        distance_threshold: float = 0.05,
        icp_threshold: float = 0.1,
        max_iterations: int = 50,
        cluster_eps: float = 0.05,
        min_cluster_points: int = 50,
        **kwargs
    ):
        super().__init__(
            distance_threshold=distance_threshold,
            icp_threshold=icp_threshold,
            max_iterations=max_iterations,
            cluster_eps=cluster_eps,
            min_cluster_points=min_cluster_points,
            **kwargs
        )
        self.distance_threshold = distance_threshold
        self.icp_threshold = icp_threshold
        self.max_iterations = max_iterations
        self.cluster_eps = cluster_eps
        self.min_cluster_points = min_cluster_points
        self.name = "ICP-Based"
    
    def detect(
        self,
        reference: np.ndarray,
        comparison: np.ndarray
    ) -> List[DetectedBox]:
        if o3d is None:
            raise ImportError("Open3D required for ICP-based detection")
        
        # Convert to Open3D
        ref_pcd = o3d.geometry.PointCloud()
        ref_pcd.points = o3d.utility.Vector3dVector(reference[:, :3])
        
        comp_pcd = o3d.geometry.PointCloud()
        comp_pcd.points = o3d.utility.Vector3dVector(comparison[:, :3])
        
        # Run ICP for refined alignment
        reg_result = o3d.pipelines.registration.registration_icp(
            comp_pcd, ref_pcd,
            self.icp_threshold,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.max_iterations
            )
        )
        
        # Transform comparison cloud
        comp_transformed = comparison.copy()
        comp_xyz_homo = np.hstack([comparison[:, :3], np.ones((len(comparison), 1))])
        comp_transformed[:, :3] = (reg_result.transformation @ comp_xyz_homo.T).T[:, :3]
        
        # Use distance-based detection on aligned clouds
        dist_detector = DistanceBasedDetector(
            distance_threshold=self.distance_threshold,
            cluster_eps=self.cluster_eps,
            min_cluster_points=self.min_cluster_points
        )
        
        return dist_detector.detect(reference, comp_transformed)


# ==================== Multi-Stage Method (Ours) ====================

class MultiStageDetector(BaseChangeDetector):
    """
    Multi-Stage Geometric Change Detection.
    
    Pipeline:
    1. Compute overlap region (ROI)
    2. Distance-based change point identification
    3. Boundary filtering to remove edge artifacts
    4. Adaptive DBSCAN clustering
    5. Bounding box generation with padding
    6. Graph-based box merging
    7. Enclosing box suppression
    
    This is the proposed method adapted from the original implementation.
    """
    
    def __init__(
        self,
        distance_threshold: float = 0.06,
        proximity_threshold: float = 0.03,
        boundary_threshold: float = 0.10,
        eps: float = 0.015,
        min_samples: int = 5,
        tolerance_factor: float = 1.05,
        roi_scale: float = 0.96,
        roi_coverage_thresh: float = 0.5,
        box_budget_per_class: int = 5,
        merge_iou_thr: float = 0.20,
        merge_center_thr: float = 0.07,
        nms_iou_thr: float = 0.70,
        **kwargs
    ):
        super().__init__(
            distance_threshold=distance_threshold,
            proximity_threshold=proximity_threshold,
            boundary_threshold=boundary_threshold,
            eps=eps,
            min_samples=min_samples,
            tolerance_factor=tolerance_factor,
            roi_scale=roi_scale,
            roi_coverage_thresh=roi_coverage_thresh,
            box_budget_per_class=box_budget_per_class,
            **kwargs
        )
        
        self.distance_threshold = distance_threshold
        self.proximity_threshold = proximity_threshold
        self.boundary_threshold = boundary_threshold
        self.eps = eps
        self.min_samples = min_samples
        self.tolerance_factor = tolerance_factor
        self.roi_scale = roi_scale
        self.roi_coverage_thresh = roi_coverage_thresh
        self.box_budget_per_class = box_budget_per_class
        self.merge_iou_thr = merge_iou_thr
        self.merge_center_thr = merge_center_thr
        self.nms_iou_thr = nms_iou_thr
        
        self.name = "Multi-Stage (Ours)"
    
    def detect(
        self,
        reference: np.ndarray,
        comparison: np.ndarray
    ) -> List[DetectedBox]:
        """
        Main detection pipeline.
        """
        ref_xyz = reference[:, :3]
        comp_xyz = comparison[:, :3]
        
        # Step 1: Compute overlap region
        overlap_points, roi_bounds = self._compute_overlap_roi(ref_xyz, comp_xyz)
        
        if overlap_points is None or len(overlap_points) == 0:
            return []
        
        # Step 2: Identify change points with distance threshold
        removed_points, added_points = self._identify_change_points(
            ref_xyz, comp_xyz, roi_bounds
        )
        
        # Step 3: Apply boundary filtering
        removed_points = self._boundary_filter(removed_points, roi_bounds)
        added_points = self._boundary_filter(added_points, roi_bounds)
        
        # Step 4 & 5: Cluster and create boxes
        removed_boxes = self._cluster_and_create_boxes(
            removed_points, "Remove", roi_bounds
        )
        added_boxes = self._cluster_and_create_boxes(
            added_points, "Add", roi_bounds
        )
        
        all_boxes = removed_boxes + added_boxes
        
        # Step 6: Select top-k by point support
        all_boxes = self._select_topk_by_support(
            all_boxes, removed_points, added_points
        )
        
        # Step 7: Graph-based box merging
        all_boxes = self._merge_boxes_graph(all_boxes)
        
        # Step 8: Suppress enclosing boxes
        all_boxes = self._suppress_enclosing_boxes(all_boxes)
        
        return all_boxes
    
    def _compute_overlap_roi(
        self,
        ref_xyz: np.ndarray,
        comp_xyz: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[np.ndarray, np.ndarray]]]:
        """Compute overlap region between two point clouds."""
        ref_tree = cKDTree(ref_xyz)
        comp_tree = cKDTree(comp_xyz)
        
        dist_threshold = self.distance_threshold * self.tolerance_factor
        
        # Find overlapping points
        overlap_points = []
        for point in ref_xyz:
            dist, _ = comp_tree.query(point, k=1)
            if dist <= dist_threshold:
                overlap_points.append(point)
        
        overlap_points = np.array(overlap_points)
        
        if len(overlap_points) == 0:
            return None, None
        
        # Compute ROI bounds
        roi_min = overlap_points.min(axis=0)
        roi_max = overlap_points.max(axis=0)
        
        # Dilate ROI
        roi_min, roi_max = dilate_aabb(roi_min, roi_max, self.roi_scale)
        
        return overlap_points, (roi_min, roi_max)
    
    def _identify_change_points(
        self,
        ref_xyz: np.ndarray,
        comp_xyz: np.ndarray,
        roi_bounds: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Identify removed and added points using distance threshold."""
        roi_min, roi_max = roi_bounds
        
        ref_tree = cKDTree(ref_xyz)
        comp_tree = cKDTree(comp_xyz)
        
        prox_threshold = self.proximity_threshold * self.tolerance_factor
        
        # Find removed points (in ref, far from comp)
        removed = []
        for i, point in enumerate(ref_xyz):
            # Check if in ROI
            if not np.all((point >= roi_min) & (point <= roi_max)):
                continue
            
            dist, _ = comp_tree.query(point, k=1)
            if dist > prox_threshold:
                removed.append(point)
        
        # Find added points (in comp, far from ref)
        added = []
        for i, point in enumerate(comp_xyz):
            if not np.all((point >= roi_min) & (point <= roi_max)):
                continue
            
            dist, _ = ref_tree.query(point, k=1)
            if dist > prox_threshold:
                added.append(point)
        
        return np.array(removed), np.array(added)
    
    def _boundary_filter(
        self,
        points: np.ndarray,
        roi_bounds: Tuple[np.ndarray, np.ndarray]
    ) -> np.ndarray:
        """Remove points too close to ROI boundary."""
        if len(points) == 0:
            return points
        
        roi_min, roi_max = roi_bounds
        boundary_thr = self.boundary_threshold * self.tolerance_factor
        
        # Keep points not too close to boundary
        mask = np.ones(len(points), dtype=bool)
        for i, point in enumerate(points):
            if (np.any(np.abs(point - roi_min) < boundary_thr) or
                np.any(np.abs(point - roi_max) < boundary_thr)):
                mask[i] = False
        
        return points[mask]
    
    def _cluster_and_create_boxes(
        self,
        points: np.ndarray,
        label: str,
        roi_bounds: Tuple[np.ndarray, np.ndarray]
    ) -> List[DetectedBox]:
        """Cluster change points and create bounding boxes."""
        if len(points) < self.min_samples:
            return []
        
        # Clean points
        points = points[~np.isnan(points).any(axis=1)]
        points = points[~np.isinf(points).any(axis=1)]
        
        if len(points) < self.min_samples:
            return []
        
        # Adaptive eps based on point density
        eps_local = max(self.eps, 1.6 * median_nn_spacing(points))
        
        # Cluster
        clustering = DBSCAN(eps=eps_local, min_samples=self.min_samples).fit(points)
        
        boxes = []
        roi_min, roi_max = roi_bounds
        
        for cluster_id in set(clustering.labels_):
            if cluster_id == -1:
                continue
            
            cluster_mask = clustering.labels_ == cluster_id
            cluster_points = points[cluster_mask]
            
            if len(cluster_points) < 10 or len(cluster_points) > 100000:
                continue
            
            # Create box
            min_bound = cluster_points.min(axis=0)
            max_bound = cluster_points.max(axis=0)
            
            # Add adaptive padding
            delta = max(PADDING_SCALE * median_nn_spacing(cluster_points), DELTA_MIN)
            min_bound -= delta
            max_bound += delta
            
            # ROI coverage check
            in_roi = points_in_aabb_mask(cluster_points, roi_min, roi_max)
            coverage = float(np.mean(in_roi)) if len(in_roi) > 0 else 0.0
            center = (min_bound + max_bound) / 2
            center_in_roi = np.all((center >= roi_min) & (center <= roi_max))
            
            if coverage < self.roi_coverage_thresh and not center_in_roi:
                continue
            
            boxes.append(DetectedBox(
                min_bound=min_bound,
                max_bound=max_bound,
                label=label,
                num_points=len(cluster_points)
            ))
        
        return boxes
    
    def _select_topk_by_support(
        self,
        boxes: List[DetectedBox],
        removed_points: np.ndarray,
        added_points: np.ndarray
    ) -> List[DetectedBox]:
        """Select top-k boxes per class by point support."""
        add_boxes = [b for b in boxes if b.label == "Add"]
        rem_boxes = [b for b in boxes if b.label == "Remove"]
        
        def score_and_select(box_list, support_points, k):
            if not box_list or len(support_points) == 0:
                return box_list[:k] if len(box_list) <= k else []
            
            scored = []
            for box in box_list:
                mask = points_in_aabb_mask(support_points, box.min_bound, box.max_bound)
                count = int(np.sum(mask))
                scored.append((count, box.diagonal, box))
            
            scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
            return [t[2] for t in scored[:k]]
        
        selected_add = score_and_select(add_boxes, added_points, self.box_budget_per_class)
        selected_rem = score_and_select(rem_boxes, removed_points, self.box_budget_per_class)
        
        return selected_add + selected_rem
    
    def _merge_boxes_graph(self, boxes: List[DetectedBox]) -> List[DetectedBox]:
        """Graph-based box merging."""
        add_boxes = [b for b in boxes if b.label == "Add"]
        rem_boxes = [b for b in boxes if b.label == "Remove"]
        
        merged_add = self._merge_class_boxes(add_boxes, "Add")
        merged_rem = self._merge_class_boxes(rem_boxes, "Remove")
        
        return merged_add + merged_rem
    
    def _merge_class_boxes(self, boxes: List[DetectedBox], label: str) -> List[DetectedBox]:
        """Merge boxes of the same class using graph-based approach."""
        if len(boxes) <= 1:
            return boxes
        
        n = len(boxes)
        
        # Build adjacency graph
        adj = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                iou = aabb_iou(
                    boxes[i].min_bound, boxes[i].max_bound,
                    boxes[j].min_bound, boxes[j].max_bound
                )
                
                center_dist = np.linalg.norm(
                    (boxes[i].min_bound + boxes[i].max_bound) / 2 -
                    (boxes[j].min_bound + boxes[j].max_bound) / 2
                )
                
                if iou >= self.merge_iou_thr or center_dist <= self.merge_center_thr:
                    adj[i].append(j)
                    adj[j].append(i)
        
        # Find connected components
        seen = [False] * n
        components = []
        
        for i in range(n):
            if seen[i]:
                continue
            
            component = []
            stack = [i]
            seen[i] = True
            
            while stack:
                u = stack.pop()
                component.append(u)
                for v in adj[u]:
                    if not seen[v]:
                        seen[v] = True
                        stack.append(v)
            
            components.append(component)
        
        # Merge each component
        merged = []
        for comp in components:
            comp_boxes = [boxes[i] for i in comp]
            
            # Union of all boxes in component
            all_mins = np.array([b.min_bound for b in comp_boxes])
            all_maxs = np.array([b.max_bound for b in comp_boxes])
            
            union_min = all_mins.min(axis=0)
            union_max = all_maxs.max(axis=0)
            
            total_points = sum(b.num_points for b in comp_boxes)
            
            merged.append(DetectedBox(
                min_bound=union_min,
                max_bound=union_max,
                label=label,
                num_points=total_points
            ))
        
        # NMS on merged boxes
        return self._nms_boxes(merged)
    
    def _nms_boxes(self, boxes: List[DetectedBox]) -> List[DetectedBox]:
        """Non-maximum suppression on boxes."""
        if len(boxes) <= 1:
            return boxes
        
        # Sort by volume (larger first)
        boxes = sorted(boxes, key=lambda b: b.volume, reverse=True)
        
        keep = []
        suppressed = set()
        
        for i, box_i in enumerate(boxes):
            if i in suppressed:
                continue
            
            keep.append(box_i)
            
            for j in range(i + 1, len(boxes)):
                if j in suppressed:
                    continue
                
                iou = aabb_iou(
                    box_i.min_bound, box_i.max_bound,
                    boxes[j].min_bound, boxes[j].max_bound
                )
                
                if iou >= self.nms_iou_thr:
                    suppressed.add(j)
        
        return keep
    
    def _suppress_enclosing_boxes(self, boxes: List[DetectedBox]) -> List[DetectedBox]:
        """Remove boxes that completely enclose smaller boxes (keep smaller)."""
        add_boxes = [b for b in boxes if b.label == "Add"]
        rem_boxes = [b for b in boxes if b.label == "Remove"]
        
        def process(box_list):
            if len(box_list) <= 1:
                return box_list
            
            keep = [True] * len(box_list)
            
            for i in range(len(box_list)):
                if not keep[i]:
                    continue
                
                for j in range(i + 1, len(box_list)):
                    if not keep[j]:
                        continue
                    
                    # Check if i contains j
                    if (np.all(box_list[i].min_bound <= box_list[j].min_bound) and
                        np.all(box_list[i].max_bound >= box_list[j].max_bound)):
                        keep[i] = False  # Remove larger (enclosing) box
                        break
                    
                    # Check if j contains i
                    if (np.all(box_list[j].min_bound <= box_list[i].min_bound) and
                        np.all(box_list[j].max_bound >= box_list[i].max_bound)):
                        keep[j] = False
            
            return [b for b, k in zip(box_list, keep) if k]
        
        return process(add_boxes) + process(rem_boxes)


# ==================== Factory Function ====================

def get_method(name: str, **kwargs) -> BaseChangeDetector:
    """
    Factory function to get change detection method by name.
    
    Args:
        name: Method name
        **kwargs: Method-specific parameters
    
    Returns:
        BaseChangeDetector instance
    """
    methods = {
        'distance': DistanceBasedDetector,
        'distance_based': DistanceBasedDetector,
        'octree': OctreeBasedDetector,
        'octree_based': OctreeBasedDetector,
        'icp': ICPBasedDetector,
        'icp_based': ICPBasedDetector,
        'multi_stage': MultiStageDetector,
        'multistage': MultiStageDetector,
        'ours': MultiStageDetector,
    }
    
    name_lower = name.lower().replace('-', '_')
    if name_lower not in methods:
        raise ValueError(f"Unknown method: {name}. Available: {list(methods.keys())}")
    
    return methods[name_lower](**kwargs)


# ==================== Test ====================

if __name__ == "__main__":
    print("Testing change detection methods...")
    
    # Create synthetic test data
    np.random.seed(42)
    
    # Reference: room with some objects
    reference = np.random.randn(5000, 3) * 2
    
    # Comparison: same room with changes
    comparison = reference.copy()
    
    # Simulate removal
    remove_mask = np.random.random(len(comparison)) > 0.9
    comparison = comparison[~remove_mask]
    
    # Simulate addition
    added = np.random.randn(300, 3) * 0.5 + np.array([3, 0, 0])
    comparison = np.vstack([comparison, added])
    
    print(f"Reference: {len(reference)} points")
    print(f"Comparison: {len(comparison)} points")
    
    # Test each method
    for method_name in ['distance', 'octree', 'multi_stage']:
        print(f"\n{method_name}:")
        
        detector = get_method(method_name)
        boxes = detector.detect(reference, comparison)
        
        add_count = sum(1 for b in boxes if b.label == "Add")
        rem_count = sum(1 for b in boxes if b.label == "Remove")
        
        print(f"  Detected {len(boxes)} boxes ({add_count} Add, {rem_count} Remove)")
