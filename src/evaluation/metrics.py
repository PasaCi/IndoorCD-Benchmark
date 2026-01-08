"""
IndoorCD Evaluation Metrics
===========================
Bounding box based evaluation for 3D change detection.
Uses IoU-based matching similar to object detection evaluation.

Label Format:
{
    "filename": "001-2.pcd",
    "objects": [
        {"name": "Add" or "Remove", "vertices": [[x,y,z], ...]}  # 8 corners
    ]
}
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json
from pathlib import Path


@dataclass
class BoundingBox:
    """Represents a 3D bounding box."""
    vertices: np.ndarray  # (8, 3) array of corner points
    label: str  # "Add" or "Remove"
    confidence: float = 1.0
    
    @property
    def min_bound(self) -> np.ndarray:
        return self.vertices.min(axis=0)
    
    @property
    def max_bound(self) -> np.ndarray:
        return self.vertices.max(axis=0)
    
    @property
    def center(self) -> np.ndarray:
        return (self.min_bound + self.max_bound) / 2
    
    @property
    def dimensions(self) -> np.ndarray:
        return self.max_bound - self.min_bound
    
    @property
    def volume(self) -> float:
        dims = np.maximum(self.dimensions, 0)
        return float(dims[0] * dims[1] * dims[2])
    
    @classmethod
    def from_dict(cls, obj_dict: Dict) -> 'BoundingBox':
        """Create from label dictionary."""
        vertices = np.array(obj_dict['vertices'], dtype=float)
        return cls(vertices=vertices, label=obj_dict['name'])
    
    @classmethod
    def from_min_max(cls, min_bound: np.ndarray, max_bound: np.ndarray, 
                     label: str) -> 'BoundingBox':
        """Create from min/max bounds."""
        min_b = np.asarray(min_bound)
        max_b = np.asarray(max_bound)
        
        # Generate 8 corners
        vertices = np.array([
            [min_b[0], min_b[1], min_b[2]],
            [max_b[0], min_b[1], min_b[2]],
            [max_b[0], max_b[1], min_b[2]],
            [min_b[0], max_b[1], min_b[2]],
            [min_b[0], min_b[1], max_b[2]],
            [max_b[0], min_b[1], max_b[2]],
            [max_b[0], max_b[1], max_b[2]],
            [min_b[0], max_b[1], max_b[2]],
        ])
        return cls(vertices=vertices, label=label)


def compute_aabb_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """
    Compute Intersection over Union between two axis-aligned bounding boxes.
    
    Args:
        box1, box2: BoundingBox instances
    
    Returns:
        IoU value between 0 and 1
    """
    min1, max1 = box1.min_bound, box1.max_bound
    min2, max2 = box2.min_bound, box2.max_bound
    
    # Intersection bounds
    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)
    
    # Intersection dimensions (clamped to 0)
    inter_dims = np.maximum(inter_max - inter_min, 0)
    inter_volume = float(inter_dims[0] * inter_dims[1] * inter_dims[2])
    
    # Union = V1 + V2 - intersection
    union_volume = box1.volume + box2.volume - inter_volume
    
    if union_volume <= 0:
        return 0.0
    
    return inter_volume / union_volume


@dataclass
class DetectionMetrics:
    """Container for detection evaluation metrics."""
    # Per-class metrics
    precision: Dict[str, float] = field(default_factory=dict)
    recall: Dict[str, float] = field(default_factory=dict)
    f1: Dict[str, float] = field(default_factory=dict)
    ap: Dict[str, float] = field(default_factory=dict)  # Average Precision
    
    # Aggregated metrics
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_f1: float = 0.0
    mean_ap: float = 0.0
    
    # Matched IoU statistics
    mean_iou_matched: float = 0.0
    
    # Counts
    total_tp: int = 0
    total_fp: int = 0
    total_fn: int = 0
    num_scenes: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'macro_precision': self.macro_precision,
            'macro_recall': self.macro_recall,
            'macro_f1': self.macro_f1,
            'mean_iou_matched': self.mean_iou_matched,
            'total_tp': self.total_tp,
            'total_fp': self.total_fp,
            'total_fn': self.total_fn,
            'num_scenes': self.num_scenes,
        }
    
    def __str__(self) -> str:
        lines = [
            "Detection Metrics:",
            f"  Macro Precision: {self.macro_precision:.4f}",
            f"  Macro Recall: {self.macro_recall:.4f}",
            f"  Macro F1: {self.macro_f1:.4f}",
            f"  Mean IoU (matched): {self.mean_iou_matched:.4f}",
            f"  TP: {self.total_tp}, FP: {self.total_fp}, FN: {self.total_fn}",
        ]
        for cls in ['Add', 'Remove']:
            if cls in self.precision:
                lines.append(f"  {cls}: P={self.precision[cls]:.3f}, "
                           f"R={self.recall[cls]:.3f}, F1={self.f1[cls]:.3f}")
        return "\n".join(lines)


def parse_predictions(pred_boxes: List[Dict]) -> List[BoundingBox]:
    """Parse prediction list to BoundingBox objects."""
    boxes = []
    for obj in pred_boxes:
        if 'vertices' in obj and 'name' in obj:
            if obj['name'] in ['Add', 'Remove']:
                boxes.append(BoundingBox.from_dict(obj))
    return boxes


def parse_ground_truth(gt_data: Dict) -> List[BoundingBox]:
    """Parse ground truth JSON to BoundingBox objects."""
    if gt_data is None:
        return []
    
    boxes = []
    for obj in gt_data.get('objects', []):
        if 'vertices' in obj and 'name' in obj:
            if obj['name'] in ['Add', 'Remove']:
                boxes.append(BoundingBox.from_dict(obj))
    return boxes


def match_boxes_greedy(
    predictions: List[BoundingBox],
    ground_truths: List[BoundingBox],
    iou_threshold: float = 0.25
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """
    Greedy matching of predictions to ground truths based on IoU.
    
    Only matches boxes of the same class.
    
    Args:
        predictions: List of predicted bounding boxes
        ground_truths: List of ground truth bounding boxes
        iou_threshold: Minimum IoU for a valid match
    
    Returns:
        Tuple of:
        - matches: List of (pred_idx, gt_idx, iou) tuples
        - unmatched_preds: List of unmatched prediction indices (FP)
        - unmatched_gts: List of unmatched ground truth indices (FN)
    """
    if not predictions or not ground_truths:
        return ([], list(range(len(predictions))), list(range(len(ground_truths))))
    
    # Compute IoU matrix (only for same-class pairs)
    iou_pairs = []
    for pi, pred in enumerate(predictions):
        for gi, gt in enumerate(ground_truths):
            if pred.label == gt.label:  # Same class
                iou = compute_aabb_iou(pred, gt)
                if iou >= iou_threshold:
                    iou_pairs.append((iou, pi, gi))
    
    # Sort by IoU descending
    iou_pairs.sort(reverse=True, key=lambda x: x[0])
    
    # Greedy matching
    matched_preds = set()
    matched_gts = set()
    matches = []
    
    for iou, pi, gi in iou_pairs:
        if pi not in matched_preds and gi not in matched_gts:
            matches.append((pi, gi, iou))
            matched_preds.add(pi)
            matched_gts.add(gi)
    
    unmatched_preds = [i for i in range(len(predictions)) if i not in matched_preds]
    unmatched_gts = [i for i in range(len(ground_truths)) if i not in matched_gts]
    
    return matches, unmatched_preds, unmatched_gts


def evaluate_single_scene(
    predictions: List[BoundingBox],
    ground_truths: List[BoundingBox],
    iou_threshold: float = 0.25,
    classes: List[str] = ['Add', 'Remove']
) -> Dict:
    """
    Evaluate predictions for a single scene.
    
    Args:
        predictions: Predicted bounding boxes
        ground_truths: Ground truth bounding boxes
        iou_threshold: IoU threshold for matching
        classes: List of class names to evaluate
    
    Returns:
        Dict with per-class TP, FP, FN counts and matched IoUs
    """
    results = {cls: {'TP': 0, 'FP': 0, 'FN': 0, 'IoUs': []} for cls in classes}
    
    # Separate by class
    pred_by_class = {cls: [] for cls in classes}
    gt_by_class = {cls: [] for cls in classes}
    
    for box in predictions:
        if box.label in classes:
            pred_by_class[box.label].append(box)
    
    for box in ground_truths:
        if box.label in classes:
            gt_by_class[box.label].append(box)
    
    # Match within each class
    for cls in classes:
        matches, unmatched_preds, unmatched_gts = match_boxes_greedy(
            pred_by_class[cls], 
            gt_by_class[cls],
            iou_threshold
        )
        
        results[cls]['TP'] = len(matches)
        results[cls]['FP'] = len(unmatched_preds)
        results[cls]['FN'] = len(unmatched_gts)
        results[cls]['IoUs'] = [m[2] for m in matches]
    
    return results


def evaluate_dataset(
    all_predictions: List[List[BoundingBox]],
    all_ground_truths: List[List[BoundingBox]],
    iou_threshold: float = 0.25,
    classes: List[str] = ['Add', 'Remove']
) -> DetectionMetrics:
    """
    Evaluate predictions across entire dataset.
    
    Args:
        all_predictions: List of predictions per scene
        all_ground_truths: List of ground truths per scene
        iou_threshold: IoU threshold for matching
        classes: List of class names
    
    Returns:
        DetectionMetrics object with aggregated results
    """
    totals = {cls: {'TP': 0, 'FP': 0, 'FN': 0, 'IoUs': []} for cls in classes}
    
    for preds, gts in zip(all_predictions, all_ground_truths):
        scene_results = evaluate_single_scene(preds, gts, iou_threshold, classes)
        
        for cls in classes:
            totals[cls]['TP'] += scene_results[cls]['TP']
            totals[cls]['FP'] += scene_results[cls]['FP']
            totals[cls]['FN'] += scene_results[cls]['FN']
            totals[cls]['IoUs'].extend(scene_results[cls]['IoUs'])
    
    # Compute per-class metrics
    metrics = DetectionMetrics(num_scenes=len(all_predictions))
    all_ious = []
    
    for cls in classes:
        tp = totals[cls]['TP']
        fp = totals[cls]['FP']
        fn = totals[cls]['FN']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics.precision[cls] = precision
        metrics.recall[cls] = recall
        metrics.f1[cls] = f1
        
        metrics.total_tp += tp
        metrics.total_fp += fp
        metrics.total_fn += fn
        
        all_ious.extend(totals[cls]['IoUs'])
    
    # Compute macro averages
    if classes:
        metrics.macro_precision = np.mean([metrics.precision[c] for c in classes])
        metrics.macro_recall = np.mean([metrics.recall[c] for c in classes])
        metrics.macro_f1 = np.mean([metrics.f1[c] for c in classes])
    
    # Mean IoU of matched boxes
    if all_ious:
        metrics.mean_iou_matched = float(np.mean(all_ious))
    
    return metrics


class BenchmarkEvaluator:
    """
    Main evaluator class for IndoorCD benchmark.
    """
    
    def __init__(
        self,
        iou_threshold: float = 0.25,
        classes: List[str] = ['Add', 'Remove']
    ):
        """
        Initialize evaluator.
        
        Args:
            iou_threshold: IoU threshold for box matching
            classes: Classes to evaluate
        """
        self.iou_threshold = iou_threshold
        self.classes = classes
        self.results = {}
    
    def evaluate_method(
        self,
        method_name: str,
        predictions: List[List[Dict]],  # List of prediction dicts per scene
        ground_truths: List[Dict],  # List of GT dicts per scene
    ) -> DetectionMetrics:
        """
        Evaluate a single method's predictions.
        
        Args:
            method_name: Name of the method
            predictions: List of prediction lists (one per scene)
            ground_truths: List of ground truth dicts (one per scene)
        
        Returns:
            DetectionMetrics for this method
        """
        # Parse to BoundingBox objects
        all_preds = [parse_predictions(p.get('objects', []) if isinstance(p, dict) else p) 
                     for p in predictions]
        all_gts = [parse_ground_truth(gt) for gt in ground_truths]
        
        # Evaluate
        metrics = evaluate_dataset(
            all_preds, all_gts, 
            self.iou_threshold, 
            self.classes
        )
        
        self.results[method_name] = metrics
        return metrics
    
    def evaluate_from_files(
        self,
        method_name: str,
        pred_dir: str,
        gt_dir: str
    ) -> DetectionMetrics:
        """
        Evaluate predictions from JSON files.
        
        Args:
            method_name: Name of the method
            pred_dir: Directory containing prediction JSON files
            gt_dir: Directory containing ground truth JSON files
        
        Returns:
            DetectionMetrics for this method
        """
        pred_dir = Path(pred_dir)
        gt_dir = Path(gt_dir)
        
        all_preds = []
        all_gts = []
        
        for pred_file in sorted(pred_dir.glob("*.json")):
            gt_file = gt_dir / pred_file.name
            
            if not gt_file.exists():
                continue
            
            with open(pred_file, 'r') as f:
                pred_data = json.load(f)
            with open(gt_file, 'r') as f:
                gt_data = json.load(f)
            
            all_preds.append(parse_predictions(pred_data.get('objects', [])))
            all_gts.append(parse_ground_truth(gt_data))
        
        # Evaluate
        metrics = evaluate_dataset(
            all_preds, all_gts,
            self.iou_threshold,
            self.classes
        )
        
        self.results[method_name] = metrics
        return metrics
    
    def compare_methods(self) -> Dict:
        """Compare all evaluated methods."""
        comparison = {
            'methods': list(self.results.keys()),
            'iou_threshold': self.iou_threshold,
            'by_metric': {}
        }
        
        for metric in ['macro_precision', 'macro_recall', 'macro_f1', 'mean_iou_matched']:
            comparison['by_metric'][metric] = {
                method: getattr(self.results[method], metric)
                for method in self.results
            }
        
        # Ranking by F1
        comparison['ranking'] = sorted(
            self.results.keys(),
            key=lambda m: self.results[m].macro_f1,
            reverse=True
        )
        
        return comparison
    
    def generate_latex_table(self) -> str:
        """Generate LaTeX table for paper."""
        if not self.results:
            return ""
        
        methods = list(self.results.keys())
        
        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += "\\caption{Change Detection Results on IndoorCD (IoU threshold = "
        latex += f"{self.iou_threshold})}}}\n"
        latex += "\\label{tab:results}\n"
        latex += "\\begin{tabular}{l|cc|cc|ccc}\n"
        latex += "\\toprule\n"
        latex += "\\multirow{2}{*}{Method} & \\multicolumn{2}{c|}{Add} & "
        latex += "\\multicolumn{2}{c|}{Remove} & \\multicolumn{3}{c}{Overall} \\\\\n"
        latex += " & P & R & P & R & P & R & F1 \\\\\n"
        latex += "\\midrule\n"
        
        for method in methods:
            m = self.results[method]
            add_p = m.precision.get('Add', 0)
            add_r = m.recall.get('Add', 0)
            rem_p = m.precision.get('Remove', 0)
            rem_r = m.recall.get('Remove', 0)
            
            latex += f"{method} & {add_p:.3f} & {add_r:.3f} & "
            latex += f"{rem_p:.3f} & {rem_r:.3f} & "
            latex += f"{m.macro_precision:.3f} & {m.macro_recall:.3f} & "
            latex += f"{m.macro_f1:.3f} \\\\\n"
        
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        return latex
    
    def generate_per_category_table(self) -> str:
        """Generate detailed per-category results table."""
        if not self.results:
            return ""
        
        methods = list(self.results.keys())
        
        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += "\\caption{Per-Category Detection Performance}\n"
        latex += "\\label{tab:per_category}\n"
        latex += "\\begin{tabular}{l|ccc|ccc}\n"
        latex += "\\toprule\n"
        latex += " & \\multicolumn{3}{c|}{Add} & \\multicolumn{3}{c}{Remove} \\\\\n"
        latex += "Method & Precision & Recall & F1 & Precision & Recall & F1 \\\\\n"
        latex += "\\midrule\n"
        
        for method in methods:
            m = self.results[method]
            latex += f"{method} & "
            latex += f"{m.precision.get('Add', 0):.3f} & "
            latex += f"{m.recall.get('Add', 0):.3f} & "
            latex += f"{m.f1.get('Add', 0):.3f} & "
            latex += f"{m.precision.get('Remove', 0):.3f} & "
            latex += f"{m.recall.get('Remove', 0):.3f} & "
            latex += f"{m.f1.get('Remove', 0):.3f} \\\\\n"
        
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        return latex


if __name__ == "__main__":
    # Test evaluation
    print("Testing evaluation metrics...")
    
    # Create sample boxes
    gt_boxes = [
        BoundingBox.from_min_max([0, 0, 0], [1, 1, 1], "Add"),
        BoundingBox.from_min_max([2, 2, 2], [3, 3, 3], "Remove"),
    ]
    
    pred_boxes = [
        BoundingBox.from_min_max([0.1, 0.1, 0.1], [0.9, 0.9, 0.9], "Add"),  # Good match
        BoundingBox.from_min_max([5, 5, 5], [6, 6, 6], "Add"),  # FP
    ]
    
    # Test IoU
    iou = compute_aabb_iou(gt_boxes[0], pred_boxes[0])
    print(f"IoU between GT and Pred Add box: {iou:.4f}")
    
    # Test single scene evaluation
    results = evaluate_single_scene(pred_boxes, gt_boxes, iou_threshold=0.25)
    print(f"\nSingle scene results: {results}")
    
    # Test evaluator
    evaluator = BenchmarkEvaluator(iou_threshold=0.25)
    metrics = evaluate_dataset([pred_boxes], [gt_boxes], iou_threshold=0.25)
    print(f"\nDataset metrics:\n{metrics}")
