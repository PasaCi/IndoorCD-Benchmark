"""
IndoorCD Dataset Loader - Adapted for actual dataset structure
==============================================================
Handles the specific structure:
- Data: Dataset/Data/{room_id}/{room_id}-{scene_id}.pcd
- Labels: Dataset/Label/{room_id}-{scene_id}.json
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
from dataclasses import dataclass
import random

try:
    import open3d as o3d
except ImportError:
    o3d = None
    print("Warning: Open3D not installed.")


@dataclass
class ScenePair:
    """Represents a single scene pair (before/after) with metadata."""
    room_id: str
    scene_id: str  # 2, 3, 4, or 5
    reference_path: Path
    comparison_path: Path
    label_path: Optional[Path]
    change_type: str  # Add, Remove, Move, Composite, or Unknown
    
    @property
    def pair_id(self) -> str:
        return f"{self.room_id}-{self.scene_id}"


# Change type mapping based on scene_id convention
# XXX-2: Remove (objects removed from reference)
# XXX-3: Add (objects added to comparison)
# XXX-4: Move (objects relocated)
# XXX-5: Composite (multiple change types)
SCENE_ID_TO_CHANGE_TYPE = {
    '2': 'Remove',
    '3': 'Add', 
    '4': 'Move',
    '5': 'Composite'
}


class IndoorCDDataset:
    """
    Dataset class for IndoorCD benchmark.
    
    Expected structure:
    root/
    ├── Data/
    │   ├── 001/
    │   │   ├── 001-1.pcd  (reference)
    │   │   ├── 001-2.pcd  (Add)
    │   │   ├── 001-3.pcd  (Remove)
    │   │   ├── 001-4.pcd  (Move)
    │   │   └── 001-5.pcd  (Composite)
    │   └── ...
    └── Label/
        ├── 001-2.json
        └── ...
    """
    
    CHANGE_CATEGORIES = {
        0: 'NoChange',
        1: 'Add',
        2: 'Remove',
        3: 'Move',
        4: 'Composite'
    }
    
    CATEGORY_TO_ID = {v: k for k, v in CHANGE_CATEGORIES.items()}
    
    def __init__(
        self,
        root_path: str,
        data_subdir: str = "Data",
        label_subdir: str = "Label",
        file_extension: str = ".pcd"
    ):
        """
        Initialize dataset.
        
        Args:
            root_path: Root directory containing Data and Label folders
            data_subdir: Subdirectory name for point cloud data
            label_subdir: Subdirectory name for labels
            file_extension: Point cloud file extension (.pcd, .ply, etc.)
        """
        self.root_path = Path(root_path)
        self.data_dir = self.root_path / data_subdir
        self.label_dir = self.root_path / label_subdir
        self.file_extension = file_extension
        
        self.pairs: List[ScenePair] = []
        self.rooms: List[str] = []
        
        self._scan_dataset()
    
    def _scan_dataset(self):
        """Scan dataset directory and index all scene pairs."""
        if not self.data_dir.exists():
            print(f"Warning: Data directory {self.data_dir} does not exist.")
            return
        
        # Find all room directories
        room_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        for room_dir in room_dirs:
            room_id = room_dir.name
            self.rooms.append(room_id)
            
            # Find reference file (-1)
            reference_file = room_dir / f"{room_id}-1{self.file_extension}"
            if not reference_file.exists():
                # Try alternative naming
                ref_candidates = list(room_dir.glob(f"*-1{self.file_extension}"))
                if ref_candidates:
                    reference_file = ref_candidates[0]
                else:
                    print(f"Warning: No reference file found for room {room_id}")
                    continue
            
            # Find comparison files (-2, -3, -4, -5)
            for scene_id in ['2', '3', '4', '5']:
                comparison_file = room_dir / f"{room_id}-{scene_id}{self.file_extension}"
                
                if not comparison_file.exists():
                    # Try alternative naming
                    comp_candidates = list(room_dir.glob(f"*-{scene_id}{self.file_extension}"))
                    if comp_candidates:
                        comparison_file = comp_candidates[0]
                    else:
                        continue
                
                # Find corresponding label
                label_file = self.label_dir / f"{room_id}-{scene_id}.json"
                if not label_file.exists():
                    label_file = None
                
                # Determine change type
                change_type = SCENE_ID_TO_CHANGE_TYPE.get(scene_id, 'Unknown')
                
                pair = ScenePair(
                    room_id=room_id,
                    scene_id=scene_id,
                    reference_path=reference_file,
                    comparison_path=comparison_file,
                    label_path=label_file,
                    change_type=change_type
                )
                self.pairs.append(pair)
        
        print(f"Found {len(self.pairs)} scene pairs from {len(self.rooms)} rooms")
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single scene pair.
        
        Returns:
            Dict with keys:
            - 'room_id': Room identifier
            - 'scene_id': Scene identifier (2-5)
            - 'pair_id': Combined identifier
            - 'change_type': Type of change (Add, Remove, Move, Composite)
            - 'reference': Reference point cloud (N, 3) or (N, 6)
            - 'comparison': Comparison point cloud (M, 3) or (M, 6)
            - 'labels': Ground truth bounding boxes (if available)
            - 'reference_path': Path to reference file
            - 'comparison_path': Path to comparison file
        """
        pair = self.pairs[idx]
        
        result = {
            'room_id': pair.room_id,
            'scene_id': pair.scene_id,
            'pair_id': pair.pair_id,
            'change_type': pair.change_type,
            'reference': self.load_point_cloud(pair.reference_path),
            'comparison': self.load_point_cloud(pair.comparison_path),
            'labels': self.load_labels(pair.label_path),
            'reference_path': str(pair.reference_path),
            'comparison_path': str(pair.comparison_path),
        }
        
        return result
    
    def get_pair_by_id(self, pair_id: str) -> Optional[Dict]:
        """Get scene pair by its ID (e.g., '001-2')."""
        for i, pair in enumerate(self.pairs):
            if pair.pair_id == pair_id:
                return self[i]
        return None
    
    def get_pairs_by_room(self, room_id: str) -> List[Dict]:
        """Get all scene pairs for a specific room."""
        results = []
        for i, pair in enumerate(self.pairs):
            if pair.room_id == room_id:
                results.append(self[i])
        return results
    
    def get_pairs_by_change_type(self, change_type: str) -> List[int]:
        """Get indices of pairs with specific change type."""
        return [i for i, pair in enumerate(self.pairs) 
                if pair.change_type == change_type]
    
    @staticmethod
    def load_point_cloud(path: Path) -> Optional[np.ndarray]:
        """
        Load point cloud from file.
        
        Returns:
            np.ndarray of shape (N, 3) for XYZ or (N, 6) for XYZRGB
        """
        if path is None or not Path(path).exists():
            return None
        
        path = Path(path)
        
        if o3d is None:
            raise ImportError("Open3D required for loading point clouds")
        
        pcd = o3d.io.read_point_cloud(str(path))
        points = np.asarray(pcd.points)
        
        if len(points) == 0:
            return None
        
        # Include colors if available
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
            return np.hstack([points, colors])
        
        return points
    
    @staticmethod
    def load_labels(path: Optional[Path]) -> Optional[Dict]:
        """
        Load ground truth labels from JSON file.
        
        Expected format:
        {
            "filename": "001-2.pcd",
            "objects": [
                {
                    "name": "Add" or "Remove",
                    "vertices": [[x1,y1,z1], ..., [x8,y8,z8]]  # 8 corners
                },
                ...
            ]
        }
        
        Returns:
            Dict with 'filename' and 'objects' keys, or None
        """
        if path is None or not Path(path).exists():
            return None
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error loading labels from {path}: {e}")
            return None
    
    def get_statistics(self) -> Dict:
        """Calculate dataset statistics."""
        stats = {
            'total_pairs': len(self.pairs),
            'total_rooms': len(self.rooms),
            'pairs_by_change_type': defaultdict(int),
            'pairs_with_labels': 0,
            'point_counts': {'reference': [], 'comparison': []},
        }
        
        for pair in self.pairs:
            stats['pairs_by_change_type'][pair.change_type] += 1
            if pair.label_path is not None:
                stats['pairs_with_labels'] += 1
        
        # Sample point counts (first 50 pairs)
        sample_size = min(50, len(self.pairs))
        for i in range(sample_size):
            try:
                data = self[i]
                if data['reference'] is not None:
                    stats['point_counts']['reference'].append(len(data['reference']))
                if data['comparison'] is not None:
                    stats['point_counts']['comparison'].append(len(data['comparison']))
            except:
                continue
        
        if stats['point_counts']['reference']:
            stats['avg_points_reference'] = np.mean(stats['point_counts']['reference'])
            stats['avg_points_comparison'] = np.mean(stats['point_counts']['comparison'])
        
        stats['pairs_by_change_type'] = dict(stats['pairs_by_change_type'])
        
        return stats


def create_fixed_split(
    dataset: IndoorCDDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    by_room: bool = True,
    save_path: Optional[str] = None
) -> Tuple[List[int], List[int], List[int]]:
    """
    Create reproducible train/val/test split.
    
    IMPORTANT: For academic consistency, this split should be fixed
    and used across all experiments.
    
    Args:
        dataset: IndoorCDDataset instance
        train_ratio, val_ratio, test_ratio: Split ratios (must sum to 1.0)
        seed: Random seed for reproducibility
        by_room: If True, split by room (no room appears in multiple splits)
        save_path: If provided, save split indices to JSON
    
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    np.random.seed(seed)
    random.seed(seed)
    
    if by_room:
        # Split by room to avoid data leakage
        rooms = list(dataset.rooms)
        random.shuffle(rooms)
        
        n_rooms = len(rooms)
        n_train = int(n_rooms * train_ratio)
        n_val = int(n_rooms * val_ratio)
        
        train_rooms = set(rooms[:n_train])
        val_rooms = set(rooms[n_train:n_train + n_val])
        test_rooms = set(rooms[n_train + n_val:])
        
        train_indices = [i for i, p in enumerate(dataset.pairs) if p.room_id in train_rooms]
        val_indices = [i for i, p in enumerate(dataset.pairs) if p.room_id in val_rooms]
        test_indices = [i for i, p in enumerate(dataset.pairs) if p.room_id in test_rooms]
        
        split_info = {
            'seed': seed,
            'by_room': True,
            'train_rooms': sorted(list(train_rooms)),
            'val_rooms': sorted(list(val_rooms)),
            'test_rooms': sorted(list(test_rooms)),
        }
    else:
        # Random split by pair
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        
        n_train = int(len(indices) * train_ratio)
        n_val = int(len(indices) * val_ratio)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        split_info = {
            'seed': seed,
            'by_room': False,
        }
    
    # Add pair IDs for reference
    split_info['train_pairs'] = [dataset.pairs[i].pair_id for i in train_indices]
    split_info['val_pairs'] = [dataset.pairs[i].pair_id for i in val_indices]
    split_info['test_pairs'] = [dataset.pairs[i].pair_id for i in test_indices]
    split_info['train_count'] = len(train_indices)
    split_info['val_count'] = len(val_indices)
    split_info['test_count'] = len(test_indices)
    
    # Save if path provided
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        print(f"Split saved to {save_path}")
    
    return train_indices, val_indices, test_indices


def load_split(split_path: str, dataset: IndoorCDDataset) -> Tuple[List[int], List[int], List[int]]:
    """
    Load a previously saved split.
    
    Args:
        split_path: Path to split JSON file
        dataset: IndoorCDDataset instance
    
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    with open(split_path, 'r') as f:
        split_info = json.load(f)
    
    # Create pair_id to index mapping
    pair_id_to_idx = {p.pair_id: i for i, p in enumerate(dataset.pairs)}
    
    train_indices = [pair_id_to_idx[pid] for pid in split_info['train_pairs'] 
                     if pid in pair_id_to_idx]
    val_indices = [pair_id_to_idx[pid] for pid in split_info['val_pairs'] 
                   if pid in pair_id_to_idx]
    test_indices = [pair_id_to_idx[pid] for pid in split_info['test_pairs'] 
                    if pid in pair_id_to_idx]
    
    return train_indices, val_indices, test_indices


def get_split_by_change_type(
    dataset: IndoorCDDataset,
    indices: List[int]
) -> Dict[str, List[int]]:
    """
    Organize indices by change type.
    
    Args:
        dataset: IndoorCDDataset instance
        indices: List of indices to organize
    
    Returns:
        Dict mapping change_type to list of indices
    """
    result = defaultdict(list)
    for idx in indices:
        change_type = dataset.pairs[idx].change_type
        result[change_type].append(idx)
    return dict(result)


if __name__ == "__main__":
    # Test with example path
    import sys
    
    # Default test path (adjust as needed)
    if len(sys.argv) > 1:
        root_path = sys.argv[1]
    else:
        # Windows path example
        root_path = r"C:\Users\topra\Desktop\Ph.D\New Datasets\Files\Dataset"
    
    print(f"Testing dataset loader with: {root_path}")
    
    dataset = IndoorCDDataset(root_path)
    print(f"\nFound {len(dataset)} scene pairs")
    
    if len(dataset) > 0:
        # Print statistics
        stats = dataset.get_statistics()
        print(f"\nDataset Statistics:")
        print(f"  Total pairs: {stats['total_pairs']}")
        print(f"  Total rooms: {stats['total_rooms']}")
        print(f"  Pairs with labels: {stats['pairs_with_labels']}")
        print(f"  By change type: {stats['pairs_by_change_type']}")
        
        # Create and save split
        train_idx, val_idx, test_idx = create_fixed_split(
            dataset,
            seed=42,
            by_room=True,
            save_path="dataset_split.json"
        )
        
        print(f"\nSplit created:")
        print(f"  Train: {len(train_idx)} pairs")
        print(f"  Val: {len(val_idx)} pairs")
        print(f"  Test: {len(test_idx)} pairs")
        
        # Test loading a pair
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\nSample pair: {sample['pair_id']}")
            print(f"  Change type: {sample['change_type']}")
            print(f"  Reference shape: {sample['reference'].shape if sample['reference'] is not None else None}")
            print(f"  Comparison shape: {sample['comparison'].shape if sample['comparison'] is not None else None}")
            print(f"  Has labels: {sample['labels'] is not None}")
