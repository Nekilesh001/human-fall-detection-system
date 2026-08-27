"""
PyTorch Dataset Loader for URFD RGB Baseline
Loads processed temporal windows (.npz) lazily, converts uint8 to float32,
normalizes with ImageNet mean/std, and exposes train/val/test partitions.
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class URFDRGBDataset(Dataset):
    """
    Lazy-loading PyTorch Dataset for URFD 50-frame RGB temporal windows.
    """
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    LABEL_MAP = {"NORMAL": 0, "FALL": 1}

    def __init__(self, partition="train", root_dir=None, apply_augmentations=False):
        super().__init__()
        self.partition = partition
        
        if root_dir is None:
            self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        else:
            self.root_dir = root_dir

        manifest_path = os.path.join(
            self.root_dir, "processed_data", "URFD_RGB_baseline", "processed_manifest.csv"
        )
        
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Processed manifest not found at {manifest_path}")

        df_all = pd.read_csv(manifest_path)
        self.df = df_all[df_all["partition"] == partition].reset_index(drop=True)
        self.apply_augmentations = apply_augmentations and (partition == "train")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_rel = str(row["processed_sample_path"]).replace("/", os.sep)
        sample_abs = os.path.join(self.root_dir, sample_rel)

        if not os.path.exists(sample_abs):
            raise FileNotFoundError(f"Sample array file not found at {sample_abs}")

        # Lazy load numpy array
        with np.load(sample_abs) as data:
            frames_uint8 = data["frames"]  # Shape: (50, 240, 320, 3), uint8

        # 1. Convert uint8 (T, H, W, C) -> FloatTensor (T, C, H, W) scaled to [0.0, 1.0]
        tensor_frames = torch.from_numpy(frames_uint8).permute(0, 3, 1, 2).float() / 255.0

        # 2. Optional Training Augmentations (Horizontal Flip across all 50 frames consistently)
        if self.apply_augmentations:
            if torch.rand(1).item() > 0.5:
                tensor_frames = torch.flip(tensor_frames, dims=[3])  # Flip width dimension

        # 3. Standard ImageNet Normalization: (x - mean) / std per frame
        normalized_frames = (tensor_frames - self.IMAGENET_MEAN) / self.IMAGENET_STD

        # Label encoding: NORMAL -> 0, FALL -> 1
        label_int = self.LABEL_MAP[row["label"]]
        label_tensor = torch.tensor(label_int, dtype=torch.long)

        return {
            "frames": normalized_frames,  # Shape: (50, 3, 240, 320), float32
            "label": label_tensor,        # scalar (0 or 1)
            "event_id": row["event_id"],
            "video_id": row["video_id"],
            "window_id": row["window_id"],
            "camera_id": row["camera_id"],
        }


class URFDRGBFeatureDataset(Dataset):
    """
    Lazy-loading PyTorch Dataset for URFD precomputed ResNet-18 (50, 512) feature matrices.
    """
    LABEL_MAP = {"NORMAL": 0, "FALL": 1}

    def __init__(self, partition="train", root_dir=None):
        super().__init__()
        self.partition = partition
        
        if root_dir is None:
            self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        else:
            self.root_dir = root_dir

        manifest_path = os.path.join(
            self.root_dir, "processed_data", "URFD_RGB_baseline", "processed_features_manifest.csv"
        )
        
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Processed feature manifest not found at {manifest_path}")

        df_all = pd.read_csv(manifest_path)
        self.df = df_all[df_all["partition"] == partition].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_rel = str(row["processed_feature_path"]).replace("/", os.sep)
        sample_abs = os.path.join(self.root_dir, sample_rel)

        if not os.path.exists(sample_abs):
            raise FileNotFoundError(f"Feature array file not found at {sample_abs}")

        with np.load(sample_abs) as data:
            feats_np = data["features"]  # Shape: (50, 512), float32

        feats_tensor = torch.from_numpy(feats_np).float()

        label_int = self.LABEL_MAP[row["label"]]
        label_tensor = torch.tensor(label_int, dtype=torch.long)

        return {
            "features": feats_tensor,    # Shape: (50, 512), float32
            "label": label_tensor,       # scalar (0 or 1)
            "event_id": row["event_id"],
            "video_id": row["video_id"],
            "window_id": row["window_id"],
            "camera_id": row["camera_id"],
        }


if __name__ == "__main__":
    # Test dataset loader instantiation across partitions
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    print("=== TESTING URFD RGB DATASET LOADER ===")
    for p in ["train", "val", "test"]:
        ds = URFDRGBDataset(partition=p, root_dir=root)
        print(f"\nPartition '{p}': {len(ds)} samples")
        if len(ds) > 0:
            sample = ds[0]
            print(f"  Sample 0 Window ID: {sample['window_id']}")
            print(f"  Event ID: {sample['event_id']}")
            print(f"  Camera ID: {sample['camera_id']}")
            print(f"  Frames Shape: {sample['frames'].shape}, Dtype: {sample['frames'].dtype}")
            print(f"  Label Tensor: {sample['label']} ({'FALL' if sample['label'] == 1 else 'NORMAL'})")
            print(f"  Tensor Min: {sample['frames'].min():.3f}, Max: {sample['frames'].max():.3f}")
