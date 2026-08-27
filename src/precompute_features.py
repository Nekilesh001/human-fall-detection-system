"""
Feature Precomputation Pipeline for URFD RGB Baseline
Extracts 512-dim spatial embeddings per frame from frozen ResNet-18
for all 360 temporal windows and saves lightweight feature representations.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

# Set root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

def precompute_features():
    print("=" * 70)
    print("URFD RGB BASELINE: FEATURE PRECOMPUTATION")
    print("=" * 70)

    # 1. Setup Device & Load Pretrained ResNet-18 Backbone
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device        : {device}")
    
    weights = models.ResNet18_Weights.DEFAULT
    resnet = models.resnet18(weights=weights)
    resnet.fc = nn.Identity()
    
    # Freeze backbone parameters
    for param in resnet.parameters():
        param.requires_grad = False
        
    resnet.to(device)
    resnet.eval() # Ensure BatchNorm & Dropout stay strictly in eval mode
    print(f"Backbone Model          : Frozen ImageNet ResNet-18 (fc=Identity, eval mode)")

    # 2. Load Processed RGB Manifest
    manifest_path = os.path.join(ROOT_DIR, "processed_data", "URFD_RGB_baseline", "processed_manifest.csv")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Processed RGB manifest not found at: {manifest_path}")

    df_manifest = pd.read_csv(manifest_path)
    total_samples = len(df_manifest)
    print(f"Manifest Loaded         : {total_samples} samples from {manifest_path}")

    # Output directory for features
    features_base_dir = os.path.join(ROOT_DIR, "processed_data", "URFD_RGB_baseline", "features")
    os.makedirs(features_base_dir, exist_ok=True)

    for partition in ["train", "val", "test"]:
        os.makedirs(os.path.join(features_base_dir, partition), exist_ok=True)

    # ImageNet Mean & Std for preprocessing
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    feature_manifest_rows = []

    start_time = time.perf_counter()
    total_frames_processed = 0

    print("\nExtracting features...")
    for idx, row in df_manifest.iterrows():
        sample_rel = str(row["processed_sample_path"]).replace("/", os.sep)
        sample_abs = os.path.join(ROOT_DIR, sample_rel)

        if not os.path.exists(sample_abs):
            raise FileNotFoundError(f"RGB sample not found at: {sample_abs}")

        # Load RGB array (50, 240, 320, 3) uint8
        with np.load(sample_abs) as data:
            frames_uint8 = data["frames"]

        T, H, W, C = frames_uint8.shape
        total_frames_processed += T

        # Preprocess tensor on device
        tensor_frames = torch.from_numpy(frames_uint8).permute(0, 3, 1, 2).float().to(device) / 255.0
        norm_frames = (tensor_frames - IMAGENET_MEAN) / IMAGENET_STD

        # Forward pass through frozen ResNet-18
        with torch.no_grad():
            feats = resnet(norm_frames) # Shape: (50, 512)

        feats_np = feats.cpu().numpy().astype(np.float32)

        # Save feature npz file
        partition = row["partition"]
        window_id = row["window_id"]
        feature_filename = f"{window_id}_features.npz"
        feature_rel_path = os.path.join("processed_data", "URFD_RGB_baseline", "features", partition, feature_filename)
        feature_abs_path = os.path.join(ROOT_DIR, feature_rel_path)

        np.savez_compressed(
            feature_abs_path,
            features=feats_np, # Shape: (50, 512) float32
            event_id=row["event_id"],
            video_id=row["video_id"],
            camera_id=row["camera_id"],
            partition=partition,
            label=row["label"],
            window_id=window_id,
            original_sample_path=row["processed_sample_path"]
        )

        feature_manifest_rows.append({
            "window_id": window_id,
            "event_id": row["event_id"],
            "video_id": row["video_id"],
            "camera_id": row["camera_id"],
            "partition": partition,
            "label": row["label"],
            "processed_feature_path": feature_rel_path.replace(os.sep, "/"),
            "original_sample_path": row["processed_sample_path"],
            "frame_count": T,
            "feature_dim": feats_np.shape[1]
        })

        if (idx + 1) % 50 == 0 or (idx + 1) == total_samples:
            elapsed = time.perf_counter() - start_time
            print(f"  Processed {idx + 1}/{total_samples} windows ({elapsed:.1f}s)")

    end_time = time.perf_counter()
    total_extraction_time = end_time - start_time

    # Save feature manifest
    df_feature_manifest = pd.DataFrame(feature_manifest_rows)
    feature_manifest_path = os.path.join(ROOT_DIR, "processed_data", "URFD_RGB_baseline", "processed_features_manifest.csv")
    df_feature_manifest.to_csv(feature_manifest_path, index=False)
    print(f"\nFeature Manifest Saved : {feature_manifest_path}")

    # Calculate storage stats
    total_storage_bytes = sum(
        os.path.getsize(os.path.join(ROOT_DIR, str(r["processed_feature_path"]).replace("/", os.sep)))
        for r in feature_manifest_rows
    )
    avg_file_size_kb = (total_storage_bytes / total_samples) / 1024

    print("\n" + "=" * 70)
    print("PRECOMPUTATION BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Total Extraction Time   : {total_extraction_time:.2f} seconds")
    print(f"Total Windows Processed : {total_samples}")
    print(f"Total Frames Processed  : {total_frames_processed}")
    print(f"Average Time per Window : {(total_extraction_time / total_samples) * 1000:.1f} ms")
    print(f"Total Output Storage    : {total_storage_bytes / (1024 * 1024):.2f} MB")
    print(f"Average Feature Size    : {avg_file_size_kb:.2f} KB / file")
    print("=" * 70)

    return df_feature_manifest

if __name__ == "__main__":
    precompute_features()
