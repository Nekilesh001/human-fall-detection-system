"""
Feature Precomputation Script for Le2i Baseline (Frozen ResNet-18)
Extracts 512-dim spatial embeddings per frame from frozen ResNet-18 for all preprocessed Le2i windows.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

def precompute_le2i_features():
    print("=" * 70)
    print("LE2I BASELINE: FEATURE PRECOMPUTATION")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device        : {device}")

    weights = models.ResNet18_Weights.DEFAULT
    resnet = models.resnet18(weights=weights)
    resnet.fc = nn.Identity()

    for param in resnet.parameters():
        param.requires_grad = False

    resnet.to(device)
    resnet.eval()
    print("Backbone Model          : Frozen ImageNet ResNet-18 (fc=Identity, eval mode)")

    manifest_path = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "processed_manifest.csv")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Le2i processed manifest missing at {manifest_path}")

    df_manifest = pd.read_csv(manifest_path)
    total_samples = len(df_manifest)
    print(f"Manifest Loaded         : {total_samples} samples from {manifest_path}")

    features_dir = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "features")
    os.makedirs(features_dir, exist_ok=True)

    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    feature_manifest_rows = []
    start_time = time.perf_counter()

    print("\nExtracting Le2i features...")
    for idx, row in df_manifest.iterrows():
        sample_rel = str(row["processed_sample_path"]).replace("/", os.sep)
        sample_abs = os.path.join(ROOT_DIR, sample_rel)

        if not os.path.exists(sample_abs):
            raise FileNotFoundError(f"Le2i sample array missing at {sample_abs}")

        with np.load(sample_abs) as data:
            frames_uint8 = data["frames"] # (50, 240, 320, 3) uint8

        tensor_frames = torch.from_numpy(frames_uint8).permute(0, 3, 1, 2).float().to(device) / 255.0
        norm_frames = (tensor_frames - IMAGENET_MEAN) / IMAGENET_STD

        with torch.no_grad():
            feats = resnet(norm_frames) # Shape: (50, 512)

        feats_np = feats.cpu().numpy().astype(np.float32)

        window_id = row["window_id"]
        feature_filename = f"{window_id}_features.npz"
        feature_rel_path = os.path.join("processed_data", "Le2i_baseline", "features", feature_filename)
        feature_abs_path = os.path.join(ROOT_DIR, feature_rel_path)

        np.savez_compressed(
            feature_abs_path,
            features=feats_np,
            event_id=row["event_id"],
            video_id=row["video_id"],
            camera_id=row["camera_id"],
            location=row["location"],
            partition=row["partition"],
            label=row["label"],
            window_id=window_id,
            f_start=row["f_start"],
            f_end=row["f_end"],
            win_start_frame=row["win_start_frame"],
            win_end_frame=row["win_end_frame"]
        )

        feature_manifest_rows.append({
            "window_id": window_id,
            "event_id": row["event_id"],
            "video_id": row["video_id"],
            "camera_id": row["camera_id"],
            "location": row["location"],
            "partition": row["partition"],
            "label": row["label"],
            "processed_feature_path": feature_rel_path.replace(os.sep, "/"),
            "original_sample_path": row["processed_sample_path"],
            "f_start": row["f_start"],
            "f_end": row["f_end"],
            "win_start_frame": row["win_start_frame"],
            "win_end_frame": row["win_end_frame"]
        })

        if (idx + 1) % 500 == 0 or (idx + 1) == total_samples:
            elapsed = time.perf_counter() - start_time
            print(f"  Processed {idx + 1}/{total_samples} Le2i feature windows ({elapsed:.1f}s)")

    end_time = time.perf_counter()
    total_time = end_time - start_time

    df_feat_manifest = pd.DataFrame(feature_manifest_rows)
    feat_manifest_path = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "processed_features_manifest.csv")
    df_feat_manifest.to_csv(feat_manifest_path, index=False)

    total_storage_bytes = sum(
        os.path.getsize(os.path.join(ROOT_DIR, str(r["processed_feature_path"]).replace("/", os.sep)))
        for r in feature_manifest_rows
    )

    print("\n" + "=" * 70)
    print("LE2I FEATURE PRECOMPUTATION BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Total Extraction Time   : {total_time:.2f} seconds")
    print(f"Total Feature Windows   : {total_samples}")
    print(f"Average Time per Window : {(total_time / total_samples) * 1000:.1f} ms")
    print(f"Total Output Storage    : {total_storage_bytes / (1024 * 1024):.2f} MB")
    print(f"Feature Manifest Saved  : {feat_manifest_path}")
    print("=" * 70)

    return df_feat_manifest

if __name__ == "__main__":
    precompute_le2i_features()
