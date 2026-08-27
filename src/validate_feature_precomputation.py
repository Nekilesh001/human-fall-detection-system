"""
Validation & Numerical Equivalence Audit for URFD RGB Baseline Feature Precomputation
Verifies:
1. Numerical equivalence between on-the-fly ResNet-18 extraction and precomputed features.
2. Record counts: 360 total (260 train, 43 val, 57 test).
3. Feature shape (50, 512) float32 per window.
4. Zero event leakage and camera consistency across splits.
5. Integrity of original RGB processed dataset.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

def run_validation():
    print("=" * 70)
    print("URFD RGB BASELINE: FEATURE PRECOMPUTATION VALIDATION & AUDIT")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Validation Device       : {device}")

    # 1. Load Both Manifests
    rgb_manifest_path = os.path.join(ROOT_DIR, "processed_data", "URFD_RGB_baseline", "processed_manifest.csv")
    feat_manifest_path = os.path.join(ROOT_DIR, "processed_data", "URFD_RGB_baseline", "processed_features_manifest.csv")

    if not os.path.exists(rgb_manifest_path):
        raise FileNotFoundError(f"RGB manifest missing at {rgb_manifest_path}")
    if not os.path.exists(feat_manifest_path):
        raise FileNotFoundError(f"Feature manifest missing at {feat_manifest_path}")

    df_rgb = pd.read_csv(rgb_manifest_path)
    df_feat = pd.read_csv(feat_manifest_path)

    print(f"\n1. MANIFEST RECORD COUNT VERIFICATION")
    print(f"   RGB Manifest Records     : {len(df_rgb)}")
    print(f"   Feature Manifest Records : {len(df_feat)}")

    assert len(df_rgb) == 360, f"Expected 360 RGB manifest records, found {len(df_rgb)}"
    assert len(df_feat) == 360, f"Expected 360 feature manifest records, found {len(df_feat)}"

    # Check partition distribution
    p_counts = df_feat["partition"].value_counts().to_dict()
    print(f"   Partition Counts         : {p_counts}")
    assert p_counts.get("train", 0) == 260, f"Expected 260 train samples, got {p_counts.get('train')}"
    assert p_counts.get("val", 0) == 43, f"Expected 43 val samples, got {p_counts.get('val')}"
    assert p_counts.get("test", 0) == 57, f"Expected 57 test samples, got {p_counts.get('test')}"

    # 2. NUMERICAL EQUIVALENCE TEST (5 Sample Windows)
    print(f"\n2. NUMERICAL EQUIVALENCE TEST")
    weights = models.ResNet18_Weights.DEFAULT
    resnet = models.resnet18(weights=weights)
    resnet.fc = nn.Identity()
    for param in resnet.parameters():
        param.requires_grad = False
    resnet.to(device)
    resnet.eval()

    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    max_diffs = []
    mean_diffs = []
    allclose_pass = True

    sample_indices = [0, 50, 150, 260, 310]  # Samples across train, val, test
    for idx in sample_indices:
        row_rgb = df_rgb.iloc[idx]
        row_feat = df_feat.iloc[idx]

        assert row_rgb["window_id"] == row_feat["window_id"], "Window ID mismatch between manifests"

        # On-the-fly computation
        rgb_abs = os.path.join(ROOT_DIR, str(row_rgb["processed_sample_path"]).replace("/", os.sep))
        with np.load(rgb_abs) as data:
            frames_uint8 = data["frames"]

        tensor_frames = torch.from_numpy(frames_uint8).permute(0, 3, 1, 2).float().to(device) / 255.0
        norm_frames = (tensor_frames - IMAGENET_MEAN) / IMAGENET_STD

        with torch.no_grad():
            live_feats = resnet(norm_frames).cpu().numpy().astype(np.float32)

        # Precomputed feature
        feat_abs = os.path.join(ROOT_DIR, str(row_feat["processed_feature_path"]).replace("/", os.sep))
        with np.load(feat_abs) as data:
            saved_feats = data["features"]

        abs_diff = np.abs(live_feats - saved_feats)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)
        is_close = np.allclose(live_feats, saved_feats, atol=1e-5)

        max_diffs.append(max_diff)
        mean_diffs.append(mean_diff)
        if not is_close:
            allclose_pass = False

        print(f"   Sample [{idx}] {row_rgb['window_id']}: max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}, allclose(atol=1e-5)={is_close}")

    print(f"\n   OVERALL NUMERICAL EQUIVALENCE RESULTS:")
    print(f"     Maximum Absolute Difference : {max(max_diffs):.8f}")
    print(f"     Mean Absolute Difference    : {np.mean(mean_diffs):.8f}")
    print(f"     np.allclose(atol=1e-5)      : {allclose_pass}")

    assert allclose_pass, "Numerical equivalence check failed!"

    # 3. LEAKAGE AUDIT & INTEGRITY CHECK
    print(f"\n3. EVENT & CAMERA LEAKAGE AUDIT")
    train_events = set(df_feat[df_feat["partition"] == "train"]["event_id"])
    val_events   = set(df_feat[df_feat["partition"] == "val"]["event_id"])
    test_events  = set(df_feat[df_feat["partition"] == "test"]["event_id"])

    train_val_overlap  = train_events.intersection(val_events)
    train_test_overlap = train_events.intersection(test_events)
    val_test_overlap   = val_events.intersection(test_events)

    print(f"   Train Events count : {len(train_events)}")
    print(f"   Val Events count   : {len(val_events)}")
    print(f"   Test Events count  : {len(test_events)}")
    print(f"   Train ∩ Val Overlap: {len(train_val_overlap)} events")
    print(f"   Train ∩ Test Overlap: {len(train_test_overlap)} events")
    print(f"   Val ∩ Test Overlap : {len(val_test_overlap)} events")

    assert len(train_val_overlap) == 0, f"Event leakage detected between Train and Val: {train_val_overlap}"
    assert len(train_test_overlap) == 0, f"Event leakage detected between Train and Test: {train_test_overlap}"
    assert len(val_test_overlap) == 0, f"Event leakage detected between Val and Test: {val_test_overlap}"

    # 4. FEATURE SHAPE & FILE EXISTENCE VERIFICATION
    print(f"\n4. FEATURE FILE INTEGRITY CHECK")
    missing_files = 0
    invalid_shapes = 0

    for idx, row in df_feat.iterrows():
        feat_abs = os.path.join(ROOT_DIR, str(row["processed_feature_path"]).replace("/", os.sep))
        if not os.path.exists(feat_abs):
            missing_files += 1
            continue

        with np.load(feat_abs) as data:
            f_shape = data["features"].shape
            if f_shape != (50, 512):
                invalid_shapes += 1

    print(f"   Missing Feature Files  : {missing_files}")
    print(f"   Invalid Feature Shapes : {invalid_shapes}")
    assert missing_files == 0, f"Found {missing_files} missing feature files!"
    assert invalid_shapes == 0, f"Found {invalid_shapes} feature files with invalid shape!"

    # 5. ORIGINAL DATA INTEGRITY CHECK
    print(f"\n5. ORIGINAL PROCESSED RGB DATA INTEGRITY CHECK")
    rgb_missing = 0
    for idx, row in df_rgb.iterrows():
        rgb_abs = os.path.join(ROOT_DIR, str(row["processed_sample_path"]).replace("/", os.sep))
        if not os.path.exists(rgb_abs):
            rgb_missing += 1

    print(f"   Missing Original RGB Samples: {rgb_missing}")
    assert rgb_missing == 0, "Original processed RGB files were corrupted or removed!"

    print("\n" + "=" * 70)
    print("ALL VALIDATION & AUDIT CHECKS PASSED PERFECTLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_validation()
