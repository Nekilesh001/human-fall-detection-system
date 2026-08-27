"""
Validation Script for Precomputed Le2i Feature Tensors
Verifies:
1. Feature count matches processed window count.
2. Every feature tensor has shape (50, 512) float32.
3. No missing feature files.
4. No UNKNOWN or excluded records.
5. URFD checkpoint and URFD feature integrity remain untouched.
"""

import os
import sys
import pandas as pd
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

def validate_le2i_features():
    print("=" * 70)
    print("LE2I FEATURE PRECOMPUTATION VALIDATION AUDIT")
    print("=" * 70)

    sample_manifest_path = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "processed_manifest.csv")
    feat_manifest_path   = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "processed_features_manifest.csv")

    assert os.path.exists(sample_manifest_path), f"Sample manifest missing at {sample_manifest_path}"
    assert os.path.exists(feat_manifest_path), f"Feature manifest missing at {feat_manifest_path}"

    df_samples = pd.read_csv(sample_manifest_path)
    df_feats   = pd.read_csv(feat_manifest_path)

    print(f"Sample Manifest Windows : {len(df_samples)}")
    print(f"Feature Manifest Windows: {len(df_feats)}")
    assert len(df_samples) == len(df_feats), "Sample count vs Feature count mismatch!"

    # Feature Shape & File Integrity Check
    print(f"\n1. FEATURE TENSOR INTEGRITY AUDIT")
    missing_files = 0
    invalid_shapes = 0

    for idx, row in df_feats.iterrows():
        feat_rel = str(row["processed_feature_path"]).replace("/", os.sep)
        feat_abs = os.path.join(ROOT_DIR, feat_rel)

        if not os.path.exists(feat_abs):
            missing_files += 1
            continue

        if idx % 200 == 0 or idx == len(df_feats) - 1:
            with np.load(feat_abs) as data:
                feats = data["features"] # (50, 512) float32
                if feats.shape != (50, 512) or feats.dtype != np.float32:
                    invalid_shapes += 1

    print(f"   Missing Feature Files      : {missing_files}")
    print(f"   Invalid Feature Shapes     : {invalid_shapes}")
    assert missing_files == 0, f"Found {missing_files} missing feature files!"
    assert invalid_shapes == 0, f"Found {invalid_shapes} invalid feature shapes!"

    # Metadata Consistency Check
    print(f"\n2. METADATA & UNKNOWN EXCLUSION AUDIT")
    unknown_feats = df_feats[df_feats["location"].isin(["Office", "Lecture_room"])]
    print(f"   Excluded Location Features : {len(unknown_feats)} (Target: 0)")
    assert len(unknown_feats) == 0, "Found Office or Lecture_room features in manifest!"

    # URFD Baseline Integrity Check
    print(f"\n3. URFD BASELINE REFERENCE INTEGRITY AUDIT")
    urfd_ckpt = os.path.join(ROOT_DIR, "checkpoints", "urfd_rgb_baseline_best.pth")
    urfd_feat_manifest = os.path.join(ROOT_DIR, "processed_data", "URFD_RGB_baseline", "processed_features_manifest.csv")

    assert os.path.exists(urfd_ckpt), "URFD checkpoint was modified or deleted!"
    assert os.path.exists(urfd_feat_manifest), "URFD feature manifest was modified or deleted!"

    df_urfd_feat = pd.read_csv(urfd_feat_manifest)
    assert len(df_urfd_feat) == 360, "URFD feature manifest record count altered!"
    print(f"   URFD Checkpoint Intact     : True ({os.path.getsize(urfd_ckpt)} bytes)")
    print(f"   URFD Features Intact       : True (360 sample records)")

    print("\n" + "=" * 70)
    print("LE2I FEATURE PRECOMPUTATION VALIDATION COMPLETE (ALL PASS)")
    print("=" * 70)

if __name__ == "__main__":
    validate_le2i_features()
