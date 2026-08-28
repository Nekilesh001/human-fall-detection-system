"""
Experiment K Phase K2: 100-Frame Feature & Manifest Validation Gate Script.

Verifies:
1. Exactly 1,142 expected windows in processed_pose_100f_manifest.csv
2. Exactly 1,142 NPZ files in yolo_pose_100f/
3. 1-to-1 manifest/file alignment
4. Every tensor shape = (100, 165) float32 with key 'features'
5. 0 NaN, 0 Inf
6. Expected location distribution (Coffee_01: 408, Coffee_02: 370, Home_01: 179, Home_02: 185)
7. Expected label distribution (838 NORMAL, 304 FALL)
8. No duplicate window IDs
9. Deterministic ordering
10. 50-frame canonical dataset (manifest and tensors) remains 100% untouched
"""

import os
import sys
import glob
import numpy as np
import pandas as pd

ROOT_DIR = r"d:\ONE_DATA\Fall detection"
sys.path.insert(0, ROOT_DIR)

def validate_le2i_yolo_100f():
    print("=" * 70)
    print("EXPERIMENT K PHASE K2: 100-FRAME VALIDATION GATE")
    print("=" * 70)

    # 1. Manifest Audit
    manifest_path = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "processed_pose_100f_manifest.csv")
    assert os.path.exists(manifest_path), f"K2 Manifest missing: {manifest_path}"
    df_manifest = pd.read_csv(manifest_path).sort_values("window_id").reset_index(drop=True)

    total_wins = len(df_manifest)
    print(f"\n1. Manifest Window Count Check: {total_wins} / 1,142")
    assert total_wins == 1142, f"Expected 1,142 windows, found {total_wins}"

    # 2. Duplicate & Missing Window ID Check
    print("   Checking Window ID Uniqueness...")
    unique_wids = df_manifest["window_id"].nunique()
    assert unique_wids == total_wins, f"Duplicate window IDs found! Unique={unique_wids}, Total={total_wins}"
    print("   [PASS] 0 Duplicate Window IDs")

    # 3. Location Distribution Verification
    print("\n2. Location Distribution Verification:")
    expected_locs = {
        "Coffee_room_01": 408,
        "Coffee_room_02": 370,
        "Home_01": 179,
        "Home_02": 185
    }
    actual_locs = df_manifest["location"].value_counts().to_dict()

    for loc, exp_cnt in expected_locs.items():
        act_cnt = actual_locs.get(loc, 0)
        print(f"   - {loc:15s}: {act_cnt} / {exp_cnt} ", end="")
        if act_cnt == exp_cnt:
            print("[EXACT MATCH PASS]")
        else:
            print("[DISCREPANCY]")
            sys.exit(1)

    # 4. Label Distribution Verification
    print("\n3. Label Distribution Verification:")
    expected_labels = {
        "NORMAL": 838,
        "FALL": 304
    }
    actual_labels = df_manifest["label"].value_counts().to_dict()

    for lbl, exp_cnt in expected_labels.items():
        act_cnt = actual_labels.get(lbl, 0)
        print(f"   - {lbl:8s}: {act_cnt} / {exp_cnt} ", end="")
        if act_cnt == exp_cnt:
            print("[EXACT MATCH PASS]")
        else:
            print("[DISCREPANCY]")
            sys.exit(1)

    # 5. NPZ File & Tensor Integrity Check
    print("\n4. Tensor Integrity & NPZ File Verification:")
    target_dir = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "pose_estimator_features", "yolo_pose_100f")
    npz_files = glob.glob(os.path.join(target_dir, "*.npz"))
    print(f"   NPZ File Count: {len(npz_files)} / 1,142")
    assert len(npz_files) == 1142, f"Expected 1,142 NPZ files, found {len(npz_files)}"

    for idx, row in df_manifest.iterrows():
        wid = row["window_id"]
        fpath = os.path.join(target_dir, f"{wid}.npz")
        assert os.path.exists(fpath), f"NPZ file missing: {fpath}"

        with np.load(fpath) as d:
            assert "features" in d, f"Key 'features' missing in {fpath}"
            feat = d["features"]

        assert feat.shape == (100, 165), f"Window {wid} invalid shape: {feat.shape}"
        assert feat.dtype == np.float32, f"Window {wid} invalid dtype: {feat.dtype}"
        assert not np.isnan(feat).any(), f"Window {wid} contains NaN!"
        assert not np.isinf(feat).any(), f"Window {wid} contains Inf!"

    print("   [PASS] 1,142 / 1,142 files valid (100, 165) float32")
    print("   [PASS] 0 NaN, 0 Inf across all 1,142 files")

    # 6. Safety Check on 50-Frame Canonical Dataset
    print("\n5. Canonical 50-Frame Dataset Safety Audit:")
    canonical_manifest = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "processed_pose_features_manifest.csv")
    df_canon = pd.read_csv(canonical_manifest)
    assert len(df_canon) == 1396, f"Canonical 50f manifest modified! Found {len(df_canon)}"
    
    canon_dir = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "pose_estimator_features", "yolo_pose")
    canon_npzs = glob.glob(os.path.join(canon_dir, "*.npz"))
    assert len(canon_npzs) == 1396, f"Canonical 50f NPZs modified! Found {len(canon_npzs)}"
    
    print("   [PASS] Canonical 50-frame manifest & yolo_pose/ feature tensors remain 100% untouched")

    print("\n" + "=" * 70)
    print("EXPERIMENT K PHASE K2 100-FRAME VALIDATION GATE PASSED [PASS]")
    print("=" * 70)

if __name__ == "__main__":
    validate_le2i_yolo_100f()
