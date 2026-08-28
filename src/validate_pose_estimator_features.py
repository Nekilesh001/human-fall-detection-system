"""
Validation Script for Experiment H1 Pose Estimator Feature Tensors.
Validates:
1. File Count: Exactly 1,396 feature files per estimator (4,188 total).
2. Tensor Shape: (50, 165) float32 per window.
3. Integrity: 0 missing files, 0 invalid shapes, 0 NaN/Inf errors.
4. Manifest Alignment: Deterministic 1-to-1 match with processed_pose_features_manifest.csv.
5. Location Breakdown: Pose detection rates per physical location.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

ROOT_DIR = r"d:\ONE_DATA\Fall detection"
sys.path.insert(0, ROOT_DIR)

def validate_pose_estimator_features():
    print("=" * 70)
    print("EXPERIMENT H1: POSE ESTIMATOR FEATURE TENSOR VALIDATION GATE")
    print("=" * 70)

    pose_manifest_path = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "processed_pose_features_manifest.csv")
    assert os.path.exists(pose_manifest_path), f"Manifest missing: {pose_manifest_path}"
    df_manifest = pd.read_csv(pose_manifest_path)
    df_manifest = df_manifest.sort_values("window_id").reset_index(drop=True)

    estimators = {
        "mediapipe": {"name": "H1: MediaPipe Pose", "dir": os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "pose_estimator_features", "mediapipe")},
        "yolo_pose": {"name": "H2: YOLO Pose",     "dir": os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "pose_estimator_features", "yolo_pose")},
        "rtmpose":   {"name": "H3: RTMPose",       "dir": os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "pose_estimator_features", "rtmpose")}
    }

    all_gate_pass = True

    for est_key, est_meta in estimators.items():
        print(f"\n[{est_meta['name']}] Validating Tensors in {os.path.relpath(est_meta['dir'], ROOT_DIR)}...")

        assert os.path.exists(est_meta["dir"]), f"Directory missing: {est_meta['dir']}"
        files = glob_files = [f for f in os.listdir(est_meta["dir"]) if f.endswith(".npz")]

        print(f"   - File Count Check         : {len(files):,}/1,396 files ", end="")
        if len(files) == 1396:
            print("[PASS ✅]")
        else:
            print("[FAIL ❌]")
            all_gate_pass = False

        invalid_shapes = 0
        nan_inf_count = 0
        loc_stats = {
            "Coffee_room_01": {"total_wins": 0, "total_frames": 0, "det_frames": 0, "zero_wins": 0},
            "Coffee_room_02": {"total_wins": 0, "total_frames": 0, "det_frames": 0, "zero_wins": 0},
            "Home_01":        {"total_wins": 0, "total_frames": 0, "det_frames": 0, "zero_wins": 0},
            "Home_02":        {"total_wins": 0, "total_frames": 0, "det_frames": 0, "zero_wins": 0}
        }

        for idx, row in df_manifest.iterrows():
            wid = row["window_id"]
            loc = row["location"]
            fpath = os.path.join(est_meta["dir"], f"{wid}.npz")

            if not os.path.exists(fpath):
                invalid_shapes += 1
                continue

            with np.load(fpath) as d:
                feat = d["features"] # (50, 165)

            if feat.shape != (50, 165) or feat.dtype != np.float32:
                invalid_shapes += 1
            if np.isnan(feat).any() or np.isinf(feat).any():
                nan_inf_count += 1

            vis_per_frame = feat[:, 2:99:3]
            det_f = int(np.sum((vis_per_frame > 0).any(axis=1)))

            loc_stats[loc]["total_wins"] += 1
            loc_stats[loc]["total_frames"] += 50
            loc_stats[loc]["det_frames"] += det_f
            if det_f == 0:
                loc_stats[loc]["zero_wins"] += 1

        print(f"   - Tensor Shape & Dtype     : {1396 - invalid_shapes}/1,396 valid (50, 165) float32 ", end="")
        if invalid_shapes == 0:
            print("[PASS ✅]")
        else:
            print(f"[FAIL ❌ ({invalid_shapes} invalid)]")
            all_gate_pass = False

        print(f"   - NaN / Inf Error Check    : {nan_inf_count} errors ", end="")
        if nan_inf_count == 0:
            print("[PASS ✅]")
        else:
            print("[FAIL ❌]")
            all_gate_pass = False

        print(f"\n   Location Detection Breakdown:")
        print(f"     {'Location':15s} | {'Windows':^8s} | {'Frames':^8s} | {'Detected':^8s} | {'Det Rate':^10s} | {'Undetected Wins':^16s} | {'Rating':^12s}")
        print("     " + "-" * 85)
        for loc, ldata in loc_stats.items():
            rate = (ldata["det_frames"] / ldata["total_frames"]) * 100.0 if ldata["total_frames"] > 0 else 0.0
            if rate >= 90.0:
                rating = "EXCELLENT ✅"
            elif rate >= 70.0:
                rating = "GOOD ✅"
            elif rate >= 40.0:
                rating = "FAIR ⚠️"
            else:
                rating = "LOW ⚠️"
            print(f"     {loc:15s} | {ldata['total_wins']:^8d} | {ldata['total_frames']:^8d} | {ldata['det_frames']:^8d} | {rate:9.1f}% | {ldata['zero_wins']:^16d} | {rating:^12s}")

    print("\n" + "=" * 70)
    if all_gate_pass:
        print("PHASE H1 VALIDATION GATE PASS — PROCEED TO PHASE H2/H3 TRAINING")
    else:
        print("CRITICAL WARNING: PHASE H1 VALIDATION FAILED")
    print("=" * 70)

if __name__ == "__main__":
    validate_pose_estimator_features()
