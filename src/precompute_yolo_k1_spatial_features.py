"""
Experiment K Phase K1: Precomputes 187-D Spatial-Augmented Feature Tensors for K1.

Inputs:
- Source Directory: processed_data/Le2i_baseline/pose_estimator_features/yolo_pose/
- Input Tensor Shape: (50, 165) float32

Derived Features (22 spatial/body-configuration metrics):
1. 12 Joint Flexion & Orientation Angles (Knees, Hips, Elbows, Shoulders, Spine Inclination, Neck, Leg Verticals)
2. 2 Bounding Box / Aspect Ratio Features (Width/Height Aspect Ratio, BBox Area Ratio)
3. 4 Normalized Joint Heights relative to Torso Length (Head, L/R Wrists, Ankles)
4. 4 Torso Deformation & Lateral Tilt Metrics (Torso Scale Ratio, Shoulder Tilt, Hip Tilt, Torso Aspect Ratio)

Outputs:
- Target Directory: processed_data/Le2i_baseline/pose_estimator_features/yolo_pose_k1/
- Target Tensor Shape: (50, 187) float32
- Summary JSON: R&D/ML_Baseline/results/yolo_k1_precomputation_summary.json
"""

import os
import sys
import glob
import time
import json
import numpy as np
import pandas as pd

ROOT_DIR = r"d:\ONE_DATA\Fall detection"
sys.path.insert(0, ROOT_DIR)

def derive_22_spatial_features(feat_165):
    """
    Derives 22 spatial body-geometry features from (T, 165) YOLO Pose tensor.
    """
    T = feat_165.shape[0]
    out_22 = np.zeros((T, 22), dtype=np.float32)
    
    def get_kp(feat, idx):
        return feat[:, idx * 3 : idx * 3 + 2] # (T, 2)
        
    kp_nose       = get_kp(feat_165, 0)
    kp_l_shoulder = get_kp(feat_165, 11)
    kp_r_shoulder = get_kp(feat_165, 12)
    kp_l_elbow    = get_kp(feat_165, 13)
    kp_r_elbow    = get_kp(feat_165, 14)
    kp_l_wrist    = get_kp(feat_165, 15)
    kp_r_wrist    = get_kp(feat_165, 16)
    kp_l_hip      = get_kp(feat_165, 23)
    kp_r_hip      = get_kp(feat_165, 24)
    kp_l_knee     = get_kp(feat_165, 25)
    kp_r_knee     = get_kp(feat_165, 26)
    kp_l_ankle    = get_kp(feat_165, 27)
    kp_r_ankle    = get_kp(feat_165, 28)
    
    mid_shoulder  = (kp_l_shoulder + kp_r_shoulder) / 2.0
    mid_hip       = (kp_l_hip + kp_r_hip) / 2.0
    torso_vec     = mid_shoulder - mid_hip
    torso_len     = np.linalg.norm(torso_vec, axis=1) + 1e-6 # (T,)
    
    def angle_3p(A, B, C):
        BA = A - B
        BC = C - B
        dot = np.sum(BA * BC, axis=1)
        norm_ba = np.linalg.norm(BA, axis=1) + 1e-6
        norm_bc = np.linalg.norm(BC, axis=1) + 1e-6
        cos_a = np.clip(dot / (norm_ba * norm_bc), -1.0, 1.0)
        return np.arccos(cos_a)
        
    # 1. 12 Joint Flexion & Orientation Angles
    out_22[:, 0]  = angle_3p(kp_l_hip, kp_l_knee, kp_l_ankle)       # L Knee
    out_22[:, 1]  = angle_3p(kp_r_hip, kp_r_knee, kp_r_ankle)       # R Knee
    out_22[:, 2]  = angle_3p(kp_l_shoulder, kp_l_hip, kp_l_knee)   # L Hip
    out_22[:, 3]  = angle_3p(kp_r_shoulder, kp_r_hip, kp_r_knee)   # R Hip
    out_22[:, 4]  = angle_3p(kp_l_shoulder, kp_l_elbow, kp_l_wrist) # L Elbow
    out_22[:, 5]  = angle_3p(kp_r_shoulder, kp_r_elbow, kp_r_wrist) # R Elbow
    out_22[:, 6]  = angle_3p(kp_l_hip, kp_l_shoulder, kp_l_elbow)   # L Shoulder
    out_22[:, 7]  = angle_3p(kp_r_hip, kp_r_shoulder, kp_r_elbow)   # R Shoulder
    
    # Spine inclination angle relative to upward vertical (0, -1)
    cos_spine = np.clip(-torso_vec[:, 1] / torso_len, -1.0, 1.0)
    out_22[:, 8]  = np.arccos(cos_spine)                            # Spine inclination
    out_22[:, 9]  = angle_3p(kp_nose, mid_shoulder, mid_hip)        # Neck angle
    
    # Leg vertical angles
    l_leg_vec = kp_l_ankle - kp_l_hip
    r_leg_vec = kp_r_ankle - kp_r_hip
    out_22[:, 10] = np.arccos(np.clip(l_leg_vec[:, 1] / (np.linalg.norm(l_leg_vec, axis=1) + 1e-6), -1.0, 1.0))
    out_22[:, 11] = np.arccos(np.clip(r_leg_vec[:, 1] / (np.linalg.norm(r_leg_vec, axis=1) + 1e-6), -1.0, 1.0))
    
    # 2. 2 Bounding Box / Aspect Ratio Features
    for t in range(T):
        xs = feat_165[t, 0:99:3]
        ys = feat_165[t, 1:99:3]
        valid = (xs > 0) | (ys > 0)
        if np.any(valid):
            w_bb = np.max(xs[valid]) - np.min(xs[valid])
            h_bb = np.max(ys[valid]) - np.min(ys[valid])
        else:
            w_bb, h_bb = 0.0, 0.0
        out_22[t, 12] = w_bb / (h_bb + 1e-6)
        out_22[t, 13] = w_bb * h_bb
        
    # 3. 4 Normalized Joint Heights (relative to torso length)
    out_22[:, 14] = (kp_nose[:, 1] - mid_hip[:, 1]) / torso_len
    out_22[:, 15] = (kp_l_wrist[:, 1] - mid_hip[:, 1]) / torso_len
    out_22[:, 16] = (kp_r_wrist[:, 1] - mid_hip[:, 1]) / torso_len
    out_22[:, 17] = ((kp_l_ankle[:, 1] + kp_r_ankle[:, 1]) / 2.0 - mid_hip[:, 1]) / torso_len
    
    # 4. 4 Torso Deformation / Lateral Tilt
    mean_t_len = np.mean(torso_len) + 1e-6
    out_22[:, 18] = torso_len / mean_t_len
    
    sh_diff = kp_r_shoulder - kp_l_shoulder
    hip_diff = kp_r_hip - kp_l_hip
    out_22[:, 19] = np.arctan2(sh_diff[:, 1], sh_diff[:, 0])
    out_22[:, 20] = np.arctan2(hip_diff[:, 1], hip_diff[:, 0])
    
    sh_width = np.linalg.norm(sh_diff, axis=1)
    out_22[:, 21] = sh_width / torso_len
    
    return out_22

def precompute_yolo_k1_spatial_features():
    print("=" * 70)
    print("EXPERIMENT K PHASE K1: PRECOMPUTING 187-D SPATIAL FEATURE TENSORS")
    print("=" * 70)

    src_dir = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "pose_estimator_features", "yolo_pose")
    target_dir = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "pose_estimator_features", "yolo_pose_k1")
    os.makedirs(target_dir, exist_ok=True)

    manifest_path = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "processed_pose_features_manifest.csv")
    df_manifest = pd.read_csv(manifest_path).sort_values("window_id").reset_index(drop=True)

    start_time = time.time()
    extracted_count = 0
    total_bytes = 0

    for idx, row in df_manifest.iterrows():
        wid = row["window_id"]
        src_npz = os.path.join(src_dir, f"{wid}.npz")
        target_npz = os.path.join(target_dir, f"{wid}.npz")

        assert os.path.exists(src_npz), f"Source NPZ missing: {src_npz}"

        with np.load(src_npz) as d:
            feat_165 = d["features"].astype(np.float32)

        assert feat_165.shape == (50, 165), f"Expected shape (50, 165), got {feat_165.shape}"

        feat_22 = derive_22_spatial_features(feat_165)
        feat_187 = np.hstack([feat_165, feat_22]).astype(np.float32)

        np.savez_compressed(target_npz, features=feat_187)
        
        sz = os.path.getsize(target_npz)
        total_bytes += sz
        extracted_count += 1

        if (extracted_count % 300 == 0) or (extracted_count == len(df_manifest)):
            print(f"   Processed {extracted_count}/{len(df_manifest)} windows...")

    elapsed = time.time() - start_time
    print(f"\nExtracted {extracted_count} K1 feature tensors in {elapsed:.2f} seconds.")
    print(f"Total Storage Size: {total_bytes / (1024 * 1024):.2f} MB.")

    summary = {
        "experiment": "Experiment K Phase K1 (Spatial Feature Precomputation)",
        "total_windows": extracted_count,
        "feature_dim": 187,
        "base_dim": 165,
        "derived_dim": 22,
        "tensor_shape": [50, 187],
        "dtype": "float32",
        "total_storage_mb": round(total_bytes / (1024 * 1024), 2),
        "precomputation_time_sec": round(elapsed, 2),
        "target_directory": target_dir
    }

    res_dir = os.path.join(ROOT_DIR, "R&D", "ML_Baseline", "results")
    os.makedirs(res_dir, exist_ok=True)
    summary_path = os.path.join(res_dir, "yolo_k1_precomputation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary JSON to {summary_path}")

if __name__ == "__main__":
    precompute_yolo_k1_spatial_features()
