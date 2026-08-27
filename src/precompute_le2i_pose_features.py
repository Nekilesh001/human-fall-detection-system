"""
Precompute Le2i MediaPipe Pose Keypoint Features for Experiment E (E1, E2, E3).

Inputs:
- Raw Le2i videos specified in processed_features_manifest.csv (1,396 windows)
- models/pose_landmarker_full.task

Outputs:
- processed_data/Le2i_baseline/pose_features/e1/{window_id}_e1.npz  (50, 99)
- processed_data/Le2i_baseline/pose_features/e2/{window_id}_e2.npz  (50, 165)
- processed_data/Le2i_baseline/pose_features/e3/{window_id}_e3.npz  (50, 173)
- processed_data/Le2i_baseline/processed_pose_features_manifest.csv
- R&D/ML_Baseline/results/pose_precomputation_summary.json
"""

import os
import sys
import time
import json
import cv2
import pandas as pd
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

ROOT_DIR = r"d:\ONE_DATA\Fall detection"

def compute_pose_features_for_window(frames, landmarker):
    """
    Computes E1 (50, 99), E2 (50, 165), and E3 (50, 173) feature matrices for a 50-frame window.
    """
    raw_landmarks_50 = [] # 50 frames x 33 landmarks x (x, y, v)
    detection_mask_50 = []

    for img in frames:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        res = landmarker.detect(mp_image)

        if res.pose_landmarks and len(res.pose_landmarks) > 0:
            detection_mask_50.append(True)
            lm_list = res.pose_landmarks[0]
            pts = []
            for lm in lm_list:
                pts.append([lm.x, lm.y, lm.visibility])
            raw_landmarks_50.append(np.array(pts, dtype=np.float32))
        else:
            detection_mask_50.append(False)
            raw_landmarks_50.append(np.zeros((33, 3), dtype=np.float32))

    raw_landmarks_50 = np.array(raw_landmarks_50, dtype=np.float32) # (50, 33, 3)

    # 1. E1: Centering & Torso Length Normalization
    e1_frames = []
    e3_descriptors = []

    for t in range(50):
        if not detection_mask_50[t]:
            e1_frames.append(np.zeros((99,), dtype=np.float32))
            e3_descriptors.append(np.zeros((8,), dtype=np.float32))
            continue

        lm = raw_landmarks_50[t] # (33, 3)
        # Hip midpoint (landmarks 23, 24)
        x_hip = (lm[23, 0] + lm[24, 0]) / 2.0
        y_hip = (lm[23, 1] + lm[24, 1]) / 2.0

        # Shoulder midpoint (landmarks 11, 12)
        x_sh = (lm[11, 0] + lm[12, 0]) / 2.0
        y_sh = (lm[11, 1] + lm[12, 1]) / 2.0

        torso_len = np.sqrt((x_sh - x_hip)**2 + (y_sh - y_hip)**2) + 1e-6

        # Centering and scale normalization
        norm_lm = np.copy(lm)
        norm_lm[:, 0] = (lm[:, 0] - x_hip) / torso_len
        norm_lm[:, 1] = (lm[:, 1] - y_hip) / torso_len
        # visibility remains lm[:, 2]

        e1_vec = norm_lm.reshape(-1) # (99,)
        e1_frames.append(e1_vec)

        # Derived physics descriptors (8-D)
        valid_pts = lm[lm[:, 2] > 0.1]
        if len(valid_pts) > 0:
            min_x, max_x = np.min(valid_pts[:, 0]), np.max(valid_pts[:, 0])
            min_y, max_y = np.min(valid_pts[:, 1]), np.max(valid_pts[:, 1])
            aspect_ratio = (max_y - min_y) / (max_x - min_x + 1e-6)
        else:
            aspect_ratio = 0.0

        torso_angle = np.arctan2(y_sh - y_hip, x_sh - x_hip)

        # Velocities computed across sequence
        desc = np.array([
            x_hip, y_hip, x_sh, y_sh,
            torso_angle, aspect_ratio,
            0.0, 0.0 # Placeholder for vertical vel & trajectory distance
        ], dtype=np.float32)
        e3_descriptors.append(desc)

    e1_matrix = np.array(e1_frames, dtype=np.float32) # (50, 99)
    e3_desc_matrix = np.array(e3_descriptors, dtype=np.float32) # (50, 8)

    # Compute frame-to-frame joint velocities (66-D) for E2
    vel_frames = []
    for t in range(50):
        if t == 0 or not detection_mask_50[t] or not detection_mask_50[t-1]:
            vel_frames.append(np.zeros((66,), dtype=np.float32))
        else:
            dx = e1_matrix[t, 0::3] - e1_matrix[t-1, 0::3] # 33 dx
            dy = e1_matrix[t, 1::3] - e1_matrix[t-1, 1::3] # 33 dy
            d_xy = np.column_stack([dx, dy]).reshape(-1)
            vel_frames.append(d_xy)

    vel_matrix = np.array(vel_frames, dtype=np.float32) # (50, 66)

    # Update vertical velocity and trajectory distance in E3
    for t in range(50):
        if t > 0 and detection_mask_50[t] and detection_mask_50[t-1]:
            # Hip vertical velocity
            e3_desc_matrix[t, 6] = e3_desc_matrix[t, 1] - e3_desc_matrix[t-1, 1] # dy_hip
            # Trajectory distance
            e3_desc_matrix[t, 7] = np.sqrt((e3_desc_matrix[t, 0] - e3_desc_matrix[t-1, 0])**2 + (e3_desc_matrix[t, 1] - e3_desc_matrix[t-1, 1])**2)

    # Construct final matrices
    e2_matrix = np.hstack([e1_matrix, vel_matrix]) # (50, 165)
    e3_matrix = np.hstack([e2_matrix, e3_desc_matrix]) # (50, 173)

    return e1_matrix, e2_matrix, e3_matrix, detection_mask_50

def precompute_le2i_pose_features():
    print("=" * 70)
    print("PRECOMPUTING LE2I MEDIAPIPE POSE FEATURES (EXPERIMENT E)")
    print("=" * 70)

    model_path = os.path.join(ROOT_DIR, "models", "pose_landmarker_full.task")
    assert os.path.exists(model_path), f"Task model missing: {model_path}"

    base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    manifest_path = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "processed_flow_features_manifest.csv")
    df_manifest = pd.read_csv(manifest_path)

    out_e1_dir = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "pose_features", "e1")
    out_e2_dir = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "pose_features", "e2")
    out_e3_dir = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "pose_features", "e3")

    os.makedirs(out_e1_dir, exist_ok=True)
    os.makedirs(out_e2_dir, exist_ok=True)
    os.makedirs(out_e3_dir, exist_ok=True)

    grouped_videos = df_manifest.groupby("raw_video_path")
    print(f"Processing 1,396 windows across {len(grouped_videos)} unique videos...")

    start_time = time.perf_counter()
    new_rows = []
    
    loc_stats = {
        "Coffee_room_01": {"total_frames": 0, "detected_frames": 0, "total_wins": 0, "completely_undetected_wins": 0},
        "Coffee_room_02": {"total_frames": 0, "detected_frames": 0, "total_wins": 0, "completely_undetected_wins": 0},
        "Home_01":        {"total_frames": 0, "detected_frames": 0, "total_wins": 0, "completely_undetected_wins": 0},
        "Home_02":        {"total_frames": 0, "detected_frames": 0, "total_wins": 0, "completely_undetected_wins": 0}
    }

    processed_windows_count = 0

    for raw_video_path, grp in grouped_videos:
        video_abs = os.path.join(ROOT_DIR, str(raw_video_path).replace("/", os.sep))
        assert os.path.exists(video_abs), f"Video missing: {video_abs}"

        cap = cv2.VideoCapture(video_abs)
        frame_buffer = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(cv2.resize(frame, (320, 240)), cv2.COLOR_BGR2RGB)
            frame_buffer.append(rgb)
        cap.release()

        for _, win_row in grp.iterrows():
            win_id  = win_row["window_id"]
            location = win_row["location"]
            start_f = int(win_row["win_start_frame"]) - 1
            end_f   = int(win_row["win_end_frame"])

            win_frames = []
            for f_idx in range(start_f, end_f):
                idx = max(0, min(f_idx, len(frame_buffer) - 1))
                win_frames.append(frame_buffer[idx])

            e1_mat, e2_mat, e3_mat, mask_50 = compute_pose_features_for_window(win_frames, landmarker)

            e1_rel = f"processed_data/Le2i_baseline/pose_features/e1/{win_id}_e1.npz"
            e2_rel = f"processed_data/Le2i_baseline/pose_features/e2/{win_id}_e2.npz"
            e3_rel = f"processed_data/Le2i_baseline/pose_features/e3/{win_id}_e3.npz"

            np.savez_compressed(os.path.join(ROOT_DIR, e1_rel.replace("/", os.sep)), features=e1_mat)
            np.savez_compressed(os.path.join(ROOT_DIR, e2_rel.replace("/", os.sep)), features=e2_mat)
            np.savez_compressed(os.path.join(ROOT_DIR, e3_rel.replace("/", os.sep)), features=e3_mat)

            row_dict = win_row.to_dict()
            row_dict["e1_feature_path"] = e1_rel
            row_dict["e2_feature_path"] = e2_rel
            row_dict["e3_feature_path"] = e3_rel
            new_rows.append(row_dict)

            # Accumulate detection stats
            n_det = sum(mask_50)
            loc_stats[location]["total_frames"] += 50
            loc_stats[location]["detected_frames"] += n_det
            loc_stats[location]["total_wins"] += 1
            if n_det == 0:
                loc_stats[location]["completely_undetected_wins"] += 1

            processed_windows_count += 1
            if processed_windows_count % 200 == 0 or processed_windows_count == 1396:
                print(f"  Processed {processed_windows_count}/1,396 pose windows in {time.perf_counter() - start_time:.1f}s")

    landmarker.close()
    elapsed = time.perf_counter() - start_time

    # Save Manifest
    df_pose_manifest = pd.DataFrame(new_rows)
    pose_manifest_path = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "processed_pose_features_manifest.csv")
    df_pose_manifest.to_csv(pose_manifest_path, index=False)

    # Prepare Precomputation Summary JSON
    summary_data = {
        "total_windows": processed_windows_count,
        "elapsed_seconds": elapsed,
        "manifest_path": pose_manifest_path,
        "location_stats": loc_stats
    }

    res_dir = os.path.join(ROOT_DIR, "R&D", "ML_Baseline", "results")
    os.makedirs(res_dir, exist_ok=True)
    json_path = os.path.join(res_dir, "pose_precomputation_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print("\n" + "=" * 70)
    print("LE2I POSE FEATURE PRECOMPUTATION COMPLETE")
    print("=" * 70)
    print(f"Total Extraction Time : {elapsed:.2f} seconds (~{elapsed/60.0:.1f} minutes)")
    print(f"Pose Manifest Saved   : {pose_manifest_path}")

if __name__ == "__main__":
    precompute_le2i_pose_features()
