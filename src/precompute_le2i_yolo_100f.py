"""
Experiment K Phase K2: Precomputes 100-Frame (100, 165) YOLO Pose Feature Tensors & Manifest.

Inputs:
- Source dataset: Le2i videos (127 videos across Coffee_room_01, Coffee_room_02, Home_01, Home_02)
- Window length: 100 frames (4.0s at 25 FPS)
- Stride: 25 frames (1.0s overlap)

Outputs:
- Manifest: processed_data/Le2i_baseline/processed_pose_100f_manifest.csv (1,142 rows)
- Feature Directory: processed_data/Le2i_baseline/pose_estimator_features/yolo_pose_100f/ (1,142 NPZ files)
- Summary JSON: R&D/ML_Baseline/results/yolo_k2_100f_precomputation_summary.json
"""

import os
import sys
import glob
import time
import json
import cv2
import numpy as np
import pandas as pd

ROOT_DIR = r"d:\ONE_DATA\Fall detection"
sys.path.insert(0, ROOT_DIR)

COCO_TO_CANONICAL_33 = {
    0: 0,   # Nose
    1: 2,   # L_Eye
    2: 5,   # R_Eye
    3: 7,   # L_Ear
    4: 8,   # R_Ear
    5: 11,  # L_Shoulder
    6: 12,  # R_Shoulder
    7: 13,  # L_Elbow
    8: 14,  # R_Elbow
    9: 15,  # L_Wrist
    10: 16, # R_Wrist
    11: 23, # L_Hip
    12: 24, # R_Hip
    13: 25, # L_Knee
    14: 26, # R_Knee
    15: 27, # L_Ankle
    16: 28  # R_Ankle
}

class YOLOPoseExtractor:
    def __init__(self):
        from ultralytics import YOLO
        model_path = os.path.join(ROOT_DIR, "models", "yolov8n-pose.pt")
        if not os.path.exists(model_path):
            self.model = YOLO("yolov8n-pose.pt")
        else:
            self.model = YOLO(model_path)

    def extract_landmarks(self, frame_bgr):
        raw_33 = np.zeros((33, 3), dtype=np.float32)
        h_img, w_img = frame_bgr.shape[:2]
        results = self.model.predict(frame_bgr, verbose=False, conf=0.25)
        
        if len(results) > 0 and len(results[0].keypoints) > 0 and len(results[0].keypoints.data) > 0:
            kpts_data = results[0].keypoints.data[0].cpu().numpy()
            confs = results[0].keypoints.conf[0].cpu().numpy() if results[0].keypoints.conf is not None else np.ones(17)

            for coco_idx, can_idx in COCO_TO_CANONICAL_33.items():
                if coco_idx < len(kpts_data):
                    x_px, y_px = kpts_data[coco_idx][:2]
                    conf = float(confs[coco_idx]) if coco_idx < len(confs) else 0.5
                    raw_33[can_idx] = [x_px / float(w_img), y_px / float(h_img), conf]
            return raw_33, True
        return raw_33, False

def compute_165d_pose_features_100f(raw_window_33):
    # raw_window_33: (100, 33, 3)
    T = raw_window_33.shape[0]
    norm_window_99 = np.zeros((T, 99), dtype=np.float32)

    for t in range(T):
        frame_raw = raw_window_33[t]
        vis = frame_raw[:, 2]

        if np.sum(vis > 0) == 0:
            norm_window_99[t] = 0.0
            continue

        hip_center = 0.5 * (frame_raw[23, :2] + frame_raw[24, :2])
        sh_center  = 0.5 * (frame_raw[11, :2] + frame_raw[12, :2])
        torso_len  = np.linalg.norm(sh_center - hip_center)

        if torso_len < 1e-5:
            torso_len = 1.0

        norm_coords = (frame_raw[:, :2] - hip_center) / torso_len
        frame_99 = np.zeros((33, 3), dtype=np.float32)
        frame_99[:, :2] = norm_coords
        frame_99[:, 2]  = vis
        norm_window_99[t] = frame_99.flatten()

    # Velocity Derivation (66-D)
    pos_coords_T = norm_window_99.reshape(T, 33, 3)[:, :, :2]
    vel_T = np.zeros((T, 33, 2), dtype=np.float32)
    vel_T[1:] = pos_coords_T[1:] - pos_coords_T[:-1]
    vel_66 = vel_T.reshape(T, 66)

    # 165-D Feature Vector (100, 165)
    features_165 = np.hstack([norm_window_99, vel_66]).astype(np.float32)
    return features_165

def precompute_le2i_yolo_100f():
    print("=" * 70)
    print("EXPERIMENT K PHASE K2: PRECOMPUTING 100-FRAME YOLO POSE FEATURES & MANIFEST")
    print("=" * 70)

    src_manifest_path = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "processed_pose_features_manifest.csv")
    df_src = pd.read_csv(src_manifest_path)

    # Get unique video paths
    unique_videos = df_src.groupby("raw_video_path").first().reset_index()
    print(f"Total Unique Videos to Process: {len(unique_videos)}")

    target_dir = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "pose_estimator_features", "yolo_pose_100f")
    os.makedirs(target_dir, exist_ok=True)

    extractor = YOLOPoseExtractor()

    k2_manifest_rows = []
    win_len = 100
    stride = 25
    start_time = time.time()
    total_bytes = 0

    for v_idx, v_row in unique_videos.iterrows():
        v_path = v_row["raw_video_path"]
        loc = v_row["location"]
        event_id = v_row["event_id"]
        v_name = os.path.basename(v_path)

        # Fall Annotation bounds
        ann_file = v_path.replace("Videos", "Annotation_files").replace(".avi", ".txt").replace(".mp4", ".txt")
        f_start, f_end = 0, 0
        if os.path.exists(ann_file):
            with open(ann_file) as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            if len(lines) >= 2:
                try:
                    f_start = int(lines[0])
                    f_end = int(lines[1])
                except ValueError:
                    pass

        cap = cv2.VideoCapture(v_path)
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        
        # Extract raw landmarks for all frames in video
        raw_frames_33 = []
        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                break
            raw_33, _ = extractor.extract_landmarks(frame_bgr)
            raw_frames_33.append(raw_33)
        cap.release()

        n_frames = len(raw_frames_33)
        raw_frames_33 = np.array(raw_frames_33) # (N, 33, 3)

        w_count = 0
        for w_start in range(0, n_frames - win_len + 1, stride):
            w_end = w_start + win_len - 1
            
            if f_start > 0 and f_end > 0:
                overlap = max(0, min(w_end, f_end) - max(w_start, f_start) + 1)
                lbl = "FALL" if overlap > 0 else "NORMAL"
            else:
                lbl = "NORMAL"

            wid = f"{event_id}_k2w{w_count:03d}"
            
            raw_win_33 = raw_frames_33[w_start : w_start + win_len] # (100, 33, 3)
            feat_165_100f = compute_165d_pose_features_100f(raw_win_33) # (100, 165)

            target_npz = os.path.join(target_dir, f"{wid}.npz")
            np.savez_compressed(target_npz, features=feat_165_100f)

            sz = os.path.getsize(target_npz)
            total_bytes += sz

            k2_manifest_rows.append({
                "window_id": wid,
                "event_id": event_id,
                "video_id": v_name,
                "camera_id": "cam_01",
                "location": loc,
                "partition": "LOLO",
                "label": lbl,
                "processed_sample_path": target_npz,
                "raw_video_path": v_path,
                "source_fps": fps,
                "source_frames": n_frames,
                "f_start": f_start,
                "f_end": f_end,
                "win_start_frame": w_start,
                "win_end_frame": w_end
            })
            w_count += 1

        print(f"   [{v_idx+1:3d}/{len(unique_videos)}] {event_id:45s} -> {w_count:2d} 100f windows")

    elapsed = time.time() - start_time

    # Sort & Save Manifest
    df_k2_manifest = pd.DataFrame(k2_manifest_rows).sort_values("window_id").reset_index(drop=True)
    manifest_target_path = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "processed_pose_100f_manifest.csv")
    df_k2_manifest.to_csv(manifest_target_path, index=False)

    print(f"\nExtracted {len(df_k2_manifest)} 100f windows in {elapsed:.2f} seconds.")
    print(f"Manifest Saved: {manifest_target_path}")
    print(f"Total Storage Size: {total_bytes / (1024 * 1024):.2f} MB")

    summary = {
        "experiment": "Experiment K Phase K2 (100-Frame Precomputation)",
        "total_windows": len(df_k2_manifest),
        "feature_dim": 165,
        "tensor_shape": [100, 165],
        "dtype": "float32",
        "total_storage_mb": round(total_bytes / (1024 * 1024), 2),
        "precomputation_time_sec": round(elapsed, 2),
        "target_manifest": manifest_target_path,
        "target_directory": target_dir,
        "label_counts": df_k2_manifest["label"].value_counts().to_dict(),
        "location_counts": df_k2_manifest["location"].value_counts().to_dict()
    }

    res_dir = os.path.join(ROOT_DIR, "R&D", "ML_Baseline", "results")
    os.makedirs(res_dir, exist_ok=True)
    summary_path = os.path.join(res_dir, "yolo_k2_100f_precomputation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary Saved: {summary_path}")

if __name__ == "__main__":
    precompute_le2i_yolo_100f()
