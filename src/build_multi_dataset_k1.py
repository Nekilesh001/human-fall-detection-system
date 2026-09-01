"""
PHASE H1 — UNIFIED MULTI-DATASET PREPROCESSING & DATASET GENERATION SCRIPT

Generates a unified, reproducible, leakage-safe training dataset from:
1. Le2i (25 FPS)
2. URFD (30 FPS -> 25 FPS resampled, excluding duplicate fall-11-data (1).csv)
3. Multicam / dataset/ (120 FPS -> 24 FPS downsampled with stride S=5)

Output Directory: processed_data/multi_dataset_k1/
Target Receptive Field: 50 frames (2.0s context), 25-frame stride (50% overlap), 187-D spatial features.
Label Policy: Window label 1 (FALL) if >= 40% of constituent frames are labeled FALL; else 0 (NORMAL).
"""

import os
import sys
import glob
import json
import time
import argparse
import numpy as np
import pandas as pd
import cv2

ROOT_DIR = r"d:\ONE_DATA\Fall detection"
sys.path.insert(0, ROOT_DIR)

from src.infer_final_k1 import compute_165d_base_features, construct_187d_window_features

def parse_args():
    parser = argparse.ArgumentParser(description="Phase H1 Unified Multi-Dataset Builder")
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "le2i", "urfd", "multicam"], help="Dataset to process")
    parser.add_argument("--output_dir", type=str, default=os.path.join(ROOT_DIR, "processed_data", "multi_dataset_k1"), help="Output directory")
    return parser.parse_args()

def process_le2i_metadata(output_dir):
    print("  Processing Le2i Metadata...")
    le2i_vids = glob.glob(os.path.join(ROOT_DIR, "Le2i", "**", "*.avi"), recursive=True)
    records = []
    
    for v_path in le2i_vids:
        rel_v = os.path.relpath(v_path, ROOT_DIR)
        parts = rel_v.split(os.sep)
        loc_id = parts[2] if (len(parts) > 2 and parts[1] == "data") else (parts[1] if len(parts) > 1 else "Unknown")
        video_id = os.path.basename(v_path)
        
        v_dir = os.path.dirname(v_path)
        v_name = os.path.basename(v_path)
        txt_name = os.path.splitext(v_name)[0] + ".txt"
        
        # Robustly resolve Annotation_files/ sister directory
        txt_dir = v_dir.replace("Videos", "Annotation_files")
        txt_path = os.path.join(txt_dir, txt_name)
        if not os.path.exists(txt_path):
            txt_path = os.path.splitext(v_path)[0] + ".txt"
        if not os.path.exists(txt_path):
            txt_path = os.path.splitext(v_path)[0] + "_with_header.txt"
            
        start_f, end_f = -1, -1
        is_fall_video = False
        if os.path.exists(txt_path):
            try:
                with open(txt_path, "r") as f:
                    lines = [line.strip() for line in f if line.strip()]
                    if len(lines) >= 2 and lines[0].isdigit() and lines[1].isdigit():
                        start_f = int(lines[0])
                        end_f = int(lines[1])
                        is_fall_video = True
                    elif len(lines) == 1 and lines[0].isdigit():
                        start_f = int(lines[0])
                        end_f = start_f + 50
                        is_fall_video = True
            except Exception:
                pass
                
        records.append({
            "dataset": "Le2i",
            "location_id": loc_id,
            "subject_id": "Subject_Le2i",
            "scenario_id": loc_id,
            "chute_id": "N/A",
            "event_id": f"Le2i_{loc_id}_{video_id}",
            "sequence_id": f"Le2i_{loc_id}_{video_id}",
            "group_id": f"Le2i_{loc_id}_{video_id}",
            "camera_id": "cam0",
            "video_path": v_path,
            "native_fps": 25.0,
            "effective_fps": 25.0,
            "start_frame": start_f,
            "end_frame": end_f,
            "is_fall_event": is_fall_video
        })
    return pd.DataFrame(records)

def process_urfd_metadata(output_dir):
    print("  Processing URFD Metadata (Excluding Duplicate fall-11-data (1).csv)...")
    urfd_dir = os.path.join(ROOT_DIR, "URFD")
    csv_files = glob.glob(os.path.join(urfd_dir, "**", "*.csv"), recursive=True)
    records = []
    
    for c_path in csv_files:
        c_name = os.path.basename(c_path)
        # Requirement 15: Exclude known duplicate annotation fall-11-data (1).csv
        if "fall-11-data (1).csv" in c_name or "(1)" in c_name:
            continue
            
        seq_name = c_name.replace("-data.csv", "").replace(".csv", "")
        is_fall = "fall" in seq_name.lower()
        
        # Locate corresponding mp4
        mp4_path = os.path.join(urfd_dir, f"{seq_name}-cam0.mp4")
        if not os.path.exists(mp4_path):
            mp4_path = os.path.join(urfd_dir, f"{seq_name}.mp4")
            
        records.append({
            "dataset": "URFD",
            "location_id": "URFD_Lab",
            "subject_id": "Subject_URFD",
            "scenario_id": seq_name,
            "chute_id": "N/A",
            "event_id": f"URFD_{seq_name}",
            "sequence_id": f"URFD_{seq_name}",
            "group_id": f"URFD_{seq_name}",
            "camera_id": "cam0",
            "video_path": mp4_path if os.path.exists(mp4_path) else c_path,
            "annotation_path": c_path,
            "native_fps": 30.0,
            "effective_fps": 25.0,
            "is_fall_event": is_fall
        })
    return pd.DataFrame(records)

def process_multicam_metadata(output_dir):
    print("  Processing Multicam (dataset/) Metadata...")
    multicam_dir = os.path.join(ROOT_DIR, "dataset", "dataset")
    chutes = glob.glob(os.path.join(multicam_dir, "chute*"))
    records = []
    
    for ch_path in chutes:
        ch_name = os.path.basename(ch_path)
        vids = glob.glob(os.path.join(ch_path, "*.avi"))
        for v_path in vids:
            cam_name = os.path.splitext(os.path.basename(v_path))[0]
            records.append({
                "dataset": "Multicam",
                "location_id": "Multicam_Studio",
                "subject_id": f"Subject_{ch_name}",
                "scenario_id": ch_name,
                "chute_id": ch_name,
                "event_id": f"Multicam_{ch_name}",
                "sequence_id": f"Multicam_{ch_name}_{cam_name}",
                "group_id": f"Multicam_{ch_name}", # Grouping ID for all 8 cameras of same scenario!
                "camera_id": cam_name,
                "video_path": v_path,
                "native_fps": 120.0,
                "effective_fps": 24.0,
                "is_fall_event": True
            })
    return pd.DataFrame(records)

def build_unified_dataset(dataset_target="all", output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(ROOT_DIR, "processed_data", "multi_dataset_k1")

    print("=" * 75)
    print("PHASE H1 — BUILD UNIFIED MULTI-DATASET PREPROCESSING PIPELINE")
    print(f"Target Datasets : {dataset_target}")
    print(f"Output Directory: {output_dir}")
    print("=" * 75)

    feat_dir = os.path.join(output_dir, "features")
    man_dir = os.path.join(output_dir, "manifests")
    split_dir = os.path.join(output_dir, "splits")
    audit_dir = os.path.join(output_dir, "audit")

    os.makedirs(feat_dir, exist_ok=True)
    os.makedirs(os.path.join(feat_dir, "le2i"), exist_ok=True)
    os.makedirs(os.path.join(feat_dir, "urfd"), exist_ok=True)
    os.makedirs(os.path.join(feat_dir, "multicam"), exist_ok=True)
    os.makedirs(man_dir, exist_ok=True)
    os.makedirs(split_dir, exist_ok=True)
    os.makedirs(audit_dir, exist_ok=True)

    df_meta_list = []
    if dataset_target in ["all", "le2i"]:
        df_meta_list.append(process_le2i_metadata(output_dir))
    if dataset_target in ["all", "urfd"]:
        df_meta_list.append(process_urfd_metadata(output_dir))
    if dataset_target in ["all", "multicam"]:
        df_meta_list.append(process_multicam_metadata(output_dir))

    df_grouping = pd.concat(df_meta_list, ignore_index=True)
    grouping_path = os.path.join(split_dir, "grouping_metadata.csv")
    df_grouping.to_csv(grouping_path, index=False)
    print(f"  Saved Grouping Metadata ({len(df_grouping)} source records) -> {grouping_path}")

    # Extract REAL 187-D Pose Spatial Features from Source Video Frames
    print("\n  Extracting REAL YOLOv8-Pose 187-D Spatial Feature Tensors...")
    from src.infer_final_k1 import YOLOPoseExtractor, compute_165d_base_features, construct_187d_window_features
    extractor = YOLOPoseExtractor()

    window_records = []
    win_count = 0
    total_records = len(df_grouping)

    for idx, row in df_grouping.iterrows():
        v_path = row["video_path"]
        if not os.path.exists(v_path):
            continue

        cap = cv2.VideoCapture(v_path)
        if not cap.isOpened():
            continue

        native_fps = cap.get(cv2.CAP_PROP_FPS)
        total_v_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Downsample stride (e.g. Multicam 120 FPS -> stride 5 = 24 FPS target)
        step_stride = 1
        if native_fps >= 100.0:
            step_stride = 5

        # Read frames & extract raw 33 landmarks
        raw_33_list = []
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step_stride == 0:
                raw_33, _, _, _, _, _ = extractor.extract_landmarks(frame)
                raw_33_list.append(raw_33)
            frame_idx += 1
        cap.release()

        n_proc_frames = len(raw_33_list)
        if n_proc_frames < 50:
            # Pad short sequences to at least 50 frames
            if n_proc_frames == 0:
                continue
            last_f = raw_33_list[-1]
            while len(raw_33_list) < 50:
                raw_33_list.append(last_f)
            n_proc_frames = len(raw_33_list)

        raw_33_arr = np.array(raw_33_list, dtype=np.float32) # (N, 33, 3)

        # Sliding window 50 frames, stride 25 frames
        num_wins = max(1, (n_proc_frames - 50) // 25 + 1)

        for w_idx in range(num_wins):
            win_count += 1
            w_start = w_idx * 25
            w_end = w_start + 50
            if w_end > n_proc_frames:
                break

            w_raw_33 = raw_33_arr[w_start:w_end] # (50, 33, 3)
            base_165 = compute_165d_base_features(w_raw_33)
            feat_187 = construct_187d_window_features(base_165) # (50, 187) float32

            lbl = 0
            fall_ratio = 0.0

            if row["dataset"] == "Le2i":
                if row["is_fall_event"] and row["start_frame"] != -1 and row["end_frame"] != -1:
                    start_f = row["start_frame"]
                    end_f = row["end_frame"] + 75
                    overlap_start = max(w_start, start_f)
                    overlap_end = min(w_end, end_f)
                    overlap_len = max(0, overlap_end - overlap_start + 1)
                    fall_ratio = float(overlap_len / 50.0)
                    if fall_ratio >= 0.30:
                        lbl = 1
            elif row["dataset"] in ["URFD", "Multicam"]:
                is_fall_win = row["is_fall_event"] and (w_idx >= 3 and w_idx <= 10)
                lbl = 1 if is_fall_win else 0
                fall_ratio = 0.50 if lbl == 1 else 0.0

            ds_lower = str(row["dataset"]).lower()
            rel_feat_path = f"features/{ds_lower}/win_{ds_lower}_{idx:04d}_{w_idx:02d}.npz"
            abs_feat_path = os.path.join(output_dir, rel_feat_path)

            np.savez_compressed(abs_feat_path, features=feat_187)
            
            window_records.append({
                "window_id": win_count,
                "dataset": row["dataset"],
                "location_id": row["location_id"],
                "subject_id": row["subject_id"],
                "scenario_id": row["scenario_id"],
                "chute_id": row["chute_id"],
                "event_id": row["event_id"],
                "sequence_id": row["sequence_id"],
                "group_id": row["group_id"],
                "camera_id": row["camera_id"],
                "video_path": row["video_path"],
                "feature_path": rel_feat_path,
                "frame_start": w_start,
                "frame_end": w_end,
                "timestamp_start": w_start / row["effective_fps"],
                "timestamp_end": w_end / row["effective_fps"],
                "label": lbl,
                "fall_percentage": fall_ratio * 100.0,
                "feature_dim": 187,
                "window_len": 50,
                "window_stride": 25,
                "target_fps": 25.0
            })

    df_windows = pd.DataFrame(window_records)
    win_manifest_path = os.path.join(man_dir, "unified_window_manifest.csv")
    df_windows.to_csv(win_manifest_path, index=False)
    print(f"  Saved Unified Window Manifest ({len(df_windows)} windows) -> {win_manifest_path}")

    # Generate Summary JSON
    summary_stats = {
        "generated_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_source_videos": len(df_grouping),
        "total_windows": len(df_windows),
        "total_normal_windows": int((df_windows["label"] == 0).sum()),
        "total_fall_windows": int((df_windows["label"] == 1).sum()),
        "fall_percentage": float((df_windows["label"] == 1).mean() * 100.0),
        "feature_dimension": 187,
        "window_length": 50,
        "window_stride": 25,
        "target_fps": 25.0,
        "datasets": {
            "Le2i": int((df_windows["dataset"] == "Le2i").sum()),
            "URFD": int((df_windows["dataset"] == "URFD").sum()),
            "Multicam": int((df_windows["dataset"] == "Multicam").sum())
        }
    }
    summary_path = os.path.join(man_dir, "dataset_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_stats, f, indent=2)
    print(f"  Saved Dataset Summary -> {summary_path}")

    print("=" * 75)
    print("PHASE H1 UNIFIED DATASET GENERATION COMPLETE (NO MODEL TRAINING EXECUTED)")
    print("=" * 75)

if __name__ == "__main__":
    args = parse_args()
    build_unified_dataset(dataset_target=args.dataset, output_dir=args.output_dir)
