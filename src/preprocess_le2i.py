"""
Le2i Preprocessing Pipeline for Zero-Shot Cross-Dataset Baseline Evaluation
Processes the 127 verified supervised Le2i videos into 50-frame, 25 FPS, 320x240 RGB temporal windows.
Applies +30px top and +30px bottom vertical zero-padding for Home_02 (320x180) to preserve anatomical aspect ratio.
Strictly excludes all 63 UNKNOWN records (Office, Lecture_room, and 3 malformed annotation files).
"""

import os
import sys
import glob
import time
import cv2
import pandas as pd
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

MALFORMED_REL_IDS = {
    "Coffee_room_01/video (26)",
    "Coffee_room_02/video (50)",
    "Coffee_room_02/video (52)"
}

SUPERVISED_LOCATIONS = ["Coffee_room_01", "Coffee_room_02", "Home_01", "Home_02"]

def preprocess_le2i():
    print("=" * 70)
    print("LE2I PREPROCESSING PIPELINE (ZERO-SHOT BASELINE)")
    print("=" * 70)

    le2i_base_dir = os.path.join(ROOT_DIR, "Le2i", "data")
    if not os.path.exists(le2i_base_dir):
        le2i_base_dir = os.path.join(ROOT_DIR, "Le2i")

    output_base_dir = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline")
    os.makedirs(output_base_dir, exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, "samples"), exist_ok=True)

    manifest_rows = []
    total_videos_processed = 0
    total_windows_generated = 0
    skipped_videos = []

    start_time = time.perf_counter()

    for loc in SUPERVISED_LOCATIONS:
        loc_dir = os.path.join(le2i_base_dir, loc)
        vids = sorted(
            glob.glob(os.path.join(loc_dir, "**", "*.avi"), recursive=True) +
            glob.glob(os.path.join(loc_dir, "**", "*.mp4"), recursive=True) +
            glob.glob(os.path.join(loc_dir, "**", "*.mpg"), recursive=True)
        )

        print(f"\nProcessing Location: {loc} ({len(vids)} raw videos found)")

        for v_path in vids:
            v_name = os.path.basename(v_path)
            v_id = os.path.splitext(v_name)[0]
            rel_id = f"{loc}/{v_id}"

            # Check if malformed record
            if rel_id in MALFORMED_REL_IDS:
                skipped_videos.append((rel_id, "EXCLUDED_MALFORMED"))
                continue

            # Annotation Check
            anno_txt_name = f"{v_id}.txt"
            anno_path = os.path.join(os.path.dirname(v_path), "..", "Annotation_files", anno_txt_name)
            if not os.path.exists(anno_path):
                anno_path = os.path.join(os.path.dirname(v_path), "..", "Annotations_files", anno_txt_name)
                if not os.path.exists(anno_path):
                    anno_path = os.path.join(os.path.dirname(v_path), "..", "Annotation", anno_txt_name)

            has_anno = os.path.exists(anno_path)
            f_start, f_end = None, None
            video_label = "NORMAL"

            if has_anno:
                try:
                    with open(anno_path, "r") as f:
                        lines = [line.strip() for line in f if line.strip()]
                    if len(lines) >= 2:
                        f_start = int(lines[0])
                        f_end = int(lines[1])
                        if f_start > 0 and f_end > 0:
                            video_label = "FALL"
                        else:
                            video_label = "NORMAL"
                    else:
                        skipped_videos.append((rel_id, "EXCLUDED_MALFORMED_ANNO"))
                        continue
                except Exception as e:
                    skipped_videos.append((rel_id, f"EXCLUDED_READ_ERROR: {e}"))
                    continue

            # Video Frame Extraction
            cap = cv2.VideoCapture(v_path)
            if not cap.isOpened():
                skipped_videos.append((rel_id, "FAILED_TO_OPEN_VIDEO"))
                continue

            src_fps = float(cap.get(cv2.CAP_PROP_FPS))
            if src_fps <= 0: src_fps = 25.0
            
            raw_frames = []
            while True:
                ret, frame = cap.read()
                if not ret: break
                # Convert BGR -> RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Resizing & Padding Logic
                H_src, W_src, _ = frame_rgb.shape
                if loc == "Home_02" and W_src == 320 and H_src == 180:
                    # Vertical Zero-Padding (+30px top, +30px bottom)
                    frame_processed = cv2.copyMakeBorder(
                        frame_rgb, top=30, bottom=30, left=0, right=0,
                        borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
                    )
                else:
                    # Lanczos Resizing to 320x240
                    frame_processed = cv2.resize(frame_rgb, (320, 240), interpolation=cv2.INTER_LANCZOS4)

                raw_frames.append(frame_processed)

            cap.release()

            N_frames = len(raw_frames)
            if N_frames < 50:
                skipped_videos.append((rel_id, f"SKIPPED_SHORT_VIDEO ({N_frames} frames < 50)"))
                continue

            total_videos_processed += 1

            # Windowing: W=50 frames, S=25 frames
            W = 50
            S = 25
            win_count = 0

            for start_idx in range(0, N_frames - W + 1, S):
                end_idx = start_idx + W
                win_frames = np.stack(raw_frames[start_idx:end_idx], axis=0) # (50, 240, 320, 3) uint8

                # Determine Window Label
                # Frame indexing: 1-indexed for annotation matching (start_idx+1 to end_idx)
                f_win_start = start_idx + 1
                f_win_end = end_idx

                if video_label == "NORMAL":
                    win_label = "NORMAL"
                else:
                    # FALL video
                    if f_win_end < f_start:
                        win_label = "NORMAL" # Pre-fall phase
                    elif f_win_start > f_end:
                        win_label = "FALL"   # Post-fall phase
                    else:
                        # Active fall overlap check
                        active_overlap = max(0, min(f_win_end, f_end) - max(f_win_start, f_start) + 1)
                        if active_overlap >= 10: # >= 20% of window (10 out of 50 frames)
                            win_label = "FALL"
                        else:
                            win_label = "NORMAL"

                window_id = f"Le2i_{loc}_{v_id}_w{win_count:03d}"
                win_filename = f"{window_id}.npz"
                win_rel_path = os.path.join("processed_data", "Le2i_baseline", "samples", win_filename)
                win_abs_path = os.path.join(ROOT_DIR, win_rel_path)

                np.savez_compressed(win_abs_path, frames=win_frames)

                manifest_rows.append({
                    "window_id": window_id,
                    "event_id": f"Le2i_{loc}_{v_id}",
                    "video_id": v_id,
                    "camera_id": "cam0",
                    "location": loc,
                    "partition": "test", # Zero-shot cross-dataset evaluation test set
                    "label": win_label,
                    "processed_sample_path": win_rel_path.replace(os.sep, "/"),
                    "raw_video_path": v_path,
                    "source_fps": src_fps,
                    "source_frames": N_frames,
                    "f_start": f_start if f_start is not None else -1,
                    "f_end": f_end if f_end is not None else -1,
                    "win_start_frame": f_win_start,
                    "win_end_frame": f_win_end
                })

                win_count += 1
                total_windows_generated += 1

            print(f"  Processed {v_id:15s} | Frames: {N_frames:4d} | Windows: {win_count:3d} | Label: {video_label}")

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    # Save manifest
    df_manifest = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(output_base_dir, "processed_manifest.csv")
    df_manifest.to_csv(manifest_path, index=False)

    print("\n" + "=" * 70)
    print("LE2I PREPROCESSING BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Total Execution Time      : {elapsed:.2f} seconds")
    print(f"Total Supervised Videos   : {total_videos_processed} / 127")
    print(f"Total Windows Generated   : {total_windows_generated}")
    print(f"  - FALL Windows          : {sum(df_manifest['label'] == 'FALL')}")
    print(f"  - NORMAL Windows        : {sum(df_manifest['label'] == 'NORMAL')}")
    print(f"Skipped / Excluded Videos : {len(skipped_videos)}")
    print(f"Manifest Saved            : {manifest_path}")
    print("=" * 70)

    return df_manifest

if __name__ == "__main__":
    preprocess_le2i()
