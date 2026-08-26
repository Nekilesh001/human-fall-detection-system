"""
URFD RGB Baseline Preprocessing Pipeline
Reads raw URFD RGB frames, applies deterministic 25 FPS resampling, Lanczos 320x240 resizing,
50-frame temporal windowing with 25-frame stride, and writes processed samples + manifest.
"""

import os
import glob
import time
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from config import PreprocessingConfig

def load_source_frames_and_timestamps(rgb_folder, csv_path):
    # Find all PNG frames sorted
    png_files = sorted(glob.glob(os.path.join(rgb_folder, "**", "*.png"), recursive=True))
    if not png_files:
        return [], []

    # Read timestamps from CSV if available
    timestamps_ms = []
    if csv_path and os.path.exists(csv_path):
        try:
            df_csv = pd.read_csv(csv_path, header=None)
            if len(df_csv.columns) >= 2:
                timestamps_ms = df_csv[1].values.tolist()
        except Exception:
            pass

    # Fallback to uniform 30 FPS timestamps if CSV timestamps count mismatch
    if len(timestamps_ms) != len(png_files):
        timestamps_ms = [i * (1000.0 / 30.0) for i in range(len(png_files))]

    return png_files, timestamps_ms

def resample_and_resize(png_files, source_timestamps_ms, config):
    if not png_files:
        return [], []

    total_duration_ms = source_timestamps_ms[-1] - source_timestamps_ms[0]
    target_period_ms = 1000.0 / config.TARGET_FPS
    target_num_frames = int(np.floor(total_duration_ms / target_period_ms)) + 1
    
    target_timestamps_ms = [i * target_period_ms for i in range(target_num_frames)]
    
    resampled_frames = []
    resampled_timestamps = []

    for ts in target_timestamps_ms:
        idx = np.argmin(np.abs(np.array(source_timestamps_ms) - ts))
        img_path = png_files[idx]
        
        # Load and resize using Lanczos
        img = Image.open(img_path).convert("RGB")
        img_resized = img.resize((config.TARGET_WIDTH, config.TARGET_HEIGHT), Image.LANCZOS)
        arr = np.array(img_resized, dtype=np.uint8) # Shape: (240, 320, 3)
        
        resampled_frames.append(arr)
        resampled_timestamps.append(source_timestamps_ms[idx])

    return resampled_frames, resampled_timestamps

def process_video_record(row, root_dir, out_dir, config):
    event_id = row['event_id']
    video_id = row['video_id']
    camera_id = row['camera_id']
    label = row['label']
    partition = PreprocessingConfig.get_event_partition(event_id)

    rgb_folder = os.path.join(root_dir, str(row['rgb_path']).replace("/", os.sep))
    csv_path = os.path.join(root_dir, str(row['timestamp_path']).replace("/", os.sep))

    png_files, timestamps_ms = load_source_frames_and_timestamps(rgb_folder, csv_path)
    if not png_files:
        return [], f"Missing PNG frames for {video_id}"

    resampled_frames, resampled_timestamps = resample_and_resize(png_files, timestamps_ms, config)
    
    num_resampled = len(resampled_frames)
    if num_resampled < config.WINDOW_SIZE:
        return [], f"Insufficient resampled frames ({num_resampled} < {config.WINDOW_SIZE}) for {video_id}"

    # Temporal window generation
    manifest_records = []
    w_idx = 0
    
    for start_idx in range(0, num_resampled - config.WINDOW_SIZE + 1, config.STRIDE):
        end_idx = start_idx + config.WINDOW_SIZE
        window_frames = np.stack(resampled_frames[start_idx:end_idx], axis=0) # (50, 240, 320, 3)
        
        window_id = f"{video_id}_w{w_idx:03d}"
        partition_dir = os.path.join(out_dir, partition)
        os.makedirs(partition_dir, exist_ok=True)
        
        sample_filename = f"{window_id}.npz"
        sample_path = os.path.join(partition_dir, sample_filename)
        
        # Save compressed numpy sample array
        np.savez_compressed(sample_path, frames=window_frames)
        
        rel_sample_path = os.path.relpath(sample_path, root_dir).replace("\\", "/")
        
        start_ts = resampled_timestamps[start_idx]
        end_ts = resampled_timestamps[end_idx - 1]
        
        manifest_records.append({
            "dataset": "URFD",
            "event_id": event_id,
            "video_id": video_id,
            "camera_id": camera_id,
            "partition": partition,
            "label": label,
            "window_id": window_id,
            "start_frame": start_idx,
            "end_frame": end_idx - 1,
            "start_timestamp": round(start_ts, 2),
            "end_timestamp": round(end_ts, 2),
            "num_frames": config.WINDOW_SIZE,
            "source_fps": row['fps'],
            "target_fps": config.TARGET_FPS,
            "width": config.TARGET_WIDTH,
            "height": config.TARGET_HEIGHT,
            "source_video_path": str(row['video_path']).replace("\\", "/"),
            "processed_sample_path": rel_sample_path
        })
        w_idx += 1

    return manifest_records, None

def run_pipeline(subset_events=None):
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    manifest_path = os.path.join(root_dir, "R&D", "Dataset_Analysis", "dataset_manifest.csv")
    
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest file not found at {manifest_path}")
        return

    df_manifest = pd.read_csv(manifest_path)
    urfd_df = df_manifest[df_manifest['dataset'] == 'URFD']

    if subset_events:
        urfd_df = urfd_df[urfd_df['event_id'].isin(subset_events)]
        print(f"=== RUNNING URFD PREPROCESSING SUBSET TEST ({len(subset_events)} events) ===")
    else:
        print("=== RUNNING FULL URFD RGB PREPROCESSING PIPELINE ===")

    config = PreprocessingConfig()
    out_dir = os.path.join(root_dir, "processed_data", "URFD_RGB_baseline")
    os.makedirs(out_dir, exist_ok=True)

    all_window_records = []
    errors = []
    t0 = time.time()

    for idx, row in urfd_df.iterrows():
        records, err = process_video_record(row, root_dir, out_dir, config)
        if err:
            errors.append(err)
        else:
            all_window_records.extend(records)

    elapsed_sec = time.time() - t0
    
    df_processed = pd.DataFrame(all_window_records)
    proc_manifest_path = os.path.join(out_dir, "processed_manifest.csv")
    df_processed.to_csv(proc_manifest_path, index=False)

    print(f"\nPipeline finished in {elapsed_sec:.2f} seconds.")
    print(f"Total processed windows created: {len(df_processed)}")
    print(f"Processed manifest saved at: {proc_manifest_path}")

    if errors:
        print(f"Processing Errors/Skipped ({len(errors)}): {errors}")

    return df_processed, errors, elapsed_sec

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="URFD Preprocessing Pipeline")
    parser.add_argument("--subset", action="store_true", help="Run small test subset")
    args = parser.parse_args()

    if args.subset:
        # Small subset: 1 train fall ('fall-01'), 1 val adl ('adl-04'), 1 test fall ('fall-07')
        test_subset = ['fall-01', 'adl-04', 'fall-07']
        run_pipeline(subset_events=test_subset)
    else:
        run_pipeline()
