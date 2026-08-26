"""
Automated Validation & Data Leakage Verification Utility for URFD Preprocessing
"""

import os
import pandas as pd
import numpy as np

def validate_preprocessing_output(proc_manifest_path):
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    if not os.path.exists(proc_manifest_path):
        return {"status": "FAIL", "reason": f"Manifest not found at {proc_manifest_path}"}

    df = pd.read_csv(proc_manifest_path)
    print(f"\n================ VALIDATING PROCESSED MANIFEST ===============")
    print(f"Manifest Path: {proc_manifest_path}")
    print(f"Total Window Records: {len(df)}")

    # 1. EVENT LEAKAGE
    train_events = set(df[df['partition'] == 'train']['event_id'])
    val_events = set(df[df['partition'] == 'val']['event_id'])
    test_events = set(df[df['partition'] == 'test']['event_id'])

    tv_intersect = train_events.intersection(val_events)
    tt_intersect = train_events.intersection(test_events)
    vt_intersect = val_events.intersection(test_events)

    event_leakage_passed = (len(tv_intersect) == 0 and len(tt_intersect) == 0 and len(vt_intersect) == 0)

    # 2. CAMERA LEAKAGE
    cam_leakage_passed = True
    fall_df = df[df['label'] == 'FALL']
    fall_event_parts = fall_df.groupby('event_id')['partition'].unique()
    for ev_id, parts in fall_event_parts.items():
        if len(parts) > 1:
            cam_leakage_passed = False
            print(f"CAM LEAKAGE DETECTED in {ev_id}: Partitions = {parts}")

    # 3. WINDOW COUNT & 4. RESOLUTION & 8. MANIFEST INTEGRITY
    broken_paths = 0
    invalid_shapes = 0

    for idx, row in df.iterrows():
        sample_rel = str(row['processed_sample_path']).replace("/", os.sep)
        sample_abs = os.path.join(root_dir, sample_rel)
        
        if not os.path.exists(sample_abs):
            broken_paths += 1
        else:
            try:
                data = np.load(sample_abs)
                frames = data['frames'] # Expected shape: (50, 240, 320, 3)
                if frames.shape != (50, 240, 320, 3):
                    invalid_shapes += 1
            except Exception:
                broken_paths += 1

    # 6. LABEL CONSISTENCY
    invalid_fall_labels = df[(df['label'] == 'FALL') & (~df['event_id'].str.startswith('fall-'))]
    invalid_adl_labels = df[(df['label'] == 'NORMAL') & (~df['event_id'].str.startswith('adl-'))]
    label_consistency_passed = (len(invalid_fall_labels) == 0 and len(invalid_adl_labels) == 0)

    # 9. PARTITION INTEGRITY REPORT
    partition_summary = []
    for part in ['train', 'val', 'test']:
        sub = df[df['partition'] == part]
        partition_summary.append({
            "partition": part,
            "unique_events": sub['event_id'].nunique(),
            "unique_videos": sub['video_id'].nunique(),
            "FALL_windows": len(sub[sub['label'] == 'FALL']),
            "NORMAL_windows": len(sub[sub['label'] == 'NORMAL']),
            "total_windows": len(sub)
        })

    summary_df = pd.DataFrame(partition_summary)

    results = {
        "event_leakage_passed": event_leakage_passed,
        "camera_leakage_passed": cam_leakage_passed,
        "broken_paths": broken_paths,
        "invalid_shapes": invalid_shapes,
        "label_consistency_passed": label_consistency_passed,
        "partition_summary": summary_df
    }

    print("\n--- Automated Validation Summary ---")
    print(f"1. Event Leakage Check: {'PASSED' if event_leakage_passed else 'FAILED'}")
    print(f"2. Camera Leakage Check: {'PASSED' if cam_leakage_passed else 'FAILED'}")
    print(f"3 & 4. Sample Integrity Check: {broken_paths} broken paths, {invalid_shapes} invalid shapes")
    print(f"6. Label Consistency Check: {'PASSED' if label_consistency_passed else 'FAILED'}")
    print("\nPartition Breakdown:")
    print(summary_df.to_string(index=False))

    return results

if __name__ == "__main__":
    proc_manifest_path = r"d:\ONE_DATA\Fall detection\processed_data\URFD_RGB_baseline\processed_manifest.csv"
    validate_preprocessing_output(proc_manifest_path)
