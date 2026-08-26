"""
Master Dataset Manifest Read-Only Validation & Integrity Check Utility
"""

import os
import pandas as pd

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    manifest_path = os.path.join(root_dir, "R&D", "Dataset_Analysis", "dataset_manifest.csv")
    
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest file not found at {manifest_path}")
        return

    df = pd.read_csv(manifest_path)
    print(f"Loaded manifest from {manifest_path} ({len(df)} records)")

    missing_video_files = []
    missing_annotations = []
    
    for idx, row in df.iterrows():
        v_rel = str(row['video_path']).replace("/", os.sep)
        v_path = os.path.join(root_dir, v_rel)
        if row['video_path'] != "UNKNOWN" and not os.path.exists(v_path):
            missing_video_files.append(row['video_id'])
            
        ann_rel = str(row['annotation_path']).replace("/", os.sep)
        ann_path = os.path.join(root_dir, ann_rel)
        if row['annotation_path'] != "UNKNOWN" and not os.path.exists(ann_path):
            missing_annotations.append((row['video_id'], row['annotation_path']))

    v_counts = df['video_id'].value_counts()
    duplicate_video_ids = v_counts[v_counts > 1].index.tolist()

    event_cam_counts = df.groupby(['dataset', 'event_id', 'camera_id']).size()
    duplicate_event_cams = event_cam_counts[event_cam_counts > 1].index.tolist()

    conflicting_event_labels = []
    event_label_groups = df.groupby(['dataset', 'event_id'])['label'].unique()
    for (ds, ev), labels in event_label_groups.items():
        if len(labels) > 1:
            conflicting_event_labels.append((ds, ev, list(labels)))

    print("\n--- Integrity Verification Results ---")
    print(f"Missing Video Files: {len(missing_video_files)}")
    print(f"Missing Annotation Files Referenced: {len(missing_annotations)}")
    print(f"Duplicate Video IDs: {len(duplicate_video_ids)}")
    print(f"Duplicate Event/Camera Pairs: {len(duplicate_event_cams)}")
    print(f"Conflicting Event Labels: {len(conflicting_event_labels)}")

if __name__ == "__main__":
    main()
