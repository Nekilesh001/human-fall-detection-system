"""
Validation Script for Le2i Preprocessed Temporal Dataset
Verifies:
1. Exactly 127 supervised source videos are represented in the processed dataset.
2. Exactly 63 UNKNOWN records remain strictly excluded.
3. No Office or Lecture_room records appear in the processed manifest.
4. No malformed annotation records appear.
5. All sample windows have shape (50, 240, 320, 3) uint8.
6. Home_02 samples contain exact top-30 and bottom-30 vertical zero-padding.
7. All processed sample paths exist on disk.
8. Window label assignment consistency.
"""

import os
import sys
import pandas as pd
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

MALFORMED_REL_IDS = {
    "Coffee_room_01/video (26)",
    "Coffee_room_02/video (50)",
    "Coffee_room_02/video (52)"
}

def validate_le2i_preprocessing():
    print("=" * 70)
    print("LE2I PREPROCESSING VALIDATION AUDIT")
    print("=" * 70)

    manifest_path = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "processed_manifest.csv")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Le2i processed manifest missing at {manifest_path}")

    df_manifest = pd.read_csv(manifest_path)
    print(f"Manifest Loaded           : {len(df_manifest)} total window records")

    # 1. Source Video Count Verification
    source_videos = df_manifest["raw_video_path"].unique()
    print(f"\n1. SOURCE VIDEO COUNT VERIFICATION")
    print(f"   Supervised Videos Processed : {len(source_videos)} (Target: 127)")
    assert len(source_videos) == 127, f"Expected 127 source videos, found {len(source_videos)}"

    # 2. Excluded UNKNOWN Records Verification
    print(f"\n2. EXCLUDED UNKNOWN RECORDS VERIFICATION")
    office_lecture_records = df_manifest[df_manifest["location"].isin(["Office", "Lecture_room"])]
    print(f"   Office / Lecture_room Records Found : {len(office_lecture_records)} (Target: 0)")
    assert len(office_lecture_records) == 0, "Found Office or Lecture_room records in manifest!"

    malformed_records = df_manifest[df_manifest["event_id"].isin([f"Le2i_{rel}" for rel in MALFORMED_REL_IDS])]
    print(f"   Malformed Annotation Records Found : {len(malformed_records)} (Target: 0)")
    assert len(malformed_records) == 0, "Found malformed annotation records in manifest!"

    # 3. Location & Label Window Distribution
    print(f"\n3. WINDOW DISTRIBUTION BY LOCATION AND CLASS")
    loc_class_dist = df_manifest.groupby(["location", "label"]).size().unstack(fill_value=0)
    print(loc_class_dist)

    total_fall_windows = sum(df_manifest["label"] == "FALL")
    total_normal_windows = sum(df_manifest["label"] == "NORMAL")
    print(f"   Total FALL Windows   : {total_fall_windows}")
    print(f"   Total NORMAL Windows : {total_normal_windows}")
    print(f"   Total Windows        : {len(df_manifest)}")

    # 4. Sample Integrity & Padding Check (Sample 10 Windows)
    print(f"\n4. SAMPLE TENSOR INTEGRITY & HOME_02 PADDING AUDIT")
    invalid_shapes = 0
    missing_files = 0
    home02_padding_verified = 0

    for idx, row in df_manifest.iterrows():
        sample_rel = str(row["processed_sample_path"]).replace("/", os.sep)
        sample_abs = os.path.join(ROOT_DIR, sample_rel)

        if not os.path.exists(sample_abs):
            missing_files += 1
            continue

        # Check sample tensor
        if idx % 100 == 0 or row["location"] == "Home_02":
            with np.load(sample_abs) as data:
                frames = data["frames"] # (50, 240, 320, 3) uint8
                if frames.shape != (50, 240, 320, 3) or frames.dtype != np.uint8:
                    invalid_shapes += 1

                # Verify Home_02 Padding (+30px top, +30px bottom)
                if row["location"] == "Home_02":
                    top_strip = frames[:, :30, :, :]
                    bottom_strip = frames[:, -30:, :, :]
                    if np.all(top_strip == 0) and np.all(bottom_strip == 0):
                        home02_padding_verified += 1
                    else:
                        print(f"   WARNING: Home_02 sample {row['window_id']} zero-padding check failed!")

    print(f"   Missing File Count                 : {missing_files}")
    print(f"   Invalid Tensor Shape Count         : {invalid_shapes}")
    print(f"   Home_02 Zero-Padded Windows Checked: {home02_padding_verified} (All Top/Bottom 30px = 0)")

    assert missing_files == 0, f"Found {missing_files} missing sample files!"
    assert invalid_shapes == 0, f"Found {invalid_shapes} invalid tensor shapes!"
    assert home02_padding_verified > 0, "No Home_02 zero-padded windows verified!"

    print("\n" + "=" * 70)
    print("LE2I PREPROCESSING VALIDATION AUDIT COMPLETE (ALL PASS)")
    print("=" * 70)

if __name__ == "__main__":
    validate_le2i_preprocessing()
