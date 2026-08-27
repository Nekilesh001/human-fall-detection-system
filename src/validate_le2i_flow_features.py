"""
Validation Audit Script for Precomputed Le2i Optical Flow Features
Verifies:
1. All 1,396 flow feature tensors exist and are readable.
2. Tensor shape is exactly (49, 512) float32.
3. 0 missing files, 0 NaNs, 0 Infs.
4. 0 UNKNOWN records present.
"""

import os
import sys
import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def validate_flow_features():
    print("=" * 70)
    print("LE2I OPTICAL FLOW FEATURE VALIDATION AUDIT")
    print("=" * 70)

    flow_manifest_path = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "processed_flow_features_manifest.csv")
    assert os.path.exists(flow_manifest_path), f"Flow manifest missing at {flow_manifest_path}"

    df_flow = pd.read_csv(flow_manifest_path)
    total_records = len(df_flow)
    print(f"Loaded Flow Feature Manifest : {total_records} records (Target: 1,396)")
    assert total_records == 1396, f"Expected 1,396 records, got {total_records}"

    missing_files = 0
    invalid_shapes = 0
    nan_inf_errors = 0

    for idx, row in df_flow.iterrows():
        rel_path = str(row["flow_feature_path"]).replace("/", os.sep)
        abs_path = os.path.join(ROOT_DIR, rel_path)

        if not os.path.exists(abs_path):
            missing_files += 1
            continue

        try:
            with np.load(abs_path) as data:
                feats = data["features"]

            if feats.shape != (49, 512):
                invalid_shapes += 1

            if np.isnan(feats).any() or np.isinf(feats).any():
                nan_inf_errors += 1

        except Exception as e:
            print(f"Error reading {abs_path}: {e}")
            missing_files += 1

    print(f"\n1. FEATURE INTEGRITY VERIFICATION:")
    print(f"   Missing Feature Files      : {missing_files} (Target: 0)")
    print(f"   Invalid Tensor Shapes      : {invalid_shapes} (Target: 0)")
    print(f"   NaN / Inf Errors           : {nan_inf_errors} (Target: 0)")

    # Excluded location audit
    excluded_locs = ["Office", "Lecture_room"]
    excluded_count = len(df_flow[df_flow["location"].isin(excluded_locs)])
    print(f"\n2. EXCLUDED UNKNOWN RECORDS AUDIT:")
    print(f"   Excluded Location Records  : {excluded_count} (Target: 0)")

    assert missing_files == 0, "Missing feature files detected!"
    assert invalid_shapes == 0, "Invalid feature tensor shapes detected!"
    assert nan_inf_errors == 0, "NaN or Inf values detected!"
    assert excluded_count == 0, "Excluded UNKNOWN records found in flow manifest!"

    print("\n" + "=" * 70)
    print("LE2I OPTICAL FLOW FEATURE VALIDATION COMPLETE (ALL PASS)")
    print("=" * 70)

if __name__ == "__main__":
    validate_flow_features()
