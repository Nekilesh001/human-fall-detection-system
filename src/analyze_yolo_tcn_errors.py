"""
Experiment J Phase J1: Error & Failure Analysis Script for Champion Model (YOLO Pose + 1D TCN).

Performs deterministic read-only inference across all 4 LOLO test folds:
- checkpoints/le2i_yolo_temporal/tcn/fold_1_best.pth
- checkpoints/le2i_yolo_temporal/tcn/fold_2_best.pth
- checkpoints/le2i_yolo_temporal/tcn/fold_3_best.pth
- checkpoints/le2i_yolo_temporal/tcn/fold_4_best.pth

Generates:
- R&D/ML_Baseline/results/yolo_tcn_window_predictions.csv (1,396 rows)
- R&D/ML_Baseline/results/yolo_tcn_error_analysis.csv (90 error rows)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT_DIR = r"d:\ONE_DATA\Fall detection"
sys.path.insert(0, ROOT_DIR)

from src.train_le2i_yolo_temporal import ModelI3_TCN, YoloPoseDataset, set_seed

def analyze_yolo_tcn_errors():
    print("=" * 70)
    print("EXPERIMENT J PHASE J1: YOLO POSE 1D TCN SOTA ERROR ANALYSIS")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device}")

    manifest_path = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "processed_pose_features_manifest.csv")
    df_manifest = pd.read_csv(manifest_path).sort_values("window_id").reset_index(drop=True)
    yolo_dir = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "pose_estimator_features", "yolo_pose")

    res_json_path = os.path.join(ROOT_DIR, "R&D", "ML_Baseline", "results", "yolo_temporal_benchmark_results.json")
    with open(res_json_path) as f:
        res_data = json.load(f)

    tcn_folds = res_data["tcn"]["folds"]
    tau_stars = {f["fold"]: f["tau_star"] for f in tcn_folds}

    locations = ["Coffee_room_01", "Coffee_room_02", "Home_01", "Home_02"]

    window_predictions = []

    for fold_idx, test_loc in enumerate(locations, 1):
        set_seed(42 + fold_idx)
        tau_star = tau_stars[fold_idx]

        test_df = df_manifest[df_manifest["location"] == test_loc].reset_index(drop=True)
        test_ds = YoloPoseDataset(test_df, yolo_dir)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

        ckpt_path = os.path.join(ROOT_DIR, "checkpoints", "le2i_yolo_temporal", "tcn", f"fold_{fold_idx}_best.pth")
        assert os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}"

        model = ModelI3_TCN().to(device)
        model.load_state_dict(torch.load(ckpt_path))
        model.eval()

        test_probs = []
        with torch.no_grad():
            for x_b, _ in test_loader:
                x_b = x_b.to(device)
                out = model(x_b)
                probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                test_probs.extend(probs)

        test_probs = np.array(test_probs)

        for idx, row in test_df.iterrows():
            wid = row["window_id"]
            raw_label = str(row["label"]).upper()
            y_true = 1 if raw_label in ["FALL", "1"] else 0
            p_fall = float(test_probs[idx])
            y_pred = 1 if p_fall >= tau_star else 0
            is_correct = (y_true == y_pred)

            if y_true == 1 and y_pred == 1:
                cat = "TP"
            elif y_true == 0 and y_pred == 0:
                cat = "TN"
            elif y_true == 0 and y_pred == 1:
                cat = "FP"
            else:
                cat = "FN"

            conf_margin = abs(p_fall - tau_star)
            err_conf = p_fall if cat == "FP" else (1.0 - p_fall) if cat == "FN" else 0.0

            # Kinematic & feature metrics
            fpath = os.path.join(yolo_dir, f"{wid}.npz")
            with np.load(fpath) as d:
                feat = d["features"] # (50, 165)
            
            # Keypoint visibilities (canonical indices 2, 5, 8, ... up to 98)
            vis = feat[:, 2:99:3]
            mean_vis = float(np.mean(vis))

            # Vertical position & velocity of hips (indices 33:36 and velocity 99:165)
            # Velocity features dY for hips/torso
            y_vels = feat[:, 100:165:2] # vertical velocity features
            max_vert_vel = float(np.max(y_vels))
            min_vert_vel = float(np.min(y_vels))

            window_predictions.append({
                "window_id": wid,
                "event_id": row["event_id"],
                "video_path": row["raw_video_path"],
                "location": test_loc,
                "fold": fold_idx,
                "ground_truth_label": raw_label,
                "ground_truth_int": y_true,
                "predicted_prob": round(p_fall, 6),
                "threshold_used": tau_star,
                "predicted_class": y_pred,
                "is_correct": is_correct,
                "error_type": cat,
                "confidence_margin": round(conf_margin, 6),
                "error_confidence": round(err_conf, 6),
                "mean_keypoint_vis": round(mean_vis, 4),
                "max_vert_vel": round(max_vert_vel, 6),
                "min_vert_vel": round(min_vert_vel, 6)
            })

    df_all = pd.DataFrame(window_predictions)

    # Output CSV paths
    res_dir = os.path.join(ROOT_DIR, "R&D", "ML_Baseline", "results")
    os.makedirs(res_dir, exist_ok=True)

    all_csv_path = os.path.join(res_dir, "yolo_tcn_window_predictions.csv")
    err_csv_path = os.path.join(res_dir, "yolo_tcn_error_analysis.csv")

    df_all.to_csv(all_csv_path, index=False)

    df_err = df_all[df_all["error_type"].isin(["FP", "FN"])].reset_index(drop=True)
    df_err.to_csv(err_csv_path, index=False)

    # ----------------------------------------------------------------------
    # Programmatic Verification Gate
    # ----------------------------------------------------------------------
    print("\n=== PROGRAMMATIC ERROR MATRIX VERIFICATION GATE ===")
    total_wins = len(df_all)
    tp_cnt = len(df_all[df_all["error_type"] == "TP"])
    tn_cnt = len(df_all[df_all["error_type"] == "TN"])
    fp_cnt = len(df_all[df_all["error_type"] == "FP"])
    fn_cnt = len(df_all[df_all["error_type"] == "FN"])
    err_cnt = len(df_err)

    print(f"Total Windows        : {total_wins} (Target: 1,396)")
    print(f"TP Count             : {tp_cnt} (Target: 300)")
    print(f"TN Count             : {tn_cnt} (Target: 1,006)")
    print(f"FP Count             : {fp_cnt} (Target: 59)")
    print(f"FN Count             : {fn_cnt} (Target: 31)")
    print(f"Total Error Windows  : {err_cnt} (Target: 90)")

    assert total_wins == 1396, f"Total windows mismatch: {total_wins}"
    assert tp_cnt == 300, f"TP mismatch: {tp_cnt}"
    assert tn_cnt == 1006, f"TN mismatch: {tn_cnt}"
    assert fp_cnt == 59, f"FP mismatch: {fp_cnt}"
    assert fn_cnt == 31, f"FN mismatch: {fn_cnt}"
    assert err_cnt == 90, f"Error count mismatch: {err_cnt}"

    print("\nPer-Fold Verification:")
    expected_fold_matrix = {
        1: {"loc": "Coffee_room_01", "tp": 162, "fp": 20, "tn": 310, "fn": 10},
        2: {"loc": "Coffee_room_02", "tp": 45,  "fp": 14, "tn": 349, "fn": 2},
        3: {"loc": "Home_01",        "tp": 73,  "fp": 14, "tn": 135, "fn": 17},
        4: {"loc": "Home_02",        "tp": 20,  "fp": 11, "tn": 212, "fn": 2}
    }

    for fold_num, exp in expected_fold_matrix.items():
        f_df = df_all[df_all["fold"] == fold_num]
        f_tp = len(f_df[f_df["error_type"] == "TP"])
        f_fp = len(f_df[f_df["error_type"] == "FP"])
        f_tn = len(f_df[f_df["error_type"] == "TN"])
        f_fn = len(f_df[f_df["error_type"] == "FN"])

        print(f"  Fold {fold_num} ({exp['loc']:15s}): TP={f_tp}/{exp['tp']}, FP={f_fp}/{exp['fp']}, TN={f_tn}/{exp['tn']}, FN={f_fn}/{exp['fn']} ", end="")
        if (f_tp, f_fp, f_tn, f_fn) == (exp['tp'], exp['fp'], exp['tn'], exp['fn']):
            print("[EXACT MATCH PASS]")
        else:
            print("[DISCREPANCY]")
            sys.exit(1)

    print("\n" + "=" * 70)
    print("EXPERIMENT J PHASE J1 EXTRACTION & VERIFICATION COMPLETE — 100% MATCH [PASS]")
    print("=" * 70)

if __name__ == "__main__":
    analyze_yolo_tcn_errors()
