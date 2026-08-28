"""
Experiment #18: K1 Decision Threshold & Operating Point Robustness Analysis.
Read-Only analysis of frozen K1 SOTA checkpoints (checkpoints/le2i_yolo_k1/fold_{1..4}_best.pth).

Inputs:
- processed_data/Le2i_baseline/processed_pose_features_manifest.csv
- processed_data/Le2i_baseline/pose_estimator_features/yolo_pose_k1/
- checkpoints/le2i_yolo_k1/fold_{1..4}_best.pth

Outputs:
- R&D/ML_Baseline/results/exp18_k1_threshold_sweep.json
- R&D/ML_Baseline/results/exp18_k1_threshold_sweep.csv
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

from src.train_le2i_yolo_k1_spatial import ModelK1_SpatialTCN, YoloK1Dataset
from src.train_le2i_yolo_temporal import compute_metrics, set_seed

def analyze_k1_threshold_robustness():
    print("=" * 70)
    print("EXPERIMENT #18: K1 DECISION THRESHOLD ROBUSTNESS ANALYSIS")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    manifest_path = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "processed_pose_features_manifest.csv")
    df_manifest = pd.read_csv(manifest_path).sort_values("window_id").reset_index(drop=True)
    k1_dir = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "pose_estimator_features", "yolo_pose_k1")
    ckpt_dir = os.path.join(ROOT_DIR, "checkpoints", "le2i_yolo_k1")

    locations = ["Coffee_room_01", "Coffee_room_02", "Home_01", "Home_02"]
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

    # Collect predictions for each fold
    fold_probs_targets = []

    for fold_idx, test_loc in enumerate(locations, 1):
        set_seed(42 + fold_idx)
        test_df = df_manifest[df_manifest["location"] == test_loc].reset_index(drop=True)
        test_ds = YoloK1Dataset(test_df, k1_dir)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

        ckpt_path = os.path.join(ckpt_dir, f"fold_{fold_idx}_best.pth")
        assert os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}"

        model = ModelK1_SpatialTCN().to(device)
        model.load_state_dict(torch.load(ckpt_path))
        model.eval()

        test_probs, test_targets = [], []
        with torch.no_grad():
            for x_b, y_b in test_loader:
                x_b = x_b.to(device)
                out = model(x_b)
                probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                test_probs.extend(probs)
                test_targets.extend(y_b.numpy())

        fold_probs_targets.append({
            "fold": fold_idx,
            "test_location": test_loc,
            "probs": np.array(test_probs),
            "targets": np.array(test_targets)
        })

    # Execute Threshold Sweep
    sweep_results_by_tau = {}
    csv_rows = []

    print("\nExecuting Threshold Sweep across 13 Decision Thresholds...\n")
    print(f" {'Tau':>6s} | {'Mean F1':>8s} | {'F1 Std':>8s} | {'Mean Rec':>8s} | {'Mean Spec':>9s} | {'Total FP':>8s} | {'Total FN':>8s}")
    print("-" * 72)

    for tau in thresholds:
        tau_str = f"{tau:.2f}"
        fold_metrics = []
        tot_tp, tot_fp, tot_tn, tot_fn = 0, 0, 0, 0

        for item in fold_probs_targets:
            f_idx = item["fold"]
            loc = item["test_location"]
            m = compute_metrics(item["targets"], item["probs"], threshold=tau)
            
            acc = float(np.mean((item["probs"] >= tau).astype(int) == item["targets"]))
            
            m["accuracy"] = acc
            m["fold"] = f_idx
            m["test_location"] = loc
            fold_metrics.append(m)

            tot_tp += m["tp"]
            tot_fp += m["fp"]
            tot_tn += m["tn"]
            tot_fn += m["fn"]

            csv_rows.append({
                "threshold": tau,
                "fold": f_idx,
                "test_location": loc,
                "f1": m["f1"],
                "precision": m["precision"],
                "recall": m["recall"],
                "specificity": m["specificity"],
                "accuracy": acc,
                "tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"]
            })

        mean_f1   = float(np.mean([fm["f1"] for fm in fold_metrics]))
        std_f1    = float(np.std([fm["f1"] for fm in fold_metrics]))
        mean_prec = float(np.mean([fm["precision"] for fm in fold_metrics]))
        mean_rec  = float(np.mean([fm["recall"] for fm in fold_metrics]))
        mean_spec = float(np.mean([fm["specificity"] for fm in fold_metrics]))
        mean_acc  = float(np.mean([fm["accuracy"] for fm in fold_metrics]))

        sweep_results_by_tau[tau_str] = {
            "threshold": tau,
            "lolo_f1_mean": mean_f1,
            "lolo_f1_std": std_f1,
            "lolo_precision_mean": mean_prec,
            "lolo_recall_mean": mean_rec,
            "lolo_specificity_mean": mean_spec,
            "lolo_accuracy_mean": mean_acc,
            "total_tp": tot_tp,
            "total_fp": tot_fp,
            "total_tn": tot_tn,
            "total_fn": tot_fn,
            "fold_details": fold_metrics
        }

        print(f" {tau:6.2f} | {mean_f1*100:7.2f}% | ±{std_f1*100:5.2f}% | {mean_rec*100:7.2f}% | {mean_spec*100:8.2f}% | {tot_fp:8d} | {tot_fn:8d}")

    # Find Optimal Operating Points
    best_f1_tau = max(sweep_results_by_tau.items(), key=lambda x: x[1]["lolo_f1_mean"])
    high_rec_tau = max(sweep_results_by_tau.items(), key=lambda x: (x[1]["lolo_recall_mean"] >= 0.95, x[1]["lolo_f1_mean"]))

    print("\n" + "=" * 70)
    print("EXPERIMENT #18 OPERATING POINT AUDIT SUMMARY")
    print("=" * 70)
    print(f"  - Peak Mean F1 Threshold (@ Tau={best_f1_tau[0]}): {best_f1_tau[1]['lolo_f1_mean']*100:.2f}% ± {best_f1_tau[1]['lolo_f1_std']*100:.2f}%")
    print(f"  - High-Recall Operating Point (@ Tau={high_rec_tau[0]}): Recall={high_rec_tau[1]['lolo_recall_mean']*100:.2f}%, F1={high_rec_tau[1]['lolo_f1_mean']*100:.2f}%")

    res_dir = os.path.join(ROOT_DIR, "R&D", "ML_Baseline", "results")
    os.makedirs(res_dir, exist_ok=True)

    json_path = os.path.join(res_dir, "exp18_k1_threshold_sweep.json")
    with open(json_path, "w") as f:
        json.dump(sweep_results_by_tau, f, indent=2)

    csv_path = os.path.join(res_dir, "exp18_k1_threshold_sweep.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    print(f"\nResults Saved: {json_path} & {csv_path}")

if __name__ == "__main__":
    analyze_k1_threshold_robustness()
