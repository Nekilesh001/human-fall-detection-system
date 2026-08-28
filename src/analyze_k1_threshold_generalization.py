"""
Experiment #19: Leakage-Free Decision Threshold Generalization Analysis for Model K1.
Read-Only analysis of frozen K1 SOTA checkpoints (checkpoints/le2i_yolo_k1/fold_{1..4}_best.pth).

Protocol:
1. Reconstruct exact inner validation split (20% event-stratified from outer train) per fold.
2. Evaluate P(FALL) strictly on inner validation set.
3. Sweep tau in [0.30, 0.90] in increments of 0.01 to find tau*_inner maximizing inner val F1.
4. Deterministic tie-breaker: Select threshold closest to 0.50 if F1 ties.
5. Freeze tau*_inner and evaluate ONCE on unseen outer test location.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

ROOT_DIR = r"d:\ONE_DATA\Fall detection"
sys.path.insert(0, ROOT_DIR)

from src.train_le2i_yolo_k1_spatial import ModelK1_SpatialTCN, YoloK1Dataset
from src.train_le2i_yolo_temporal import compute_metrics, set_seed

def select_inner_val_threshold(val_targets, val_probs):
    thresholds = np.linspace(0.30, 0.90, 601) # Increments of 0.001 / 0.01
    best_f1 = -1.0
    best_tau = 0.50
    candidates = []

    for tau in thresholds:
        preds = (val_probs >= tau).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(val_targets, preds, average='binary', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            candidates = [(tau, f1)]
        elif abs(f1 - best_f1) < 1e-6:
            candidates.append((tau, f1))

    # Tie-breaker: Select threshold closest to 0.50
    candidates.sort(key=lambda x: (abs(x[0] - 0.50), x[0]))
    best_tau = candidates[0][0]
    return float(best_tau), float(best_f1)

def analyze_k1_threshold_generalization():
    print("=" * 70)
    print("EXPERIMENT #19: LEAKAGE-FREE THRESHOLD GENERALIZATION ANALYSIS")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    manifest_path = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "processed_pose_features_manifest.csv")
    df_manifest = pd.read_csv(manifest_path).sort_values("window_id").reset_index(drop=True)
    k1_dir = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "pose_estimator_features", "yolo_pose_k1")
    ckpt_dir = os.path.join(ROOT_DIR, "checkpoints", "le2i_yolo_k1")

    locations = ["Coffee_room_01", "Coffee_room_02", "Home_01", "Home_02"]

    fold_results = []
    selected_taus = []
    csv_rows = []

    print("\nPhase 19A & 19B: Selecting Thresholds Strictly from Inner Validation Predictions...\n")

    for fold_idx, test_loc in enumerate(locations, 1):
        set_seed(42 + fold_idx)

        test_df = df_manifest[df_manifest["location"] == test_loc].reset_index(drop=True)
        train_val_df = df_manifest[df_manifest["location"] != test_loc].reset_index(drop=True)

        unique_events = train_val_df["event_id"].unique()
        event_labels = [1 if str(train_val_df[train_val_df["event_id"] == ev]["label"].iloc[0]).upper() in ["FALL", "1"] else 0 for ev in unique_events]
        
        try:
            tr_events, val_events = train_test_split(unique_events, test_size=0.20, random_state=42, stratify=event_labels)
        except ValueError:
            tr_events, val_events = train_test_split(unique_events, test_size=0.20, random_state=42)

        inner_val_df = train_val_df[train_val_df["event_id"].isin(val_events)].reset_index(drop=True)

        val_ds  = YoloK1Dataset(inner_val_df, k1_dir)
        test_ds = YoloK1Dataset(test_df, k1_dir)

        val_loader  = DataLoader(val_ds, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

        ckpt_path = os.path.join(ckpt_dir, f"fold_{fold_idx}_best.pth")
        assert os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}"

        model = ModelK1_SpatialTCN().to(device)
        model.load_state_dict(torch.load(ckpt_path))
        model.eval()

        # 1. Inner Validation Predictions (LEAKAGE-FREE THRESHOLD TUNING)
        val_probs, val_targets = [], []
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b = x_b.to(device)
                out = model(x_b)
                probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                val_probs.extend(probs)
                val_targets.extend(y_b.numpy())

        val_probs = np.array(val_probs)
        val_targets = np.array(val_targets)
        
        tau_inner, val_f1_inner = select_inner_val_threshold(val_targets, val_probs)
        selected_taus.append(tau_inner)

        # 2. Outer Test Predictions (EVALUATED ONCE WITH FROZEN tau_inner)
        test_probs, test_targets = [], []
        with torch.no_grad():
            for x_b, y_b in test_loader:
                x_b = x_b.to(device)
                out = model(x_b)
                probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                test_probs.extend(probs)
                test_targets.extend(y_b.numpy())

        test_probs = np.array(test_probs)
        test_targets = np.array(test_targets)

        m_opt = compute_metrics(test_targets, test_probs, threshold=tau_inner)
        m_def = compute_metrics(test_targets, test_probs, threshold=0.50)
        m_55  = compute_metrics(test_targets, test_probs, threshold=0.55)

        acc = float(np.mean((test_probs >= tau_inner).astype(int) == test_targets))

        print(f"  Fold {fold_idx} ({test_loc:15s}) | Selected Tau*_inner = {tau_inner:.4f} (Val F1={val_f1_inner:.4f})")
        print(f"    -> Outer Test @ Tau*_inner : F1={m_opt['f1']:.4f} | Rec={m_opt['recall']:.4f} | Spec={m_opt['specificity']:.4f} | FP={m_opt['fp']} | FN={m_opt['fn']}")
        print(f"    -> Outer Test @ Tau=0.50   : F1={m_def['f1']:.4f} | Rec={m_def['recall']:.4f} | Spec={m_def['specificity']:.4f} | FP={m_def['fp']} | FN={m_def['fn']}")
        print(f"    -> Outer Test @ Tau=0.55   : F1={m_55['f1']:.4f} | Rec={m_55['recall']:.4f} | Spec={m_55['specificity']:.4f} | FP={m_55['fp']} | FN={m_55['fn']}\n")

        fold_results.append({
            "fold": fold_idx,
            "test_location": test_loc,
            "tau_inner_selected": tau_inner,
            "val_f1_inner": val_f1_inner,
            "f1_test_selected": m_opt["f1"],
            "f1_test_default_050": m_def["f1"],
            "f1_test_sweep_055": m_55["f1"],
            "precision": m_opt["precision"],
            "recall": m_opt["recall"],
            "specificity": m_opt["specificity"],
            "accuracy": acc,
            "tp": m_opt["tp"], "fp": m_opt["fp"], "tn": m_opt["tn"], "fn": m_opt["fn"]
        })

        csv_rows.append({
            "fold": fold_idx,
            "test_location": test_loc,
            "tau_inner_selected": tau_inner,
            "val_f1_inner": val_f1_inner,
            "f1_test_selected": m_opt["f1"],
            "f1_test_default_050": m_def["f1"],
            "f1_test_sweep_055": m_55["f1"],
            "precision": m_opt["precision"],
            "recall": m_opt["recall"],
            "specificity": m_opt["specificity"],
            "accuracy": acc,
            "tp": m_opt["tp"], "fp": m_opt["fp"], "tn": m_opt["tn"], "fn": m_opt["fn"]
        })

    # Summary Statistics
    mean_tau_selected = float(np.mean(selected_taus))
    std_tau_selected  = float(np.std(selected_taus))

    lolo_f1_selected = float(np.mean([r["f1_test_selected"] for r in fold_results]))
    lolo_std_selected = float(np.std([r["f1_test_selected"] for r in fold_results]))

    lolo_f1_def050 = float(np.mean([r["f1_test_default_050"] for r in fold_results]))
    lolo_f1_swp055 = float(np.mean([r["f1_test_sweep_055"] for r in fold_results]))

    print("=" * 70)
    print("EXPERIMENT #19 SUMMARY & GENERALIZATION VERDICT")
    print("=" * 70)
    print(f"  - Inner-Val Selected Thresholds Across Folds : {[round(t, 4) for t in selected_taus]}")
    print(f"  - Mean / Std Selected Threshold (Tau*_inner) : {mean_tau_selected:.4f} ± {std_tau_selected:.4f}")
    print(f"  - Generalization Mean LOLO F1 (@ Tau*_inner) : {lolo_f1_selected*100:.2f}% ± {lolo_std_selected*100:.2f}%")
    print(f"  - Baseline Mean LOLO F1 (@ Tau=0.50)         : {lolo_f1_def050*100:.2f}%")
    print(f"  - Outer-Sweep Mean LOLO F1 (@ Tau=0.55)      : {lolo_f1_swp055*100:.2f}%")

    summary_json = {
        "experiment": "Experiment #19: K1 Threshold Generalization Analysis",
        "inner_selected_thresholds": selected_taus,
        "mean_tau_selected": mean_tau_selected,
        "std_tau_selected": std_tau_selected,
        "lolo_f1_generalization_mean": lolo_f1_selected,
        "lolo_f1_generalization_std": lolo_std_selected,
        "lolo_f1_default_050_mean": lolo_f1_def050,
        "lolo_f1_sweep_055_mean": lolo_f1_swp055,
        "fold_details": fold_results
    }

    res_dir = os.path.join(ROOT_DIR, "R&D", "ML_Baseline", "results")
    os.makedirs(res_dir, exist_ok=True)

    json_path = os.path.join(res_dir, "exp19_threshold_generalization.json")
    with open(json_path, "w") as f:
        json.dump(summary_json, f, indent=2)

    csv_path = os.path.join(res_dir, "exp19_threshold_generalization.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    print(f"\nResults Saved: {json_path} & {csv_path}")

if __name__ == "__main__":
    analyze_k1_threshold_generalization()
