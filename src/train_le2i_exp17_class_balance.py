"""
Experiment #17: Class Balancing & Oversampling Strategies Benchmark Pipeline.
Model: ModelK1_SpatialTCN (86,434 params, 187-D Input)

Variants Evaluated:
1. EXP17-A: K1 Control (Unweighted CE Loss, Standard DataLoader)
2. EXP17-B: Class-Weighted Loss (Weighted CE Loss, w_fall = N_norm / N_fall)
3. EXP17-C: Random Oversampling (Duplicates FALL samples in inner_train_df)
4. EXP17-D: Balanced Batch Sampler (WeightedRandomSampler with equal 50/50 class probability)

Safety & Protocol Rules:
- All balancing techniques applied STRICTLY to inner_train_df.
- Inner validation and outer test partitions remain 100% unweighted and un-oversampled.
- Threshold tau* tuned strictly on inner validation predictions.
- 4-Fold LOLO evaluation on unseen outer test locations.
"""

import os
import sys
import random
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT_DIR = r"d:\ONE_DATA\Fall detection"
sys.path.insert(0, ROOT_DIR)

from src.train_le2i_yolo_k1_spatial import ModelK1_SpatialTCN, YoloK1Dataset
from src.train_le2i_yolo_temporal import compute_metrics, find_best_threshold, set_seed

def train_le2i_exp17_class_balance():
    print("=" * 70)
    print("EXPERIMENT #17: CLASS BALANCING & OVERSAMPLING BENCHMARK")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    manifest_path = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "processed_pose_features_manifest.csv")
    df_manifest = pd.read_csv(manifest_path).sort_values("window_id").reset_index(drop=True)
    k1_dir = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "pose_estimator_features", "yolo_pose_k1")

    locations = ["Coffee_room_01", "Coffee_room_02", "Home_01", "Home_02"]
    
    ckpt_root = os.path.join(ROOT_DIR, "checkpoints", "le2i_exp17_class_balance")
    variant_dirs = {
        "exp17_a_control":          os.path.join(ckpt_root, "control"),
        "exp17_b_weighted_loss":    os.path.join(ckpt_root, "weighted_loss"),
        "exp17_c_oversampling":     os.path.join(ckpt_root, "oversampling"),
        "exp17_d_balanced_sampler": os.path.join(ckpt_root, "balanced_sampler")
    }
    for d in variant_dirs.values():
        os.makedirs(d, exist_ok=True)

    results_matrix = {
        "exp17_a_control":          {"name": "EXP17-A: K1 Control (Unweighted)",    "parameters": 86434, "folds": []},
        "exp17_b_weighted_loss":    {"name": "EXP17-B: Class-Weighted Loss",        "parameters": 86434, "folds": []},
        "exp17_c_oversampling":     {"name": "EXP17-C: Random Oversampling",         "parameters": 86434, "folds": []},
        "exp17_d_balanced_sampler": {"name": "EXP17-D: Balanced Batch Sampler",     "parameters": 86434, "folds": []}
    }

    variants = ["exp17_a_control", "exp17_b_weighted_loss", "exp17_c_oversampling", "exp17_d_balanced_sampler"]

    for var_key in variants:
        var_name = results_matrix[var_key]["name"]
        ckpt_dir = variant_dirs[var_key]

        print(f"\n" + "=" * 70)
        print(f"BENCHMARKING VARIANT: {var_name}")
        print("=" * 70)

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

            inner_train_df = train_val_df[train_val_df["event_id"].isin(tr_events)].reset_index(drop=True)
            inner_val_df   = train_val_df[train_val_df["event_id"].isin(val_events)].reset_index(drop=True)

            n_norm = (inner_train_df["label"].str.upper() == "NORMAL").sum()
            n_fall = (inner_train_df["label"].str.upper() == "FALL").sum()
            w_fall = float(n_norm) / float(n_fall) if n_fall > 0 else 1.0

            # Configure Dataset / Sampler / Loss for Variant
            if var_key == "exp17_c_oversampling":
                # Oversample FALL in inner_train_df
                train_norm_df = inner_train_df[inner_train_df["label"].str.upper() == "NORMAL"]
                train_fall_df = inner_train_df[inner_train_df["label"].str.upper() == "FALL"]
                
                # Sample with replacement to match length of normal samples
                oversampled_fall_df = train_fall_df.sample(n=len(train_norm_df), replace=True, random_state=42 + fold_idx)
                train_df_effective = pd.concat([train_norm_df, oversampled_fall_df]).sample(frac=1.0, random_state=42).reset_index(drop=True)
            else:
                train_df_effective = inner_train_df

            train_ds = YoloK1Dataset(train_df_effective, k1_dir)
            val_ds   = YoloK1Dataset(inner_val_df, k1_dir)
            test_ds  = YoloK1Dataset(test_df, k1_dir)

            if var_key == "exp17_d_balanced_sampler":
                labels = [1 if str(row["label"]).upper() in ["FALL", "1"] else 0 for _, row in train_df_effective.iterrows()]
                samples_weight = [1.0 / n_norm if l == 0 else 1.0 / n_fall for l in labels]
                sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight), replacement=True)
                train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler)
            else:
                train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

            val_loader  = DataLoader(val_ds, batch_size=32, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

            model = ModelK1_SpatialTCN().to(device)

            if var_key == "exp17_b_weighted_loss":
                class_weights = torch.tensor([1.0, w_fall], dtype=torch.float32).to(device)
                criterion = nn.CrossEntropyLoss(weight=class_weights)
            else:
                criterion = nn.CrossEntropyLoss()

            optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

            best_val_f1 = -1.0
            best_ckpt_path = os.path.join(ckpt_dir, f"fold_{fold_idx}_best.pth")

            for epoch in range(1, 101):
                model.train()
                for x_b, y_b in train_loader:
                    x_b, y_b = x_b.to(device), y_b.to(device)
                    optimizer.zero_grad()
                    out = model(x_b)
                    loss = criterion(out, y_b)
                    loss.backward()
                    optimizer.step()

                model.eval()
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
                v_metrics = compute_metrics(val_targets, val_probs, threshold=0.5)

                if v_metrics["f1"] > best_val_f1:
                    best_val_f1 = v_metrics["f1"]
                    torch.save(model.state_dict(), best_ckpt_path)

            # Evaluate Best Model
            model.load_state_dict(torch.load(best_ckpt_path))
            model.eval()

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
            tau_star, val_f1_star = find_best_threshold(val_targets, val_probs)

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

            m_def = compute_metrics(test_targets, test_probs, threshold=0.50)
            m_opt = compute_metrics(test_targets, test_probs, threshold=tau_star)

            print(f"   Fold {fold_idx} ({test_loc:15s}) | Tau=0.50: F1={m_def['f1']:.4f} | Tau*={tau_star:.2f}: F1={m_opt['f1']:.4f} | Rec={m_opt['recall']:.4f} | Spec={m_opt['specificity']:.4f}")

            results_matrix[var_key]["folds"].append({
                "fold": fold_idx,
                "test_location": test_loc,
                "tau_star": tau_star,
                "val_f1_star": val_f1_star,
                "f1_default": m_def["f1"],
                "f1_optimal": m_opt["f1"],
                "precision": m_opt["precision"],
                "recall": m_opt["recall"],
                "specificity": m_opt["specificity"],
                "roc_auc": m_opt["roc_auc"],
                "pr_auc": m_opt["pr_auc"],
                "tp": m_opt["tp"], "fp": m_opt["fp"], "tn": m_opt["tn"], "fn": m_opt["fn"]
            })

    # Summary Calculations
    print("\n" + "=" * 70)
    print("EXPERIMENT #17 CLASS BALANCING LOLO SUMMARY")
    print("=" * 70)

    for key, data in results_matrix.items():
        f1_opt_list = [r["f1_optimal"] for r in data["folds"]]
        f1_def_list = [r["f1_default"] for r in data["folds"]]
        data["lolo_f1_default_mean"] = float(np.mean(f1_def_list))
        data["lolo_f1_optimal_mean"] = float(np.mean(f1_opt_list))
        data["lolo_f1_optimal_std"]  = float(np.std(f1_opt_list))
        
        print(f"  {data['name']:40s} | F1 (@ 0.50): {data['lolo_f1_default_mean']*100:.2f}% | F1 (@ Tau*): {data['lolo_f1_optimal_mean']*100:.2f}% ± {data['lolo_f1_optimal_std']*100:.2f}%")

    res_dir = os.path.join(ROOT_DIR, "R&D", "ML_Baseline", "results")
    os.makedirs(res_dir, exist_ok=True)
    json_path = os.path.join(res_dir, "exp17_class_balance_results.json")
    with open(json_path, "w") as f:
        json.dump(results_matrix, f, indent=2)

    csv_rows = []
    for key, data in results_matrix.items():
        for r in data["folds"]:
            csv_rows.append({
                "variant_key": key,
                "variant_name": data["name"],
                "fold": r["fold"],
                "test_location": r["test_location"],
                "tau_star": r["tau_star"],
                "f1_default": r["f1_default"],
                "f1_optimal": r["f1_optimal"],
                "precision": r["precision"],
                "recall": r["recall"],
                "specificity": r["specificity"],
                "roc_auc": r["roc_auc"],
                "pr_auc": r["pr_auc"],
                "tp": r["tp"], "fp": r["fp"], "tn": r["tn"], "fn": r["fn"]
            })
    csv_path = os.path.join(res_dir, "exp17_class_balance_results.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    print(f"\nResults Saved: {json_path} & {csv_path}")

if __name__ == "__main__":
    train_le2i_exp17_class_balance()
