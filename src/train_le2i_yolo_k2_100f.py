"""
Experiment K Phase K2: 100-Frame Temporal Context YOLO Pose 1D TCN Benchmark Pipeline.
Model: ModelK2_100fTCN (83,618 params)

Inputs:
- Manifest: processed_data/Le2i_baseline/processed_pose_100f_manifest.csv (1,142 rows)
- Feature Directory: processed_data/Le2i_baseline/pose_estimator_features/yolo_pose_100f/ ((100, 165) float32)

Outputs:
- checkpoints/le2i_yolo_k2_100f/fold_{1..4}_best.pth
- R&D/ML_Baseline/results/yolo_k2_100f_benchmark_results.json
- R&D/ML_Baseline/results/yolo_k2_100f_benchmark_results.csv
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
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT_DIR = r"d:\ONE_DATA\Fall detection"
sys.path.insert(0, ROOT_DIR)

from src.train_le2i_yolo_temporal import ModelI3_TCN, compute_metrics, find_best_threshold, set_seed

class YoloK2Dataset(Dataset):
    def __init__(self, df, feature_dir):
        self.df = df.reset_index(drop=True)
        self.feature_dir = feature_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wid = row["window_id"]
        raw_label = str(row["label"]).upper()
        label = 1 if raw_label in ["FALL", "1"] else 0
        fpath = os.path.join(self.feature_dir, f"{wid}.npz")
        
        with np.load(fpath) as d:
            feat = d["features"].astype(np.float32) # (100, 165)
        
        return torch.tensor(feat, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def train_le2i_yolo_k2_100f():
    print("=" * 70)
    print("EXPERIMENT K PHASE K2: 100-FRAME TEMPORAL TCN BENCHMARK")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    manifest_path = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "processed_pose_100f_manifest.csv")
    df_manifest = pd.read_csv(manifest_path).sort_values("window_id").reset_index(drop=True)
    k2_dir = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "pose_estimator_features", "yolo_pose_100f")

    locations = ["Coffee_room_01", "Coffee_room_02", "Home_01", "Home_02"]
    ckpt_dir = os.path.join(ROOT_DIR, "checkpoints", "le2i_yolo_k2_100f")
    os.makedirs(ckpt_dir, exist_ok=True)

    fold_results = []

    for fold_idx, test_loc in enumerate(locations, 1):
        set_seed(42 + fold_idx)

        test_df = df_manifest[df_manifest["location"] == test_loc].reset_index(drop=True)
        train_val_df = df_manifest[df_manifest["location"] != test_loc].reset_index(drop=True)

        unique_events = train_val_df["event_id"].unique()
        event_labels = [1 if str(train_val_df[train_val_df["event_id"] == ev]["label"].iloc[0]).upper() in ["FALL", "1"] else 0 for ev in unique_events]
        
        try:
            tr_events, val_events = train_test_split(
                unique_events, test_size=0.20, random_state=42, stratify=event_labels
            )
        except ValueError:
            tr_events, val_events = train_test_split(
                unique_events, test_size=0.20, random_state=42
            )

        inner_train_df = train_val_df[train_val_df["event_id"].isin(tr_events)].reset_index(drop=True)
        inner_val_df   = train_val_df[train_val_df["event_id"].isin(val_events)].reset_index(drop=True)

        train_ds = YoloK2Dataset(inner_train_df, k2_dir)
        val_ds   = YoloK2Dataset(inner_val_df, k2_dir)
        test_ds  = YoloK2Dataset(test_df, k2_dir)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)
        test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)

        model = ModelI3_TCN().to(device) # Base 1D TCN (83,618 params)
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

        fold_results.append({
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

    lolo_f1_def = np.mean([r["f1_default"] for r in fold_results])
    lolo_f1_opt = np.mean([r["f1_optimal"] for r in fold_results])
    lolo_f1_std = np.std([r["f1_optimal"] for r in fold_results])

    print(f"\n   [Model K2: 100-Frame Temporal TCN] LOLO Summary:")
    print(f"     - Mean LOLO F1 (@ 0.50)   : {lolo_f1_def * 100:.2f}%")
    print(f"     - Mean LOLO F1 (@ Tau*)   : {lolo_f1_opt * 100:.2f}% ± {lolo_f1_std * 100:.2f}%")

    k2_benchmark_results = {
        "k2_100f_tcn": {
            "name": "Model K2: 100-Frame Temporal TCN",
            "parameters": 83618,
            "lolo_f1_default_mean": float(lolo_f1_def),
            "lolo_f1_optimal_mean": float(lolo_f1_opt),
            "lolo_f1_optimal_std": float(lolo_f1_std),
            "folds": fold_results
        }
    }

    res_dir = os.path.join(ROOT_DIR, "R&D", "ML_Baseline", "results")
    os.makedirs(res_dir, exist_ok=True)
    json_path = os.path.join(res_dir, "yolo_k2_100f_benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(k2_benchmark_results, f, indent=2)

    csv_rows = []
    for r in fold_results:
        csv_rows.append({
            "model": "Model K2: 100-Frame Temporal TCN",
            "parameters": 83618,
            "fold": r["fold"],
            "test_location": r["test_location"],
            "tau_star": r["tau_star"],
            "f1_default": r["f1_default"],
            "f1_optimal": r["f1_optimal"],
            "precision": r["precision"],
            "recall": r["recall"],
            "specificity": r["specificity"],
            "roc_auc": r["roc_auc"],
            "pr_auc": r["pr_auc"]
        })
    csv_path = os.path.join(res_dir, "yolo_k2_100f_benchmark_results.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    print("\n" + "=" * 70)
    print("EXPERIMENT K PHASE K2 TRAINING COMPLETE — RESULTS SAVED")
    print("=" * 70)

if __name__ == "__main__":
    train_le2i_yolo_k2_100f()
