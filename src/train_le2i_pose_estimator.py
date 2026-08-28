"""
Experiment H Phase H2: Controlled 4-Fold LOLO Training Pipeline for Pose Estimators
(H1: MediaPipe Pose, H2: YOLO Pose, H3: RTMPose).

Scientific Protocol:
- Controlled Classifier Architecture: 21,314-parameter Pose+Velocity MLP Control
- Input Tensor Shape: (B, 50, 165)
- Outer Protocol: 4-Fold Leave-One-Location-Out (LOLO)
- Inner Protocol: 80/20 Event-Stratified Train/Validation Split for Early Stopping and tau* Tuning
- Device: PyTorch GPU CUDA (RTX 4060)

Outputs:
- checkpoints/le2i_pose_estimator/{mediapipe,yolo_pose,rtmpose}/fold_{1..4}_best.pth
- R&D/ML_Baseline/results/pose_estimator_benchmark_results.json
- R&D/ML_Baseline/results/pose_estimator_benchmark_results.csv
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

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ----------------------------------------------------------------------
# Controlled Downstream Classifier (21,314 Parameters)
# ----------------------------------------------------------------------
class PoseVelocityMLP(nn.Module):
    def __init__(self, in_dim=165, hidden_dim=64, num_classes=2, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dim * 2, hidden_dim) # 330 -> 64
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes) # 64 -> 2

    def forward(self, x):
        # x: (B, 50, 165)
        mean_feat = x.mean(dim=1)
        std_feat  = x.std(dim=1)
        pooled    = torch.cat([mean_feat, std_feat], dim=1) # (B, 330)
        h         = self.dropout(self.relu(self.fc1(pooled))) # (B, 64)
        out       = self.fc2(h) # (B, 2)
        return out

# ----------------------------------------------------------------------
# Dataset Class
# ----------------------------------------------------------------------
class PoseEstimatorDataset(Dataset):
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
            feat = d["features"].astype(np.float32) # (50, 165)
        
        return torch.tensor(feat, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def compute_metrics(y_true, y_probs, threshold=0.5):
    y_pred = (y_probs >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_true, y_probs)
    except Exception:
        roc_auc = 0.5
        
    try:
        pr_auc = average_precision_score(y_true, y_probs)
    except Exception:
        pr_auc = 0.0

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "f1": float(f1),
        "precision": float(p),
        "recall": float(r),
        "specificity": float(spec),
        "sensitivity": float(sens),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
    }

def find_best_threshold(y_true, y_probs):
    thresholds = np.linspace(0.1, 0.9, 81)
    best_tau = 0.5
    best_f1 = -1.0
    
    for tau in thresholds:
        m = compute_metrics(y_true, y_probs, threshold=tau)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_tau = tau
            
    return best_tau, best_f1

def train_le2i_pose_estimator():
    print("=" * 70)
    print("EXPERIMENT H PHASE H2: CONTROLLED POSE ESTIMATOR 4-FOLD LOLO TRAINING")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    manifest_path = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "processed_pose_features_manifest.csv")
    df_manifest = pd.read_csv(manifest_path).sort_values("window_id").reset_index(drop=True)

    estimators = {
        "mediapipe": {"name": "H1: MediaPipe Pose", "dir": os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "pose_estimator_features", "mediapipe")},
        "yolo_pose": {"name": "H2: YOLO Pose",     "dir": os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "pose_estimator_features", "yolo_pose")},
        "rtmpose":   {"name": "H3: RTMPose",       "dir": os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "pose_estimator_features", "rtmpose")}
    }

    locations = ["Coffee_room_01", "Coffee_room_02", "Home_01", "Home_02"]
    all_benchmark_results = {}

    for est_key, est_meta in estimators.items():
        print("\n" + "=" * 70)
        print(f"TRAINING 4-FOLD LOLO BENCHMARK FOR {est_meta['name'].upper()}")
        print("=" * 70)

        ckpt_dir = os.path.join(ROOT_DIR, "checkpoints", "le2i_pose_estimator", est_key)
        os.makedirs(ckpt_dir, exist_ok=True)

        fold_results = []

        for fold_idx, test_loc in enumerate(locations, 1):
            set_seed(42 + fold_idx)

            test_df = df_manifest[df_manifest["location"] == test_loc].reset_index(drop=True)
            train_val_df = df_manifest[df_manifest["location"] != test_loc].reset_index(drop=True)

            # Event-stratified inner train/val split
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

            # DataLoaders
            train_ds = PoseEstimatorDataset(inner_train_df, est_meta["dir"])
            val_ds   = PoseEstimatorDataset(inner_val_df, est_meta["dir"])
            test_ds  = PoseEstimatorDataset(test_df, est_meta["dir"])

            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
            val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)
            test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)

            model = PoseVelocityMLP().to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

            best_val_f1 = -1.0
            best_ckpt_path = os.path.join(ckpt_dir, f"fold_{fold_idx}_best.pth")

            # Training Loop
            for epoch in range(1, 101):
                model.train()
                for x_b, y_b in train_loader:
                    x_b, y_b = x_b.to(device), y_b.to(device)
                    optimizer.zero_grad()
                    out = model(x_b)
                    loss = criterion(out, y_b)
                    loss.backward()
                    optimizer.step()

                # Inner Validation Evaluation
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
                    torch.save(model.state_state_dict() if hasattr(model, 'state_state_dict') else model.state_dict(), best_ckpt_path)

            # Load Best Checkpoint for Outer Test Evaluation
            model.load_state_dict(torch.load(best_ckpt_path))
            model.eval()

            # Tune tau* on Inner Validation Set
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

            # Outer Test Inference
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

            # Evaluate at default 0.50 and optimal tau*
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

        # Summarize Estimator Performance across 4 Folds
        lolo_f1_def = np.mean([r["f1_default"] for r in fold_results])
        lolo_f1_opt = np.mean([r["f1_optimal"] for r in fold_results])
        lolo_f1_std = np.std([r["f1_optimal"] for r in fold_results])

        print(f"\n   [{est_meta['name']}] LOLO Summary:")
        print(f"     - Mean LOLO F1 (@ 0.50)   : {lolo_f1_def * 100:.2f}%")
        print(f"     - Mean LOLO F1 (@ Tau*)   : {lolo_f1_opt * 100:.2f}% ± {lolo_f1_std * 100:.2f}%")

        all_benchmark_results[est_key] = {
            "name": est_meta["name"],
            "lolo_f1_default_mean": float(lolo_f1_def),
            "lolo_f1_optimal_mean": float(lolo_f1_opt),
            "lolo_f1_optimal_std": float(lolo_f1_std),
            "folds": fold_results
        }

    # Save Machine-Readable Results
    res_dir = os.path.join(ROOT_DIR, "R&D", "ML_Baseline", "results")
    os.makedirs(res_dir, exist_ok=True)
    json_path = os.path.join(res_dir, "pose_estimator_benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(all_benchmark_results, f, indent=2)

    # Save CSV Summary
    csv_rows = []
    for est_key, est_res in all_benchmark_results.items():
        for r in est_res["folds"]:
            csv_rows.append({
                "estimator": est_res["name"],
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
    csv_path = os.path.join(res_dir, "pose_estimator_benchmark_results.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    print("\n" + "=" * 70)
    print("EXPERIMENT H PHASE H2 TRAINING COMPLETE — RESULTS SAVED")
    print("=" * 70)

if __name__ == "__main__":
    train_le2i_pose_estimator()
