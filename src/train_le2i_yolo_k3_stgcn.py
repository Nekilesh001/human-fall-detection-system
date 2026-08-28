"""
Experiment K Phase K3: Spatial-Temporal Graph Convolutional Network (ST-GCN) Pipeline.
Model: ModelK3_STGCN (107,778 params)

Inputs:
- Dynamic conversion from processed_data/Le2i_baseline/pose_estimator_features/yolo_pose/
- Input Tensor Shape per Window: (5, 50, 17) float32 [Channels: X, Y, V, dX, dY]

Outputs:
- checkpoints/le2i_yolo_k3_stgcn/fold_{1..4}_best.pth
- R&D/ML_Baseline/results/yolo_k3_stgcn_benchmark_results.json
- R&D/ML_Baseline/results/yolo_k3_stgcn_benchmark_results.csv
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

from src.train_le2i_yolo_temporal import compute_metrics, find_best_threshold, set_seed

COCO_POPULATED_CANONICAL_INDICES = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# ----------------------------------------------------------------------
# ST-GCN Graph Adjacency Formulation (COCO-17, 3 Partitions)
# ----------------------------------------------------------------------
def build_coco_stgcn_adjacency():
    V = 17
    edges = [
        (0,1), (0,2), (1,3), (2,4),          # Head
        (5,6), (5,7), (7,9), (6,8), (8,10),  # Arms / Shoulders
        (5,11), (6,12), (11,12),             # Torso
        (11,13), (13,15), (12,14), (14,16)   # Legs / Hips
    ]
    A = np.zeros((3, V, V), dtype=np.float32)
    
    # Partition 0: Self-loops
    for i in range(V):
        A[0, i, i] = 1.0

    # Partitions 1 and 2: Centripetal / Centrifugal based on distance to hip center (11, 12)
    hip_dist = [0, 1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2]

    for i, j in edges:
        if hip_dist[i] < hip_dist[j]:
            A[1, i, j] = 1.0
            A[2, j, i] = 1.0
        elif hip_dist[i] > hip_dist[j]:
            A[2, i, j] = 1.0
            A[1, j, i] = 1.0
        else:
            A[1, i, j] = 1.0
            A[1, j, i] = 1.0

    for k in range(3):
        deg = np.sum(A[k], axis=1)
        deg[deg == 0] = 1.0
        A[k] = A[k] / deg[:, None]
        
    return torch.tensor(A, dtype=torch.float32)

class SpatialGraphConv(nn.Module):
    def __init__(self, in_c, out_c, K=3):
        super().__init__()
        self.K = K
        self.conv = nn.Conv2d(in_c, out_c * K, kernel_size=1)
        
    def forward(self, x, A_tensor):
        # x: (B, in_c, T, V), A_tensor: (3, V, V)
        N, C, T, V = x.size()
        h = self.conv(x).view(N, self.K, -1, T, V)
        out = torch.einsum('nkctv,kvw->nctw', h, A_tensor)
        return out

class STGCNBlock(nn.Module):
    def __init__(self, in_c, out_c, A_tensor, stride=1, dropout=0.2):
        super().__init__()
        self.register_buffer('A', A_tensor)
        self.gconv = SpatialGraphConv(in_c, out_c)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
        self.tconv = nn.Conv2d(out_c, out_c, kernel_size=(9, 1), padding=(4, 0), stride=(stride, 1))
        self.bn2 = nn.BatchNorm2d(out_c)
        self.drop = nn.Dropout(dropout)
        
        if in_c != out_c or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_c)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        out = self.relu(self.bn1(self.gconv(x, self.A)))
        out = self.drop(self.bn2(self.tconv(out)))
        return self.relu(out + res)

class ModelK3_STGCN(nn.Module):
    """K3: Spatial-Temporal Graph Convolutional Network (107,778 params)"""
    def __init__(self, in_channels=5, num_classes=2, dropout_p=0.5):
        super().__init__()
        A_tensor = build_coco_stgcn_adjacency()
        self.b1 = STGCNBlock(in_channels, 32, A_tensor)
        self.b2 = STGCNBlock(32, 64, A_tensor)
        self.b3 = STGCNBlock(64, 64, A_tensor)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        # x: (B, 5, 50, 17)
        h = self.b1(x)
        h = self.b2(h)
        h = self.b3(h)
        pooled = self.pool(h).view(h.size(0), -1)
        out = self.dropout(self.relu(self.fc1(pooled)))
        return self.fc2(out)

class YoloK3Dataset(Dataset):
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
            feat_165 = d["features"].astype(np.float32) # (50, 165)
        
        # Slicing (5, 50, 17) float32
        T = feat_165.shape[0]
        V = len(COCO_POPULATED_CANONICAL_INDICES)
        stgcn_tensor = np.zeros((5, T, V), dtype=np.float32)

        for c_idx, can_idx in enumerate(COCO_POPULATED_CANONICAL_INDICES):
            stgcn_tensor[0, :, c_idx] = feat_165[:, can_idx * 3]     # X
            stgcn_tensor[1, :, c_idx] = feat_165[:, can_idx * 3 + 1] # Y
            stgcn_tensor[2, :, c_idx] = feat_165[:, can_idx * 3 + 2] # V
            stgcn_tensor[3, :, c_idx] = feat_165[:, 99 + can_idx * 2]     # dX
            stgcn_tensor[4, :, c_idx] = feat_165[:, 99 + can_idx * 2 + 1] # dY
        
        return torch.tensor(stgcn_tensor, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def train_le2i_yolo_k3_stgcn():
    print("=" * 70)
    print("EXPERIMENT K PHASE K3: ST-GCN GRAPH MODEL BENCHMARK")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    manifest_path = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "processed_pose_features_manifest.csv")
    df_manifest = pd.read_csv(manifest_path).sort_values("window_id").reset_index(drop=True)
    yolo_dir = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "pose_estimator_features", "yolo_pose")

    locations = ["Coffee_room_01", "Coffee_room_02", "Home_01", "Home_02"]
    ckpt_dir = os.path.join(ROOT_DIR, "checkpoints", "le2i_yolo_k3_stgcn")
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

        train_ds = YoloK3Dataset(inner_train_df, yolo_dir)
        val_ds   = YoloK3Dataset(inner_val_df, yolo_dir)
        test_ds  = YoloK3Dataset(test_df, yolo_dir)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)
        test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)

        model = ModelK3_STGCN().to(device)
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

    print(f"\n   [Model K3: ST-GCN Graph Model] LOLO Summary:")
    print(f"     - Mean LOLO F1 (@ 0.50)   : {lolo_f1_def * 100:.2f}%")
    print(f"     - Mean LOLO F1 (@ Tau*)   : {lolo_f1_opt * 100:.2f}% ± {lolo_f1_std * 100:.2f}%")

    k3_benchmark_results = {
        "k3_stgcn": {
            "name": "Model K3: ST-GCN Graph Model",
            "parameters": 107778,
            "lolo_f1_default_mean": float(lolo_f1_def),
            "lolo_f1_optimal_mean": float(lolo_f1_opt),
            "lolo_f1_optimal_std": float(lolo_f1_std),
            "folds": fold_results
        }
    }

    res_dir = os.path.join(ROOT_DIR, "R&D", "ML_Baseline", "results")
    os.makedirs(res_dir, exist_ok=True)
    json_path = os.path.join(res_dir, "yolo_k3_stgcn_benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(k3_benchmark_results, f, indent=2)

    csv_rows = []
    for r in fold_results:
        csv_rows.append({
            "model": "Model K3: ST-GCN Graph Model",
            "parameters": 107778,
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
    csv_path = os.path.join(res_dir, "yolo_k3_stgcn_benchmark_results.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    print("\n" + "=" * 70)
    print("EXPERIMENT K PHASE K3 TRAINING COMPLETE — RESULTS SAVED")
    print("=" * 70)

if __name__ == "__main__":
    train_le2i_yolo_k3_stgcn()
