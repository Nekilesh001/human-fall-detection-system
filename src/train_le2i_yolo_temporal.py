"""
Experiment I: Controlled YOLO Pose Temporal Architecture Benchmark Pipeline.
Models Benchmarked:
- I0: YOLO Pose Control MLP (21,314 params)
- I1: YOLO Pose 1-Layer GRU (46,498 params)
- I2: YOLO Pose 1-Layer LSTM (61,282 params)
- I3: YOLO Pose 1D TCN (83,618 params)
- I4: YOLO Pose 1-Layer Transformer (46,242 params)

Outputs:
- checkpoints/le2i_yolo_temporal/{control,gru,lstm,tcn,transformer}/fold_{1..4}_best.pth
- R&D/ML_Baseline/results/yolo_temporal_benchmark_results.json
- R&D/ML_Baseline/results/yolo_temporal_benchmark_results.csv
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
# Model Architectures
# ----------------------------------------------------------------------
class ModelI0_MLP(nn.Module):
    """I0: Control MLP (21,314 params)"""
    def __init__(self, in_dim=165, hidden_dim=64, num_classes=2, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dim * 2, hidden_dim) # 330 -> 64
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes) # 64 -> 2

    def forward(self, x):
        mean_feat = x.mean(dim=1)
        std_feat  = x.std(dim=1)
        pooled    = torch.cat([mean_feat, std_feat], dim=1) # (B, 330)
        h         = self.dropout(self.relu(self.fc1(pooled)))
        return self.fc2(h)

class ModelI1_GRU(nn.Module):
    """I1: 1-Layer GRU (46,498 params)"""
    def __init__(self, input_dim=165, hidden_dim=64, fc_dim=32, dropout_p=0.5):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, fc_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(fc_dim, 2)

    def forward(self, x):
        out, h_n = self.gru(x)
        final_h = h_n[-1]
        out = self.dropout(self.relu(self.fc1(final_h)))
        return self.fc2(out)

class ModelI2_LSTM(nn.Module):
    """I2: 1-Layer LSTM (61,282 params)"""
    def __init__(self, input_dim=165, hidden_dim=64, fc_dim=32, dropout_p=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, fc_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(fc_dim, 2)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        final_h = h_n[-1]
        out = self.dropout(self.relu(self.fc1(final_h)))
        return self.fc2(out)

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        out = self.drop1(self.relu1(self.conv1(x)))
        out = self.drop2(self.relu2(self.conv2(out)))
        if out.size(2) != res.size(2):
            out = out[:, :, :res.size(2)]
        return self.relu(out + res)

class ModelI3_TCN(nn.Module):
    """I3: 1D TCN (83,618 params)"""
    def __init__(self, input_dim=165, num_channels=[64, 64], kernel_size=3, fc_dim=32, dropout_p=0.5):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_ch = input_dim if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, stride=1, dilation=dilation_size, padding=padding, dropout=0.2))
        self.tcn = nn.Sequential(*layers)
        self.fc1 = nn.Linear(num_channels[-1] * 2, fc_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(fc_dim, 2)

    def forward(self, x):
        x_t = x.permute(0, 2, 1)
        feat = self.tcn(x_t)
        mean_p = torch.mean(feat, dim=2)
        max_p, _  = torch.max(feat, dim=2)
        pooled = torch.cat([mean_p, max_p], dim=1)
        out = self.dropout(self.relu(self.fc1(pooled)))
        return self.fc2(out)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class ModelI4_Transformer(nn.Module):
    """I4: Transformer Encoder (46,242 params)"""
    def __init__(self, input_dim=165, d_model=64, nhead=4, dim_feedforward=128, fc_dim=32, dropout_p=0.5):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=50)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc1 = nn.Linear(d_model, fc_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(fc_dim, 2)

    def forward(self, x):
        h = self.pos_encoder(self.proj(x))
        out = self.transformer(h)
        pooled = torch.mean(out, dim=1)
        out = self.dropout(self.relu(self.fc1(pooled)))
        return self.fc2(out)

# ----------------------------------------------------------------------
# Dataset Class
# ----------------------------------------------------------------------
class YoloPoseDataset(Dataset):
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

def train_le2i_yolo_temporal():
    print("=" * 70)
    print("EXPERIMENT I: YOLO POSE TEMPORAL ARCHITECTURE BENCHMARK")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    manifest_path = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "processed_pose_features_manifest.csv")
    df_manifest = pd.read_csv(manifest_path).sort_values("window_id").reset_index(drop=True)
    yolo_dir = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "pose_estimator_features", "yolo_pose")

    models = {
        "control":    {"name": "I0: Control MLP",  "cls": ModelI0_MLP,  "params": 21314},
        "gru":        {"name": "I1: 1-Layer GRU",  "cls": ModelI1_GRU,  "params": 46498},
        "lstm":       {"name": "I2: 1-Layer LSTM", "cls": ModelI2_LSTM, "params": 61282},
        "tcn":        {"name": "I3: 1D TCN",       "cls": ModelI3_TCN,  "params": 83618},
        "transformer":{"name": "I4: Transformer",  "cls": ModelI4_Transformer, "params": 46242}
    }

    locations = ["Coffee_room_01", "Coffee_room_02", "Home_01", "Home_02"]
    all_benchmark_results = {}

    for m_key, m_meta in models.items():
        print("\n" + "=" * 70)
        print(f"TRAINING 4-FOLD LOLO BENCHMARK FOR {m_meta['name'].upper()} ({m_meta['params']:,} PARAMS)")
        print("=" * 70)

        ckpt_dir = os.path.join(ROOT_DIR, "checkpoints", "le2i_yolo_temporal", m_key)
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

            train_ds = YoloPoseDataset(inner_train_df, yolo_dir)
            val_ds   = YoloPoseDataset(inner_val_df, yolo_dir)
            test_ds  = YoloPoseDataset(test_df, yolo_dir)

            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
            val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)
            test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)

            model = m_meta["cls"]().to(device)
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

        print(f"\n   [{m_meta['name']}] LOLO Summary:")
        print(f"     - Mean LOLO F1 (@ 0.50)   : {lolo_f1_def * 100:.2f}%")
        print(f"     - Mean LOLO F1 (@ Tau*)   : {lolo_f1_opt * 100:.2f}% ± {lolo_f1_std * 100:.2f}%")

        all_benchmark_results[m_key] = {
            "name": m_meta["name"],
            "parameters": m_meta["params"],
            "lolo_f1_default_mean": float(lolo_f1_def),
            "lolo_f1_optimal_mean": float(lolo_f1_opt),
            "lolo_f1_optimal_std": float(lolo_f1_std),
            "folds": fold_results
        }

    res_dir = os.path.join(ROOT_DIR, "R&D", "ML_Baseline", "results")
    os.makedirs(res_dir, exist_ok=True)
    json_path = os.path.join(res_dir, "yolo_temporal_benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(all_benchmark_results, f, indent=2)

    csv_rows = []
    for m_key, m_res in all_benchmark_results.items():
        for r in m_res["folds"]:
            csv_rows.append({
                "model": m_res["name"],
                "parameters": m_res["parameters"],
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
    csv_path = os.path.join(res_dir, "yolo_temporal_benchmark_results.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    print("\n" + "=" * 70)
    print("EXPERIMENT I TRAINING COMPLETE — RESULTS SAVED")
    print("=" * 70)

if __name__ == "__main__":
    train_le2i_yolo_temporal()
