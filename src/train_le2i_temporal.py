"""
4-Fold Leave-One-Location-Out (LOLO) Training Script for Experiment G: Temporal Architecture Benchmark.

Models Trained:
- G1: 1-Layer GRU (46,498 params)
- G2: 1-Layer LSTM (61,282 params)
- G3: 1D TCN (83,618 params)
- G4: 1-Layer Transformer Encoder (46,242 params)

Reference Control:
- G0: Canonical E2 Pose+Velocity Control MLP (21,314 params, 72.23% F1 - Untouched)

Enforces:
- Strict seed isolation: set_seed(42) before EACH model AND BEFORE EACH FOLD.
- 4 LOLO Folds (Coffee_room_01, Coffee_room_02, Home_01, Home_02).
- Inner event-level 80/20 train/val split.
- Class weights calculated ONLY from outer training locations.
- Checkpoints saved under checkpoints/le2i_temporal/{gru, lstm, tcn, transformer}/fold_{1..4}_best.pth.
- Inner validation threshold tuning tau* in [0.05, 0.95].
"""

import os
import sys
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT_DIR = r"d:\ONE_DATA\Fall detection"
sys.path.insert(0, ROOT_DIR)

from src.train_baseline import compute_metrics

LABEL_MAP = {"NORMAL": 0, "FALL": 1}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Le2iPoseE2Dataset(Dataset):
    def __init__(self, df_manifest, root_dir):
        self.df = df_manifest.reset_index(drop=True)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feat_rel = str(row["e2_feature_path"]).replace("/", os.sep)
        feat_abs = os.path.join(self.root_dir, feat_rel)

        with np.load(feat_abs) as data:
            feat_np = data["features"] # (50, 165)

        label_int = LABEL_MAP[row["label"]]
        return torch.tensor(feat_np, dtype=torch.float32), torch.tensor(label_int, dtype=torch.long), row["window_id"]

# ----------------------------------------------------------------------
# Model Architectures
# ----------------------------------------------------------------------
class ModelG1_GRU(nn.Module):
    """G1: 1-Layer GRU (46,498 params)"""
    def __init__(self, input_dim=165, hidden_dim=64, fc_dim=32, dropout_p=0.5):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, fc_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(fc_dim, 2)

    def forward(self, x):
        # x: (B, 50, 165)
        out, h_n = self.gru(x) # h_n: (1, B, 64)
        final_h = h_n[-1] # (B, 64)
        out = self.dropout(self.relu(self.fc1(final_h)))
        return self.fc2(out)

class ModelG2_LSTM(nn.Module):
    """G2: 1-Layer LSTM (61,282 params)"""
    def __init__(self, input_dim=165, hidden_dim=64, fc_dim=32, dropout_p=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, fc_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(fc_dim, 2)

    def forward(self, x):
        # x: (B, 50, 165)
        out, (h_n, c_n) = self.lstm(x) # h_n: (1, B, 64)
        final_h = h_n[-1] # (B, 64)
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

class ModelG3_TCN(nn.Module):
    """G3: 1D Temporal Convolutional Network (83,618 params)"""
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
        self.fc1 = nn.Linear(num_channels[-1] * 2, fc_dim) # Mean + Max pooling (128 -> 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(fc_dim, 2)

    def forward(self, x):
        # x: (B, 50, 165) -> permute to (B, 165, 50)
        x_t = x.permute(0, 2, 1)
        feat = self.tcn(x_t) # (B, 64, 50)
        mean_p = torch.mean(feat, dim=2) # (B, 64)
        max_p, _  = torch.max(feat, dim=2)  # (B, 64)
        pooled = torch.cat([mean_p, max_p], dim=1) # (B, 128)
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

class ModelG4_Transformer(nn.Module):
    """G4: Lightweight Transformer Encoder (46,242 params)"""
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
        # x: (B, 50, 165)
        h = self.proj(x) # (B, 50, 64)
        h = self.pos_encoder(h)
        feat = self.transformer(h) # (B, 50, 64)
        mean_p = torch.mean(feat, dim=1) # (B, 64)
        out = self.dropout(self.relu(self.fc1(mean_p)))
        return self.fc2(out)

# ----------------------------------------------------------------------
# Training Pipeline
# ----------------------------------------------------------------------
def train_le2i_temporal():
    print("=" * 70)
    print("EXPERIMENT G: LE2I TEMPORAL ARCHITECTURE BENCHMARK TRAINING")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device}")

    pose_manifest_path = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "processed_pose_features_manifest.csv")
    assert os.path.exists(pose_manifest_path), f"Manifest missing: {pose_manifest_path}"
    df_manifest = pd.read_csv(pose_manifest_path)
    df_manifest = df_manifest.sort_values("window_id").reset_index(drop=True)

    folds = {
        "Fold 1": {"num": 1, "test": ["Coffee_room_01"], "train": ["Coffee_room_02", "Home_01", "Home_02"]},
        "Fold 2": {"num": 2, "test": ["Coffee_room_02"], "train": ["Coffee_room_01", "Home_01", "Home_02"]},
        "Fold 3": {"num": 3, "test": ["Home_01"], "train": ["Coffee_room_01", "Coffee_room_02", "Home_02"]},
        "Fold 4": {"num": 4, "test": ["Home_02"], "train": ["Coffee_room_01", "Coffee_room_02", "Home_01"]}
    }

    models_meta = {
        "gru":         {"name": "Model G1 (1-Layer GRU)", "dir": "gru", "cls": ModelG1_GRU},
        "lstm":        {"name": "Model G2 (1-Layer LSTM)", "dir": "lstm", "cls": ModelG2_LSTM},
        "tcn":         {"name": "Model G3 (1D TCN)", "dir": "tcn", "cls": ModelG3_TCN},
        "transformer": {"name": "Model G4 (Transformer Encoder)", "dir": "transformer", "cls": ModelG4_Transformer}
    }

    base_ckpt_dir = os.path.join(ROOT_DIR, "checkpoints", "le2i_temporal")
    os.makedirs(base_ckpt_dir, exist_ok=True)

    all_fold_results = []

    for m_key, m_info in models_meta.items():
        print("\n" + "#" * 70)
        print(f"STARTING MODEL VARIANT: {m_info['name'].upper()}")
        print("#" * 70)

        m_ckpt_dir = os.path.join(base_ckpt_dir, m_info["dir"])
        os.makedirs(m_ckpt_dir, exist_ok=True)

        dummy_model = m_info["cls"]()
        param_count = sum(p.numel() for p in dummy_model.parameters() if p.requires_grad)
        print(f"Verified Model Architecture: {m_info['name']} | Trainable Params: {param_count:,}")

        for fold_name, f_info in folds.items():
            fold_num = f_info["num"]
            test_locs  = f_info["test"]
            train_locs = f_info["train"]

            print("\n" + "-" * 70)
            print(f"[{m_info['name']}] {fold_name}: Outer Test = {test_locs[0]}")
            print("-" * 70)

            # ENFORCE STRICT SEED RESET BEFORE EVERY MODEL AND EVERY FOLD
            set_seed(42)

            outer_train_df = df_manifest[df_manifest["location"].isin(train_locs)].copy()
            outer_test_df  = df_manifest[df_manifest["location"].isin(test_locs)].copy()

            # Class weights (computed ONLY from outer training locations)
            N_outer_train = len(outer_train_df)
            N_train_norm  = sum(outer_train_df["label"] == "NORMAL")
            N_train_fall  = sum(outer_train_df["label"] == "FALL")
            w_norm = N_outer_train / (2.0 * N_train_norm)
            w_fall = N_outer_train / (2.0 * N_train_fall)
            class_weights = torch.tensor([w_norm, w_fall], dtype=torch.float).to(device)

            # Inner Event Split (80% Inner Train, 20% Inner Validation)
            outer_train_events = sorted(outer_train_df["event_id"].unique())
            np.random.seed(42)
            shuffled_events = np.random.permutation(outer_train_events)
            n_val_events = max(1, int(len(outer_train_events) * 0.20))

            inner_val_events = set(shuffled_events[:n_val_events])
            inner_tr_events  = set(shuffled_events[n_val_events:])

            inner_tr_df  = outer_train_df[outer_train_df["event_id"].isin(inner_tr_events)].copy()
            inner_val_df = outer_train_df[outer_train_df["event_id"].isin(inner_val_events)].copy()

            loader_tr   = DataLoader(Le2iPoseE2Dataset(inner_tr_df, ROOT_DIR), batch_size=32, shuffle=True)
            loader_val  = DataLoader(Le2iPoseE2Dataset(inner_val_df, ROOT_DIR), batch_size=32, shuffle=False)
            loader_test = DataLoader(Le2iPoseE2Dataset(outer_test_df, ROOT_DIR), batch_size=32, shuffle=False)

            model = m_info["cls"]().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
            criterion = nn.CrossEntropyLoss(weight=class_weights)

            best_val_f1 = -1.0
            best_epoch = 0
            fold_ckpt_path = os.path.join(m_ckpt_dir, f"fold_{fold_num}_best.pth")

            fold_start_time = time.perf_counter()

            # 50 Epochs Training
            for epoch in range(1, 51):
                model.train()
                train_loss = 0.0
                for bx, by, _ in loader_tr:
                    bx, by = bx.to(device), by.to(device)
                    optimizer.zero_grad()
                    logits = model(bx)
                    loss = criterion(logits, by)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * len(by)

                # Inner Val Evaluation
                model.eval()
                val_probs, val_targets = [], []
                with torch.no_grad():
                    for bx, by, _ in loader_val:
                        bx = bx.to(device)
                        probs = torch.softmax(model(bx), dim=1)[:, 1].cpu().numpy()
                        val_probs.extend(probs)
                        val_targets.extend(by.numpy())

                val_f1 = compute_metrics(val_targets, val_probs, threshold=0.50)["f1"]

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_epoch = epoch
                    torch.save(model.state_dict(), fold_ckpt_path)

            fold_train_time = time.perf_counter() - fold_start_time

            # Inner Threshold Search on Best Checkpoint
            model.load_state_dict(torch.load(fold_ckpt_path, map_location=device))
            model.eval()

            val_probs, val_targets = [], []
            with torch.no_grad():
                for bx, by, _ in loader_val:
                    bx = bx.to(device)
                    probs = torch.softmax(model(bx), dim=1)[:, 1].cpu().numpy()
                    val_probs.extend(probs)
                    val_targets.extend(by.numpy())

            best_tau = 0.50
            best_tau_f1 = -1.0
            for tau_cand in np.arange(0.05, 0.96, 0.05):
                m_cand = compute_metrics(val_targets, val_probs, threshold=float(tau_cand))
                if m_cand["f1"] > best_tau_f1:
                    best_tau_f1 = m_cand["f1"]
                    best_tau = float(tau_cand)

            print(f"  Best Ep: {best_epoch} | Inner Val F1: {best_val_f1:.4f} | Selected tau*: {best_tau:.2f} | Train Time: {fold_train_time:.2f}s")

            # Outer Held-Out Test Evaluation
            t_infer_start = time.perf_counter()
            test_probs, test_targets = [], []
            with torch.no_grad():
                for bx, by, _ in loader_test:
                    bx = bx.to(device)
                    probs = torch.softmax(model(bx), dim=1)[:, 1].cpu().numpy()
                    test_probs.extend(probs)
                    test_targets.extend(by.numpy())
            infer_time = time.perf_counter() - t_infer_start
            infer_latency_ms = (infer_time / len(outer_test_df)) * 1000.0

            m_def = compute_metrics(test_targets, test_probs, threshold=0.50)
            m_opt = compute_metrics(test_targets, test_probs, threshold=best_tau)

            outer_test_df["pred_prob"] = test_probs
            event_sens_list = []
            for ev_id, ev_grp in outer_test_df.groupby("event_id"):
                if (ev_grp["label"] == "FALL").any():
                    if (ev_grp["pred_prob"] >= 0.50).any():
                        event_sens_list.append(1.0)
                    else:
                        event_sens_list.append(0.0)

            event_sens = np.mean(event_sens_list) * 100.0 if event_sens_list else 0.0

            print(f"  Outer Test @ 0.50 -> Acc: {m_def['accuracy']:.4f}, Sens: {m_def['sensitivity']:.4f}, Spec: {m_def['specificity']:.4f}, F1: {m_def['f1']:.4f}, CM: {m_def['confusion_matrix']}")
            print(f"  Outer Test @ tau* -> Acc: {m_opt['accuracy']:.4f}, Sens: {m_opt['sensitivity']:.4f}, Spec: {m_opt['specificity']:.4f}, F1: {m_opt['f1']:.4f}, CM: {m_opt['confusion_matrix']}")

            all_fold_results.append({
                "model_key": m_key,
                "model_name": m_info["name"],
                "fold_num": fold_num,
                "fold_name": fold_name,
                "test_location": test_locs[0],
                "trainable_params": param_count,
                "best_epoch": best_epoch,
                "best_tau": best_tau,
                "train_time_s": fold_train_time,
                "infer_latency_ms": infer_latency_ms,
                "acc_050": m_def["accuracy"],
                "prec_050": m_def["precision"],
                "sens_050": m_def["sensitivity"],
                "spec_050": m_def["specificity"],
                "f1_050": m_def["f1"],
                "event_sens_050": event_sens,
                "acc_tau": m_opt["accuracy"],
                "prec_tau": m_opt["precision"],
                "sens_tau": m_opt["sensitivity"],
                "spec_tau": m_opt["specificity"],
                "f1_tau": m_opt["f1"]
            })

    # Save Results CSV
    res_dir = os.path.join(ROOT_DIR, "R&D", "ML_Baseline", "results", "le2i_temporal")
    os.makedirs(res_dir, exist_ok=True)
    df_res = pd.DataFrame(all_fold_results)
    csv_path = os.path.join(res_dir, "temporal_fold_results.csv")
    df_res.to_csv(csv_path, index=False)

    print("\n" + "=" * 70)
    print("EXPERIMENT G: TEMPORAL BENCHMARK TRAINING SUMMARY")
    print("=" * 70)
    for m_key, m_info in models_meta.items():
        sub_df = df_res[df_res["model_key"] == m_key]
        print(f"\n{m_info['name']}:")
        print(f"  Accuracy   : {sub_df['acc_050'].mean():.4f} ± {sub_df['acc_050'].std():.4f}")
        print(f"  Precision  : {sub_df['prec_050'].mean():.4f} ± {sub_df['prec_050'].std():.4f}")
        print(f"  Recall/Sens: {sub_df['sens_050'].mean():.4f} ± {sub_df['sens_050'].std():.4f}")
        print(f"  Specificity: {sub_df['spec_050'].mean():.4f} ± {sub_df['spec_050'].std():.4f}")
        print(f"  F1 Score   : {sub_df['f1_050'].mean():.4f} ± {sub_df['f1_050'].std():.4f}")
        print(f"  Event Sens : {sub_df['event_sens_050'].mean():.2f}% ± {sub_df['event_sens_050'].std():.2f}%")

    print(f"\nResults CSV Saved to: {csv_path}")

if __name__ == "__main__":
    train_le2i_temporal()
