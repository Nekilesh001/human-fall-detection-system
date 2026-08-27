"""
Experiment C: Le2i Temporal Representation Ablation Training Pipeline
Trains and evaluates 3 controlled temporal variants across 4 LOLO folds:
- Model A: Mean-Only Baseline (32,962 params)
- Model B: Mean + Std Control (65,730 params)
- Model C: 1-Layer GRU (113,122 params)
Enforces strict outer test isolation, fold-specific class weights, and event-level inner validation.
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

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.model import URFDRGBFeatureBaseline as TemporalMeanStdControl
from src.train_baseline import compute_metrics

LABEL_MAP = {"NORMAL": 0, "FALL": 1}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Le2iFeatureDataset(Dataset):
    def __init__(self, df_manifest, root_dir):
        self.df = df_manifest.reset_index(drop=True)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feat_rel = str(row["processed_feature_path"]).replace("/", os.sep)
        feat_abs = os.path.join(self.root_dir, feat_rel)

        with np.load(feat_abs) as data:
            feats_np = data["features"] # (50, 512) float32

        label_int = LABEL_MAP[row["label"]]
        return torch.from_numpy(feats_np).float(), torch.tensor(label_int, dtype=torch.long), row["window_id"]

# Model A: Mean-Only Baseline
class TemporalMeanBaseline(nn.Module):
    def __init__(self, in_features=512, hidden_dim=64, num_classes=2, dropout_p=0.5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x: (B, 50, 512)
        mean_pooled = torch.mean(x, dim=1) # (B, 512)
        return self.classifier(mean_pooled)

    def get_parameter_counts(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"trainable": trainable}

# Model C: 1-Layer Sequential GRU
class TemporalGRUBaseline(nn.Module):
    def __init__(self, input_size=512, hidden_size=64, num_layers=1, num_classes=2, dropout_p=0.5):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x: (B, 50, 512)
        out, h_n = self.gru(x)
        last_hidden = h_n[-1] # (B, 64)
        return self.classifier(last_hidden)

    def get_parameter_counts(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"trainable": trainable}

def train_ablation_experiment():
    print("=" * 70)
    print("EXPERIMENT C: LE2I TEMPORAL REPRESENTATION ABLATION TRAINING")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device        : {device}")

    feat_manifest_path = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "processed_features_manifest.csv")
    df_feats = pd.read_csv(feat_manifest_path)
    total_samples = len(df_feats)
    print(f"Loaded Feature Manifest : {total_samples} samples from {feat_manifest_path}")

    base_ckpt_dir = os.path.join(ROOT_DIR, "checkpoints", "le2i_temporal_ablation")
    results_dir   = os.path.join(ROOT_DIR, "R&D", "ML_Baseline", "results", "le2i_temporal_ablation")
    os.makedirs(results_dir, exist_ok=True)

    model_configs = {
        "mean": {"name": "Model A (Mean-Only)", "cls": TemporalMeanBaseline, "dir": "mean"},
        "mean_std": {"name": "Model B (Mean+Std Control)", "cls": TemporalMeanStdControl, "dir": "mean_std"},
        "gru": {"name": "Model C (1-Layer GRU)", "cls": TemporalGRUBaseline, "dir": "gru"}
    }

    folds = {
        "Fold 1": {"test": ["Coffee_room_01"], "train": ["Coffee_room_02", "Home_01", "Home_02"]},
        "Fold 2": {"test": ["Coffee_room_02"], "train": ["Coffee_room_01", "Home_01", "Home_02"]},
        "Fold 3": {"test": ["Home_01"], "train": ["Coffee_room_01", "Coffee_room_02", "Home_02"]},
        "Fold 4": {"test": ["Home_02"], "train": ["Coffee_room_01", "Coffee_room_02", "Home_01"]}
    }

    all_ablation_results = []
    start_total_time = time.perf_counter()

    for m_key, m_meta in model_configs.items():
        print("\n" + "#" * 70)
        print(f"STARTING ABLATION VARIANT: {m_meta['name'].upper()}")
        print("#" * 70)

        m_ckpt_dir = os.path.join(base_ckpt_dir, m_meta["dir"])
        os.makedirs(m_ckpt_dir, exist_ok=True)

        for fold_num, (fold_name, f_info) in enumerate(folds.items(), start=1):
            print("\n" + "-" * 70)
            print(f"[{m_meta['name']}] {fold_name}: Outer Test = {f_info['test'][0]}")
            print("-" * 70)

            set_seed(42)

            test_locs = f_info["test"]
            train_locs = f_info["train"]

            outer_train_df = df_feats[df_feats["location"].isin(train_locs)].copy()
            outer_test_df  = df_feats[df_feats["location"].isin(test_locs)].copy()

            # Class Weights
            N_outer_train = len(outer_train_df)
            N_train_norm  = sum(outer_train_df["label"] == "NORMAL")
            N_train_fall  = sum(outer_train_df["label"] == "FALL")
            w_norm = N_outer_train / (2.0 * N_train_norm)
            w_fall = N_outer_train / (2.0 * N_train_fall)
            class_weights = torch.tensor([w_norm, w_fall], dtype=torch.float).to(device)

            # Inner Event Split (80% Inner Train, 20% Inner Val)
            outer_train_events = sorted(outer_train_df["event_id"].unique())
            np.random.seed(42)
            shuffled_events = np.random.permutation(outer_train_events)
            n_val_events = max(1, int(len(outer_train_events) * 0.20))
            
            inner_val_events = set(shuffled_events[:n_val_events])
            inner_tr_events  = set(shuffled_events[n_val_events:])

            inner_tr_df  = outer_train_df[outer_train_df["event_id"].isin(inner_tr_events)].copy()
            inner_val_df = outer_train_df[outer_train_df["event_id"].isin(inner_val_events)].copy()

            # Loaders
            ds_inner_tr   = Le2iFeatureDataset(inner_tr_df, ROOT_DIR)
            ds_inner_val  = Le2iFeatureDataset(inner_val_df, ROOT_DIR)
            ds_outer_test = Le2iFeatureDataset(outer_test_df, ROOT_DIR)

            loader_tr   = DataLoader(ds_inner_tr, batch_size=32, shuffle=True)
            loader_val  = DataLoader(ds_inner_val, batch_size=32, shuffle=False)
            loader_test = DataLoader(ds_outer_test, batch_size=32, shuffle=False)

            # Model Instantiation
            model = m_meta["cls"]().to(device)
            param_counts = model.get_parameter_counts()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
            criterion = nn.CrossEntropyLoss(weight=class_weights)

            best_val_f1 = -1.0
            best_epoch = 0
            fold_ckpt_path = os.path.join(m_ckpt_dir, f"fold_{fold_num}_best.pth")

            # Training Loop (50 Epochs)
            fold_start_time = time.perf_counter()

            for epoch in range(1, 51):
                model.train()
                train_loss = 0.0
                for batch_x, batch_y, _ in loader_tr:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * len(batch_y)

                # Inner Val Eval
                model.eval()
                val_probs, val_targets = [], []
                with torch.no_grad():
                    for batch_x, batch_y, _ in loader_val:
                        batch_x = batch_x.to(device)
                        logits = model(batch_x)
                        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                        val_probs.extend(probs)
                        val_targets.extend(batch_y.numpy())

                val_f1 = compute_metrics(val_targets, val_probs, threshold=0.50)["f1"]

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_epoch = epoch
                    torch.save(model.state_dict(), fold_ckpt_path)

            fold_train_time = time.perf_counter() - fold_start_time

            # Inner Threshold Tuning on Best Checkpoint
            model.load_state_dict(torch.load(fold_ckpt_path, map_location=device))
            model.eval()

            val_probs, val_targets = [], []
            with torch.no_grad():
                for batch_x, batch_y, _ in loader_val:
                    batch_x = batch_x.to(device)
                    logits = model(batch_x)
                    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    val_probs.extend(probs)
                    val_targets.extend(batch_y.numpy())

            best_tau = 0.50
            best_tau_f1 = -1.0
            for tau_cand in np.arange(0.05, 0.96, 0.05):
                m_cand = compute_metrics(val_targets, val_probs, threshold=float(tau_cand))
                if m_cand["f1"] > best_tau_f1:
                    best_tau_f1 = m_cand["f1"]
                    best_tau = float(tau_cand)

            # Outer Held-Out Test Evaluation
            test_probs, test_targets = [], []
            test_start_time = time.perf_counter()
            with torch.no_grad():
                for batch_x, batch_y, _ in loader_test:
                    batch_x = batch_x.to(device)
                    logits = model(batch_x)
                    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    test_probs.extend(probs)
                    test_targets.extend(batch_y.numpy())

            test_latency_ms = ((time.perf_counter() - test_start_time) / len(ds_outer_test)) * 1000.0

            m_outer_050 = compute_metrics(test_targets, test_probs, threshold=0.50)
            m_outer_tau = compute_metrics(test_targets, test_probs, threshold=best_tau)

            # Event Sensitivity
            outer_test_df["prob_fall"] = test_probs
            event_groups = outer_test_df.groupby("event_id")
            fall_events_count = 0
            detected_events_050 = 0
            detected_events_tau = 0

            for event_id, grp in event_groups:
                if grp["f_start"].iloc[0] > 0:
                    fall_events_count += 1
                    if (grp["prob_fall"] >= 0.50).any():
                        detected_events_050 += 1
                    if (grp["prob_fall"] >= best_tau).any():
                        detected_events_tau += 1

            event_sens_050 = (detected_events_050 / fall_events_count) * 100.0 if fall_events_count > 0 else 100.0
            event_sens_tau = (detected_events_tau / fall_events_count) * 100.0 if fall_events_count > 0 else 100.0

            print(f"  Best Ep: {best_epoch} | Inner Val F1: {best_val_f1:.4f} | Selected tau*: {best_tau:.2f}")
            print(f"  Outer Test @ 0.50 -> Acc: {m_outer_050['accuracy']:.4f}, Sens: {m_outer_050['sensitivity']:.4f}, Spec: {m_outer_050['specificity']:.4f}, F1: {m_outer_050['f1']:.4f}, CM: {m_outer_050['confusion_matrix']}")
            print(f"  Outer Test @ tau* -> Acc: {m_outer_tau['accuracy']:.4f}, Sens: {m_outer_tau['sensitivity']:.4f}, Spec: {m_outer_tau['specificity']:.4f}, F1: {m_outer_tau['f1']:.4f}, CM: {m_outer_tau['confusion_matrix']}")

            all_ablation_results.append({
                "model_key": m_key,
                "model_name": m_meta["name"],
                "trainable_params": param_counts["trainable"],
                "fold": fold_name,
                "test_location": f_info['test'][0],
                "best_epoch": best_epoch,
                "best_val_f1": best_val_f1,
                "selected_tau": best_tau,
                "acc_050": m_outer_050["accuracy"],
                "prec_050": m_outer_050["precision"],
                "sens_050": m_outer_050["sensitivity"],
                "spec_050": m_outer_050["specificity"],
                "f1_050": m_outer_050["f1"],
                "event_sens_050": event_sens_050,
                "cm_050": str(m_outer_050["confusion_matrix"]),
                "acc_tau": m_outer_tau["accuracy"],
                "prec_tau": m_outer_tau["precision"],
                "sens_tau": m_outer_tau["sensitivity"],
                "spec_tau": m_outer_tau["specificity"],
                "f1_tau": m_outer_tau["f1"],
                "event_sens_tau": event_sens_tau,
                "cm_tau": str(m_outer_tau["confusion_matrix"]),
                "latency_ms": test_latency_ms,
                "train_time_sec": fold_train_time
            })

    total_exp_time = time.perf_counter() - start_total_time

    # Save CSV
    df_ablation = pd.DataFrame(all_ablation_results)
    results_csv_path = os.path.join(results_dir, "ablation_fold_results.csv")
    df_ablation.to_csv(results_csv_path, index=False)

    print("\n" + "=" * 70)
    print("EXPERIMENT C: LE2I TEMPORAL ABLATION SUMMARY")
    print("=" * 70)
    print(f"Total Execution Time: {total_exp_time:.2f} seconds")
    print(f"Results CSV Saved   : {results_csv_path}")

    # Summary table across models @ tau=0.50
    print("\nLOLO MEAN METRICS ACROSS TEMPORAL MODEL VARIANTS (@ tau = 0.50):")
    for m_key in ["mean", "mean_std", "gru"]:
        sub = df_ablation[df_ablation["model_key"] == m_key]
        print(f"\n{sub['model_name'].iloc[0]} ({sub['trainable_params'].iloc[0]:,} params):")
        print(f"  Accuracy   : {sub['acc_050'].mean():.4f} ± {sub['acc_050'].std():.4f}")
        print(f"  Precision  : {sub['prec_050'].mean():.4f} ± {sub['prec_050'].std():.4f}")
        print(f"  Recall/Sens: {sub['sens_050'].mean():.4f} ± {sub['sens_050'].std():.4f}")
        print(f"  Specificity: {sub['spec_050'].mean():.4f} ± {sub['spec_050'].std():.4f}")
        print(f"  F1 Score   : {sub['f1_050'].mean():.4f} ± {sub['f1_050'].std():.4f}")
        print(f"  Event Sens : {sub['event_sens_050'].mean():.2f}% ± {sub['event_sens_050'].std():.2f}%")

    print("=" * 70)

    return df_ablation

if __name__ == "__main__":
    train_ablation_experiment()
