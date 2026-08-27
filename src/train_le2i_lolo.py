"""
Experiment B: Le2i Supervised 4-Fold LOLO Baseline Training Pipeline
Trains baseline MLP classifiers (65,730 params) across 4 Leave-One-Location-Out folds.
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

from src.model import URFDRGBFeatureBaseline
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

def train_lolo_experiment():
    print("=" * 70)
    print("EXPERIMENT B: LE2I SUPERVISED 4-FOLD LOLO BASELINE TRAINING")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device        : {device}")

    feat_manifest_path = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "processed_features_manifest.csv")
    if not os.path.exists(feat_manifest_path):
        raise FileNotFoundError(f"Feature manifest missing at {feat_manifest_path}")

    df_feats = pd.read_csv(feat_manifest_path)
    total_samples = len(df_feats)
    print(f"Loaded Feature Manifest : {total_samples} samples from {feat_manifest_path}")

    ckpt_dir = os.path.join(ROOT_DIR, "checkpoints", "le2i_lolo")
    results_dir = os.path.join(ROOT_DIR, "R&D", "ML_Baseline", "results", "le2i_lolo")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    folds = {
        "Fold 1": {"test": ["Coffee_room_01"], "train": ["Coffee_room_02", "Home_01", "Home_02"]},
        "Fold 2": {"test": ["Coffee_room_02"], "train": ["Coffee_room_01", "Home_01", "Home_02"]},
        "Fold 3": {"test": ["Home_01"], "train": ["Coffee_room_01", "Coffee_room_02", "Home_02"]},
        "Fold 4": {"test": ["Home_02"], "train": ["Coffee_room_01", "Coffee_room_02", "Home_01"]}
    }

    fold_results = []
    start_total_time = time.perf_counter()

    for fold_num, (fold_name, f_info) in enumerate(folds.items(), start=1):
        print("\n" + "=" * 70)
        print(f"{fold_name.upper()}: Outer Test Location = {f_info['test'][0]}")
        print("=" * 70)

        set_seed(42)

        test_locs = f_info["test"]
        train_locs = f_info["train"]

        outer_train_df = df_feats[df_feats["location"].isin(train_locs)].copy()
        outer_test_df  = df_feats[df_feats["location"].isin(test_locs)].copy()

        # Leakage Safeguard Audits
        assert set(outer_train_df["location"]).isdisjoint(set(outer_test_df["location"])), "Location leakage!"
        assert set(outer_train_df["event_id"]).isdisjoint(set(outer_test_df["event_id"])), "Event leakage!"

        # Fold-Specific Class Weight Calculation
        N_outer_train = len(outer_train_df)
        N_train_norm  = sum(outer_train_df["label"] == "NORMAL")
        N_train_fall  = sum(outer_train_df["label"] == "FALL")

        w_norm = N_outer_train / (2.0 * N_train_norm)
        w_fall = N_outer_train / (2.0 * N_train_fall)

        class_weights = torch.tensor([w_norm, w_fall], dtype=torch.float).to(device)

        print(f"Outer Train Windows : {N_outer_train} (FALL={N_train_fall}, NORM={N_train_norm}) | Events: {len(outer_train_df['event_id'].unique())}")
        print(f"Outer Test Windows  : {len(outer_test_df)} (FALL={sum(outer_test_df['label']=='FALL')}, NORM={sum(outer_test_df['label']=='NORMAL')}) | Events: {len(outer_test_df['event_id'].unique())}")
        print(f"Calculated Weights  : NORMAL = {w_norm:.4f}, FALL = {w_fall:.4f}")

        # Inner Event-Level Validation Split (80% Inner Train, 20% Inner Val)
        outer_train_events = sorted(outer_train_df["event_id"].unique())
        np.random.seed(42)
        shuffled_events = np.random.permutation(outer_train_events)
        n_val_events = max(1, int(len(outer_train_events) * 0.20))
        
        inner_val_events = set(shuffled_events[:n_val_events])
        inner_tr_events  = set(shuffled_events[n_val_events:])

        assert inner_val_events.isdisjoint(inner_tr_events), "Inner split event leakage!"

        inner_tr_df  = outer_train_df[outer_train_df["event_id"].isin(inner_tr_events)].copy()
        inner_val_df = outer_train_df[outer_train_df["event_id"].isin(inner_val_events)].copy()

        print(f"Inner Split         : Inner Train Events={len(inner_tr_events)} ({len(inner_tr_df)} wins) | Inner Val Events={len(inner_val_events)} ({len(inner_val_df)} wins)")

        # Data Loaders
        ds_inner_tr  = Le2iFeatureDataset(inner_tr_df, ROOT_DIR)
        ds_inner_val = Le2iFeatureDataset(inner_val_df, ROOT_DIR)
        ds_outer_test = Le2iFeatureDataset(outer_test_df, ROOT_DIR)

        loader_tr  = DataLoader(ds_inner_tr, batch_size=32, shuffle=True)
        loader_val = DataLoader(ds_inner_val, batch_size=32, shuffle=False)
        loader_test = DataLoader(ds_outer_test, batch_size=32, shuffle=False)

        # Model Initialization
        model = URFDRGBFeatureBaseline(dropout_p=0.5).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        best_val_f1 = -1.0
        best_epoch = 0
        fold_ckpt_path = os.path.join(ckpt_dir, f"fold_{fold_num}_best.pth")

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

            train_loss /= len(ds_inner_tr)

            # Inner Validation Evaluation
            model.eval()
            val_probs = []
            val_targets = []

            with torch.no_grad():
                for batch_x, batch_y, _ in loader_val:
                    batch_x = batch_x.to(device)
                    logits = model(batch_x)
                    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    val_probs.extend(probs)
                    val_targets.extend(batch_y.numpy())

            val_metrics = compute_metrics(val_targets, val_probs, threshold=0.50)
            val_f1 = val_metrics["f1"]

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                torch.save(model.state_dict(), fold_ckpt_path)

            if epoch in [1, 10, 25, 50] or epoch == best_epoch:
                print(f"  Epoch [{epoch:02d}/50] - Loss: {train_loss:.4f} | Inner Val F1 (@0.50): {val_f1:.4f} (Best: {best_val_f1:.4f} @ Ep {best_epoch})")

        fold_train_time = time.perf_counter() - fold_start_time
        print(f"\nTraining Complete for {fold_name}. Best Checkpoint Saved from Epoch {best_epoch} (Val F1: {best_val_f1:.4f})")

        # Inner Threshold Tuning on Best Checkpoint
        model.load_state_dict(torch.load(fold_ckpt_path, map_location=device))
        model.eval()

        val_probs = []
        val_targets = []
        with torch.no_grad():
            for batch_x, batch_y, _ in loader_val:
                batch_x = batch_x.to(device)
                logits = model(batch_x)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                val_probs.extend(probs)
                val_targets.extend(batch_y.numpy())

        # Grid search inner validation threshold tau in [0.05, 0.95]
        best_tau = 0.50
        best_tau_f1 = -1.0
        for tau_cand in np.arange(0.05, 0.96, 0.05):
            m_cand = compute_metrics(val_targets, val_probs, threshold=float(tau_cand))
            if m_cand["f1"] > best_tau_f1:
                best_tau_f1 = m_cand["f1"]
                best_tau = float(tau_cand)

        print(f"Selected Inner Validation Threshold tau* = {best_tau:.2f} (Inner Val F1: {best_tau_f1:.4f})")

        # Outer Held-Out Test Evaluation (Evaluated ONCE at tau=0.50 and tau=best_tau)
        test_probs = []
        test_targets = []
        test_win_ids = []

        test_start_time = time.perf_counter()
        with torch.no_grad():
            for batch_x, batch_y, win_ids in loader_test:
                batch_x = batch_x.to(device)
                logits = model(batch_x)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                test_probs.extend(probs)
                test_targets.extend(batch_y.numpy())
                test_win_ids.extend(win_ids)
        test_latency_ms = ((time.perf_counter() - test_start_time) / len(ds_outer_test)) * 1000.0

        m_outer_050 = compute_metrics(test_targets, test_probs, threshold=0.50)
        m_outer_tau = compute_metrics(test_targets, test_probs, threshold=best_tau)

        # Outer Event-Level Sensitivity Evaluation
        outer_test_df["prob_fall"] = test_probs
        event_groups = outer_test_df.groupby("event_id")
        fall_events_count = 0
        detected_events_050 = 0
        detected_events_tau = 0

        for event_id, grp in event_groups:
            f_start = grp["f_start"].iloc[0]
            if f_start > 0:
                fall_events_count += 1
                if (grp["prob_fall"] >= 0.50).any():
                    detected_events_050 += 1
                if (grp["prob_fall"] >= best_tau).any():
                    detected_events_tau += 1

        event_sens_050 = (detected_events_050 / fall_events_count) * 100.0 if fall_events_count > 0 else 100.0
        event_sens_tau = (detected_events_tau / fall_events_count) * 100.0 if fall_events_count > 0 else 100.0

        print(f"\nOuter Test Results for {f_info['test'][0]} @ tau = 0.50:")
        print(f"  Accuracy: {m_outer_050['accuracy']:.4f}, Sens: {m_outer_050['sensitivity']:.4f}, Spec: {m_outer_050['specificity']:.4f}, F1: {m_outer_050['f1']:.4f}, CM: {m_outer_050['confusion_matrix']}")
        print(f"  Event Sensitivity: {event_sens_050:.2f}% ({detected_events_050}/{fall_events_count})")

        print(f"\nOuter Test Results for {f_info['test'][0]} @ Selected tau* ({best_tau:.2f}):")
        print(f"  Accuracy: {m_outer_tau['accuracy']:.4f}, Sens: {m_outer_tau['sensitivity']:.4f}, Spec: {m_outer_tau['specificity']:.4f}, F1: {m_outer_tau['f1']:.4f}, CM: {m_outer_tau['confusion_matrix']}")
        print(f"  Event Sensitivity: {event_sens_tau:.2f}% ({detected_events_tau}/{fall_events_count})")

        fold_results.append({
            "fold": fold_name,
            "test_location": f_info['test'][0],
            "train_windows": N_outer_train,
            "test_windows": len(outer_test_df),
            "train_events": len(outer_train_df['event_id'].unique()),
            "test_events": len(outer_test_df['event_id'].unique()),
            "fall_events": fall_events_count,
            "w_norm": w_norm,
            "w_fall": w_fall,
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

    # Save CSV Results
    df_results = pd.DataFrame(fold_results)
    results_csv_path = os.path.join(results_dir, "lolo_fold_results.csv")
    df_results.to_csv(results_csv_path, index=False)

    print("\n" + "=" * 70)
    print("LE2I SUPERVISED 4-FOLD LOLO CROSS-VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Total Experiment Time   : {total_exp_time:.2f} seconds")
    print(f"Results CSV Saved       : {results_csv_path}")

    # Calculate Mean ± Std across Folds @ tau=0.50 and selected tau*
    print("\nLOLO SUMMARY METRICS (MEAN ± STD ACROSS 4 FOLDS):")
    print("At Default Threshold (tau = 0.50):")
    print(f"  Accuracy   : {df_results['acc_050'].mean():.4f} ± {df_results['acc_050'].std():.4f}")
    print(f"  Precision  : {df_results['prec_050'].mean():.4f} ± {df_results['prec_050'].std():.4f}")
    print(f"  Recall/Sens: {df_results['sens_050'].mean():.4f} ± {df_results['sens_050'].std():.4f}")
    print(f"  Specificity: {df_results['spec_050'].mean():.4f} ± {df_results['spec_050'].std():.4f}")
    print(f"  F1 Score   : {df_results['f1_050'].mean():.4f} ± {df_results['f1_050'].std():.4f}")
    print(f"  Event Sens : {df_results['event_sens_050'].mean():.2f}% ± {df_results['event_sens_050'].std():.2f}%")

    print("\nAt Selected Inner Validation Threshold (tau*):")
    print(f"  Accuracy   : {df_results['acc_tau'].mean():.4f} ± {df_results['acc_tau'].std():.4f}")
    print(f"  Precision  : {df_results['prec_tau'].mean():.4f} ± {df_results['prec_tau'].std():.4f}")
    print(f"  Recall/Sens: {df_results['sens_tau'].mean():.4f} ± {df_results['sens_tau'].std():.4f}")
    print(f"  Specificity: {df_results['spec_tau'].mean():.4f} ± {df_results['spec_tau'].std():.4f}")
    print(f"  F1 Score   : {df_results['f1_tau'].mean():.4f} ± {df_results['f1_tau'].std():.4f}")
    print(f"  Event Sens : {df_results['event_sens_tau'].mean():.2f}% ± {df_results['event_sens_tau'].std():.2f}%")

    print("=" * 70)

    return df_results

if __name__ == "__main__":
    train_lolo_experiment()
