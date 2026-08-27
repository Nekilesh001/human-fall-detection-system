"""
Experiment D: Le2i Optical Flow & Dual-Stream Fusion Training Pipeline
Trains and evaluates 3 controlled models across 4 LOLO folds:
- Model D1 (Optical Flow-Only Baseline, 65,730 params)
- Model D2 (RGB Control Baseline, 65,730 params)
- Model D3 (RGB + Optical Flow Dual-Stream Fusion, 131,266 params)
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

from src.train_baseline import compute_metrics
from src.model import URFDRGBFeatureBaseline as ModelD2_RGBControl

LABEL_MAP = {"NORMAL": 0, "FALL": 1}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Le2iDualStreamDataset(Dataset):
    def __init__(self, df_manifest, root_dir):
        self.df = df_manifest.reset_index(drop=True)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load RGB feature (50, 512)
        rgb_rel = str(row["processed_feature_path"]).replace("/", os.sep)
        rgb_abs = os.path.join(self.root_dir, rgb_rel)
        with np.load(rgb_abs) as data:
            rgb_np = data["features"]

        # Load Flow feature (49, 512)
        flow_rel = str(row["flow_feature_path"]).replace("/", os.sep)
        flow_abs = os.path.join(self.root_dir, flow_rel)
        with np.load(flow_abs) as data:
            flow_np = data["features"]

        label_int = LABEL_MAP[row["label"]]
        return (
            torch.from_numpy(rgb_np).float(),
            torch.from_numpy(flow_np).float(),
            torch.tensor(label_int, dtype=torch.long),
            row["window_id"]
        )

# Model D1: Flow-Only Baseline (65,730 params)
class ModelD1_FlowOnly(nn.Module):
    def __init__(self, dropout_p=0.5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(64, 2)
        )

    def forward(self, x_flow):
        # x_flow: (B, 49, 512)
        mean_feat = torch.mean(x_flow, dim=1) # (B, 512)
        std_feat  = torch.std(x_flow, dim=1)  # (B, 512)
        pooled = torch.cat([mean_feat, std_feat], dim=1) # (B, 1024)
        return self.classifier(pooled)

    def get_parameter_counts(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"trainable": trainable}

# Model D3: Dual-Stream RGB + Flow Fusion (131,266 params)
class ModelD3_RGBFlowFusion(nn.Module):
    def __init__(self, dropout_p=0.5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(2048, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(64, 2)
        )

    def forward(self, x_rgb, x_flow):
        # x_rgb: (B, 50, 512), x_flow: (B, 49, 512)
        rgb_mean, rgb_std   = torch.mean(x_rgb, dim=1), torch.std(x_rgb, dim=1)
        flow_mean, flow_std = torch.mean(x_flow, dim=1), torch.std(x_flow, dim=1)

        rgb_vec  = torch.cat([rgb_mean, rgb_std], dim=1)   # (B, 1024)
        flow_vec = torch.cat([flow_mean, flow_std], dim=1) # (B, 1024)

        fused = torch.cat([rgb_vec, flow_vec], dim=1) # (B, 2048)
        return self.classifier(fused)

    def get_parameter_counts(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"trainable": trainable}

def train_flow_experiments():
    print("=" * 70)
    print("EXPERIMENT D: OPTICAL FLOW & DUAL-STREAM FUSION TRAINING")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device}")

    flow_manifest_path = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "processed_flow_features_manifest.csv")
    df_manifest = pd.read_csv(flow_manifest_path)
    total_samples = len(df_manifest)
    print(f"Loaded Flow Feature Manifest: {total_samples} samples")

    base_ckpt_dir = os.path.join(ROOT_DIR, "checkpoints", "le2i_optical_flow")
    results_dir   = os.path.join(ROOT_DIR, "R&D", "ML_Baseline", "results", "le2i_optical_flow")
    os.makedirs(results_dir, exist_ok=True)

    models_meta = {
        "flow": {"name": "Model D1 (Optical Flow-Only)", "type": "flow", "dir": "flow"},
        "rgb_control": {"name": "Model D2 (RGB Control)", "type": "rgb", "dir": "rgb_control"},
        "rgb_flow": {"name": "Model D3 (RGB+Flow Fusion)", "type": "fusion", "dir": "rgb_flow"}
    }

    folds = {
        "Fold 1": {"test": ["Coffee_room_01"], "train": ["Coffee_room_02", "Home_01", "Home_02"]},
        "Fold 2": {"test": ["Coffee_room_02"], "train": ["Coffee_room_01", "Home_01", "Home_02"]},
        "Fold 3": {"test": ["Home_01"], "train": ["Coffee_room_01", "Coffee_room_02", "Home_02"]},
        "Fold 4": {"test": ["Home_02"], "train": ["Coffee_room_01", "Coffee_room_02", "Home_01"]}
    }

    all_results = []
    start_total_time = time.perf_counter()

    for m_key, m_info in models_meta.items():
        print("\n" + "#" * 70)
        print(f"STARTING MODEL VARIANT: {m_info['name'].upper()}")
        print("#" * 70)

        m_ckpt_dir = os.path.join(base_ckpt_dir, m_info["dir"])
        os.makedirs(m_ckpt_dir, exist_ok=True)

        for fold_num, (fold_name, f_info) in enumerate(folds.items(), start=1):
            print("\n" + "-" * 70)
            print(f"[{m_info['name']}] {fold_name}: Outer Test = {f_info['test'][0]}")
            print("-" * 70)

            set_seed(42)

            test_locs = f_info["test"]
            train_locs = f_info["train"]

            outer_train_df = df_manifest[df_manifest["location"].isin(train_locs)].copy()
            outer_test_df  = df_manifest[df_manifest["location"].isin(test_locs)].copy()

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

            # Datasets & Loaders
            ds_tr   = Le2iDualStreamDataset(inner_tr_df, ROOT_DIR)
            ds_val  = Le2iDualStreamDataset(inner_val_df, ROOT_DIR)
            ds_test = Le2iDualStreamDataset(outer_test_df, ROOT_DIR)

            loader_tr   = DataLoader(ds_tr, batch_size=32, shuffle=True)
            loader_val  = DataLoader(ds_val, batch_size=32, shuffle=False)
            loader_test = DataLoader(ds_test, batch_size=32, shuffle=False)

            # Instantiate model
            if m_info["type"] == "flow":
                model = ModelD1_FlowOnly().to(device)
            elif m_info["type"] == "rgb":
                model = ModelD2_RGBControl().to(device)
            else:
                model = ModelD3_RGBFlowFusion().to(device)

            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
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
                for batch_rgb, batch_flow, batch_y, _ in loader_tr:
                    batch_rgb, batch_flow, batch_y = batch_rgb.to(device), batch_flow.to(device), batch_y.to(device)
                    optimizer.zero_grad()

                    if m_info["type"] == "flow":
                        logits = model(batch_flow)
                    elif m_info["type"] == "rgb":
                        logits = model(batch_rgb)
                    else:
                        logits = model(batch_rgb, batch_flow)

                    loss = criterion(logits, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * len(batch_y)

                # Inner Val Evaluation
                model.eval()
                val_probs, val_targets = [], []
                with torch.no_grad():
                    for batch_rgb, batch_flow, batch_y, _ in loader_val:
                        batch_rgb, batch_flow = batch_rgb.to(device), batch_flow.to(device)

                        if m_info["type"] == "flow":
                            logits = model(batch_flow)
                        elif m_info["type"] == "rgb":
                            logits = model(batch_rgb)
                        else:
                            logits = model(batch_rgb, batch_flow)

                        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                        val_probs.extend(probs)
                        val_targets.extend(batch_y.numpy())

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
                for batch_rgb, batch_flow, batch_y, _ in loader_val:
                    batch_rgb, batch_flow = batch_rgb.to(device), batch_flow.to(device)
                    if m_info["type"] == "flow":
                        logits = model(batch_flow)
                    elif m_info["type"] == "rgb":
                        logits = model(batch_rgb)
                    else:
                        logits = model(batch_rgb, batch_flow)

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
                for batch_rgb, batch_flow, batch_y, _ in loader_test:
                    batch_rgb, batch_flow = batch_rgb.to(device), batch_flow.to(device)
                    if m_info["type"] == "flow":
                        logits = model(batch_flow)
                    elif m_info["type"] == "rgb":
                        logits = model(batch_rgb)
                    else:
                        logits = model(batch_rgb, batch_flow)

                    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    test_probs.extend(probs)
                    test_targets.extend(batch_y.numpy())

            test_latency_ms = ((time.perf_counter() - test_start_time) / len(ds_test)) * 1000.0

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

            all_results.append({
                "model_key": m_key,
                "model_name": m_info["name"],
                "trainable_params": trainable_params,
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
    df_results = pd.DataFrame(all_results)
    results_csv_path = os.path.join(results_dir, "flow_fold_results.csv")
    df_results.to_csv(results_csv_path, index=False)

    print("\n" + "=" * 70)
    print("EXPERIMENT D: OPTICAL FLOW SUMMARY")
    print("=" * 70)
    print(f"Total Execution Time: {total_exp_time:.2f} seconds")
    print(f"Results CSV Saved   : {results_csv_path}")

    print("\nLOLO MEAN METRICS ACROSS EXPERIMENT D VARIANTS (@ tau = 0.50):")
    for m_key in ["flow", "rgb_control", "rgb_flow"]:
        sub = df_results[df_results["model_key"] == m_key]
        print(f"\n{sub['model_name'].iloc[0]} ({sub['trainable_params'].iloc[0]:,} params):")
        print(f"  Accuracy   : {sub['acc_050'].mean():.4f} ± {sub['acc_050'].std():.4f}")
        print(f"  Precision  : {sub['prec_050'].mean():.4f} ± {sub['prec_050'].std():.4f}")
        print(f"  Recall/Sens: {sub['sens_050'].mean():.4f} ± {sub['sens_050'].std():.4f}")
        print(f"  Specificity: {sub['spec_050'].mean():.4f} ± {sub['spec_050'].std():.4f}")
        print(f"  F1 Score   : {sub['f1_050'].mean():.4f} ± {sub['f1_050'].std():.4f}")
        print(f"  Event Sens : {sub['event_sens_050'].mean():.2f}% ± {sub['event_sens_050'].std():.2f}%")

    print("=" * 70)

    return df_results

if __name__ == "__main__":
    train_flow_experiments()
