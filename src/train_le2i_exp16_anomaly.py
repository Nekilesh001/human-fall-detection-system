"""
Experiment #16: Unsupervised Anomaly Detection Benchmark Pipeline.
Models:
1. M16-A: 1D Conv Autoencoder (84,763 params, MSE Reconstruction Error)
2. M16-B: One-Class SVM (RBF Kernel, 374-D Pooled Features)
3. M16-C: Isolation Forest (100 Trees, 374-D Pooled Features)

Safety & Protocol Rules:
- All models trained EXCLUSIVELY ON NORMAL SAMPLES (y=0) from outer training partitions.
- Zero fall samples exposed to models during fitting.
- Threshold tau* selected strictly on inner validation predictions.
- 4-Fold LOLO evaluation on unseen outer test locations.
- Inputs: 187-D spatial feature tensors precomputed in Exp K1 (processed_data/Le2i_baseline/pose_estimator_features/yolo_pose_k1/).
"""

import os
import sys
import random
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT_DIR = r"d:\ONE_DATA\Fall detection"
sys.path.insert(0, ROOT_DIR)

from src.train_le2i_yolo_temporal import compute_metrics, set_seed

# ----------------------------------------------------------------------
# 1. Model M16-A: 1D Conv Autoencoder (84,763 params)
# ----------------------------------------------------------------------
class ConvAutoencoder1D(nn.Module):
    def __init__(self, in_c=187, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_c, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, latent_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, in_c, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # x: (B, 50, 187) -> permute to (B, 187, 50)
        x_t = x.permute(0, 2, 1)
        z = self.encoder(x_t)
        recon_t = self.decoder(z)
        recon = recon_t.permute(0, 2, 1) # (B, 50, 187)
        return recon

def compute_ae_anomaly_score(model, tensor_b, device):
    # tensor_b: (B, 50, 187)
    model.eval()
    with torch.no_grad():
        tensor_b = tensor_b.to(device)
        recon_b = model(tensor_b)
        # MSE per window: average over (50, 187)
        mse_per_win = torch.mean((tensor_b - recon_b) ** 2, dim=(1, 2)).cpu().numpy()
    return mse_per_win

# ----------------------------------------------------------------------
# 2. Dataset Loader for Anomaly Detection
# ----------------------------------------------------------------------
class AnomalyDataset(Dataset):
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
            feat_187 = d["features"].astype(np.float32) # (50, 187)
        
        return torch.tensor(feat_187, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def extract_pooled_features(df, feature_dir):
    # Returns (N, 374) float32 [Mean(187) + Std(187)] and labels (N,)
    X_list, Y_list = [], []
    for _, row in df.iterrows():
        wid = row["window_id"]
        raw_label = str(row["label"]).upper()
        label = 1 if raw_label in ["FALL", "1"] else 0
        fpath = os.path.join(feature_dir, f"{wid}.npz")
        with np.load(fpath) as d:
            feat = d["features"].astype(np.float32) # (50, 187)
        mean_feat = np.mean(feat, axis=0) # 187
        std_feat  = np.std(feat, axis=0)  # 187
        pooled = np.concatenate([mean_feat, std_feat]) # 374
        X_list.append(pooled)
        Y_list.append(label)
    return np.array(X_list, dtype=np.float32), np.array(Y_list, dtype=np.int64)

# ----------------------------------------------------------------------
# 3. Anomaly Threshold Tuning (Optimizing F1 on Validation)
# ----------------------------------------------------------------------
def find_best_anomaly_threshold(y_true, scores):
    min_s, max_s = np.min(scores), np.max(scores)
    thresholds = np.linspace(min_s, max_s, 200)
    best_tau, best_f1 = 0.5, -1.0
    for tau in thresholds:
        preds = (scores >= tau).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, preds, average='binary', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = tau
    return float(best_tau), float(best_f1)

# ----------------------------------------------------------------------
# 4. Main Experiment #16 Training Loop
# ----------------------------------------------------------------------
def train_le2i_exp16_anomaly():
    print("=" * 70)
    print("EXPERIMENT #16: UNSUPERVISED ANOMALY DETECTION BENCHMARK")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    manifest_path = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "processed_pose_features_manifest.csv")
    df_manifest = pd.read_csv(manifest_path).sort_values("window_id").reset_index(drop=True)
    k1_dir = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "pose_estimator_features", "yolo_pose_k1")

    locations = ["Coffee_room_01", "Coffee_room_02", "Home_01", "Home_02"]
    
    ckpt_root = os.path.join(ROOT_DIR, "checkpoints", "le2i_exp16_anomaly")
    ckpt_ae_dir     = os.path.join(ckpt_root, "ae")
    ckpt_ocsvm_dir  = os.path.join(ckpt_root, "ocsvm")
    ckpt_iforest_dir= os.path.join(ckpt_root, "iforest")
    
    os.makedirs(ckpt_ae_dir, exist_ok=True)
    os.makedirs(ckpt_ocsvm_dir, exist_ok=True)
    os.makedirs(ckpt_iforest_dir, exist_ok=True)

    results_matrix = {
        "m16_a_conv_ae": {"name": "Model M16-A: 1D Conv Autoencoder", "parameters": 84763, "folds": []},
        "m16_b_ocsvm":   {"name": "Model M16-B: One-Class SVM",        "parameters": "Non-parametric", "folds": []},
        "m16_c_iforest": {"name": "Model M16-C: Isolation Forest",     "parameters": "Non-parametric", "folds": []}
    }

    for fold_idx, test_loc in enumerate(locations, 1):
        set_seed(42 + fold_idx)
        print(f"\n" + "-" * 60)
        print(f"FOLD {fold_idx} / 4 — TEST LOCATION: {test_loc}")
        print("-" * 60)

        test_df = df_manifest[df_manifest["location"] == test_loc].reset_index(drop=True)
        train_val_df = df_manifest[df_manifest["location"] != test_loc].reset_index(drop=True)

        # STRICT SAFETY AUDIT: Extract ONLY NORMAL samples (y=0) for model fitting
        normal_train_val_df = train_val_df[train_val_df["label"].str.upper() == "NORMAL"].reset_index(drop=True)
        fall_train_val_df   = train_val_df[train_val_df["label"].str.upper() == "FALL"].reset_index(drop=True)

        print(f"  Outer Train Partition: {len(train_val_df)} Total (Normal={len(normal_train_val_df)}, Fall={len(fall_train_val_df)})")
        print(f"  Outer Test Partition : {len(test_df)} Total (Normal={(test_df['label']=='NORMAL').sum()}, Fall={(test_df['label']=='FALL').sum()})")
        assert len(normal_train_val_df) > 0, "No normal training samples found!"

        # Inner Train / Validation Split (Grouped by event_id)
        unique_events = train_val_df["event_id"].unique()
        event_labels = [1 if str(train_val_df[train_val_df["event_id"] == ev]["label"].iloc[0]).upper() in ["FALL", "1"] else 0 for ev in unique_events]
        
        try:
            tr_events, val_events = train_test_split(unique_events, test_size=0.20, random_state=42, stratify=event_labels)
        except ValueError:
            tr_events, val_events = train_test_split(unique_events, test_size=0.20, random_state=42)

        # Inner train for AE fitting: STRICTLY NORMAL windows in tr_events
        inner_train_norm_df = train_val_df[(train_val_df["event_id"].isin(tr_events)) & (train_val_df["label"].str.upper() == "NORMAL")].reset_index(drop=True)
        
        # Inner val for threshold tuning: ALL windows in val_events (Normal + Fall)
        inner_val_df = train_val_df[train_val_df["event_id"].isin(val_events)].reset_index(drop=True)

        print(f"  [SAFETY VERIFIED] Normal-only Training Samples Used for Model Fitting: {len(inner_train_norm_df)}")

        # --------------------------------------------------------------
        # M16-A: 1D Conv Autoencoder Training
        # --------------------------------------------------------------
        print(f"\n  Training M16-A: 1D Conv Autoencoder (84,763 params)...")
        ae_train_ds = AnomalyDataset(inner_train_norm_df, k1_dir)
        ae_val_ds   = AnomalyDataset(inner_val_df, k1_dir)
        ae_test_ds  = AnomalyDataset(test_df, k1_dir)

        ae_train_loader = DataLoader(ae_train_ds, batch_size=32, shuffle=True)
        ae_val_loader   = DataLoader(ae_val_ds, batch_size=32, shuffle=False)
        ae_test_loader  = DataLoader(ae_test_ds, batch_size=32, shuffle=False)

        model_ae = ConvAutoencoder1D().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model_ae.parameters(), lr=1e-3, weight_decay=1e-4)

        best_val_loss = float('inf')
        ckpt_ae_path = os.path.join(ckpt_ae_dir, f"fold_{fold_idx}_best.pth")

        for epoch in range(1, 101):
            model_ae.train()
            for x_b, _ in ae_train_loader:
                x_b = x_b.to(device)
                optimizer.zero_grad()
                recon_b = model_ae(x_b)
                loss = criterion(recon_b, x_b)
                loss.backward()
                optimizer.step()

            model_ae.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_b, _ in ae_val_loader:
                    x_b = x_b.to(device)
                    recon_b = model_ae(x_b)
                    val_loss += criterion(recon_b, x_b).item() * x_b.size(0)
            val_loss /= len(ae_val_ds)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model_ae.state_dict(), ckpt_ae_path)

        model_ae.load_state_dict(torch.load(ckpt_ae_path))
        model_ae.eval()

        # Inner Val Anomaly Scores & Threshold Tuning
        val_ae_scores, val_targets = [], []
        for x_b, y_b in ae_val_loader:
            scores = compute_ae_anomaly_score(model_ae, x_b, device)
            val_ae_scores.extend(scores)
            val_targets.extend(y_b.numpy())
        
        tau_ae, val_f1_ae = find_best_anomaly_threshold(np.array(val_targets), np.array(val_ae_scores))

        # Outer Test Evaluation
        test_ae_scores, test_targets = [], []
        for x_b, y_b in ae_test_loader:
            scores = compute_ae_anomaly_score(model_ae, x_b, device)
            test_ae_scores.extend(scores)
            test_targets.extend(y_b.numpy())

        test_ae_scores = np.array(test_ae_scores)
        test_targets   = np.array(test_targets)

        m_ae_def = compute_metrics(test_targets, test_ae_scores, threshold=0.01) # Nominal default
        m_ae_opt = compute_metrics(test_targets, test_ae_scores, threshold=tau_ae)

        results_matrix["m16_a_conv_ae"]["folds"].append({
            "fold": fold_idx, "test_location": test_loc, "tau_star": tau_ae, "val_f1_star": val_f1_ae,
            "f1_default": m_ae_def["f1"], "f1_optimal": m_ae_opt["f1"], "precision": m_ae_opt["precision"],
            "recall": m_ae_opt["recall"], "specificity": m_ae_opt["specificity"], "roc_auc": m_ae_opt["roc_auc"],
            "pr_auc": m_ae_opt["pr_auc"], "tp": m_ae_opt["tp"], "fp": m_ae_opt["fp"], "tn": m_ae_opt["tn"], "fn": m_ae_opt["fn"]
        })
        print(f"  M16-A Conv-AE  | Tau*={tau_ae:.5f}: F1={m_ae_opt['f1']:.4f} | Rec={m_ae_opt['recall']:.4f} | Spec={m_ae_opt['specificity']:.4f}")

        # --------------------------------------------------------------
        # M16-B: One-Class SVM Fitting
        # --------------------------------------------------------------
        print(f"  Fitting M16-B: One-Class SVM (RBF Kernel, 374-D)...")
        X_tr_norm, _ = extract_pooled_features(inner_train_norm_df, k1_dir)
        X_val, Y_val = extract_pooled_features(inner_val_df, k1_dir)
        X_te, Y_te   = extract_pooled_features(test_df, k1_dir)

        ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
        ocsvm.fit(X_tr_norm)

        ckpt_ocsvm_path = os.path.join(ckpt_ocsvm_dir, f"fold_{fold_idx}_model.pkl")
        with open(ckpt_ocsvm_path, "wb") as f:
            pickle.dump(ocsvm, f)

        # Anomaly score: negative decision function (higher = more anomalous)
        val_ocsvm_scores = -ocsvm.decision_function(X_val)
        tau_ocsvm, val_f1_ocsvm = find_best_anomaly_threshold(Y_val, val_ocsvm_scores)

        test_ocsvm_scores = -ocsvm.decision_function(X_te)
        m_ocsvm_def = compute_metrics(Y_te, test_ocsvm_scores, threshold=0.0)
        m_ocsvm_opt = compute_metrics(Y_te, test_ocsvm_scores, threshold=tau_ocsvm)

        results_matrix["m16_b_ocsvm"]["folds"].append({
            "fold": fold_idx, "test_location": test_loc, "tau_star": tau_ocsvm, "val_f1_star": val_f1_ocsvm,
            "f1_default": m_ocsvm_def["f1"], "f1_optimal": m_ocsvm_opt["f1"], "precision": m_ocsvm_opt["precision"],
            "recall": m_ocsvm_opt["recall"], "specificity": m_ocsvm_opt["specificity"], "roc_auc": m_ocsvm_opt["roc_auc"],
            "pr_auc": m_ocsvm_opt["pr_auc"], "tp": m_ocsvm_opt["tp"], "fp": m_ocsvm_opt["fp"], "tn": m_ocsvm_opt["tn"], "fn": m_ocsvm_opt["fn"]
        })
        print(f"  M16-B OC-SVM   | Tau*={tau_ocsvm:.5f}: F1={m_ocsvm_opt['f1']:.4f} | Rec={m_ocsvm_opt['recall']:.4f} | Spec={m_ocsvm_opt['specificity']:.4f}")

        # --------------------------------------------------------------
        # M16-C: Isolation Forest Fitting
        # --------------------------------------------------------------
        print(f"  Fitting M16-C: Isolation Forest (100 Trees, 374-D)...")
        iforest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42 + fold_idx)
        iforest.fit(X_tr_norm)

        ckpt_iforest_path = os.path.join(ckpt_iforest_dir, f"fold_{fold_idx}_model.pkl")
        with open(ckpt_iforest_path, "wb") as f:
            pickle.dump(iforest, f)

        # Anomaly score: negative score_samples (higher = more anomalous)
        val_iforest_scores = -iforest.score_samples(X_val)
        tau_iforest, val_f1_iforest = find_best_anomaly_threshold(Y_val, val_iforest_scores)

        test_iforest_scores = -iforest.score_samples(X_te)
        m_iforest_def = compute_metrics(Y_te, test_iforest_scores, threshold=0.5)
        m_iforest_opt = compute_metrics(Y_te, test_iforest_scores, threshold=tau_iforest)

        results_matrix["m16_c_iforest"]["folds"].append({
            "fold": fold_idx, "test_location": test_loc, "tau_star": tau_iforest, "val_f1_star": val_f1_iforest,
            "f1_default": m_iforest_def["f1"], "f1_optimal": m_iforest_opt["f1"], "precision": m_iforest_opt["precision"],
            "recall": m_iforest_opt["recall"], "specificity": m_iforest_opt["specificity"], "roc_auc": m_iforest_opt["roc_auc"],
            "pr_auc": m_iforest_opt["pr_auc"], "tp": m_iforest_opt["tp"], "fp": m_iforest_opt["fp"], "tn": m_iforest_opt["tn"], "fn": m_iforest_opt["fn"]
        })
        print(f"  M16-C iForest  | Tau*={tau_iforest:.5f}: F1={m_iforest_opt['f1']:.4f} | Rec={m_iforest_opt['recall']:.4f} | Spec={m_iforest_opt['specificity']:.4f}")

    # Compute Aggregate LOLO Summary for each model
    for key, data in results_matrix.items():
        f1_opt_list = [r["f1_optimal"] for r in data["folds"]]
        f1_def_list = [r["f1_default"] for r in data["folds"]]
        data["lolo_f1_default_mean"] = float(np.mean(f1_def_list))
        data["lolo_f1_optimal_mean"] = float(np.mean(f1_opt_list))
        data["lolo_f1_optimal_std"]  = float(np.std(f1_opt_list))

    print("\n" + "=" * 70)
    print("EXPERIMENT #16 LOLO BENCHMARK SUMMARY")
    print("=" * 70)
    for key, data in results_matrix.items():
        print(f"  {data['name']:35s} | LOLO F1 (@ Tau*): {data['lolo_f1_optimal_mean']*100:.2f}% ± {data['lolo_f1_optimal_std']*100:.2f}%")

    res_dir = os.path.join(ROOT_DIR, "R&D", "ML_Baseline", "results")
    os.makedirs(res_dir, exist_ok=True)
    
    json_path = os.path.join(res_dir, "exp16_anomaly_benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(results_matrix, f, indent=2)

    csv_rows = []
    for key, data in results_matrix.items():
        for r in data["folds"]:
            csv_rows.append({
                "model_key": key,
                "model_name": data["name"],
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

    csv_path = os.path.join(res_dir, "exp16_anomaly_benchmark_results.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    print(f"\nResults Saved: {json_path} & {csv_path}")

if __name__ == "__main__":
    train_le2i_exp16_anomaly()
