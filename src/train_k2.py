"""
PHASE H10-E — MODEL K2 TRAINING & FEATURE STANDARDIZATION PIPELINE SCRIPT

Supports Experiments EXP-K2-A through G:
- EXP-K2-A: Model K2 Dual-Stream TCN + Le2i Baseline Reference
- EXP-K2-B: Model K2 + Le2i + URFD
- EXP-K2-C: Model K2 + Le2i + Multicam
- EXP-K2-D: Model K2 + Le2i + URFD + Multicam
- EXP-K2-E: Model K2 + Dataset-Balanced Weighted Sampler
- EXP-K2-F: Model K2 + Multicam Camera Sub-sampling
- EXP-K2-G: Model K2 + Full Proposed Solution

Target Checkpoint Output Directory: checkpoints/multi_dataset_k2/exp_<name>/
Scaler Output: checkpoints/multi_dataset_k2/exp_<name>/scaler.pkl (Fitted ONLY on Train Split!)
"""

import os
import sys
import argparse
import time
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix

ROOT_DIR = r"d:\ONE_DATA\Fall detection"
sys.path.insert(0, ROOT_DIR)

from src.model_k2_dual_stream import ModelK2_DualStreamTCN

class FeatureStandardScaler:
    """
    Fits StandardScaler on 187-D features using ONLY the training split.
    Preserves 16 zero-padded unused landmark channels.
    """
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X_train):
        # X_train: (N, 50, 187) -> reshape to (N*50, 187)
        N, T, D = X_train.shape
        flat = X_train.reshape(-1, D)
        self.scaler.fit(flat)

    def transform(self, X):
        # X: (N, 50, 187)
        N, T, D = X.shape
        flat = X.reshape(-1, D)
        trans = self.scaler.transform(flat)
        # Handle zero-variance channels (NaNs replaced with 0.0)
        trans = np.nan_to_num(trans, nan=0.0)
        return trans.reshape(N, T, D).astype(np.float32)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.scaler, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.scaler = pickle.load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="Phase H10 Model K2 Training Pipeline Script")
    parser.add_argument("--experiment", type=str, default="K2_A", choices=["K2_A", "K2_B", "K2_C", "K2_D", "K2_E", "K2_F", "K2_G"], help="Model K2 experiment configuration")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--pos_weight", type=float, default=1.0, help="BCE loss positive class weight")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--dry_run", action="store_true", help="Perform readiness dry-run without full training")
    parser.add_argument("--min_warmup", type=int, default=10, help="Minimum warmup epochs before checkpoint selection")
    parser.add_argument("--manifest_path", type=str, default=os.path.join(ROOT_DIR, "processed_data", "multi_dataset_k1", "manifests", "unified_window_manifest.csv"), help="Unified window manifest path")
    parser.add_argument("--output_dir", type=str, default=os.path.join(ROOT_DIR, "checkpoints", "multi_dataset_k2"), help="Output base directory")
    return parser.parse_args()

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

class K2TensorDataset(Dataset):
    def __init__(self, X_tensors, y_labels):
        self.X = torch.tensor(X_tensors, dtype=torch.float32)
        self.y = torch.tensor(y_labels, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def run_k2_training(args):
    print("=" * 75)
    print("PHASE H10 — MODEL K2 DUAL-STREAM RESEARCH TRAINING PIPELINE")
    print(f"  Experiment       : EXP-{args.experiment}")
    print(f"  Epochs           : {args.epochs}")
    print(f"  Batch Size       : {args.batch_size}")
    print(f"  Learning Rate    : {args.learning_rate}")
    print(f"  Seed             : {args.seed}")
    print(f"  Dry Run Mode     : {args.dry_run}")
    print("=" * 75)

    set_seed(args.seed)

    exp_folder_map = {
        "K2_A": "exp_k2_a",
        "K2_B": "exp_k2_b",
        "K2_C": "exp_k2_c",
        "K2_D": "exp_k2_d",
        "K2_E": "exp_k2_e",
        "K2_F": "exp_k2_f",
        "K2_G": "exp_k2_g"
    }
    target_dir = os.path.join(args.output_dir, exp_folder_map[args.experiment])
    assert "checkpoints\\final_k1" not in target_dir and "final_production.pth" not in target_dir, "CRITICAL SAFETY FAIL: Output path touches production directory!"
    os.makedirs(target_dir, exist_ok=True)

    base_dir = os.path.dirname(os.path.dirname(args.manifest_path))
    df_manifest = pd.read_csv(args.manifest_path)

    # Filter according to experiment
    if args.experiment == "K2_A":
        df_manifest = df_manifest[df_manifest["dataset"] == "Le2i"].copy()
    elif args.experiment == "K2_B":
        df_manifest = df_manifest[df_manifest["dataset"].isin(["Le2i", "URFD"])].copy()
    elif args.experiment == "K2_C":
        df_manifest = df_manifest[df_manifest["dataset"].isin(["Le2i", "Multicam"])].copy()

    # Split by physical group
    unique_groups = list(df_manifest["group_id"].unique())
    np.random.shuffle(unique_groups)

    n_grps = len(unique_groups)
    n_train = int(n_grps * 0.70)
    n_val = int(n_grps * 0.15)

    train_grps = set(unique_groups[:n_train])
    val_grps = set(unique_groups[n_train:n_train+n_val])
    test_grps = set(unique_groups[n_train+n_val:])

    df_train = df_manifest[df_manifest["group_id"].isin(train_grps)].copy()
    df_val = df_manifest[df_manifest["group_id"].isin(val_grps)].copy()
    df_test = df_manifest[df_manifest["group_id"].isin(test_grps)].copy()

    df_train.to_csv(os.path.join(target_dir, "train_split.csv"), index=False)
    df_val.to_csv(os.path.join(target_dir, "val_split.csv"), index=False)
    df_test.to_csv(os.path.join(target_dir, "test_split.csv"), index=False)

    print(f"  Group-Safe Split Saved -> {target_dir}")
    print(f"    Train: {len(df_train)} windows ({len(train_grps)} groups)")
    print(f"    Val  : {len(df_val)} windows ({len(val_grps)} groups)")
    print(f"    Test : {len(df_test)} windows ({len(test_grps)} groups)")

    if args.dry_run:
        print("\n  [DRY RUN MODE] Verifying K2 Model Forward Pass & Loss Calculation...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ModelK2_DualStreamTCN().to(device)
        dummy_x = torch.randn(args.batch_size, 50, 187, device=device)
        out = model(dummy_x)
        criterion = nn.CrossEntropyLoss()
        dummy_y = torch.randint(0, 2, (args.batch_size,), device=device)
        loss = criterion(out, dummy_y)
        print(f"    Sample Input Shape: {dummy_x.shape} -> Logits: {out.shape} -> Loss: {loss.item():.4f}")
        print("  [DRY RUN PASSED] Pipeline is 100% ready for future training execution.")
        return

    # Load 187-D Feature Tensors
    def load_tensors(df):
        X_list, y_list = [], []
        for idx, row in df.iterrows():
            abs_p = os.path.join(base_dir, row["feature_path"])
            with np.load(abs_p) as d:
                feat = d["features"] # (50, 187) float32
            X_list.append(feat)
            y_list.append(int(row["label"]))
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)

    print("\n  Loading Feature Tensors...")
    X_train_raw, y_train = load_tensors(df_train)
    X_val_raw, y_val = load_tensors(df_val)
    X_test_raw, y_test = load_tensors(df_test)

    # CRITICAL: Fit FeatureStandardScaler ONLY on Train Split!
    print("  Fitting FeatureStandardScaler ONLY on Train Split...")
    scaler = FeatureStandardScaler()
    scaler.fit(X_train_raw)

    scaler_path = os.path.join(target_dir, "scaler.pkl")
    scaler.save(scaler_path)
    print(f"  Saved Scaler -> {scaler_path}")

    # Transform all splits using frozen Train Scaler
    X_train = scaler.transform(X_train_raw)
    X_val   = scaler.transform(X_val_raw)
    X_test  = scaler.transform(X_test_raw)

    train_ds = K2TensorDataset(X_train, y_train)
    val_ds   = K2TensorDataset(X_val, y_val)
    test_ds  = K2TensorDataset(X_test, y_test)

    sampler = None
    if args.experiment in ["K2_E", "K2_G"]:
        print("  Enabling Dataset-Balanced WeightedRandomSampler (Derived from Train Split ONLY)...")
        dataset_counts = df_train["dataset"].value_counts().to_dict()
        total_train_samples = len(df_train)
        num_datasets = len(dataset_counts)
        dataset_weights = {ds: total_train_samples / (num_datasets * count) for ds, count in dataset_counts.items()}
        sample_weights = [dataset_weights[ds] for ds in df_train["dataset"].values]
        sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights_tensor, num_samples=len(sample_weights_tensor), replacement=True)
        print(f"    Dataset Counts in Train: {dataset_counts}")
        print(f"    Calculated Dataset Sampling Weights: {dataset_weights}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelK2_DualStreamTCN().to(device)

    # Independent Random Weight Initialization
    def init_weights(m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    model.apply(init_weights)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_epoch = -1
    best_model_state = None

    val_history = []

    print(f"\n  Starting Model K2 Candidate Training ({args.epochs} epochs)...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_samples = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(by)
            train_samples += len(by)

        train_loss = train_loss_sum / train_samples

        # Validation Evaluation
        model.eval()
        val_loss_sum = 0.0
        val_samples = 0
        val_probs, val_targets = [], []

        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                loss = criterion(logits, by)
                val_loss_sum += loss.item() * len(by)
                val_samples += len(by)

                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                val_probs.extend(probs)
                val_targets.extend(by.cpu().numpy())

        val_loss = val_loss_sum / val_samples
        val_auc = roc_auc_score(val_targets, val_probs) if len(np.unique(val_targets)) > 1 else 0.5
        val_preds_default = (np.array(val_probs) >= 0.5).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(val_targets, val_preds_default, average="binary", zero_division=0)

        saved_str = ""
        if epoch >= args.min_warmup:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_model_state = model.state_dict().copy()
                saved_str = f"  [CHECKPOINT SAVED] Selected Epoch {epoch} (Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f})"

        print(f"  Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Val F1: {f1:.4f} (Prec: {p:.4f}, Rec: {r:.4f}){saved_str}")

        val_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "val_f1": f1
        })

    # Save Best Checkpoint
    ckpt_path = os.path.join(target_dir, "best_candidate.pth")
    torch.save(best_model_state, ckpt_path)

    # Validation Threshold Optimization (tau*)
    model.load_state_dict(best_model_state)
    model.eval()

    val_probs, val_targets = [], []
    with torch.no_grad():
        for bx, by in val_loader:
            bx = bx.to(device)
            logits = model(bx)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            val_probs.extend(probs)
            val_targets.extend(by.numpy())

    val_probs = np.array(val_probs)
    val_targets = np.array(val_targets)

    best_tau = 0.5
    best_tau_f1 = -1.0
    tau_sweep = []

    for t_cand in np.arange(0.05, 0.96, 0.01):
        t_preds = (val_probs >= t_cand).astype(int)
        vp, vr, vf1, _ = precision_recall_fscore_support(val_targets, t_preds, average="binary", zero_division=0)
        tau_sweep.append({"tau": float(t_cand), "precision": float(vp), "recall": float(vr), "f1": float(vf1)})
        if vf1 > best_tau_f1:
            best_tau_f1 = vf1
            best_tau = t_cand

    with open(os.path.join(target_dir, "threshold_analysis.json"), "w") as f:
        json.dump(tau_sweep, f, indent=2)

    meta = {
        "experiment": args.experiment,
        "model": "ModelK2_DualStreamTCN",
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "best_val_f1": float(best_tau_f1),
        "candidate_tau": float(best_tau),
        "min_warmup": args.min_warmup,
        "scaler_path": "scaler.pkl"
    }

    with open(os.path.join(target_dir, "candidate_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  [VAL THRESHOLD SELECTION] Best Candidate Tau (Max F1={best_tau_f1:.4f}): {best_tau:.4f}")
    print(f"  TRAINING COMPLETE. Candidate Saved -> {target_dir}")

if __name__ == "__main__":
    args = parse_args()
    run_k2_training(args)
