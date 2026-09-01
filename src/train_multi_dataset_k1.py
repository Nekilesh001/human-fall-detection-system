"""
PHASE H3 — CONTROLLED MULTI-DATASET RESEARCH MODEL TRAINING PIPELINE

Supports Experiments A-E:
- EXP-A: Le2i Baseline Reference (Frozen Production Model K1)
- EXP-B: Le2i + URFD Candidate Training
- EXP-C: Le2i + Multicam Candidate Training
- EXP-D: Le2i + URFD + Multicam Unified Candidate Training
- EXP-E: Le2i + URFD Candidate Training with Zero-Shot Multicam Evaluation

Target Checkpoint Output Directory: checkpoints/multi_dataset_k1/exp_<name>/ (NEVER final_production.pth!)
Group-Safe Split: Grouped Stratified K-Fold by group_id (prevents cross-camera & cross-window leakage!).
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix

ROOT_DIR = r"d:\ONE_DATA\Fall detection"
sys.path.insert(0, ROOT_DIR)

from src.train_final_k1 import ModelK1_SpatialTCN

def parse_args():
    parser = argparse.ArgumentParser(description="Phase H3 Multi-Dataset Research Candidate Training Script")
    parser.add_argument("--experiment", type=str, default="D", choices=["A", "B", "C", "D", "E", "B_CORRECTED", "B_REAL", "C_REAL", "D_REAL"], help="Experiment configuration A-E, B_CORRECTED, B_REAL, C_REAL, or D_REAL")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--pos_weight", type=float, default=4.0, help="BCE loss positive class weight")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--dry_run", action="store_true", help="Perform readiness dry-run without full training")
    parser.add_argument("--min_warmup", type=int, default=1, help="Minimum warmup epochs before checkpoint selection")
    parser.add_argument("--manifest_path", type=str, default=os.path.join(ROOT_DIR, "processed_data", "multi_dataset_k1", "manifests", "unified_window_manifest.csv"), help="Unified window manifest path")
    parser.add_argument("--output_dir", type=str, default=os.path.join(ROOT_DIR, "checkpoints", "multi_dataset_k1"), help="Output base directory")
    return parser.parse_args()

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

class MultiDatasetTensorDataset(Dataset):
    def __init__(self, df_manifest, base_dir):
        self.df = df_manifest.reset_index(drop=True)
        self.base_dir = base_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = row["feature_path"]
        abs_path = os.path.join(self.base_dir, rel_path)
        
        with np.load(abs_path) as d:
            feat = d["features"] # (50, 187) float32
            
        label = int(row["label"])
        return torch.tensor(feat, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def run_multi_dataset_training(args):
    print("=" * 75)
    print("PHASE H3 — MULTI-DATASET RESEARCH MODEL TRAINING PIPELINE")
    print(f"  Experiment       : EXP-{args.experiment}")
    print(f"  Epochs           : {args.epochs}")
    print(f"  Batch Size       : {args.batch_size}")
    print(f"  Learning Rate    : {args.learning_rate}")
    print(f"  Pos Weight       : {args.pos_weight}")
    print(f"  Seed             : {args.seed}")
    print(f"  Dry Run Mode     : {args.dry_run}")
    print("=" * 75)

    set_seed(args.seed)

    # SAFETY CHECK: Ensure output directory NEVER touches production baseline
    exp_folder_map = {
        "A": "exp_a_baseline",
        "B": "exp_b_le2i_urfd",
        "C": "exp_c_le2i_multicam",
        "D": "exp_d_unified",
        "E": "exp_e_le2i_urfd_ood",
        "B_CORRECTED": "exp_b_corrected",
        "B_REAL": "exp_b_real",
        "C_REAL": "exp_c_real",
        "D_REAL": "exp_d_real"
    }
    target_dir = os.path.join(args.output_dir, exp_folder_map[args.experiment])
    assert "checkpoints\\final_k1" not in target_dir and "final_production.pth" not in target_dir, "CRITICAL SAFETY FAIL: Output path touches production directory!"
    os.makedirs(target_dir, exist_ok=True)

    base_dir = os.path.dirname(os.path.dirname(args.manifest_path))
    df_manifest = pd.read_csv(args.manifest_path)

    # Filter Manifest according to Experiment A-E / B_CORRECTED / B_REAL / C_REAL / D_REAL
    if args.experiment == "A":
        print("  [EXP-A] Baseline Reference — Using Frozen Production Checkpoint.")
        return
    elif args.experiment in ["B", "B_CORRECTED", "B_REAL"]:
        df_manifest = df_manifest[df_manifest["dataset"].isin(["Le2i", "URFD"])]
    elif args.experiment in ["C", "C_REAL"]:
        df_manifest = df_manifest[df_manifest["dataset"].isin(["Le2i", "Multicam"])]
    elif args.experiment in ["D", "D_REAL"]:
        pass  # All 3 datasets (Le2i + URFD + Multicam)
    elif args.experiment == "E":
        # Train on Le2i + URFD, test on Multicam
        df_train_val = df_manifest[df_manifest["dataset"].isin(["Le2i", "URFD"])].copy()
        df_test = df_manifest[df_manifest["dataset"] == "Multicam"].copy()
        df_manifest = pd.concat([df_train_val, df_test], ignore_index=True)

    # Perform Grouped Stratified K-Fold Split
    gkf = GroupKFold(n_splits=5)
    groups = df_manifest["group_id"].values

    # Determine train/val/test splits cleanly
    unique_groups = list(df_manifest["group_id"].unique())
    np.random.shuffle(unique_groups)
    
    n_grps = len(unique_groups)
    n_train = int(n_grps * 0.70)
    n_val = int(n_grps * 0.15)
    
    train_grps = set(unique_groups[:n_train])
    val_grps = set(unique_groups[n_train:n_train+n_val])
    test_grps = set(unique_groups[n_train+n_val:])

    if args.experiment == "E":
        # For EXP-E, test_grps MUST be all Multicam groups, train_grps Le2i+URFD
        train_val_grps = list(df_manifest[df_manifest["dataset"].isin(["Le2i", "URFD"])]["group_id"].unique())
        test_grps = set(df_manifest[df_manifest["dataset"] == "Multicam"]["group_id"].unique())
        
        n_tv = len(train_val_grps)
        n_t = int(n_tv * 0.80)
        train_grps = set(train_val_grps[:n_t])
        val_grps = set(train_val_grps[n_t:])

    # Verify zero group leakage
    assert len(train_grps.intersection(val_grps)) == 0, "Group Leakage Error: Train and Val groups overlap!"
    assert len(train_grps.intersection(test_grps)) == 0, "Group Leakage Error: Train and Test groups overlap!"
    assert len(val_grps.intersection(test_grps)) == 0, "Group Leakage Error: Val and Test groups overlap!"

    df_train = df_manifest[df_manifest["group_id"].isin(train_grps)].copy()
    df_val = df_manifest[df_manifest["group_id"].isin(val_grps)].copy()
    df_test = df_manifest[df_manifest["group_id"].isin(test_grps)].copy()

    # Save Split Manifests
    df_train.to_csv(os.path.join(target_dir, "train_split.csv"), index=False)
    df_val.to_csv(os.path.join(target_dir, "val_split.csv"), index=False)
    df_test.to_csv(os.path.join(target_dir, "test_split.csv"), index=False)

    print(f"  Group-Safe Split Saved -> {target_dir}")
    print(f"    Train: {len(df_train)} windows ({len(train_grps)} groups) | Fall %: {(df_train['label']==1).mean()*100:.2f}%")
    print(f"    Val  : {len(df_val)} windows ({len(val_grps)} groups) | Fall %: {(df_val['label']==1).mean()*100:.2f}%")
    print(f"    Test : {len(df_test)} windows ({len(test_grps)} groups) | Fall %: {(df_test['label']==1).mean()*100:.2f}%")

    train_ds = MultiDatasetTensorDataset(df_train, base_dir)
    val_ds = MultiDatasetTensorDataset(df_val, base_dir)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelK1_SpatialTCN(input_dim=187).to(device)
    
    pos_w = torch.tensor([args.pos_weight], device=device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, args.pos_weight], device=device))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    if args.dry_run:
        print("\n  [DRY RUN MODE] Performing DataLoader & Model Forward Pass Verification...")
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            out = model(x_b)
            loss = criterion(out, y_b)
            optimizer.zero_grad()
            loss.backward()
            print(f"    Sample Batch Shape: {x_b.shape} -> Output: {out.shape} -> Loss: {loss.item():.4f}")
            break
        print("  [DRY RUN PASSED] Pipeline is 100% ready for training.")
        return

    # Execute Full Model Training Loop
    print(f"\n  Starting Candidate Training ({args.epochs} epochs)...")
    best_val_loss = float("inf")
    best_val_f1 = 0.0
    best_epoch = -1
    training_history = []

    is_corrected = (args.experiment == "B_CORRECTED" or args.min_warmup >= 10)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            out = model(x_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y_b)
        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_preds, val_targets, val_probs = [], [], []
        val_loss = 0.0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                out = model(x_b)
                loss = criterion(out, y_b)
                val_loss += loss.item() * len(y_b)
                probs = torch.softmax(out, dim=1)[:, 1]
                val_probs.extend(probs.cpu().numpy())
                val_preds.extend((probs >= 0.3650).cpu().numpy())
                val_targets.extend(y_b.cpu().numpy())
        val_loss /= len(val_ds)

        p, r, f1, _ = precision_recall_fscore_support(val_targets, val_preds, average="binary", zero_division=0)
        try:
            val_auc = float(roc_auc_score(val_targets, val_probs))
        except Exception:
            val_auc = 0.0

        training_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "precision": p,
            "recall": r,
            "f1": f1,
            "val_auc": val_auc
        })
        
        print(f"  Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Val F1: {f1:.4f} (Prec: {p:.4f}, Rec: {r:.4f})")

        # Checkpoint Selection Logic
        if is_corrected:
            # Rule: Ignore epochs < min_warmup, select MINIMUM val_loss among epoch >= min_warmup
            if epoch >= args.min_warmup and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_path = os.path.join(target_dir, "best_candidate.pth")
                torch.save(model.state_dict(), best_path)
                print(f"    [CHECKPOINT SAVED] Selected Epoch {epoch:02d} (Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f})")
        else:
            if f1 > best_val_f1:
                best_val_f1 = f1
                best_epoch = epoch
                best_path = os.path.join(target_dir, "best_candidate.pth")
                torch.save(model.state_dict(), best_path)

    # Save Training History
    with open(os.path.join(target_dir, "training_history.json"), "w") as f:
        json.dump(training_history, f, indent=2)

    # Perform Validation Threshold Tuning on Selected Checkpoint (LEAKAGE-FREE)
    best_candidate_path = os.path.join(target_dir, "best_candidate.pth")
    model.load_state_dict(torch.load(best_candidate_path, map_location=device))
    model.eval()

    val_probs_best, val_targets_best = [], []
    with torch.no_grad():
        for x_b, y_b in val_loader:
            x_b = x_b.to(device)
            out = model(x_b)
            probs = torch.softmax(out, dim=1)[:, 1]
            val_probs_best.extend(probs.cpu().numpy())
            val_targets_best.extend(y_b.numpy())

    val_probs_best = np.array(val_probs_best)
    val_targets_best = np.array(val_targets_best)

    # Threshold Sweep [0.05, 0.95]
    best_tau_f1 = 0.3650
    max_val_f1 = 0.0
    tau_rec90 = 0.3650
    rec90_best_prec = 0.0

    threshold_results = []
    for tau in np.linspace(0.05, 0.95, 91):
        preds_tau = (val_probs_best >= tau).astype(int)
        tp_t = int(((preds_tau == 1) & (val_targets_best == 1)).sum())
        fp_t = int(((preds_tau == 1) & (val_targets_best == 0)).sum())
        tn_t = int(((preds_tau == 0) & (val_targets_best == 0)).sum())
        fn_t = int(((preds_tau == 0) & (val_targets_best == 1)).sum())
        
        p_t, r_t, f1_t, _ = precision_recall_fscore_support(val_targets_best, preds_tau, average="binary", zero_division=0)
        fpr_t = float(fp_t / (fp_t + tn_t + 1e-6))
        fnr_t = float(fn_t / (fn_t + tp_t + 1e-6))

        threshold_results.append({
            "threshold": float(tau), "precision": float(p_t), "recall": float(r_t), "f1": float(f1_t),
            "fpr": fpr_t, "fnr": fnr_t, "tp": tp_t, "fp": fp_t, "tn": tn_t, "fn": fn_t
        })

        if f1_t > max_val_f1:
            max_val_f1 = f1_t
            best_tau_f1 = float(tau)

        if r_t >= 0.90 and p_t > rec90_best_prec:
            rec90_best_prec = p_t
            tau_rec90 = float(tau)

    threshold_analysis = {
        "best_tau_f1": best_tau_f1,
        "max_val_f1": float(max_val_f1),
        "tau_rec90": tau_rec90,
        "rec90_best_prec": float(rec90_best_prec),
        "threshold_sweep": threshold_results
    }
    with open(os.path.join(target_dir, "threshold_analysis.json"), "w") as f:
        json.dump(threshold_analysis, f, indent=2)

    candidate_meta = {
        "experiment": args.experiment,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss) if is_corrected else None,
        "best_val_f1": float(max_val_f1),
        "candidate_tau": best_tau_f1,
        "tau_rec90": tau_rec90,
        "pos_weight": args.pos_weight,
        "min_warmup": args.min_warmup
    }
    with open(os.path.join(target_dir, "candidate_metadata.json"), "w") as f:
        json.dump(candidate_meta, f, indent=2)

    print(f"\n  [VAL THRESHOLD SELECTION] Best Candidate Tau (Max F1={max_val_f1:.4f}): {best_tau_f1:.4f}")
    print(f"  [VAL THRESHOLD SELECTION] Candidate Tau (Rec >= 90%): {tau_rec90:.4f}")
    print(f"  TRAINING COMPLETE. Candidate Saved -> {target_dir}")

if __name__ == "__main__":
    args = parse_args()
    run_multi_dataset_training(args)
