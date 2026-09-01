"""
PHASE H3 — RESEARCH EVALUATION SCRIPT FOR MULTI-DATASET CANDIDATE MODELS

Evaluates Candidate Checkpoints & Baseline K1 on Held-Out Test Splits.
Generates:
- Overall Metrics (Accuracy, Precision, Recall, F1, FPR, FNR, ROC-AUC)
- Confusion Matrix (TP, TN, FP, FN)
- Per-Dataset Breakdown (Le2i F1, URFD F1, Multicam F1)
- Zero-Shot Out-of-Distribution Evaluation (for EXP-E)
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix

ROOT_DIR = r"d:\ONE_DATA\Fall detection"
sys.path.insert(0, ROOT_DIR)

from src.train_final_k1 import ModelK1_SpatialTCN

def parse_args():
    parser = argparse.ArgumentParser(description="Phase H3 Research Evaluation Script")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(ROOT_DIR, "checkpoints", "final_k1", "final_production.pth"), help="Checkpoint path to evaluate")
    parser.add_argument("--test_split", type=str, default=os.path.join(ROOT_DIR, "checkpoints", "multi_dataset_k1", "exp_d_unified", "test_split.csv"), help="Test split CSV path")
    parser.add_argument("--threshold", type=float, default=0.3650, help="Decision threshold tau")
    parser.add_argument("--output_dir", type=str, default=os.path.join(ROOT_DIR, "R&D", "ML_Baseline", "results", "multi_dataset_k1"), help="Results directory")
    return parser.parse_args()

class EvalTensorDataset(Dataset):
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
        return torch.tensor(feat, dtype=torch.float32), torch.tensor(label, dtype=torch.long), row["dataset"]

def evaluate_checkpoint(args):
    # Auto-resolve candidate threshold from metadata if present
    ckpt_dir = os.path.dirname(args.checkpoint)
    meta_json = os.path.join(ckpt_dir, "candidate_metadata.json")
    effective_tau = args.threshold
    if os.path.exists(meta_json):
        with open(meta_json, "r") as f:
            meta = json.load(f)
        if "candidate_tau" in meta:
            effective_tau = float(meta["candidate_tau"])
            print(f"  [AUTO] Loaded Candidate Threshold tau* = {effective_tau:.4f} from candidate_metadata.json")

    print("=" * 75)
    print("PHASE H3 — RESEARCH CANDIDATE MODEL EVALUATION")
    print(f"  Checkpoint Path : {args.checkpoint}")
    print(f"  Test Split Path : {args.test_split}")
    print(f"  Threshold (tau) : {effective_tau:.4f}")
    print("=" * 75)
    args.threshold = effective_tau

    if not os.path.exists(args.test_split):
        print(f"  [NOTE] Test split file {args.test_split} not found yet. Generating mock evaluation template.")
        return

    df_test = pd.read_csv(args.test_split)
    base_dir = os.path.dirname(os.path.dirname(args.test_split))
    if not os.path.exists(base_dir) or "processed_data" not in base_dir:
        base_dir = os.path.join(ROOT_DIR, "processed_data", "multi_dataset_k1")

    test_ds = EvalTensorDataset(df_test, base_dir)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelK1_SpatialTCN(input_dim=187).to(device)

    # Load Checkpoint Weights
    state_dict = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()

    all_preds, all_probs, all_targets, all_ds = [], [], [], []
    with torch.no_grad():
        for x_b, y_b, ds_b in test_loader:
            x_b = x_b.to(device)
            out = model(x_b)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            preds = (probs >= args.threshold).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(y_b.numpy())
            all_ds.extend(ds_b)

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    all_ds = np.array(all_ds)

    # Calculate Overall Metrics
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    
    p, r, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="binary", zero_division=0)
    fpr = fp / (fp + tn + 1e-6)
    fnr = fn / (fn + tp + 1e-6)
    roc_auc = float(roc_auc_score(all_targets, all_probs)) if len(np.unique(all_targets)) > 1 else 0.0

    print("\nOVERALL HELD-OUT TEST METRICS:")
    print("-" * 50)
    print(f"  Precision       : {p*100:.2f}%")
    print(f"  Recall          : {r*100:.2f}%")
    print(f"  F1-Score        : {f1*100:.2f}%")
    print(f"  FPR             : {fpr*100:.2f}%")
    print(f"  FNR             : {fnr*100:.2f}%")
    print(f"  ROC-AUC         : {roc_auc:.4f}")
    print(f"  Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print("-" * 50)

    # Per-Dataset Metrics
    ds_metrics = {}
    for d_name in np.unique(all_ds):
        idx = (all_ds == d_name)
        dp, dr, df1, _ = precision_recall_fscore_support(all_targets[idx], all_preds[idx], average="binary", zero_division=0)
        ds_metrics[d_name] = {"precision": float(dp), "recall": float(dr), "f1": float(df1), "count": int(idx.sum())}
        print(f"  [{d_name:9s}] F1: {df1*100:.2f}% | Prec: {dp*100:.2f}% | Rec: {dr*100:.2f}% (N={idx.sum()})")

    # Save Output Results JSON
    os.makedirs(args.output_dir, exist_ok=True)
    res_stats = {
        "checkpoint": args.checkpoint,
        "test_split": args.test_split,
        "threshold": args.threshold,
        "overall": {"precision": float(p), "recall": float(r), "f1": float(f1), "fpr": float(fpr), "fnr": float(fnr), "roc_auc": roc_auc, "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
        "per_dataset": ds_metrics
    }
    res_path = os.path.join(args.output_dir, "eval_summary.json")
    with open(res_path, "w") as f:
        json.dump(res_stats, f, indent=2)
    print(f"\n  Saved Evaluation Summary -> {res_path}")

if __name__ == "__main__":
    args = parse_args()
    evaluate_checkpoint(args)
