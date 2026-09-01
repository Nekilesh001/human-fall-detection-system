"""
PHASE H11 — MODEL K2 CANDIDATE EVALUATION SCRIPT

Loads trained Model K2 checkpoint, fitted FeatureStandardScaler, and candidate threshold tau*.
Transforms held-out test features using the frozen train scaler, evaluates predictions, and calculates:
- Precision, Recall, F1, FPR, FNR, ROC-AUC, PR-AUC, Confusion Matrix
- Per-dataset metrics
- Train / Val / Test Probability Diagnostics
"""

import os
import sys
import argparse
import json
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score, confusion_matrix

ROOT_DIR = r"d:\ONE_DATA\Fall detection"
sys.path.insert(0, ROOT_DIR)

from src.model_k2_dual_stream import ModelK2_DualStreamTCN
from src.train_k2 import FeatureStandardScaler

def parse_args():
    parser = argparse.ArgumentParser(description="Model K2 Research Candidate Evaluation Script")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_candidate.pth")
    parser.add_argument("--test_split", type=str, required=True, help="Path to test_split.csv")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    return parser.parse_args()

def evaluate_k2_candidate(args):
    print("=" * 75)
    print("PHASE H11 — MODEL K2 RESEARCH CANDIDATE EVALUATION")
    print(f"  Checkpoint Path : {args.checkpoint}")
    print(f"  Test Split Path : {args.test_split}")
    print("=" * 75)

    os.makedirs(args.output_dir, exist_ok=True)
    exp_dir = os.path.dirname(args.checkpoint)
    base_dir = os.path.join(ROOT_DIR, "processed_data", "multi_dataset_k1")

    # Load Scaler
    scaler_path = os.path.join(exp_dir, "scaler.pkl")
    assert os.path.exists(scaler_path), f"Missing scaler: {scaler_path}"
    scaler = FeatureStandardScaler()
    scaler.load(scaler_path)

    # Load Threshold tau*
    meta_path = os.path.join(exp_dir, "candidate_metadata.json")
    assert os.path.exists(meta_path), f"Missing candidate_metadata.json: {meta_path}"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    tau = meta["candidate_tau"]
    print(f"  [AUTO] Loaded Candidate Threshold tau* = {tau:.4f} from candidate_metadata.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelK2_DualStreamTCN().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    df_test = pd.read_csv(args.test_split)

    def load_and_transform(df):
        X_raw, y_list = [], []
        for idx, row in df.iterrows():
            abs_p = os.path.join(base_dir, row["feature_path"])
            with np.load(abs_p) as d:
                feat = d["features"] # (50, 187)
            X_raw.append(feat)
            y_list.append(int(row["label"]))
        X_raw = np.array(X_raw, dtype=np.float32)
        X_trans = scaler.transform(X_raw)
        return X_trans, np.array(y_list, dtype=np.int64)

    X_test, y_test = load_and_transform(df_test)

    # Run Model Inference on Test Split
    with torch.no_grad():
        tensor_x = torch.tensor(X_test, dtype=torch.float32).to(device)
        logits = model(tensor_x)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    preds = (probs >= tau).astype(int)

    # Overall Test Metrics
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    roc_auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else 0.5
    pr_auc  = average_precision_score(y_test, probs) if len(np.unique(y_test)) > 1 else 0.5

    print("\nOVERALL HELD-OUT TEST METRICS:")
    print("-" * 50)
    print(f"  Precision       : {prec*100:.2f}%")
    print(f"  Recall          : {rec*100:.2f}%")
    print(f"  F1-Score        : {f1*100:.2f}%")
    print(f"  FPR             : {fpr*100:.2f}%")
    print(f"  FNR             : {fnr*100:.2f}%")
    print(f"  ROC-AUC         : {roc_auc:.4f}")
    print(f"  PR-AUC          : {pr_auc:.4f}")
    print(f"  Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print("-" * 50)

    # Per-Dataset Metrics
    per_ds = {}
    for ds_name in df_test["dataset"].unique():
        ds_mask = (df_test["dataset"] == ds_name).values
        sub_y = y_test[ds_mask]
        sub_p = preds[ds_mask]
        sub_prob = probs[ds_mask]
        
        sp, sr, sf1, _ = precision_recall_fscore_support(sub_y, sub_p, average="binary", zero_division=0)
        stn, sfp, sfn, stp = confusion_matrix(sub_y, sub_p, labels=[0, 1]).ravel()
        sfpr = sfp / (sfp + stn) if (sfp + stn) > 0 else 0.0
        sfnr = sfn / (sfn + stp) if (sfn + stp) > 0 else 0.0
        sauc = roc_auc_score(sub_y, sub_prob) if len(np.unique(sub_y)) > 1 else 0.5
        
        per_ds[ds_name] = {
            "precision": float(sp),
            "recall": float(sr),
            "f1": float(sf1),
            "fpr": float(sfpr),
            "fnr": float(sfnr),
            "roc_auc": float(sauc),
            "count": int(len(sub_y)),
            "fall_count": int((sub_y == 1).sum())
        }
        print(f"  [{ds_name:<9}] F1: {sf1*100:.2f}% | Prec: {sp*100:.2f}% | Rec: {sr*100:.2f}% | FPR: {sfpr*100:.2f}% (N={len(sub_y)}, Fall={int((sub_y==1).sum())})")

    # Probability Diagnostics across Train, Val, Test Folds
    df_tr = pd.read_csv(os.path.join(exp_dir, "train_split.csv"))
    df_va = pd.read_csv(os.path.join(exp_dir, "val_split.csv"))

    def get_diag(df_split):
        X_s, y_s = load_and_transform(df_split)
        with torch.no_grad():
            tx = torch.tensor(X_s, dtype=torch.float32).to(device)
            lg = model(tx)
            pb = torch.softmax(lg, dim=1)[:, 1].cpu().numpy()
        m_fall = float(pb[y_s==1].mean()) if (y_s==1).sum() > 0 else 0.0
        m_norm = float(pb[y_s==0].mean()) if (y_s==0).sum() > 0 else 0.0
        auc_s = roc_auc_score(y_s, pb) if len(np.unique(y_s)) > 1 else 0.5
        pr_s  = average_precision_score(y_s, pb) if len(np.unique(y_s)) > 1 else 0.5
        return {"N": len(y_s), "mean_fall_prob": m_fall, "mean_norm_prob": m_norm, "roc_auc": auc_s, "pr_auc": pr_s}

    tr_diag = get_diag(df_tr)
    va_diag = get_diag(df_va)
    te_diag = {"N": len(y_test), "mean_fall_prob": float(probs[y_test==1].mean()) if (y_test==1).sum()>0 else 0.0, "mean_norm_prob": float(probs[y_test==0].mean()) if (y_test==0).sum()>0 else 0.0, "roc_auc": float(roc_auc), "pr_auc": float(pr_auc)}

    eval_summary = {
        "checkpoint": args.checkpoint,
        "test_split": args.test_split,
        "threshold": float(tau),
        "overall": {
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "fpr": float(fpr),
            "fnr": float(fnr),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn)
        },
        "per_dataset": per_ds,
        "diagnostics": {
            "train": tr_diag,
            "validation": va_diag,
            "test": te_diag
        }
    }

    out_file = os.path.join(args.output_dir, "eval_summary.json")
    with open(out_file, "w") as f:
        json.dump(eval_summary, f, indent=2)

    print(f"  Saved Evaluation Summary -> {out_file}")

if __name__ == "__main__":
    args = parse_args()
    evaluate_k2_candidate(args)
