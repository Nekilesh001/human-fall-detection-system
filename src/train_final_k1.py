"""
PHASE F2/F3 — FINAL K1 TRAINING PIPELINE (OPTIONS A + B)

Option A: 4-Fold LOLO Training & Evaluation
    - Trains one K1 model per LOLO fold on (3 locations train + 20% inner-val)
    - Selects checkpoint using inner-validation F1 (leakage-free)
    - Selects threshold tau* from inner-validation predictions (leakage-free)
    - Evaluates on held-out outer-test location
    - Saves per-window predictions for every outer-test window
    - Checkpoints: checkpoints/final_k1/fold_{1..4}_best.pth

Option B: Full-Data Production Retrain
    - Retrains K1 on ALL 1,396 windows (no held-out test)
    - Uses tau = 0.4923 (mean leakage-free inner-val threshold from Option A)
    - No evaluation possible — performance estimated from Option A LOLO benchmark
    - Checkpoint: checkpoints/final_k1/final_production.pth

Outputs:
    checkpoints/final_k1/fold_1_best.pth
    checkpoints/final_k1/fold_2_best.pth
    checkpoints/final_k1/fold_3_best.pth
    checkpoints/final_k1/fold_4_best.pth
    checkpoints/final_k1/final_production.pth

    R&D/ML_Baseline/results/final_k1/final_test_predictions.csv
    R&D/ML_Baseline/results/final_k1/final_test_metrics.json
    R&D/ML_Baseline/results/final_k1/final_test_metrics.csv
    R&D/ML_Baseline/results/final_k1/final_confusion_matrix.json
    R&D/ML_Baseline/results/final_k1/final_threshold.json

    R&D/ML_Baseline/final/final_k1_training_report.md
    R&D/ML_Baseline/final/final_k1_evaluation_report.md

SAFETY:
    - Does NOT write to checkpoints/le2i_yolo_k1/ (frozen research checkpoints)
    - Does NOT modify any existing manifest, NPZ, or result file
    - Does NOT use outer-test labels for threshold selection
    - Does NOT use outer-test labels for checkpoint selection
"""

import os
import sys
import json
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

ROOT_DIR = r"d:\ONE_DATA\Fall detection"
sys.path.insert(0, ROOT_DIR)

# ---------------------------------------------------------------------------
# Import shared utilities from the validated K1 training pipeline
# ---------------------------------------------------------------------------
from src.train_le2i_yolo_temporal import TemporalBlock, compute_metrics, find_best_threshold, set_seed

# ---------------------------------------------------------------------------
# Model definition — IDENTICAL to validated K1 (must not be changed)
# ---------------------------------------------------------------------------

class ModelK1_SpatialTCN(nn.Module):
    """
    K1: 1D TCN with 187-D Spatial Feature Input.
    Trainable parameters: 86,434
    Total parameters (incl. BatchNorm buffers): 89,250
    Architecture: 2 Residual TCN Blocks (64 ch, dil=[1,2]) ->
                  Mean+Max Pool -> Linear(128->32) -> Linear(32->2)
    """
    def __init__(
        self,
        input_dim=187,
        num_channels=None,
        kernel_size=3,
        fc_dim=32,
        dropout_p=0.5,
    ):
        super().__init__()
        if num_channels is None:
            num_channels = [64, 64]
        layers = []
        for i, out_ch in enumerate(num_channels):
            dilation_size = 2 ** i
            in_ch = input_dim if i == 0 else num_channels[i - 1]
            padding = (kernel_size - 1) * dilation_size
            layers.append(
                TemporalBlock(
                    in_ch, out_ch, kernel_size,
                    stride=1, dilation=dilation_size,
                    padding=padding, dropout=0.2,
                )
            )
        self.tcn = nn.Sequential(*layers)
        self.fc1 = nn.Linear(num_channels[-1] * 2, fc_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(fc_dim, 2)

    def forward(self, x):
        # x: (B, 50, 187) -> (B, 187, 50)
        x_t = x.permute(0, 2, 1)
        feat = self.tcn(x_t)
        mean_p = torch.mean(feat, dim=2)
        max_p, _ = torch.max(feat, dim=2)
        pooled = torch.cat([mean_p, max_p], dim=1)
        out = self.dropout(self.relu(self.fc1(pooled)))
        return self.fc2(out)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class YoloK1Dataset(Dataset):
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
            feat = d["features"].astype(np.float32)  # (50, 187)
        return torch.tensor(feat, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def classify_errors(gt, pred):
    errors = []
    for g, p in zip(gt, pred):
        if g == 1 and p == 1:
            errors.append("TP")
        elif g == 0 and p == 0:
            errors.append("TN")
        elif g == 0 and p == 1:
            errors.append("FP")
        else:
            errors.append("FN")
    return errors


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ---------------------------------------------------------------------------
# Option A — 4-Fold LOLO Training
# ---------------------------------------------------------------------------

def run_option_a(df_manifest, k1_dir, device, ckpt_dir, res_dir):
    print("=" * 70)
    print("OPTION A — 4-FOLD LOLO TRAINING & EVALUATION")
    print("=" * 70)

    locations = ["Coffee_room_01", "Coffee_room_02", "Home_01", "Home_02"]

    fold_results = []
    all_predictions = []

    for fold_idx, test_loc in enumerate(locations, 1):
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx} / 4 — Outer Test Location: {test_loc}")
        print(f"{'='*70}")

        set_seed(42 + fold_idx)

        # Outer split
        test_df      = df_manifest[df_manifest["location"] == test_loc].reset_index(drop=True)
        train_val_df = df_manifest[df_manifest["location"] != test_loc].reset_index(drop=True)

        # Inner split (event-grouped, stratified)
        unique_events = train_val_df["event_id"].unique()
        event_labels = [
            1 if str(train_val_df[train_val_df["event_id"] == ev]["label"].iloc[0]).upper()
                 in ["FALL", "1"] else 0
            for ev in unique_events
        ]
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

        print(f"  Outer test  : {len(test_df)} windows ({test_loc})")
        print(f"  Inner train : {len(inner_train_df)} windows")
        print(f"  Inner val   : {len(inner_val_df)} windows")

        train_ds = YoloK1Dataset(inner_train_df, k1_dir)
        val_ds   = YoloK1Dataset(inner_val_df, k1_dir)
        test_ds  = YoloK1Dataset(test_df, k1_dir)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=0)
        test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=0)

        model = ModelK1_SpatialTCN().to(device)
        total_p, trainable_p = count_params(model)
        if fold_idx == 1:
            print(f"\n  Model Parameters: {trainable_p:,} trainable / {total_p:,} total")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        best_val_f1 = -1.0
        best_ckpt_path = os.path.join(ckpt_dir, f"fold_{fold_idx}_best.pth")

        # Safety: do NOT overwrite frozen le2i_yolo_k1 checkpoints
        assert "le2i_yolo_k1" not in best_ckpt_path, \
            "SAFETY: Attempted to write to frozen checkpoint directory!"

        print(f"\n  Training 100 epochs...")
        for epoch in range(1, 101):
            model.train()
            for x_b, y_b in train_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                out = model(x_b)
                loss = criterion(out, y_b)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            val_probs, val_targets = [], []
            with torch.no_grad():
                for x_b, y_b in val_loader:
                    x_b = x_b.to(device)
                    out = model(x_b)
                    probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                    val_probs.extend(probs)
                    val_targets.extend(y_b.numpy())

            val_probs_arr = np.array(val_probs)
            val_targets_arr = np.array(val_targets)
            v_metrics = compute_metrics(val_targets_arr, val_probs_arr, threshold=0.5)

            if v_metrics["f1"] > best_val_f1:
                best_val_f1 = v_metrics["f1"]
                torch.save(model.state_dict(), best_ckpt_path)

            if epoch % 20 == 0 or epoch == 100:
                print(f"    Epoch {epoch:3d}/100 | Val F1={v_metrics['f1']:.4f} | Best Val F1={best_val_f1:.4f}")

        # Load best checkpoint
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device, weights_only=True))
        model.eval()

        # Inner-val threshold selection (leakage-free)
        val_probs, val_targets = [], []
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b = x_b.to(device)
                out = model(x_b)
                probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                val_probs.extend(probs)
                val_targets.extend(y_b.numpy())
        val_probs_arr = np.array(val_probs)
        val_targets_arr = np.array(val_targets)
        tau_star, val_f1_star = find_best_threshold(val_targets_arr, val_probs_arr)
        print(f"\n  Inner-val threshold: tau*={tau_star:.4f} (Val F1={val_f1_star:.4f})")

        # Outer-test inference
        test_probs, test_targets = [], []
        with torch.no_grad():
            for x_b, y_b in test_loader:
                x_b = x_b.to(device)
                out = model(x_b)
                probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                test_probs.extend(probs)
                test_targets.extend(y_b.numpy())
        test_probs_arr = np.array(test_probs)
        test_targets_arr = np.array(test_targets)

        m_def = compute_metrics(test_targets_arr, test_probs_arr, threshold=0.50)
        m_opt = compute_metrics(test_targets_arr, test_probs_arr, threshold=tau_star)

        print(f"  Outer Test @ tau=0.50 : F1={m_def['f1']:.4f} | Rec={m_def['recall']:.4f} | Spec={m_def['specificity']:.4f}")
        print(f"  Outer Test @ tau*={tau_star:.2f}: F1={m_opt['f1']:.4f} | Rec={m_opt['recall']:.4f} | Spec={m_opt['specificity']:.4f}")

        # Per-window predictions — use positional enumerate for safe index alignment
        pred_labels = (test_probs_arr >= tau_star).astype(int)
        error_types = classify_errors(test_targets_arr.tolist(), pred_labels.tolist())

        for pos_i, (_, row) in enumerate(test_df.iterrows()):
            all_predictions.append({
                "fold": fold_idx,
                "location": test_loc,
                "event_id": row["event_id"],
                "video_id": row["video_id"],
                "window_id": row["window_id"],
                "frame_start": int(row["win_start_frame"]),
                "frame_end": int(row["win_end_frame"]),
                "ground_truth": int(test_targets_arr[pos_i]),
                "fall_probability": float(test_probs_arr[pos_i]),
                "decision_threshold": float(tau_star),
                "predicted_label": int(pred_labels[pos_i]),
                "correct": bool(pred_labels[pos_i] == test_targets_arr[pos_i]),
                "error_type": error_types[pos_i],
            })

        fold_results.append({
            "fold": fold_idx,
            "test_location": test_loc,
            "tau_star": float(tau_star),
            "val_f1_star": float(val_f1_star),
            "f1_default": float(m_def["f1"]),
            "f1_optimal": float(m_opt["f1"]),
            "precision": float(m_opt["precision"]),
            "recall": float(m_opt["recall"]),
            "specificity": float(m_opt["specificity"]),
            "roc_auc": float(m_opt["roc_auc"]),
            "pr_auc": float(m_opt["pr_auc"]),
            "tp": int(m_opt["tp"]),
            "fp": int(m_opt["fp"]),
            "tn": int(m_opt["tn"]),
            "fn": int(m_opt["fn"]),
        })

    # Aggregate LOLO metrics
    f1_def_vals  = [r["f1_default"] for r in fold_results]
    f1_opt_vals  = [r["f1_optimal"] for r in fold_results]
    tau_vals     = [r["tau_star"] for r in fold_results]
    lolo_f1_def  = float(np.mean(f1_def_vals))
    lolo_f1_opt  = float(np.mean(f1_opt_vals))
    lolo_f1_std  = float(np.std(f1_opt_vals))
    mean_tau     = float(np.mean(tau_vals))

    print("\n" + "=" * 70)
    print("OPTION A — LOLO SUMMARY")
    print("=" * 70)
    for r in fold_results:
        print(f"  Fold {r['fold']} ({r['test_location']:15s}) | "
              f"tau=0.50: F1={r['f1_default']:.4f} | "
              f"tau*={r['tau_star']:.4f}: F1={r['f1_optimal']:.4f} | "
              f"TP={r['tp']} FP={r['fp']} TN={r['tn']} FN={r['fn']}")
    print(f"\n  LOLO Mean F1 (@ tau=0.50) : {lolo_f1_def * 100:.2f}%")
    print(f"  LOLO Mean F1 (@ tau*)     : {lolo_f1_opt * 100:.2f}% +/- {lolo_f1_std * 100:.2f}%")
    print(f"  Mean tau*_inner            : {mean_tau:.4f}")

    # Save artifacts
    os.makedirs(res_dir, exist_ok=True)

    # Per-window predictions CSV
    pred_df = pd.DataFrame(all_predictions)
    pred_path = os.path.join(res_dir, "final_test_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"\n  Saved per-window predictions: {pred_path} ({len(pred_df)} rows)")

    # Aggregated confusion matrix
    agg_cm = {
        "total_tp": sum(r["tp"] for r in fold_results),
        "total_fp": sum(r["fp"] for r in fold_results),
        "total_tn": sum(r["tn"] for r in fold_results),
        "total_fn": sum(r["fn"] for r in fold_results),
        "per_fold": [{"fold": r["fold"], "tp": r["tp"], "fp": r["fp"],
                      "tn": r["tn"], "fn": r["fn"]} for r in fold_results],
    }
    cm_path = os.path.join(res_dir, "final_confusion_matrix.json")
    with open(cm_path, "w") as f:
        json.dump(agg_cm, f, indent=2)

    # Threshold file
    thresh_data = {
        "per_fold_tau_star": [{"fold": r["fold"], "tau_star": r["tau_star"]} for r in fold_results],
        "mean_tau_star": mean_tau,
        "deployment_threshold": mean_tau,
        "selection_method": "inner-validation F1 maximization (leakage-free)",
    }
    thresh_path = os.path.join(res_dir, "final_threshold.json")
    with open(thresh_path, "w") as f:
        json.dump(thresh_data, f, indent=2)

    # Metrics JSON
    metrics_data = {
        "option": "A — 4-Fold LOLO Training",
        "model": "Model K1: YOLO Pose 187-D Spatial TCN",
        "trainable_parameters": 86434,
        "total_parameters": 89250,
        "lolo_mean_f1_tau_050": lolo_f1_def,
        "lolo_mean_f1_tau_star": lolo_f1_opt,
        "lolo_f1_std_tau_star": lolo_f1_std,
        "mean_tau_star": mean_tau,
        "reference_benchmark": "86.65% LOLO Mean F1 (Exp 19, leakage-free)",
        "folds": fold_results,
    }
    metrics_json_path = os.path.join(res_dir, "final_test_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_data, f, indent=2)

    # Metrics CSV
    csv_rows = []
    for r in fold_results:
        csv_rows.append({
            "model": "Model K1: YOLO Pose 187-D Spatial TCN",
            "fold": r["fold"],
            "test_location": r["test_location"],
            "tau_star": r["tau_star"],
            "f1_at_0.50": r["f1_default"],
            "f1_at_tau_star": r["f1_optimal"],
            "precision": r["precision"],
            "recall": r["recall"],
            "specificity": r["specificity"],
            "roc_auc": r["roc_auc"],
            "pr_auc": r["pr_auc"],
            "tp": r["tp"], "fp": r["fp"], "tn": r["tn"], "fn": r["fn"],
        })
    metrics_csv_path = os.path.join(res_dir, "final_test_metrics.csv")
    pd.DataFrame(csv_rows).to_csv(metrics_csv_path, index=False)

    print(f"  Saved metrics JSON : {metrics_json_path}")
    print(f"  Saved metrics CSV  : {metrics_csv_path}")
    print(f"  Saved confusion    : {cm_path}")
    print(f"  Saved threshold    : {thresh_path}")

    return fold_results, mean_tau, lolo_f1_opt, lolo_f1_std


# ---------------------------------------------------------------------------
# Option B — Full-Data Production Retrain
# ---------------------------------------------------------------------------

def run_option_b(df_manifest, k1_dir, device, ckpt_dir, deployment_tau):
    print("\n" + "=" * 70)
    print("OPTION B — FULL-DATA PRODUCTION RETRAIN")
    print("=" * 70)
    print(f"  Training on ALL {len(df_manifest)} windows (no held-out test)")
    print(f"  Deployment threshold: tau = {deployment_tau:.4f} (from Option A inner-val)")

    set_seed(42)

    full_ds = YoloK1Dataset(df_manifest, k1_dir)
    full_loader = DataLoader(full_ds, batch_size=32, shuffle=True, num_workers=0)

    model = ModelK1_SpatialTCN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    prod_ckpt_path = os.path.join(ckpt_dir, "final_production.pth")
    assert "le2i_yolo_k1" not in prod_ckpt_path, \
        "SAFETY: Attempted to write to frozen checkpoint directory!"

    print(f"\n  Training 100 epochs on all data...")
    for epoch in range(1, 101):
        model.train()
        epoch_loss = 0.0
        for x_b, y_b in full_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            out = model(x_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 20 == 0 or epoch == 100:
            print(f"    Epoch {epoch:3d}/100 | Mean Loss={epoch_loss / len(full_loader):.4f}")

    torch.save(model.state_dict(), prod_ckpt_path)
    prod_sz = os.path.getsize(prod_ckpt_path) / (1024 * 1024)
    print(f"\n  Saved production checkpoint: {prod_ckpt_path} ({prod_sz:.2f} MB)")
    print(f"  Deployment threshold (tau): {deployment_tau:.4f}")
    print("  NOTE: No held-out evaluation available. Performance estimated from Option A LOLO benchmark.")

    return prod_ckpt_path


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def write_evaluation_report(report_dir, fold_results, mean_tau, lolo_f1_opt, lolo_f1_std, res_dir):
    os.makedirs(report_dir, exist_ok=True)

    ref_benchmark = 86.65
    delta = lolo_f1_opt * 100 - ref_benchmark
    match_status = "REPRODUCED" if abs(delta) < 0.5 else "DIVERGED (investigate)"

    lines = [
        "# Final K1 Evaluation Report",
        "",
        "**Model**: K1 — YOLO Pose + 187-D Spatial Features + 1D Residual TCN",
        "**Phase**: F1 Final Training & Evaluation",
        f"**Date**: {time.strftime('%Y-%m-%d')}",
        "**Protocol**: 4-Fold Leave-One-Location-Out (LOLO), leakage-free",
        "",
        "---",
        "",
        "## 1. LOLO Benchmark Comparison",
        "",
        f"| Metric | Reference (Exp 19) | This Run (Option A) | Delta |",
        f"| :--- | :---: | :---: | :---: |",
        f"| LOLO Mean F1 (tau*) | 86.65% | {lolo_f1_opt*100:.2f}% | {delta:+.2f}% |",
        f"| Cross-Loc Variance  | ±5.64% | ±{lolo_f1_std*100:.2f}% | — |",
        f"| Mean tau*_inner     | 0.4923 | {mean_tau:.4f} | — |",
        f"| Benchmark Status    | — | **{match_status}** | — |",
        "",
        "---",
        "",
        "## 2. Per-Fold Results",
        "",
        "| Fold | Test Location | tau* | F1 @ 0.50 | F1 @ tau* | Recall | Specificity | TP | FP | TN | FN |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in fold_results:
        lines.append(
            f"| {r['fold']} | {r['test_location']} | {r['tau_star']:.4f} | "
            f"{r['f1_default']:.4f} | {r['f1_optimal']:.4f} | "
            f"{r['recall']:.4f} | {r['specificity']:.4f} | "
            f"{r['tp']} | {r['fp']} | {r['tn']} | {r['fn']} |"
        )

    agg_tp = sum(r["tp"] for r in fold_results)
    agg_fp = sum(r["fp"] for r in fold_results)
    agg_tn = sum(r["tn"] for r in fold_results)
    agg_fn = sum(r["fn"] for r in fold_results)

    lines += [
        f"| **All** | **Aggregated** | {mean_tau:.4f} | — | "
        f"**{lolo_f1_opt*100:.2f}%** | — | — | "
        f"**{agg_tp}** | **{agg_fp}** | **{agg_tn}** | **{agg_fn}** |",
        "",
        "---",
        "",
        "## 3. Aggregated Confusion Matrix",
        "",
        "| | Predicted FALL | Predicted NORMAL |",
        "| :--- | :---: | :---: |",
        f"| **Actual FALL**   | TP = {agg_tp} | FN = {agg_fn} |",
        f"| **Actual NORMAL** | FP = {agg_fp} | TN = {agg_tn} |",
        "",
        "---",
        "",
        "## 4. Artifacts",
        "",
        f"- Per-window predictions: `{res_dir}/final_test_predictions.csv`",
        f"- Metrics JSON: `{res_dir}/final_test_metrics.json`",
        f"- Confusion matrix: `{res_dir}/final_confusion_matrix.json`",
        f"- Threshold file: `{res_dir}/final_threshold.json`",
        "",
        "---",
        "",
        "## 5. Leakage-Free Certification",
        "",
        "- Outer-test locations were excluded from all training operations",
        "- Threshold tau* selected from inner-validation predictions only",
        "- No outer-test labels accessed during model selection or threshold tuning",
        "- Inner split is event-grouped (no window-level cross-contamination)",
        "- Fixed random seeds ensure reproducibility",
        "",
    ]

    report_path = os.path.join(report_dir, "final_k1_evaluation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved evaluation report: {report_path}")


def write_training_report(report_dir, fold_results, mean_tau, prod_ckpt_path):
    os.makedirs(report_dir, exist_ok=True)

    lines = [
        "# Final K1 Training Report",
        "",
        "**Model**: K1 — YOLO Pose + 187-D Spatial Features + 1D Residual TCN",
        "**Phase**: F1 Final Training",
        f"**Date**: {time.strftime('%Y-%m-%d')}",
        "",
        "---",
        "",
        "## 1. Hyperparameters",
        "",
        "| Parameter | Value |",
        "| :--- | :--- |",
        "| Architecture | 2-Block Residual 1D TCN |",
        "| Input Dim | 187-D |",
        "| Channels | [64, 64] |",
        "| Kernel Size | 3 |",
        "| Dilations | [1, 2] |",
        "| Pooling | Mean + Max |",
        "| FC Dim | 32 |",
        "| Dropout | 0.5 (training) |",
        "| Optimizer | Adam |",
        "| Learning Rate | 1e-3 |",
        "| Weight Decay | 1e-4 |",
        "| Epochs | 100 |",
        "| Batch Size | 32 |",
        "| Checkpoint criterion | Max inner-val F1 |",
        "",
        "---",
        "",
        "## 2. Option A — LOLO Fold Checkpoint Summary",
        "",
        "| Fold | Test Location | tau* | Best Inner-Val F1 | Checkpoint |",
        "| :---: | :--- | :---: | :---: | :--- |",
    ]
    for r in fold_results:
        lines.append(
            f"| {r['fold']} | {r['test_location']} | {r['tau_star']:.4f} | "
            f"{r['val_f1_star']:.4f} | `checkpoints/final_k1/fold_{r['fold']}_best.pth` |"
        )

    lines += [
        "",
        "---",
        "",
        "## 3. Option B — Production Checkpoint",
        "",
        f"- Trained on all 1,396 windows",
        f"- Deployment threshold: tau = {mean_tau:.4f}",
        f"- Checkpoint: `{prod_ckpt_path}`",
        "- No held-out evaluation. Performance estimated from Option A LOLO benchmark (86.65% LOLO Mean F1).",
        "",
    ]

    report_path = os.path.join(report_dir, "final_k1_training_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved training report: {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("PHASE F2/F3 — FINAL K1 TRAINING PIPELINE (OPTIONS A + B)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device}" +
          (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))

    # Paths
    manifest_path = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline",
                                 "processed_pose_features_manifest.csv")
    k1_dir        = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline",
                                 "pose_estimator_features", "yolo_pose_k1")
    ckpt_dir      = os.path.join(ROOT_DIR, "checkpoints", "final_k1")
    res_dir       = os.path.join(ROOT_DIR, "R&D", "ML_Baseline", "results", "final_k1")
    report_dir    = os.path.join(ROOT_DIR, "R&D", "ML_Baseline", "final")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    # Safety guard — never touch frozen research checkpoints
    frozen_dir = os.path.join(ROOT_DIR, "checkpoints", "le2i_yolo_k1")
    assert os.path.abspath(ckpt_dir) != os.path.abspath(frozen_dir), \
        "SAFETY: ckpt_dir must not equal frozen le2i_yolo_k1 directory!"

    # Load manifest
    df_manifest = pd.read_csv(manifest_path).sort_values("window_id").reset_index(drop=True)
    print(f"\nManifest loaded: {len(df_manifest)} windows")
    print(f"Label distribution:\n{df_manifest['label'].value_counts().to_string()}")

    t_start = time.time()

    # === Option A ===
    fold_results, mean_tau, lolo_f1_opt, lolo_f1_std = run_option_a(
        df_manifest, k1_dir, device, ckpt_dir, res_dir
    )

    # === Option B ===
    prod_ckpt_path = run_option_b(df_manifest, k1_dir, device, ckpt_dir, mean_tau)

    elapsed = time.time() - t_start

    # === Reports ===
    print("\n" + "=" * 70)
    print("GENERATING REPORTS")
    print("=" * 70)
    write_evaluation_report(report_dir, fold_results, mean_tau, lolo_f1_opt, lolo_f1_std, res_dir)
    write_training_report(report_dir, fold_results, mean_tau, prod_ckpt_path)

    print("\n" + "=" * 70)
    print("PHASE F1 TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Total Training Time  : {elapsed / 60:.1f} minutes")
    print(f"  LOLO Mean F1 (tau*)  : {lolo_f1_opt * 100:.2f}% +/- {lolo_f1_std * 100:.2f}%")
    print(f"  Mean tau*_inner      : {mean_tau:.4f}")
    print(f"  Reference Benchmark  : 86.65% (Exp 19, leakage-free)")
    print(f"  Production Checkpoint: {prod_ckpt_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
