"""
Checkpoint Reproducibility Audit Script for Experiment #16 Anomaly Models.
Verifies 100% exact match reproduction across all 12 trained checkpoints/model files.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT_DIR = r"d:\ONE_DATA\Fall detection"
sys.path.insert(0, ROOT_DIR)

from src.train_le2i_exp16_anomaly import (
    ConvAutoencoder1D, AnomalyDataset, extract_pooled_features, compute_ae_anomaly_score
)
from src.train_le2i_yolo_temporal import compute_metrics, set_seed

def evaluate_le2i_exp16_anomaly():
    print("=" * 70)
    print("EXPERIMENT #16: CHECKPOINT REPRODUCIBILITY AUDIT")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device}")

    manifest_path = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "processed_pose_features_manifest.csv")
    df_manifest = pd.read_csv(manifest_path).sort_values("window_id").reset_index(drop=True)
    k1_dir = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "pose_estimator_features", "yolo_pose_k1")

    res_json_path = os.path.join(ROOT_DIR, "R&D", "ML_Baseline", "results", "exp16_anomaly_benchmark_results.json")
    assert os.path.exists(res_json_path), f"Results JSON missing: {res_json_path}"
    
    with open(res_json_path) as f:
        stored_results = json.load(f)

    ckpt_root = os.path.join(ROOT_DIR, "checkpoints", "le2i_exp16_anomaly")
    locations = ["Coffee_room_01", "Coffee_room_02", "Home_01", "Home_02"]

    # 1. Audit Model M16-A: 1D Conv-AE
    print("\n1. AUDITING MODEL M16-A: 1D CONV AUTOENCODER CHECKPOINTS...")
    ae_folds = stored_results["m16_a_conv_ae"]["folds"]
    for fold_idx, test_loc in enumerate(locations, 1):
        set_seed(42 + fold_idx)
        test_df = df_manifest[df_manifest["location"] == test_loc].reset_index(drop=True)
        test_ds = AnomalyDataset(test_df, k1_dir)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

        ckpt_path = os.path.join(ckpt_root, "ae", f"fold_{fold_idx}_best.pth")
        assert os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}"

        model = ConvAutoencoder1D().to(device)
        model.load_state_dict(torch.load(ckpt_path))
        model.eval()

        test_scores, test_targets = [], []
        for x_b, y_b in test_loader:
            scores = compute_ae_anomaly_score(model, x_b, device)
            test_scores.extend(scores)
            test_targets.extend(y_b.numpy())

        tau_star = ae_folds[fold_idx - 1]["tau_star"]
        stored_f1 = ae_folds[fold_idx - 1]["f1_optimal"]
        m = compute_metrics(np.array(test_targets), np.array(test_scores), threshold=tau_star)

        diff = abs(m["f1"] - stored_f1)
        print(f"  Fold {fold_idx} ({test_loc:15s}) | Reproduced F1={m['f1']:.4f} (Stored={stored_f1:.4f}) | Diff={diff:.6f} ", end="")
        assert diff < 1e-5, f"Fold {fold_idx} reproduction mismatch!"
        print("[MATCH PASS]")

    # 2. Audit Model M16-B: One-Class SVM
    print("\n2. AUDITING MODEL M16-B: ONE-CLASS SVM SAVED MODELS...")
    ocsvm_folds = stored_results["m16_b_ocsvm"]["folds"]
    for fold_idx, test_loc in enumerate(locations, 1):
        test_df = df_manifest[df_manifest["location"] == test_loc].reset_index(drop=True)
        X_te, Y_te = extract_pooled_features(test_df, k1_dir)

        model_path = os.path.join(ckpt_root, "ocsvm", f"fold_{fold_idx}_model.pkl")
        assert os.path.exists(model_path), f"Model file missing: {model_path}"

        with open(model_path, "rb") as f:
            ocsvm = pickle.load(f)

        test_scores = -ocsvm.decision_function(X_te)
        tau_star = ocsvm_folds[fold_idx - 1]["tau_star"]
        stored_f1 = ocsvm_folds[fold_idx - 1]["f1_optimal"]
        m = compute_metrics(Y_te, test_scores, threshold=tau_star)

        diff = abs(m["f1"] - stored_f1)
        print(f"  Fold {fold_idx} ({test_loc:15s}) | Reproduced F1={m['f1']:.4f} (Stored={stored_f1:.4f}) | Diff={diff:.6f} ", end="")
        assert diff < 1e-5, f"Fold {fold_idx} reproduction mismatch!"
        print("[MATCH PASS]")

    # 3. Audit Model M16-C: Isolation Forest
    print("\n3. AUDITING MODEL M16-C: ISOLATION FOREST SAVED MODELS...")
    iforest_folds = stored_results["m16_c_iforest"]["folds"]
    for fold_idx, test_loc in enumerate(locations, 1):
        test_df = df_manifest[df_manifest["location"] == test_loc].reset_index(drop=True)
        X_te, Y_te = extract_pooled_features(test_df, k1_dir)

        model_path = os.path.join(ckpt_root, "iforest", f"fold_{fold_idx}_model.pkl")
        assert os.path.exists(model_path), f"Model file missing: {model_path}"

        with open(model_path, "rb") as f:
            iforest = pickle.load(f)

        test_scores = -iforest.score_samples(X_te)
        tau_star = iforest_folds[fold_idx - 1]["tau_star"]
        stored_f1 = iforest_folds[fold_idx - 1]["f1_optimal"]
        m = compute_metrics(Y_te, test_scores, threshold=tau_star)

        diff = abs(m["f1"] - stored_f1)
        print(f"  Fold {fold_idx} ({test_loc:15s}) | Reproduced F1={m['f1']:.4f} (Stored={stored_f1:.4f}) | Diff={diff:.6f} ", end="")
        assert diff < 1e-5, f"Fold {fold_idx} reproduction mismatch!"
        print("[MATCH PASS]")

    print("\n" + "=" * 70)
    print("ALL 12 ANOMALY CHECKPOINTS & MODELS VERIFIED -- 100% EXACT REPRODUCTION PASSED [PASS]")
    print("=" * 70)

if __name__ == "__main__":
    evaluate_le2i_exp16_anomaly()
