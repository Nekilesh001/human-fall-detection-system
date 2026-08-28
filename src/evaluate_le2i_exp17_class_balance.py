"""
Checkpoint Verification & Reproducibility Audit Script for Experiment #17.
Verifies 100% exact match reproduction across all 16 trained checkpoints.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT_DIR = r"d:\ONE_DATA\Fall detection"
sys.path.insert(0, ROOT_DIR)

from src.train_le2i_yolo_k1_spatial import ModelK1_SpatialTCN, YoloK1Dataset
from src.train_le2i_yolo_temporal import compute_metrics, set_seed

def evaluate_le2i_exp17_class_balance():
    print("=" * 70)
    print("EXPERIMENT #17: CHECKPOINT REPRODUCIBILITY AUDIT")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device}")

    manifest_path = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "processed_pose_features_manifest.csv")
    df_manifest = pd.read_csv(manifest_path).sort_values("window_id").reset_index(drop=True)
    k1_dir = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "pose_estimator_features", "yolo_pose_k1")

    res_json_path = os.path.join(ROOT_DIR, "R&D", "ML_Baseline", "results", "exp17_class_balance_results.json")
    assert os.path.exists(res_json_path), f"Results JSON missing: {res_json_path}"
    
    with open(res_json_path) as f:
        stored_results = json.load(f)

    ckpt_root = os.path.join(ROOT_DIR, "checkpoints", "le2i_exp17_class_balance")
    locations = ["Coffee_room_01", "Coffee_room_02", "Home_01", "Home_02"]

    variants = {
        "exp17_a_control":          "control",
        "exp17_b_weighted_loss":    "weighted_loss",
        "exp17_c_oversampling":     "oversampling",
        "exp17_d_balanced_sampler": "balanced_sampler"
    }

    for var_key, sub_dir in variants.items():
        var_name = stored_results[var_key]["name"]
        print(f"\nAUDITING VARIANT: {var_name}...")
        stored_folds = stored_results[var_key]["folds"]

        for fold_idx, test_loc in enumerate(locations, 1):
            set_seed(42 + fold_idx)
            test_df = df_manifest[df_manifest["location"] == test_loc].reset_index(drop=True)
            test_ds = YoloK1Dataset(test_df, k1_dir)
            test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

            ckpt_path = os.path.join(ckpt_root, sub_dir, f"fold_{fold_idx}_best.pth")
            assert os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}"

            model = ModelK1_SpatialTCN().to(device)
            model.load_state_dict(torch.load(ckpt_path))
            model.eval()

            test_probs, test_targets = [], []
            with torch.no_grad():
                for x_b, y_b in test_loader:
                    x_b = x_b.to(device)
                    out = model(x_b)
                    probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                    test_probs.extend(probs)
                    test_targets.extend(y_b.numpy())

            test_probs = np.array(test_probs)
            test_targets = np.array(test_targets)

            tau_star = stored_folds[fold_idx - 1]["tau_star"]
            stored_f1 = stored_folds[fold_idx - 1]["f1_optimal"]
            
            m = compute_metrics(test_targets, test_probs, threshold=tau_star)
            diff = abs(m["f1"] - stored_f1)

            print(f"  Fold {fold_idx} ({test_loc:15s}) | Reproduced F1={m['f1']:.4f} (Stored={stored_f1:.4f}) | Diff={diff:.6f} ", end="")
            assert diff < 1e-5, f"Fold {fold_idx} reproduction mismatch!"
            print("[MATCH PASS]")

    print("\n" + "=" * 70)
    print("ALL 16 CHECKPOINTS VERIFIED -- 100% EXACT REPRODUCTION PASSED [PASS]")
    print("=" * 70)

if __name__ == "__main__":
    evaluate_le2i_exp17_class_balance()
