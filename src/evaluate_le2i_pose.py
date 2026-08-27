"""
Reproducibility Verification Script for Experiment E (Pose Models E1, E2, E3).
Re-loads all 12 checkpoints from checkpoints/le2i_pose/{e1, e2, e3}/fold_{1..4}_best.pth
and verifies 100% exact reproduction of outer-test metrics.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT_DIR = r"d:\ONE_DATA\Fall detection"
sys.path.insert(0, ROOT_DIR)

from src.train_baseline import compute_metrics
from src.train_le2i_pose import (
    Le2iPoseDataset,
    ModelE1_PoseGeometry,
    ModelE2_PoseVelocity,
    ModelE3_PoseMotionGeometry
)

def evaluate_le2i_pose():
    print("=" * 70)
    print("EXPERIMENT E: CHECKPOINT EVALUATION & REPRODUCIBILITY AUDIT")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device}")

    pose_manifest_path = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "processed_pose_features_manifest.csv")
    assert os.path.exists(pose_manifest_path), f"Manifest missing: {pose_manifest_path}"
    df_manifest = pd.read_csv(pose_manifest_path)
    df_manifest = df_manifest.sort_values("window_id").reset_index(drop=True)

    results_csv_path = os.path.join(ROOT_DIR, "R&D", "ML_Baseline", "results", "le2i_pose", "pose_fold_results.csv")
    assert os.path.exists(results_csv_path), f"Results CSV missing: {results_csv_path}"
    df_results = pd.read_csv(results_csv_path)

    folds = {
        "Fold 1": {"num": 1, "test": ["Coffee_room_01"]},
        "Fold 2": {"num": 2, "test": ["Coffee_room_02"]},
        "Fold 3": {"num": 3, "test": ["Home_01"]},
        "Fold 4": {"num": 4, "test": ["Home_02"]}
    }

    models_meta = {
        "e1": {"name": "Model E1 (Pose Geometry)", "dir": "e1", "cls": ModelE1_PoseGeometry, "feat_col": "e1_feature_path"},
        "e2": {"name": "Model E2 (Pose + Velocity)", "dir": "e2", "cls": ModelE2_PoseVelocity, "feat_col": "e2_feature_path"},
        "e3": {"name": "Model E3 (Pose Motion Geometry)", "dir": "e3", "cls": ModelE3_PoseMotionGeometry, "feat_col": "e3_feature_path"}
    }

    base_ckpt_dir = os.path.join(ROOT_DIR, "checkpoints", "le2i_pose")
    all_matched = True

    for m_key, m_info in models_meta.items():
        print(f"\nVerifying {m_info['name']}...")

        for fold_name, f_info in folds.items():
            fold_num = f_info["num"]
            test_loc = f_info["test"][0]

            ckpt_path = os.path.join(base_ckpt_dir, m_info["dir"], f"fold_{fold_num}_best.pth")
            assert os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}"

            test_df = df_manifest[df_manifest["location"] == test_loc].copy()
            loader_test = DataLoader(Le2iPoseDataset(test_df, m_info["feat_col"], ROOT_DIR), batch_size=32, shuffle=False)

            model = m_info["cls"]().to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.eval()

            test_probs, test_targets = [], []
            with torch.no_grad():
                for bx, by, _ in loader_test:
                    bx = bx.to(device)
                    probs = torch.softmax(model(bx), dim=1)[:, 1].cpu().numpy()
                    test_probs.extend(probs)
                    test_targets.extend(by.numpy())

            row_match = df_results[
                (df_results["model_key"] == m_key) & (df_results["fold_num"] == fold_num)
            ].iloc[0]

            tau_star = float(row_match["best_tau"])
            m_def = compute_metrics(test_targets, test_probs, threshold=0.50)
            m_opt = compute_metrics(test_targets, test_probs, threshold=tau_star)

            exp_f1_050 = float(row_match["f1_050"])
            exp_f1_tau = float(row_match["f1_tau"])

            match_050 = np.isclose(m_def["f1"], exp_f1_050, atol=1e-4)
            match_tau = np.isclose(m_opt["f1"], exp_f1_tau, atol=1e-4)

            if not (match_050 and match_tau):
                all_matched = False
                print(f"  ❌ MISMATCH in {fold_name} ({test_loc}): Exp @0.50={exp_f1_050:.4f}, Got={m_def['f1']:.4f}")
            else:
                print(f"  Fold {fold_num} ({test_loc:15s}) | Reproduced @ 0.50: F1={m_def['f1']:.4f} | Reproduced @ tau* ({tau_star:.2f}): F1={m_opt['f1']:.4f}")

    print("\n" + "=" * 70)
    if all_matched:
        print("ALL 12 POSE CHECKPOINTS VERIFIED — 100% REPRODUCIBILITY CONFIRMED (ALL PASS)")
    else:
        print("CRITICAL WARNING: SOME CHECKPOINTS FAILED REPRODUCIBILITY VERIFICATION")
    print("=" * 70)

if __name__ == "__main__":
    evaluate_le2i_pose()
