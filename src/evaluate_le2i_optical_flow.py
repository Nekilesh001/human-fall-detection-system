"""
Verification Script for Saved Experiment D Optical Flow Checkpoints
Re-loads all 12 saved checkpoints (3 model variants x 4 LOLO folds) and reproduces outer test metrics.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.train_baseline import compute_metrics
from src.train_le2i_optical_flow import (
    Le2iDualStreamDataset,
    ModelD1_FlowOnly,
    ModelD2_RGBControl,
    ModelD3_RGBFlowFusion
)

def evaluate_flow_checkpoints():
    print("=" * 70)
    print("EXPERIMENT D: CHECKPOINT EVALUATION & REPRODUCIBILITY AUDIT")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flow_manifest_path = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "processed_flow_features_manifest.csv")
    df_manifest = pd.read_csv(flow_manifest_path)

    results_csv_path = os.path.join(ROOT_DIR, "R&D", "ML_Baseline", "results", "le2i_optical_flow", "flow_fold_results.csv")
    assert os.path.exists(results_csv_path), f"Results CSV missing at {results_csv_path}"
    df_saved = pd.read_csv(results_csv_path)

    models_meta = {
        "flow": {"name": "Model D1 (Flow-Only)", "cls": ModelD1_FlowOnly, "type": "flow", "dir": "flow"},
        "rgb_control": {"name": "Model D2 (RGB Control)", "cls": ModelD2_RGBControl, "type": "rgb", "dir": "rgb_control"},
        "rgb_flow": {"name": "Model D3 (RGB+Flow Fusion)", "cls": ModelD3_RGBFlowFusion, "type": "fusion", "dir": "rgb_flow"}
    }

    folds = {
        "Fold 1": {"test": ["Coffee_room_01"], "num": 1},
        "Fold 2": {"test": ["Coffee_room_02"], "num": 2},
        "Fold 3": {"test": ["Home_01"], "num": 3},
        "Fold 4": {"test": ["Home_02"], "num": 4}
    }

    base_ckpt_dir = os.path.join(ROOT_DIR, "checkpoints", "le2i_optical_flow")

    for m_key, m_info in models_meta.items():
        print(f"\nVerifying {m_info['name']}...")

        for fold_name, f_info in folds.items():
            fold_num = f_info["num"]
            test_loc = f_info["test"][0]
            ckpt_path = os.path.join(base_ckpt_dir, m_info["dir"], f"fold_{fold_num}_best.pth")

            assert os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}"

            model = m_info["cls"]().to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.eval()

            test_df = df_manifest[df_manifest["location"] == test_loc].copy()
            ds_test = Le2iDualStreamDataset(test_df, ROOT_DIR)
            loader_test = DataLoader(ds_test, batch_size=32, shuffle=False)

            test_probs, test_targets = [], []
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

            saved_row = df_saved[(df_saved["model_key"] == m_key) & (df_saved["fold"] == fold_name)].iloc[0]
            sel_tau = float(saved_row["selected_tau"])

            m_050 = compute_metrics(test_targets, test_probs, threshold=0.50)
            m_tau = compute_metrics(test_targets, test_probs, threshold=sel_tau)

            print(f"  {fold_name} ({test_loc:15s}) | Reproduced @ 0.50: F1={m_050['f1']:.4f} | Reproduced @ {sel_tau:.2f}: F1={m_tau['f1']:.4f}")

            assert np.isclose(m_050['f1'], saved_row['f1_050'], atol=1e-4), f"F1 mismatch for {m_key} {fold_name} at tau=0.50!"
            assert np.isclose(m_tau['f1'], saved_row['f1_tau'], atol=1e-4), f"F1 mismatch for {m_key} {fold_name} at tau={sel_tau}!"

    print("\n" + "=" * 70)
    print("ALL 12 OPTICAL FLOW CHECKPOINTS VERIFIED — 100% REPRODUCIBILITY CONFIRMED (ALL PASS)")
    print("=" * 70)

if __name__ == "__main__":
    evaluate_flow_checkpoints()
