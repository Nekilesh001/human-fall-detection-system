"""
Verification Script for Saved Experiment C Checkpoints
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
from src.train_le2i_lolo import Le2iFeatureDataset
from src.train_le2i_ablation import TemporalMeanBaseline, TemporalMeanStdControl, TemporalGRUBaseline

def evaluate_ablation_checkpoints():
    print("=" * 70)
    print("EXPERIMENT C: CHECKPOINT EVALUATION & REPRODUCIBILITY VERIFICATION AUDIT")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_manifest_path = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "processed_features_manifest.csv")
    df_feats = pd.read_csv(feat_manifest_path)

    results_csv_path = os.path.join(ROOT_DIR, "R&D", "ML_Baseline", "results", "le2i_temporal_ablation", "ablation_fold_results.csv")
    assert os.path.exists(results_csv_path), f"Results CSV missing at {results_csv_path}"
    df_saved = pd.read_csv(results_csv_path)

    model_configs = {
        "mean": {"name": "Model A (Mean-Only)", "cls": TemporalMeanBaseline, "dir": "mean"},
        "mean_std": {"name": "Model B (Mean+Std Control)", "cls": TemporalMeanStdControl, "dir": "mean_std"},
        "gru": {"name": "Model C (1-Layer GRU)", "cls": TemporalGRUBaseline, "dir": "gru"}
    }

    folds = {
        "Fold 1": {"test": ["Coffee_room_01"], "num": 1},
        "Fold 2": {"test": ["Coffee_room_02"], "num": 2},
        "Fold 3": {"test": ["Home_01"], "num": 3},
        "Fold 4": {"test": ["Home_02"], "num": 4}
    }

    base_ckpt_dir = os.path.join(ROOT_DIR, "checkpoints", "le2i_temporal_ablation")

    for m_key, m_meta in model_configs.items():
        print(f"\nVerifying {m_meta['name']}...")

        for fold_name, f_info in folds.items():
            fold_num = f_info["num"]
            test_loc = f_info["test"][0]
            ckpt_path = os.path.join(base_ckpt_dir, m_meta["dir"], f"fold_{fold_num}_best.pth")

            assert os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}"

            model = m_meta["cls"]().to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.eval()

            test_df = df_feats[df_feats["location"] == test_loc].copy()
            ds_test = Le2iFeatureDataset(test_df, ROOT_DIR)
            loader_test = DataLoader(ds_test, batch_size=32, shuffle=False)

            test_probs, test_targets = [], []
            with torch.no_grad():
                for batch_x, batch_y, _ in loader_test:
                    batch_x = batch_x.to(device)
                    logits = model(batch_x)
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
    print("ALL 12 CHECKPOINTS VERIFIED — 100% REPRODUCIBILITY CONFIRMED (ALL PASS)")
    print("=" * 70)

if __name__ == "__main__":
    evaluate_ablation_checkpoints()
