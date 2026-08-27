"""
Verification and Evaluation Script for Saved Le2i LOLO Checkpoints
Loads saved checkpoints (checkpoints/le2i_lolo/fold_{1..4}_best.pth) and reproduces outer test metrics.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.model import URFDRGBFeatureBaseline
from src.train_baseline import compute_metrics
from src.train_le2i_lolo import Le2iFeatureDataset

def evaluate_lolo_checkpoints():
    print("=" * 70)
    print("LE2I LOLO CHECKPOINT EVALUATION & VERIFICATION AUDIT")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_manifest_path = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "processed_features_manifest.csv")
    df_feats = pd.read_csv(feat_manifest_path)

    folds = {
        "Fold 1": {"test": ["Coffee_room_01"], "num": 1},
        "Fold 2": {"test": ["Coffee_room_02"], "num": 2},
        "Fold 3": {"test": ["Home_01"], "num": 3},
        "Fold 4": {"test": ["Home_02"], "num": 4}
    }

    results_csv_path = os.path.join(ROOT_DIR, "R&D", "ML_Baseline", "results", "le2i_lolo", "lolo_fold_results.csv")
    assert os.path.exists(results_csv_path), f"Results CSV missing at {results_csv_path}"
    df_saved = pd.read_csv(results_csv_path)

    for fold_name, f_info in folds.items():
        fold_num = f_info["num"]
        test_loc = f_info["test"][0]
        ckpt_path = os.path.join(ROOT_DIR, "checkpoints", "le2i_lolo", f"fold_{fold_num}_best.pth")

        assert os.path.exists(ckpt_path), f"Checkpoint missing for {fold_name} at {ckpt_path}"

        model = URFDRGBFeatureBaseline(dropout_p=0.5).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        test_df = df_feats[df_feats["location"] == test_loc].copy()
        ds_test = Le2iFeatureDataset(test_df, ROOT_DIR)
        loader_test = DataLoader(ds_test, batch_size=32, shuffle=False)

        test_probs = []
        test_targets = []
        with torch.no_grad():
            for batch_x, batch_y, _ in loader_test:
                batch_x = batch_x.to(device)
                logits = model(batch_x)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                test_probs.extend(probs)
                test_targets.extend(batch_y.numpy())

        saved_row = df_saved[df_saved["fold"] == fold_name].iloc[0]
        sel_tau = float(saved_row["selected_tau"])

        m_050 = compute_metrics(test_targets, test_probs, threshold=0.50)
        m_tau = compute_metrics(test_targets, test_probs, threshold=sel_tau)

        print(f"\n{fold_name} Verification (Test Location: {test_loc}):")
        print(f"  Checkpoint File     : {ckpt_path}")
        print(f"  Reproduced @ 0.50   : Acc={m_050['accuracy']:.4f}, Sens={m_050['sensitivity']:.4f}, Spec={m_050['specificity']:.4f}, F1={m_050['f1']:.4f}")
        print(f"  Reproduced @ {sel_tau:.2f}  : Acc={m_tau['accuracy']:.4f}, Sens={m_tau['sensitivity']:.4f}, Spec={m_tau['specificity']:.4f}, F1={m_tau['f1']:.4f}")

        assert np.isclose(m_050['f1'], saved_row['f1_050'], atol=1e-4), f"F1 mismatch for {fold_name} at tau=0.50!"
        assert np.isclose(m_tau['f1'], saved_row['f1_tau'], atol=1e-4), f"F1 mismatch for {fold_name} at tau={sel_tau}!"

    print("\n" + "=" * 70)
    print("LE2I LOLO CHECKPOINT EVALUATION VERIFICATION COMPLETE (ALL PASS)")
    print("=" * 70)

if __name__ == "__main__":
    evaluate_lolo_checkpoints()
