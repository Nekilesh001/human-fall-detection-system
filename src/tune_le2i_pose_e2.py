"""
Phase 3 & Phase 4 Script for Experiment F: Controlled Hyperparameter Search & Final Freeze for Model E2.

Hyperparameters Evaluated:
- Learning Rate: [1e-4, 3e-4, 1e-3, 3e-3]
- Weight Decay : [0, 1e-5, 1e-4, 1e-3, 1e-2]
- Dropout      : [0.2, 0.3, 0.5]
- Batch Size   : [8, 16, 32]

Selection Criterion: STRICTLY MEAN INNER VALIDATION F1 SCORE across 4 LOLO folds.
Outer test locations are 100% HIDDEN until the final configuration is frozen.

Outputs:
- R&D/ML_Baseline/results/le2i_pose_e2/hyperparameter_trials.csv
- R&D/ML_Baseline/results/le2i_pose_e2/final_configuration.json
- R&D/ML_Baseline/results/le2i_pose_e2/final_results.csv
- Checkpoints: checkpoints/le2i_pose_e2_optimized/fold_{1..4}_best.pth
"""

import os
import sys
import time
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT_DIR = r"d:\ONE_DATA\Fall detection"
sys.path.insert(0, ROOT_DIR)

from src.train_baseline import compute_metrics
from src.train_le2i_pose import Le2iPoseDataset, ModelE2_PoseVelocity, set_seed

def tune_le2i_pose_e2():
    print("=" * 70)
    print("EXPERIMENT F: CONTROLLED HYPERPARAMETER SEARCH FOR MODEL E2")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device}")

    pose_manifest_path = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "processed_pose_features_manifest.csv")
    assert os.path.exists(pose_manifest_path), f"Manifest missing: {pose_manifest_path}"
    df_manifest = pd.read_csv(pose_manifest_path)
    df_manifest = df_manifest.sort_values("window_id").reset_index(drop=True)

    folds = {
        "Fold 1": {"num": 1, "test": ["Coffee_room_01"], "train": ["Coffee_room_02", "Home_01", "Home_02"]},
        "Fold 2": {"num": 2, "test": ["Coffee_room_02"], "train": ["Coffee_room_01", "Home_01", "Home_02"]},
        "Fold 3": {"num": 3, "test": ["Home_01"], "train": ["Coffee_room_01", "Coffee_room_02", "Home_02"]},
        "Fold 4": {"num": 4, "test": ["Home_02"], "train": ["Coffee_room_01", "Coffee_room_02", "Home_01"]}
    }

    # Define 12 Controlled Hyperparameter Trial Configurations
    trials = [
        {"trial_id": 0,  "lr": 1e-3, "weight_decay": 1e-2, "dropout_p": 0.5, "batch_size": 32, "desc": "Original E2 Baseline"},
        {"trial_id": 1,  "lr": 1e-4, "weight_decay": 1e-4, "dropout_p": 0.3, "batch_size": 16, "desc": "Low LR, Low WD, Drop 0.3"},
        {"trial_id": 2,  "lr": 3e-4, "weight_decay": 1e-4, "dropout_p": 0.3, "batch_size": 16, "desc": "Mid LR (3e-4), Drop 0.3"},
        {"trial_id": 3,  "lr": 1e-3, "weight_decay": 1e-4, "dropout_p": 0.3, "batch_size": 16, "desc": "Base LR (1e-3), Low WD, Drop 0.3"},
        {"trial_id": 4,  "lr": 3e-3, "weight_decay": 1e-4, "dropout_p": 0.3, "batch_size": 16, "desc": "High LR (3e-3), Drop 0.3"},
        {"trial_id": 5,  "lr": 3e-4, "weight_decay": 0.0,  "dropout_p": 0.2, "batch_size": 16, "desc": "No WD, Drop 0.2"},
        {"trial_id": 6,  "lr": 3e-4, "weight_decay": 1e-5, "dropout_p": 0.2, "batch_size": 16, "desc": "Tiny WD (1e-5), Drop 0.2"},
        {"trial_id": 7,  "lr": 3e-4, "weight_decay": 1e-3, "dropout_p": 0.2, "batch_size": 16, "desc": "Moderate WD (1e-3), Drop 0.2"},
        {"trial_id": 8,  "lr": 3e-4, "weight_decay": 1e-4, "dropout_p": 0.5, "batch_size": 16, "desc": "Mid LR, Drop 0.5"},
        {"trial_id": 9,  "lr": 3e-4, "weight_decay": 1e-4, "dropout_p": 0.3, "batch_size": 8,  "desc": "Small Batch Size (8)"},
        {"trial_id": 10, "lr": 3e-4, "weight_decay": 1e-4, "dropout_p": 0.3, "batch_size": 32, "desc": "Large Batch Size (32)"},
        {"trial_id": 11, "lr": 1e-3, "weight_decay": 1e-4, "dropout_p": 0.2, "batch_size": 16, "desc": "Base LR, Low WD, Drop 0.2"}
    ]

    trial_results = []
    start_time = time.perf_counter()

    print(f"\nEvaluating {len(trials)} Hyperparameter Trials across 4 Folds (STRICT INNER VALIDATION)...")

    for t_cfg in trials:
        t_id = t_cfg["trial_id"]
        lr = t_cfg["lr"]
        wd = t_cfg["weight_decay"]
        dp = t_cfg["dropout_p"]
        bs = t_cfg["batch_size"]

        inner_val_f1s = []

        for fold_name, f_info in folds.items():
            train_locs = f_info["train"]

            # STRICT SEED RESET BEFORE EVERY TRIAL AND EVERY FOLD
            set_seed(42)

            outer_train_df = df_manifest[df_manifest["location"].isin(train_locs)].copy()

            # Class weights (computed ONLY from outer training data)
            N_outer_train = len(outer_train_df)
            N_train_norm  = sum(outer_train_df["label"] == "NORMAL")
            N_train_fall  = sum(outer_train_df["label"] == "FALL")
            w_norm = N_outer_train / (2.0 * N_train_norm)
            w_fall = N_outer_train / (2.0 * N_train_fall)
            class_weights = torch.tensor([w_norm, w_fall], dtype=torch.float).to(device)

            # Inner Event Split (80% Inner Train, 20% Inner Validation)
            outer_train_events = sorted(outer_train_df["event_id"].unique())
            np.random.seed(42)
            shuffled_events = np.random.permutation(outer_train_events)
            n_val_events = max(1, int(len(outer_train_events) * 0.20))

            inner_val_events = set(shuffled_events[:n_val_events])
            inner_tr_events  = set(shuffled_events[n_val_events:])

            inner_tr_df  = outer_train_df[outer_train_df["event_id"].isin(inner_tr_events)].copy()
            inner_val_df = outer_train_df[outer_train_df["event_id"].isin(inner_val_events)].copy()

            loader_tr  = DataLoader(Le2iPoseDataset(inner_tr_df, "e2_feature_path", ROOT_DIR), batch_size=bs, shuffle=True)
            loader_val = DataLoader(Le2iPoseDataset(inner_val_df, "e2_feature_path", ROOT_DIR), batch_size=bs, shuffle=False)

            model = ModelE2_PoseVelocity(dropout_p=dp).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            criterion = nn.CrossEntropyLoss(weight=class_weights)

            best_val_f1 = -1.0

            for epoch in range(1, 51):
                model.train()
                for bx, by, _ in loader_tr:
                    bx, by = bx.to(device), by.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(bx), by)
                    loss.backward()
                    optimizer.step()

                model.eval()
                val_probs, val_targets = [], []
                with torch.no_grad():
                    for bx, by, _ in loader_val:
                        bx = bx.to(device)
                        probs = torch.softmax(model(bx), dim=1)[:, 1].cpu().numpy()
                        val_probs.extend(probs)
                        val_targets.extend(by.numpy())

                val_f1 = compute_metrics(val_targets, val_probs, threshold=0.50)["f1"]
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1

            inner_val_f1s.append(best_val_f1)

        mean_inner_val_f1 = np.mean(inner_val_f1s)
        std_inner_val_f1  = np.std(inner_val_f1s)

        trial_results.append({
            "trial_id": t_id,
            "lr": lr,
            "weight_decay": wd,
            "dropout_p": dp,
            "batch_size": bs,
            "description": t_cfg["desc"],
            "mean_inner_val_f1": mean_inner_val_f1,
            "std_inner_val_f1": std_inner_val_f1,
            "fold_1_val_f1": inner_val_f1s[0],
            "fold_2_val_f1": inner_val_f1s[1],
            "fold_3_val_f1": inner_val_f1s[2],
            "fold_4_val_f1": inner_val_f1s[3]
        })

        print(f"   Trial {t_id:02d} ({t_cfg['desc']:35s}): Mean Inner Val F1 = {mean_inner_val_f1:.4f} ± {std_inner_val_f1:.4f}")

    # Save Trial Results CSV
    res_dir = os.path.join(ROOT_DIR, "R&D", "ML_Baseline", "results", "le2i_pose_e2")
    os.makedirs(res_dir, exist_ok=True)
    df_trials = pd.DataFrame(trial_results)
    trials_csv_path = os.path.join(res_dir, "hyperparameter_trials.csv")
    df_trials.to_csv(trials_csv_path, index=False)

    # SELECT BEST CONFIGURATION STRICTLY BASED ON MEAN INNER VALIDATION F1
    best_trial_row = df_trials.sort_values(by="mean_inner_val_f1", ascending=False).iloc[0]
    best_trial_id  = int(best_trial_row["trial_id"])
    best_cfg       = trials[best_trial_id]

    print("\n" + "=" * 70)
    print("FREEZING BEST HYPERPARAMETER CONFIGURATION (INNER VAL F1 SELECTION)")
    print("=" * 70)
    print(f"Selected Trial ID      : Trial {best_trial_id}")
    print(f"Description            : {best_cfg['desc']}")
    print(f"Learning Rate          : {best_cfg['lr']}")
    print(f"Weight Decay           : {best_cfg['weight_decay']}")
    print(f"Dropout Rate           : {best_cfg['dropout_p']}")
    print(f"Batch Size             : {best_cfg['batch_size']}")
    print(f"Mean Inner Val F1      : {best_trial_row['mean_inner_val_f1']:.4f} ± {best_trial_row['std_inner_val_f1']:.4f}")

    # Save Final Configuration JSON
    final_cfg_path = os.path.join(res_dir, "final_configuration.json")
    with open(final_cfg_path, "w") as f:
        json.dump(best_cfg, f, indent=2)

    # NOW TRAIN FROZEN BEST CONFIGURATION FOR OUTER TEST EVALUATION
    print("\n" + "#" * 70)
    print("TRAINING FROZEN OPTIMIZED MODEL E2 ON ALL 4 FOLDS")
    print("#" * 70)

    opt_ckpt_dir = os.path.join(ROOT_DIR, "checkpoints", "le2i_pose_e2_optimized")
    os.makedirs(opt_ckpt_dir, exist_ok=True)

    final_fold_results = []

    for fold_name, f_info in folds.items():
        fold_num  = f_info["num"]
        test_locs = f_info["test"]
        train_locs = f_info["train"]

        print(f"\n[{best_cfg['desc']}] {fold_name}: Outer Test = {test_locs[0]}")

        set_seed(42)

        outer_train_df = df_manifest[df_manifest["location"].isin(train_locs)].copy()
        outer_test_df  = df_manifest[df_manifest["location"].isin(test_locs)].copy()

        N_outer_train = len(outer_train_df)
        w_norm = N_outer_train / (2.0 * sum(outer_train_df["label"] == "NORMAL"))
        w_fall = N_outer_train / (2.0 * sum(outer_train_df["label"] == "FALL"))
        class_weights = torch.tensor([w_norm, w_fall], dtype=torch.float).to(device)

        outer_train_events = sorted(outer_train_df["event_id"].unique())
        np.random.seed(42)
        shuffled_events = np.random.permutation(outer_train_events)
        n_val_events = max(1, int(len(outer_train_events) * 0.20))

        inner_val_events = set(shuffled_events[:n_val_events])
        inner_tr_events  = set(shuffled_events[n_val_events:])

        inner_tr_df  = outer_train_df[outer_train_df["event_id"].isin(inner_tr_events)].copy()
        inner_val_df = outer_train_df[outer_train_df["event_id"].isin(inner_val_events)].copy()

        loader_tr   = DataLoader(Le2iPoseDataset(inner_tr_df, "e2_feature_path", ROOT_DIR), batch_size=best_cfg["batch_size"], shuffle=True)
        loader_val  = DataLoader(Le2iPoseDataset(inner_val_df, "e2_feature_path", ROOT_DIR), batch_size=best_cfg["batch_size"], shuffle=False)
        loader_test = DataLoader(Le2iPoseDataset(outer_test_df, "e2_feature_path", ROOT_DIR), batch_size=best_cfg["batch_size"], shuffle=False)

        model = ModelE2_PoseVelocity(dropout_p=best_cfg["dropout_p"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=best_cfg["lr"], weight_decay=best_cfg["weight_decay"])
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        best_val_f1 = -1.0
        best_epoch = 0
        fold_ckpt_path = os.path.join(opt_ckpt_dir, f"fold_{fold_num}_best.pth")

        for epoch in range(1, 51):
            model.train()
            for bx, by, _ in loader_tr:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()

            model.eval()
            val_probs, val_targets = [], []
            with torch.no_grad():
                for bx, by, _ in loader_val:
                    bx = bx.to(device)
                    probs = torch.softmax(model(bx), dim=1)[:, 1].cpu().numpy()
                    val_probs.extend(probs)
                    val_targets.extend(by.numpy())

            val_f1 = compute_metrics(val_targets, val_probs, threshold=0.50)["f1"]
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                torch.save(model.state_dict(), fold_ckpt_path)

        # Threshold Search on Inner Validation
        model.load_state_dict(torch.load(fold_ckpt_path, map_location=device))
        model.eval()

        val_probs, val_targets = [], []
        with torch.no_grad():
            for bx, by, _ in loader_val:
                bx = bx.to(device)
                probs = torch.softmax(model(bx), dim=1)[:, 1].cpu().numpy()
                val_probs.extend(probs)
                val_targets.extend(by.numpy())

        best_tau = 0.50
        best_tau_f1 = -1.0
        for tau_cand in np.arange(0.05, 0.96, 0.05):
            m_cand = compute_metrics(val_targets, val_probs, threshold=float(tau_cand))
            if m_cand["f1"] > best_tau_f1:
                best_tau_f1 = m_cand["f1"]
                best_tau = float(tau_cand)

        # Outer Held-Out Test Evaluation
        test_probs, test_targets = [], []
        with torch.no_grad():
            for bx, by, _ in loader_test:
                bx = bx.to(device)
                probs = torch.softmax(model(bx), dim=1)[:, 1].cpu().numpy()
                test_probs.extend(probs)
                test_targets.extend(by.numpy())

        m_def = compute_metrics(test_targets, test_probs, threshold=0.50)
        m_opt = compute_metrics(test_targets, test_probs, threshold=best_tau)

        outer_test_df["pred_prob"] = test_probs
        event_sens_list = []
        for ev_id, ev_grp in outer_test_df.groupby("event_id"):
            if (ev_grp["label"] == "FALL").any():
                if (ev_grp["pred_prob"] >= 0.50).any():
                    event_sens_list.append(1.0)
                else:
                    event_sens_list.append(0.0)

        event_sens = np.mean(event_sens_list) * 100.0 if event_sens_list else 0.0

        print(f"  Best Ep: {best_epoch} | Inner Val F1: {best_val_f1:.4f} | Selected tau*: {best_tau:.2f}")
        print(f"  Outer Test @ 0.50 -> Acc: {m_def['accuracy']:.4f}, Sens: {m_def['sensitivity']:.4f}, Spec: {m_def['specificity']:.4f}, F1: {m_def['f1']:.4f}, CM: {m_def['confusion_matrix']}")
        print(f"  Outer Test @ tau* -> Acc: {m_opt['accuracy']:.4f}, Sens: {m_opt['sensitivity']:.4f}, Spec: {m_opt['specificity']:.4f}, F1: {m_opt['f1']:.4f}, CM: {m_opt['confusion_matrix']}")

        final_fold_results.append({
            "trial_id": best_trial_id,
            "fold_num": fold_num,
            "fold_name": fold_name,
            "test_location": test_locs[0],
            "best_epoch": best_epoch,
            "best_tau": best_tau,
            "acc_050": m_def["accuracy"],
            "prec_050": m_def["precision"],
            "sens_050": m_def["sensitivity"],
            "spec_050": m_def["specificity"],
            "f1_050": m_def["f1"],
            "event_sens_050": event_sens,
            "acc_tau": m_opt["accuracy"],
            "prec_tau": m_opt["precision"],
            "sens_tau": m_opt["sensitivity"],
            "spec_tau": m_opt["specificity"],
            "f1_tau": m_opt["f1"]
        })

    # Save Final Results CSV
    df_final_res = pd.DataFrame(final_fold_results)
    final_csv_path = os.path.join(res_dir, "final_results.csv")
    df_final_res.to_csv(final_csv_path, index=False)

    print("\n" + "=" * 70)
    print("OPTIMIZED MODEL E2 TRAINING COMPLETE — RESULTS SAVED")
    print("=" * 70)
    print(f"Optimized E2 LOLO Mean Accuracy   : {df_final_res['acc_050'].mean():.4f} ± {df_final_res['acc_050'].std():.4f}")
    print(f"Optimized E2 LOLO Mean Precision  : {df_final_res['prec_050'].mean():.4f} ± {df_final_res['prec_050'].std():.4f}")
    print(f"Optimized E2 LOLO Mean Recall/Sens: {df_final_res['sens_050'].mean():.4f} ± {df_final_res['sens_050'].std():.4f}")
    print(f"Optimized E2 LOLO Mean Specificity: {df_final_res['spec_050'].mean():.4f} ± {df_final_res['spec_050'].std():.4f}")
    print(f"Optimized E2 LOLO Mean F1 Score   : {df_final_res['f1_050'].mean():.4f} ± {df_final_res['f1_050'].std():.4f}")
    print(f"Optimized E2 Event Sensitivity    : {df_final_res['event_sens_050'].mean():.2f}% ± {df_final_res['event_sens_050'].std():.2f}%")

if __name__ == "__main__":
    tune_le2i_pose_e2()
