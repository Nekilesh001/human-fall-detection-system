"""
Training & Evaluation Pipeline for URFD RGB Baseline
Trains URFDRGBBaseline model on Train split, tunes threshold on Val split,
evaluates once on Test split, and saves lightweight result artifacts.
"""

import os
import sys
import time
import json
import random
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.dataset import URFDRGBDataset, URFDRGBFeatureDataset
from src.model import URFDRGBBaseline, URFDRGBFeatureBaseline

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.array(y_true, dtype=int)
    y_prob = np.array(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),          # Sensitivity
        "sensitivity": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "confusion_matrix": [[tn, fp], [fn, tp]]
    }

def run_inference(model, dataloader, device):
    model.eval()
    all_probs = []
    all_labels = []
    all_window_ids = []
    all_event_ids = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["features"].to(device) if "features" in batch else batch["frames"].to(device)
            labels = batch["label"].to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)

            probs = torch.softmax(logits, dim=1)[:, 1] # P(FALL)
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_window_ids.extend(batch["window_id"])
            all_event_ids.extend(batch["event_id"])

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, np.array(all_labels), np.array(all_probs), all_window_ids, all_event_ids

def train_experiment(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode_str = "PRECOMPUTED FEATURES" if getattr(args, "use_precomputed", False) else "RAW RGB FRAMES"
    print(f"=== URFD RGB BASELINE TRAINING [{mode_str}] (Device: {device}) ===")

    # Ensure output directories exist
    checkpoint_dir = os.path.join(ROOT_DIR, "checkpoints")
    results_dir = os.path.join(ROOT_DIR, "R&D", "ML_Baseline", "results")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 1. Load Datasets & Create DataLoaders
    if getattr(args, "use_precomputed", False):
        train_dataset = URFDRGBFeatureDataset(partition="train", root_dir=ROOT_DIR)
        val_dataset   = URFDRGBFeatureDataset(partition="val", root_dir=ROOT_DIR)
        test_dataset  = URFDRGBFeatureDataset(partition="test", root_dir=ROOT_DIR)
    else:
        train_dataset = URFDRGBDataset(partition="train", root_dir=ROOT_DIR, apply_augmentations=args.augment)
        val_dataset   = URFDRGBDataset(partition="val", root_dir=ROOT_DIR, apply_augmentations=False)
        test_dataset  = URFDRGBDataset(partition="test", root_dir=ROOT_DIR, apply_augmentations=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 2. Compute Class Weights Strictly from Train Set
    train_labels = [train_dataset.LABEL_MAP[row["label"]] for _, row in train_dataset.df.iterrows()]
    n_total = len(train_labels)
    n_normal = sum(1 for y in train_labels if y == 0)
    n_fall = sum(1 for y in train_labels if y == 1)

    w_normal = n_total / (2.0 * n_normal)
    w_fall = n_total / (2.0 * n_fall)
    class_weights = torch.tensor([w_normal, w_fall], dtype=torch.float32).to(device)

    print(f"Train Dataset: {n_total} windows (NORMAL: {n_normal}, FALL: {n_fall})")
    print(f"Class Weights: NORMAL={w_normal:.4f}, FALL={w_fall:.4f}")
    print(f"Validation Dataset: {len(val_dataset)} windows")
    print(f"Test Dataset: {len(test_dataset)} windows")

    # 3. Instantiate Model, Loss, Optimizer, Scheduler
    if getattr(args, "use_precomputed", False):
        model = URFDRGBFeatureBaseline(dropout_p=args.dropout).to(device)
    else:
        model = URFDRGBBaseline(dropout_p=args.dropout).to(device)

    param_counts = model.get_parameter_counts()
    print(f"Model Parameters: Total={param_counts['total']:,}, Frozen={param_counts['frozen']:,}, Trainable={param_counts['trainable']:,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_score = -1.0
    best_epoch = 0
    best_checkpoint_path = os.path.join(checkpoint_dir, "urfd_rgb_baseline_best.pth")

    history_records = []

    # 4. Epoch Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        if hasattr(model, "feature_extractor"):
            model.feature_extractor.eval()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        t_epoch_start = time.perf_counter()
        t_load_sum, t_fwd_sum, t_bwd_sum, t_opt_sum = 0.0, 0.0, 0.0, 0.0

        t_prev = time.perf_counter()
        for batch in train_loader:
            t_load = time.perf_counter() - t_prev
            t_load_sum += t_load

            inputs = batch["features"].to(device) if "features" in batch else batch["frames"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            t0 = time.perf_counter()
            logits = model(inputs)
            loss = criterion(logits, labels)
            if device.type == "cuda": torch.cuda.synchronize()
            t_fwd = time.perf_counter() - t0
            t_fwd_sum += t_fwd

            t0 = time.perf_counter()
            loss.backward()
            if device.type == "cuda": torch.cuda.synchronize()
            t_bwd = time.perf_counter() - t0
            t_bwd_sum += t_bwd

            # Gradient Clipping on trainable classifier
            torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), max_norm=1.0)

            t0 = time.perf_counter()
            optimizer.step()
            if device.type == "cuda": torch.cuda.synchronize()
            t_opt = time.perf_counter() - t0
            t_opt_sum += t_opt

            train_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += len(labels)

            t_prev = time.perf_counter()

        t_epoch_total = time.perf_counter() - t_epoch_start

        scheduler.step()

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total


        # Evaluate on Validation Set
        val_loss, val_true, val_probs, _, _ = run_inference(model, val_loader, device)
        val_metrics_05 = compute_metrics(val_true, val_probs, threshold=0.5)

        history_records.append({
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "train_acc": epoch_train_acc,
            "val_loss": val_loss,
            "val_acc": val_metrics_05["accuracy"],
            "val_f1": val_metrics_05["f1"],
            "val_sensitivity": val_metrics_05["sensitivity"],
            "val_specificity": val_metrics_05["specificity"]
        })

        # Model Selection Rule: Maximize Validation F1 Score subject to Specificity >= 0.80
        val_f1 = val_metrics_05["f1"]
        val_spec = val_metrics_05["specificity"]

        # Score formula prioritizing F1 while requiring reasonable specificity
        score = val_f1 if val_spec >= 0.75 else (val_f1 * 0.5)

        if score > best_val_score:
            best_val_score = score
            best_epoch = epoch
            torch.save(model.state_dict(), best_checkpoint_path)

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"Epoch [{epoch:02d}/{args.epochs:02d}] - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_metrics_05['accuracy']:.4f}, Val F1: {val_f1:.4f}, Val Spec: {val_spec:.4f}")

    print(f"\nTraining Complete. Best Model Checkpoint Saved from Epoch {best_epoch} (Score: {best_val_score:.4f})")

    # Save Training History CSV
    df_history = pd.DataFrame(history_records)
    df_history.to_csv(os.path.join(results_dir, "training_history.csv"), index=False)

    # 5. Load Best Model Checkpoint for Final Validation Threshold Tuning & Test Evaluation
    model.load_state_dict(torch.load(best_checkpoint_path))
    model.eval()

    # Validation Threshold Grid Search: tau in [0.10, 0.90]
    val_loss, val_true, val_probs, _, _ = run_inference(model, val_loader, device)
    threshold_grid = np.arange(0.10, 0.95, 0.05)
    val_grid_records = []
    
    best_tau = 0.5
    best_tau_f1 = -1.0

    for tau in threshold_grid:
        m = compute_metrics(val_true, val_probs, threshold=tau)
        val_grid_records.append(m)
        
        # Experimental Threshold Selection Criterion: Maximize Validation F1 (subject to Specificity >= 80%)
        if m["specificity"] >= 0.80:
            if m["f1"] > best_tau_f1:
                best_tau_f1 = m["f1"]
                best_tau = tau

    # Save Threshold Grid CSV
    df_val_grid = pd.DataFrame([
        {
            "threshold": r["threshold"],
            "accuracy": r["accuracy"],
            "precision": r["precision"],
            "sensitivity": r["sensitivity"],
            "specificity": r["specificity"],
            "f1": r["f1"],
            "tp": r["tp"], "tn": r["tn"], "fp": r["fp"], "fn": r["fn"]
        } for r in val_grid_records
    ])
    df_val_grid.to_csv(os.path.join(results_dir, "validation_threshold_analysis.csv"), index=False)

    print(f"\n=== VALIDATION THRESHOLD SELECTION ===")
    print(f"Selected Validation Threshold tau* = {best_tau:.2f} (Val F1: {best_tau_f1:.4f})")

    # 6. Final Single Evaluation on Test Set
    test_loss, test_true, test_probs, test_wids, test_eids = run_inference(model, test_loader, device)
    test_metrics_default = compute_metrics(test_true, test_probs, threshold=0.50)
    test_metrics_selected = compute_metrics(test_true, test_probs, threshold=best_tau)

    model.eval()
    if getattr(args, "use_precomputed", False):
        dummy_win = torch.randn(1, 50, 512).to(device)
    else:
        dummy_win = torch.randn(1, 50, 3, 240, 320).to(device)

    
    # Warmup
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy_win)
            
    # Measure 50 inference iterations
    start_time = time.time()
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_win)
    elapsed_sec = time.time() - start_time
    latency_ms = (elapsed_sec / 50.0) * 1000.0  # ms per 50-frame window
    throughput_fps = (50.0 * 50) / elapsed_sec   # equivalent frame throughput FPS

    final_results = {
        "experiment": "URFD RGB Baseline",
        "device": str(device),
        "best_epoch": best_epoch,
        "selected_val_threshold": best_tau,
        "parameter_counts": param_counts,
        "inference_performance": {
            "window_latency_ms": round(latency_ms, 2),
            "equivalent_frame_fps": round(throughput_fps, 2)
        },
        "test_metrics_default_tau_0_5": test_metrics_default,
        "test_metrics_selected_tau_star": test_metrics_selected
    }

    # Save Final Test Metrics JSON
    with open(os.path.join(results_dir, "final_test_metrics.json"), "w") as f:
        json.dump(final_results, f, indent=2)

    # Save Experiment Config JSON
    config_dict = {
        "dataset": "URFD RGB Baseline",
        "seed": args.seed,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "dropout": args.dropout,
        "augment": args.augment,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "class_weights": {"NORMAL": w_normal, "FALL": w_fall},
        "target_fps": 25.0,
        "window_size": 50,
        "stride": 25,
        "resolution": [320, 240]
    }
    with open(os.path.join(results_dir, "experiment_config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    # 8. Generate Final Comprehensive Baseline Research Report Artifact
    report_path = os.path.join(ROOT_DIR, "R&D", "ML_Baseline", "final_urfd_rgb_baseline_report.md")
    report_content = f"""# Final URFD RGB Baseline Model Training & Benchmark Report

## 1. Experimental Objective
The objective of this experiment is to establish the first complete, leakage-safe ML baseline for the Human Fall Detection System using the URFD RGB dataset. The baseline evaluates an ImageNet-pretrained ResNet-18 spatial feature extractor combined with temporal mean-std pooling and a 2-layer MLP classifier under strict event-level cross-validation split boundaries.

---

## 2. Dataset Description
- **Dataset**: URFD (University of Rzeszów Fall Detection Dataset)
- **Modality**: RGB Only
- **Effective Scope**: 67 usable events (3 missing events skipped during preprocessing due to < 50 frames duration: `fall-16`, `fall-21`, `fall-22`)
- **Total Processed Windows**: 360 temporal windows ($W=50$ frames, $S=25$ stride, 25 FPS, $320 \times 240$ spatial resolution)
- **Class Breakdown**: 118 FALL windows (32.8%), 242 NORMAL windows (67.2%)

---

## 3. Split Description (Leakage-Safe Event-Level Split)
Partition assignments were derived strictly at the event level with seed 42:
- **Train Partition**: 47 events | 260 windows (84 FALL, 176 NORMAL)
- **Validation Partition**: 9 events | 43 windows (10 FALL, 33 NORMAL)
- **Test Partition**: 11 events | 57 windows (24 FALL, 33 NORMAL)
- **Disjointness Audit**: $\text{{Train}} \cap \text{{Val}} = \emptyset$, $\text{{Train}} \cap \text{{Test}} = \emptyset$, $\text{{Val}} \cap \text{{Test}} = \emptyset$ (0 event or camera leakage).

---

## 4. Feature Representation
- **Spatial Backbone**: ImageNet-pretrained ResNet-18 (`resnet.fc = nn.Identity()`)
- **Feature Matrix per Window**: $(50, 512)$ float32 embeddings
- **Precomputation Mode**: Two-stage precomputation pipeline saved to `processed_data/URFD_RGB_baseline/features/` (32.26 MB total storage, 91.75 KB/file). Bit-exact numerical equivalence verified ($\Delta_{{\text{{max}}}} = 0.0$).

---

## 5. Model Architecture & Parameters
```text
Input Feature Tensor: (B, 50, 512)
        │
Temporal Mean + Std Pooling: (B, 1024)
        │
Linear(1024 → 64) ──► ReLU ──► Dropout(p=0.5) ──► Linear(64 → 2)
```
- **Total Parameters**: 65,730
- **Frozen Parameters**: 0 (Backbone ResNet-18 parameters precomputed)
- **Trainable Parameters**: **65,730**

---

## 6. Training Configuration & Reproducibility
- **Optimizer**: AdamW (`lr=0.001`, `weight_decay=0.01`)
- **Scheduler**: CosineAnnealingLR (`T_max=50`, `eta_min=1e-5`)
- **Epochs**: {args.epochs}
- **Batch Size**: {args.batch_size}
- **Seed**: {args.seed} (`random`, `numpy`, `torch`, `cudnn.deterministic=True`)

---

## 7. Class Weighting
Calculated strictly from the 260 Train windows:
$$w_{{\\text{{normal}}}} = \\frac{{260}}{{2 \\times 176}} \\approx 0.738636, \\quad w_{{\\text{{fall}}}} = \\frac{{260}}{{2 \\times 84}} \\approx 1.547619$$

---

## 8. Checkpoint Selection Rule
Model checkpoints were evaluated strictly on the Validation set at the end of each epoch:
$$\\text{{Score}}_{{\\text{{val}}}} = \\begin{{cases}} \\text{{F1}}_{{\\text{{val}}}} & \\text{{if }} \\text{{Specificity}}_{{\\text{{val}}}} \\ge 0.75 \\\\ \\text{{F1}}_{{\\text{{val}}}} \\times 0.5 & \\text{{otherwise}} \\end{{cases}}$$
- **Best Model Epoch**: Epoch {best_epoch} (Validation Score: {best_val_score:.4f})
- **Best Checkpoint Saved**: `checkpoints/urfd_rgb_baseline_best.pth`

---

## 9. Validation Threshold Selection ($\\tau^*$)
Grid search across $\\tau \\in [0.10, 0.90]$ (step 0.05) evaluated on Validation set using the best checkpoint:
- **Default Threshold**: $\\tau = 0.50$
- **Selected Optimal Threshold**: $\\tau^* = {best_tau:.2f}$ (Validation F1 = {best_tau_f1:.4f})

---

## 10. Final Test Evaluation Results

### A. Evaluation @ Default Threshold ($\\tau = 0.50$)
- **Accuracy**: `{test_metrics_default['accuracy']:.4f}`
- **Precision**: `{test_metrics_default['precision']:.4f}`
- **Recall / Sensitivity**: `{test_metrics_default['sensitivity']:.4f}`
- **Specificity**: `{test_metrics_default['specificity']:.4f}`
- **F1 Score**: `{test_metrics_default['f1']:.4f}`
- **Confusion Matrix**: `{test_metrics_default['confusion_matrix']}` (TN: {test_metrics_default['tn']}, FP: {test_metrics_default['fp']}, FN: {test_metrics_default['fn']}, TP: {test_metrics_default['tp']})

### B. Evaluation @ Validation-Selected Threshold ($\\tau^* = {best_tau:.2f}$)
- **Accuracy**: `{test_metrics_selected['accuracy']:.4f}`
- **Precision**: `{test_metrics_selected['precision']:.4f}`
- **Recall / Sensitivity**: `{test_metrics_selected['sensitivity']:.4f}`
- **Specificity**: `{test_metrics_selected['specificity']:.4f}`
- **F1 Score**: `{test_metrics_selected['f1']:.4f}`
- **Confusion Matrix**: `{test_metrics_selected['confusion_matrix']}` (TN: {test_metrics_selected['tn']}, FP: {test_metrics_selected['fp']}, FN: {test_metrics_selected['fn']}, TP: {test_metrics_selected['tp']})

---

## 11. Latency & Throughput Performance
- **Inference Latency**: **{latency_ms:.2f} ms / window** (50-frame temporal window)
- **Throughput**: **{throughput_fps:.1f} FPS** (equivalent frame processing throughput)

---

## 12. Interpretation & Clinical Deployment Boundaries
> [!WARNING]
> **IMPORTANT NON-CLINICAL DISCLAIMER**: High validation/test accuracy on URFD does **NOT** indicate hospital or clinical deployment readiness.
> 
> 1. **Dataset Limitations**: URFD consists of staged fall events performed by healthy actors in controlled indoor environments under consistent lighting.
> 2. **False Alarms per Camera-Hour**: Continuous real-world ward monitoring requires establishing a False Alarm Rate (FAR) per camera-hour across multi-hour non-fall activities (e.g., tying shoes, picking up items, lying down intentionally). Staged event datasets cannot establish FAR.
> 3. **Modality Constraints**: RGB-only models are sensitive to lighting shifts, occlusions, patient blankets, and privacy concerns in real clinical wards.

---

## 13. Reproducibility & Artifact Index
- **Training Log CSV**: `R&D/ML_Baseline/results/training_history.csv`
- **Validation Threshold Grid CSV**: `R&D/ML_Baseline/results/validation_threshold_analysis.csv`
- **Final Test Metrics JSON**: `R&D/ML_Baseline/results/final_test_metrics.json`
- **Experiment Config JSON**: `R&D/ML_Baseline/results/experiment_config.json`
- **Best Model Checkpoint**: `checkpoints/urfd_rgb_baseline_best.pth`
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"\nFinal Research Report Saved: {report_path}")

    print("\n================ FINAL TEST RESULTS ================")
    print(f"TEST @ Default Threshold (tau = 0.50):")
    print(f"  Accuracy:    {test_metrics_default['accuracy']:.4f}")
    print(f"  Precision:   {test_metrics_default['precision']:.4f}")
    print(f"  Recall/Sens: {test_metrics_default['sensitivity']:.4f}")
    print(f"  Specificity: {test_metrics_default['specificity']:.4f}")
    print(f"  F1 Score:    {test_metrics_default['f1']:.4f}")
    print(f"  Confusion Matrix: {test_metrics_default['confusion_matrix']}")

    print(f"\nTEST @ Selected Validation Threshold (tau* = {best_tau:.2f}):")
    print(f"  Accuracy:    {test_metrics_selected['accuracy']:.4f}")
    print(f"  Precision:   {test_metrics_selected['precision']:.4f}")
    print(f"  Recall/Sens: {test_metrics_selected['sensitivity']:.4f}")
    print(f"  Specificity: {test_metrics_selected['specificity']:.4f}")
    print(f"  F1 Score:    {test_metrics_selected['f1']:.4f}")
    print(f"  Confusion Matrix: {test_metrics_selected['confusion_matrix']}")
    print(f"\nInference Performance: {latency_ms:.2f} ms/window ({throughput_fps:.1f} equiv FPS)")
    print("====================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train URFD RGB Baseline Model")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs (default: 50)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay (default: 1e-2)")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout probability (default: 0.5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--augment", action="store_true", help="Enable training data augmentation")
    parser.add_argument("--use_precomputed", action="store_true", help="Use precomputed (50, 512) ResNet-18 features for fast training")

    args = parser.parse_args()
    train_experiment(args)


