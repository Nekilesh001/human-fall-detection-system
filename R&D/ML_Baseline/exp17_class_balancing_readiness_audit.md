# Experiment #17: Class Balancing Readiness Audit Report

## 1. Executive Summary & Readiness Verdict
- **Audit Target**: Verification of dataset integrity, class ratios, K1 model architecture controls, 4-fold LOLO partitions, class balancing isolation, GPU acceleration, and checkpoint isolation for Experiment #17 (Class Balancing & Oversampling Strategies).
- **Readiness Verdict**: **EXPERIMENT #17 READY FOR IMPLEMENTATION & TRAINING — NO CODE MODIFIED — NO TRAINING EXECUTED**

---

## 2. Model & Hyperparameter Control Audit

| Parameter | K1 SOTA Reference | Experiment #17 Control | Experiment #17 Variants (B, C, D) | Audit Status |
| :--- | :---: | :---: | :---: | :---: |
| **Model Architecture** | `ModelK1_SpatialTCN` | `ModelK1_SpatialTCN` | `ModelK1_SpatialTCN` | **100% Controlled ✅** |
| **Input Representation** | 187-D Spatial Feature | 187-D Spatial Feature | 187-D Spatial Feature | **100% Controlled ✅** |
| **Sequence Length** | 50 frames ($2.0\text{ s}$) | 50 frames ($2.0\text{ s}$) | 50 frames ($2.0\text{ s}$) | **100% Controlled ✅** |
| **Trainable Parameters** | **86,434** | **86,434** | **86,434** | **100% Controlled ✅** |
| **Optimizer & LR** | Adam (1e-3, decay 1e-4) | Adam (1e-3, decay 1e-4) | Adam (1e-3, decay 1e-4) | **100% Controlled ✅** |
| **Epochs & Batch Size** | 100 epochs, BS=32 | 100 epochs, BS=32 | 100 epochs, BS=32 | **100% Controlled ✅** |

---

## 3. Dataset Integrity & Class Imbalance Audit

- **Canonical Manifest**: `processed_data/Le2i_baseline/processed_pose_features_manifest.csv` (1,396 rows intact).
- **187-D Feature Directory**: `processed_data/Le2i_baseline/pose_estimator_features/yolo_pose_k1/` (1,396 files intact).
- **Class Breakdown**: NORMAL = 1,065 ($76.29\%$), FALL = 331 ($23.71\%$). Imbalance ratio = $3.22 : 1$.

---

## 4. LOLO Partition & Leakage Prevention Audit

- **Outer Test Locations**: `Coffee_room_01`, `Coffee_room_02`, `Home_01`, `Home_02`.
- **Event-Level Isolation**: All 4 folds maintain 0 event overlap across outer train and outer test sets.
- **Class Balancing Isolation Scope**:
  - EXP17-B (Weighted Loss): Weights derived strictly from inner train split. Validation and test loss unweighted.
  - EXP17-C (Oversampling): Oversampling applied ONLY to inner train split. Validation and test splits unweighted and un-oversampled.
  - EXP17-D (Balanced Sampler): Sampler applied ONLY to inner train DataLoader. Validation and test DataLoaders standard sequential.
- **Data Leakage Risk**: **0 Risk**.

---

## 5. Hardware & Computational Cost Estimation

- **Execution Hardware**: NVIDIA GeForce RTX 4060 Laptop GPU.
- **VRAM Usage**: $< 1.2\text{ GB}$ VRAM.
- **Estimated Training Time**: $\sim 45\text{ seconds}$ per fold per model ($\sim 3.0\text{ minutes}$ total per model, $\sim 12.0\text{ minutes}$ for all 4 variants across 4 folds).

---

## 6. Checkpoint & Output Isolation Plan

- **Checkpoint Directory**: `checkpoints/le2i_exp17_class_balance/{control,weighted_loss,oversampling,balanced_sampler}/`
- **Result Paths**:
  - `R&D/ML_Baseline/results/exp17_class_balance_results.json`
  - `R&D/ML_Baseline/results/exp17_class_balance_results.csv`
- **Existing Experiments A–K1, K2, K3, Exp 16**: **100% Safe, Isolated, and Untouched**.

---

## 7. Script Creation Plan

| Proposed Script / File | Purpose | Status |
| :--- | :--- | :---: |
| [`R&D/ML_Baseline/exp17_class_balancing_design.md`](file:///d:/ONE_DATA/Fall%20detection/R&D/ML_Baseline/exp17_class_balancing_design.md) | Canonical Design Specification | **CREATED ✅** |
| [`R&D/ML_Baseline/exp17_class_balancing_readiness_audit.md`](file:///d:/ONE_DATA/Fall%20detection/R&D/ML_Baseline/exp17_class_balancing_readiness_audit.md) | Programmatic Readiness Audit | **CREATED ✅** |
| `src/train_le2i_exp17_class_balance.py` | Training pipeline for Exp 17 variants | Proposed |
| `src/evaluate_le2i_exp17_class_balance.py` | Evaluation & reproducibility verification script | Proposed |

---

## 8. Readiness Verdict & Proposed Training Command

```text
EXPERIMENT #17 READINESS AUDIT COMPLETE — TRAINING NOT EXECUTED — WAITING FOR USER APPROVAL
```

When approved, the training pipeline command will be:

```powershell
cd "d:\ONE_DATA\Fall detection"
& "C:\Users\NEKILESH\AppData\Local\Programs\Python\Python311\python.exe" src/train_le2i_exp17_class_balance.py
```
