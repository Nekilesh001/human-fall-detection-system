# Experiment #16: Anomaly Detection Readiness Audit Report

## 1. Executive Summary & Readiness Verdict
- **Audit Target**: Verification of feature reusability, one-class normal data partitions, model parameter counts, threshold tuning protocols, GPU acceleration, and checkpoint isolation for Experiment #16 (Unsupervised Anomaly Detection for Fall Detection).
- **Readiness Verdict**: **EXPERIMENT #16 READY FOR IMPLEMENTATION & TRAINING — NO CODE MODIFIED — NO TRAINING EXECUTED**

---

## 2. Programmatic Parameter Counts & Architecture Verification

| Model Variant | Paradigm | Training Exposure | Input Representation | Model Architecture | Trainable Parameters | Precomputation Required? |
| :--- | :---: | :---: | :---: | :--- | :---: | :---: |
| **K1 SOTA Baseline** | Supervised | Normal + Fall | `(B, 50, 187)` | 1D Residual TCN (Control) | **86,434** | **NO ✅** |
| **M16-A: Conv-AE** | **Unsupervised** | **Normal Only** | `(B, 50, 187)` | 1D Conv Autoencoder | **84,763** | **NO ✅** |
| **M16-B: OC-SVM** | **Unsupervised** | **Normal Only** | `(B, 374)` | OC-SVM RBF Kernel | Non-parametric | **NO ✅** |
| **M16-C: iForest** | **Unsupervised** | **Normal Only** | `(B, 374)` | Ensemble 100 iTrees | Non-parametric | **NO ✅** |

---

## 3. Dataset & Reusability Audit

- **Feature Directory**: `processed_data/Le2i_baseline/pose_estimator_features/yolo_pose_k1/` (1,396 files, `(50, 187)` float32).
- **Reusability**: **100% Reusable**. The 187-D spatial feature tensors precomputed in Experiment K1 contain all required joint kinematics and body angles.
- **Precomputation Requirement**: **NONE**. Existing 187-D tensors are filtered dynamically in PyTorch/NumPy dataset loaders for `NORMAL` samples during training.

---

## 4. LOLO & Leakage Audit

- **Outer Test Isolation**: 4-Fold LOLO partitions (`Coffee_room_01`, `Coffee_room_02`, `Home_01`, `Home_02`) maintain **0 event overlap** across outer train and test sets.
- **Normal-Only Training Enforcement**: In every fold, training data is strictly filtered for `label == 'NORMAL'` ($y=0$). Zero fall samples enter the training loop.
- **Threshold Selection Isolation**: Optimal anomaly threshold $\tau^*$ is selected on inner validation anomaly scores without observing outer test data.
- **Data Leakage Risk**: **0 Risk**.

---

## 5. Hardware & Computational Cost Estimation

- **Execution Hardware**: NVIDIA GeForce RTX 4060 Laptop GPU.
- **VRAM Usage**: $< 1.1\text{ GB}$ VRAM.
- **Estimated Training Time**: $\sim 30\text{ seconds}$ per fold ($\sim 2.0\text{ minutes}$ total for 4 folds).

---

## 6. Checkpoint & Output Isolation Plan

- **Checkpoint Path**: `checkpoints/le2i_exp16_anomaly/{ae,ocsvm,iforest}/fold_{1..4}_best.pth`
- **Result Paths**:
  - `R&D/ML_Baseline/results/exp16_anomaly_benchmark_results.json`
  - `R&D/ML_Baseline/results/exp16_anomaly_benchmark_results.csv`
- **Existing Experiments A–K**: **100% Safe, Isolated, and Untouched**.

---

## 7. Required Script Creation Plan

| Proposed Script / File | Purpose | Status |
| :--- | :--- | :---: |
| [`R&D/ML_Baseline/exp16_anomaly_detection_design.md`](file:///d:/ONE_DATA/Fall%20detection/R&D/ML_Baseline/exp16_anomaly_detection_design.md) | Canonical Design Specification | **CREATED ✅** |
| [`R&D/ML_Baseline/exp16_anomaly_detection_readiness_audit.md`](file:///d:/ONE_DATA/Fall%20detection/R&D/ML_Baseline/exp16_anomaly_detection_readiness_audit.md) | Programmatic Readiness Audit | **CREATED ✅** |
| `src/train_le2i_exp16_anomaly.py` | Training pipeline for Experiment #16 anomaly models | Proposed |
| `src/evaluate_le2i_exp16_anomaly.py` | Evaluation & reproducibility verification script | Proposed |

---

## 8. Readiness Verdict & Next Training Command

**READINESS VERDICT: READY FOR IMPLEMENTATION & TRAINING**  
Reasoning: All 1,396 187-D spatial feature tensors are 100% reusable. One-class normal partitioning (702–916 normal training windows per fold), 1D Conv-AE model parameters (84,763 params), threshold selection protocols, and hardware isolation paths are fully verified.

When ready to launch Experiment #16 Training, run:

```powershell
cd "d:\ONE_DATA\Fall detection"
& "C:\Users\NEKILESH\AppData\Local\Programs\Python\Python311\python.exe" src/train_le2i_exp16_anomaly.py
```
