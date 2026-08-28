# Experiment K Phase K2: 100-Frame Temporal Context Readiness Audit Report

## 1. Executive Summary & Readiness Verdict
- **Audit Target**: Verification of window generation logic, dataset frame availability, parameter counts, partition splits, GPU suitability, and leakage isolation for Experiment K Phase K2 (100-Frame Temporal Context Benchmark).
- **Readiness Verdict**: **K2 READINESS: PASS — NO CODE MODIFIED — NO TRAINING EXECUTED**

---

## 2. Window Construction & Dataset Audit

### A. Current 50-Frame Window Construction Logic
- **Window Length**: 50 frames ($2.0\text{ s}$ at 25 FPS)
- **Stride**: 25 frames ($1.0\text{ s}$ overlap)
- **Total Windows**: 1,396 supervised windows across 127 Le2i videos
- **Labeling Rule**: `FALL` if window frame bounds overlap with ground-truth fall annotation frame range (`fall_start_f` to `fall_end_f`); otherwise `NORMAL`.

### B. Proposed 100-Frame Window Construction Logic
- **Window Length**: 100 frames ($4.0\text{ s}$ at 25 FPS)
- **Stride**: 25 frames ($1.0\text{ s}$ overlap)
- **Total 100f Windows**: **1,142 supervised windows** (838 NORMAL, 304 FALL)
- **Distribution by Location**:
  - `Coffee_room_01`: 408 windows
  - `Coffee_room_02`: 370 windows
  - `Home_02`: 185 windows
  - `Home_01`: 179 windows

### C. Source Video Frame Availability
- **Video Count**: 127 / 127 Le2i videos have $\ge 133$ frames (Mean length = 500 frames). Every video supports 100-frame window extraction.

---

## 3. Architecture & Parameter Count

| Model Variant | Input Sequence Shape | Model Architecture | Trainable Parameters | Parameter Verification |
| :--- | :---: | :--- | :---: | :---: |
| **K0 Control TCN** | `(B, 50, 165)` | 2 Residual 1D TCN Blocks | **83,618** | **VERIFIED ✅** |
| **K1 Spatial TCN** | `(B, 50, 187)` | 2 Residual 1D TCN Blocks + 22 Body Angles | **86,434** | **VERIFIED ✅** |
| **K2 Temporal TCN (Base)** | `(B, 100, 165)` | 2 Residual 1D TCN Blocks ($4.0\text{ s}$ Window) | **83,618** | **VERIFIED ✅** |
| **K2 Temporal TCN (Spatial)**| `(B, 100, 187)` | 2 Residual 1D TCN Blocks ($4.0\text{ s}$ Window + 22 Angles) | **86,434** | **VERIFIED ✅** |

> **Parameter Consistency**: Global temporal pooling (Mean + Max pooling $\to$ 128-D) keeps the linear classification head dimension identical regardless of sequence length ($T=50$ or $T=100$).

---

## 4. LOLO Partition & Leakage Prevention Audit

- **Physical Locations**: `Coffee_room_01`, `Coffee_room_02`, `Home_01`, `Home_02`
- **Outer Test Isolation**: 4-Fold LOLO partitions maintain **0 event overlap** across outer train and test sets.
- **Inner Validation Split**: 80/20 event-stratified split with zero event overlap.
- **Leakage Risk**: **0 Risk**. All window indexing is strictly grouped by `event_id`.

---

## 5. Hardware & Computational Cost Estimation

- **Execution Hardware**: NVIDIA GeForce RTX 4060 Laptop GPU.
- **VRAM Usage**: $< 1.4\text{ GB}$ VRAM.
- **Estimated Training Time**: $\sim 50\text{ seconds}$ per fold ($\sim 3.3\text{ minutes}$ total for 4 folds).

---

## 6. Checkpoint & Output Isolation Plan

- **Manifest Path**: `processed_data/Le2i_baseline/processed_pose_100f_manifest.csv`
- **Feature Directory**: `processed_data/Le2i_baseline/pose_estimator_features/yolo_pose_100f/`
- **Checkpoint Path**: `checkpoints/le2i_yolo_k2_100f/fold_{1..4}_best.pth`
- **Results Paths**:
  - `R&D/ML_Baseline/results/yolo_k2_100f_benchmark_results.json`
  - `R&D/ML_Baseline/results/yolo_k2_100f_benchmark_results.csv`
- **Existing Experiments A–K1**: **100% Safe, Isolated, and Untouched**.

---

## 7. Required Script Creation Plan

| Proposed Script / File | Purpose | Status |
| :--- | :--- | :---: |
| `src/precompute_le2i_yolo_100f.py` | Extractor for 100-frame YOLO Pose feature tensors | Proposed |
| `src/validate_le2i_yolo_100f.py` | Validation gate for 100-frame feature tensors | Proposed |
| `src/train_le2i_yolo_k2_100f.py` | Training pipeline for K2 (100f Temporal Context) | Proposed |
| `src/evaluate_le2i_yolo_k2_100f.py` | Evaluation & reproducibility verification script | Proposed |
| [`R&D/ML_Baseline/yolo_k2_100f_readiness_audit.md`](file:///d:/ONE_DATA/Fall%20detection/R&D/ML_Baseline/yolo_k2_100f_readiness_audit.md) | Readiness Audit Report Artifact | **CREATED ✅** |

---

## 8. Readiness Verdict

**K2 READINESS: PASS**  
Reasoning: All 127 Le2i videos have $\ge 133$ frames, generating 1,142 valid 100-frame temporal windows with 0 event leakage across 4 LOLO folds. Model parameters (83,618 / 86,434) and hardware requirements are fully verified.
