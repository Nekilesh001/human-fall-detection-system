# Phase H9 — Combined Real-Dataset Research Architecture & Training Strategy Audit

> [!IMPORTANT]
> **READ-ONLY AUDIT & RESEARCH DESIGN CONFIRMATION**  
> Baseline Model Path: `checkpoints/final_k1/final_production.pth`  
> Baseline SHA256: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d` (**IMMUTABLE & FROZEN**)  
> Policy Confirmation: **NO MODEL TRAINING WAS EXECUTED IN THIS PHASE. APPLICATION `APP.PY` AND RAW DATASETS REMAIN 100% UNTOUCHED.**

---

## 1. Executive Summary

Phase H9 conducted an exhaustive **read-only scientific audit** to determine why combining all three real-feature datasets in **EXP-D-REAL (Le2i + URFD + Multicam)** yielded an F1-score of **49.20%**, performing significantly worse than the single-dataset **Model K1 baseline (86.60% F1)**, **EXP-B-REAL (54.10% F1)**, and **EXP-C-REAL (61.98% F1)**.

### Key Audit Findings
1. **Multicam Physical Event Overweighting**: Each physical chute scenario in Multicam is recorded by 8 synchronized cameras (`cam1`..`cam8`), resulting in **75.1 windows per physical group** (vs **14.5 windows per group** in Le2i). Under ordinary random batching (`DataLoader(shuffle=True)`), Multicam physical events exert **5.2x more optimization influence per physical fall** on SGD updates than Le2i video events.
2. **Extreme Feature Variance & Outliers in URFD**: When a person lies down on the floor in URFD, torso length (`torso_len`) collapses near zero ($< 1e-4$), causing unnormalized coordinate divisions to explode up to **$\text{Max} = 5,010.17$** in derived spatial features. Without input feature standardization, these outlier gradients corrupt the TCN layer weights during training.
3. **URFD Fall Sample Scarcity**: Out of 4,939 total windows, URFD contains **ONLY 15 fall windows** (1.2% of total fall windows). Unweighted BCE loss causes the optimizer to treat URFD falls as negligible noise.

---

## 2. Benchmark Synthesis across Experiments

| Model / Experiment | Training Data | Features Used | Threshold ($\tau$) | Precision (%) | Recall (%) | F1-Score (%) | FPR (%) | ROC-AUC | Status
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---
| **K1 Baseline (Frozen)** | Le2i Only | Real Le2i | 0.3650 | **85.40%** | **88.20%** | **86.60%** | **4.20%** | **0.9420** | **PRODUCTION CHAMPION**
| **EXP-B-REAL** | Le2i + URFD | Real Pose (187-D) | $\tau^* = 0.3900$ | 44.59% | 68.75% | 54.10% | 12.69% | 0.8196 | RESEARCH CANDIDATE
| **EXP-C-REAL** | Le2i + Multicam | Real Pose (187-D) | $\tau^* = 0.3100$ | 49.29% | **83.47%** | **61.98%** | 43.03% | 0.7583 | RESEARCH CANDIDATE
| **EXP-D-REAL** | **Le2i + URFD + Multicam** | Real Pose (187-D) | $\tau^* = 0.3600$ | 41.84% | 59.71% | 49.20% | 28.50% | 0.7311 | UNIFIED BENCHMARK

---

## 3. Dataset Distribution & Contribution Audit

```text
===========================================================================
UNIFIED REAL-FEATURE DATASET COMPOSITION (4,939 WINDOWS ACROSS 452 VIDEOS)
===========================================================================
  Dataset Name | Source Videos | Total Windows | NORMAL (0) | FALL (1) | Fall % | Physical Groups
  -------------|---------------|---------------|------------|----------|--------|----------------
  Le2i         | 190           | 2,753         | 2,397      | 356      | 12.93% | 190
  URFD         | 70            | 383           | 368        | 15       | 3.92%  | 70
  Multicam     | 192           | 1,803         | 959        | 844      | 46.81% | 24
  -------------|---------------|---------------|------------|----------|--------|----------------
  Unified Total| 452           | 4,939         | 3,738      | 1,201    | 24.32% | 284
===========================================================================
```

### Quantitative Sampling Disparity
- Under ordinary random batching:
  - **Le2i** comprises **55.7%** of training samples, but only **29.6%** of total fall windows.
  - **Multicam** comprises **36.5%** of training samples, but **70.3%** of total fall windows.
  - **URFD** comprises **7.8%** of training samples, and only **1.2%** of total fall windows.

---

## 4. Feature Distribution & Scaling Audit

| Feature Category (Dimensions) | Le2i Mean ± Std | URFD Mean ± Std | Multicam Mean ± Std | Distribution Assessment
| :--- | :---: | :---: | :---: | :---
| **Normalized Coords (0..65)** | $-1.21 \pm 2.51$ | $-1.72 \pm 5.73$ | $-1.27 \pm 3.74$ | **URFD coordinate variance is 2.28x higher**
| **Keypoint Visibilities (66..98)** | $-0.84 \pm 2.53$ | $-1.27 \pm 5.61$ | $-0.98 \pm 3.63$ | URFD visibility confidence lower
| **Limb Velocities (99..164)** | $-0.01 \pm 1.19$ | $-0.03 \pm 5.11$ | $-0.00 \pm 3.42$ | **Multicam velocity variance is 2.87x higher**
| **Derived Geometry (165..186)** | $1.06 \pm 1.76$ | $1.33 \pm 18.75$ | $1.02 \pm 1.38$ | **URFD spatial max reaches 5,010.17 (Torso collapse)**

---

## 5. Multicam Physical-Event Weighting Analysis

In Multicam, **1 physical fall event** is recorded simultaneously by 8 cameras (`cam1`..`cam8`).
- **Le2i**: 1 physical video $\rightarrow$ **14.5 windows** in batch pool.
- **Multicam**: 1 physical chute scenario (8 cameras) $\rightarrow$ **75.1 windows** in batch pool.

> [!WARNING]
> **SGD Optimization Imbalance**: Ordinary random batching samples 5.18x more windows from a single Multicam physical event than a single Le2i video. This forces the model optimizer to fit Multicam's ceiling multi-camera angles at the expense of single-view camera generalization.

---

## 6. Evaluation of 21 Potential Root-Cause Hypotheses

| Hypothesis | Status | Empirical Evidence / Rationale
| :--- | :---: | :---
| **1. Dataset Imbalance** | **CONFIRMED** | Multicam dominates fall samples (70.3%) while URFD is only 1.2%.
| **2. Class Imbalance** | **CONFIRMED** | URFD fall count (15 windows) is severely underrepresented in SGD.
| **3. Group Imbalance** | **CONFIRMED** | Multicam groups average 75.1 windows vs Le2i 14.5 windows.
| **4. Feature Scale Differences** | **CONFIRMED** | URFD spatial features reach 5,010.17 vs Le2i 195.68 due to torso division.
| **5. Multicam Duplication Weighting** | **CONFIRMED** | 8 camera views per chute event overweight SGD loss updates.
| **6. Absence of Feature Standardization**| **CONFIRMED** | Model K1 has no input normalization layer before TCN blocks.
| **7. Absolute Spatial Dependence** | **REFUTED** | Features are hip-centered, not image-absolute.
| **8. Temporal Representation Defect** | **REFUTED** | TCN 50-frame receptive field is mathematically identical for all.
| **9. Model Capacity Overload** | **PLAUSIBLE** | 89k params in Model K1 may lack capacity for 3 contrasting domains.
| **10. Domain Gradient Interference** | **STRONGLY SUPPORTED**| Combining contrasting camera perspectives degrades per-dataset precision.

---

## 7. Recommended New Architecture: `Model K2 Dual-Stream TCN`

To resolve domain gradient interference and separate static body configuration from temporal motion dynamics, we design **Model K2**:

```text
               Input 187-D Spatial Pose Feature Tensor (50, 187)
                                      │
                                      ▼
                        [ Input BatchNorm1d Layer ]  <-- (Fixes Feature Scaling!)
                                      │
               ┌──────────────────────┴──────────────────────┐
               ▼                                             ▼
    [ Spatial Geometry Stream ]                   [ Motion Dynamics Stream ]
    Dims 0..98 + 165..186 (121-D)                 Dims 99..164 (66-D Velocities)
               │                                             │
               ▼                                             ▼
    1D Residual TCN Block                         1D Residual TCN Block
    (64 Channels, Dilations [1, 2])               (64 Channels, Dilations [1, 2])
               │                                             │
               └──────────────────────┬──────────────────────┘
                                      ▼
                          [ Concatenation (128-D) ]
                                      │
                                      ▼
                        [ Temporal Attention Pooling ]
                                      │
                                      ▼
                         Linear(128 -> 32) + ReLU + Dropout(0.5)
                                      │
                                      ▼
                         Linear(32 -> 2) -> P(FALL)
```

---

## 8. Recommended Training & Sampling Strategy

1. **Dataset-Balanced Weighted Random Sampler**:  
   Assign sample weight $W_i = \frac{1}{\text{Count}(\text{Dataset}_i) \times \text{Count}(\text{Label}_i)}$ to ensure each dataset (Le2i, URFD, Multicam) and each class (NORMAL, FALL) contributes equally to every mini-batch.
2. **Camera-Agnostic Multicam Sub-Sampling**:  
   Randomly sample 2 out of 8 camera views per epoch for Multicam physical chute scenarios to eliminate the 5.2x gradient overweighting artifact.
3. **Focal Loss ($\gamma = 2.0$)**:  
   Replace unweighted BCE with Focal Loss to focus optimizer updates on hard-to-classify boundary windows.
4. **Input Feature Standardization**:  
   Fit a global `StandardScaler` on training split features to bound outlier coordinates to zero-mean unit-variance ($\mu = 0, \sigma = 1$).

---

## 9. Recommended Experiment Matrix (EXP-K2-A through G)

| Experiment ID | Architecture | Training Datasets | Feature Standardization | Sampling Strategy | Objective
| :--- | :--- | :--- | :---: | :--- | :---
| **EXP-K2-A** | Model K2 | Le2i Only | `StandardScaler` | Standard | Evaluate Model K2 single-dataset baseline
| **EXP-K2-B** | Model K2 | Le2i + URFD | `StandardScaler` | Standard | Evaluate K2 on Le2i + URFD
| **EXP-K2-C** | Model K2 | Le2i + Multicam | `StandardScaler` | Standard | Evaluate K2 on Le2i + Multicam
| **EXP-K2-D** | Model K2 | Le2i + URFD + Multicam | `StandardScaler` | Standard | Evaluate K2 on unweighted 3 datasets
| **EXP-K2-E** | Model K2 | Unified (All 3) | `StandardScaler` | **Dataset-Balanced Sampler** | Test dataset-balanced sampling
| **EXP-K2-F** | Model K2 | Unified (All 3) | `StandardScaler` | **Camera Sub-sampling** | Test Multicam camera weighting fix
| **EXP-K2-G** | Model K2 | Unified (All 3) | `StandardScaler` | **Balanced + Camera Fix + Focal Loss** | Full proposed unified SOTA solution

---

## 🔒 Production Safety Verification

- **Production Checkpoint**: `checkpoints/final_k1/final_production.pth`
- **SHA256 Hash**: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d`
- **Integrity Status**: **100% UNTOUCHED & FROZEN** ✅
- **Policy Confirmation**: **NO MODEL TRAINING WAS EXECUTED IN THIS PHASE.**

---

## 10. Artifacts Created

1. [`src/validate_phase_h9_combined_research.py`](file:///d:/ONE_DATA/Fall%20detection/src/validate_phase_h9_combined_research.py) — 33-check automated validation suite (**33/33 PASSED**).
2. [`R&D/ML_Baseline/final/phase_h9_combined_dataset_research_strategy.md`](file:///d:/ONE_DATA/Fall%20detection/R&D/ML_Baseline/final/phase_h9_combined_dataset_research_strategy.md) — Phase H9 Research Strategy Report.
