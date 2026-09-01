# Phase H13 — EXP-K2-C Candidate Model Results Report

> [!IMPORTANT]
> **READ-ONLY BASELINE SAFETY & ISOLATION CONFIRMATION**  
> Baseline Model Path: `checkpoints/final_k1/final_production.pth`  
> Baseline SHA256: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d` (**IMMUTABLE & FROZEN**)  
> Policy Confirmation: **EXP-K2-C CANDIDATE CHECKPOINT IS ISOLATED UNDER `checkpoints/multi_dataset_k2/exp_k2_c/best_candidate.pth`. PRODUCTION BASELINE REMAINS 100% UNTOUCHED.**

---

## 1. Executive Summary

Phase H13 executed candidate model training for **EXP-K2-C (Le2i + Multicam)** using **Model K2 Dual-Stream TCN** and real YOLOv8-pose 187-D spatial feature tensors. URFD was **100% EXCLUDED** from this experiment. Feature standardization was performed using `FeatureStandardScaler` fitted **ONLY** on training split features. Synchronized 8-camera chute grouping was strictly preserved.

### Key Performance Findings (@ Candidate Tau $\tau^* = 0.1900$)
- **Held-Out Test F1-Score**: **65.05%** (vs 61.98% in EXP-C-REAL — a **$+3.07\%$ improvement in multi-dataset F1!**).
- **Held-Out Test Recall**: **81.05%** (TP = 201, FN = 47).
- **Held-Out Test Precision**: **54.32%** (TP = 201, FP = 169).
- **Held-Out Test ROC-AUC**: **0.7924** (vs 0.7583 in EXP-C-REAL — a **$+0.0341$ jump in ROC-AUC ranking accuracy!**).
- **Held-Out Test PR-AUC**: **0.6073**.
- **Held-Out Test FPR**: **34.14%** (FP = 169, TN = 326).
- **Held-Out Test FNR**: **18.95%**.

---

## 2. Training Configuration & Hardware Details

| Parameter | Value |
| :--- | :--- |
| **Experiment Name** | EXP-K2-C |
| **Training Datasets** | Le2i + Multicam (URFD 100% Excluded) |
| **Architecture** | `ModelK2_DualStreamTCN` (Dual-Stream Spatio-Temporal TCN) |
| **Feature Normalization** | `FeatureStandardScaler` (Fitted **ONLY** on Train Split!) |
| **Hardware Device** | NVIDIA RTX 4060 Laptop GPU (PyTorch CUDA Active) |
| **Epochs** | 30 |
| **Batch Size** | 32 |
| **Learning Rate** | 0.001 (AdamW) |
| **Positive Class Weight (`pos_weight`)** | 1.0 (Unweighted BCE) |
| **Minimum Warmup (`min_warmup`)** | 10 Epochs |
| **Random Seed** | 42 |

---

## 3. Group-Safe Split Statistics

```text
===========================================================================
EXP-K2-C GROUP-SAFE SPLIT STATISTICS (LE2I + MULTICAM)
===========================================================================
  Split Fold | Window Count | Group Count | NORMAL (0) | FALL (1) | Fall %
  -----------|--------------|-------------|------------|----------|--------
  Train      | 3,121        | 149         | 2,375      | 746      | 23.90%
  Val        |   692        |  32         |   486      | 206      | 29.77%
  Test       |   743        |  33         |   495      | 248      | 33.38%
  -------------------------------------------------------------------------
  Total      | 4,556        | 214         | 3,356      | 1,200    | 26.34%
===========================================================================
```

---

## 4. Epoch-by-Epoch Training & Validation Progression

```text
===========================================================================
EXP-K2-C 30-EPOCH TRAINING PROGRESSION
===========================================================================
  Epoch | Train Loss | Val Loss | Val ROC-AUC | Val Prec (%) | Val Rec (%) | Val F1 (%) | Selection Status
  ------|------------|----------|-------------|--------------|-------------|------------|-------------------
  01    | 0.6468     | 0.4419   | 0.8590      | 66.25%       | 25.73%      | 37.06%     | Warmup (Ignored)
  05    | 0.3954     | 0.4396   | 0.8709      | 75.61%       | 30.10%      | 43.06%     | Warmup (Ignored)
  10    | 0.3496     | 0.4275   | 0.8764      | 72.47%       | 62.62%      | 67.19%     | Candidate Checkpoint
  12    | 0.3499     | 0.4166   | 0.8840      | 75.33%       | 54.85%      | 63.48%     | Candidate Checkpoint
  13    | 0.3358     | 0.4151   | 0.8855      | 65.86%       | 79.61%      | 72.09%     | Candidate Checkpoint
  14    | 0.3217     | 0.4061   | 0.8851      | 71.20%       | 66.02%      | 68.51%     | BEST CHECKPOINT (Min Val Loss)
  15    | 0.3396     | 0.4211   | 0.8840      | 67.59%       | 70.87%      | 69.19%     | Higher Val Loss
  20    | 0.2904     | 0.4428   | 0.8712      | 67.24%       | 56.80%      | 61.58%     | Higher Val Loss
  28    | 0.2487     | 0.4570   | 0.8778      | 67.19%       | 62.62%      | 64.82%     | Higher Val Loss
  30    | 0.2157     | 0.5643   | 0.8768      | 71.51%       | 62.14%      | 66.49%     | Higher Val Loss
===========================================================================
```

- **Selected Checkpoint Epoch**: **Epoch 14**
- **Best Validation Loss**: **0.4061**
- **Candidate Operating Threshold ($\tau^*$)**: **0.1900** (Derived from Validation probabilities ONLY)

---

## 5. Held-Out Test Evaluation Results (@ $\tau^* = 0.1900$)

```text
===========================================================================
EXP-K2-C — HELD-OUT TEST METRICS (N = 743 Windows @ tau* = 0.1900)
===========================================================================
  Metric                | Value
  ----------------------|--------------------------------------------------
  Precision             | 54.32%
  Recall                | 81.05% (TP = 201, FN = 47)
  F1-Score              | 65.05%
  FPR                   | 34.14% (FP = 169, TN = 326)
  FNR                   | 18.95%
  ROC-AUC               | 0.7924
  PR-AUC                | 0.6073
  Confusion Matrix      | TP = 201 | FP = 169 | TN = 326 | FN = 47
===========================================================================
```

### Per-Dataset Test Breakdown

| Dataset Name | Test Samples (N) | Precision (%) | Recall (%) | F1-Score (%) | FPR (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Le2i** | 319 | 41.75% | **76.79%** | **54.09%** | 22.81% |
| **Multicam** | 424 | 59.18% | **82.29%** | **68.85%** | 46.98% |

---

## 6. Model Output Probability Diagnostics across Folds

```text
===========================================================================
PROBABILITY DIAGNOSTICS BY SPLIT FOLD
===========================================================================
  Split Fold | Sample Count (N) | Group Count | Mean FALL Prob | Mean NORMAL Prob | ROC-AUC | PR-AUC
  -----------|------------------|-------------|----------------|------------------|---------|-------
  TRAIN      | 3,121            | 149         | 0.7850         | 0.0820           | 0.9620  | 0.8910
  VAL        |   692            |  32         | 0.5810         | 0.1420           | 0.8851  | 0.7650
  TEST       |   743            |  33         | 0.5140         | 0.1680           | 0.7924  | 0.6073
===========================================================================
```

---

## 7. Comparison Matrix: Baseline Production K1 vs. EXP-K2-A vs. EXP-K2-B vs. EXP-K2-C

| Model / Experiment | Training Data | Features / Normalization | Operating Threshold | Precision | Recall | F1-Score | FPR | FNR | ROC-AUC | Status
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---
| **K1 Baseline (Frozen)** | Le2i Only | Real Le2i (Unscaled) | 0.3650 | **85.40%** | **88.20%** | **86.60%** | **4.20%** | **11.80%** | **0.9420** | **PRODUCTION CHAMPION**
| **EXP-K2-A** | Le2i Only | Model K2 + StandardScaler | $\tau^* = 0.4800$ | 56.06% | 66.07% | 60.66% | **10.43%** | 33.93% | **0.8845** | K2 SINGLE-DATASET BASELINE
| **EXP-K2-B** | Le2i + URFD | Model K2 + StandardScaler | $\tau^* = 0.3800$ | 42.67% | 66.67% | 52.03% | 13.31% | 33.33% | 0.8696 | K2 DUAL-DATASET CANDIDATE
| **EXP-C-REAL (K1)** | Le2i + Multicam | Model K1 (Unscaled) | $\tau^* = 0.3100$ | 49.29% | 83.47% | 61.98% | 43.03% | 16.53% | 0.7583 | OLD K1 CANDIDATE
| **EXP-K2-C** | **Le2i + Multicam** | **Model K2 + StandardScaler** | **$\tau^* = 0.1900$** | **54.32%** | **81.05%** | **65.05%** | **34.14%** | **18.95%** | **0.7924** | **K2 MULTI-VIEW CANDIDATE**

*Key Takeaway: Model K2 Dual-Stream TCN achieved **F1 = 65.05%** and **ROC-AUC = 0.7924** on Le2i + Multicam, outperforming old K1 EXP-C-REAL (**F1 = 61.98%, ROC-AUC = 0.7583**) while significantly reducing FPR from 43.03% to 34.14%!*

---

## 8. Summary of Files Created & Modified

### Files Created (2 Files)
1. [`src/validate_phase_h13_k2_c.py`](file:///d:/ONE_DATA/Fall%20detection/src/validate_phase_h13_k2_c.py) — 32-check automated validation suite (**32/32 PASSED**).
2. [`R&D/ML_Baseline/final/phase_h13_exp_k2_c_results.md`](file:///d:/ONE_DATA/Fall%20detection/R&D/ML_Baseline/final/phase_h13_exp_k2_c_results.md) — Phase H13 EXP-K2-C Results Report.

### Files Modified (0 Files)
- **NONE** (Zero modifications to existing code or production files).

---

## 🔒 Final Production Safety Declarations

- **Explicit Confirmation 1**: Production Model K1 (`checkpoints/final_k1/final_production.pth`) remains **100% UNTOUCHED & FROZEN** (SHA256: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d`).
- **Explicit Confirmation 2**: Streamlit application `app.py` remains **100% UNTOUCHED**.
- **Explicit Confirmation 3**: Raw datasets (`Le2i/`, `URFD/`, `dataset/`) remain **100% UNTOUCHED**.
- **Explicit Confirmation 4**: URFD was **100% EXCLUDED** from EXP-K2-C training and evaluation.
- **Explicit Confirmation 5**: EXP-K2-C candidate training completed successfully with outputs isolated in `checkpoints/multi_dataset_k2/exp_k2_c/`.
