# Phase H11 — EXP-K2-A Candidate Model Results Report

> [!IMPORTANT]
> **READ-ONLY BASELINE SAFETY & ISOLATION CONFIRMATION**  
> Baseline Model Path: `checkpoints/final_k1/final_production.pth`  
> Baseline SHA256: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d` (**IMMUTABLE & FROZEN**)  
> Policy Confirmation: **EXP-K2-A CANDIDATE CHECKPOINT IS ISOLATED UNDER `checkpoints/multi_dataset_k2/exp_k2_a/best_candidate.pth`. PRODUCTION BASELINE REMAINS 100% UNTOUCHED.**

---

## 1. Executive Summary

Phase H11 executed the first **controlled candidate training experiment for Model K2 (EXP-K2-A)** using **Le2i ONLY** real YOLOv8-pose 187-D spatial feature tensors. Feature standardization was performed using `FeatureStandardScaler` fitted **ONLY** on the training split features.

### Key Performance Findings (@ Candidate Tau $\tau^* = 0.4800$)
- **Held-Out Test ROC-AUC**: **0.8845** (Continuous probabilities).
- **Held-Out Test PR-AUC**: **0.6346**.
- **Held-Out Test Precision**: **56.06%** (TP = 37, FP = 29).
- **Held-Out Test Recall**: **66.07%** (TP = 37, FN = 19).
- **Held-Out Test F1-Score**: **60.66%**.
- **Held-Out Test FPR**: **10.43%** (FP = 29, TN = 249).
- **Held-Out Test FNR**: **33.93%**.

---

## 2. Training Configuration & Hardware Details

| Parameter | Value |
| :--- | :--- |
| **Experiment Name** | EXP-K2-A |
| **Training Dataset** | Le2i Only (URFD & Multicam 100% Excluded) |
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
EXP-K2-A GROUP-SAFE SPLIT STATISTICS (LE2I ONLY)
===========================================================================
  Split Fold | Window Count | Group Count | NORMAL (0) | FALL (1) | Fall %
  -----------|--------------|-------------|------------|----------|--------
  Train      | 1,988        | 133         | 1,732      | 256      | 12.88%
  Val        |   431        |  28         |   388      |  43      |  9.98%
  Test       |   334        |  29         |   278      |  56      | 16.77%
  -------------------------------------------------------------------------
  Total      | 2,753        | 190         | 2,398      | 355      | 12.90%
===========================================================================
```

---

## 4. Epoch-by-Epoch Training & Validation Progression

```text
===========================================================================
EXP-K2-A 30-EPOCH TRAINING PROGRESSION
===========================================================================
  Epoch | Train Loss | Val Loss | Val ROC-AUC | Val Prec (%) | Val Rec (%) | Val F1 (%) | Selection Status
  ------|------------|----------|-------------|--------------|-------------|------------|-------------------
  01    | 0.6040     | 0.3036   | 0.8519      | 47.17%       | 58.14%      | 52.08%     | Warmup (Ignored)
  05    | 0.3551     | 0.2523   | 0.8768      | 41.18%       | 16.28%      | 23.33%     | Warmup (Ignored)
  10    | 0.2470     | 0.2576   | 0.8431      | 41.18%       | 16.28%      | 23.33%     | BEST CHECKPOINT (Min Val Loss)
  11    | 0.2227     | 0.2717   | 0.8407      | 44.44%       | 37.21%      | 40.51%     | Higher Val Loss
  15    | 0.1840     | 0.3037   | 0.8491      | 47.06%       | 55.81%      | 51.06%     | Higher Val Loss
  20    | 0.1601     | 0.3608   | 0.8452      | 49.15%       | 67.44%      | 56.86%     | Higher Val Loss
  24    | 0.1474     | 0.3395   | 0.8849      | 48.48%       | 74.42%      | 58.72%     | Higher Val Loss
  30    | 0.1185     | 0.3832   | 0.8727      | 50.85%       | 69.77%      | 58.82%     | Higher Val Loss
===========================================================================
```

- **Selected Checkpoint Epoch**: **Epoch 10**
- **Best Validation Loss**: **0.2576**
- **Candidate Operating Threshold ($\tau^*$)**: **0.4800** (Derived from Validation probabilities ONLY)

---

## 5. Held-Out Test Evaluation Results (@ $\tau^* = 0.4800$)

```text
===========================================================================
EXP-K2-A — HELD-OUT TEST METRICS (N = 334 Windows @ tau* = 0.4800)
===========================================================================
  Metric                | Value
  ----------------------|--------------------------------------------------
  Precision             | 56.06%
  Recall                | 66.07% (TP = 37, FN = 19)
  F1-Score              | 60.66%
  FPR                   | 10.43% (FP = 29, TN = 249)
  FNR                   | 33.93%
  ROC-AUC               | 0.8845
  PR-AUC                | 0.6346
  Confusion Matrix      | TP = 37 | FP = 29 | TN = 249 | FN = 19
===========================================================================
```

---

## 6. Model Output Probability Diagnostics across Folds

```text
===========================================================================
PROBABILITY DIAGNOSTICS BY SPLIT FOLD
===========================================================================
  Split Fold | Sample Count (N) | Mean FALL Prob | Mean NORMAL Prob | ROC-AUC | PR-AUC
  -----------|------------------|----------------|------------------|---------|-------
  TRAIN      | 1,988            | 0.7240         | 0.0510           | 0.9680  | 0.8920
  VAL        |   431            | 0.4910         | 0.1120           | 0.8431  | 0.5840
  TEST       |   334            | 0.5280         | 0.1080           | 0.8845  | 0.6346
===========================================================================
```

---

## 7. Comparison Matrix: Baseline Production K1 vs. EXP-K2-A

| Model / Experiment | Training Data | Features / Normalization | Operating Threshold | Precision | Recall | F1-Score | FPR | FNR | ROC-AUC | Status
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---
| **K1 Baseline (Frozen)** | Le2i Only | Real Le2i (Unscaled) | 0.3650 | **85.40%** | **88.20%** | **86.60%** | **4.20%** | **11.80%** | **0.9420** | **PRODUCTION CHAMPION**
| **EXP-K2-A** | Le2i Only | **Model K2 + StandardScaler** | **$\tau^* = 0.4800$** | **56.06%** | **66.07%** | **60.66%** | **10.43%** | **33.93%** | **0.8845** | **K2 SINGLE-DATASET BASELINE**

---

## 8. Summary of Files Created & Modified

### Files Created (4 Files)
1. [`src/model_k2_dual_stream.py`](file:///d:/ONE_DATA/Fall%20detection/src/model_k2_dual_stream.py) — Independent Model K2 Dual-Stream TCN architecture implementation.
2. [`src/train_k2.py`](file:///d:/ONE_DATA/Fall%20detection/src/train_k2.py) — Model K2 training pipeline script with `FeatureStandardScaler` (fitted ONLY on train split).
3. [`src/evaluate_k2.py`](file:///d:/ONE_DATA/Fall%20detection/src/evaluate_k2.py) — Model K2 candidate evaluation script.
4. [`src/validate_phase_h11_k2_a.py`](file:///d:/ONE_DATA/Fall%20detection/src/validate_phase_h11_k2_a.py) — 25-check automated validation suite (**25/25 PASSED**).

### Files Modified (0 Files)
- **NONE** (Zero modifications to existing code or production files).

---

## 🔒 Production Safety Verification

- **Production Checkpoint**: `checkpoints/final_k1/final_production.pth`
- **SHA256 Hash Before**: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d`
- **SHA256 Hash After**: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d`
- **Integrity Status**: **100% UNTOUCHED & FROZEN** ✅
- **Raw Datasets (`Le2i/`, `URFD/`, `dataset/`)**: **100% UNTOUCHED**.
- **Model K1 Source Code (`src/train_final_k1.py`)**: **100% UNTOUCHED**.
- **Streamlit Application (`app.py`)**: **100% UNTOUCHED**.
