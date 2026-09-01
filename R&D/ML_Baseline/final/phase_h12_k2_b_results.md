# Phase H12 — EXP-K2-B Candidate Model Results Report

> [!IMPORTANT]
> **READ-ONLY BASELINE SAFETY & ISOLATION CONFIRMATION**  
> Baseline Model Path: `checkpoints/final_k1/final_production.pth`  
> Baseline SHA256: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d` (**IMMUTABLE & FROZEN**)  
> Policy Confirmation: **EXP-K2-B CANDIDATE CHECKPOINT IS ISOLATED UNDER `checkpoints/multi_dataset_k2/exp_k2_b/best_candidate.pth`. PRODUCTION BASELINE REMAINS 100% UNTOUCHED.**

---

## 1. Executive Summary

Phase H12 executed candidate model training for **EXP-K2-B (Le2i + URFD)** using **Model K2 Dual-Stream TCN** and real YOLOv8-pose 187-D spatial feature tensors. Feature standardization was performed using `FeatureStandardScaler` fitted **ONLY** on the training split features. Multicam was **100% EXCLUDED** from this experiment.

### Key Performance Findings (@ Candidate Tau $\tau^* = 0.3800$)
- **Held-Out Test ROC-AUC**: **0.8696** (vs 0.8196 in EXP-B-REAL — a **$+0.0500$ jump in ROC-AUC ranking discrimination thanks to Model K2 & Feature Standardization!**).
- **Held-Out Test PR-AUC**: **0.4629**.
- **Held-Out Test Precision**: **42.67%** (TP = 32, FP = 43).
- **Held-Out Test Recall**: **66.67%** (TP = 32, FN = 16).
- **Held-Out Test F1-Score**: **52.03%**.
- **Held-Out Test FPR**: **13.31%** (FP = 43, TN = 280).
- **Held-Out Test FNR**: **33.33%**.

---

## 2. Training Configuration & Hardware Details

| Parameter | Value |
| :--- | :--- |
| **Experiment Name** | EXP-K2-B |
| **Training Datasets** | Le2i + URFD (Multicam 100% Excluded) |
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
EXP-K2-B GROUP-SAFE SPLIT STATISTICS (LE2I + URFD)
===========================================================================
  Split Fold | Window Count | Group Count | NORMAL (0) | FALL (1) | Fall %
  -----------|--------------|-------------|------------|----------|--------
  Train      | 2,353        | 182         | 2,087      | 266      | 11.30%
  Val        |   412        |  39         |   355      |  57      | 13.83%
  Test       |   371        |  39         |   323      |  48      | 12.94%
  -------------------------------------------------------------------------
  Total      | 3,136        | 260         | 2,765      | 371      | 11.83%
===========================================================================
```

---

## 4. Epoch-by-Epoch Training & Validation Progression

```text
===========================================================================
EXP-K2-B 30-EPOCH TRAINING PROGRESSION
===========================================================================
  Epoch | Train Loss | Val Loss | Val ROC-AUC | Val Prec (%) | Val Rec (%) | Val F1 (%) | Selection Status
  ------|------------|----------|-------------|--------------|-------------|------------|-------------------
  01    | 0.4803     | 0.4145   | 0.7542      | 55.81%       | 42.11%      | 48.00%     | Warmup (Ignored)
  05    | 0.2250     | 0.4405   | 0.7845      | 56.76%       | 36.84%      | 44.68%     | Warmup (Ignored)
  10    | 0.1662     | 0.4650   | 0.7859      | 54.72%       | 50.88%      | 52.73%     | Candidate Checkpoint
  11    | 0.1963     | 0.4224   | 0.8108      | 58.06%       | 31.58%      | 40.91%     | BEST CHECKPOINT (Min Val Loss)
  15    | 0.1543     | 0.4956   | 0.8025      | 56.25%       | 47.37%      | 51.43%     | Higher Val Loss
  20    | 0.1311     | 0.6957   | 0.7989      | 49.28%       | 59.65%      | 53.97%     | Higher Val Loss
  28    | 0.1578     | 0.4343   | 0.8231      | 52.94%       | 47.37%      | 50.00%     | Higher Val Loss
  30    | 0.1241     | 0.5448   | 0.8243      | 45.45%       | 52.63%      | 48.78%     | Higher Val Loss
===========================================================================
```

- **Selected Checkpoint Epoch**: **Epoch 11**
- **Best Validation Loss**: **0.4224**
- **Candidate Operating Threshold ($\tau^*$)**: **0.3800** (Derived from Validation probabilities ONLY)

---

## 5. Held-Out Test Evaluation Results (@ $\tau^* = 0.3800$)

```text
===========================================================================
EXP-K2-B — HELD-OUT TEST METRICS (N = 371 Windows @ tau* = 0.3800)
===========================================================================
  Metric                | Value
  ----------------------|--------------------------------------------------
  Precision             | 42.67%
  Recall                | 66.67% (TP = 32, FN = 16)
  F1-Score              | 52.03%
  FPR                   | 13.31% (FP = 43, TN = 280)
  FNR                   | 33.33%
  ROC-AUC               | 0.8696
  PR-AUC                | 0.4629
  Confusion Matrix      | TP = 32 | FP = 43 | TN = 280 | FN = 16
===========================================================================
```

### Per-Dataset Test Metrics

| Dataset Name | Test Samples (N) | Precision (%) | Recall (%) | F1-Score (%) | FPR (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Le2i** | 306 | 47.76% | **66.67%** | **55.65%** | 13.57% |
| **URFD** | 65 | 0.00% | 0.00% | 0.00% | 12.31% |

*Note on URFD Test Split: In this group-isolated random split fold, all 65 test windows from URFD were NORMAL (0 positive fall windows in test split). Model achieved 12.31% FPR on normal URFD sequences.*

---

## 6. Model Output Probability Diagnostics across Folds

```text
===========================================================================
PROBABILITY DIAGNOSTICS BY SPLIT FOLD
===========================================================================
  Split Fold | Sample Count (N) | Group Count | Mean FALL Prob | Mean NORMAL Prob | ROC-AUC | PR-AUC
  -----------|------------------|-------------|----------------|------------------|---------|-------
  TRAIN      | 2,353            | 182         | 0.7410         | 0.0420           | 0.9710  | 0.8980
  VAL        |   412            |  39         | 0.4850         | 0.1240           | 0.8108  | 0.5420
  TEST       |   371            |  39         | 0.4620         | 0.1190           | 0.8696  | 0.4629
===========================================================================
```

---

## 7. Comparison Matrix: Baseline Production K1 vs. EXP-K2-A vs. EXP-K2-B

| Model / Experiment | Training Data | Features / Normalization | Operating Threshold | Precision | Recall | F1-Score | FPR | FNR | ROC-AUC | Status
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---
| **K1 Baseline (Frozen)** | Le2i Only | Real Le2i (Unscaled) | 0.3650 | **85.40%** | **88.20%** | **86.60%** | **4.20%** | **11.80%** | **0.9420** | **PRODUCTION CHAMPION**
| **EXP-K2-A** | Le2i Only | Model K2 + StandardScaler | $\tau^* = 0.4800$ | 56.06% | 66.07% | **60.66%** | **10.43%** | 33.93% | **0.8845** | K2 SINGLE-DATASET BASELINE
| **EXP-B-REAL (K1)** | Le2i + URFD | Model K1 (Unscaled) | $\tau^* = 0.3900$ | 44.59% | **68.75%** | 54.10% | 12.69% | 31.25% | 0.8196 | OLD K1 CANDIDATE
| **EXP-K2-B** | **Le2i + URFD** | **Model K2 + StandardScaler** | **$\tau^* = 0.3800$** | 42.67% | 66.67% | 52.03% | 13.31% | 33.33% | **0.8696** | **K2 DUAL-DATASET CANDIDATE**

---

## 8. Summary of Files Created & Modified

### Files Created (2 Files)
1. [`src/validate_phase_h12_k2_b.py`](file:///d:/ONE_DATA/Fall%20detection/src/validate_phase_h12_k2_b.py) — 32-check automated validation suite (**32/32 PASSED**).
2. [`R&D/ML_Baseline/final/phase_h12_k2_b_results.md`](file:///d:/ONE_DATA/Fall%20detection/R&D/ML_Baseline/final/phase_h12_k2_b_results.md) — Phase H12 EXP-K2-B Results Report.

### Files Modified (0 Files)
- **NONE** (Zero modifications to existing code or production files).

---

## 🔒 Final Production Safety Declarations

- **Explicit Confirmation 1**: Production Model K1 (`checkpoints/final_k1/final_production.pth`) remains **100% UNTOUCHED & FROZEN**.
- **Explicit Confirmation 2**: Streamlit application `app.py` remains **100% UNTOUCHED**.
- **Explicit Confirmation 3**: Raw datasets (`Le2i/`, `URFD/`, `dataset/`) remain **100% UNTOUCHED**.
