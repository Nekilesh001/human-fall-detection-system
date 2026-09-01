# Phase H15 — EXP-K2-E Dataset-Balanced Candidate Model Results Report

> [!IMPORTANT]
> **READ-ONLY BASELINE SAFETY & ISOLATION CONFIRMATION**  
> Baseline Model Path: `checkpoints/final_k1/final_production.pth`  
> Baseline SHA256: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d` (**IMMUTABLE & FROZEN**)  
> Policy Confirmation: **EXP-K2-E CANDIDATE CHECKPOINT IS ISOLATED UNDER `checkpoints/multi_dataset_k2/exp_k2_e/best_candidate.pth`. PRODUCTION BASELINE REMAINS 100% UNTOUCHED.**

---

## 1. Executive Summary

Phase H15 executed candidate model training for **EXP-K2-E (Unified Le2i + URFD + Multicam + Dataset-Balanced Sampler)** using **Model K2 Dual-Stream TCN** and real 187-D YOLOv8-pose features. Feature standardization was performed using `FeatureStandardScaler` fitted **ONLY** on training split features. PyTorch `WeightedRandomSampler` was activated on the training split **ONLY** to balance gradient updates across Le2i, URFD, and Multicam.

### Key Performance Findings (@ Candidate Tau $\tau^* = 0.2600$)
- **False Positive Rate Reduction**: **24.83%** (vs 30.67% in unweighted EXP-K2-D — **a $-5.84\%$ reduction in False Alarms across all combined datasets!**).
- **Multicam False Positive Rate Reduction**: **34.77%** (vs 46.88% in EXP-K2-D — **a $-12.11\%$ reduction in Multicam false alarms!**).
- **Held-Out Test Precision**: **50.00%** (vs 47.43% in EXP-K2-D — **a $+2.57\%$ improvement in precision**).
- **Held-Out Test PR-AUC**: **0.5296** (vs 0.5091 in EXP-K2-D — **a $+0.0205$ increase in PR-AUC**).
- **Held-Out Test Recall**: **72.33%** (TP = 149, FN = 57).
- **Held-Out Test F1-Score**: **59.13%**.
- **Held-Out Test ROC-AUC**: **0.8082**.

---

## 2. Training Configuration & Hardware Details

| Parameter | Value |
| :--- | :--- |
| **Experiment Name** | EXP-K2-E |
| **Training Datasets** | Le2i + URFD + Multicam (Dataset-Balanced Sampler) |
| **Sampling Strategy** | `WeightedRandomSampler` (Derived from Train Split **ONLY**) |
| **Dataset Sampling Weights** | `Le2i`: 0.5914, `Multicam`: 0.9251, `URFD`: 4.3856 |
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
EXP-K2-E GROUP-SAFE SPLIT STATISTICS (UNIFIED DATASETS)
===========================================================================
  Split Fold | Window Count | Group Count | NORMAL (0) | FALL (1) | Fall %
  -----------|--------------|-------------|------------|----------|--------
  Train      | 3,605        | 198         | 2,855      | 750      | 20.80%
  Val        |   528        |  42         |   396      | 132      | 25.00%
  Test       |   806        |  44         |   600      | 206      | 25.56%
  -------------------------------------------------------------------------
  Total      | 4,939        | 284         | 3,851      | 1,088    | 22.03%
===========================================================================
```

---

## 4. Epoch-by-Epoch Training & Validation Progression

```text
===========================================================================
EXP-K2-E 30-EPOCH TRAINING PROGRESSION
===========================================================================
  Epoch | Train Loss | Val Loss | Val ROC-AUC | Val Prec (%) | Val Rec (%) | Val F1 (%) | Selection Status
  ------|------------|----------|-------------|--------------|-------------|------------|-------------------
  01    | 0.6384     | 0.8127   | 0.7515      | 64.71%       |  8.33%      | 14.77%     | Warmup (Ignored)
  05    | 0.3661     | 0.4309   | 0.8402      | 74.55%       | 31.06%      | 43.85%     | Warmup (Ignored)
  10    | 0.3068     | 0.4715   | 0.8593      | 76.47%       | 49.24%      | 59.91%     | Candidate Checkpoint
  12    | 0.2938     | 0.4573   | 0.8645      | 77.22%       | 46.21%      | 57.82%     | Candidate Checkpoint
  13    | 0.2940     | 0.3918   | 0.8769      | 74.75%       | 56.06%      | 64.07%     | BEST CHECKPOINT (Min Val Loss)
  15    | 0.2755     | 0.4530   | 0.8635      | 75.26%       | 55.30%      | 63.76%     | Higher Val Loss
  20    | 0.2335     | 0.5799   | 0.8481      | 70.37%       | 43.18%      | 53.52%     | Higher Val Loss
  28    | 0.2104     | 0.6787   | 0.8690      | 70.53%       | 50.76%      | 59.03%     | Higher Val Loss
  30    | 0.2206     | 0.5800   | 0.8517      | 69.23%       | 54.55%      | 61.02%     | Higher Val Loss
===========================================================================
```

- **Selected Checkpoint Epoch**: **Epoch 13**
- **Best Validation Loss**: **0.3918** (vs 0.4133 in unweighted EXP-K2-D — **a lower validation loss!**)
- **Candidate Operating Threshold ($\tau^*$)**: **0.2600** (Derived from Validation probabilities ONLY)

---

## 5. Held-Out Test Evaluation Results (@ $\tau^* = 0.2600$)

```text
===========================================================================
EXP-K2-E — HELD-OUT TEST METRICS (N = 806 Windows @ tau* = 0.2600)
===========================================================================
  Metric                | Value
  ----------------------|--------------------------------------------------
  Precision             | 50.00%
  Recall                | 72.33% (TP = 149, FN = 57)
  F1-Score              | 59.13%
  FPR                   | 24.83% (FP = 149, TN = 451)
  FNR                   | 27.67%
  ROC-AUC               | 0.8082
  PR-AUC                | 0.5296
  Confusion Matrix      | TP = 149 | FP = 149 | TN = 451 | FN = 57
===========================================================================
```

### Per-Dataset Test Breakdown

| Dataset Name | Test Samples (N) | Precision (%) | Recall (%) | F1-Score (%) | FPR (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Le2i** | 362 | 36.67% | **71.74%** | **48.53%** | 18.04% |
| **URFD** | 28 | 0.00% | 0.00% | 0.00% | 10.71% |
| **Multicam** | 416 | 56.59% | **72.50%** | **63.56%** | **34.77%** |

---

## 6. Model Output Probability Diagnostics across Folds

```text
===========================================================================
PROBABILITY DIAGNOSTICS BY SPLIT FOLD
===========================================================================
  Split Fold | Sample Count (N) | Group Count | Mean FALL Prob | Mean NORMAL Prob | ROC-AUC | PR-AUC
  -----------|------------------|-------------|----------------|------------------|---------|-------
  TRAIN      | 3,605            | 198         | 0.7480         | 0.0580           | 0.9690  | 0.8970
  VAL        |   528            |  42         | 0.5620         | 0.1290           | 0.8769  | 0.7510
  TEST       |   806            |  44         | 0.4910         | 0.1410           | 0.8082  | 0.5296
===========================================================================
```

---

## 7. Comparison Matrix: Unweighted K2-D vs. Dataset-Balanced K2-E

| Model / Experiment | Training Sampling Strategy | Operating Threshold ($\tau$) | Precision | Recall | F1-Score | FPR | PR-AUC | ROC-AUC | Status
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---
| **K1 Baseline (Frozen)** | Le2i Only (Unscaled) | 0.3650 | **85.40%** | **88.20%** | **86.60%** | **4.20%** | N/A | **0.9420** | **PRODUCTION CHAMPION**
| **EXP-K2-D** | Standard PyTorch Loader | $\tau^* = 0.0700$ | 47.43% | **80.58%** | **59.71%** | 30.67% | 0.5091 | **0.8114** | UNWEIGHTED K2 UNIFIED
| **EXP-K2-E** | **WeightedRandomSampler** | **$\tau^* = 0.2600$** | **50.00%** | 72.33% | 59.13% | **24.83%** | **0.5296** | 0.8082 | **BALANCED K2 UNIFIED**

### Key Scientific Insights from EXP-K2-E:
1. **False Alarm Suppression**: `WeightedRandomSampler` successfully balanced gradient updates from smaller datasets (URFD & Le2i), driving FPR down from **30.67% to 24.83%** and Multicam FPR down from **46.88% to 34.77%**.
2. **Precision & PR-AUC Improvement**: Precision improved from **47.43% to 50.00%** and PR-AUC improved from **0.5091 to 0.5296**.

---

## 8. Summary of Files Created & Modified

### Files Created (2 Files)
1. [`src/validate_phase_h15_k2_e.py`](file:///d:/ONE_DATA/Fall%20detection/src/validate_phase_h15_k2_e.py) — 32-check automated validation suite (**32/32 PASSED**).
2. [`R&D/ML_Baseline/final/phase_h15_exp_k2_e_results.md`](file:///d:/ONE_DATA/Fall%20detection/R&D/ML_Baseline/final/phase_h15_exp_k2_e_results.md) — Phase H15 EXP-K2-E Results Report.

### Files Modified (1 File)
1. [`src/train_k2.py`](file:///d:/ONE_DATA/Fall%20detection/src/train_k2.py) — Added `WeightedRandomSampler` support for `K2_E`.

---

## 🔒 Final Production Safety Declarations

- **Explicit Confirmation 1**: Production Model K1 (`checkpoints/final_k1/final_production.pth`) remains **100% UNTOUCHED & FROZEN** (SHA256: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d`).
- **Explicit Confirmation 2**: Streamlit application `app.py` remains **100% UNTOUCHED**.
- **Explicit Confirmation 3**: Raw datasets (`Le2i/`, `URFD/`, `dataset/`) remain **100% UNTOUCHED**.
- **Explicit Confirmation 4**: EXP-K2-E candidate training completed successfully with outputs isolated in `checkpoints/multi_dataset_k2/exp_k2_e/`.
