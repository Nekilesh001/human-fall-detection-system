# Phase H14 — EXP-K2-D Unified Real Multi-Dataset Candidate Results Report

> [!IMPORTANT]
> **READ-ONLY BASELINE SAFETY & ISOLATION CONFIRMATION**  
> Baseline Model Path: `checkpoints/final_k1/final_production.pth`  
> Baseline SHA256: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d` (**IMMUTABLE & FROZEN**)  
> Policy Confirmation: **EXP-K2-D CANDIDATE CHECKPOINT IS ISOLATED UNDER `checkpoints/multi_dataset_k2/exp_k2_d/best_candidate.pth`. PRODUCTION BASELINE REMAINS 100% UNTOUCHED.**

---

## 1. Executive Summary

Phase H14 executed candidate model training for **EXP-K2-D (Unified Le2i + URFD + Multicam)** using **Model K2 Dual-Stream TCN** and real 187-D YOLOv8-pose features. Feature standardization was performed using `FeatureStandardScaler` fitted **ONLY** on training split features.

### Key Performance Findings (@ Candidate Tau $\tau^* = 0.0700$)
- **Held-Out Test F1-Score**: **59.71%** (vs 49.20% in EXP-D-REAL — a **$+10.51\%$ boost in unified multi-dataset F1!**).
- **Held-Out Test ROC-AUC**: **0.8114** (vs 0.7311 in EXP-D-REAL — a **$+0.0803$ jump in ROC-AUC!**).
- **Held-Out Test Recall**: **80.58%** (TP = 166, FN = 40).
- **Held-Out Test Precision**: **47.43%** (TP = 166, FP = 184).
- **Held-Out Test PR-AUC**: **0.5091**.
- **Held-Out Test FPR**: **30.67%** (FP = 184, TN = 416).
- **Held-Out Test FNR**: **19.42%**.

---

## 2. Training Configuration & Hardware Details

| Parameter | Value |
| :--- | :--- |
| **Experiment Name** | EXP-K2-D |
| **Training Datasets** | Le2i + URFD + Multicam (Unified 3 Datasets) |
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
EXP-K2-D GROUP-SAFE SPLIT STATISTICS (LE2I + URFD + MULTICAM)
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
EXP-K2-D 30-EPOCH TRAINING PROGRESSION
===========================================================================
  Epoch | Train Loss | Val Loss | Val ROC-AUC | Val Prec (%) | Val Rec (%) | Val F1 (%) | Selection Status
  ------|------------|----------|-------------|--------------|-------------|------------|-------------------
  01    | 0.6373     | 0.4792   | 0.8356      | 58.54%       | 18.18%      | 27.75%     | Warmup (Ignored)
  05    | 0.3928     | 0.3845   | 0.8838      | 77.27%       | 38.64%      | 51.52%     | Warmup (Ignored)
  10    | 0.3398     | 0.4673   | 0.8553      | 71.43%       | 45.45%      | 55.56%     | Candidate Checkpoint
  11    | 0.3237     | 0.4464   | 0.8590      | 68.24%       | 43.94%      | 53.46%     | Candidate Checkpoint
  12    | 0.3228     | 0.4279   | 0.8696      | 74.47%       | 53.03%      | 61.95%     | Candidate Checkpoint
  13    | 0.3208     | 0.4133   | 0.8715      | 65.25%       | 58.33%      | 61.60%     | BEST CHECKPOINT (Min Val Loss)
  14    | 0.3018     | 0.4141   | 0.8728      | 67.27%       | 56.06%      | 61.16%     | Higher Val Loss
  20    | 0.2604     | 0.4913   | 0.8795      | 71.88%       | 52.27%      | 60.53%     | Higher Val Loss
  28    | 0.2322     | 0.5491   | 0.8607      | 62.39%       | 51.52%      | 56.43%     | Higher Val Loss
  30    | 0.2121     | 0.6476   | 0.8794      | 71.74%       | 50.00%      | 58.93%     | Higher Val Loss
===========================================================================
```

- **Selected Checkpoint Epoch**: **Epoch 13**
- **Best Validation Loss**: **0.4133**
- **Candidate Operating Threshold ($\tau^*$)**: **0.0700** (Derived from Validation probabilities ONLY)

---

## 5. Held-Out Test Evaluation Results (@ $\tau^* = 0.0700$)

```text
===========================================================================
EXP-K2-D — HELD-OUT TEST METRICS (N = 806 Windows @ tau* = 0.0700)
===========================================================================
  Metric                | Value
  ----------------------|--------------------------------------------------
  Precision             | 47.43%
  Recall                | 80.58% (TP = 166, FN = 40)
  F1-Score              | 59.71%
  FPR                   | 30.67% (FP = 184, TN = 416)
  FNR                   | 19.42%
  ROC-AUC               | 0.8114
  PR-AUC                | 0.5091
  Confusion Matrix      | TP = 166 | FP = 184 | TN = 416 | FN = 40
===========================================================================
```

### Per-Dataset Test Breakdown

| Dataset Name | Test Samples (N) | Precision (%) | Recall (%) | F1-Score (%) | FPR (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Le2i** | 362 | 37.11% | **78.26%** | **50.35%** | 19.30% |
| **URFD** | 28 | 0.00% | 0.00% | 0.00% | 10.71% |
| **Multicam** | 416 | 52.00% | **81.25%** | **63.41%** | 46.88% |

---

## 6. Model Output Probability Diagnostics across Folds

```text
===========================================================================
PROBABILITY DIAGNOSTICS BY SPLIT FOLD
===========================================================================
  Split Fold | Sample Count (N) | Group Count | Mean FALL Prob | Mean NORMAL Prob | ROC-AUC | PR-AUC
  -----------|------------------|-------------|----------------|------------------|---------|-------
  TRAIN      | 3,605            | 198         | 0.7620         | 0.0650           | 0.9650  | 0.8940
  VAL        |   528            |  42         | 0.5420         | 0.1380           | 0.8715  | 0.7420
  TEST       |   806            |  44         | 0.4850         | 0.1520           | 0.8114  | 0.5091
===========================================================================
```

---

## 7. Grand Benchmarking Comparison Across All Model K1 and K2 Experiments

| Model / Experiment | Training Data | Features / Scaling | Operating Threshold ($\tau$) | Precision | Recall | F1-Score | FPR | FNR | ROC-AUC | Status
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---
| **K1 Baseline (Frozen)** | Le2i Only | Real Le2i (Unscaled) | 0.3650 | **85.40%** | **88.20%** | **86.60%** | **4.20%** | **11.80%** | **0.9420** | **PRODUCTION CHAMPION**
| **EXP-K2-A** | Le2i Only | Model K2 + StandardScaler | $\tau^* = 0.4800$ | 56.06% | 66.07% | **60.66%** | **10.43%** | 33.93% | **0.8845** | K2 SINGLE-DATASET
| **EXP-K2-B** | Le2i + URFD | Model K2 + StandardScaler | $\tau^* = 0.3800$ | 42.67% | 66.67% | 52.03% | 13.31% | 33.33% | **0.8696** | K2 DUAL-DATASET
| **EXP-K2-C** | Le2i + Multicam | Model K2 + StandardScaler | $\tau^* = 0.1900$ | **54.32%** | **81.05%** | **65.05%** | 34.14% | **18.95%** | 0.7924 | K2 MULTI-VIEW
| **EXP-D-REAL (K1)** | Le2i+URFD+Multicam | Model K1 (Unscaled) | $\tau^* = 0.3600$ | 41.84% | 59.71% | 49.20% | 28.50% | 40.29% | 0.7311 | OLD K1 UNIFIED
| **EXP-K2-D** | **Unified (All 3)** | **Model K2 + StandardScaler** | **$\tau^* = 0.0700$** | 47.43% | **80.58%** | **59.71%** | 30.67% | 19.42% | **0.8114** | **K2 UNIFIED BENCHMARK**

### Key Scientific Insights from EXP-K2-D:
1. **Model K2 vs Model K1 on Unified Data**: Model K2 achieved **F1 = 59.71%** and **ROC-AUC = 0.8114**, massively outperforming Model K1's EXP-D-REAL (**F1 = 49.20%, ROC-AUC = 0.7311** — a **$+10.51\%$ F1 gain and $+0.0803$ ROC-AUC gain**).
2. **Domain Gradient Interference Analysis**: While Model K2 handles feature standardization much better, combining all 3 datasets without dataset rebalancing still shows domain conflict between Multicam (8 camera views per chute) and Le2i/URFD.

---

## 8. Summary of Files Created & Modified

### Files Created (2 Files)
1. [`src/validate_phase_h14_k2_d.py`](file:///d:/ONE_DATA/Fall%20detection/src/validate_phase_h14_k2_d.py) — 38-check automated validation suite (**38/38 PASSED**).
2. [`R&D/ML_Baseline/final/phase_h14_exp_k2_d_results.md`](file:///d:/ONE_DATA/Fall%20detection/R&D/ML_Baseline/final/phase_h14_exp_k2_d_results.md) — Phase H14 EXP-K2-D Results Report.

### Files Modified (0 Files)
- **NONE** (Zero modifications to existing code or production files).

---

## 🔒 Final Production Safety Declarations

- **Explicit Confirmation 1**: Production Model K1 (`checkpoints/final_k1/final_production.pth`) remains **100% UNTOUCHED & FROZEN** (SHA256: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d`).
- **Explicit Confirmation 2**: Streamlit application `app.py` remains **100% UNTOUCHED**.
- **Explicit Confirmation 3**: Raw datasets (`Le2i/`, `URFD/`, `dataset/`) remain **100% UNTOUCHED**.
- **Explicit Confirmation 4**: EXP-K2-D candidate training completed successfully with outputs isolated in `checkpoints/multi_dataset_k2/exp_k2_d/`.
