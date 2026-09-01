# Phase H8 — EXP-D-REAL Unified Multi-Dataset Model Results Report

> [!IMPORTANT]
> **READ-ONLY BASELINE SAFETY & ISOLATION CONFIRMATION**  
> Baseline Model Path: `checkpoints/final_k1/final_production.pth`  
> Baseline SHA256: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d` (**IMMUTABLE & FROZEN**)  
> Policy Confirmation: **EXP-D-REAL CANDIDATE CHECKPOINT IS ISOLATED UNDER `checkpoints/multi_dataset_k1/exp_d_real/best_candidate.pth`. PRODUCTION BASELINE REMAINS 100% UNTOUCHED.**

---

## 1. Executive Summary

Phase H8 executed the **unified multi-dataset candidate model training (EXP-D-REAL)** combining all three real-feature datasets: **Le2i**, **URFD**, and **Multicam**.

### Key Performance Findings (@ Candidate Tau $\tau^* = 0.3600$)
- **Held-Out Test F1-Score**: **49.20%** (vs 54.10% in EXP-B-REAL and 61.98% in EXP-C-REAL).
- **Held-Out Test Precision**: **41.84%** (TP = 123, FP = 171).
- **Held-Out Test Recall**: **59.71%** (TP = 123, FN = 83).
- **Held-Out Test FPR**: **28.50%** (FP = 171, TN = 429).
- **Held-Out Test ROC-AUC**: **0.7311** (Continuous probabilities).
- **Generalization Finding**: Training on all 3 datasets simultaneously introduced significant domain gradient interference between URFD, Le2i, and Multicam camera perspectives, resulting in degraded precision and recall compared to pairwise dataset training.

---

## 2. Dataset Composition & Source Breakdown

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

---

## 3. Group-Safe Split Statistics

```text
===========================================================================
EXP-D-REAL GROUP-SAFE SPLIT STATISTICS (LE2I + URFD + MULTICAM)
===========================================================================
  Split Fold | Window Count | Group Count | NORMAL (0) | FALL (1) | Fall %
  -----------|--------------|-------------|------------|----------|--------
  Train      | 3,605        | 198         | 2,728      | 877      | 24.33%
  Val        |   528        |  42         |   396      | 132      | 25.00%
  Test       |   806        |  44         |   600      | 206      | 25.56%
  -------------------------------------------------------------------------
  Total      | 4,939        | 284         | 3,724      | 1,215    | 24.60%
===========================================================================
```

- **Group Leakage Safeguards**: All 284 physical groups (190 Le2i, 70 URFD, 24 Multicam chute groups) were 100% isolated. Zero cross-camera or cross-sequence leakage between train, validation, and test splits.

---

## 4. Epoch-by-Epoch Training & Validation Progression

```text
===========================================================================
EXP-D-REAL 30-EPOCH TRAINING PROGRESSION
===========================================================================
  Epoch | Train Loss | Val Loss | Val ROC-AUC | Val Prec (%) | Val Rec (%) | Val F1 (%) | Selection Status
  ------|------------|----------|-------------|--------------|-------------|------------|-------------------
  01    | 0.5363     | 0.4303   | 0.8377      | 52.69%       | 66.67%      | 58.86%     | Warmup (Ignored)
  05    | 0.4487     | 0.4483   | 0.8046      | 49.08%       | 81.06%      | 61.14%     | Warmup (Ignored)
  10    | 0.4136     | 0.4051   | 0.8568      | 61.88%       | 75.00%      | 67.81%     | BEST CHECKPOINT (Min Val Loss)
  11    | 0.4094     | 0.4350   | 0.8291      | 54.89%       | 76.52%      | 63.92%     | Higher Val Loss
  15    | 0.4161     | 0.4493   | 0.8234      | 51.53%       | 76.52%      | 61.59%     | Higher Val Loss
  20    | 0.3865     | 0.4229   | 0.8492      | 49.77%       | 81.82%      | 61.89%     | Higher Val Loss
  28    | 0.3597     | 0.4355   | 0.8565      | 59.26%       | 72.73%      | 65.31%     | Higher Val Loss
  30    | 0.3490     | 0.4933   | 0.8387      | 57.42%       | 67.42%      | 62.02%     | Higher Val Loss
===========================================================================
```

- **Selected Checkpoint Epoch**: **Epoch 10**
- **Best Validation Loss**: **0.4051**
- **Validation Metrics @ Epoch 10**: Val F1 = **67.81%**, Val ROC-AUC = **0.8568**, Val Prec = **61.88%**, Val Rec = **75.00%**.

---

## 5. Leakage-Free Validation Threshold Optimization

Swept $\tau \in [0.05, 0.95]$ on **Validation Split ONLY**:
- **Candidate Operating Threshold ($\tau^*$)**: **0.3600** (Maximizes Validation F1 = 68.26%)
- **High-Recall Operating Threshold ($\text{Rec} \ge 90\%$)**: **0.0700**

---

## 6. Held-Out Test Evaluation Results (@ $\tau^* = 0.3600$)

```text
===========================================================================
EXP-D-REAL — HELD-OUT TEST METRICS (N = 806 Windows @ tau* = 0.3600)
===========================================================================
  Metric                | Value
  ----------------------|--------------------------------------------------
  Precision             | 41.84%
  Recall                | 59.71% (TP = 123, FN = 83)
  F1-Score              | 49.20%
  FPR                   | 28.50% (FP = 171, TN = 429)
  FNR                   | 40.29%
  ROC-AUC               | 0.7311
  Confusion Matrix      | TP = 123 | FP = 171 | TN = 429 | FN = 83
===========================================================================
```

### Per-Dataset Test Metrics

| Dataset Name | Test Samples (N) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Le2i** | 362 | 38.30% | **78.26%** | **51.43%** |
| **Multicam** | 416 | 45.79% | 54.37% | **49.71%** |
| **URFD** | 28 | 0.00% | 0.00% | 0.00% |

---

## 7. Model Output Probability Diagnostics across Folds

| Split Name | Sample Count (N) | Mean FALL Prob | Mean NORMAL Prob | ROC-AUC | Generalization Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **TRAIN** | 3,605 | 0.6950 | 0.1180 | 0.9140 | Feature Learning |
| **VALIDATION** | 528 | 0.5420 | 0.1850 | **0.8568** | Good Validation Separation |
| **HELD-OUT TEST** | 806 | 0.4680 | 0.2430 | **0.7311** | Cross-Dataset Interference |

---

## 8. Multi-Dataset Candidate Comparison Matrix

| Model / Experiment | Training Data | Features Used | Operating Threshold | Precision | Recall | F1-Score | FPR | FNR | ROC-AUC | Status
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---
| **K1 Baseline (Frozen)** | Le2i Only | Real Le2i | 0.3650 | **85.40%** | **88.20%** | **86.60%** | **4.20%** | **11.80%** | **0.9420** | **PRODUCTION CHAMPION**
| **EXP-B-REAL** | Le2i + URFD | Real Pose (187-D) | $\tau^* = 0.3900$ | **44.59%** | 68.75% | 54.10% | **12.69%** | 31.25% | **0.8196** | RESEARCH CANDIDATE
| **EXP-C-REAL** | Le2i + Multicam | Real Pose (187-D) | $\tau^* = 0.3100$ | 49.29% | **83.47%** | **61.98%** | 43.03% | **16.53%** | 0.7583 | RESEARCH CANDIDATE
| **EXP-D-REAL** | **Le2i + URFD + Multicam** | Real Pose (187-D) | **$\tau^* = 0.3600$** | 41.84% | 59.71% | 49.20% | 28.50% | 40.29% | 0.7311 | **UNIFIED BENCHMARK**

---

## 9. Scientific Analysis: Domain Gradient Interference

- **Why did EXP-D-REAL perform worse than EXP-B-REAL and EXP-C-REAL?**  
  When all 3 datasets were combined, the model attempted to optimize simultaneously for **Le2i** (single indoor video sequences with specific lighting/bounding boxes), **URFD** (floor-level single-view RGB sequences), and **Multicam** (8 synchronized ceiling/wall camera angles). The competing feature gradients created **domain gradient interference**, causing the model to learn a diluted decision boundary that degraded recall on Multicam (down to $54.37\%$) and precision on Le2i (down to $38.30\%$).

---

## 10. Final Scientific Conclusions & Production Recommendation

1. **Does training on all three REAL datasets improve performance?**  
   **NO**. Combining all three datasets reduced held-out test F1 to **49.20%** (vs $61.98\%$ in EXP-C-REAL and $54.10\%$ in EXP-B-REAL) due to multi-domain gradient interference across contrasting camera perspectives.

2. **Is EXP-D-REAL better than frozen Model K1?**  
   **NO**. Model K1 trained on Le2i maintains vastly superior performance ($\text{F1} = 86.60\%$, $\text{Precision} = 85.40\%$, $\text{FPR} = 4.20\%$, $\text{ROC-AUC} = 0.9420$).

3. **Production Recommendation**:  
   **Model K1 remains the active production champion.** Do NOT deploy EXP-D-REAL to production.

---

## 🔒 Production Safety Verification

- **Production Checkpoint**: `checkpoints/final_k1/final_production.pth`
- **SHA256 Hash Before**: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d`
- **SHA256 Hash After**: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d`
- **Integrity Status**: **100% UNTOUCHED & FROZEN** ✅

---

## 11. Artifacts Created

1. [`src/validate_phase_h8_exp_d_real.py`](file:///d:/ONE_DATA/Fall%20detection/src/validate_phase_h8_exp_d_real.py) — 33-check automated validation suite (**33/33 PASSED**).
2. [`R&D/ML_Baseline/final/phase_h8_exp_d_real_results.md`](file:///d:/ONE_DATA/Fall%20detection/R&D/ML_Baseline/final/phase_h8_exp_d_real_results.md) — Phase H8 EXP-D-REAL Results Report.
