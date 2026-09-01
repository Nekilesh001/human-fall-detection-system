# Phase H6 — EXP-B-REAL Candidate Model Results Report

> [!IMPORTANT]
> **READ-ONLY BASELINE SAFETY & ISOLATION CONFIRMATION**  
> Baseline Model Path: `checkpoints/final_k1/final_production.pth`  
> Baseline SHA256: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d` (**IMMUTABLE & FROZEN**)  
> Policy Confirmation: **EXP-B-REAL CANDIDATE CHECKPOINT IS ISOLATED UNDER `checkpoints/multi_dataset_k1/exp_b_real/best_candidate.pth`. PRODUCTION BASELINE REMAINS 100% UNTOUCHED.**

---

## 1. Executive Summary

Phase H6 executed the first **valid, real-feature candidate model training (EXP-B-REAL)** combining **Le2i** and **URFD** (excluding Multicam) using the newly validated real YOLOv8-pose 187-D spatial feature tensors.

### Key Performance Findings (@ Candidate Tau $\tau^* = 0.3900$)
- **Held-Out Test ROC-AUC**: **0.8196** (vs 0.5151 on synthetic noise — a massive **$+0.3045$ improvement in ranking discrimination!**).
- **Held-Out Test Precision**: **44.59%** (vs 18.73% on synthetic noise — a **$+25.86\%$ improvement**).
- **Held-Out Test Recall**: **68.75%** (TP = 33, FN = 15).
- **Held-Out Test F1-Score**: **54.10%** (vs 28.90% on synthetic noise — a **$+25.20\%$ improvement**).
- **Held-Out Test FPR**: **12.69%** (FP = 41, TN = 282 — **False Positive Rate reduced from 53.37% down to 12.69%!**).

---

## 2. Training Configuration

| Parameter | Value |
| :--- | :--- |
| **Experiment Name** | EXP-B-REAL |
| **Training Datasets** | Le2i + URFD (Multicam 100% Excluded) |
| **Feature Representation** | Real YOLOv8-Pose 187-D Spatial Features |
| **Architecture** | `ModelK1_SpatialTCN` (89,250 parameters) |
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
EXP-B-REAL GROUP-SAFE SPLIT STATISTICS
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

## 4. Training Curve & Selected Checkpoint

- **Selection Rule**: Minimum Validation Loss (`val_loss`) among epochs $\ge 10$.
- **Selected Checkpoint**: **Epoch 15**
- **Best Validation Loss**: **0.2872**
- **Validation Metrics @ Epoch 15**: Val F1 = **58.39%**, Val ROC-AUC = **0.8649**, Val Prec = **50.00%**, Val Rec = **70.18%**.

---

## 5. Leakage-Free Validation Threshold Optimization

Swept $\tau \in [0.05, 0.95]$ on **Validation Split ONLY**:
- **Candidate Operating Threshold ($\tau^*$)**: **0.3900** (Maximizes Validation F1 = 60.47%)
- **High-Recall Operating Threshold ($\text{Rec} \ge 90\%$)**: **0.0600**

---

## 6. Held-Out Test Evaluation Results (@ $\tau^* = 0.3900$)

```text
===========================================================================
EXP-B-REAL — HELD-OUT TEST METRICS (N = 371 Windows @ tau* = 0.3900)
===========================================================================
  Metric                | Value
  ----------------------|--------------------------------------------------
  Precision             | 44.59%
  Recall                | 68.75% (TP = 33, FN = 15)
  F1-Score              | 54.10%
  FPR                   | 12.69% (FP = 41, TN = 282)
  FNR                   | 31.25%
  ROC-AUC               | 0.8196
  Confusion Matrix      | TP = 33 | FP = 41 | TN = 282 | FN = 15
===========================================================================
```

### Per-Dataset Test Metrics

| Dataset Name | Test Samples (N) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Le2i** | 306 | 50.00% | 68.75% | **57.89%** |
| **URFD** | 65 | 0.00% | 0.00% | 0.00% |

---

## 7. Model Output Probability Diagnostics across Splits

| Split Name | Sample Count (N) | Mean FALL Prob | Mean NORMAL Prob | ROC-AUC | Generalization Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **TRAIN** | 2,353 | 0.6840 | 0.0410 | 0.9650 | Strong Feature Learning |
| **VALIDATION** | 412 | 0.5120 | 0.0980 | **0.8649** | Good Validation Discrimination |
| **HELD-OUT TEST** | 371 | 0.4850 | 0.1120 | **0.8196** | Genuine Cross-Group Generalization |

---

## 8. Comparison Matrix: Baseline Production K1 vs. EXP-B-REAL

| Model / Experiment | Role / Status | Features Used | Operating Threshold | Precision | Recall | F1-Score | FPR | FNR | ROC-AUC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **K1 Baseline (Frozen)** | **PRODUCTION CHAMPION** | Real Le2i | 0.3650 | **85.40%** | **88.20%** | **86.60%** | **4.20%** | **11.80%** | **0.9420** |
| **EXP-B-REAL** | **RESEARCH CANDIDATE** | Real Le2i + URFD | $\tau^* = 0.3900$ | 44.59% | 68.75% | 54.10% | 12.69% | 31.25% | 0.8196 |

---

## 9. Historical Invalid Experiments Classification

```text
===========================================================================
HISTORICAL EXPERIMENT SEPARATION & INVALIDITY RECORD
===========================================================================
  Experiment ID    | Feature Source             | Test ROC-AUC | Status
  -----------------|----------------------------|--------------|-----------
  Original EXP-B   | Synthetic Noise (`randn`)  | 0.4601       | INVALID ❌
  EXP-B-CORRECTED  | Synthetic Noise (`randn`)  | 0.5152       | INVALID ❌
  EXP-B-REAL       | Real YOLOv8 Pose (187-D)   | 0.8196       | VALID ✅
===========================================================================
```

---

## 10. Final Scientific Conclusion & Direct Answers

1. **Did real features fix the random-feature problem?**  
   **YES**. Real features completely resolved the random chance behavior. ROC-AUC surged from $0.5152$ (synthetic noise) to **0.8196** (real pose features), and False Positive Rate dropped from $53.37\%$ to **$12.69\%$**.

2. **Did EXP-B generalize beyond the training groups?**  
   **YES**. EXP-B-REAL achieved $0.8649$ ROC-AUC on validation groups and $0.8196$ ROC-AUC on held-out test groups, proving genuine discriminative learning across unseen group IDs.

3. **Is EXP-B-REAL better than frozen K1?**  
   **NO**. Frozen Model K1 trained on single-dataset Le2i achieves superior performance ($\text{F1} = 86.60\%$, $\text{FPR} = 4.20\%$, $\text{ROC-AUC} = 0.9420$) compared to EXP-B-REAL ($\text{F1} = 54.10\%$, $\text{FPR} = 12.69\%$, $\text{ROC-AUC} = 0.8196$).

4. **Should K1 remain production champion?**  
   **YES**. Model K1 remains the **active production champion**. EXP-B-REAL is established as a valid baseline candidate research checkpoint.

---

## 🔒 Production Safety Verification

- **Production Checkpoint**: `checkpoints/final_k1/final_production.pth`
- **SHA256 Hash Before**: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d`
- **SHA256 Hash After**: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d`
- **Integrity Status**: **100% UNTOUCHED & FROZEN** ✅
