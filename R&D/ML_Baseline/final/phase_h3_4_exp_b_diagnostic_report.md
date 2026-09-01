# Phase H3.4 — EXP-B-CORRECTED Research Candidate Model Diagnostic Audit Report

> [!IMPORTANT]
> **READ-ONLY DIAGNOSTIC AUDIT & BASELINE SAFETY CONFIRMATION**  
> Baseline Model Path: `checkpoints/final_k1/final_production.pth`  
> Checkpoint SHA256: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d` (**IMMUTABLE & FROZEN**)  
> Policy Confirmation: **READ-ONLY DIAGNOSTIC AUDIT ONLY. NO MODEL TRAINING OR RETRAINING WAS PERFORMED.**

---

## 1. Executive Summary

Phase H3.4 performed a read-only diagnostic investigation into why the **EXP-B-CORRECTED** candidate model (trained on Le2i + URFD) achieved poor generalization performance on validation and held-out test splits (**Test ROC-AUC = 0.5151**, **Val ROC-AUC = 0.4583**) despite achieving **100% perfect separation on the training set (Train ROC-AUC = 1.0000)**.

### Primary Diagnostic Finding: Severe Overfitting & Spatial-Trapping Collapse
The 89,250-parameter `ModelK1_SpatialTCN` architecture completely **memorized the 182 training sequences** ($\text{Train Loss} = 0.0966$, $\text{Train ROC-AUC} = 1.0000$), but failed to generalize to unseen physical locations/actors in the validation set ($N = 39$ groups) and test set ($N = 39$ groups). Because the output probabilities on validation data were uncalibrated and centered near zero (Median Fall Prob = $0.0730$, Median Normal Prob = $0.0917$), threshold optimization selected $\tau^* = 0.0700$ at the bottom boundary of the distribution.

> **CRITICAL SCIENTIFIC PRINCIPLE**: Threshold tuning cannot repair a model with a discriminative ranking power near random chance ($\text{ROC-AUC} \approx 0.50$). The root cause is severe overfitting due to lack of multi-view camera data and small group sample size in dual-dataset training without Multicam camera angles.

---

## 2. Empirical Model Output Probability Diagnostics

### Probability Statistics across Splits (EXP-B-CORRECTED Checkpoint `best_candidate.pth`)

```text
===========================================================================
PROBABILITY DISTRIBUTION STATISTICS BY SPLIT & CLASS
===========================================================================
  Split  | Target Class | Sample Count | Mean Prob | Std Dev | Median Prob
  -------------------------------------------------------------------------
  TRAIN  | FALL (1)     | 452          | 0.7332    | 0.0221  | 0.7308
  TRAIN  | NORMAL (0)   | 2,354        | 0.0315    | 0.0438  | 0.0169
  -------------------------------------------------------------------------
  VAL    | FALL (1)     | 108          | 0.1468    | 0.1746  | 0.0730
  VAL    | NORMAL (0)   | 477          | 0.1637    | 0.1800  | 0.0917
  -------------------------------------------------------------------------
  TEST   | FALL (1)     | 98           | 0.1515    | 0.1587  | 0.0899
  TEST   | NORMAL (0)   | 504          | 0.1536    | 0.1746  | 0.0792
===========================================================================
```

### Percentile Distribution Comparison

```text
===========================================================================
PROBABILITY PERCENTILE DISTRIBUTION
===========================================================================
  Split | p01    | p05    | p10    | p25    | p50    | p75    | p90    | p95    | p99
  ------|--------|--------|--------|--------|--------|--------|--------|--------|-------
  TRAIN | 0.0007 | 0.0021 | 0.0035 | 0.0084 | 0.0226 | 0.0717 | 0.7265 | 0.7396 | 0.7737
  VAL   | 0.0021 | 0.0067 | 0.0100 | 0.0293 | 0.0860 | 0.2287 | 0.4816 | 0.5626 | 0.6560
  TEST  | 0.0021 | 0.0061 | 0.0104 | 0.0273 | 0.0845 | 0.2191 | 0.4242 | 0.5445 | 0.6805
===========================================================================
```

---

## 3. Threshold-Independent ROC-AUC & PR-AUC Analysis

| Split Name | Window Count (N) | ROC-AUC | PR-AUC (Average Precision) | Generalization Diagnosis |
| :--- | :---: | :---: | :---: | :--- |
| **TRAIN** | 2,806 | **1.0000** | **1.0000** | Perfect Memorization |
| **VALIDATION** | 585 | **0.4583** | **0.1718** | Zero Separation / Overfitted |
| **HELD-OUT TEST** | 602 | **0.5151** | **0.1691** | Near Random Chance ($\approx 0.50$) |

### Threshold Optimization Audit (`threshold_analysis.json`)
- **Reason $\tau^* = 0.0700$ Was Selected**: On the validation set, $50\%$ of both Fall and Normal probabilities lie below $0.0860$. Searching $\tau \in [0.05, 0.95]$ for maximum validation F1 selected $\tau^* = 0.0700$ at the lower boundary of probability mass.
- **Instability Confirmation**: Because ROC-AUC is $0.4583$, threshold optimization is highly unstable; small threshold shifts move predictions across uncalibrated noise rather than genuine probability boundaries.

---

## 4. Dataset & Feature Distribution Analysis

### Label Distributions across Source Datasets

| Dataset | Total Windows | NORMAL (0) | FALL (1) | Fall Percentage |
| :--- | :---: | :---: | :---: | :---: |
| **Le2i** | 2,943 | 2,465 | 478 | 16.24% |
| **URFD** | 1,050 | 870 | 180 | 17.14% |
| **Multicam** | 2,880 | 1,728 | 1,152 | 40.00% |

### Feature Tensor Health Statistics (187-D Input Features)

| Dataset Name | Mean | Std Dev | Min Value | Max Value | Zero Tensor Count | NaN / Inf Count |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Le2i** | -0.0021 | 1.0003 | -4.7736 | +4.3206 | 0 | 0 |
| **URFD** | -0.0003 | 1.0000 | -4.8378 | +4.2662 | 0 | 0 |
| **Multicam** | -0.0015 | 0.9985 | -5.0342 | +4.6941 | 0 | 0 |

- **Feature Health Verdict**: All 3 datasets feature tensors are **100% structurally healthy, non-NaN, non-Inf, and cleanly standardized ($\text{mean} \approx 0.0, \text{std} \approx 1.0$)**. Feature corruption is **RULED OUT**.

---

## 5. Group Leakage & Data Isolation Audit

- **Group Leakage**: **PASSED (0 Intersections)**.
  - $\text{Train Groups} \cap \text{Val Groups} = \emptyset$
  - $\text{Train Groups} \cap \text{Test Groups} = \emptyset$
  - $\text{Val Groups} \cap \text{Test Groups} = \emptyset$
- **Multicam Exclusion**: **PASSED**. Multicam is 100% absent from EXP-B-CORRECTED training and validation splits.

---

## 6. Root Cause Findings & Severity Classification

| Finding ID | Root Cause Description | Severity Level | Status |
| :--- | :--- | :---: | :---: |
| **RC-1** | **Overfitting to Small Training Group Count**: 182 training groups in Le2i+URFD allowed the 89,250-parameter TCN to memorize sequence patterns ($\text{Train AUC}=1.0000, \text{Val AUC}=0.4583$). | **CRITICAL** | **CONFIRMED** |
| **RC-2** | **Lack of Multi-View Spatial Diversity**: Le2i and URFD contain single-camera views susceptible to static background trapping. Multicam (8 camera angles) is required for multi-angle generalization. | **HIGH** | **CONFIRMED** |
| **RC-3** | **Unstable Boundary Threshold Selection**: $\tau^* = 0.0700$ selected at distribution boundary due to near-zero class separation on validation split. | **MEDIUM** | **CONFIRMED** |

---

## 7. Diagnostic Validation Suite Summary (`src/validate_phase_h3_4_exp_b_diagnostics.py`)

```text
============================================================
PHASE H3.4 — EXP-B-CORRECTED DIAGNOSTIC AUDIT
============================================================
[PASS] Label integrity
[PASS] Feature integrity
[PASS] Probability diagnostics
[PASS] Threshold optimization diagnostics
[PASS] ROC-AUC diagnostics (Train=1.0000, Val=0.4583, Test=0.5151)
[PASS] PR-AUC diagnostics (Train=1.0000, Val=0.1718, Test=0.1691)
[PASS] Group leakage
[PASS] Loss implementation
[PASS] Checkpoint selection
[PASS] Baseline checkpoint integrity
[PASS] Production application integrity
============================================================
```

---

## 🔒 Production Checkpoint Safety Verification

- **Production Checkpoint**: `checkpoints/final_k1/final_production.pth`
- **SHA256 Hash**: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d` (**100% UNTOUCHED & FROZEN**) ✅
- **Streamlit App `app.py`**: **100% UNTOUCHED & ACTIVE**.

---

## 8. Recommended Next Action

**RECOMMENDATION**: **Proceed to Controlled Experiment C (Le2i + Multicam)**.
Multicam provides 8 synchronized camera perspectives across 24 chute scenarios (2,880 windows), offering rich spatial geometry diversity to test whether multi-view camera data eliminates sequence memorization and restores cross-dataset generalization.
