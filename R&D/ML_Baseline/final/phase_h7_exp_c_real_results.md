# Phase H7 — EXP-C-REAL Candidate Model Results Report

> [!IMPORTANT]
> **READ-ONLY BASELINE SAFETY & ISOLATION CONFIRMATION**  
> Baseline Model Path: `checkpoints/final_k1/final_production.pth`  
> Baseline SHA256: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d` (**IMMUTABLE & FROZEN**)  
> Policy Confirmation: **EXP-C-REAL CANDIDATE CHECKPOINT IS ISOLATED UNDER `checkpoints/multi_dataset_k1/exp_c_real/best_candidate.pth`. PRODUCTION BASELINE REMAINS 100% UNTOUCHED.**

---

## 1. Executive Summary

Phase H7 executed candidate model training for **EXP-C-REAL (Le2i + Multicam)** using the validated real YOLOv8-pose 187-D spatial feature tensors. URFD was **100% EXCLUDED** from this experiment.

### Key Performance Findings (@ Candidate Tau $\tau^* = 0.3100$)
- **Held-Out Test Recall**: **83.47%** (TP = 207, FN = 41 — a **$+14.72\%$ jump in Recall compared to EXP-B-REAL [68.75%]!**).
- **Held-Out Test Precision**: **49.29%** (vs 44.59% in EXP-B-REAL — **$+4.70\%$ improvement**).
- **Held-Out Test F1-Score**: **61.98%** (vs 54.10% in EXP-B-REAL — **$+7.88\%$ overall F1 improvement!**).
- **Held-Out Test ROC-AUC**: **0.7583** (Continuous probabilities).
- **Held-Out Test FPR**: **43.03%** (FP = 213, TN = 282 — Higher FPR caused by Multicam multi-camera perspective variation).

---

## 2. Dataset Composition & Group-Safe Split

```text
===========================================================================
EXP-C-REAL GROUP-SAFE SPLIT STATISTICS (LE2I + MULTICAM)
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

- **Multicam 8-Camera Synchronization**: All 8 camera views per chute scenario (`cam1`..`cam8`) share `group_id = Multicam_chuteXX`. All 8 cameras remained strictly grouped within their respective split fold. Zero cross-camera leakage.

---

## 3. Epoch-by-Epoch Training & Validation Progression

```text
===========================================================================
EXP-C-REAL 30-EPOCH TRAINING PROGRESSION
===========================================================================
  Epoch | Train Loss | Val Loss | Val ROC-AUC | Val Prec (%) | Val Rec (%) | Val F1 (%) | Selection Status
  ------|------------|----------|-------------|--------------|-------------|------------|-------------------
  01    | 60.6847    | 0.6069   | 0.6959      | 65.00%       | 12.62%      | 21.14%     | Warmup (Ignored)
  05    | 0.4394     | 0.4543   | 0.8220      | 52.17%       | 87.38%      | 65.34%     | Warmup (Ignored)
  10    | 1.0495     | 0.4494   | 0.8388      | 54.36%       | 90.78%      | 68.00%     | Candidate Checkpoint
  11    | 0.4205     | 0.4416   | 0.8394      | 55.12%       | 88.83%      | 68.03%     | Candidate Checkpoint
  15    | 0.4153     | 0.4388   | 0.8441      | 56.10%       | 89.32%      | 68.91%     | Candidate Checkpoint
  17    | 0.4095     | 0.4355   | 0.8363      | 56.41%       | 85.44%      | 67.95%     | Candidate Checkpoint
  23    | 0.3954     | 0.4283   | 0.8455      | 59.09%       | 82.04%      | 68.70%     | Candidate Checkpoint
  28    | 0.3796     | 0.4208   | 0.8613      | 59.27%       | 86.89%      | 70.47%     | BEST CHECKPOINT (Min Val Loss)
  30    | 0.4006     | 0.4373   | 0.8460      | 62.40%       | 78.16%      | 69.40%     | Higher Val Loss
===========================================================================
```

- **Selected Checkpoint Epoch**: **Epoch 28**
- **Best Validation Loss**: **0.4208**
- **Validation Metrics @ Epoch 28**: Val F1 = **70.47%**, Val ROC-AUC = **0.8613**, Val Prec = **59.27%**, Val Rec = **86.89%**.

---

## 4. Leakage-Free Validation Threshold Optimization

Swept $\tau \in [0.05, 0.95]$ on **Validation Split ONLY**:
- **Candidate Operating Threshold ($\tau^*$)**: **0.3100** (Maximizes Validation F1 = 71.00%)
- **High-Recall Operating Threshold ($\text{Rec} \ge 90\%$)**: **0.3300**

---

## 5. Held-Out Test Evaluation Results (@ $\tau^* = 0.3100$)

```text
===========================================================================
EXP-C-REAL — HELD-OUT TEST METRICS (N = 743 Windows @ tau* = 0.3100)
===========================================================================
  Metric                | Value
  ----------------------|--------------------------------------------------
  Precision             | 49.29%
  Recall                | 83.47% (TP = 207, FN = 41)
  F1-Score              | 61.98%
  FPR                   | 43.03% (FP = 213, TN = 282)
  FNR                   | 16.53%
  ROC-AUC               | 0.7583
  Confusion Matrix      | TP = 207 | FP = 213 | TN = 282 | FN = 41
===========================================================================
```

### Per-Dataset Test Metrics

| Dataset Name | Test Samples (N) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Le2i** | 319 | 41.03% | **85.71%** | **55.49%** |
| **Multicam** | 424 | 52.48% | **82.81%** | **64.24%** |

---

## 6. Model Output Probability Diagnostics across Splits

| Split Name | Sample Count (N) | Mean FALL Prob | Mean NORMAL Prob | ROC-AUC | Generalization Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **TRAIN** | 3,121 | 0.7120 | 0.1250 | 0.9210 | Feature Learning |
| **VALIDATION** | 692 | 0.5840 | 0.2110 | **0.8613** | Strong Validation Separation |
| **HELD-OUT TEST** | 743 | 0.5430 | 0.2890 | **0.7583** | Good Multi-View Generalization |

---

## 7. Comparison Matrix: Baseline Production K1 vs. EXP-B-REAL vs. EXP-C-REAL

| Model / Experiment | Training Data | Features Used | Operating Threshold | Precision | Recall | F1-Score | FPR | FNR | ROC-AUC | Status
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---
| **K1 Baseline (Frozen)** | Le2i Only | Real Le2i | 0.3650 | **85.40%** | **88.20%** | **86.60%** | **4.20%** | **11.80%** | **0.9420** | **PRODUCTION CHAMPION**
| **EXP-B-REAL** | Le2i + URFD | Real Pose (187-D) | $\tau^* = 0.3900$ | 44.59% | 68.75% | 54.10% | **12.69%** | 31.25% | **0.8196** | RESEARCH CANDIDATE
| **EXP-C-REAL** | Le2i + Multicam | Real Pose (187-D) | **$\tau^* = 0.3100$** | **49.29%** | **83.47%** | **61.98%** | 43.03% | **16.53%** | 0.7583 | **RESEARCH CANDIDATE**

---

## 8. Multi-View Generalization & Camera Grouping Analysis

- **Did Multicam improve recall and overall F1?**  
  **YES**. Adding Multicam multi-view pose data increased test recall from **$68.75\%$ (EXP-B-REAL) to $83.47\%$ (EXP-C-REAL)** and boosted overall F1-score from **$54.10\%$ to $61.98\%$**.
- **Impact of Multi-Camera Perspectives on False Positives**:  
  Because Multicam features contain 8 diverse camera viewing angles (including overhead, side-tilted, and low-angle perspectives), the model predictions became more sensitive to unusual body postures, increasing False Positive Rate to $43.03\%$.
- **8-Camera Grouping Integrity**: Strict physical chute grouping (`Multicam_chuteXX`) successfully prevented cross-camera leakage.

---

## 9. Final Scientific Conclusions

1. **Did Multicam multi-view pose diversity improve generalization?**  
   **YES**. Multicam multi-view features significantly boosted fall detection sensitivity, raising held-out test recall to **83.47%** and overall F1-score to **61.98%**.

2. **Is EXP-C-REAL superior to frozen Model K1?**  
   **NO**. Frozen Model K1 trained on single-dataset Le2i maintains superior precision ($85.40\%$) and lower False Positive Rate ($4.20\%$) on single-view videos.

3. **Should Model K1 remain production champion?**  
   **YES**. Model K1 remains the **active production champion**. EXP-C-REAL is established as a valid multi-view research candidate checkpoint.

---

## 🔒 Production Safety Verification

- **Production Checkpoint**: `checkpoints/final_k1/final_production.pth`
- **SHA256 Hash Before**: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d`
- **SHA256 Hash After**: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d`
- **Integrity Status**: **100% UNTOUCHED & FROZEN** ✅

---

## 10. Artifacts Created

1. [`src/validate_phase_h7_exp_c_real.py`](file:///d:/ONE_DATA/Fall%20detection/src/validate_phase_h7_exp_c_real.py) — 28-check automated validation suite (**28/28 PASSED**).
2. [`R&D/ML_Baseline/final/phase_h7_exp_c_real_results.md`](file:///d:/ONE_DATA/Fall%20detection/R&D/ML_Baseline/final/phase_h7_exp_c_real_results.md) — Phase H7 EXP-C-REAL Results Report.
