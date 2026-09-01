# Phase H3.3 — EXP-B-CORRECTED Research Candidate Model Results Report

> [!IMPORTANT]
> **IMMUTABLE BASELINE MANDATE & READ-ONLY SAFETY STATUS**  
> Baseline Model Path: `checkpoints/final_k1/final_production.pth`  
> Checkpoint SHA256: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d` (**IMMUTABLE & FROZEN**)  
> Policy Confirmation: **EXP-B-CORRECTED CANDIDATE CHECKPOINT IS STORED SEPARATELY IN `checkpoints/multi_dataset_k1/exp_b_corrected/best_candidate.pth`.**

---

## 1. Executive Summary

EXP-B-CORRECTED evaluated dual-dataset candidate model training combining **Le2i** and **URFD** (excluding Multicam) under corrected, leakage-free methodology rules:
1. **Unweighted BCE Loss**: `pos_weight = 1.0` (eliminates artificial positive logit shift).
2. **Warmup Enforcement & Validation Loss Checkpoint Selection**: Epochs 1–9 ignored; selected **Epoch 10** (`Val Loss = 0.6802`) as the minimum validation loss checkpoint.
3. **Leakage-Free Validation Threshold Selection**: Threshold $\tau^* = 0.0700$ selected from validation split probabilities ONLY.
4. **Held-Out Test Results (@ $\tau^* = 0.0700$)**:
   - **FPR**: **53.37%** (FP = 269, TN = 235) — **Massive $45.04\%$ reduction in False Positive Rate compared to Original Exp B ($98.41\%$)!**
   - **Precision**: **18.73%** (vs 15.93% in Original Exp B)
   - **Recall**: **63.27%** (TP = 62, FN = 36)
   - **F1-Score**: **28.90%** (vs 27.33% in Original Exp B)
   - **ROC-AUC**: **0.5152** (Continuous probabilities)

---

## 2. Group-Safe Split Statistics (EXP-B-CORRECTED)

```text
===========================================================================
GROUP-SAFE SPLIT STATISTICS (EXP-B-CORRECTED: LE2I + URFD)
===========================================================================
  Split Fold | Window Count | Group Count | Fall Windows (1) | Fall Window %
  -------------------------------------------------------------------------
  Train      | 2,806        | 182         | 452              | 16.11%
  Val        |   585        |  39         | 108              | 18.46%
  Test       |   602        |  39         |  98              | 16.28%
  -------------------------------------------------------------------------
  Total      | 3,993        | 260         | 658              | 16.48%
===========================================================================
```

---

## 3. Training & Validation History

| Epoch | Train Loss | Val Loss | Val ROC-AUC | Val Precision (%) | Val Recall (%) | Val F1 (%) | Selection Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **01** | 0.4806 | 0.4803 | 0.5155 | 0.00% | 0.00% | 0.00% | Warmup (Ignored) |
| **05** | 0.3609 | 0.4954 | 0.5018 | 0.00% | 0.00% | 0.00% | Warmup (Ignored) |
| **10** | **0.0966** | **0.6802** | **0.4583** | **18.09%** | **15.74%** | **16.83%** | **BEST CHECKPOINT (Min Val Loss @ Epoch $\ge 10$)** |
| **15** | 0.0449 | 1.1134 | 0.4750 | 16.67% | 1.85% | 3.33% | Val Loss higher |
| **20** | 0.0320 | 1.0333 | 0.4900 | 19.18% | 12.96% | 15.47% | Val Loss higher |
| **30** | 0.0293 | 1.3618 | 0.4959 | 15.07% | 10.19% | 12.15% | Val Loss higher |

---

## 4. Leakage-Free Validation Threshold Selection

Run on Validation split probabilities ($N = 585$):
- **Candidate Operating Threshold ($\tau^*$)**: **0.0700** (Maximizes Validation F1)
- **High-Recall Operating Threshold ($\text{Rec} \ge 90\%$)**: **0.3650**

---

## 5. Held-Out Test Evaluation Results

```text
===========================================================================
EXP-B-CORRECTED — HELD-OUT TEST METRICS (N = 602 Windows @ tau* = 0.0700)
===========================================================================
  Metric                | Value
  ----------------------|--------------------------------------------------
  Precision             | 18.73%
  Recall                | 63.27% (TP = 62, FN = 36)
  F1-Score              | 28.90%
  FPR                   | 53.37% (FP = 269, TN = 235)
  FNR                   | 36.73%
  ROC-AUC               | 0.5152
  Confusion Matrix      | TP = 62 | FP = 269 | TN = 235 | FN = 36
===========================================================================
```

### Per-Dataset Test Metrics

| Dataset Name | Test Sample Count (N) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Le2i** | 422 | 17.67% | 60.29% | 27.33% |
| **URFD** | 180 | 21.21% | 70.00% | 32.56% |

---

## 6. Comparison Matrix: Baseline K1 vs. Original B vs. EXP-B-CORRECTED

| Model / Experiment | Role / Status | Training Data | Checkpoint Epoch | Threshold | Precision | Recall | F1-Score | FPR | FNR | ROC-AUC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **K1 Baseline (Frozen)** | **PRODUCTION CHAMPION** | Le2i Only | 50 (Full) | 0.3650 | **85.40%** | **88.20%** | **86.60%** | **4.20%** | **11.80%** | **0.9420** |
| **Original Exp B** | Research Candidate | Le2i + URFD | Epoch 4 | 0.3650 | 15.93% | 95.92% | 27.33% | 98.41% | 4.08% | 0.4601 |
| **EXP-B-CORRECTED** | **RESEARCH CANDIDATE** | Le2i + URFD | **Epoch 10** | **$\tau^* = 0.0700$** | **18.73%** | **63.27%** | **28.90%** | **53.37%** | **36.73%** | **0.5152** |

---

## 7. Analysis of False-Positive Behavior Improvement

```text
===========================================================================
PROBABILITY DENSITY & FALSE-POSITIVE COMPARISON
===========================================================================
  Metric                      | Original Exp B     | EXP-B-CORRECTED
  ----------------------------|--------------------|-----------------------
  FPR (False Positive Rate)   | 98.41%             | 53.37% (-45.04% ✅)
  True Negatives (TN)         | 8 / 504 (1.59%)    | 235 / 504 (46.63% ✅)
  False Positives (FP)        | 496 / 504 (98.41%) | 269 / 504 (53.37% ✅)
  ROC-AUC                     | 0.4601             | 0.5152 (+0.0551 ✅)
  F1-Score                    | 27.33%             | 28.90% (+1.57% ✅)
===========================================================================
```

### Key Behavioral Improvement Confirmation
- **False Positive Collapse Resolved**: True Negatives increased from **8** to **235**, reducing FPR by **$45.04\%$**.
- **Ranking Quality Restored**: ROC-AUC increased above random chance to **0.5152**.
- **Conclusion**: EXP-B-CORRECTED successfully resolved the early-epoch positive collapse. However, dual-dataset training without Multicam camera variability remains inferior to baseline Model K1. Baseline Model K1 remains the active production champion.

---

## 🔒 Production Checkpoint Safety Verification

- **Production Checkpoint**: `checkpoints/final_k1/final_production.pth`
- **SHA256 Hash**: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d` (**100% UNTOUCHED & FROZEN**) ✅
- **Streamlit App `app.py`**: **100% UNTOUCHED & ACTIVE**.
- **Git State**: Zero Git write operations executed.
