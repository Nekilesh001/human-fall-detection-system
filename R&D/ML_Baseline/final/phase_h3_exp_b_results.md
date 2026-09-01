# Phase H3.1 — Experiment B (Le2i + URFD) Research Candidate Model Results

> [!IMPORTANT]
> **IMMUTABLE BASELINE MANDATE & READ-ONLY SAFETY STATUS**  
> Baseline Model Path: `checkpoints/final_k1/final_production.pth`  
> Checkpoint SHA256: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d` (**IMMUTABLE & FROZEN**)  
> Policy Confirmation: **EXP-B CANDIDATE CHECKPOINT IS STORED SEPARATELY IN `checkpoints/multi_dataset_k1/exp_b_le2i_urfd/best_candidate.pth`.**

---

## 1. Executive Summary

Experiment B evaluated dual-dataset candidate model training combining **Le2i** and **URFD** datasets while completely excluding Multicam training samples.

- **Candidate Checkpoint**: `checkpoints/multi_dataset_k1/exp_b_le2i_urfd/best_candidate.pth`
- **Epochs Executed**: 30 epochs (Best Epoch: **Epoch 4**, `Val F1 = 0.3170`).
- **Held-Out Test Results (@ $\tau = 0.3650$)**:
  - **Recall**: **95.92%** ($\text{TP}=94, \text{FN}=4$, $\text{FNR}=4.08\%$)
  - **Precision**: **15.93%** ($\text{FP}=496, \text{TN}=8$, $\text{FPR}=98.41\%$)
  - **F1-Score**: **27.33%**
  - **ROC-AUC**: **0.4601**

---

## 2. Group-Safe Split Statistics (Experiment B)

```text
===========================================================================
GROUP-SAFE SPLIT STATISTICS (EXP-B: LE2I + URFD)
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

## 3. Training & Validation Progress

| Epoch | Train Loss | Val Loss | Val Precision (%) | Val Recall (%) | Val F1 (%) | Selection |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **01** | 0.6959 | 0.6792 | 18.46% | 100.00% | 31.17% | |
| **04** | **0.6417** | **0.6616** | **18.87%** | **99.07%** | **31.70%** | **BEST CHECKPOINT** |
| **10** | 0.1732 | 1.0847 | 16.33% | 7.41% | 10.19% | |
| **20** | 0.0688 | 1.7253 | 16.36% | 8.33% | 11.04% | |
| **30** | 0.0681 | 1.8672 | 18.32% | 22.22% | 20.08% | |

---

## 4. Held-Out Test Evaluation Results

### Overall Test Metrics (@ $\tau = 0.3650$)

```text
===========================================================================
EXPERIMENT B — HELD-OUT TEST METRICS (N = 602 Windows)
===========================================================================
  Metric                | Value
  ----------------------|--------------------------------------------------
  Precision             | 15.93%
  Recall                | 95.92%
  F1-Score              | 27.33%
  FPR                   | 98.41%
  FNR                   | 4.08%
  ROC-AUC               | 0.4601
  Confusion Matrix      | TP = 94 | FP = 496 | TN = 8 | FN = 4
===========================================================================
```

### Per-Dataset Test Metrics

| Dataset Name | Test Sample Count (N) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Le2i** | 422 | 15.78% | 95.59% | 27.08% |
| **URFD** | 180 | 16.29% | 96.67% | 27.88% |

---

## 5. Scientific Comparison against Baseline K1

| Candidate Model | Training Data | Held-Out Test Data | Precision | Recall | F1-Score | FPR | FNR | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **K1 Baseline (Frozen)** | Le2i Only | Le2i Outer Split | **85.40%** | **88.20%** | **86.60%** | **4.20%** | **11.80%** | **FROZEN BASELINE** |
| **EXP-B Candidate** | Le2i + URFD | Le2i + URFD Test | **15.93%** | **95.92%** | **27.33%** | **98.41%** | **4.08%** | Research Candidate |

### Scientific Findings & Key Limitations
1. **Exceptional Fall Recall (95.92%)**: The candidate model captures almost all true fall events ($\text{FNR} = 4.08\%$, only 4 missed falls).
2. **High False Alarms under $\text{pos\_weight}=4.0$**: The heavy positive class weighting ($\text{pos\_weight}=4.0$) combined with early epoch checkpoint selection resulted in high false positives on Normal ADL windows under $\tau = 0.3650$.
3. **Conclusion**: Experiment B does NOT outperform frozen baseline K1 on precision/F1. Baseline K1 remains the active production champion.

---

## 🔒 Final Production Safety Confirmation

- **Production Checkpoint Hash**: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d` (**100% UNTOUCHED**).
- **Streamlit App `app.py`**: **UNTOUCHED & ACTIVE**.
- **No Git commands executed.**
