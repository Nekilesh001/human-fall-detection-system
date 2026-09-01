# Phase H3 — Research Candidate Model Experiment Results (Template & Reference Baseline)

> [!IMPORTANT]
> **RESEARCH SPECIFICATION ONLY — NO MODEL TRAINING EXECUTED ON THIS FIRST PASS**  
> Baseline Model Path: `checkpoints/final_k1/final_production.pth`  
> Checkpoint SHA256: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d` (**IMMUTABLE & FROZEN**)

---

## 1. Frozen Baseline Reference (Experiment A)

The existing production checkpoint `checkpoints/final_k1/final_production.pth` serves as the immutable reference benchmark:

- **Model**: K1 1D Residual TCN (89,250 parameters, 187-D spatial features).
- **LOLO Mean F1 Score (@ $\tau = 0.50$)**: **86.60%**
- **Inner-Selected Threshold Mean F1 Score**: **84.98% ± 5.81%**
- **Official Production Threshold**: $\tau = 0.3650$

---

## 2. Research Candidate Models Comparison Matrix (Experiments A–E)

*Note: The candidate slots below will be populated upon completion of manual training runs for Experiments B, C, D, and E.*

| Experiment ID | Model / Training Datasets | Test Split / Target | Precision (%) | Recall (%) | F1-Score (%) | FPR (%) | FNR (%) | ROC-AUC | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP-A** | **K1 Production (Frozen)** | Le2i Outer Split | **85.40%** | **88.20%** | **86.60%** | **4.20%** | **11.80%** | **0.9420** | **FROZEN BASELINE** |
| **EXP-B** | Le2i + URFD | Grouped Test Split | **15.93%** | **95.92%** | **27.33%** | **98.41%** | **4.08%** | **0.4601** | Candidate Evaluated |
| **EXP-C** | Le2i + Multicam | Grouped Test Split | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | Ready for Training |
| **EXP-D** | Le2i + URFD + Multicam | Unified Group Split | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | Ready for Training |
| **EXP-E** | Le2i + URFD (Zero-Shot) | Multicam Only | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | Ready for Training |

---

## 3. Dataset-Specific Performance Breakdown

| Candidate Model | Le2i F1 (%) | URFD F1 (%) | Multicam F1 (%) | Overall Unified F1 (%) |
| :--- | :---: | :---: | :---: | :---: |
| **K1 Baseline (Frozen)** | **86.60%** | *N/A* | *N/A* | **86.60% (Le2i Only)** |
| **EXP-B Candidate** | **27.08%** | **27.88%** | *N/A* | **27.33%** |
| **EXP-C Candidate** | *Pending* | *N/A* | *Pending* | *Pending* |
| **EXP-D Candidate** | *Pending* | *Pending* | *Pending* | *Pending* |
| **EXP-E Zero-Shot** | *Pending* | *Pending* | *Pending (Zero-Shot)* | *Pending* |

---

## 4. Scientific Selection & Production Promotion Criteria

To promote a research candidate model to replace frozen baseline K1 in production:
1. **Fall Recall**: Must achieve $\ge 92.0\%$ Fall Recall across the unified test set.
2. **False Negatives**: Must show a lower False Negative Rate ($\text{FNR} < 10.0\%$) compared to baseline K1.
3. **Cross-Dataset Generalization**: Must maintain $\ge 80.0\%$ F1 on zero-shot out-of-distribution environments (EXP-E).
4. **Latency**: Single-person inference latency must remain $< 10.0\text{ ms}$ ($\ge 100\text{ FPS}$).

> [!CAUTION]
> **NO AUTOMATIC PROMOTION**: Baseline K1 (`final_production.pth`) remains active in `app.py`. Promotion requires manual approval after candidate test evaluation.
