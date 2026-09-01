# Phase H3.2 — Multi-Dataset Candidate Model Training & Evaluation Methodology Design Report

> [!IMPORTANT]
> **IMMUTABLE BASELINE MANDATE & READ-ONLY SAFETY STATUS**  
> Baseline Model Path: `checkpoints/final_k1/final_production.pth`  
> Checkpoint SHA256: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d` (**IMMUTABLE & FROZEN**)  
> Policy Statement: **READ-ONLY DESIGN & METHODOLOGY PHASE. NO MODEL TRAINING WAS EXECUTED.**

---

## 1. Current Experiment B Methodology & Confirmed Weaknesses

### Current Implementation Overview
- **Training Script**: `src/train_multi_dataset_k1.py`
- **Dataset**: Le2i + URFD (2,806 train windows, 585 val windows, 602 test windows)
- **Loss Function**: `nn.CrossEntropyLoss(weight=[1.0, 4.0])`
- **Checkpoint Selection**: `if val_f1 > best_val_f1:` at fixed $\tau = 0.3650$ without warmup.
- **Evaluation Script**: `src/evaluate_multi_dataset_k1.py` at fixed production threshold $\tau = 0.3650$.

### Confirmed Weaknesses

| Weakness Area | Code Behavior | Empirical Impact | Status |
| :--- | :--- | :--- | :---: |
| **Early Stopping Vulnerability** | Selected Epoch 4 (`Val F1 = 31.70%`) before convergence | Model selected while probabilities compressed in $[0.34, 0.48]$ | **CONFIRMED FROM CODE** |
| **Trivial Positive Bias** | Raw F1 at $\tau = 0.3650$ favored early "all-positive" predictions | $98.41\%$ False Positive Rate on test set | **CONFIRMED FROM CODE** |
| **Fixed Threshold Coupling** | Checked validation F1 at fixed $\tau = 0.3650$ instead of optimizing $\tau^*$ | Fixed production threshold applied to uncalibrated candidate logits | **CONFIRMED FROM CODE** |
| **Aggressive Loss Weighting** | `pos_weight = 4.0` in early epochs pushed logits positive | Logits for Class 1 shifted positive ($\sim 0.41$) early in training | **CONFIRMED FROM CODE** |

---

## 2. Code Inspection & Verification Results

### A. Training Pipeline (`src/train_multi_dataset_k1.py`)
- **Group Splitting**: **CONFIRMED**. Splits cleanly on 284 physical `group_id` units (70% Train, 15% Val, 15% Test). Zero group leakage.
- **Validation Threshold**: **CONFIRMED**. Fixed at $\tau = 0.3650$. No threshold optimization was performed during training.
- **Warmup Epochs**: **CONFIRMED**. None. Epoch 1–4 eligible for best checkpoint selection.
- **Validation ROC-AUC**: **CONFIRMED**. Not tracked or used for checkpoint selection during training.

### B. Evaluation Pipeline (`src/evaluate_multi_dataset_k1.py`)
- **ROC-AUC Calculation**: **CONFIRMED FROM CODE**. Calculated directly from **continuous Softmax probabilities** `probs = torch.softmax(out, dim=1)[:, 1]`, NOT thresholded predictions.
- **Metrics Calculation**: **CONFIRMED**. Precision, Recall, F1, FPR, FNR, and Confusion Matrix calculated cleanly from binary predictions `preds = (probs >= tau)`.

---

## 3. Candidate Threshold Policy & Scientific Leakage Prevention

```text
===========================================================================
LEAKAGE-FREE CANDIDATE MODEL EVALUATION PIPELINE
===========================================================================
  TRAIN SPLIT (70%)
        ↓
  VALIDATION SPLIT (15%)
    1. Select Best Model Checkpoint (using Val Loss / Val ROC-AUC @ Epoch >= 10)
    2. Tune & Freeze Candidate Operating Threshold tau* on Val Probabilities
        ↓
  HELD-OUT TEST SPLIT (15%)
    3. Evaluate Frozen Candidate (Checkpoint + tau*) ONCE on Test Set
===========================================================================
```

> [!NOTE]
> **Strict Separation Rule**: The production threshold $\tau = 0.3650$ belongs exclusively to the frozen K1 production model. Each candidate model MUST have its own operating threshold $\tau^*$ selected on validation data before test set evaluation.

---

## 4. Checkpoint Selection & `pos_weight` Analysis

### A. Recommended Checkpoint Selection Policy
We evaluated 5 design options for checkpoint selection:
- **Option A**: Minimum warmup epoch ($\text{epoch} \ge 10$).
- **Option B**: Minimum Validation Loss (`val_loss`).
- **Option C**: Maximum Validation ROC-AUC (`val_auc`).
- **Option D**: Validation F1 at optimized threshold ($\text{F1}^*$).
- **Option E**: Balanced Objective score.

**Recommended Policy**: **Validation Loss + Warmup ($\text{epoch} \ge 10$)**
```python
if epoch >= 10 and val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save(model.state_dict(), best_checkpoint_path)
```
*Rationale*: Validation loss measures probability calibration and generalization error without biasing predictions toward trivial positive collapses early in training.

### B. `pos_weight` Class Weighting Analysis
- Total Train Windows (EXP-B): 2,806 windows
- NORMAL (0): 2,354 ($83.89\%$) | FALL (1): 452 ($16.11\%$)
- Natural Class Imbalance Ratio: $\frac{2354}{452} = 5.21 : 1$

While `pos_weight = 4.0` reflects the natural imbalance ratio, when combined with early uncalibrated epochs, it pushes output logits positive.
**Recommended Loss Weight**: Standard unweighted BCE (`pos_weight = 1.0`) or moderate weight (`pos_weight = 2.0`), relying on validation threshold tuning $\tau^*$ to handle class imbalance.

---

## 5. Controlled Configuration for Next Experiment (Experiment B Corrected)

```text
===========================================================================
EXPERIMENT B (CORRECTED) METHODOLOGY SPECIFICATION
===========================================================================
  Parameter                 | Value / Setting
  --------------------------|----------------------------------------------
  Seed                      | 42 (Identical)
  Group Split               | 284 Group IDs (Identical 70/15/15 split)
  Features & Input          | (50, 187) float32 (Identical)
  Architecture              | ModelK1_SpatialTCN (Identical)
  Datasets Used             | Le2i + URFD (Identical)
  Loss Weight (pos_weight)  | 1.0 (Unweighted BCE) or 2.0
  Checkpoint Selection      | Minimum Validation Loss after Epoch >= 10
  Threshold Selection       | Tuned tau* on Validation Split to maximize F1
  Test Set Evaluation       | Single Unbiased Pass using Checkpoint + tau*
===========================================================================
```

---

## 🔒 Production Checkpoint Safety Verification

- **Production Checkpoint**: `checkpoints/final_k1/final_production.pth`
- **SHA256 Hash**: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d` (**100% UNTOUCHED & FROZEN**) ✅
- **Streamlit App `app.py`**: **UNTOUCHED & ACTIVE**.
- **Git State**: Zero Git write operations executed.

---

## 6. Artifacts Created

1. [`R&D/ML_Baseline/final/phase_h3_2_candidate_training_methodology.md`](file:///d:/ONE_DATA/Fall%20detection/R&D/ML_Baseline/final/phase_h3_2_candidate_training_methodology.md) — Comprehensive Candidate Training & Evaluation Methodology Design Report.
