# Final K1 Evaluation Report

**Model**: K1 — YOLO Pose + 187-D Spatial Features + 1D Residual TCN
**Phase**: F1 Final Training & Evaluation
**Date**: 2026-08-28
**Protocol**: 4-Fold Leave-One-Location-Out (LOLO), leakage-free

---

## 1. LOLO Benchmark Comparison

| Metric | Reference (Exp 19) | This Run (Option A) | Delta |
| :--- | :---: | :---: | :---: |
| LOLO Mean F1 (tau*) | 86.65% | 84.98% | -1.67% |
| Cross-Loc Variance  | ±5.64% | ±5.81% | — |
| Mean tau*_inner     | 0.4923 | 0.3650 | — |
| Benchmark Status    | — | **DIVERGED (investigate)** | — |

---

## 2. Per-Fold Results

| Fold | Test Location | tau* | F1 @ 0.50 | F1 @ tau* | Recall | Specificity | TP | FP | TN | FN |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | Coffee_room_01 | 0.3400 | 0.9188 | 0.9222 | 0.9651 | 0.9333 | 166 | 22 | 308 | 6 |
| 2 | Coffee_room_02 | 0.2000 | 0.9020 | 0.8868 | 1.0000 | 0.9669 | 47 | 12 | 351 | 0 |
| 3 | Home_01 | 0.4800 | 0.7739 | 0.7739 | 0.8556 | 0.7852 | 77 | 32 | 117 | 13 |
| 4 | Home_02 | 0.4400 | 0.8696 | 0.8163 | 0.9091 | 0.9686 | 20 | 7 | 216 | 2 |
| **All** | **Aggregated** | 0.3650 | — | **84.98%** | — | — | **310** | **73** | **992** | **21** |

---

## 3. Aggregated Confusion Matrix

| | Predicted FALL | Predicted NORMAL |
| :--- | :---: | :---: |
| **Actual FALL**   | TP = 310 | FN = 21 |
| **Actual NORMAL** | FP = 73 | TN = 992 |

---

## 4. Artifacts

- Per-window predictions: `d:\ONE_DATA\Fall detection\R&D\ML_Baseline\results\final_k1/final_test_predictions.csv`
- Metrics JSON: `d:\ONE_DATA\Fall detection\R&D\ML_Baseline\results\final_k1/final_test_metrics.json`
- Confusion matrix: `d:\ONE_DATA\Fall detection\R&D\ML_Baseline\results\final_k1/final_confusion_matrix.json`
- Threshold file: `d:\ONE_DATA\Fall detection\R&D\ML_Baseline\results\final_k1/final_threshold.json`

---

## 5. Leakage-Free Certification

- Outer-test locations were excluded from all training operations
- Threshold tau* selected from inner-validation predictions only
- No outer-test labels accessed during model selection or threshold tuning
- Inner split is event-grouped (no window-level cross-contamination)
- Fixed random seeds ensure reproducibility
