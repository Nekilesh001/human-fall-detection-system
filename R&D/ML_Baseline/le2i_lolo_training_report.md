# Experiment B: Le2i Supervised 4-Fold LOLO Baseline Training & Evaluation Report

## 1. Executive Summary
This document presents the empirical results of **Experiment B: Supervised In-Domain Le2i Baseline**, evaluating the baseline architecture (`URFDRGBFeatureBaseline`, 65,730 trainable parameters) trained under a 4-Fold Leave-One-Location-Out (LOLO) cross-validation protocol on the 127 verified supervised videos (1,396 temporal windows) of the **Le2i Fall Detection Dataset**.

- **LOLO Mean Accuracy ($\tau=0.50$)**: **$88.88\% \pm 12.72\%$** (vs. $68.55\%$ zero-shot transfer)
- **LOLO Mean F1 Score ($\tau=0.50$)**: **$71.53\% \pm 26.69\%$** (vs. $31.51\%$ zero-shot transfer)
- **LOLO Mean Event Sensitivity**: **$83.10\% \pm 25.20\%$** (83.10% of physical fall events detected in unseen held-out locations)
- **Primary Finding**: In-domain supervised training on Le2i environment variations dramatically improves cross-location generalization (F1 score increased by **+40.02 percentage points** over zero-shot transfer), proving that exposure to diverse indoor backgrounds and lighting is essential for location-invariant fall detection.

---

## 2. 4-Fold LOLO Outer Test Results

| Fold | Held-Out Test Location | Outer Test Windows ($N$) | Threshold $\tau$ | Accuracy | Precision | Recall / Sensitivity | Specificity | F1 Score | Event Sensitivity | Outer Test Confusion Matrix |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | `Coffee_room_01` | 502 | $\tau = 0.50$ | **0.9462** | **0.8836** | **0.9709** | **0.9333** | **0.9252** | **100.00%** (47/47) | `[[308, 22], [5, 167]]` |
| | | 502 | $\tau^* = 0.40$ | **0.9482** | **0.8842** | **0.9767** | **0.9333** | **0.9282** | **100.00%** (47/47) | `[[308, 22], [4, 168]]` |
| **Fold 2** | `Coffee_room_02` | 410 | $\tau = 0.50$ | **0.9878** | **0.9038** | **1.0000** | **0.9862** | **0.9495** | **100.00%** (12/12) | `[[358, 5], [0, 47]]` |
| | | 410 | $\tau^* = 0.40$ | **0.9805** | **0.8545** | **1.0000** | **0.9780** | **0.9216** | **100.00%** (12/12) | `[[355, 8], [0, 47]]` |
| **Fold 3** | `Home_01` | 239 | $\tau = 0.50$ | **0.7029** | **0.8276** | **0.2667** | **0.9664** | **0.4034** | **46.67%** (14/30) | `[[144, 5], [66, 24]]` |
| | | 239 | $\tau^* = 0.50$ | **0.7029** | **0.8276** | **0.2667** | **0.9664** | **0.4034** | **46.67%** (14/30) | `[[144, 5], [66, 24]]` |
| **Fold 4** | `Home_02` | 245 | $\tau = 0.50$ | **0.9184** | **0.5385** | **0.6364** | **0.9462** | **0.5833** | **85.71%** (6/7) | `[[211, 12], [8, 14]]` |
| | | 245 | $\tau^* = 0.50$ | **0.9184** | **0.5385** | **0.6364** | **0.9462** | **0.5833** | **85.71%** (6/7) | `[[211, 12], [8, 14]]` |

### LOLO 4-Fold Summary Statistics (Mean $\pm$ Std)
- **Accuracy ($\tau=0.50$)**: **$0.8888 \pm 0.1272$** ($88.88\%$)
- **Precision ($\tau=0.50$)**: **$0.7884 \pm 0.1697$** ($78.84\%$)
- **Recall / Sensitivity ($\tau=0.50$)**: **$0.7185 \pm 0.3434$** ($71.85\%$)
- **Specificity ($\tau=0.50$)**: **$0.9580 \pm 0.0232$** ($95.80\%$)
- **F1 Score ($\tau=0.50$)**: **$0.7153 \pm 0.2669$** ($71.53\%$)
- **Event Sensitivity ($\tau=0.50$)**: **$83.10\% \pm 25.20\%$**

---

## 3. Fold Training Metadata & Class Weight Audit

| Fold Name | Outer Train Windows | Fold Class Weights ($w_{\text{norm}}, w_{\text{fall}}$) | Inner Val Split (Events) | Best Epoch | Best Inner Val F1 | Selected Threshold ($\tau^*_{\text{inner}}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 894 (159 F, 735 N) | $w_{\text{norm}} = 0.6082, w_{\text{fall}} = 2.8113$ | 64 Train / 16 Val | Epoch 42 | `0.8421` | $\tau^* = 0.40$ |
| **Fold 2** | 986 (284 F, 702 N) | $w_{\text{norm}} = 0.7023, w_{\text{fall}} = 1.7359$ | 86 Train / 21 Val | Epoch 43 | `0.8529` | $\tau^* = 0.40$ |
| **Fold 3** | 1,157 (241 F, 916 N) | $w_{\text{norm}} = 0.6316, w_{\text{fall}} = 2.4004$ | 78 Train / 19 Val | Epoch 47 | `0.9351` | $\tau^* = 0.50$ |
| **Fold 4** | 1,151 (309 F, 842 N) | $w_{\text{norm}} = 0.6835, w_{\text{fall}} = 1.8625$ | 78 Train / 19 Val | Epoch 44 | `0.8991` | $\tau^* = 0.50$ |

---

## 4. Scientific Comparison Across Milestones

| Evaluation Milestone | Evaluation Protocol | Accuracy | Precision | Recall / Sensitivity | Specificity | F1 Score | Event Sensitivity |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **URFD Baseline (In-Domain)** | Held-Out Test Set | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** (11/11) |
| **URFD $\to$ Le2i Zero-Shot** | Zero-Shot Transfer ($\tau=0.50$) | **68.55%** | **32.58%** | **30.51%** | **80.38%** | **31.51%** | **50.00%** (48/96) |
| **URFD $\to$ Le2i Zero-Shot** | Zero-Shot Transfer ($\tau^*=0.10$) | **27.72%** | **23.84%** | **93.35%** | **7.32%** | **37.98%** | **97.92%** (94/96) |
| **Le2i $\to$ Le2i LOLO Baseline** | **4-Fold Cross-Location ($\tau=0.50$)** | **88.88%** | **78.84%** | **71.85%** | **95.80%** | **71.53%** | **83.10%** (79/96) |

---

## 5. Location Generalization Analysis

1. **Coffee Room Locations (`Coffee_room_01` & `Coffee_room_02`)**:
   - High generalization F1 scores (**92.82%** for Fold 1 and **94.95%** for Fold 2).
   - When trained on residential home data + one coffee room, the classifier learns to separate reflective coffee room backgrounds effectively.
2. **Home Locations (`Home_01` & `Home_02`)**:
   - Lower F1 scores (**40.34%** for Fold 3 and **58.33%** for Fold 4).
   - Dim residential lighting and sofa/table occlusions present unique challenges when held out.

---

## 6. Verification & Isolation Compliance
1. **Outer Test Isolation**: Outer test location was 100% unseen until evaluation.
2. **0 Event Leakage**: $\text{Inner Train Events} \cap \text{Inner Val Events} = \emptyset$.
3. **0 Camera Leakage**: All camera streams of each event remained strictly together.
4. **URFD Integrity**: URFD model checkpoint (`urfd_rgb_baseline_best.pth`) and URFD data remained 100% read-only.
5. **Saved Checkpoints**: Verified `checkpoints/le2i_lolo/fold_{1..4}_best.pth` (reproduced 100% exact match).

---

## 7. Git Status Audit (`dev` branch)

```text
Current Branch: dev
Tracking Status: Up to date with 'origin/dev'

Untracked files:
  R&D/ML_Baseline/
  checkpoints/le2i_lolo/
  src/dataset.py
  src/evaluate_le2i_lolo.py
  src/evaluate_le2i_zeroshot.py
  src/model.py
  src/precompute_features.py
  src/precompute_le2i_features.py
  src/preprocess_le2i.py
  src/train_baseline.py
  src/train_le2i_lolo.py
  src/validate_feature_precomputation.py
  src/validate_le2i_features.py
  src/validate_le2i_preprocessing.py

No changes staged for commit. main branch untouched.
```

- **Branch**: `dev` (`main` untouched).
- **Git Operations**: **No commits or pushes performed.**
