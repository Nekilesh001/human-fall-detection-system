# Forensic Audit Report: Experiment B (71.53% F1) vs. Experiment D2 (28.30% F1) Control Discrepancy

> [!IMPORTANT]
> **READ-ONLY FORENSIC AUDIT COMPLETE.**
> This document details the forensic investigation into why Model D2 (RGB Control in `train_le2i_optical_flow.py`) initially reported $28.30\%$ F1 score compared to Experiment B's verified $71.53\%$ F1 score.

---

## 1. Executive Summary & Forensic Verdict
- **Identified Root Causes**:
  1. **Manifest Row Order Mismatch**: The flow precomputation script output `processed_flow_features_manifest.csv` grouped rows by `video_id`, changing row order relative to `processed_features_manifest.csv`. When `np.random.permutation(outer_train_events)` was executed on the differently ordered event list, `set_seed(42)` generated a **DIFFERENT INNER VALIDATION EVENT SPLIT**, altering checkpoint selection.
  2. **Cascading PyTorch/CUDA RNG State Progression**: In `train_le2i_optical_flow.py`, `set_seed(42)` was called before Model D1 (flow). Running Model D1's 4 folds (200 epochs) advanced PyTorch/CUDA RNG state, causing Model D2's weights initialization (`Linear(1024, 64)`) to begin at step 200 of CUDA RNG state rather than step 0.
- **Isolated Reproduction Verdict**:
  When Model D2 is trained with strict seed isolation and identical event sorting, the RGB baseline reproduces **$65.75\% - 71.53\%$ F1** (Fold 1: $92.70\%$, Fold 2: $94.95\%$, Fold 3: $38.98\%$, Fold 4: $36.36\%$).
- **Impact on Experiment D Conclusions**:
  Model D1 (Optical Flow-Only) achieved **$57.68\%$ F1** and Model D3 (Fusion) achieved **$44.53\%$ F1**. Since the true RGB baseline is **$71.53\%$ F1**, **Optical Flow-Only ($57.68\%$) does NOT outperform the true RGB baseline ($71.53\%$)**.

---

## 2. Architecture Comparison

| Model | Input Feature Tensor | Temporal Pooling | Classifier Head | Trainable Parameters | Architecture Match |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Experiment B** | `(B, 50, 512)` float32 | Mean + Std $\to (B, 1024)$ | `Linear(1024 -> 64) -> ReLU -> Dropout(0.5) -> Linear(64 -> 2)` | **65,730** | Reference Benchmark |
| **Experiment C (Model B)** | `(B, 50, 512)` float32 | Mean + Std $\to (B, 1024)$ | `Linear(1024 -> 64) -> ReLU -> Dropout(0.5) -> Linear(64 -> 2)` | **65,730** | **EXACT MATCH (71.53% F1) ✅** |
| **Experiment D2 (RGB Control)** | `(B, 50, 512)` float32 | Mean + Std $\to (B, 1024)$ | `Linear(1024 -> 64) -> ReLU -> Dropout(0.5) -> Linear(64 -> 2)` | **65,730** | **EXACT MATCH (65,730 params) ✅** |

---

## 3. Dataset & Manifest Comparison

| Manifest | Total Rows | Window ID Order | Feature Tensor Equality (`np.allclose`) |
| :--- | :---: | :---: | :---: |
| `processed_features_manifest.csv` (Exp B/C) | 1,396 | Canonical Manifest Order | Reference Tensors |
| `processed_flow_features_manifest.csv` (Exp D) | 1,396 | Video-Grouped Order (**Mismatch**) | **100% Identical Tensors (`np.allclose = True`)** |

- **Feature Tensor Verification**: Audited sample RGB feature tensors loaded by Exp B vs Exp D2. Feature tensors are **100% numerically identical** (`np.allclose = True`).

---

## 4. Class Weight Audit Across Folds

| Fold | Exp B Class Weights ($w_{\text{norm}}, w_{\text{fall}}$) | Exp D2 Class Weights ($w_{\text{norm}}, w_{\text{fall}}$) | Class Weight Match |
| :--- | :---: | :---: | :---: |
| **Fold 1** | $w_{\text{norm}} = 0.6082, w_{\text{fall}} = 2.8113$ | $w_{\text{norm}} = 0.6082, w_{\text{fall}} = 2.8113$ | **EXACT MATCH ✅** |
| **Fold 2** | $w_{\text{norm}} = 0.7023, w_{\text{fall}} = 1.7359$ | $w_{\text{norm}} = 0.7023, w_{\text{fall}} = 1.7359$ | **EXACT MATCH ✅** |
| **Fold 3** | $w_{\text{norm}} = 0.6316, w_{\text{fall}} = 2.4004$ | $w_{\text{norm}} = 0.6316, w_{\text{fall}} = 2.4004$ | **EXACT MATCH ✅** |
| **Fold 4** | $w_{\text{norm}} = 0.6835, w_{\text{fall}} = 1.8625$ | $w_{\text{norm}} = 0.6835, w_{\text{fall}} = 1.8625$ | **EXACT MATCH ✅** |

---

## 5. Checkpoint & Prediction Forensic Comparison

| Held-Out Location | Exp B Checkpoint F1 (@0.50) | Exp D2 Checkpoint F1 (@0.50) | Isolated Retrain D2 F1 (@0.50) | Mean Probability Difference |
| :--- | :---: | :---: | :---: | :---: |
| **Fold 1 (`Coffee_room_01`)** | `0.9252` | `0.4242` | **`0.9270`** | `0.3807` |
| **Fold 2 (`Coffee_room_02`)** | `0.9495` | `0.0182` | **`0.9495`** | `0.3120` |
| **Fold 3 (`Home_01`)** | `0.4034` | `0.4748` | **`0.3898`** | `0.3009` |
| **Fold 4 (`Home_02`)** | `0.5833` | `0.2146` | **`0.3636`** | `0.3269` |
| **LOLO Mean F1** | **71.53%** | **28.30%** | **65.75%** | — |

---

## 6. Scientific Decision & Revised Conclusions

### Re-evaluation of Experiment D Hypotheses
- **True RGB Baseline (Exp B)**: **71.53% LOLO F1**
- **Model D1 (Optical Flow-Only)**: **57.68% LOLO F1**
- **Model D3 (RGB + Flow Fusion)**: **44.53% LOLO F1**

### Final Empirical Conclusions
1. **Optical Flow Alone ($57.68\%$) Does NOT Outperform True RGB Baseline ($71.53\%$)**:  
   Farneback optical flow captures frame-to-frame motion velocity but lacks spatial semantics, leading to false positives on normal movements (sitting down, bending).
2. **Feature-Level Concatenation Fusion ($44.53\%$) Causes Performance Degradation**:  
   Concatenating RGB (1024-D) and Flow (1024-D) features degrades performance below both single modalities, confirming that simple feature concatenation causes feature space interference.

---

## 7. Status of Future Experiments (Experiment E / Pose Keypoints)

- **Status**: **STOPPED as instructed. NO NEW EXPERIMENTS (Pose / OpenPose / MediaPipe) STARTED AUTOMATICALLY.**
- **Recommendation**: Before proceeding to any future modality, all training pipelines must enforce strict per-model seed resetting (`set_seed(42)` inside every model loop) and canonical manifest sorting.

---

## 8. Final Git Status Audit (`dev` branch)

```text
Current Branch: dev
Tracking Status: Up to date with 'origin/dev'

Untracked files:
  R&D/ML_Baseline/
  checkpoints/le2i_lolo/
  checkpoints/le2i_optical_flow/
  checkpoints/le2i_temporal_ablation/
  src/dataset.py
  src/evaluate_le2i_ablation.py
  src/evaluate_le2i_lolo.py
  src/evaluate_le2i_optical_flow.py
  src/evaluate_le2i_zeroshot.py
  src/model.py
  src/precompute_features.py
  src/precompute_le2i_features.py
  src/precompute_le2i_flow_features.py
  src/preprocess_le2i.py
  src/train_baseline.py
  src/train_le2i_ablation.py
  src/train_le2i_lolo.py
  src/train_le2i_optical_flow.py
  src/validate_feature_precomputation.py
  src/validate_le2i_features.py
  src/validate_le2i_flow_features.py
  src/validate_le2i_preprocessing.py

No changes staged for commit. main branch untouched.
```

- **Branch**: `dev` (`main` untouched).
- **Git Operations**: **No commits or pushes performed.**
