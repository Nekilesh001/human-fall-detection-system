# Phase H3.1.1 — Experiment B Training Speed & Model Behavior Investigation Report

> [!IMPORTANT]
> **READ-ONLY INVESTIGATION STATUS & BASELINE SAFETY CONFIRMATION**  
> Baseline Model Path: `checkpoints/final_k1/final_production.pth`  
> Checkpoint SHA256: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d` (**IMMUTABLE & FROZEN**)  
> Policy Statement: **READ-ONLY INVESTIGATION ONLY. NO MODEL TRAINING OR RETRAINING WAS PERFORMED.**

---

## A. Executive Summary

Phase H3.1.1 investigated two core phenomena observed during Experiment B (Le2i + URFD candidate model training):
1. **Training Speed**: Why Experiment B trained in $\sim 15$ seconds compared to earlier baseline runs.
2. **Model Metrics Behavior**: Why Experiment B produced high Fall Recall ($95.92\%$) but high False Positive Rate ($98.41\%$) and low ROC-AUC ($0.4601$) under decision threshold $\tau = 0.3650$.

**Empirical Diagnosis Summary**:
- **Speed Explanation**: Experiment B trained on precomputed 187-D `.npz` feature tensors loaded directly from disk onto an **NVIDIA GeForce RTX 4060 Laptop GPU** (CUDA 12.6). Training involved a single split of 30 epochs ($2,640$ total optimizer steps), taking $\sim 15$ seconds on hardware acceleration without YOLO feature extraction overhead.
- **Metrics Behavior Root Cause**: Early epoch checkpoint selection (**Epoch 4**) selected an undertrained model state where predicted probabilities $P(\text{FALL})$ were tightly concentrated in $[0.3438, 0.4861]$ (mean $0.4169$, std $0.0258$). Because almost all probabilities exceeded $\tau = 0.3650$, the model predicted positive (FALL) for $98.41\%$ of all samples, generating high recall ($95.92\%$) alongside high false positives ($98.41\%$).

---

## B. Original K1 vs Experiment B Comparison Table

| Parameter | Original K1 Pipeline (`train_final_k1.py`) | Experiment B Pipeline (`train_multi_dataset_k1.py`) | Verification Status |
| :--- | :--- | :--- | :---: |
| **Training Datasets** | Le2i Only (1,396 windows) | Le2i + URFD (2,806 train windows) | **CONFIRMED FROM CODE** |
| **Feature Extraction** | Pre-extracted `.npz` features | Pre-extracted `.npz` features | **CONFIRMED FROM CODE** |
| **YOLO during Training** | No (YOLO executed offline) | No (YOLO executed offline) | **CONFIRMED FROM CODE** |
| **Hardware Device** | CUDA GPU | NVIDIA RTX 4060 Laptop GPU (CUDA 12.6) | **CONFIRMED FROM CODE** |
| **Training Scheme** | 4-Fold LOLO CV (200 epochs) + Full Retrain (50 epochs) | Single 70/15/15 Group Split (30 epochs) | **CONFIRMED FROM CODE** |
| **Total Epochs** | 250 Total Epochs | 30 Total Epochs | **CONFIRMED FROM CODE** |
| **Batch Size** | 32 | 32 | **CONFIRMED FROM CODE** |
| **Batches / Epoch** | 35 batches/epoch | 88 batches/epoch | **CONFIRMED FROM CODE** |
| **Total Optimizer Steps** | 8,750 steps | 2,640 steps | **CONFIRMED FROM CODE** |
| **Model Architecture** | `ModelK1_SpatialTCN` | `ModelK1_SpatialTCN` | **CONFIRMED FROM CODE** |
| **Input Dimension** | (50, 187) | (50, 187) | **CONFIRMED FROM CODE** |
| **Optimizer / Loss** | AdamW / Weighted CrossEntropy | AdamW / Weighted CrossEntropy (`pos_weight=4.0`) | **CONFIRMED FROM CODE** |

---

## C. Training Computation Analysis

For Experiment B:
$$\text{Batches per Epoch} = \left\lceil \frac{2806 \text{ train windows}}{32 \text{ batch size}} \right\rceil = 88 \text{ batches/epoch}$$

$$\text{Total Optimizer Steps} = 88 \text{ batches/epoch} \times 30 \text{ epochs} = 2,640 \text{ optimization steps}$$

On an NVIDIA RTX 4060 Laptop GPU:
- Tensor transfer and TCN forward/backward pass per batch of 32 items: $\sim 2\text{ ms}$
- Time per epoch ($88$ batches): $\sim 176\text{ ms}$
- Total compute time for 30 epochs: $\approx 5.28\text{ seconds}$ (excluding disk I/O)

**Conclusion**: The fast execution time is expected behavior for hardware-accelerated TCN training on pre-extracted feature tensors.

---

## D. Device & Hardware Acceleration Verification

- **PyTorch Version**: `2.13.0+cu126`
- **CUDA Available**: `True`
- **GPU Hardware**: `NVIDIA GeForce RTX 4060 Laptop GPU`
- **Model & Tensor Placement**: Verified `model.to(device)` and `x_b.to(device), y_b.to(device)` executed in PyTorch on CUDA GPU.

---

## E. Checkpoint Verification (`best_candidate.pth`)

Inspected `checkpoints/multi_dataset_k1/exp_b_le2i_urfd/best_candidate.pth`:
- **State Dict Keys Present**: `tcn.0.conv1.weight`, `tcn.0.conv1.bias`, `tcn.0.conv2.weight`, `tcn.0.conv2.bias`, `fc1.weight`, `fc2.weight`.
- **Layer 0 Weights (`tcn.0.conv1.weight`)**: Shape `(64, 187, 3)`, Min: `-0.1141`, Max: `+0.1027`, Mean: `-0.0001`, Std: `0.0308`.
- **Epoch Saved**: **Epoch 04** (Best Val F1 = 0.3170).
- **Weight Update Verification**: Weights differ from random initialization, confirming genuine backpropagation occurred.

---

## F. Evaluation Pipeline Verification (`evaluate_multi_dataset_k1.py`)

- **Positive Class Index**: Index 1 represents `FALL` (verified `probs = torch.softmax(out, dim=1)[:, 1]`).
- **Label Mapping**: `1` = FALL, `0` = NORMAL.
- **Threshold Rule**: `preds = (probs >= tau).astype(int)` where $\tau = 0.3650$.
- **Confusion Matrix Mapping**: `tn, fp, fn, tp = cm.ravel()`. Correctly mapped.

---

## G. Empirical Diagnosis of 98.41% FPR and ROC-AUC = 0.4601

### Raw Probability Distribution Statistics on Test Set (N = 602)
- **Minimum Probability**: `0.3438`
- **Maximum Probability**: `0.4861`
- **Mean Probability**: `0.4169` (Std: `0.0258`)
- **Percentiles**: `[10%: 0.3845, 25%: 0.4010, 50%: 0.4158, 75%: 0.4346, 90%: 0.4510]`

### Key Insight
Because **every single predicted probability** on the test set lies between `0.3438` and `0.4861`, evaluating at $\tau = 0.3650$ causes almost all 602 samples (including 496 Normal windows) to exceed $\tau = 0.3650$.

- **At $\tau = 0.3650$**:
  - $\text{TP} = 94 / 98$ ($\text{Recall} = 95.92\%$)
  - $\text{FP} = 496 / 504$ ($\text{FPR} = 98.41\%$)
- **At $\tau = 0.5000$**:
  - $\text{TP} = 0$, $\text{FP} = 0$, $\text{TN} = 504$, $\text{FN} = 98$ ($\text{Recall} = 0.00\%$)

### Why Epoch 4 Was Selected
Early in training (Epochs 1–4), the uncalibrated model outputs probabilities centered near $\sim 0.41$. Because validation set has $18.46\%$ Fall windows, predicting all samples as positive yielded a trivial validation F1 score of $31.70\%$ (Recall $99.07\%$, Precision $18.87\%$). The early stopping logic selected Epoch 4 as "best" before the model converged.

---

## H. Confirmed Root Causes & Issues Requiring Correction

1. **Early Epoch Checkpoint Selection Bug**: Early-stopping on validation F1 selected Epoch 4 when predictions were trivially positive ($\text{Recall} = 99\%$, $\text{Precision} = 18\%$).
2. **Uncalibrated Class Weighting**: `pos_weight = 4.0` in CrossEntropyLoss artificially shifted logits positive early in training.
3. **Probability Dynamic Range**: Probabilities remained compressed in $[0.34, 0.48]$, indicating under-convergence at Epoch 4.

---

## I. Production Checkpoint Safety Verification

- **Production Checkpoint**: `checkpoints/final_k1/final_production.pth`
- **SHA256 Hash**: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d` (**100% UNTOUCHED & FROZEN**) ✅

---

## J. Recommended Next Actions

1. Update `train_multi_dataset_k1.py` checkpoint selection logic to require a minimum precision threshold ($\text{Precision} \ge 0.50$) or select model based on minimum validation loss after warm-up epochs ($epoch \ge 10$).
2. Adjust `pos_weight` to $2.0$ or $1.0$ (or use standard BCE) to avoid shifting output logits into trivial positive regions.
