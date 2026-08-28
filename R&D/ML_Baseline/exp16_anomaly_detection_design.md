# Research Design: Unsupervised Anomaly Detection for Fall Detection (Experiment #16)

> [!IMPORTANT]
> **EXPERIMENTAL DESIGN ONLY — READINESS AUDIT PHASE ONLY — NO CODE MODIFIED — NO TRAINING EXECUTED**  
> This document specifies the controlled experimental design, model architectures, one-class training protocols, decision threshold selection, and 4-fold LOLO evaluation strategies for Experiment #16: Unsupervised Anomaly Detection for Fall Detection.

---

## 1. Scientific Motivation & Core Paradigm Shift

In all previous experiments (Experiments A through K), fall detection was formulated as a **supervised binary classification problem** where models were trained on both `NORMAL` ADL activities ($y=0$) and `FALL` events ($y=1$).

Experiment #16 shifts to an **Unsupervised Anomaly Detection Paradigm**:
> *"Can human falls be accurately detected as anomalous departures from normal daily activity patterns by models trained EXCLUSIVELY ON NORMAL ADL SAMPLES ($y=0$), without ever observing a single fall sample during training?"*

### Why Anomaly Detection for Real-World Deployment?
1. **Extreme Real-World Class Imbalance**: In real-world smart-home or hospital deployments, fall events represent $< 0.1\%$ of daily video streams, while normal activities (walking, sitting, cooking) account for $> 99.9\%$.
2. **Unseen Fall Trajectories**: Supervised classifiers risk overfitting to specific fall styles. Anomaly detectors flag ANY rapid or structural deviation from normal movement as an anomaly.

---

## 2. Benchmark Model Architectures & Formulations

All anomaly detection variants consume the winning **187-D Spatial Feature Tensors `(50, 187)` float32** precomputed in Experiment K1 from `processed_data/Le2i_baseline/pose_estimator_features/yolo_pose_k1/`:

```text
Experiment #16 Anomaly Model Matrix:

M16-A (1D Conv-AE)   : 1D Temporal Convolutional Autoencoder -> 84,763 params  [MSE Reconstruction Error]
M16-B (One-Class SVM): OC-SVM with RBF Kernel (374-D Pooled)  -> Non-parametric [Boundary Margin Score]
M16-C (Isolation For): Ensemble of 100 Isolation Trees         -> Non-parametric [Path Length Anomaly Score]
```

### Model M16-A: 1D Temporal Convolutional Autoencoder (1D Conv-AE)
- **Encoder**: 2 Residual 1D Conv Layers (`187 -> 64 -> 32` channels, `kernel_size=3`, padding=1) $\to$ Latent representation $Z \in \mathbb{R}^{32 \times 50}$.
- **Decoder**: 2 Transposed 1D Conv / Up-Sampling Layers (`32 -> 64 -> 187` channels, `kernel_size=3`, padding=1) $\to$ Reconstructed sequence $\hat{X} \in \mathbb{R}^{50 \times 187}$.
- **Trainable Parameters**: **84,763 parameters** (Fair control vs K1 1D TCN 86,434 params).
- **Training Loss (Normal Samples Only)**: Mean Squared Reconstruction Error:
  $$\mathcal{L}_{\text{MSE}}(X, \hat{X}) = \frac{1}{50 \times 187} \sum_{t=1}^{50} \sum_{c=1}^{187} (X_{t, c} - \hat{X}_{t, c})^2$$
- **Inference Anomaly Score**: $S_{\text{AE}}(X) = \mathcal{L}_{\text{MSE}}(X, \hat{X})$.

### Model M16-B: One-Class SVM (OC-SVM)
- **Feature Vector**: Mean + Std Pooled 374-D vector derived from `(50, 187)` tensor.
- **Kernel**: Radial Basis Function (RBF) kernel ($\nu=0.05, \gamma=\text{scale}$).
- **Inference Anomaly Score**: $S_{\text{OC-SVM}}(X) = -f_{\text{OC-SVM}}(X)$ (Distance to hyperplane boundary).

### Model M16-C: Isolation Forest (iForest)
- **Feature Vector**: Mean + Std Pooled 374-D vector derived from `(50, 187)` tensor.
- **Ensemble Size**: $100$ isolation trees fitted on `NORMAL` training samples.
- **Inference Anomaly Score**: $S_{\text{iForest}}(X) \in [0, 1]$ (Average path length).

---

## 3. Scientific Controls & 4-Fold LOLO Protocol

To ensure 100% fair scientific comparison against the supervised K1 baseline ($86.60\%$ LOLO Mean F1), Experiment #16 enforces strict protocol controls:

| Protocol Parameter | Specification | Scientific Rationale |
| :--- | :--- | :--- |
| **Dataset** | Le2i 187-D Spatial Feature Tensors | 100% feature consistency with K1 |
| **Training Set** | Outer Train `NORMAL` Samples ONLY | Zero fall exposure during training |
| **Inner Validation** | 80/20 Event Split (Normal + Fall) | Tunes anomaly threshold $\tau^*$ without test leakage |
| **Outer Test** | Unseen Physical Location (Normal + Fall) | Zero test location data in training |
| **Random Seed** | `set_seed(42)` per fold/model | 100% deterministic reproducibility |
| **Evaluation Metrics** | LOLO Mean F1 (@ $\tau^*$), ROC-AUC, PR-AUC | Direct comparison with K1 baseline |

---

## 4. 4-Fold LOLO Partitioning Protocol for Anomaly Detection

| Fold | Outer Test Location | Normal Training Samples (AE / OC-SVM Fit) | Outer Test Total Windows | Test FALL Windows ($y=1$) | Test NORMAL Windows ($y=0$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Fold 1** | `Coffee_room_01` | 735 | 502 | 172 | 330 |
| **Fold 2** | `Coffee_room_02` | 702 | 410 | 47 | 363 |
| **Fold 3** | `Home_01` | 916 | 239 | 90 | 149 |
| **Fold 4** | `Home_02` | 842 | 245 | 22 | 223 |

---

## 5. Decision Threshold Selection Without Test Leakage

1. **Model Training**: Anomaly model $M$ is trained **EXCLUSIVELY** on outer train `NORMAL` samples ($y=0$).
2. **Inner Validation Threshold Tuning**:
   - Anomaly scores $S(X)$ are computed for inner validation samples (containing both `NORMAL` and `FALL` event windows).
   - An optimal anomaly threshold $\tau^*$ is selected by maximizing F1 score on inner validation predictions:
     $$\tau^* = \arg\max_{\tau} \text{F1}\Big(\{S(X_i) \ge \tau\}_{i \in \text{InnerVal}}\Big)$$
3. **Outer Test Evaluation**: Frozen model $M$ and frozen threshold $\tau^*$ are applied to outer test windows.

---

## 6. Primary & Secondary Evaluation Metrics

- **Primary Benchmark Metric**: **LOLO Mean F1 (@ $\tau^*$)**
- **Secondary Metrics**:
  - Area Under ROC Curve (**ROC-AUC**)
  - Area Under Precision-Recall Curve (**PR-AUC**)
  - Fall Recall / Sensitivity ($\text{TP} / (\text{TP} + \text{FN})$)
  - Normal Specificity ($\text{TN} / (\text{TN} + \text{FP})$)
  - Cross-Location Variance ($\sigma$)
  - **Supervised vs Unsupervised Gap**: $\Delta \text{F1} = \text{F1}_{\text{Supervised (K1)}} - \text{F1}_{\text{Anomaly}}$

---

## 7. Target Baseline Comparison Matrix

| Benchmark Experiment | Paradigm | Training Data Exposure | Feature Input | Model Architecture | Expected / Benchmark F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Exp K1 (CHAMPION SOTA)** | **Supervised** | Normal + Fall | 187-D Spatial | 1D Residual TCN (86.4K) | **$86.60\%$ LOLO F1** |
| **Exp M16-A (Conv-AE)** | **Unsupervised** | **Normal Only** | 187-D Spatial | 1D Conv-AE (84.7K) | Target Anomaly Baseline |
| **Exp M16-B (OC-SVM)** | **Unsupervised** | **Normal Only** | 374-D Pooled | OC-SVM RBF Kernel | Benchmark |
| **Exp M16-C (iForest)** | **Unsupervised** | **Normal Only** | 374-D Pooled | Ensemble 100 iTrees | Benchmark |
