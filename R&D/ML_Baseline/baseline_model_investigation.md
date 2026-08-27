# First ML Baseline Model Design & Feasibility Investigation

## Executive Summary
This document formalizes the **architectural design, feature extraction strategy, overfitting mitigation, class balancing, evaluation protocols, and computational feasibility** for the first machine learning baseline of our Human Fall Detection System.

The objective is to establish a **trustworthy, reproducible, and leak-free initial baseline** on the processed URFD RGB dataset ($W=50$ frames @ 25 FPS, $320 \times 240$ resolution) before introducing complex pose-based, 3D CNN, or Transformer models.

---

## 1. Current Dataset Characteristics & Input Representation

### Dataset Inventory (URFD RGB Baseline)
- **Processed Scope**: 67 physical events (27 Fall events, 40 ADL events), 94 video streams.
- **Total Processed Samples**: **360 temporal window clips**.
- **Input Tensor Shape per Sample**: $(W, H, W_s, C) = (50, 240, 320, 3)$ with data type `uint8`.
- **Temporal Parameters**: Window size $W = 50$ frames (2.0s at 25.0 FPS), Stride $S = 25$ frames (1.0s, 50% overlap).

### Partition & Label Breakdown

| Partition | Unique Events | Unique Videos | FALL Windows (Class 1) | NORMAL Windows (Class 0) | Total Windows | Class Imbalance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 47 | 66 | 84 | 176 | **260** | 32.3% Fall / 67.7% Normal |
| **Validation** | 9 | 12 | 10 | 33 | **43** | 23.3% Fall / 76.7% Normal |
| **Test** | 11 | 16 | 24 | 33 | **57** | 42.1% Fall / 57.9% Normal |
| **TOTAL** | **67** | **94** | **118** | **242** | **360** | **32.8% Fall / 67.2% Normal** |

---

## 2. Research Question & Design Philosophy

> **Research Question**: What is the simplest, scientifically meaningful ML architecture that can consume the existing 50-frame RGB temporal windows and establish a trustworthy baseline before investigating complex 3D CNN, pose-based, or transformer models?

### Core Design Philosophy
1. **Low Overfitting Risk**: The dataset contains 360 windows from **67 physical events**. Training millions of spatial parameters end-to-end on 67 physical events will cause severe overfitting to room geometry and participant apparel.
2. **Frozen Spatial Feature Extraction**: Leverage rich, generalizable 2D spatial features from ImageNet-pretrained CNN backbones without updating backbone weights.
3. **Lightweight Trainable Classifier**: Train only a low-parameter temporal aggregation head or sequence classifier ($< 100,000$ trainable parameters).
4. **Strict Leakage-Safe Protocol**: All train, val, and test partitions are fixed by event ID (`seed=42`).

---

## 3. Candidate Architecture Investigation

Four primary candidate families were investigated:

```
Candidate 1: Pretrained 2D CNN (Frozen) ──> Temporal Mean Pooling ──────> Linear Classifier
Candidate 2: Pretrained 2D CNN (Frozen) ──> Temporal (Mean + Std) Pooling ──> MLP Classifier (RECOMMENDED)
Candidate 3: Pretrained 2D CNN (Frozen) ──> 1-Layer BiGRU Sequence Model ─> Linear Classifier
Candidate 4: End-to-End 3D Video CNN (R3D-18 / MC3-18) ─────────────────────> Linear Classifier
```

### Comparative Analysis of Candidate Families

| Candidate Family | Spatial Feature Source | Temporal Aggregation Method | Trainable Parameters | Overfitting Risk | Suitability for First Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Candidate 1** | Pretrained 2D CNN (Frozen) | Temporal Global Average | ~1,025 params | **Negligible** | Good simple baseline |
| **Candidate 2 (Recommended)** | **Pretrained 2D CNN (Frozen)** | **Temporal Mean + Std** | **~66,114 params** | **Very Low** | **PRIMARY BASELINE (RECOMMENDED)** |
| **Candidate 3** | Pretrained 2D CNN (Frozen) | 1-Layer Bidirectional GRU | ~263,938 params | Low | Secondary Ablation |
| **Candidate 4** | End-to-End 3D Video CNN | 3D Spatiotemporal Convolutions| ~33.2M params | **EXCEEDINGLY HIGH** | DEFERRED (High overfitting on 67 events) |

---

## 4. CNN Spatial Backbone Investigation

Pretrained 2D CNN backbones were evaluated for extracting per-frame feature vectors $\mathbf{z}_t \in \mathbb{R}^D$:

| Backbone Architecture | ImageNet Top-1 Accuracy | Parameter Count | Output Feature Dim ($D$) | Relative Feature Extractor Time (per frame) | Memory Footprint (FP32) | Transferability Rating |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ResNet-18 (Recommended)**| **69.8%** | **11.7M** | **512** | **Fastest (~0.8ms)** | **~45 MB** | **EXCELLENT (Stable features)** |
| **MobileNetV3-Large** | 75.2% | 5.4M | 960 | Fast (~1.1ms) | ~22 MB | Good (Edge mobile focus) |
| **EfficientNet-B0** | 77.1% | 5.3M | 1,280 | Moderate (~1.6ms) | ~21 MB | High (Slightly higher dim) |

### Backbone Recommendation: ResNet-18 (Frozen)
- **Why ResNet-18**: ResNet-18 produces a compact $D=512$ feature vector per frame. It is computationally lightweight, exceptionally stable during feature extraction, and has proven feature representation for human pose and spatial orientation.
- **Freeze Policy**: The ResNet-18 spatial backbone **MUST BE FROZEN** (`param.requires_grad = False`).

---

## 5. Dataset Size & Overfitting Analysis

### Statistical Independence vs. Sample Count
- The processed dataset contains 360 total window clips.
- Because windows are generated with stride $S=25$ (50% overlap), adjacent windows within a video share 25 frames.
- Crucially, the **true degrees of freedom is bounded by the 67 physical events**, NOT the 360 window clips.

### Overfitting Risk Demonstration
- Training a 3D CNN (33M parameters) end-to-end on 47 training events (260 windows) yields a parameter-to-sample ratio of $> 127,000 : 1$. The network will memorize fixed room lighting, floor patterns, and participant clothing, resulting in 100% training accuracy but poor test generalization.
- In contrast, using a **Frozen ResNet-18 + Temporal Mean-Std Pooling MLP** reduces trainable parameters to **~66,114**, yielding a safe parameter-to-sample ratio of ~254 : 1.

---

## 6. Class Imbalance Analysis & Strategy

### Dataset Class Distribution
- **FALL Windows (Class 1)**: 118 windows (32.8%)
- **NORMAL Windows (Class 0)**: 242 windows (67.2%)
- **Imbalance Ratio**: $\approx 1 : 2.05$ (Normal windows are double the Fall windows).

### Strategy Evaluation

| Strategy | Description | Pros | Cons | Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| **1. Class-Weighted Loss** | Multiply Fall loss by $w_{\text{fall}} \approx 1.52$, Normal loss by $w_{\text{norm}} \approx 0.74$. | Preserves all sample gradients; zero variance increase. | None | **RECOMMENDED FOR FIRST BASELINE** |
| **2. Weighted Random Sampling**| Resample minority Fall windows during mini-batch creation. | Equal class frequency per epoch. | Re-uses exact same 50-frame clips repeatedly. | Secondary option |
| **3. Decision Threshold Tuning**| Select optimal classification threshold $\tau$ on Val set. | Direct control over Precision vs Sensitivity trade-off. | Requires validation set tuning. | **RECOMMENDED (Post-Hoc)** |

#### Recommended Imbalance Strategy
Use **Class-Weighted Cross-Entropy Loss** during training, defined as:
$$w_c = \frac{N_{\text{total}}}{2 \times N_c} \implies w_{\text{normal}} = \frac{260}{2 \times 176} \approx 0.7386, \quad w_{\text{fall}} = \frac{260}{2 \times 84} \approx 1.5476$$
Combine with post-hoc decision threshold selection $\tau \in [0.1, 0.9]$ tuned on the Validation set.

---

## 7. Data Augmentation Policy

### Evaluation of Augmentations

1. **Random Horizontal Flip ($p = 0.5$)**: **APPROVED**. Left-right body motion symmetry is biologically valid and preserves fall semantics.
2. **Color Jitter (Brightness $\pm 10\%$, Contrast $\pm 10\%$)**: **APPROVED**. Simulates minor camera exposure and lighting shifts.
3. **Random Vertical Flip**: **PROHIBITED**. Inverts gravity, turning standing postures into fallen postures and corrupting label semantics.
4. **Random Rotation ($> 15^\circ$)**: **PROHIBITED**. Rotates upright body axes, confusing posture classification.
5. **Random Erasing / Cutout**: **PROHIBITED**. May remove the person entirely from low-resolution $320 \times 240$ frames.

---

## 8. Frozen Evaluation Protocol

### Metrics Required
- **Classification Performance**:
  - Precision: $\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$
  - Recall / Sensitivity: $\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$
  - F1 Score: $\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
  - Specificity: $\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}$
  - Confusion Matrix: $[\text{TN}, \text{FP}; \text{FN}, \text{TP}]$
- **Computational Performance**:
  - Inference Latency (ms per 50-frame window)
  - Throughput (FPS)
  - Trainable Parameter Count & Model Size (MB)

### Strict Decision Threshold Selection Protocol
- Train the model on the **Train partition** (47 events / 260 windows).
- Compute classification probabilities $P(\text{Fall})$ on the **Validation partition** (9 events / 43 windows).
- Select optimal decision threshold $\tau^* \in [0.1, 0.9]$ that maximizes Validation F1 Score / Sensitivity.
- Apply $\tau^*$ ONCE to the **Test partition** (11 events / 57 windows) to report final baseline results. **The Test set must never be used for threshold selection or hyperparameter tuning**.

---

## 9. Deployment & Clinical Scope Limitations

> [!WARNING]
> **CLINICAL SCOPE NOTICE**:
> This experiment evaluates **segmented event-level 2.0-second windows**. It does **NOT** measure:
> - False alarm rate per camera-hour in continuous real-world operation.
> - Alert latency in continuous surveillance streaming.
> - Robustness to occlusions, multiple patients, or staff movements in hospital wards.

---

## 10. Architecture Recommendations

### A. PRIMARY FIRST BASELINE (Recommended)
**Pretrained ResNet-18 (Frozen) + Temporal Mean-Std Pooling + 2-Layer MLP Classifier**
- **Spatial Extractor**: Frozen ImageNet-pretrained ResNet-18 ($\mathbf{z}_t \in \mathbb{R}^{512}$).
- **Temporal Pooling**: Concatenate temporal mean $\boldsymbol{\mu}$ and standard deviation $\boldsymbol{\sigma}$ across $W=50$ frames:
  $$\boldsymbol{\mu} = \frac{1}{W} \sum_{t=1}^W \mathbf{z}_t, \quad \boldsymbol{\sigma} = \sqrt{\frac{1}{W} \sum_{t=1}^W (\mathbf{z}_t - \boldsymbol{\mu})^2}, \quad \mathbf{h} = [\boldsymbol{\mu}; \boldsymbol{\sigma}] \in \mathbb{R}^{1024}$$
- **Classifier Head**: Linear($1024 \to 64$) $\to$ ReLU $\to$ Dropout(0.5) $\to$ Linear($64 \to 2$).
- **Trainable Parameters**: **66,114 parameters** (~0.25 MB).
- **Why**: Zero risk of spatial overfitting, captures both average posture ($\boldsymbol{\mu}$) and motion dynamics ($\boldsymbol{\sigma}$), trains in $< 30$ seconds.

### B. SECONDARY BASELINE / ABLATION (Recommended)
1. **Ablation 1 (Mean Pooling Only)**: ResNet-18 (Frozen) + Temporal Mean Pooling $\to$ Linear($512 \to 2$). (Tests importance of motion variance $\boldsymbol{\sigma}$).
2. **Ablation 2 (BiGRU Sequence Model)**: ResNet-18 (Frozen) + 1-Layer Bidirectional GRU (Hidden size 64) $\to$ Linear($128 \to 2$). (Tests sequential temporal modeling).

### C. DEFERRED ARCHITECTURES
- **End-to-End 3D Video CNNs (R3D-18, C3D)**: Deferred due to extreme overfitting risk on 67 physical events.
- **2D Pose Estimation + ST-GCN (RTMPose / MediaPipe)**: Deferred to Phase 2 (Pose-based modeling).
- **Video Transformers (ViViT / Timesformer)**: Deferred to Phase 3 (Advanced architectures).

---

## 11. Expected Computational Requirements

- **Device**: CPU or Single CUDA GPU (e.g. NVIDIA GTX/RTX).
- **Memory Footprint**: $< 2.0\text{ GB}$ VRAM / RAM.
- **Training Time**: $< 1\text{ minute}$ for 50 epochs on Train partition.
- **Inference Latency**: $< 15\text{ ms}$ per 50-frame window on GPU ($> 65\text{ windows/sec}$).

---

## 12. Next Implementation Step

1. Create directory `R&D/ML_Baseline/` (done).
2. Review this baseline model design with team.
3. Upon approval, implement dataset loader `src/dataset.py` reading `processed_manifest.csv` and sample `.npz` files.
4. Implement baseline architecture in `src/model.py` and training script `src/train_baseline.py`.
