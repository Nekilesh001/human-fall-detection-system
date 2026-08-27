# Research Design: Pose Estimator Benchmark — MediaPipe vs YOLO Pose vs RTM Pose (Experiment H)

> [!IMPORTANT]
> **DESIGN ONLY — NO TRAINING OR PRECOMPUTATION PERFORMED YET.**
> This document specifies the scientific protocol, canonical 17-joint anatomical mapping, body-relative normalization equations, missing-keypoint handling policy, and 1-Layer LSTM classifier for Experiment H: Pose Estimator Benchmark.

---

## 1. Scientific Objective & Research Question
To determine whether the current best performance (**$73.34\%$ LOLO Mean F1**, Model G2 1-Layer LSTM) is constrained by keypoint landmark estimation quality from MediaPipe Pose, or whether alternative state-of-the-art pose estimators (**YOLO Pose** and **RTM Pose**) provide superior cross-location fall detection generalization.

### Core Scientific Question
*"Is the current 73.34% LSTM performance limited by the quality and detection robustness of the MediaPipe pose estimator rather than by the temporal sequence architecture?"*

---

## 2. Controlled Experimental Variables

To isolate pose estimator quality as the **single controlled variable**, all three benchmark variants (H1, H2, H3) enforce **identical downstream feature layout, normalization math, sequence length, and model capacity**:

- **Downstream Model Architecture**: 1-Layer LSTM (`hidden_size=64`) $\to$ `Linear(64 -> 32) -> ReLU -> Dropout(0.5) -> Linear(32 -> 2)`.
- **Trainable Parameter Count**: **41,378 parameters** (Exactly identical across H1, H2, and H3).
- **Temporal Sequence Length**: 50 frames per window ($2.0\text{ seconds}$ at 25 FPS).
- **Evaluation Protocol**: 4-Fold Leave-One-Location-Out (LOLO) cross-validation on 1,396 Le2i windows (127 physical videos).

---

## 3. Canonical 17-Joint Anatomical Mapping (COCO-17 Standard)

To enable a fair, unbiased comparison across heterogeneous pose architectures (MediaPipe's 33 landmarks vs YOLO/RTM Pose's COCO-17 keypoints), all landmark outputs are mapped to a **common 17 anatomical keypoint subset**:

```text
               1. Nose
         2. L_Eye    3. R_Eye
         4. L_Ear    5. R_Ear
      6. L_Shoulder  7. R_Shoulder
      8. L_Elbow     9. R_Elbow
     10. L_Wrist    11. R_Wrist
     12. L_Hip      13. R_Hip
     14. L_Knee     15. R_Knee
     16. L_Ankle    17. R_Ankle
```

### Landmark Index Mapping Definition

| Anatomical Keypoint Name | COCO-17 Standard Index | MediaPipe Pose Index (33-Landmark Layout) | YOLO Pose Index (COCO-17) | RTM Pose Index (COCO-17) |
| :--- | :---: | :---: | :---: | :---: |
| **Nose** | 0 | 0 | 0 | 0 |
| **Left Eye** | 1 | 2 | 1 | 1 |
| **Right Eye** | 2 | 5 | 2 | 2 |
| **Left Ear** | 3 | 7 | 3 | 3 |
| **Right Ear** | 4 | 8 | 4 | 4 |
| **Left Shoulder** | 5 | 11 | 5 | 5 |
| **Right Shoulder** | 6 | 12 | 6 | 6 |
| **Left Elbow** | 7 | 13 | 7 | 7 |
| **Right Elbow** | 8 | 14 | 8 | 8 |
| **Left Wrist** | 9 | 15 | 9 | 9 |
| **Right Wrist** | 10 | 16 | 10 | 10 |
| **Left Hip** | 11 | 23 | 11 | 11 |
| **Right Hip** | 12 | 24 | 12 | 12 |
| **Left Knee** | 13 | 25 | 13 | 13 |
| **Right Knee** | 14 | 26 | 14 | 14 |
| **Left Ankle** | 15 | 27 | 15 | 15 |
| **Right Ankle** | 16 | 28 | 16 | 16 |

---

## 4. Body-Relative Normalization & Velocity Derivation

For every frame $t \in [1, 50]$, raw keypoint pixel coordinates $(x_i, y_i)$ are transformed into scale- and translation-invariant normalized coordinates $(\hat{x}_i, \hat{y}_i)$:

### Mathematical Formulation

1. **Hip Center Reference**:
   $$\mathbf{p}_{\text{hip}} = \left( \frac{x_{11} + x_{12}}{2}, \frac{y_{11} + y_{12}}{2} \right)$$
2. **Shoulder Center Reference**:
   $$\mathbf{p}_{\text{shoulder}} = \left( \frac{x_{5} + x_{6}}{2}, \frac{y_{5} + y_{6}}{2} \right)$$
3. **Torso Normalization Length**:
   $$L_{\text{torso}} = \max\left( \|\mathbf{p}_{\text{shoulder}} - \mathbf{p}_{\text{hip}}\|_2, \epsilon \right), \quad \epsilon = 10^{-5}$$
4. **Torso-Centered & Scale-Normalized Coordinates**:
   $$\hat{x}_i = \frac{x_i - x_{\text{hip}}}{L_{\text{torso}}}, \quad \hat{y}_i = \frac{y_i - y_{\text{hip}}}{L_{\text{torso}}}$$
5. **Frame-to-Frame Joint Velocity**:
   $$d\hat{x}_i^{(t)} = \hat{x}_i^{(t)} - \hat{x}_i^{(t-1)}, \quad d\hat{y}_i^{(t)} = \hat{y}_i^{(t)} - \hat{y}_i^{(t-1)}, \quad \text{for } t \ge 2, \quad (\text{zero at } t=1)$$

### Per-Frame Feature Dimension Assembly ($D = 85$-D)
- **17 Keypoint Geometry**: $17 \times (\hat{x}_i, \hat{y}_i, v_i) = \mathbf{51\text{-D}}$
- **17 Keypoint Velocity**: $17 \times (d\hat{x}_i, d\hat{y}_i) = \mathbf{34\text{-D}}$
- **Total Vector per Frame**: $\mathbf{85\text{-D}}$
- **Input Tensor Shape**: $(B, 50, 85)$ float32.

---

## 5. Missing Keypoint & Undetected Frame Policy

- **Visibility / Confidence Threshold**: Keypoint visibility $v_i < 0.20$ is flagged as un-detected.
- **Undetected Frames**: If an estimator fails to detect a person in frame $t$, all keypoint coordinates and velocity entries are assigned zero vectors with $v_i = 0.0$.
- **Temporal Alignment**: Zero-padding preserves 50-frame temporal alignment across all 1,396 windows.

---

## 6. Model Variants to Benchmark

1. **H1 — MediaPipe Pose + 1-Layer LSTM** (85-D input, 41,378 params).
2. **H2 — YOLO Pose + 1-Layer LSTM** (85-D input, 41,378 params).
3. **H3 — RTM Pose + 1-Layer LSTM** (85-D input, 41,378 params).

---

## 7. Precomputation Footprint & Extraction Estimates

- **Total Frames to Process**: 127 supervised videos $\times$ 50 frames $\times$ windows = **69,800 total frames**.
- **Expected Storage Footprint**: 1,396 windows $\times$ 17 KB $\approx$ **23.7 MB storage per pose estimator**.
- **Extraction Latency**: ~3-5 minutes per pose estimator on GPU/CPU.
