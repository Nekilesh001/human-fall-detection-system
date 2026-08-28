# Research Design: Pose Estimator Benchmark — MediaPipe vs YOLO Pose vs RTMPose (Experiment H)

> [!IMPORTANT]
> **DESIGN & READINESS AUDIT ONLY — NO PRECOMPUTATION PERFORMED — NO TRAINING PERFORMED.**
> This document specifies the controlled scientific protocol, landmark mapping, normalization equations, missing detection policy, and future 5-phase experimental roadmap for Experiment H: Pose Estimator Benchmark.

---

## 1. Scientific Objective & Research Question

To determine whether the current best cross-location fall detection performance (**$73.34\%$ LOLO Mean F1**, Model G2 1-Layer LSTM) is constrained by keypoint estimation robustness from MediaPipe Pose, or whether alternative state-of-the-art pose estimators (**YOLO Pose** and **RTMPose**) provide superior detection stability and generalizability on unseen physical locations (especially `Home_01` and `Home_02`).

### Core Scientific Question
*"Does the choice of human pose estimator materially affect cross-location fall-detection performance when the downstream classifier architecture is strictly controlled?"*

---

## 2. Controlled Experimental Variables

To isolate pose estimator quality as the **single controlled variable**, all three benchmark variants (H1, H2, H3) enforce **identical downstream classifier architecture, feature dimensionality, sequence length, and LOLO partition splits**:

- **Downstream Classifier Sub-Network**: Canonical E2 / G0 Pose + Velocity MLP Control:
  $$\text{Input } (B, 50, 165) \xrightarrow{\text{Mean+Std}} (B, 330) \xrightarrow{\text{Linear}(330 \to 64)} \text{ReLU} \xrightarrow{\text{Dropout}(0.5)} \text{Linear}(64 \to 2)$$
- **Trainable Parameter Count**: **21,314 parameters** (Exactly identical across H1, H2, and H3).
- **Temporal Sequence Length**: 50 frames per window ($2.0\text{ seconds}$ at 25 FPS).
- **Feature Vector Dimensionality**: **165-D per frame** (99-D Pose Geometry + 66-D Joint Velocity).
- **Dataset & Partitioning**: Exact same 1,396 Le2i windows (127 supervised videos, 96 FALL, 31 NORMAL, 63 UNKNOWN excluded) across the 4 physical LOLO locations.

---

## 3. Canonical 33-Landmark Topology Mapping & Vector Layout

To guarantee complete input compatibility with the 165-D downstream classifier head while accommodating COCO-17 estimators (YOLO Pose and RTMPose), all estimator keypoints map to a **canonical 33-landmark vector layout**:

```text
               0. Nose
         2. L_Eye    5. R_Eye
         7. L_Ear    8. R_Ear
      11. L_Shoulder 12. R_Shoulder
      13. L_Elbow    14. R_Elbow
      15. L_Wrist    16. R_Wrist
      23. L_Hip      24. R_Hip
      25. L_Knee     26. R_Knee
      27. L_Ankle    28. R_Ankle
```

### Landmark Index Mapping Matrix

| Anatomical Keypoint Name | Canonical 33 Index | MediaPipe Pose Index | YOLO Pose Index (COCO-17) | RTMPose Index (COCO-17) | Mapping Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Nose** | 0 | 0 | 0 | 0 | **Direct 1-to-1 ✅** |
| **Left Eye / Right Eye** | 2, 5 | 2, 5 | 1, 2 | 1, 2 | **Direct 1-to-1 ✅** |
| **Left Ear / Right Ear** | 7, 8 | 7, 8 | 3, 4 | 3, 4 | **Direct 1-to-1 ✅** |
| **Left / Right Shoulders** | 11, 12 | 11, 12 | 5, 6 | 5, 6 | **Direct 1-to-1 ✅** |
| **Left / Right Elbows** | 13, 14 | 13, 14 | 7, 8 | 7, 8 | **Direct 1-to-1 ✅** |
| **Left / Right Wrists** | 15, 16 | 15, 16 | 9, 10 | 9, 10 | **Direct 1-to-1 ✅** |
| **Left / Right Hips** | 23, 24 | 23, 24 | 11, 12 | 11, 12 | **Direct 1-to-1 ✅** |
| **Left / Right Knees** | 25, 26 | 25, 26 | 13, 14 | 13, 14 | **Direct 1-to-1 ✅** |
| **Left / Right Ankles** | 27, 28 | 27, 28 | 15, 16 | 15, 16 | **Direct 1-to-1 ✅** |
| **Facial & Foot Detail (16 Landmarks)** | Remaining | Indices 1, 3, 4, 6, 9-10, 17-22, 29-32 | N/A | N/A | **Zero-Padded ($v=0.0$)** |

---

## 4. Scale- & Translation-Invariant Torso Normalization

For every frame $t \in [1, 50]$, raw keypoint coordinates $(x_i, y_i)$ are transformed into scale- and translation-invariant normalized coordinates $(\hat{x}_i, \hat{y}_i)$:

### Mathematical Formulation

1. **Hip Center Reference**:
   $$\mathbf{p}_{\text{hip}} = \left( \frac{x_{23} + x_{24}}{2}, \frac{y_{23} + y_{24}}{2} \right)$$
2. **Shoulder Center Reference**:
   $$\mathbf{p}_{\text{shoulder}} = \left( \frac{x_{11} + x_{12}}{2}, \frac{y_{11} + y_{12}}{2} \right)$$
3. **Torso Normalization Length**:
   $$L_{\text{torso}} = \max\left( \|\mathbf{p}_{\text{shoulder}} - \mathbf{p}_{\text{hip}}\|_2, \epsilon \right), \quad \epsilon = 10^{-5}$$
4. **Torso-Centered & Scale-Normalized Coordinates**:
   $$\hat{x}_i = \frac{x_i - x_{\text{hip}}}{L_{\text{torso}}}, \quad \hat{y}_i = \frac{y_i - y_{\text{hip}}}{L_{\text{torso}}}$$
5. **Frame-to-Frame Joint Velocity**:
   $$d\hat{x}_i^{(t)} = \hat{x}_i^{(t)} - \hat{x}_i^{(t-1)}, \quad d\hat{y}_i^{(t)} = \hat{y}_i^{(t)} - \hat{y}_i^{(t-1)}, \quad \text{for } t \ge 2, \quad (\text{zero at } t=1)$$

### Per-Frame Feature Dimension Assembly ($D = 165$-D)
- **33 Landmark Geometry**: $33 \times (\hat{x}_i, \hat{y}_i, v_i) = \mathbf{99\text{-D}}$
- **33 Landmark Velocity**: $33 \times (d\hat{x}_i, d\hat{y}_i) = \mathbf{66\text{-D}}$
- **Total Vector per Frame**: $\mathbf{165\text{-D}}$ $\to$ Input Tensor Shape: $(B, 50, 165)$ float32.

---

## 5. Missing Detection & Low Confidence Policy

- **Undetected Frame**: If an estimator fails to detect a person in frame $t$, all 33 landmark entries and velocity features are assigned zero vectors with $v_i = 0.0$.
- **Low Confidence Threshold**: Keypoints with confidence score $< 0.20$ are zeroed.
- **Temporal Alignment**: Zero-padding preserves 50-frame temporal alignment across all 1,396 windows.

---

## 6. Future 5-Phase Experimental Roadmap

- **Phase H1**: Full Pose Feature Precomputation for H1, H2, and H3 across all 1,396 Le2i windows.
- **Phase H2**: Phase 1 Precomputation Validation Gate ($0$ missing files, $0$ NaN/Inf errors).
- **Phase H3**: 4-Fold LOLO Training of Controlled Pose+Velocity MLP ($21,314$ params) for H1, H2, H3.
- **Phase H4**: Cross-Location Robustness Analysis (evaluating performance on `Home_01` and `Home_02`).
- **Phase H5**: Winning Pose Estimator Selection & Post-H Combination with Best Temporal Model (G2 LSTM).
