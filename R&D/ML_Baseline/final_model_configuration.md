# Final System Configuration: Champion Model K1 (187-D Spatial TCN)

> [!IMPORTANT]
> **FROZEN SOTA CHAMPION SYSTEM CONFIGURATION — EXPERIMENT #20 AUDITED**  
> Model K1 (YOLO Pose + 187-D Spatial Features + 1D Residual TCN) is officially locked as the **UNDISPUTED SYSTEM CHAMPION SOTA** with **$86.65\%$ LOLO Mean F1** ($\pm 5.64\%$) under the leakage-free inner-validation operating threshold policy ($\bar{\tau}^*_{\text{inner}} = 0.4923 \pm 0.0134$).

---

## 1. System Overview

- **Model Name**: Model K1 (187-D Spatial-Augmented Temporal Convolutional Network)
- **Modality**: Skeleton-Based Keypoint Sequences (2D YOLO Pose)
- **Input Feature Dimension**: 187-D float32 per frame
- **Temporal Receptive Field**: 50 frames ($2.0\text{ seconds}$ at 25 FPS)
- **Input Tensor Shape**: `(B, 50, 187)` float32
- **Trainable Parameters**: **89,250 parameters** (weights + biases + BatchNorm parameters)
- **Execution Device**: PyTorch CUDA (NVIDIA GeForce RTX 4060 Laptop GPU)
- **Official Leakage-Free Benchmark Score**: **$86.65\%$ LOLO Mean F1** ($\pm 5.64\%$)

---

## 2. Model Architecture Specification

```text
Model K1 Spatial TCN Architecture Flow:

Input Window (B, 50, 187)
       │  permute to (B, 187, 50)
       ▼
TemporalBlock Layer 1  [in=187, out=64, kernel=3, dilation=1, padding=2, dropout=0.2]
       │
       ▼
TemporalBlock Layer 2  [in=64,  out=64, kernel=3, dilation=2, padding=4, dropout=0.2]
       │
       ▼
Concat(MeanPooling, MaxPooling) over time  [Shape -> (B, 128)]
       │
       ▼
Linear(128 -> 32) -> ReLU -> Dropout(0.5)
       │
       ▼
Linear(32 -> 2) -> Logits (B, 2)
       │
       ▼
Softmax -> P(FALL) >= 0.4923 -> NORMAL / FALL
```

---

## 3. Feature Representation (187-D) Breakdown

1. **Base Normalized Coordinates (99-D)**: 33 canonical MediaPipe landmarks $\times 3$ values $(X, Y, V)$ normalized by torso length $L_{\text{torso}} = \|\text{Hip}_{\text{center}} - \text{Shoulder}_{\text{center}}\|$.
2. **Instantaneous 2D Velocities (66-D)**: Frame-to-frame coordinate differences $(dX, dY)$ for 33 canonical landmarks.
3. **Derived Body Angles & Flexions (12-D)**: Flexion angles for Left/Right Knees, Left/Right Hips, Left/Right Elbows, Left/Right Shoulders, Spine Inclination $\theta_{\text{spine}}$, Neck Angle, and Leg Verticals.
4. **Bounding Box Aspect Features (2-D)**: Aspect ratio $W/H$ and bounding box area ratio.
5. **Normalized Joint Heights (4-D)**: Heights of Head, Wrists, Ankles relative to torso length.
6. **Torso Deformation & Tilt Metrics (4-D)**: Torso scale ratio, shoulder tilt, hip tilt, torso aspect ratio.

---

## 4. Frozen Checkpoints & SHA256 Hashes

- Checkpoint Directory: `checkpoints/le2i_yolo_k1/`

| Checkpoint File | Outer Test Location | Size (Bytes) | Parameters | SHA256 Checksum |
| :--- | :--- | :---: | :---: | :--- |
| `fold_1_best.pth` | `Coffee_room_01` | 362,825 | 89,250 | `099edd6e3b549e816f90a0ec8f2bf90c311e9735da9d1ee11d1acd6d22363c21` |
| `fold_2_best.pth` | `Coffee_room_02` | 362,825 | 89,250 | `7ca9d0ec5cc310ec12f99d83c373bffbd512c992d27883a1ea3421299f7ba3fc` |
| `fold_3_best.pth` | `Home_01` | 362,825 | 89,250 | `7fb0675474349151ac2033ab943dea864bb517a47a9b18760e8eebfa94f900ab` |
| `fold_4_best.pth` | `Home_02` | 362,825 | 89,250 | `6ee5469704def6328a8f95d6b05f1e22e8b6db4e87026a72eed171b10634bb2e` |

---

## 5. Validated Decision Threshold Policy

- **Official Threshold Selection Method**: Leakage-free inner-validation selection ($\tau^*_{\text{inner}}$) tuned on 20% inner validation predictions without observing outer test data.
- **Mean Selected Threshold**: $\bar{\tau}^*_{\text{inner}} = \mathbf{0.4923 \pm 0.0134}$.
- **Deployment Rule**: Classify as `FALL` if $P(\text{FALL}) \ge 0.4923$; otherwise `NORMAL`.
- **Exploratory Reference Note**: The post-hoc outer-test threshold sweep peak of $87.45\%$ at $\tau = 0.55$ (Exp #18) is documented strictly as an exploratory upper bound, **NOT** the official un-cheated benchmark score.
