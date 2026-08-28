# Research Design: Spatial-Temporal Graph Convolutional Network (Experiment K Phase K3)

> [!IMPORTANT]
> **EXPERIMENTAL DESIGN ONLY — READINESS AUDIT & CUDA SMOKE TEST PHASE ONLY — NO TRAINING EXECUTED**  
> This document specifies the controlled experimental design, COCO-17 graph topology, 3-partition spatial graph convolution, input tensor formulation, and 4-fold LOLO ablation protocols for Model K3 (Spatial-Temporal Graph Convolutional Network).

---

## 1. Research Motivation & Scientific Question

In Experiment I and K1, combining **YOLO Pose keypoint representations** with temporal models established two benchmark milestones:
- **K0 Control (1D TCN, 165-D)**: **$83.60\%$ LOLO Mean F1** ($\pm 5.74\%$)
- **K1 Spatial (187-D Derived Features)**: **$86.60\%$ LOLO Mean F1** ($\pm 5.81\%$)

Experiment K3 addresses the core topological research question:
> *"Does explicit skeletal topology modeling with a Spatial-Temporal Graph Convolutional Network (ST-GCN) improve cross-location fall detection beyond the YOLO Pose TCN baselines?"*

---

## 2. Topological Graph Definition ($V=17$ COCO Keypoints)

The human skeleton is represented as a spatial-temporal graph $G = (V, E)$ over **17 populated COCO keypoints**:

### Graph Nodes ($V = 17$ Keypoints)
- **Head (5)**: `0` Nose, `1` L_Eye, `2` R_Eye, `3` L_Ear, `4` R_Ear
- **Upper Body (6)**: `5` L_Shoulder, `6` R_Shoulder, `7` L_Elbow, `8` R_Elbow, `9` L_Wrist, `10` R_Wrist
- **Lower Body (6)**: `11` L_Hip, `12` R_Hip, `13` L_Knee, `14` R_Knee, `15` L_Ankle, `16` R_Ankle

> **Facial Edge Rationale**: Facial keypoints ($0-4$) are explicitly included because head orientation and vertical downward descent of ears/nose during fall impact provide crucial motion cues that distinguish falling from sitting.

### Anatomical Spatial Edges ($E_{\text{spatial}}$, 16 Undirected Bones)
- **Head (4)**: $(0,1), (0,2), (1,3), (2,4)$
- **Upper Body (6)**: $(5,6), (5,7), (7,9), (6,8), (8,10), (5,11)$
- **Lower Body (6)**: $(6,12), (11,12), (11,13), (13,15), (12,14), (14,16)$

### 3-Partition Spatial Adjacency Matrix $A \in \mathbb{R}^{3 \times 17 \times 17}$
1. **Partition 0 (Self-Loops)**: $A_0 = I_{17}$ (Centripetal self-joint connections).
2. **Partition 1 (Inward Edges)**: Edges connecting nodes towards the body center of gravity (hips `11, 12`).
3. **Partition 2 (Outward Edges)**: Edges connecting nodes away from the body center of gravity towards extremities.

---

## 3. Input Tensor Formulation

The ST-GCN model consumes input tensor shape **$(B, 5, 50, 17)$ float32**:
- **Batch Size ($B$)**: 32
- **Channel Dimension ($C=5$)**: $[X, Y, V, dX, dY]$
  - $X, Y$: Normalized 2D keypoint coordinates in $[0, 1]$
  - $V$: Keypoint detection confidence score in $[0, 1]$
  - $dX, dY$: Instantaneous 2D keypoint velocities
- **Sequence Length ($T=50$)**: 50 frames ($2.0\text{ s}$ at 25 FPS)
- **Joint Count ($V=17$)**: 17 populated COCO keypoints

### Data Reuse Strategy
The `(5, 50, 17)` input tensor is sliced **dynamically in memory** directly from the canonical 165-D YOLO Pose NPZ files in `processed_data/Le2i_baseline/pose_estimator_features/yolo_pose/`. **NO new precomputation directory or video extraction required**.

---

## 4. ST-GCN Model Architecture & Verified Parameter Count

```text
Model K3 ST-GCN Architecture Flow:

Input (B, 5, 50, 17)
       │
       ▼
STGCNBlock Layer 1  [Channels: 5  -> 32,  Spatial Graph Conv + Temporal Conv(9, 1)]
       │
       ▼
STGCNBlock Layer 2  [Channels: 32 -> 64,  Spatial Graph Conv + Temporal Conv(9, 1)]
       │
       ▼
STGCNBlock Layer 3  [Channels: 64 -> 64,  Spatial Graph Conv + Temporal Conv(9, 1)]
       │
       ▼
AdaptiveAvgPool2d   [Pooling over (T, V) -> (B, 64)]
       │
       ▼
Linear(64 -> 32) -> ReLU -> Dropout(0.5) -> Linear(32 -> 2)
       │
       ▼
Output Logits (B, 2)
```

- **Verified Trainable Parameters**: **107,778 parameters** (Programmatically verified via PyTorch).

---

## 5. Fair Control & Baseline Comparisons

- **Primary Control Baseline**: **K0/I3 (YOLO Pose + 1D TCN, 165-D)** $\to$ **$83.60\%$ LOLO Mean F1**.
- **Secondary Feature Benchmark**: **K1 (YOLO Pose + 187-D Spatial TCN)** $\to$ **$86.60\%$ LOLO Mean F1**.
- **Isolation Scope**: Model K3 tests explicit skeletal graph topology independently without blending K1's 22 engineered features.

---

## 6. Scientific Controls & 4-Fold LOLO Protocol

- Same 4 physical locations (`Coffee_room_01`, `Coffee_room_02`, `Home_01`, `Home_02`).
- 0 outer test event leakage across 4 LOLO folds.
- Optimal threshold $\tau^*$ tuned strictly on inner validation predictions.
- `set_seed(42)` per fold/model.
- Execution Device: PyTorch CUDA on NVIDIA GeForce RTX 4060 Laptop GPU.
