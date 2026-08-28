# Research Design: Post-SOTA Architectural & Feature Enhancement Benchmark (Experiment K)

> [!IMPORTANT]
> **EXPERIMENTAL DESIGN ONLY — READINESS AUDIT PHASE ONLY — NO CODE MODIFIED — NO TRAINING EXECUTED**  
> This document specifies the controlled experimental design, model architectures, feature formulations, and 4-fold LOLO ablation protocols for Experiment K: Post-SOTA Architectural & Feature Enhancement Benchmark.

---

## 1. Research Motivation & Scientific Objectives

In Experiment I and J, combining **YOLO Pose keypoint representations (165-D)** with a **2-block Residual 1D TCN (Model I3)** established an all-time state-of-the-art record of **$83.60\%$ LOLO Mean F1** ($\pm 5.74\%$).

However, evidence-based failure analysis in Experiment J2 uncovered two core limitations of the 1D TCN baseline:
1. **Behavioral Ambiguity in 2D Velocity (65.6% of errors)**: Non-fall ADL activities (crouching, tying shoes, rapid sitting) exhibit 2D downward velocity peaks ($20.98$) that match or exceed true falls ($14.92$). The 1D TCN operating on 2D joint velocities alone cannot distinguish controlled descent from free-fall collapse.
2. **Short Receptive Field ($2.0\text{ seconds}$ / 50 frames)**: 50-frame windows capture only the immediate fall impact, lacking sufficient context to observe the pre-fall standing state or post-fall recovery state.

Experiment K tests two independent scientific hypotheses:
- **Hypothesis K1 (Spatial Feature Augmentation)**: Explicitly augmenting the feature vector with 3D-proxy joint angles, torso tilt, and body aspect ratio will resolve non-fall downward velocity ambiguity.
- **Hypothesis K2 (Multi-Scale Temporal Context)**: Extending temporal window length to 100 frames ($4.0\text{ seconds}$) provides complete pre-fall and post-fall context to differentiate falls from transient crouching.
- **Hypothesis K3 (Spatial-Temporal Graph Convolutional Networks)**: Structuring skeletal motion as a topological graph $G = (V, E)$ explicitly enforces anatomical bone connectivity and joint angle constraints.

---

## 2. J2 Failure Evidence Motivating Experiment K

| Failure Evidence (J2) | Quantitative Metric (J2) | Root-Cause Analysis | Target K Benchmark |
| :--- | :--- | :--- | :--- |
| **Rapid Non-Fall Descent** | FP Max Downward Velocity = $20.98$ vs TP = $14.92$ | 2D velocity alone is ambiguous | **K1 (187-D Spatial Feature)** & **K3 (ST-GCN)** |
| **Short Window Context** | 50 frames ($2.0\text{ s}$) captures impact only | Lacks pre-fall standing & post-fall recovery | **K2 (100-Frame Context)** |
| **Furniture Occlusion** | FN Keypoint Vis = $0.2746$ ($32.1\%$ lower) | Partial lower-body occlusion in `Home_01` | **K3 (ST-GCN Graph Interpolation)** |

---

## 3. Controlled Experiment K Sub-Benchmark Definitions

```text
Experiment K Controlled Architecture & Feature Matrix:

K0 (Frozen Control)  : YOLO Pose (165-D)  + 1D TCN (50f)  -> 83,618 params  [83.60% F1 SOTA Baseline]
K1 (Spatial Augment) : YOLO Pose (187-D)  + 1D TCN (50f)  -> 86,434 params  [Tests Spatial Angle Hypothesis]
K2 (Temporal Context): YOLO Pose (165-D)  + 1D TCN (100f) -> 83,618 params  [Tests 4.0s Window Hypothesis]
K3 (ST-GCN Graph)   : COCO-17 Graph (3D) + ST-GCN (50f)   -> 54,146 params  [Tests Skeletal Graph Hypothesis]
```

### K0 — Frozen Control (SOTA Baseline)
- **Architecture**: 2 Residual TCN Blocks (`num_channels=[64, 64], kernel_size=3`, dilations=[1, 2]) $\to$ Mean+Max Concatenated Pooling $\to$ Linear(128 $\to$ 32) $\to$ Linear(32 $\to$ 2).
- **Features**: Canonical 165-D YOLO Pose (99-D Pose Geometry + 66-D Joint Velocity).
- **Window Length**: 50 frames ($2.0\text{ s}$).
- **Trainable Parameters**: **83,618 parameters**.
- **Benchmark Performance**: **$83.60\%$ LOLO Mean F1** ($\pm 5.74\%$).

---

### K1 — Spatial Feature Augmentation (187-D Representation)
- **Scientific Mechanism**: Augments the 165-D base vector with **22 derived spatial & geometric features**:
  1. **Joint Angles (12 features)**: Knee flexion angles (L/R), Hip flexion angles (L/R), Elbow angles (L/R), Shoulder angles (L/R), Torso/Spine inclination angle $\theta_{\text{spine}} = \arctan2(\Delta Y_{\text{shoulder-hip}}, \Delta X_{\text{shoulder-hip}})$, Neck angle.
  2. **Bounding Box Aspect Ratio (2 features)**: Bounding box width-to-height ratio $W_{\text{bbox}} / H_{\text{bbox}}$, area ratio.
  3. **Normalized Heights (4 features)**: Head-to-floor, Wrist-to-floor, Ankle-to-floor relative to torso length.
  4. **Torso Deformation (4 features)**: Torso length ratio $L_{\text{torso}}(t) / L_{\text{torso}}(t=0)$, lateral tilt angle.
- **Input Tensor Shape**: `(B, 50, 187)` float32.
- **Data Source**: All 22 features are derived mathematically from the existing 33 keypoint coordinates $(X_i, Y_i)$ stored in `processed_data/Le2i_baseline/pose_estimator_features/yolo_pose/`. **NO video re-extraction required**.
- **Trainable Parameters**: **86,434 parameters** (+3.3% parameter change over K0).

---

### K2 — Multi-Scale Temporal Context (100-Frame Windowing)
- **Scientific Mechanism**: Increases sequence window length from 50 frames ($2.0\text{ s}$) to **100 frames ($4.0\text{ s}$ at 25 FPS)**.
- **Temporal Window Structure**:
  $$\underbrace{\text{Frames 0–25}}_{\text{Pre-Fall Standing (1.0s)}} \longrightarrow \underbrace{\text{Frames 25–75}}_{\text{Fall Descent & Impact (2.0s)}} \longrightarrow \underbrace{\text{Frames 75–100}}_{\text{Post-Fall Lying Still (1.0s)}}$$
- **Dataset Audit**: 127 out of 127 Le2i videos have $\ge 133$ frames (Mean length = 500 frames). All videos support 100-frame window extraction.
- **Input Tensor Shape**: `(B, 100, 165)` float32.
- **Trainable Parameters**: **83,618 parameters** (Identical to K0; temporal dimension processed via global pooling).

---

### K3 — Spatial-Temporal Graph Convolutional Network (ST-GCN)
- **Graph Topology Definition**:
  - **Nodes ($V$)**: 17 COCO keypoints (Nose, L/R Eye, L/R Ear, L/R Shoulder, L/R Elbow, L/R Wrist, L/R Hip, L/R Knee, L/R Ankle).
  - **Spatial Edges ($E_{\text{spatial}}$)**: 16 anatomical bones connecting adjacent joints.
  - **Temporal Edges ($E_{\text{temporal}}$)**: Trajectories linking joint $v_i(t) \to v_i(t+1)$ across 50 frames.
- **Input Tensor Shape**: `(B, 3, 50, 17)` float32 (3 channels: Normalized $X$, Normalized $Y$, Visibility $V$).
- **Architecture**: 3 ST-GCN Blocks (`channels=[32, 64, 64]`, Spatial Graph Conv with 3-partition adjacency matrix + Temporal Conv1d) $\to$ Global Spatial-Temporal Avg Pooling $\to$ Linear(64 $\to$ 2).
- **Trainable Parameters**: **54,146 parameters** (-35% parameter count compared to K0 1D TCN).

---

## 4. Scientific Controls & 4-Fold LOLO Protocol

All sub-experiments in Experiment K maintain strict protocol controls:

| Parameter | Specification | Scientific Rationale |
| :--- | :--- | :--- |
| **Dataset** | Le2i Supervised Temporal Windows | 100% dataset consistency |
| **Physical Locations** | `Coffee_room_01`, `Coffee_room_02`, `Home_01`, `Home_02` | Controlled LOLO physical partitions |
| **Outer Test Isolation** | 0 event overlap across outer folds | Zero data leakage |
| **Threshold Tuning** | Optimal $\tau^*$ tuned strictly on Inner Val | Frozen application to Outer Test |
| **Random Seed** | `set_seed(42)` per fold/model | 100% deterministic reproducibility |
| **Execution Hardware** | PyTorch GPU CUDA (RTX 4060) | Hardware consistency |

---

## 5. Target Failure Mode Mapping

| Experiment Variant | Target Failure Mode from J2 | Scientific Hypothesis |
| :--- | :--- | :--- |
| **K1 (187-D Spatial)** | Rapid Crouching / Bending FPs ($N=37$ High-Conf FPs) | Torso inclination angle $\theta_{\text{spine}}$ and knee flexion angles will distinguish controlled bending from uncontrolled falling. |
| **K2 (100f Context)** | Post-Fall Lying vs Normal Lying FPs & Boundary FNs | 4.0s context will capture post-fall stillness or post-crouch recovery, resolving temporal boundary errors. |
| **K3 (ST-GCN Graph)** | Partial Occlusion FNs ($N=21$ High-Conf FNs in `Home_01`) | Topological bone constraints $E_{\text{spatial}}$ enforce structural limb geometry even when lower body joints are partially occluded. |

---

## 6. Recommended Execution Order & Go/No-Go Decision Criteria

1. **Step 1**: Run **K1 (187-D Spatial Augmentation)** — Instant computation from existing 33-landmark tensors without creating new datasets.
2. **Step 2**: Run **K3 (ST-GCN)** — Evaluates skeletal graph topology on existing COCO-17 keypoints.
3. **Step 3**: Run **K2 (100-Frame Context)** — Precomputes 100-frame feature tensors and tests temporal context extension.

**Go/No-Go Decision Gate**: Any K variant achieving **$> 83.60\%$ LOLO Mean F1** or **$\sigma < \pm 5.74\%$** advances to the final multi-modal fusion architecture.
