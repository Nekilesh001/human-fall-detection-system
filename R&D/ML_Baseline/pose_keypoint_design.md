# Research Design: Le2i Pose Keypoint-Based Fall Detection (Experiment E)

> [!IMPORTANT]
> **DESIGN ONLY — NO TRAINING PERFORMED YET — NO FULL PRECOMPUTATION PERFORMED.**
> This document specifies the scientific protocol, landmark normalization formulas, model variants (E1, E2, E3), parameter counts, and evaluation metrics for Experiment E: Pose/Keypoint-Based Fall Detection.

---

## 1. Objective
To evaluate whether explicit human body pose keypoint geometry and joint kinematics eliminate background scene dependence and improve cross-location fall detection performance on the Le2i dataset by evaluating three controlled pose model variants under the exact same 4-Fold Leave-One-Location-Out (LOLO) cross-validation protocol.

---

## 2. Scientific Question
*"Does extracting 2D human pose geometry and motion kinematics via MediaPipe Pose eliminate spatial scene/background illumination dependence and outperform the canonical RGB baseline ($71.53\%$ F1) on unseen physical locations?"*

### Primary Research Hypotheses
1. **Hypothesis E1 (Background Invariance)**: Human body keypoints isolate body geometry and motion, discarding static room furniture, reflections, and wall contrast.
2. **Hypothesis E2 (Kinematic Velocity Benefit)**: Frame-to-frame joint velocity vectors ($d\hat{x}, d\hat{y}$) and vertical hip descent speed provide unambiguous physical downward acceleration cues during a fall.
3. **Hypothesis E3 (Low Illumination & Occlusion Challenge)**: Severe room dimness or furniture occlusions (such as in `Home_02`) reduce keypoint detection rates, requiring robust zero-padding and confidence weighting.

---

## 3. MediaPipe Pose Landmark Normalization Strategy

For each frame, MediaPipe Pose extracts **33 2D body landmarks**:
$$(x_i, y_i, v_i), \quad i \in \{0, 1, \dots, 32\}$$
where $x_i, y_i \in [0, 1]$ are raw image-normalized coordinates and $v_i \in [0, 1]$ is landmark visibility confidence.

### Centering & Scale Normalization Formula
1. **Hip Midpoint Centering**:
   $$x_{\text{hip}} = \frac{x_{23} + x_{24}}{2}, \quad y_{\text{hip}} = \frac{y_{23} + y_{24}}{2}$$
   $$x'_i = x_i - x_{\text{hip}}, \quad y'_i = y_i - y_{\text{hip}}$$
2. **Torso Scale Normalization**:
   $$x_{\text{sh}} = \frac{x_{11} + x_{12}}{2}, \quad y_{\text{sh}} = \frac{y_{11} + y_{12}}{2}$$
   $$L_{\text{torso}} = \sqrt{(x_{\text{sh}} - x_{\text{hip}})^2 + (y_{\text{sh}} - y_{\text{hip}})^2} + \epsilon$$
   $$\hat{x}_i = \frac{x'_i}{L_{\text{torso}}}, \quad \hat{y}_i = \frac{y'_i}{L_{\text{torso}}}$$

---

## 4. Controlled Model Variants

All three variants process the exact same 1,396 preprocessed Le2i windows ($W=50, S=25, 25\text{ FPS}$).

```text
Model E1 (Pose Geometry):         (B, 50, 99)  ──► Mean+Std (198) ──► Linear(198→64) ──► ReLU ──► Dropout(0.5) ──► Linear(64→2) [12,866 params]
Model E2 (Pose + Velocity):       (B, 50, 165) ──► Mean+Std (330) ──► Linear(330→64) ──► ReLU ──► Dropout(0.5) ──► Linear(64→2) [21,314 params]
Model E3 (Pose Motion Geometry):  (B, 50, 173) ──► Mean+Std (346) ──► Linear(346→64) ──► ReLU ──► Dropout(0.5) ──► Linear(64→2) [22,338 params]
```

| Model Variant | Feature Input Vector per Frame | Frame Vector Dim | Window Representation | Classifier Architecture | Trainable Parameters | Role in Experiment E |
| :--- | :--- | :---: | :---: | :--- | :---: | :--- |
| **Model E1** | 33 Normalized Landmarks $(\hat{x}_i, \hat{y}_i, v_i)$ | **99-D** | $(B, 50, 99) \to (B, 198)$ | `Linear(198 -> 64) -> ReLU -> Dropout(0.5) -> Linear(64 -> 2)` | **12,866** | Static Pose Geometry |
| **Model E2** | Static Pose (99-D) + Joint Velocity $(d\hat{x}_i, d\hat{y}_i)$ (66-D) | **165-D** | $(B, 50, 165) \to (B, 330)$ | `Linear(330 -> 64) -> ReLU -> Dropout(0.5) -> Linear(64 -> 2)` | **21,314** | Pose + Kinematic Velocity |
| **Model E3** | E2 Vector (165-D) + Derived Physics Descriptors (8-D) | **173-D** | $(B, 50, 173) \to (B, 346)$ | `Linear(346 -> 64) -> ReLU -> Dropout(0.5) -> Linear(64 -> 2)` | **22,338** | Pose Motion Geometry |

### Derived Physics Descriptors (8-D in E3)
1. Hip Center Position $(x_{\text{hip}}, y_{\text{hip}})$ [2-D]
2. Shoulder Center Position $(x_{\text{sh}}, y_{\text{sh}})$ [2-D]
3. Torso Angle $\theta = \arctan2(y_{\text{sh}} - y_{\text{hip}}, x_{\text{sh}} - x_{\text{hip}})$ [1-D]
4. Body Bounding Box Aspect Ratio $AR = \frac{\max(y) - \min(y)}{\max(x) - \min(x) + \epsilon}$ [1-D]
5. Hip Vertical Velocity $v_{y,\text{hip}} = \frac{y_{\text{hip}}(t) - y_{\text{hip}}(t-1)}{\Delta t}$ [1-D]
6. Center-of-Mass Trajectory Distance [1-D]

---

## 5. Dataset Scope & 4-Fold LOLO Protocol

- **Dataset Scope**: 1,396 windows across 127 verified supervised videos (96 FALL, 31 NORMAL).
- **Excluded Scope**: All 63 UNKNOWN records remain 100% EXCLUDED.
- **Folds**:
  - Fold 1: Outer Test = `Coffee_room_01` (894 train wins, 502 test wins)
  - Fold 2: Outer Test = `Coffee_room_02` (986 train wins, 410 test wins)
  - Fold 3: Outer Test = `Home_01` (1,157 train wins, 239 test wins)
  - Fold 4: Outer Test = `Home_02` (1,151 train wins, 245 test wins)

---

## 6. Inner Validation, Class Weighting & Threshold Strategy

- **Inner Event Split**: Inner Train (80% of outer train events) and Inner Validation (20% of outer train events) constructed with zero video event overlap ($\text{Inner Train Events} \cap \text{Inner Val Events} = \emptyset$).
- **Outer Test Isolation**: Outer held-out test location is 100% unseen (0 test windows used for training, class weights, checkpoint selection, or threshold tuning).
- **Class Weights**: Calculated programmatically per fold from outer training location windows.
- **Threshold Selection**: Inner validation threshold $\tau^*_{\text{inner}}$ searched in $[0.05, 0.95]$ to maximize Inner Validation F1.

---

## 7. Expected Results Artifacts & Output Directories

- Feature precomputation path: `processed_data/Le2i_baseline/pose_features/`.
- Checkpoints path: `checkpoints/le2i_pose_keypoint/{e1_pose, e2_vel, e3_physics}/fold_{i}_best.pth`.
- Results CSV path: `R&D/ML_Baseline/results/le2i_pose_keypoint/pose_fold_results.csv`.
- Canonical research report artifact: [`R&D/ML_Baseline/pose_keypoint_training_report.md`](file:///d:/ONE_DATA/Fall%20detection/R&D/ML_Baseline/pose_keypoint_training_report.md).
