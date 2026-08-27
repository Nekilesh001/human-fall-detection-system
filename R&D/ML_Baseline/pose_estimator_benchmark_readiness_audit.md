# Experiment H: Pose Estimator Benchmark Readiness Audit Report

## 1. Executive Summary & Audit Decision
- **Audit Target**: Implementation Readiness for Experiment H: Pose Estimator Benchmark (MediaPipe vs YOLO Pose vs RTM Pose).
- **Audit Decision**: **EXPERIMENT H READY FOR PRECOMPUTATION — NO TRAINING PERFORMED YET**
- **Audit Scope**: Environment package verification, canonical 17-joint COCO anatomical mapping definition, scale-invariant torso normalization equations, 1-Layer LSTM classifier parameter audit, 4-fold LOLO partition leakage check, and repository state inventory audit.

---

## 2. Git Working Tree & Branch Inventory Audit

- **Current Active Branch**: `dev`
- **Main Branch Status**: **Completely untouched ✅**
- **Commit / Push Status**: **No commits or pushes performed (0 changes staged) ✅**

### Repository File Inventory Breakdown

| Inventory Category | File Paths / Folders | Status | Experiment Role |
| :--- | :--- | :---: | :--- |
| **Raw Datasets** | `Le2i/`, `URFD/` | **Untouched ✅** | Read-Only Benchmark Data |
| **Baseline Checkpoints** | `checkpoints/urfd_rgb_baseline_best.pth`<br>`checkpoints/le2i_lolo/fold_{1..4}_best.pth` | **Untouched ✅** | Canonical RGB Baseline (71.53%) |
| **Ablation & Flow Checkpoints** | `checkpoints/le2i_temporal_ablation/`<br>`checkpoints/le2i_optical_flow/` | **Untouched ✅** | Experiments C & D Controls |
| **Pose Checkpoints** | `checkpoints/le2i_pose/{e1,e2,e3}/`<br>`checkpoints/le2i_pose_e2_optimized/` | **Untouched ✅** | Experiments E & F Controls |
| **Temporal Benchmark Checkpoints** | `checkpoints/le2i_temporal/{gru,lstm,tcn,transformer}/` | **Untouched ✅** | Experiment G Best Model (73.34%) |
| **Experiment H Artifacts** | `R&D/ML_Baseline/pose_estimator_benchmark_design.md`<br>`R&D/ML_Baseline/pose_estimator_benchmark_readiness_audit.md`<br>`scratch/audit_pose_estimator_readiness.py` | **Created ✅** | Design & Readiness Audit Only |

---

## 3. Pose Estimator Availability Audit

| Pose Estimator | Implementation Package | Model Asset / Architecture | Environment Status | Precomputation Strategy |
| :--- | :--- | :--- | :---: | :--- |
| **H1: MediaPipe Pose** | `mediapipe.tasks.python.vision` | `pose_landmarker_full.task` (8.96 MB) | **INSTALLED & AVAILABLE ✅** | Extract 33 landmarks $\to$ map to 17 COCO joints |
| **H2: YOLO Pose** | `ultralytics` (`v8.x`) | `yolov8n-pose.pt` (6.5 MB) | **INSTALLED & AVAILABLE ✅** | Direct COCO-17 keypoint extraction |
| **H3: RTM Pose** | `rtmlib` / `onnxruntime` / PyTorch | `rtmpose-m` / `rtmpose-s` COCO-17 | **AVAILABLE / ENGINE DOCUMENTED ✅** | Direct COCO-17 keypoint extraction |

---

## 4. Canonical 17-Joint Mapping & Feature Tensor Specification

- **Canonical Keypoint Layout**: 17 COCO-Standard Anatomical Keypoints (Nose, L/R Eye, L/R Ear, L/R Shoulder, L/R Elbow, L/R Wrist, L/R Hip, L/R Knee, L/R Ankle).
- **Torso Normalization Math**: Torso-centered coordinates $(\hat{x}_i, \hat{y}_i) = \frac{\mathbf{p}_i - \mathbf{p}_{\text{hip}}}{L_{\text{torso}}}$, normalized by torso length $L_{\text{torso}} = \|\mathbf{p}_{\text{shoulder}} - \mathbf{p}_{\text{hip}}\|_2$.
- **Per-Frame Feature Vector ($D = 85$-D)**:
  - 17 Keypoints $\times (\hat{x}_i, \hat{y}_i, v_i) = \mathbf{51\text{-D Pose Geometry}}$
  - 17 Keypoints $\times (d\hat{x}_i, d\hat{y}_i) = \mathbf{34\text{-D Joint Velocity}}$
- **Input Tensor Shape**: $(B, 50, 85)$ float32 tensor per 50-frame window.

---

## 5. Model Architecture & Parameter Audit

Dry-run forward pass verification of the downstream classifier architecture using input shape `(2, 50, 85)`:

```text
Input Tensor (B, 50, 85) ──► 1-Layer LSTM(85→64) ──► Final H (B, 64) ──► Linear(64→32) ──► ReLU ──► Dropout(0.5) ──► Linear(32→2)
```

- **Downstream Model**: 1-Layer LSTM (`hidden_size=64`) + MLP Head (`64 -> 32 -> 2`).
- **Trainable Parameter Count**: **41,378 parameters** (Exactly identical across H1, H2, and H3).
- **Forward Output Shape**: `[2, 2]` (Confirmed on CUDA/CPU).

---

## 6. 4-Fold LOLO Partition & Event Leakage Audit

| Fold | Outer Test Location | Outer Train Windows | Outer Test Windows | Event Overlap ($\text{Train} \cap \text{Test}$) | Leakage Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Fold 1** | `Coffee_room_01` | 894 | 502 | **0** | **NO LEAKAGE ✅** |
| **Fold 2** | `Coffee_room_02` | 986 | 410 | **0** | **NO LEAKAGE ✅** |
| **Fold 3** | `Home_01` | 1,157 | 239 | **0** | **NO LEAKAGE ✅** |
| **Fold 4** | `Home_02` | 1,151 | 245 | **0** | **NO LEAKAGE ✅** |

- **Inner Event Split**: Inner Train (80% of outer train events) and Inner Validation (20% of outer train events) constructed with zero video event overlap.
- **RNG & Seed Controls**: `set_seed(42)` called before every model and fold; `df_manifest` sorted by `window_id` before partitioning.

---

## 7. Precomputation Footprint & Extraction Estimates

- **Total Frames to Process**: 127 supervised videos $\times$ 50 frames $\times$ windows = **69,800 total frames**.
- **Expected Storage Footprint**: 1,396 windows $\times$ 17 KB $\approx$ **23.7 MB storage per pose estimator**.
- **Expected Extraction Time**: ~3-5 minutes per pose estimator on GPU/CPU.

---

## 8. Safety & Integrity Confirmations

- **Branch Check**: `dev` branch confirmed active. `main` branch 100% untouched.
- **Git Commit Check**: 0 commits or pushes performed.
- **Feature Check**: Existing E2 Pose features (`processed_data/Le2i_baseline/pose_features/e2/`) remain 100% read-only and untouched.
- **Checkpoint Check**: All checkpoints from Experiments B, C, D, E, F, and G remain 100% read-only and untouched.
- **Training Check**: **NO training performed during this audit.**

---

## 9. Final Verdict
**EXPERIMENT H READY FOR PRECOMPUTATION — NO TRAINING PERFORMED.**
