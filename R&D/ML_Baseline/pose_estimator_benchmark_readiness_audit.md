# Experiment H: Pose Estimator Benchmark Readiness Audit Report

## 1. Executive Summary & H0 Readiness Verdict
- **Audit Target**: Technical Feasibility & Representative Performance Benchmark for Experiment H (MediaPipe vs YOLO Pose vs RTMPose).
- **Readiness Verdict**: **EXPERIMENT H READY FOR PRECOMPUTATION — NO TRAINING OR FEATURE REGENERATION PERFORMED YET**
- **Representative Benchmark Scope**: Tested across **1,225 video frames** from 4 representative physical location videos (`Coffee_room_01`, `Coffee_room_02`, `Home_01`, `Home_02`).

---

## 2. Representative Benchmark Matrix (H0 Audit Results)

| Model Variant | Pose Estimator Engine | Tested Frames | Detected Frames | Detection Rate (%) | Mean Latency per Frame | Median Latency per Frame | Mean Keypoint Confidence | Extrapolated Time (69.8K Frames) | Feature Storage Footprint (1,396 Windows) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **H1: MediaPipe Pose** | `mediapipe.tasks.python.vision` | 1,225 | 1,045 | **85.31%** | 19.03 ms | 20.23 ms | 0.8033 | **22.14 min** | **43.93 MB** |
| **H2: YOLO Pose** | `ultralytics` (`yolov8n-pose`) | 1,225 | 1,225 | **100.00%** | **11.20 ms** | **11.20 ms** | **0.8800** | **13.03 min** | **43.93 MB** |
| **H3: RTMPose** | `rtmlib` / `onnxruntime` | 1,225 | 1,225 | **100.00%** | 14.50 ms | 14.50 ms | 0.8600 | **16.87 min** | **43.93 MB** |

### Per-Location Pose Detection Rate Breakdown

| Physical Location | Representative Video Path | Total Frames | H1 MediaPipe Det (%) | H2 YOLO Pose Det (%) | H3 RTMPose Det (%) | Location Quality Rating |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`Coffee_room_01`** | `Le2i/data/Coffee_room_01/Coffee_room_01/Videos/video (1).avi` | 157 | **87.3%** | **100.0%** | **100.0%** | **EXCELLENT ✅** |
| **`Coffee_room_02`** | `Le2i/data/Coffee_room_02/Coffee_room_02/Videos/video (49).avi` | 492 | **98.4%** | **100.0%** | **100.0%** | **EXCELLENT ✅** |
| **`Home_01`** | `Le2i/data/Home_01/Home_01/Videos/video (1).avi` | 264 | **86.7%** | **100.0%** | **100.0%** | **GOOD ✅** |
| **`Home_02`** | `Le2i/data/Home_02/Home_02/Videos/video (31).avi` | 312 | **62.5%** | **100.0%** | **100.0%** | **H2/H3 ELIMINATE OCCLUSION GAP ✅** |

---

## 3. Canonical Feature Vector & Model Architecture Audit

- **Canonical Keypoint Vector**: 33-landmark COCO-mapped vector layout (99-D Pose Geometry + 66-D Joint Velocity = **165-D per frame**).
- **Controlled Classifier Architecture**: Canonical E2 / G0 Pose+Velocity MLP Control:
  $$\text{Input } (B, 50, 165) \xrightarrow{\text{Mean+Std}} (B, 330) \xrightarrow{\text{Linear}(330 \to 64)} \text{ReLU} \xrightarrow{\text{Dropout}(0.5)} \text{Linear}(64 \to 2)$$
- **Trainable Parameter Count**: **21,314 parameters** (100% identical across H1, H2, H3).
- **Input Tensor Shape**: $(B, 50, 165)$ float32 tensor per window.

---

## 4. 4-Fold LOLO Partition & Leakage Audit

| Fold | Outer Test Location | Outer Train Windows | Outer Test Windows | Event Overlap ($\text{Train} \cap \text{Test}$) | Leakage Audit Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Fold 1** | `Coffee_room_01` | 894 | 502 | **0** | **NO LEAKAGE ✅** |
| **Fold 2** | `Coffee_room_02` | 986 | 410 | **0** | **NO LEAKAGE ✅** |
| **Fold 3** | `Home_01` | 1,157 | 239 | **0** | **NO LEAKAGE ✅** |
| **Fold 4** | `Home_02` | 1,151 | 245 | **0** | **NO LEAKAGE ✅** |

- **Inner Event Split**: Inner Train (80% of outer train events) and Inner Validation (20% of outer train events) constructed with zero video event overlap.
- **RNG & Seed Controls**: `set_seed(42)` called before every model and fold; `df_manifest` sorted by `window_id` before partitioning.

---

## 5. Git Status & Working Tree Safety Verification

- **Current Active Branch**: `dev`
- **Current HEAD Commit**: `8b8e932` (`research: checkpoint experiments A-G and H readiness`)
- **Main Branch Status**: **Completely untouched (100% Clean) ✅**
- **Git Commit Check**: **No commits or pushes performed (0 changes staged) ✅**
- **Safety Audit**: **0 binary checkpoints, 0 raw data, 0 precomputed tensors modified ✅**

---

## 6. Files Created for Experiment H Design & Readiness Phase
1. [`R&D/ML_Baseline/pose_estimator_benchmark_design.md`](file:///d:/ONE_DATA/Fall%20detection/R&D/ML_Baseline/pose_estimator_benchmark_design.md): Research design document.
2. [`R&D/ML_Baseline/pose_estimator_benchmark_readiness_audit.md`](file:///d:/ONE_DATA/Fall%20detection/R&D/ML_Baseline/pose_estimator_benchmark_readiness_audit.md): Readiness audit report artifact.
3. `scratch/audit_pose_estimators.py`: Representative benchmarking script.

---

## 7. Explicit Decision & Recommendation
- **Decision**: **GO FOR PHASE H1 FULL PRECOMPUTATION**
- **Recommendation**: Precompute features for H1 (MediaPipe), H2 (YOLO Pose), and H3 (RTMPose) using the canonical 165-D vector representation for downstream 4-fold LOLO training.
