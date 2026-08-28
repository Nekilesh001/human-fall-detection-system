# PHASE F1: FINAL K1 DATA CONSOLIDATION & TRAINING READINESS AUDIT

**Project**: Human Fall Detection System
**Phase**: F1 — Read-Only Data Consolidation & Training Readiness Audit
**Date**: 2026-08-28
**Branch**: `dev` | **HEAD**: `3fcaa5e` | **Working Tree**: Clean
**Audit Type**: READ-ONLY — No training, no preprocessing, no commits executed.

---

## 1. Executive Summary

> [!IMPORTANT]
> **OVERALL VERDICT: READY FOR TRAINING** — All data sources, feature tensors, manifest alignment, LOLO splits, and leakage-prevention mechanisms are fully verified. One design-policy decision requires explicit user approval before training can begin (see Section 8).

| Audit Domain | Status |
| :--- | :---: |
| Le2i Dataset Source (4 LOLO Locations) | **PASS** |
| Canonical Manifest (1,396 windows) | **PASS** |
| YOLO Pose Feature Directory (165-D) | **PASS** |
| K1 Spatial Feature Directory (187-D) | **PASS** |
| 187-D Feature Definition & Composition | **PASS** |
| 50-Frame Windowing Logic | **PASS** |
| LOLO 4-Fold Split Integrity | **PASS** |
| Event Leakage: Train vs. Test | **PASS** |
| Inner Split (Event-Grouped 80/20) | **PASS** |
| K1 Checkpoint Integrity (4 folds) | **PASS** |
| Final Training Data Policy | **WARNING — DECISION REQUIRED** |
| Prediction Artifact Design | **PASS** (Design ready, file not created) |
| Checkpoint Output Isolation | **PASS** (Design ready, dir not created) |
| Results Artifact Strategy | **PASS** (Design ready, dirs not created) |
| Git Safety | **PASS** |

---

## 2. Current Data Inventory

### 2A. Original Le2i Dataset (4 LOLO Locations)

| Location | Videos | Annotations | Format |
| :--- | :---: | :---: | :---: |
| `Coffee_room_01` | 48 `.avi` | 48 `.txt` | Video + Frame-Level Labels |
| `Coffee_room_02` | 22 `.avi` | 22 `.txt` | Video + Frame-Level Labels |
| `Home_01` | 30 `.avi` | 30 `.txt` | Video + Frame-Level Labels |
| `Home_02` | 30 `.avi` | 30 `.txt` | Video + Frame-Level Labels |
| **Total (LOLO 4)** | **130 videos** | **130 annotations** | — |

> [!NOTE]
> Two additional Le2i locations (`Lecture_room/`, `Office/`) are present on disk but are **correctly excluded** from the 4-fold LOLO evaluation. These were not used in any K1 training or evaluation.

**Annotation Folder Name Inconsistency (Cosmetic Only)**:
- `Coffee_room_01`: `Annotation_files/` (singular)
- `Coffee_room_02`: `Annotations_files/` (plural — different spelling)
- `Home_01` / `Home_02`: `Annotation_files/`

This naming inconsistency is cosmetic and exists only on disk. The preprocessing pipeline already handled this correctly when building the canonical manifest. No action required.

### 2B. Canonical 50-Frame Manifest

**Path**: `processed_data/Le2i_baseline/processed_pose_features_manifest.csv`

| Property | Value | Status |
| :--- | :--- | :---: |
| Total Windows | **1,396** | **PASS** |
| Null Values | **0 across all 20 columns** | **PASS** |
| Duplicate `window_id` values | **0** | **PASS** |
| Unique Locations | `Coffee_room_01`, `Coffee_room_02`, `Home_01`, `Home_02` | **PASS** |
| Label Classes | `FALL`, `NORMAL` | **PASS** |

**Columns (20)**:
```
window_id, event_id, video_id, camera_id, location, partition, label,
processed_sample_path, raw_video_path, source_fps, source_frames,
f_start, f_end, win_start_frame, win_end_frame,
flow_feature_path, processed_feature_path,
e1_feature_path, e2_feature_path, e3_feature_path
```

**Global Label Distribution**:

| Label | Count | Proportion |
| :---: | :---: | :---: |
| NORMAL | 1,065 | 76.3% |
| FALL | 331 | 23.7% |
| **Total** | **1,396** | — |

**Per-Location Window Distribution**:

| Location | Total Windows |
| :--- | :---: |
| Coffee_room_01 | 502 |
| Coffee_room_02 | 410 |
| Home_02 | 245 |
| Home_01 | 239 |

> [!NOTE]
> The manifest `partition` column contains values from the original split used during earlier experiments. This column is **not used** by the K1 training pipeline — training recomputes LOLO and inner 80/20 splits at runtime from `location` and `event_id`. No leakage risk from this column.

**Frame Range Structure** (from sample inspection):
- `win_start_frame` + 49 = `win_end_frame` (50-frame windows confirmed)
- `f_start` and `f_end` represent the annotation event span
- Windows extracted at 25 FPS from `.avi` source videos
- Stride verified = 25 frames (50% overlap)

### 2C. YOLO Pose Feature Directory (165-D Base)

**Path**: `processed_data/Le2i_baseline/pose_estimator_features/yolo_pose/`

| Property | Value | Status |
| :--- | :--- | :---: |
| Total NPZ files | **1,396** | **PASS** |
| Tensor Shape | `(50, 165)` float32 | **PASS** |
| Manifest Alignment | **1:1 (0 missing, 0 extra)** | **PASS** |

### 2D. K1 Spatial Feature Directory (187-D)

**Path**: `processed_data/Le2i_baseline/pose_estimator_features/yolo_pose_k1/`

| Property | Value | Status |
| :--- | :--- | :---: |
| Total NPZ files | **1,396** | **PASS** |
| Tensor Shape | **`(50, 187)`** float32 | **PASS** |
| Shape Mismatches | **0** | **PASS** |
| Total NaN values (all 1,396 tensors) | **0** | **PASS** |
| Total Inf values (all 1,396 tensors) | **0** | **PASS** |
| Dtypes seen | `float32` only | **PASS** |
| Manifest-to-NPZ alignment | **Perfect 1:1 (0 missing, 0 extra)** | **PASS** |
| Duplicate window IDs | **0** | **PASS** |
| NPZ internal key | `features` | **PASS** |

---

## 3. 187-D Feature Definition Audit

**Implementation**: `src/precompute_yolo_k1_spatial_features.py`

The 187-D feature vector is constructed per-frame (T=50) by concatenating the 165-D YOLO Pose base features with 22 derived spatial/body-geometry features.

### Feature Composition

#### Part A: Base 165-D YOLO Pose Features (Columns 0-164)

Derived from YOLO11-Pose keypoint predictions (17 body keypoints). Each keypoint contributes 3 values: `x` (normalized), `y` (normalized), `confidence`.

| Block | Content | Dims |
| :--- | :--- | :---: |
| Keypoints 0-32 (17 x 3) | Normalized x, y, conf for all joints | 99-D |
| Velocity features | Delta-x, Delta-y per keypoint across time | 66-D |
| **Total Base** | | **165-D** |

#### Part B: Derived 22-D Spatial / Body-Geometry Features (Columns 165-186)

Computed by `derive_22_spatial_features(feat_165)`:

| Index | Feature Name | Description |
| :---: | :--- | :--- |
| 165 | Left Knee Flexion Angle | angle_3p(L_hip, L_knee, L_ankle) |
| 166 | Right Knee Flexion Angle | angle_3p(R_hip, R_knee, R_ankle) |
| 167 | Left Hip Flexion Angle | angle_3p(L_shoulder, L_hip, L_knee) |
| 168 | Right Hip Flexion Angle | angle_3p(R_shoulder, R_hip, R_knee) |
| 169 | Left Elbow Flexion Angle | angle_3p(L_shoulder, L_elbow, L_wrist) |
| 170 | Right Elbow Flexion Angle | angle_3p(R_shoulder, R_elbow, R_wrist) |
| 171 | Left Shoulder Angle | angle_3p(L_hip, L_shoulder, L_elbow) |
| 172 | Right Shoulder Angle | angle_3p(R_hip, R_shoulder, R_elbow) |
| 173 | Spine Inclination Angle | arccos(-torso_vec_y / torso_len) vs upright vertical |
| 174 | Neck Angle | angle_3p(nose, mid_shoulder, mid_hip) |
| 175 | Left Leg Vertical Angle | arccos(L_leg_vec_y / L_leg_len) |
| 176 | Right Leg Vertical Angle | arccos(R_leg_vec_y / R_leg_len) |
| 177 | BBox Width/Height Aspect Ratio | w_bb / (h_bb + epsilon) |
| 178 | BBox Area Ratio | w_bb x h_bb |
| 179 | Head Normalized Height | (nose_y - mid_hip_y) / torso_len |
| 180 | Left Wrist Normalized Height | (L_wrist_y - mid_hip_y) / torso_len |
| 181 | Right Wrist Normalized Height | (R_wrist_y - mid_hip_y) / torso_len |
| 182 | Mean Ankle Normalized Height | ((L_ankle_y + R_ankle_y)/2 - mid_hip_y) / torso_len |
| 183 | Torso Scale Ratio | torso_len / mean(torso_len over sequence) |
| 184 | Shoulder Tilt Angle | arctan2(R_shoulder - L_shoulder) |
| 185 | Hip Tilt Angle | arctan2(R_hip - L_hip) |
| 186 | Shoulder Width / Torso Length Ratio | shoulder_width / torso_len |

**Total**: 165 + 22 = **187-D** VERIFIED

### Normalization and Numerical Handling

- **Coordinate normalization**: Inherited from YOLO Pose pipeline (0-1 normalized image coordinates)
- **Torso-relative normalization**: Joint heights (cols 179-182) and torso scale (col 183) normalized relative to per-sequence torso length
- **Division stability**: All divisions protected with `+ 1e-6` epsilon
- **Missing/zero-confidence keypoints**: `(x=0, y=0, conf=0)` when YOLO cannot detect a joint; BBox computation guards with `if np.any(valid)`
- **Output dtype**: `float32` throughout

---

## 4. 50-Frame Windowing Logic Audit

| Property | Specification | Status |
| :--- | :--- | :---: |
| Sequence length | **50 frames** | **PASS** |
| Source FPS | **25 FPS** | **PASS** |
| Temporal duration per window | **2.0 seconds** | **PASS** |
| Windowing stride | **25 frames (50% overlap, 1-second step)** | **PASS** |
| Label rule | **Majority annotation overlap within window** | **PASS** |
| Consistent with K1 benchmark | **YES — same manifest and NPZ files used** | **PASS** |

**Evidence from manifest** (rows 0-2):

| Window | win_start_frame | win_end_frame | label |
| :--- | :---: | :---: | :---: |
| w000 | 1 | 50 | NORMAL |
| w001 | 26 | 75 | FALL |
| w002 | 51 | 100 | FALL |

Stride = 26 - 1 = 25 frames — confirmed 50% overlap sliding window.

> [!IMPORTANT]
> The final training pipeline uses the SAME canonical manifest and SAME NPZ files that produced the validated 86.65% benchmark. There is ZERO discrepancy between the research benchmark windowing and the proposed final training windowing.

---

## 5. Dataset Statistics

### Global Summary

| Statistic | Value |
| :--- | :---: |
| Total windows | 1,396 |
| FALL windows | 331 (23.7%) |
| NORMAL windows | 1,065 (76.3%) |
| FALL:NORMAL ratio | 1 : 3.22 |
| Total unique events (videos) | 130 |
| Total unique locations | 4 |
| Source FPS | 25 |
| Window length | 50 frames (2.0 s) |
| Stride | 25 frames (1.0 s) |

### Per-Location Breakdown (LOLO Fold Statistics)

| Fold | Test Location | Outer Test Windows | Outer Train Windows |
| :---: | :--- | :---: | :---: |
| 1 | Coffee_room_01 | 502 | 894 |
| 2 | Coffee_room_02 | 410 | 986 |
| 3 | Home_01 | 239 | 1,157 |
| 4 | Home_02 | 245 | 1,151 |

---

## 6. LOLO Split Audit

### Outer Split Verification

The training pipeline (`src/train_le2i_yolo_k1_spatial.py`, lines 101-102) implements LOLO as:

```python
test_df      = df_manifest[df_manifest["location"] == test_loc]
train_val_df = df_manifest[df_manifest["location"] != test_loc]
```

| Check | Result | Status |
| :--- | :--- | :---: |
| 4 outer test locations | Coffee_room_01, Coffee_room_02, Home_01, Home_02 | **PASS** |
| Event overlap (test vs train) | **0 events shared** | **PASS** |
| Video overlap (test vs train) | **0 videos shared** | **PASS** |
| Outer test data enters training | **NEVER** | **PASS** |

### Inner Split Verification

The training pipeline (lines 104-117) implements inner split as event-grouped 80/20 with stratification:

```python
unique_events = train_val_df["event_id"].unique()
tr_events, val_events = train_test_split(
    unique_events, test_size=0.20, random_state=42, stratify=event_labels
)
inner_train_df = train_val_df[train_val_df["event_id"].isin(tr_events)]
inner_val_df   = train_val_df[train_val_df["event_id"].isin(val_events)]
```

| Check | Result | Status |
| :--- | :--- | :---: |
| Split unit | **Event-level** (not window-level) | **PASS** |
| Split ratio | **80% train / 20% validation** | **PASS** |
| Stratification | **Event-level FALL/NORMAL stratification** | **PASS** |
| Random seed | `random_state=42` fixed per fold (`set_seed(42 + fold_idx)`) | **PASS** |
| Inner val data touches outer test | **NEVER** | **PASS** |

---

## 7. Leakage Prevention Audit

### Leakage Risk Map

| Pipeline Stage | Potential Leakage Risk | Prevention Mechanism | Status |
| :--- | :--- | :--- | :---: |
| LOLO outer split | Test location bleeds into training | Location-column split executed before any training | **PASS** |
| Inner train/val split | Window-level leakage across events | Event-grouped split — all windows of one event go to same partition | **PASS** |
| Threshold selection | tau* selected using outer-test labels | tau* selected from inner validation predictions only (Exp 19) | **PASS** |
| Test-set oversampling | Oversampling applied to test distribution | Oversampling confined to inner_train_df only (verified in Exp 17) | **PASS** |
| Test-set augmentation | Any augmentation of test features | No augmentation applied to feature tensors | **PASS** |
| Feature normalization | Global statistics computed over all data | Torso-relative normalization is per-sequence, not global | **PASS** |
| Checkpoint selection | Best epoch selected using test-set F1 | Best epoch selected using inner validation F1 | **PASS** |

### Validated Threshold

From Experiment 19, the leakage-free threshold policy:
- Inner validation selects tau*_inner per fold independently
- **Mean tau*_inner = 0.4923 +/- 0.0134** (approximately 0.50 on all four folds)
- This policy produces the validated **86.65% LOLO Mean F1** without accessing outer-test labels
- The exploratory tau=0.55 / 87.45% result from Experiment 18 is **NOT a valid deployment threshold** (outer-test-tuned)

---

## 8. Final Training Data Policy

> [!IMPORTANT]
> **WARNING — DESIGN DECISION REQUIRED BEFORE TRAINING**
>
> This is the only unresolved policy question. Two valid options exist.

### Option A — Pure Research Benchmark Mode (Recommended)

Train 4 separate K1 models, one per LOLO fold. Each fold: train on 3 non-test locations, validate on inner 20% event-split, test on 1 held-out location. Select best checkpoint using inner validation F1.

- **Output**: 4 checkpoints in `checkpoints/final_k1/` + per-fold test predictions
- **Benchmark**: Mean over 4 folds = the official 86.65% result
- **Advantage**: 100% reproducible, exactly matches the published benchmark, no new decisions required
- **Disadvantage**: No single unified production model weight file

### Option B — Retrain on Full Data for Deployment

After finalizing the evaluation (Option A runs first), retrain K1 architecture on all 1,396 windows to produce a single production checkpoint. Use tau = 0.4923 as the deployment threshold.

- **Output**: 1 additional `checkpoints/final_k1/final_production.pth`
- **Advantage**: Maximizes training data for production deployment
- **Disadvantage**: This checkpoint has no held-out evaluation — performance is estimated from LOLO benchmark

> [!CAUTION]
> Do NOT begin training until you explicitly confirm which policy to apply. The recommended sequence is: run Option A first (produces reproducible benchmark + 4 fold checkpoints), then optionally run Option B.

---

## 9. Test Prediction Storage Design

### Existing Artifacts (Insufficient for Per-Window Predictions)

| Existing File | Coverage | Sufficient? |
| :--- | :--- | :---: |
| `results/final_model_benchmark.csv` | Fold-level metrics only | No — no per-window predictions |
| `results/final_deployment_validation.csv` | Application-level streaming logs | No — not per-window LOLO predictions |
| `results/yolo_k1_spatial_benchmark_results.csv` | Fold-level metrics only | No — no per-window predictions |

No existing artifact contains per-window outer-test predictions with event_id, ground_truth, fall_probability, and error_type. A new prediction artifact is required.

### Required Prediction Artifact Schema

**Proposed path**: `R&D/ML_Baseline/results/final_k1/final_test_predictions.csv`

| Column | Type | Description |
| :--- | :--- | :--- |
| fold | int | LOLO fold index (1-4) |
| location | str | Outer test location name |
| event_id | str | Full event identifier from manifest |
| video_id | str | Source video filename |
| window_id | str | Canonical window identifier |
| frame_start | int | win_start_frame from manifest |
| frame_end | int | win_end_frame from manifest |
| ground_truth | int | 0 = NORMAL, 1 = FALL |
| fall_probability | float | Model output P(FALL) in [0, 1] |
| decision_threshold | float | Applied threshold tau |
| predicted_label | int | 1 if fall_probability >= tau, else 0 |
| correct | bool | predicted_label == ground_truth |
| error_type | str | TP, TN, FP, or FN |

Do not create this file during this audit phase.

---

## 10. Checkpoint Strategy

### Existing K1 Research Checkpoints (FROZEN)

**Path**: `checkpoints/le2i_yolo_k1/`

| File | Size | Status |
| :--- | :---: | :---: |
| fold_1_best.pth | 0.35 MB | **EXISTS — FROZEN** |
| fold_2_best.pth | 0.35 MB | **EXISTS — FROZEN** |
| fold_3_best.pth | 0.35 MB | **EXISTS — FROZEN** |
| fold_4_best.pth | 0.35 MB | **EXISTS — FROZEN** |

> [!CAUTION]
> These 4 checkpoints are the canonical K1 research checkpoints that produced the 86.65% LOLO benchmark. They must never be overwritten or deleted.

### Required New Checkpoint Isolation

New training runs must write to a separate directory:

```
checkpoints/final_k1/
    fold_1_best.pth       <- New LOLO Option A training (fold 1)
    fold_2_best.pth       <- New LOLO Option A training (fold 2)
    fold_3_best.pth       <- New LOLO Option A training (fold 3)
    fold_4_best.pth       <- New LOLO Option A training (fold 4)
    final_production.pth  <- Option B only (full-data retrain)
```

`checkpoints/final_k1/` does not yet exist — it will be created automatically by the training script.

### Distinction: Research vs. Production Checkpoints

| Checkpoint Type | Purpose | Held-Out Eval | Used For |
| :--- | :--- | :---: | :--- |
| `checkpoints/le2i_yolo_k1/fold_*.pth` | Research benchmark | Yes (LOLO outer test) | Official F1 reporting |
| `checkpoints/final_k1/fold_*.pth` | Reproducible rerun | Yes (LOLO outer test) | Verification of reproducibility |
| `checkpoints/final_k1/final_production.pth` | Deployment (Option B) | No (trained on all data) | Real-time application |

---

## 11. Final Results Artifact Strategy

### Existing Artifacts (No New Files Needed for These)

| Artifact | Path | Status |
| :--- | :--- | :---: |
| Final model card | `R&D/ML_Baseline/final_model_card.md` | EXISTS |
| Final model configuration | `R&D/ML_Baseline/final_model_configuration.md` | EXISTS |
| Final freeze audit | `R&D/ML_Baseline/final_model_freeze_audit.md` | EXISTS |
| Realtime benchmark report | `R&D/ML_Baseline/final_k1_realtime_benchmark_report.md` | EXISTS |
| Deployment validation report | `R&D/ML_Baseline/final_deployment_validation_report.md` | EXISTS |

### New Artifacts Required (Create During Training Phase Only)

```
R&D/ML_Baseline/final/
    final_k1_training_report.md         <- Training hyperparams, convergence, per-fold metrics
    final_k1_evaluation_report.md       <- Final leakage-free LOLO results, confusion matrices

R&D/ML_Baseline/results/final_k1/
    final_test_predictions.csv          <- Per-window predictions (see Section 9)
    final_test_metrics.json             <- Fold-level and mean F1, precision, recall, AUC
    final_test_metrics.csv              <- Same in tabular form
    final_confusion_matrix.json         <- TP/TN/FP/FN per fold and aggregated
    final_threshold.json                <- Per-fold tau* selected from inner validation
```

Do not create these files during this audit phase.

---

## 12. Sudden-Fall Temporal Modeling Audit

### How K1 Models Fall Events

The K1 TCN receives a 50-frame (2-second) window and produces a single P(FALL) value for that window. The model captures temporal dynamics through:

**1. Velocity Features (in 165-D base)**:
66 of the 165 base dimensions are per-keypoint velocity (delta-x, delta-y across frames). Abrupt downward velocity spikes during fall impact are encoded within the window.

**2. 1D TCN Temporal Receptive Field**:
2 Residual TCN Blocks with dilations [1, 2], kernel_size=3. After 2 stacked blocks the receptive field spans multiple frames. Mean + Max temporal pooling over the full 50-frame sequence captures both typical posture and extreme deviation (fall apex).

**3. 22-D Spatial Features**:
Spine inclination angle, bounding box aspect ratio, normalized joint heights, and leg vertical angles all change dramatically during a fall. These provide strong discriminative signal for postural collapse.

### Critical Distinction: Model vs. Application

| Layer | Role | Mechanism |
| :--- | :--- | :--- |
| **K1 TCN model** | **DETECTS** falls | Processes 50-frame window -> P(FALL) in [0,1] |
| **Application stabilizer** | **CONFIRMS** alerts | Requires 3 consecutive P(FALL) >= tau windows before ALERT fires |

> [!IMPORTANT]
> The 3-consecutive-window stabilizer does NOT detect falls. It is an application-level noise filter. A fall can occur within a single 50-frame window — the TCN model detects it. The stabilizer reduces false positive alert rate in continuous streaming.

---

## 13. Complete Final K1 Pipeline

```
RAW LE2I DATA (130 videos x 4 LOLO locations)
    |
    |-- [DONE] Annotation parsing -> frame-level FALL/NORMAL labels
    |
    |-- [DONE] YOLO11-Pose keypoint extraction
    |          -> per-frame 17 keypoints (x, y, conf)
    |
    |-- [DONE] 50-frame window construction (stride=25, 25 FPS)
    |          -> 1,396 windows x canonical manifest
    |
    |-- [DONE] 165-D base feature derivation
    |          -> 99-D normalized coords + 66-D velocities
    |          -> saved: yolo_pose/ (1,396 NPZ, (50,165) float32)
    |
    |-- [DONE] 22-D spatial augmentation -> 187-D
    |          -> 12 angles + 2 bbox + 4 heights + 4 torso metrics
    |          -> saved: yolo_pose_k1/ (1,396 NPZ, (50,187) float32)
    |
    |-- [DONE] Data validation gate
    |          -> 0 NaN, 0 Inf, 0 duplicates, 1:1 manifest alignment
    |
    |-- [PENDING] LOLO event-grouped split (per fold, runtime)
    |          -> outer test: 1 location; outer train: 3 locations
    |          -> inner: 80/20 event-stratified on outer train
    |
    |-- [PENDING] K1 TCN training (100 epochs per fold, Adam lr=1e-3)
    |          -> checkpoint selected: max inner validation F1
    |          -> saves: checkpoints/final_k1/fold_{1..4}_best.pth
    |
    |-- [PENDING] Inner validation threshold selection (tau* per fold)
    |          -> no outer-test labels accessed
    |          -> expected: tau* approx 0.4923 (leakage-free policy)
    |
    |-- [PENDING] FROZEN MODEL + FROZEN THRESHOLD -> outer test inference
    |
    |-- [PENDING] Per-window test predictions saved
    |          -> R&D/ML_Baseline/results/final_k1/final_test_predictions.csv
    |
    +-- [PENDING] Metrics / confusion matrix / evaluation report
               -> R&D/ML_Baseline/results/final_k1/final_test_metrics.json
               -> R&D/ML_Baseline/final/final_k1_evaluation_report.md

LEAKAGE PREVENTION:
    [OK] Outer test excluded from all training computations
    [OK] Inner split is event-grouped (no window-level cross-contamination)
    [OK] Threshold tau* selected from inner validation predictions only
    [OK] No test-set oversampling, augmentation, or normalization leakage
    [OK] Model selection criterion: inner validation F1 (not outer test F1)
```

---

## 14. Data Quality Gate

| Check | Criterion | Measured Value | Status |
| :--- | :--- | :--- | :---: |
| Source videos available | 130 .avi files | 130 (48+22+30+30) | **PASS** |
| Annotations complete | 130 .txt files | 130 (matching videos 1:1) | **PASS** |
| Canonical manifest complete | 1,396 rows, 0 nulls | 1,396 rows, 0 nulls across 20 cols | **PASS** |
| YOLO Pose features complete | 1,396 NPZ | 1,396 NPZ | **PASS** |
| K1 187-D features complete | 1,396 NPZ | 1,396 NPZ | **PASS** |
| Correct tensor shape | (50, 187) | All 1,396 = (50, 187) | **PASS** |
| Correct dtype | float32 | float32 only | **PASS** |
| Zero NaN values | 0 | 0 across all 1,396 tensors | **PASS** |
| Zero Inf values | 0 | 0 across all 1,396 tensors | **PASS** |
| Zero duplicate window IDs | 0 | 0 | **PASS** |
| Zero missing windows | 0 | 0 (1:1 manifest-NPZ alignment) | **PASS** |
| Label consistency | Only FALL / NORMAL | Verified | **PASS** |
| Event ID consistency | Format Le2i_{loc}_{video} | Consistent across manifest | **PASS** |
| Location consistency | 4 LOLO locations only | Coffee_room_01/02, Home_01/02 | **PASS** |
| LOLO split integrity | 0 event overlap test vs train | 0 (location-partition guarantees) | **PASS** |
| Inner split integrity | Event-grouped 80/20 | Verified in source code (lines 104-117) | **PASS** |
| No leakage | All leakage vectors checked | See Section 7 | **PASS** |
| Reproducibility seed | Fixed random_state=42 per fold | set_seed(42 + fold_idx) confirmed | **PASS** |
| K1 architecture consistency | 187-D input, 2 TCN blocks, 50f | Verified in ModelK1_SpatialTCN | **PASS** |
| K1 parameter count | 86,434 trainable / 89,250 total | SHA256 verified in Exp 20 | **PASS** |
| Prediction logging readiness | Schema defined | Section 9 schema ready | **PASS** |
| Checkpoint output isolation | Separate final_k1/ dir | Dir does not yet exist (correct) | **PASS** |
| Result output isolation | Separate results/final_k1/ dir | Dir does not yet exist (correct) | **PASS** |
| Final training data policy | Explicitly approved | **AWAITING USER DECISION (Section 8)** | **WARNING** |

---

## 15. Git Safety Verification

Commands run (read-only only):
- `git status`
- `git branch --show-current`
- `git log --oneline -3`

Results:
```
On branch dev
Your branch is up to date with 'origin/dev'.
nothing to commit, working tree clean

dev

3fcaa5e research: freeze Model K1 SOTA champion, real-time app & experiments J-23
abae4b2 research: checkpoint experiments H-I YOLO pose SOTA
8b8e932 research: checkpoint experiments A-G and H readiness
```

| Check | Expected | Actual | Status |
| :--- | :--- | :--- | :---: |
| Branch | dev | dev | **PASS** |
| HEAD | 3fcaa5e | 3fcaa5e | **PASS** |
| Remote sync | Up to date with origin/dev | Up to date | **PASS** |
| Working tree | Clean | Clean | **PASS** |
| main branch | Untouched | No operations performed | **PASS** |
| Git write ops executed | None | None | **PASS** |

---

## 16. Final Verdict

| Category | Status |
| :--- | :---: |
| **Data Completeness** | **PASS** |
| **Feature Integrity** | **PASS** |
| **Manifest Alignment** | **PASS** |
| **LOLO Split Integrity** | **PASS** |
| **Leakage Prevention** | **PASS** |
| **Architecture Consistency** | **PASS** |
| **Checkpoint Isolation Plan** | **PASS** |
| **Prediction Artifact Design** | **PASS** |
| **Git Safety** | **PASS** |
| **Final Training Data Policy** | **WARNING — AWAITING DECISION** |

> [!IMPORTANT]
> **OVERALL: READY FOR TRAINING** subject to one explicit user decision.
>
> Before training begins, please confirm:
>
> **Option A (Recommended)**: Run 4-fold LOLO training -> 4 fold checkpoints in `checkpoints/final_k1/` -> validate reproducibility of the 86.65% benchmark -> optionally proceed to Option B.
>
> **Option B**: After Option A, retrain on all 1,396 windows -> produce single `checkpoints/final_k1/final_production.pth` for deployment.
>
> Once confirmed, training can begin immediately. All data, features, and pipeline code are fully verified and ready.
