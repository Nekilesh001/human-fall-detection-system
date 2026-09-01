# Phase H — Comprehensive Multi-Dataset Training Readiness Audit & Design Specification

> [!IMPORTANT]
> **IMMUTABLE BASELINE MANDATE & READ-ONLY AUDIT POLICY**  
> Baseline Model Path: `checkpoints/final_k1/final_production.pth`  
> Checkpoint SHA256: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d` (**IMMUTABLE & FROZEN**)  
> Operating Parameters: $\tau = 0.3650$, 50-frame buffer (2.0s context @ 25 FPS), 187-D spatial features, 3-consecutive-window confirmation.  
> Audit Policy: **Zero model retraining, zero dataset modifications, zero git commits, and zero production model changes performed.**

---

## 1. Current K1 Architecture
- **Model Type**: `ModelK1_SpatialTCN` (1D Residual Temporal Convolutional Network).
- **Parameters**: 89,250 parameters ($348.6 \text{ KB}$).
- **Structure**: 2 Residual 1D TCN Blocks (`channels=[64, 64]`, `kernel_size=3`, `dilations=[1, 2]`), Mean+Max temporal pooling, `Linear(128 -> 32)` $\to$ `ReLU` $\to$ `Dropout(0.5)` $\to$ `Linear(32 -> 2)`.
- **Receptive Field**: 50 frames ($2.0\text{s}$ context @ 25 FPS).
- **Decision Policy**: Softmax probability $P(\text{FALL}) \ge \tau = 0.3650$ with 3 consecutive window stabilization.

---

## 2. Current Preprocessing Pipeline
1. **Pose Observation**: YOLO11-Pose (`yolov8n-pose.pt` @ conf = 0.25) extracts 17 COCO keypoints mapped to 33 canonical landmarks.
2. **187-D Feature Construction**:
   - 99-D normalized keypoints ($33 \text{ joints} \times (x, y, v)$).
   - 66-D keypoint velocities ($33 \text{ joints} \times (\Delta x, \Delta y)$).
   - 22-D spatial body geometry (torso tilt, spine angle, aspect ratio, knee/elbow joint angles).
3. **Temporal Windowing**: Rolling 50-frame buffer (`stride = 25 frames` for training dataset generation).

---

## 3. Dataset Audit 1: Le2i (`Le2i/`)
- **Total Videos**: 190 videos across 4 environments (`Coffee_room_01`, `Home_01`, `Office_01`, `Lecture_room`).
- **Class Breakdown**: 96 Fall videos, 31 Normal ADL videos, 63 UNKNOWN/unannotated.
- **FPS & Resolution**: 25.0 FPS native, $320 \times 240$ resolution.
- **Annotations**: Text files (`.txt`) recording exact frame indices for fall start and fall end.
- **1,396-Window Baseline**: Confirmed correctly generated; 1,396 temporal windows ($50 \text{ frames}, \text{stride}=25$) extracted from Le2i.

---

## 4. Dataset Audit 2: URFD (`URFD/`)
- **Total Sequences**: 100 sequences (30 Fall events recorded via 2 synchronized cameras = 60 videos; 40 ADL normal sequences).
- **FPS & Resolution**: 30.0 FPS native, $640 \times 240$ resolution (dual RGB + Depth stream side-by-side).
- **Annotations**: CSV files containing frame-by-frame binary labels (`-1` = ADL, `1` = Fall).
- **Anomaly Found**: Duplicate annotation file `fall-11-data (1).csv` present alongside `fall-11-data.csv` (requires deduplication during manifest compilation).

---

## 5. Dataset Audit 3: Multicam Dataset (`dataset/`)
- **Total Videos**: 192 videos across 24 chute scenarios (`chute01` through `chute24`), each filmed simultaneously by 8 synchronized cameras (`cam1.avi` to `cam8.avi`).
- **FPS & Resolution**: 120.0 FPS high-speed capture, $720 \times 480$ resolution.
- **Class Breakdown**: 176 Fall camera streams, 16 Normal ADL camera streams.

---

## 6. Dataset Compatibility Matrix

| Property | Le2i | URFD | Multicam (`dataset/`) | Unified Representation |
| :--- | :--- | :--- | :--- | :--- |
| **Native FPS** | 25.0 FPS | 30.0 FPS | 120.0 FPS | **25.0 FPS Target** |
| **Resolution** | $320 \times 240$ | $640 \times 240$ | $720 \times 480$ | **Scale-Normalized Coordinates ($[0,1]$)** |
| **Annotation Format** | Text (`.txt`) frame boundaries | CSV (`-1`, `1`) frame labels | Scenario-level timestamps | **Frame-Level Binary (0=ADL, 1=FALL)** |
| **Cameras / Views** | Single per video | Dual (cam0, cam1) | 8 Sync Cameras (cam1..cam8) | **Camera-ID & Location Aware** |

---

## 7. Temporal FPS Conversion Plan
- **Le2i**: Native 25 FPS $\implies$ Stride = 1 (No conversion needed).
- **URFD**: Native 30 FPS $\implies$ Linear frame selection resampling ($30 \to 25 \text{ FPS}$, picking 5 out of 6 frames) or receptive field adjustment ($60 \text{ frames} / 30 \text{ FPS} = 2.0\text{s}$).
- **dataset/**: Native 120 FPS $\implies$ Temporal downsampling with stride $S=5$ ($120 / 5 = 24 \text{ FPS}$, yielding $2.08\text{s}$ receptive field over 50 frames).

---

## 8. Annotation Conversion & Transformation Strategy
- **Le2i**: Frames between `fall_start` and `fall_end + 25` labeled `1` (FALL); all other frames labeled `0` (NORMAL).
- **URFD**: CSV label `-1` mapped to `0` (NORMAL), label `1` mapped to `1` (FALL). Resampled frame indices inherit source frame labels.
- **Multicam (`dataset/`)**: Frame ranges aligned with chute impact timestamps mapped to `1` (FALL), pre-fall/post-recovery mapped to `0` (NORMAL).

---

## 9. Feature Compatibility
- **Normalized Keypoints (99-D)**: Coordinates divided by frame width $W$ and height $H$. Fully invariant to resolution differences ($320\times240$ vs $640\times240$ vs $720\times480$).
- **Velocities (66-D)**: Resampled frame step velocities normalized to per-second velocity ($\text{velocity\_px\_per\_sec} = \Delta \text{pos} \times \text{FPS}$).
- **Spatial Geometry (22-D)**: Torso angles, spine verticality, and joint aspect ratios are scale-invariant unitless metrics.

---

## 10. Label Standardization Policy
- `0` = **NORMAL** (Standing, Walking, Sitting, Bending, Normal ADL movement).
- `1` = **FALL** (Uncontrolled descent, Impact, Prolonged lying post-impact).
- **Transition Buffer**: Pre-impact descent frames included in `1` if within 0.5s of impact; post-impact lying postures up to recovery initiation labeled `1`.

---

## 11. Windowing Strategy
- **Window Length**: 50 frames ($2.0\text{s}$ temporal context).
- **Training Window Stride**: 25 frames ($50\%\text{ overlap}$).
- **Validation Window Stride**: 25 frames ($50\%\text{ overlap}$).
- **Window Labeling Rule**: Window is labeled `1` (FALL) if $\ge 40\%$ of its constituent frames carry label `1`.

---

## 12. Leakage Prevention Strategy

```
                          482 TOTAL VIDEOS
                                 │
     ┌───────────────────────────┼───────────────────────────┐
     ▼                           ▼                           ▼
Le2i (190 Vids)            URFD (100 Vids)          Multicam (192 Vids)
Group by Location        Group by Sequence ID         Group by Chute ID
  (Coffee, Home,            (fall-01..30,               (chute01..24)
   Office, Lecture)           adl-01..40)
     │                           │                           │
     └───────────────────────────┼───────────────────────────┘
                                 ▼
                     GROUPED STRATIFIED K-FOLD
            (Zero Subject / Location / Video Cross-Leakage)
```

> [!CAUTION]
> **STRICT SPLIT RULES**:
> 1. All 8 camera views of a Multicam chute scenario MUST remain in the same split fold.
> 2. Both `cam0` and `cam1` views of a URFD sequence MUST remain in the same split fold.
> 3. Adjacent temporal windows from the same video MUST NEVER be split across train and test sets.

---

## 13. Class-Balance Analysis & Strategy
- Estimated total windows across all 3 datasets: $\approx 8,500$ temporal windows.
- Class distribution: $\approx 60\%$ Fall windows / $40\%$ Normal windows.
- **Recommended Strategy**: Weighted Binary Cross-Entropy Loss ($\text{pos\_weight} \approx 1.25$) rather than destructive undersampling.

---

## 14. Proposed Unified Dataset Structure

```
d:\ONE_DATA\Fall detection\
├── processed_data/
│   ├── multi_dataset_unified/
│   │   ├── processed_features_manifest.csv
│   │   ├── features/
│   │   │   ├── win_le2i_coffee_01_0001.npz
│   │   │   ├── win_urfd_fall_01_0001.npz
│   │   │   └── win_multicam_chute01_c1_0001.npz
```

---

## 15. Proposed Future Training Experiments

| Experiment ID | Training Dataset | Validation / Test Sets | Objective |
| :--- | :--- | :--- | :--- |
| **EXP-A (Baseline)** | Le2i Only (Frozen K1) | Le2i | Measure baseline performance ($\text{F1} \approx 86.6\%$). |
| **EXP-B** | Le2i + URFD | Le2i + URFD | Test dual-dataset generalization. |
| **EXP-C** | Le2i + Multicam | Le2i + Multicam | Test high-speed multi-angle robustness. |
| **EXP-D (Unified)** | Le2i + URFD + Multicam | Unified Test Set | Evaluate unified multi-dataset model. |
| **EXP-E (Cross-Dataset)**| Le2i + URFD | Multicam (Zero-Shot) | Measure out-of-distribution cross-dataset transfer. |

---

## 16. Evaluation Methodology
Metrics reported per fold and across unified test sets:
- **Macro & Binary F1-Score**
- **Precision, Recall, Specificity**
- **Confusion Matrix** (TP, FP, TN, FN)
- **Per-Dataset Performance Breakdown** (Le2i F1 vs URFD F1 vs Multicam F1)
- **Inference Latency (ms) & FPS**

---

## 17. Prediction Logging Design
Every window prediction logged to structured CSV:
`dataset, video_id, subject_id, location_id, event_id, window_id, frame_start, frame_end, ground_truth, fall_prob, threshold, predicted_label, is_correct, error_type`

---

## 18. Baseline-vs-New-Model Comparison Strategy
The new model candidate will be evaluated in parallel against frozen `checkpoints/final_k1/final_production.pth`.  
**Promotion Criterion**: The new candidate model must achieve superior cross-dataset Recall ($\ge 92.0\%$) and lower False Negative rate without reducing single-person inference FPS below 25.0 FPS.

---

## 19. Risks and Limitations
1. **URFD Duplicate Annotation**: `fall-11-data (1).csv` must be ignored during manifest parsing.
2. **Multicam High FPS**: 120 FPS video downsampling must use precise frame indexing to avoid frame jitter.
3. **Lighting & Background Variance**: Multicam and URFD contain varied lighting conditions that require robust YOLO keypoint confidence filtering.

---

## 20. Exact Next Steps
1. Await explicit user approval of this readiness audit and design specification.
2. Run non-destructive dataset manifest builder script for unified multi-dataset tracking.
3. Feature precomputation into `processed_data/multi_dataset_unified/`.
4. Train `checkpoints/multi_dataset_k1/` candidate model and log comparative benchmarks.

---

## 🔒 Final Audit Confirmation

### A. VERIFIED FACTS
- Total local videos audited: **482 videos** (190 Le2i, 100 URFD, 192 Multicam).
- Production model checkpoint: `checkpoints/final_k1/final_production.pth` verified SHA256 `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d`.

### B. RECOMMENDATIONS
- Use weighted BCE loss ($\text{pos\_weight}=1.25$) on unified dataset.
- Group splits strictly by `location_id` and scenario/chute ID to eliminate data leakage.

### C. ITEMS REQUIRING USER APPROVAL
- Approval to execute unified dataset feature extraction into `processed_data/multi_dataset_unified/`.
- Approval to launch future candidate training run `checkpoints/multi_dataset_k1/`.

### D. ACTIONS NOT PERFORMED
- **No model retraining performed.**
- **Frozen K1 checkpoint untouched.**
- **Le2i untouched.**
- **URFD untouched.**
- **data/ untouched.**
- **No large dataset generated.**
- **No production model changed.**
- **No Streamlit behavior changed.**
- **No git add / commit / push executed.**
