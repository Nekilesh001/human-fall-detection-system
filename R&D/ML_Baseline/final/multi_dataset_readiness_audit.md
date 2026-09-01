# Multi-Dataset Readiness Audit: Le2i, URFD, and dataset/

> [!IMPORTANT]
> **READ-ONLY AUDIT POLICY — NO MODEL TRAINING EXECUTED**  
> This audit evaluates the readiness of incorporating **URFD** and **dataset/** alongside **Le2i** into a unified training pipeline.  
> Target Model Status: **DO NOT TRAIN YET**. Baseline Model K1 (`final_production.pth`) remains frozen and active in production.

---

## 1. Comparative Dataset Characteristics & Metadata Matrix

| Dataset Identifier | Total Videos / Sequences | Frame Rate (FPS) | Native Resolution | Fall Classes | Normal ADL Classes | Annotation Format | Frame-Level Temporal Boundaries |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Le2i** | 190 | 25.0 FPS | $320 \times 240$ | 130 Falls | 60 ADL | Text (`.txt` start/end frame) | Exact start & end frame indices |
| **URFD** | 100 | 30.0 FPS | $640 \times 240$ | 30 Falls | 70 ADL | CSV (`-1` ADL, `1` Fall) | Frame-by-frame binary label |
| **dataset/** | 192 | 120.0 FPS | $720 \times 480$ | High-speed Falls | High-speed ADL | Metadata / File naming | Video-level & Frame-level |

---

## 2. Technical Compatibility & Preprocessing Pipeline Requirements

### 2.1 Receptive Field & Temporal Resampling Strategy
- **Le2i**: 25 FPS native $\implies$ Direct 50-frame windowing ($2.0\text{s}$ context field).
- **URFD**: 30 FPS native $\implies$ Temporal interpolation / downsampling ($30 \to 25 \text{ FPS}$) or 60-frame windowing.
- **dataset/**: 120 FPS native $\implies$ Temporal downsampling with stride $S=5$ ($120 \to 24 \text{ FPS}$) to match the 2.0-second temporal receptive field.

### 2.2 YOLO11-Pose & 187-D Feature Extraction Readiness
- **Spatial Normalization**: Keypoints are normalized by frame width ($W$) and height ($H$), ensuring scale invariance across $320 \times 240$, $640 \times 240$, and $720 \times 480$.
- **187-D Derivation**: 99-D keypoint geometry + 66-D velocities + 22-D body angles operate uniformly across all three datasets once temporal FPS is aligned.

### 2.3 Data Leakage Prevention Strategy (PART 8 DESIGN)
Every sample generated in the unified dataset pipeline must carry mandatory metadata attributes:
- `dataset`: `"Le2i"`, `"URFD"`, or `"dataset"`
- `subject_id`: Subject identifier
- `location_id`: Camera angle / room environment
- `video_id`: Source video filename
- `event_id`: Unique fall / ADL event instance
- `window_id`: Window index within event
- `frame_start` & `frame_end`: Frame range
- `label`: `0` (Normal) or `1` (Fall)

> [!CAUTION]
> **STRICT SPLIT MANDATE**: Train/test splits MUST be grouped strictly by `location_id` and `subject_id`. Adjacent temporal windows from the same video or event MUST NEVER cross the train/test split boundary.

---

## 3. Pre-Training Readiness Verdict

- **Le2i**: **PRODUCTION READY** (Baseline K1 trained & evaluated).
- **URFD**: **AUDIT PASSED** (Ready for 25 FPS temporal resampling).
- **dataset/**: **AUDIT PASSED** (Ready for 5x temporal downsampling).
- **Next Step**: Await user explicit approval before launching multi-dataset unified dataset generation or training `checkpoints/multi_dataset_k1/`.
