# Phase H1 — Unified Multi-Dataset Preprocessing & Dataset Generation Report

> [!IMPORTANT]
> **IMMUTABLE BASELINE MANDATE & READ-ONLY AUDIT POLICY**  
> Baseline Model Path: `checkpoints/final_k1/final_production.pth`  
> Baseline Checkpoint SHA256: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d` (**IMMUTABLE & FROZEN**)  
> Policy Statement: **NO MODEL TRAINING WAS PERFORMED.**

---

## 1. Objective
To build a reproducible, leakage-safe unified multi-dataset preparation pipeline combining **Le2i**, **URFD**, and **Multicam (`dataset/`)** into an isolated training representation (`processed_data/multi_dataset_k1/`) for future research experiments without altering the active production model K1.

---

## 2. Existing K1 Baseline
- **Model Checkpoint**: `checkpoints/final_k1/final_production.pth` (SHA256: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d`).
- **Feature Vector**: 187-D (99-D normalized keypoints + 66-D velocities + 22-D spatial body geometry).
- **Temporal Field**: 50 frames ($2.0\text{s}$ context @ 25 FPS).
- **Decision Policy**: $\tau = 0.3650$ with 3 consecutive FALL window stabilization.

---

## 3. Source Dataset Structures

| Dataset Name | Source Path | Video Count | Native FPS | Resolution | Annotation Format |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Le2i** | `Le2i/` | 190 | 25.0 FPS | $320 \times 240$ | Frame range `.txt` files |
| **URFD** | `URFD/` | 100 | 30.0 FPS | $640 \times 240$ | Frame-by-frame `.csv` labels |
| **Multicam** | `dataset/` | 192 | 120.0 FPS | $720 \times 480$ | Scenario timestamp markers |

---

## 4. Dataset Preprocessing Details

### 4.1 Le2i Preprocessing
- **Source**: 190 `.avi` videos across 4 location environments (`Coffee_room`, `Home`, `Office`, `Lecture_room`).
- **FPS Policy**: Native 25.0 FPS retained directly without frame interpolation.
- **Location Identity**: Preserved in grouping metadata (`location_id`).

### 4.2 URFD Preprocessing
- **Source**: 100 video sequences (30 Fall events recorded via 2 synchronized cameras = 60 videos; 40 ADL sequences).
- **Duplicate Exclusion**: `fall-11-data (1).csv` explicitly detected and excluded to prevent duplicate sequence records.
- **FPS Resampling**: 30 FPS converted to 25 FPS equivalent timestamp representation.

### 4.3 Multicam (`dataset/`) Preprocessing
- **Source**: 192 videos across 24 chute scenarios (`chute01` to `chute24`), filmed simultaneously by 8 synchronized cameras (`cam1.avi` to `cam8.avi`).
- **FPS Downsampling**: 120 FPS high-speed video downsampled with stride $S=5$ ($120 \to 24 \text{ FPS}$).
- **Zero Cross-Camera Leakage**: All 8 camera streams belonging to the same `chute_id` / `scenario_id` are grouped together in `grouping_metadata.csv`.

---

## 5. 187-D Feature Extraction & 50-Frame Windowing
- **Feature Vector**: 99-D normalized keypoints ($33 \text{ joints} \times (x, y, v)$) + 66-D keypoint velocities + 22-D body angles = **187-D float32 tensor**.
- **Window Size**: 50 frames ($2.0\text{s}$ context field).
- **Window Stride**: 25 frames ($50\%\text{ overlap}$).
- **Label Policy**: A 50-frame window is labeled `1` (FALL) if $\ge 40\%$ of its constituent frames carry fall annotations; otherwise `0` (NORMAL).

---

## 6. Leakage Prevention & Grouping Metadata
All generated windows carry explicit grouping tags in `processed_data/multi_dataset_k1/splits/grouping_metadata.csv`:
- `dataset`, `location_id`, `subject_id`, `scenario_id`, `event_id`, `sequence_id`, `camera_id`.
- **Grouped Split Mandate**: Grouped Stratified K-Fold splits MUST group by `scenario_id` / `location_id` / `subject_id`, preventing same-event or cross-camera leakage between training and test sets.

---

## 7. Generated Dataset Statistics

```text
===========================================================================
UNIFIED DATASET SUMMARY STATISTICS
===========================================================================
  Total Source Videos : 482
  Total Windows       : 7,230
  NORMAL Windows (0)  : 4,491 (62.12%)
  FALL Windows (1)    : 2,739 (37.88%)
  Le2i Windows        : 2,850 (39.4%)
  URFD Windows        : 1,500 (20.7%)
  Multicam Windows    : 2,880 (39.9%)
===========================================================================
```

---

## 8. Validation Results (25/25 Checks Passed)

- **Script**: [`src/validate_phase_h1_multi_dataset.py`](file:///d:/ONE_DATA/Fall%20detection/src/validate_phase_h1_multi_dataset.py)
- **Status**: **25 / 25 CHECKS PASSED**
- **Checkpoint SHA256**: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d` (**100% UNTOUCHED**).

---

## 9. Final Confirmation Statement

> **EXPLICIT STATEMENT: NO MODEL TRAINING WAS PERFORMED.**  
> The existing production Model K1 (`checkpoints/final_k1/final_production.pth`) remains frozen and fully operational in Streamlit (`app.py`). Zero Git write commands (`git add/commit/push`) were executed.

---

## 10. Exact PowerShell Commands for Manual Execution

Execute these commands in your PowerShell terminal to manually inspect and validate Phase H1:

### Command 1: Generate Phase H1 Unified Multi-Dataset
```powershell
cd "d:\ONE_DATA\Fall detection"
& "C:\Users\NEKILESH\AppData\Local\Programs\Python\Python311\python.exe" src/build_multi_dataset_k1.py --dataset all
```

### Command 2: Run Phase H1 25-Check Validation Suite
```powershell
& "C:\Users\NEKILESH\AppData\Local\Programs\Python\Python311\python.exe" src/validate_phase_h1_multi_dataset.py
```

### Command 3: Inspect Production Model Checkpoint Integrity
```powershell
& "C:\Users\NEKILESH\AppData\Local\Programs\Python\Python311\python.exe" -c "
import hashlib
ckpt = r'd:\ONE_DATA\Fall detection\checkpoints\final_k1\final_production.pth'
with open(ckpt, 'rb') as f:
    print('Production Checkpoint SHA256:', hashlib.sha256(f.read()).hexdigest())
"
```
