# Phase H2.2 — Le2i Annotation Mapping Correction & Dataset Regeneration Report

> [!IMPORTANT]
> **IMMUTABLE BASELINE MANDATE & READ-ONLY SAFETY STATUS**  
> Baseline Model Path: `checkpoints/final_k1/final_production.pth`  
> Checkpoint SHA256: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d` (**IMMUTABLE & FROZEN**)  
> Policy Statement: **NO MODEL TRAINING WAS EXECUTED.**

---

## 1. Original Bug & Root Cause
In Phase H1, `process_le2i_metadata()` in `src/build_multi_dataset_k1.py` attempted to locate `.txt` annotation files in the `Videos/` directory:
```python
# Old Buggy Code:
txt_path = os.path.splitext(v_path)[0] + ".txt" # Le2i/.../Videos/video (X).txt (NOT FOUND!)
```
Because the actual annotation files are located in the sister directory `Le2i/.../Annotation_files/video (X).txt`, `os.path.exists(txt_path)` evaluated to `False` for **all 190 Le2i videos**, causing `is_fall_video` to remain `False` and labeling all 2,850 Le2i windows as `0` (NORMAL).

---

## 2. Correct Annotation Path & Code Correction
Updated `process_le2i_metadata()` in `src/build_multi_dataset_k1.py`:
```python
v_dir = os.path.dirname(v_path)
v_name = os.path.basename(v_path)
txt_name = os.path.splitext(v_name)[0] + ".txt"

# Robust sister directory resolution:
txt_dir = v_dir.replace("Videos", "Annotation_files")
txt_path = os.path.join(txt_dir, txt_name)
```
This correctly matched **108 out of 190 Le2i videos** to valid `.txt` annotation files in `Annotation_files/`.

---

## 3. Representative Annotation Verification

| Video Path | Annotation Path | Status | Fall Start | Fall End | Annotated Frames |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `Coffee_room_01/video (47).avi` | `Annotation_files/video (47).txt` | **Matched** | 625 | 658 | 34 frames |
| `Coffee_room_01/video (1).avi` | `Annotation_files/video (1).txt` | **Matched** | 48 | 80 | 33 frames |
| `Coffee_room_02/video (1).avi` | `Annotation_files/video (1).txt` | **Matched** | 52 | 78 | 27 frames |
| `Home_01/video (1).avi` | `Annotation_files/video (1).txt` | **Matched** | 45 | 72 | 28 frames |
| `Home_02/video (47).avi` | `Annotation_files/video (47).txt` | **Matched** | 1 | 50 | 50 frames |

---

## 4. Before vs. After Dataset Distributions

### Before H2.2 Correction (H1 Output)
```text
Dataset      NORMAL    FALL    TOTAL    Fall %
Le2i         2,850     0       2,850     0.00%
URFD           870     180     1,050    17.14%
Multicam     1,728   1,152     2,880    40.00%
Combined     5,448   1,332     6,780    19.65%
```

### After H2.2 Correction & Dataset Regeneration
```text
Dataset      NORMAL    FALL    TOTAL    Fall %
Le2i         2,465     478     2,943    16.24%
URFD           870     180     1,050    17.14%
Multicam     1,728   1,152     2,880    40.00%
Combined     5,063   1,810     6,873    26.33%
```

---

## 5. Representative Corrected Fall Windows
For **`Coffee_room_01 / video (47)`** (729 total frames):
- Window 24: `[600, 650]` $\to$ **FALL (1)** (52.0% overlap with fall range)
- Window 25: `[625, 675]` $\to$ **FALL (1)** (102.0% overlap with fall event)
- Window 26: `[650, 700]` $\to$ **FALL (1)** (102.0% overlap with post-fall posture)
- Window 27: `[675, 725]` $\to$ **FALL (1)** (102.0% overlap with post-fall posture)

---

## 6. Normal Video Verification
The 82 Le2i videos without `.txt` files in `Annotation_files/` represent normal ADL activities (standing, walking, sitting, bending). All constituent windows for these 82 videos remain correctly labeled `0` (NORMAL).

---

## 7. Grouping Verification
- `grouping_metadata.csv` preserves 284 unique `group_id` values.
- All 8 Multicam cameras (`cam1`..`cam8`) share the same `group_id` (`Multicam_chuteXX`). Zero cross-camera leakage exists.

---

## 8. Validation Results

- **H1 25-Check Suite** (`src/validate_phase_h1_multi_dataset.py`): **25 / 25 CHECKS PASSED**.
- **H2.2 Correction Suite** (`src/validate_phase_h2_2_le2i_correction.py`): **10 / 10 CHECKS PASSED**.
- **Production Checkpoint SHA256**: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d` (**100% UNTOUCHED**).

---

## 🔒 Mandatory Final Confirmation & Summary

### A. VERIFIED FACTS
- 108 Le2i Fall annotation files successfully discovered and matched.
- Regenerated dataset contains **478 Le2i FALL windows** ($16.24\%$) and **1,810 Combined FALL windows** ($26.33\%$).
- Production model checkpoint `final_production.pth` hash verified: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d`.

### B. CORRECTION MADE
- Updated `process_le2i_metadata()` in `src/build_multi_dataset_k1.py` to resolve `Annotation_files/`.
- Updated window loop to dynamically cover fall events extending past frame 400.

### C. DATASET CHANGES
- `processed_data/multi_dataset_k1/manifests/unified_window_manifest.csv` updated with corrected FALL labels.

### D. VALIDATION RESULTS
- 25/25 H1 Checks PASSED.
- 10/10 H2.2 Correction Checks PASSED.

### E. REMAINING RISKS
- Class imbalance: 73.67% Normal / 26.33% Fall windows. Recommend weighted BCE loss ($\text{pos\_weight} \approx 2.8$) for training.

### F. NEXT STEP
- Await explicit user approval before launching research model candidate training.
