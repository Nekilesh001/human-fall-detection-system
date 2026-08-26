# URFD RGB Baseline Preprocessing Report

## Executive Summary
This report documents the design, implementation, and automated validation of the **URFD RGB-Only Baseline Preprocessing Pipeline**. 

> [!IMPORTANT]
> **CLASSIFICATION SCOPE NOTICE**:
> This baseline uses **event-derived temporal labels** (`fall-XX` $\implies$ `FALL`, `adl-XX` $\implies$ `NORMAL`) because verified frame-level fall boundaries are not available for URFD locally. This is an **event-level baseline**, NOT a frame-level fall detection model.

---

## 1. Input Dataset Summary
- **Source Dataset**: URFD (UR Fall Detection Dataset)
- **Input Modality**: RGB Visual Stream ONLY ($640 \times 480$ pixels)
- **Source Inventory**: 70 events (30 Fall events, 40 ADL events), 100 video streams (60 Fall videos from `cam0`+`cam1`, 40 ADL videos from `cam0`).

---

## 2. Split Summary & Event Scope
- **Intended Split**: 70 events (30 Fall, 40 ADL) per `R&D/split_strategy.md`.
- **Actual Processed Scope**: **67 events (27 Fall, 40 ADL)**. 
- **Experimental Deviation**: 3 Fall events (`fall-16`, `fall-21`, `fall-22`) were cleanly skipped because their resampled sequences contain 46–47 frames, which is short of the required $W=50$ window length.
- **Partition Breakdown**:
  - **Train Partition (47 Events)**: 19 Fall events (2 excluded: `fall-16`, `fall-22`), 28 ADL events
  - **Validation Partition (9 Events)**: 3 Fall events (1 excluded: `fall-21`), 6 ADL events
  - **Test Partition (11 Events)**: 5 Fall events (**100% retained**), 6 ADL events (**100% retained**)

---

## 3. Sampling Methodology
- **Source Frame Rate**: ~30 FPS (Kinect v1 stream)
- **Target Frame Rate**: **25.0 FPS** (Resampling period $\Delta t = 40.0\text{ ms}$)
- **Resampling Method**: Deterministic nearest-neighbor timestamp matching ($t_k = k \times 40.0\text{ ms}$). Eliminates motion jitter and avoids frame duplication.

---

## 4. Spatial Preprocessing
- **Source Resolution**: $640 \times 480$ pixels
- **Target Resolution**: **$320 \times 240$ pixels, RGB (uint8)**
- **Interpolation Filter**: **Lanczos (Antialiased Area Downscaling)** to preserve human silhouette edges without high-frequency aliasing.

---

## 5. Temporal Preprocessing & Window Configuration
- **Window Length ($W$)**: **50 frames** (2.0 seconds at 25 FPS)
- **Frame Stride ($S$)**: **25 frames** (1.0 second, 50% overlap)
- **Window Rule**: Only complete 50-frame windows are generated; partial windows are strictly excluded.

---

## 6. Label Methodology
- **Event-Derived Labels**:
  - `fall-01` .. `fall-30` $\implies$ `FALL` (Event label inherited by all windows)
  - `adl-01` .. `adl-40` $\implies$ `NORMAL` (Event label inherited by all windows)

---

## 7. Output Representation & Storage Format
- **Directory Structure**:
  ```
  processed_data/URFD_RGB_baseline/
  ├── train/                   # Stored .npz sample files
  ├── val/                     # Stored .npz sample files
  ├── test/                    # Stored .npz sample files
  ├── processed_manifest.csv   # Portable sample manifest
  └── preprocessing_report.md  # Detailed preprocessing documentation
  ```
- **Sample File Format**: Compressed NumPy array files (`.npz`) containing `frames` array of shape $(50, 240, 320, 3)$ with `uint8` data type.

---

## 8. Train / Val / Test Statistics & Window Distribution

| Partition | Unique Events | Unique Videos | FALL Windows | NORMAL Windows | Total Windows | % of Dataset |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 47 | 66 | 84 | 176 | **260** | 72.2% |
| **Validation** | 9 | 12 | 10 | 33 | **43** | 11.9% |
| **Test** | 11 | 16 | 24 | 33 | **57** | 15.8% |
| **TOTAL** | **67** | **94** | **118** | **242** | **360** | **100.0%** |

---

## 9. Performance & Storage Overview
- **Processing Execution Time**: **263.02 seconds** (~4.38 minutes for full dataset).
- **Total Processed Storage Size**: **2,899.29 MB** (~2.90 GB across 360 `.npz` files).

---

## 10. Automated Validation & Leakage Verification Results

1. **Event Leakage Check**: **PASSED** (Train, Val, and Test event sets are strictly disjoint).
2. **Camera Leakage Check**: **PASSED** (All dual-camera Fall events keep `cam0` and `cam1` in the identical partition).
3. **Sample Integrity Check**: **PASSED** (0 broken paths, 0 invalid array shapes across all 360 `.npz` files).
4. **Label Consistency Check**: **PASSED** (All `FALL` windows originate from `fall-XX`; all `NORMAL` windows originate from `adl-XX`).
5. **Source Integrity Check**: **PASSED** (Raw URFD dataset files on disk remain 100% unmodified).

---

## 11. Skipped / Incomplete Sequences

The following 6 video records (~3 events) were shorter than the 50-frame minimum window requirement after 25 FPS resampling and were cleanly skipped:
- `URFD_fall-16_cam0` & `URFD_fall-16_cam1` (46 resampled frames < 50)
- `URFD_fall-21_cam0` & `URFD_fall-21_cam1` (46 resampled frames < 50)
- `URFD_fall-22_cam0` & `URFD_fall-22_cam1` (46 resampled frames < 50)

> [!NOTE]
> Both `cam0` and `cam1` for each affected event were skipped identically, preserving camera symmetry.

---

## 12. Known Limitations & Future Work
- **Lack of Frame-Level Fall Onset Labels**: Model training will evaluate event-level fall vs. ADL detection. Frame-level temporal accuracy cannot be evaluated on URFD without external frame annotations.
- **RGB Only**: Multi-modal depth maps and accelerometer signals are excluded from this initial baseline.

---

## 13. Reproducibility Instructions

To reproduce the preprocessing pipeline and recreate `processed_manifest.csv`:

```bash
# 1. Validate raw manifest
python src/validate_manifest.py

# 2. Run small test subset (optional)
python src/preprocess_urfd.py --subset

# 3. Execute full URFD RGB baseline preprocessing
python src/preprocess_urfd.py

# 4. Run automated preprocessing validation
python src/validate_preprocessing.py
```
