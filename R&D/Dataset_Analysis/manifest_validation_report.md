# Master Dataset Manifest Validation & Leakage Check Report

## Executive Summary
This report summarizes the programmatic, read-only validation and leakage verification performed on the Master Dataset Manifest (`dataset_manifest.csv`) generated for the workspace datasets:
1. **URFD (UR Fall Detection Dataset)**
2. **Le2i Fall Detection Dataset**
3. **Multiple Cameras Fall Dataset**

---

## 1. High-Level Inventory Summary

| Dataset Name | Unique Events | Total Video Records | FALL Records | NORMAL Records | UNKNOWN Records | Unique Cameras | Missing Local Annotations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **URFD** | 70 | 100 | 60 | 40 | 0 | 2 (`cam0`, `cam1`) | 100 |
| **Le2i** | 190 | 190 | 96 | 31 | 63 | 6 (`location_cam0`) | 60 (plus 3 malformed) |
| **MultiCamera** | 24 | 192 | 176 | 16 | 0 | 8 (`cam1` .. `cam8`) | 192 |
| **TOTAL** | **284** | **482** | **332** | **87** | **63** | **16** | **352** |

---

## 2. Integrity & Validation Checks

### A. Missing Video Files
- **Result**: **0 missing files** (100% of indexed video files exist on disk).

### B. Duplicate Video IDs
- **Result**: **0 duplicate video IDs** (All 482 `video_id` values are strictly unique).

### C. Duplicate Event / Camera Combinations
- **Result**: **0 duplicate event/camera pairs** (No `event_id` has redundant camera entries).

### D. Invalid Annotation References
- **Result**: **0 invalid references** (All specified `annotation_path` targets exist on disk).

### E. Missing Annotations
- **URFD**: 100 video streams lack local frame-level text files (only CSV timestamps & accelerometer logs exist).
- **Le2i**: 60 videos in `Lecture_room` and `Office` have no local annotation files; 3 videos in `Coffee_room` have malformed headers.
- **MultiCamera**: All 192 video streams lack local text annotation files (documented externally in DIRO Report 1350).

---

## 3. Data Leakage Verification

### Check 1: Multi-Camera Sharing of Event IDs
- **URFD**: All 30 Fall events share the same `event_id` across `cam0` and `cam1` (2 camera streams per fall event).
- **MultiCamera**: All 24 chute scenarios share the same `event_id` across `cam1` through `cam8` (8 camera streams per chute scenario).
- **Verification**: **PASSED**. Event grouping correctly binds multi-camera streams to prevent train/test cross-camera contamination.

### Check 2: Event Label Consistency
- **Result**: **PASSED**. 0 events have conflicting labels across camera views. Every camera observation of an event shares the identical `FALL`, `NORMAL`, or `UNKNOWN` label.

### Check 3: Unique Event Identifiers Within Datasets
- **URFD**: 70 unique `event_id` values (`fall-01`..`fall-30`, `adl-01`..`adl-40`).
- **Le2i**: 190 unique `event_id` values (`Coffee_room_01_v1`, `Home_01_v1`, etc.).
- **MultiCamera**: 24 unique `event_id` values (`chute01`..`chute24`).
- **Verification**: **PASSED**. Event IDs are strictly unique within each dataset domain.

---

## 4. Anomalies & Quality Observations

1. **Unannotated Videos (63 Records)**: 60 videos in Le2i (`Lecture_room` & `Office`) and 3 malformed header files are correctly assigned `label = UNKNOWN`.
2. **Camera Synchronization Offset**: In MultiCamera, minor frame count variations (10-30 frames) exist between `cam1` and `cam8` for the same chute event due to IP camera start/stop timing.
3. **Resolution & FPS Heterogeneity**:
   - URFD: $640 	imes 480$ @ 30 FPS
   - Le2i: $320 	imes 240$ (or $320 	imes 180$) @ 24/25 FPS
   - MultiCamera: $720 	imes 480$ @ 25 FPS (real-time)

---

## 5. Manifest Usage Declaration

The generated `dataset_manifest.csv` is validated and ready to serve as the single source of truth for:
- Group-level data splitting (GroupKFold by `event_id` and LOLO by `location`).
- Resampling and spatial normalization during preprocessing.
- Baseline evaluation pipeline integration.
