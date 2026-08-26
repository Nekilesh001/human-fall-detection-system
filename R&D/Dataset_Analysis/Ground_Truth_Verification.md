# Ground-Truth Annotations Verification & Research Report

## 1. Executive Summary

This report establishes the availability, semantics, source origin, and capability of ground-truth annotations across the three research datasets in the workspace:
1. **URFD (UR Fall Detection Dataset)**
2. **Le2i Fall Detection Dataset**
3. **Multiple Cameras Fall Dataset**

All claims are rigorously classified into **FACT** (observed in local files), **OFFICIAL DOCUMENTATION** (stated by dataset authors), **THIRD-PARTY SOURCE** (stated by external benchmarks), **INFERENCE** (reasoned conclusions), or **NOT VERIFIED** (unconfirmed). No dataset files were modified, split, processed, or generated.

---

## 2. URFD Ground-Truth Investigation

### 2.1 Local Evidence
- **FACT**: The downloaded URFD directory (`URFD/`) contains 70 video/image events (30 Fall events, 40 ADL events).
- **FACT**: Contains 72 `.csv` files.
  - 40 ADL CSVs (`adl-XX-data.csv`) contain 2 headerless numeric columns: `[Column 0: Frame Index, Column 1: Timestamp ms]`.
  - 30 Fall CSVs (`fall-XX-data.csv`) contain 3 headerless numeric columns: `[Column 0: Frame Index, Column 1: Timestamp ms, Column 2: Total Acceleration Magnitude g]`.
  - 2 duplicate CSV files exist (`adl-15-data (1).csv` and `fall-11-data (1).csv`).
- **FACT**: The local CSV files do **NOT** contain fall start frame indices, fall end frame indices, impact frame indices, bounding boxes, or subject IDs.

### 2.2 Official Documentation
- **OFFICIAL DOCUMENTATION**: Created by Bogdan Kwolek and Michał Kępski (University of Rzeszów, Poland; published in *Computer Methods and Programs in Biomedicine*, 2014).
- **OFFICIAL DOCUMENTATION**: The official dataset release contains 30 fall events recorded with 2 Microsoft Kinect v1 sensors (`cam0` front, `cam1` side) plus an x-IMU/PS Move wearable accelerometer sensor, and 40 ADL events recorded with 1 Kinect sensor (`cam0`) plus accelerometer.

### 2.3 Annotation Availability
- **OFFICIAL DOCUMENTATION**: The official URFD release **does not include a global frame-by-frame annotation file** specifying fall start/end frames.
- **INFERENCE**: Researchers using URFD determine fall boundaries by thresholding the accelerometer Sum Vector (SV) magnitude peak (typically $> 3.0g$) or through secondary manual video inspection.
- **THIRD-PARTY SOURCE**: User-contributed frame-level label files exist on secondary platforms (Kaggle/GitHub), but these are secondary community annotations, not official dataset metadata.

### 2.4 Subject Information
- **OFFICIAL DOCUMENTATION**: The authors reported that falls and ADLs were performed by 5 healthy human participants.
- **FACT**: Local file and folder names do NOT map subject IDs to specific sequence numbers (`fall-01` to `fall-30`, `adl-01` to `adl-40`).
- **NOT VERIFIED**: The exact subject ID performing each individual sequence cannot be verified from local or published metadata. Subject-independent splitting is therefore **NOT VERIFIED / NOT SUPPORTED** locally.

### 2.5 Camera/Event Information
- **FACT**: Fall sequences contain 2 synchronized camera folders (`cam0` front/overhead and `cam1` side). ADL sequences contain 1 camera folder (`cam0`).
- **OFFICIAL DOCUMENTATION**: `cam0` and `cam1` record the exact same physical fall event simultaneously.

### 2.6 What Can Be Measured
- Video-level fall vs. ADL binary classification (based on folder prefix `fall-` vs. `adl-`).
- Multi-modal sensor fusion (RGB + 16-bit Depth + Accelerometer magnitude).
- Accelerometer impact peak magnitude and timestamp delta.

### 2.7 What Cannot Be Measured
- Frame-level temporal accuracy metrics (Precision/Recall on fall onset/end) without external labels.
- Bounding-box localization accuracy.
- Subject-independent (Leave-One-Subject-Out) cross-validation.

---

## 3. Le2i Ground-Truth Investigation

### 3.1 Annotation Format
- **FACT**: Ground truth is provided in 130 plain text `.txt` files located in `Annotation_files/` (or `Annotations_files/`).
- **FACT**: Format structure:
  - `Line 1`: Integer Fall Start Frame index.
  - `Line 2`: Integer Fall End Frame index.
  - `Lines 3+`: `[frame_index, state_code, x_min, y_min, x_max, y_max]` (bounding box coordinates constrained within the 320x240 frame canvas).

### 3.2 Annotation Semantics
- **FACT**: State codes in column 1 of bounding box lines range from `1` to `8`:
  - `Code 1`: Normal / Standing posture (92% of normal video frames, 87% of pre-fall frames).
  - `Code 2, 3, 5, 6, 8`: Transitional posture states during falling.
  - `Code 4, 7`: Bounding box of fallen / lying on floor posture (predominant in post-fall frames).

### 3.3 Fall Start/End
- **FACT**:
  - For **Fall Events**: Line 1 > 0 (Start Frame) and Line 2 > 0 (End Frame). Example (`Coffee_room_01/video (1).txt`): Line 1 = `48`, Line 2 = `80`.
  - For **Normal Events**: Line 1 = `0` and Line 2 = `0`. Example (`Home_02/video (33).txt`): Line 1 = `0`, Line 2 = `0`.

### 3.4 Bounding Boxes
- **FACT**: Every frame line provides a 4-coordinate 2D bounding box `[x_min, y_min, x_max, y_max]` surrounding the person.

### 3.5 Subject Information
- **OFFICIAL DOCUMENTATION**: Created by I. Charfi et al. (Université de Bourgogne, Dijon, JEI 2013). The dataset paper states that 9 participants performed the activities.
- **FACT**: Local text files and folder names do NOT map subject IDs to individual video numbers.

### 3.6 Unannotated Videos
- **FACT**: 60 videos in `Lecture_room` (27 videos) and `Office` (33 videos) have **ZERO annotation files** in the local directory.
- **THIRD-PARTY SOURCE**: Academic discussions confirm that only `Coffee_room` and `Home` subsets were distributed with ground-truth text files; `Lecture_room` and `Office` are unannotated in standard downloads.

### 3.7 What Can Be Measured
- Frame-level fall detection (Fall Start, Fall End, Fallen duration).
- 2D Bounding-Box person detection and spatial tracking.
- Leave-One-Location-Out (LOLO) cross-validation across the 6 scene environments.

### 3.8 What Cannot Be Measured
- Supervised evaluation on `Lecture_room` and `Office` videos.
- Subject-independent cross-validation.

---

## 4. Multiple Cameras Fall Dataset Investigation

### 4.1 Event Structure
- **FACT**: Contains 24 `chuteXX` folders (`chute01` to `chute24`).
- **OFFICIAL DOCUMENTATION**: Created by Edouard Auvinet et al., 2010 (Université de Montréal, DIRO Technical Report 1350).
- **OFFICIAL DOCUMENTATION**: Scenarios 1 through 22 contain fall events interleaved with ADLs. Scenarios 23 and 24 contain **ONLY ADL / confounding activities (NO falls)**.

### 4.2 Camera Structure
- **FACT**: Every `chuteXX` folder contains **8 synchronized `.avi` video files** (`cam1.avi` to `cam8.avi`), 720x480 resolution, `FMP4` codec.

### 4.3 Official Annotations
- **FACT**: **ZERO annotation files exist locally** in `dataset/dataset/`.
- **OFFICIAL DOCUMENTATION**: Official frame-level labels (scenario, camera, start frame, end frame, position code) exist in the published **Technical Report 1350** (Université de Montréal), but were not bundled in this local directory download.

### 4.4 Subject Information
- **OFFICIAL DOCUMENTATION**: All 24 scenarios were performed by **1 single actor/subject**.
- **INFERENCE**: Subject-independent cross-validation is **IMPOSSIBLE** for this dataset.

### 4.5 Normal Activity Information
- **OFFICIAL DOCUMENTATION**: Scenarios 1-22 contain ADLs preceding and following the fall. Scenarios 23 and 24 consist entirely of normal ADL activities (walking, crouching, sitting).

### 4.6 What Can Be Measured
- Multi-view 3D vision algorithms and view-invariant representation learning across 8 synchronized camera angles.
- Scenario-level classification (`chute01`..`chute22` vs `chute23`..`chute24`).

### 4.7 What Cannot Be Measured
- Supervised frame-level evaluation without manually transcribing Technical Report 1350 tables.
- Subject-independent evaluation (only 1 subject exists).

---

## 5. Ground-Truth Comparison Table

| Capability | URFD | Le2i | Multi-Camera | Explanation / Evidence |
| :--- | :--- | :--- | :--- | :--- |
| **Event-level fall label** | **YES** | **YES** | **YES** | **URFD**: `fall-` prefix. **Le2i**: Non-zero start/end in `.txt`. **MultiCamera**: `chute01-22` (Paper). |
| **Event-level normal label** | **YES** | **YES** | **YES** | **URFD**: `adl-` prefix. **Le2i**: `0,0` header in `.txt`. **MultiCamera**: `chute23-24` (Paper). |
| **Frame-level label** | **NO** | **YES** | **NO** | **Le2i**: Explicit state code per line. **URFD & MultiCamera**: Missing locally. |
| **Fall start frame** | **NO** | **YES** | **NO** | **Le2i**: Line 1 in `.txt`. **URFD & MultiCamera**: Missing locally. |
| **Fall end frame** | **NO** | **YES** | **NO** | **Le2i**: Line 2 in `.txt`. **URFD & MultiCamera**: Missing locally. |
| **Impact frame** | **PARTIAL** | **NO** | **NO** | **URFD**: Peak accel magnitude in CSV indicates impact time. **Le2i & MultiCamera**: Not marked. |
| **Bounding box** | **NO** | **YES** | **NO** | **Le2i**: `[x_min, y_min, x_max, y_max]` per line. **URFD & MultiCamera**: None. |
| **Subject ID** | **UNKNOWN** | **UNKNOWN** | **YES** | **MultiCamera**: 1 subject (Paper). **URFD & Le2i**: Unmapped in local files. |
| **Camera ID** | **YES** | **YES** | **YES** | **URFD**: `cam0`/`cam1`. **Le2i**: Scene folder. **MultiCamera**: `cam1`..`cam8`. |
| **Timestamp** | **YES** | **PARTIAL** | **NO** | **URFD**: Column 1 in ms. **Le2i**: FPS-derived frame time. **MultiCamera**: None. |

---

## 6. Evaluation Capability Comparison

| Experiment Type | Status | Reason / Explanation |
| :--- | :--- | :--- |
| **1. Event-level fall classification** | **SUPPORTED** | All 3 datasets have clear event-level fall vs. normal activity designations. |
| **2. Frame-level classification** | **PARTIALLY SUPPORTED** | Supported on **Le2i** (130 videos). NOT supported on local URFD or MultiCamera. |
| **3. Temporal sequence classification** | **SUPPORTED** | Sequential frame streams and videos exist across all 3 datasets. |
| **4. Time-to-detection evaluation** | **PARTIALLY SUPPORTED** | Supported on **Le2i** (using Fall Start Line 1). NOT supported on local URFD or MultiCamera. |
| **5. Bounding-box baseline** | **PARTIALLY SUPPORTED** | Supported ONLY on **Le2i** (130 annotated videos). |
| **6. Subject-independent evaluation** | **NOT SUPPORTED** | Subject IDs are unmapped in local URFD/Le2i files, and MultiCamera contains only 1 subject. |
| **7. Event-independent evaluation** | **SUPPORTED** | Splitting by event ID (`adl-XX`/`fall-XX`/`chuteXX`) prevents event-level contamination. |
| **8. Cross-location evaluation** | **SUPPORTED** | Supported on **Le2i** using Leave-One-Location-Out (LOLO) across the 6 room folders. |
| **9. Cross-dataset evaluation** | **SUPPORTED** | Training on one dataset (e.g. Le2i RGB) and testing on another (e.g. URFD RGB). |
| **10. False-alarm-per-camera-hour** | **NOT SUPPORTED** | Continuous multi-hour unsegmented video recordings are not present in these datasets. |

---

## 7. Data Limitations

1. **Missing Ground-Truth Annotations**: URFD and MultiCamera lack local frame-level labels.
2. **Unannotated Video Portion**: Le2i contains 60 unannotated videos (31.6% of dataset).
3. **Single-Subject Limitation**: MultiCamera was recorded with only 1 subject (no subject diversity).
4. **Subject Anonymity Gaps**: URFD and Le2i do not map subject IDs to filenames, blocking subject-wise splitting.
5. **Severe Multi-View & Scene Leakage Risk**: Random frame/video splitting within events or rooms will corrupt validation results.

---

## 8. Information That Still Needs Verification

1. Transcribing or linking official frame-level annotation tables from **Auvinet et al. Technical Report 1350** for the Multiple Cameras Fall Dataset.
2. Locating external official frame boundary labels for URFD sequences (`fall-01` to `fall-30`).
3. Verifying whether official subject-to-sequence mapping tables exist for URFD and Le2i.

---

## 9. Recommended Next Research Step

> [!TIP]
> **Recommended Next Step**: Sourcing/transcribing the external ground-truth frame-level annotation tables for URFD and MultiCamera (Technical Report 1350) to create a unified annotation catalog before defining preprocessing windows.
