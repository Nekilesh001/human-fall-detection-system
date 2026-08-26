# Comparative Synthesis: Fall Detection Datasets (URFD, Le2i, MultiCamera)

## Executive Summary
This document synthesizes the structural, modal, spatial, and temporal characteristics of the three research datasets present in the workspace (`URFD/`, `Le2i/`, and `dataset/` [Multiple Cameras Fall Dataset]). The goal is to evaluate their strengths, weaknesses, complementarity, and data leakage risks to guide future data preparation and experimental design.

---

## 1. High-Level Comparison Matrix

| Feature / Metric | URFD (UR Fall Detection) | Le2i Fall Detection | Multiple Cameras Fall Dataset |
| :--- | :--- | :--- | :--- |
| **Directory Path** | `URFD/` | `Le2i/data/` | `dataset/dataset/` |
| **Total Events / Scenarios** | **70 events** (30 Fall, 40 ADL) | **190 events** (96 Fall, 31 ADL, 60 Unannotated) | **24 events** (24 multi-camera fall scenarios) |
| **Total Files** | 30,035 files | 321 files | 192 files |
| **Video Files Count** | 100 `.mp4` files | 190 `.avi` files | 192 `.avi` files |
| **Image Frames Count** | 29,863 `.png` frames | 0 (embedded in `.avi`) | 0 (embedded in `.avi`) |
| **Camera Views per Event** | 2 views (`cam0` front, `cam1` side) for Falls; 1 view for ADL | 1 fixed view per location | **8 synchronized views** (`cam1` .. `cam8`) per scenario |
| **Available Modalities** | **RGB, 16-bit Raw Depth (`I;16`), Accelerometer magnitude, Timestamps** | **RGB Video, Bounding Box Annotations, Fall Start/End Range** | **RGB Video only** (8 views) |
| **RGB Resolution** | 640 x 480 | 320 x 240 (320 x 180 for Home_02) | 720 x 480 |
| **Depth Resolution** | 640 x 480 (16-bit raw depth in mm) | None | None |
| **Frame Rate (FPS)** | ~30 FPS | 24 FPS (Home 01/02) / 25 FPS (Others) | 120 FPS header (~25-30 FPS real-time) |
| **Video Codec / Format** | MP4 (H.264) + PNG frame sequences | AVI (`DIB ` raw uncompressed RGB) | AVI (`FMP4` ISO MPEG-4) |
| **Frame-Level Fall Labels** | **Missing locally** (Timestamps & Accel only) | **Present locally** (in 130 `.txt` files) | **Missing locally** |
| **Environment Scenes** | Indoor laboratory setting | **6 real-world rooms** (Coffee room 1/2, Home 1/2, Lecture room, Office) | Indoor laboratory setting |
| **Local Data Quality Issues** | 2 duplicate CSVs, 1 missing depth frame, 198 nested folder anomalies | 60 unannotated videos, 3 malformed headers, 51 frame count mismatches | Slight camera-to-camera frame offset (10-30 frames per chute) |

---

## 2. Complementarity Analysis Across Datasets

### A. Modality Complementarity
- **URFD**: Only dataset with **16-bit raw Depth maps** and **wearable Accelerometer sensor data**. Crucial for multi-modal sensor fusion research (RGB + Depth + Inertial).
- **Le2i**: Rich in **real-world RGB environments** with **2D bounding boxes** tracking person location and explicit fall start/end frame labels.
- **Multiple Cameras Dataset**: Highest spatial coverage per event (**8 orthogonal camera angles**), ideal for multi-view vision algorithms and view-invariant representation learning.

### B. Environmental & Spatial Diversity
- **URFD**: 1 fixed indoor setting (low background diversity).
- **Le2i**: **Highest environmental diversity** across 6 distinct rooms with varying lighting, furniture clutter, occlusion, and shadows.
- **Multiple Cameras**: 1 room with 8 dense spatial camera viewpoints.

---

## 3. Data Leakage Risks & Cross-Validation Strategy

> [!WARNING]
> **Critical Data Leakage Hazards**
> 1. **Multi-View Leakage (URFD & MultiCamera)**: In URFD (2 views) and MultiCamera (8 views), files in the same event folder depict the **EXACT SAME physical fall**. Random video-level train/test splits will leak the identical subject pose and background into the test set.
> 2. **Scene / Room Leakage (Le2i)**: In Le2i, all videos within a location folder share the exact same room, lighting, and camera position. Random video-level splits will leak room geometry into test splits.

### Recommended Cross-Validation Protocols

| Dataset | Atomic Unit for Train/Test Splitting | Recommended Validation Strategy |
| :--- | :--- | :--- |
| **URFD** | `adl-XX` or `fall-XX` Event Group | **GroupKFold** grouped by Event Prefix (`fall-XX` / `adl-XX`) |
| **Le2i** | Location Folder (`Coffee_room_01`, `Home_01`, etc.) | **Leave-One-Location-Out (LOLO)** Cross-Validation |
| **Multiple Cameras** | `chuteXX` Scenario Directory | **GroupKFold** grouped by `chuteXX` Directory |

---

## 4. Key Limitations & Gaps to Address Before Experiments

1. **Ground-Truth Label Gaps**:
   - URFD and MultiCamera currently lack frame-by-frame fall start/end annotations in local files. Ground-truth annotation files must be sourced or linked externally.
   - Le2i has 60 unannotated videos (`Lecture_room` and `Office`).
2. **Resolution & Frame Rate Heterogeneity**:
   - Combining datasets into a unified pipeline requires handling resolutions ranging from 320x180 up to 720x480, and frame rates from 24 FPS to 30 FPS.
3. **Storage & Format Differences**:
   - URFD uses PNG frame sequences.
   - Le2i uses large uncompressed `DIB ` AVI files.
   - MultiCamera uses `FMP4` AVI files.

---

## 5. Summary & Preserved Artifact Directory

All individual dataset deep-dive analysis reports have been structured and preserved in `R&D/Dataset_Analysis/`:
- [URFD_analysis.md](file:///d:/ONE_DATA/Fall%20detection/R&D/Dataset_Analysis/URFD_analysis.md)
- [Le2i_analysis.md](file:///d:/ONE_DATA/Fall%20detection/R&D/Dataset_Analysis/Le2i_analysis.md)
- [MultiCamera_analysis.md](file:///d:/ONE_DATA/Fall%20detection/R&D/Dataset_Analysis/MultiCamera_analysis.md)
- [Dataset_Comparison.md](file:///d:/ONE_DATA/Fall%20detection/R&D/Dataset_Analysis/Dataset_Comparison.md)
