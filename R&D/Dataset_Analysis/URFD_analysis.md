# UR Fall Detection Dataset (URFD) - Analysis & Inspection Report

## Executive Summary
This document presents an empirical, non-modifying analysis of the **UR Fall Detection Dataset (URFD)** located at `d:\ONE_DATA\Fall detection\URFD`. All statistics, counts, properties, and observations in this document were derived by directly inspecting the existing filesystem without modifying, moving, copying, or altering any dataset file.

---

## 1. Directory Structure

### Workspace Top-Level Directories
The workspace folder `d:\ONE_DATA\Fall detection` contains three top-level directories:
- `URFD/` — The UR Fall Detection Dataset.
- `dataset/` — Contains a nested `dataset/chute01` to `chute24` structure (Multiple Cameras Fall Dataset).
- `Le2i/` — Contains `data/Coffee_room_01` etc. (The Le2i Fall Detection Dataset).

### Target Dataset Structure (`URFD/`)
Within `d:\ONE_DATA\Fall detection\URFD`:
- **Subdirectories**: 200 camera-modality stream subdirectories (100 RGB folders + 100 Depth folders).
- **Files**: 30,035 total files across all levels.
  - `.png` images: 29,863 files
  - `.mp4` videos: 100 files
  - `.csv` data files: 72 files

### Tree Format Diagram

```
d:\ONE_DATA\Fall detection\
├── URFD/
│   ├── adl-01-cam0-d/
│   │   └── adl-01-cam0-d/                 [Nested directory anomaly]
│   │       ├── adl-01-cam0-d-001.png
│   │       ├── adl-01-cam0-d-002.png
│   │       └── ...
│   ├── adl-01-cam0-rgb/
│   │   └── adl-01-cam0-rgb/               [Nested directory anomaly]
│   │       ├── adl-01-cam0-rgb-001.png
│   │       ├── adl-01-cam0-rgb-002.png
│   │       └── ...
│   ├── adl-01-cam0.mp4
│   ├── adl-01-data.csv
│   │   ...
│   ├── adl-15-data (1).csv                [Duplicate file anomaly]
│   ├── adl-15-data.csv
│   │   ...
│   ├── fall-01-cam0-d/
│   ├── fall-01-cam0-rgb/
│   ├── fall-01-cam0.mp4
│   ├── fall-01-cam1-d/
│   ├── fall-01-cam1-rgb/
│   ├── fall-01-cam1.mp4
│   ├── fall-01-data.csv
│   │   ...
│   ├── fall-11-data (1).csv                [Duplicate file anomaly]
│   ├── fall-11-data.csv
│   └── ... (up to fall-30-...)
```

### File Extensions & Counts in `URFD/`
| Extension | Count | Description / Role |
| :--- | :--- | :--- |
| `.png` | 29,863 | Synchronized RGB color frames and raw Depth image frames |
| `.mp4` | 100 | Video files corresponding to RGB image sequences |
| `.csv` | 72 | Headerless tabular data (Frame #, Timestamp ms, Accelerometer magnitude) |
| **Total** | **30,035** | **Total files inside `URFD`** |

### Naming Conventions
1. **Events**:
   - `adl-XX`: Normal Activities of Daily Living (Sequence index `01` to `40`, 40 total events).
   - `fall-XX`: Fall Events (Sequence index `01` to `30`, 30 total events).
2. **Cameras**:
   - `cam0`: Camera 0 (Overhead / front view Microsoft Kinect v1 sensor).
   - `cam1`: Camera 1 (Side view Microsoft Kinect v1 sensor, present only in fall sequences).
3. **Modalities**:
   - `rgb`: RGB color stream (24-bit PNG images).
   - `d`: Depth stream (16-bit raw depth PNG images).
4. **Frame Files**:
   - `{event}-{cam}-{modality}-{3-digit-frame}.png` (e.g. `adl-01-cam0-rgb-001.png`, `fall-01-cam1-d-042.png`).
5. **Video Files**:
   - `{event}-{cam}.mp4` (e.g. `adl-01-cam0.mp4`, `fall-01-cam1.mp4`).
6. **CSV Files**:
   - `{event}-data.csv` (e.g. `adl-01-data.csv`, `fall-01-data.csv`).

---

## 2. Folder Representations: Fact vs. Inference

| Folder Name Pattern | Contents Observed | Modality / Entity Represented | Fact vs. Inference |
| :--- | :--- | :--- | :--- |
| `adl-XX-cam0-rgb/` | PNG images (640x480, 24-bit RGB) | Normal activity, Camera 0, RGB color stream | **FACT**: Directly observed color images of person doing daily activities. |
| `adl-XX-cam0-d/` | PNG images (640x480, 16-bit `I;16`) | Normal activity, Camera 0, Depth map stream | **FACT**: Raw 16-bit depth maps from Microsoft Kinect sensor. |
| `fall-XX-cam0-rgb/` | PNG images (640x480, 24-bit RGB) | Fall event, Camera 0 (overhead/front), RGB stream | **FACT**: Directly observed color images of a person performing a fall. |
| `fall-XX-cam0-d/` | PNG images (640x480, 16-bit `I;16`) | Fall event, Camera 0, Depth map stream | **FACT**: 16-bit depth maps corresponding to Camera 0 fall sequence. |
| `fall-XX-cam1-rgb/` | PNG images (640x480, 24-bit RGB) | Fall event, Camera 1 (side view), RGB stream | **FACT**: Synchronized side-view color video stream of the same fall event. |
| `fall-XX-cam1-d/` | PNG images (640x480, 16-bit `I;16`) | Fall event, Camera 1 (side view), Depth stream | **FACT**: Synchronized side-view depth map sequence. |
| `URFD/` root level | MP4 files, CSV files | Multi-modal recordings of 70 events | **FACT**: Contains video containers and timestamp/accelerometer data. |
| Subject IDs | None in filenames or CSVs | Actor/subject identities | **INFERENCE**: Subjects cannot be determined from file paths or names alone. |

---

## 3. Data Types Present

| Data Type | Extension | Approx File Count | Primary Location | Example Filenames |
| :--- | :--- | :--- | :--- | :--- |
| **RGB Images** | `.png` | 14,932 | `URFD/{event}-{cam}-rgb/` | `adl-01-cam0-rgb-001.png`, `fall-01-cam0-rgb-015.png` |
| **Depth Images** | `.png` | 14,931 | `URFD/{event}-{cam}-d/` | `adl-01-cam0-d-001.png`, `fall-01-cam1-d-020.png` |
| **RGB Videos** | `.mp4` | 100 | `URFD/` | `adl-01-cam0.mp4`, `fall-01-cam0.mp4`, `fall-01-cam1.mp4` |
| **Depth Videos** | — | 0 | None (Only PNG sequences exist) | None |
| **CSV Data Files** | `.csv` | 72 | `URFD/` | `adl-01-data.csv`, `fall-01-data.csv` |
| **Accelerometer Data**| Embedded | 31 CSVs | `URFD/fall-XX-data.csv` | Column 3 in `fall-01-data.csv` to `fall-30-data.csv` |
| **Timestamps** | Embedded | 72 CSVs | `URFD/{event}-data.csv` | Column 2 (milliseconds) in all CSV files |
| **Frame Labels** | — | 0 | None found in local directory | None |
| **Subject Metadata** | — | 0 | None found in local directory | None |

---

## 4. Fall Data Analysis

- **Total Fall Sequences**: 30 distinct fall events (`fall-01` through `fall-30`).
- **Naming Pattern**: `fall-XX-camY-modality`
- **Camera Views**: Every fall sequence contains **2 synchronized camera views**:
  - `cam0`: Overhead / front-facing view.
  - `cam1`: Side-facing view.
- **Modalities per Fall Event**:
  - `cam0` RGB PNG sequence + `cam0` Depth PNG sequence + `cam0` MP4 video
  - `cam1` RGB PNG sequence + `cam1` Depth PNG sequence + `cam1` MP4 video
  - Synchronized CSV data file containing accelerometer readings.
- **Multiple Files per Fall Event**: Yes! For example, `fall-01` consists of 7 distinct files/folders:
  1. `fall-01-cam0-rgb/` (160 RGB PNG frames)
  2. `fall-01-cam0-d/` (160 Depth PNG frames)
  3. `fall-01-cam0.mp4` (160 frames video)
  4. `fall-01-cam1-rgb/` (160 RGB PNG frames)
  5. `fall-01-cam1-d/` (160 Depth PNG frames)
  6. `fall-01-cam1.mp4` (160 frames video)
  7. `fall-01-data.csv` (160 rows of timestamps & accelerometer data)

---

## 5. Normal / ADL Data Analysis

- **Total ADL Sequences**: 40 distinct normal activity events (`adl-01` through `adl-40`).
- **Organization & Naming**: Organized by `adl-XX-cam0-modality`.
- **Camera Views**: ADL events contain **only 1 camera view** (`cam0`). There is no `cam1` for ADL sequences.
- **Modalities Available**:
  - `cam0` RGB PNG sequence
  - `cam0` Depth PNG sequence
  - `cam0` MP4 video
  - `adl-XX-data.csv` (2 columns: frame index, timestamp)
- **Identifiable Activities**: File naming does not describe the specific ADL activity (e.g. sitting, walking, picking up object). Visual inspection of RGB images confirms activities such as sitting down, bending over, walking, lying down intentionally on a couch/bed.
- **Concrete Example (`adl-01`)**:
  - `adl-01-cam0-rgb/` (150 RGB PNG frames)
  - `adl-01-cam0-d/` (150 Depth PNG frames)
  - `adl-01-cam0.mp4` (150 frames video)
  - `adl-01-data.csv` (150 rows: frame # and timestamp ms)

---

## 6. Camera Information

- **Camera Identification**: Explicitly designated as `cam0` and `cam1` in folder and file names.
- **Camera Viewpoints**:
  - `cam0`: Mounted higher up, providing an angled top-down/front view.
  - `cam1`: Mounted at standing height, providing an orthogonal side view.
- **Multi-View Synchronization**:
  - For all 30 Fall events, `cam0` and `cam1` capture the **exact same physical event simultaneously**.
  - Example: `fall-01-cam0-rgb-050.png` and `fall-01-cam1-rgb-050.png` represent frame 50 of Fall #1 recorded from camera 0 and camera 1 at the same moment.
  - ADL events (`adl-01` to `adl-40`) are single-view (`cam0` only).

---

## 7. Event Relationships & Data Hierarchy

### Fall Event Structure (Dual Camera)
```
Fall Event (e.g. fall-01)
├── Camera 0 (cam0)
│   ├── RGB Image Stream: fall-01-cam0-rgb/ (160 PNGs)
│   ├── Depth Image Stream: fall-01-cam0-d/ (160 PNGs)
│   └── Video Stream: fall-01-cam0.mp4 (160 frames)
├── Camera 1 (cam1)
│   ├── RGB Image Stream: fall-01-cam1-rgb/ (160 PNGs)
│   ├── Depth Image Stream: fall-01-cam1-d/ (160 PNGs)
│   └── Video Stream: fall-01-cam1.mp4 (160 frames)
└── Sensor & Synchronization Data
    └── fall-01-data.csv (160 rows: Frame, Timestamp ms, Accel Mag g)
```

### ADL Event Structure (Single Camera)
```
ADL Event (e.g. adl-01)
├── Camera 0 (cam0)
│   ├── RGB Image Stream: adl-01-cam0-rgb/ (150 PNGs)
│   ├── Depth Image Stream: adl-01-cam0-d/ (150 PNGs)
│   └── Video Stream: adl-01-cam0.mp4 (150 frames)
└── Synchronization Data
    └── adl-01-data.csv (150 rows: Frame, Timestamp ms)
```

---

## 8. Video Inspection

- **File Count**: 100 `.mp4` video files (40 for ADL, 60 for Fall).
- **Container Format**: MP4 (ISO/IEC 14496-14 Base Media Format).
- **Resolution**: 640 x 480 pixels.
- **Frame Rate (FPS)**: ~30 FPS (derived from timestamp intervals: ~33.3 ms per frame).
- **Duration**: Varies per sequence, from ~1.8 seconds (55 frames) up to ~13.3 seconds (400 frames).
- **Frame Counts**: Exactly matches the RGB `.png` frame count in 99 out of 100 video files.
- **Redundancy**: The `.mp4` files are direct video encodes of the `.png` image frames.

---

## 9. Image Inspection

### RGB Images
- **Format**: PNG (Portable Network Graphics).
- **Dimensions**: 640 x 480 pixels.
- **Color Mode**: `RGB` (24-bit color, 8 bits per channel across 3 color channels: Red, Green, Blue).
- **Pixel Value Range**: 0 to 255 per channel.
- **Naming Pattern**: `{event}-{cam}-rgb-{3-digit-frame}.png` (e.g. `fall-05-cam0-rgb-088.png`).

### Depth Images
- **Format**: PNG (Portable Network Graphics).
- **Dimensions**: 640 x 480 pixels.
- **Color Mode**: `I;16` (16-bit unsigned integer, single channel raw depth).
- **Pixel Value Range**: 0 to 65,535 (represents distance in millimeters from the Microsoft Kinect sensor).
- **Naming Pattern**: `{event}-{cam}-d-{3-digit-frame}.png` (e.g. `fall-05-cam0-d-088.png`).
- **Frame Numbering**: Zero-padded 3-digit integer matching the RGB frame number exactly.

---

## 10. CSV & Sensor Data Inspection

- **Header Row**: None (all CSV files are headerless numeric tables).
- **Row Counts**: Ranging from 55 rows to 400 rows per file.

### Structure Breakdown

#### ADL CSV Files (`adl-XX-data.csv` - 41 files including duplicate)
- **Columns**: 2 columns
  - `Column 0`: Frame index (1-based integer: `1, 2, 3, ...`)
  - `Column 1`: Timestamp in milliseconds (`0, 33, 66, 100, 133, ...`)
- **Sample Records (`adl-01-data.csv`)**:
  ```csv
  1,0
  2,33
  3,66
  4,100
  5,133
  ```

#### Fall CSV Files (`fall-XX-data.csv` - 31 files including duplicate)
- **Columns**: 3 columns
  - `Column 0`: Frame index (1-based integer: `1, 2, 3, ...`)
  - `Column 1`: Timestamp in milliseconds (`0, 33, 66, 100, 133, ...`)
  - `Column 2`: Accelerometer total acceleration magnitude (in units of `g` gravitational acceleration). Resting value ~1.0g, peak impact values up to 11.4g.
- **Sample Records (`fall-01-data.csv`)**:
  ```csv
  1,0,1.02490
  2,33,1.00320
  3,66,1.02850
  4,100,0.98591
  5,133,0.98154
  ```

---

## 11. Subject Information

- **Observation**: There are NO subject IDs, person names, age/gender metadata, or participant tracking codes anywhere in the filenames, folder names, or CSV files within `URFD/`.
- **Finding**: Subject identity and distribution across sequences **cannot be determined** from the files present in this dataset directory.

---

## 12. Dataset Completeness & Anomalies

### 1. Frame Count Mismatches
- **`fall-03-cam1`**: `fall-03-cam1-rgb` has **216 PNG frames**, `fall-03-cam1-d` has **215 PNG frames**, `fall-03-data.csv` has **215 rows**. (1 extra RGB frame exists for camera 1).
- **`adl-37`**: `adl-37-data.csv` has **330 rows**, but `adl-37-cam0-rgb` and `adl-37-cam0-d` have **350 PNG frames**. (Missing last 20 timestamps in CSV).

### 2. Duplicate CSV Files
- `adl-15-data (1).csv` is an exact duplicate of `adl-15-data.csv` (2,608 bytes).
- `fall-11-data (1).csv` is an exact duplicate of `fall-11-data.csv` (1,965 bytes).

### 3. Nested Directory Redundancy
- 198 out of the 200 image subfolders contain a redundant nested folder with the exact same name (e.g. `URFD/adl-01-cam0-d/adl-01-cam0-d/adl-01-cam0-d-001.png`).

---

## 13. Important Observations for Future Work

1. **Dual-Camera vs. Single-Camera Asymmetry**: Falls are recorded from 2 views (`cam0` and `cam1`), while ADLs are recorded from 1 view (`cam0`).
2. **Data Redundancy**: `.mp4` files and `.png` image folders contain the same RGB visual data.
3. **Depth Format**: Depth images are stored as 16-bit raw depth (`I;16`, values in mm), requiring 16-bit image decoders.
4. **Missing Ground-Truth Annotations**: Frame-level fall labels (start of fall, impact frame, end of fall) are missing from the local dataset folder.

---

## 14. Final Summary Report

- **A. What is this dataset?** The UR Fall Detection Dataset (URFD), recorded using Microsoft Kinect v1 sensors and wearable accelerometers.
- **B. What folders are present?** 200 stream subdirectories inside `URFD/` representing 100 RGB image folders and 100 Depth image folders across 70 recorded events.
- **C. What data modalities are available?** RGB images (24-bit PNG), Depth images (16-bit `I;16` PNG in mm), RGB videos (MP4), Tri-axial accelerometer magnitude data (in `fall-XX-data.csv`), and Timestamps in ms.
- **D. How are falls represented?** By 30 fall events (`fall-01` to `fall-30`), each containing 2 camera views (`cam0` and `cam1`), RGB PNGs, Depth PNGs, MP4 video, and an accelerometer CSV.
- **E. How are normal activities represented?** By 40 ADL events (`adl-01` to `adl-40`), each containing 1 camera view (`cam0`), RGB PNGs, Depth PNGs, MP4 video, and a timestamp CSV.
- **F. How are cameras represented?** Designated as `cam0` (front/overhead) and `cam1` (side view, fall events only).
- **G. How are subjects/events represented?** Events are indexed numerically (`adl-01` to `adl-40`, `fall-01` to `fall-30`). Subject IDs are NOT present in the files.
- **H. What files appear to belong to the same event?** All files sharing the prefix `{type}-{index}` belong to the same event.
- **I. What important data-quality issues exist?** 2 duplicate CSV files, 1 missing depth frame in `fall-03-cam1`, 20 missing CSV timestamp rows in `adl-37`, 198 nested duplicate folder names, and lack of frame-level fall labels in local files.
- **J. What is still unclear?** Ground-truth fall start/end frame indices and subject demographic distribution.
- **K. What should we inspect next?** Inspect whether external official URFD ground-truth frame-level annotation files can be located or referenced.
