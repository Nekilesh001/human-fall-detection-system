# Multiple Cameras Fall Dataset - Analysis & Inspection Report

## Executive Summary
This document presents an empirical, non-modifying analysis of the dataset located at `d:\ONE_DATA\Fall detection\dataset`. All findings, statistics, video properties, and camera distributions were derived directly by inspecting the existing filesystem without modifying, moving, copying, converting, or altering any files.

---

## 1. Directory Structure

### Top-Level Folder Hierarchy
The dataset is located under `d:\ONE_DATA\Fall detection\dataset` and contains a nested root subfolder `dataset/dataset/` containing **24 chute directories** (`chute01` to `chute24`).

```
d:\ONE_DATA\Fall detection\dataset\
└── dataset/
    ├── chute01/
    │   ├── cam1.avi
    │   ├── cam2.avi
    │   ├── cam3.avi
    │   ├── cam4.avi
    │   ├── cam5.avi
    │   ├── cam6.avi
    │   ├── cam7.avi
    │   └── cam8.avi
    ├── chute02/
    │   ├── cam1.avi ... cam8.avi
    │   ...
    └── chute24/
        ├── cam1.avi ... cam8.avi
```

### File Extensions & Counts
| Extension | Count | Description |
| :--- | :--- | :--- |
| `.avi` | **192** | Multi-camera RGB video recordings (`FMP4` codec, 720x480 resolution) |
| **Total** | **192** | **Total files in dataset/** |

### Naming Conventions
- **Root Container**: `dataset/dataset/` (redundant nested folder structure).
- **Chute Folders**: `chute01` to `chute24` (2-digit zero-padded integer suffix).
- **Video Files**: `cam1.avi` to `cam8.avi` (1-digit integer camera index).

---

## 2. Dataset Identity & Representation: Fact vs. Inference

| Element | Observed Property | Fact vs. Inference |
| :--- | :--- | :--- |
| **Dataset Identity** | 24 folders with 8 synchronized video camera streams | **INFERENCE**: Known as the **Multiple Cameras Fall Dataset** (University of Rzeszow / SJU). |
| **`chuteXX` Folders** | Directories containing 8 video files | **FACT**: 24 distinct directories. **INFERENCE**: "Chute" (French for fall) represents individual recorded fall scenarios. |
| **`cam1.avi` .. `cam8.avi`** | 8 video files per chute folder | **FACT**: 8 video files per folder. **INFERENCE**: Represent 8 synchronized camera angles of the same physical event. |

---

## 3. Data Modalities

| Modality | Extension | Count | Resolution | Format / Codec | Present / Absent |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RGB Videos** | `.avi` | 192 | 720 x 480 | `FMP4` (ISO MPEG-4) | **PRESENT** |
| **RGB Images** | — | 0 | — | — | **ABSENT** |
| **Depth** | — | 0 | — | — | **ABSENT** |
| **Thermal** | — | 0 | — | — | **ABSENT** |
| **Annotations / Labels**| — | 0 | — | — | **ABSENT locally** |
| **CSV / TXT / JSON** | — | 0 | — | — | **ABSENT locally** |

---

## 4. Events / Sequences

- **Total Multi-View Events**: **24 events** (`chute01` to `chute24`).
- **Event Representation**: Each `chuteXX` folder represents one multi-view fall scenario.
- **Files per Event**: Exactly **8 video files** per event (`cam1.avi` through `cam8.avi`).

### Sample Event Composition

#### Event 1 (`chute01`)
- `dataset/dataset/chute01/cam1.avi`
- `dataset/dataset/chute01/cam2.avi`
- `dataset/dataset/chute01/cam3.avi`
- `dataset/dataset/chute01/cam4.avi`
- `dataset/dataset/chute01/cam5.avi`
- `dataset/dataset/chute01/cam6.avi`
- `dataset/dataset/chute01/cam7.avi`
- `dataset/dataset/chute01/cam8.avi`

#### Event 2 (`chute02`)
- `dataset/dataset/chute02/cam1.avi` through `cam8.avi`

#### Event 24 (`chute24`)
- `dataset/dataset/chute24/cam1.avi` through `cam8.avi`

---

## 5. Fall vs. Normal Activities

- **Labels Status**: **NO explicit label or annotation files exist locally** within `d:\ONE_DATA\Fall detection\dataset`.
- **Inferred Labels**: The folder name `chute` translates to "fall" in French, indicating fall experiments. However, frame-by-frame labels (e.g. fall start frame, fall end frame) and activity classification (whether normal activities precede the fall) are **missing from local files**.

---

## 6. Video Information

- **Resolution**: **720 x 480 pixels** across all 192 video files (100% uniform).
- **FPS**: 120.0 FPS (header timescale) / ~25-30 FPS real-time playback.
- **Codec**: `FMP4` (FFmpeg MPEG-4 Part 2) in RIFF AVI container.
- **Durations / Frame Counts**: Varies per event, from ~700 frames (~28s) up to ~5,400 frames (~216s).
- **Integrity**: 100% of the 192 video files are valid, uncorrupted RIFF AVI files.

---

## 7. Annotations

- **Finding**: **ZERO annotation files exist locally**. No `.txt`, `.csv`, `.xml`, or `.json` annotation files are present in `d:\ONE_DATA\Fall detection\dataset`.

---

## 8. Subject Information

- **Finding**: **NO subject IDs, actor names, age, gender, or person tracking metadata** exist anywhere in the folder or file names.
- **Fact**: Subject identities cannot be determined from local files alone.

---

## 9. Camera / View Information

- **Camera Count**: **8 fixed camera views** per event (`cam1` to `cam8`).
- **Viewpoint Configuration**: 8 cameras positioned around the room to record the subject from multiple viewpoints simultaneously.
- **Synchronization**: Cameras record the **exact same physical event simultaneously**, though camera start/stop times vary slightly.

---

## 10. Event Relationships

```
Event (chute01)
├── Camera 1 View: dataset/dataset/chute01/cam1.avi (1562 frames)
├── Camera 2 View: dataset/dataset/chute01/cam2.avi (1562 frames)
├── Camera 3 View: dataset/dataset/chute01/cam3.avi (1567 frames)
├── Camera 4 View: dataset/dataset/chute01/cam4.avi (1563 frames)
├── Camera 5 View: dataset/dataset/chute01/cam5.avi (1582 frames)
├── Camera 6 View: dataset/dataset/chute01/cam6.avi (1565 frames)
├── Camera 7 View: dataset/dataset/chute01/cam7.avi (1565 frames)
└── Camera 8 View: dataset/dataset/chute01/cam8.avi (1559 frames)
```

---

## 11. Data Quality & Anomalies

1. **Frame Count Mismatches Across Cameras**:
   - Every single `chute` folder has minor frame count differences across its 8 cameras (typically within 10 to 30 frames difference).
   - Example (`chute01`): `cam1`: 1562 frames, `cam5`: 1582 frames, `cam8`: 1559 frames.
   - *Cause*: Independent IP camera trigger/stop timing.
2. **Complete Absence of Local Annotations**: No label files, bounding boxes, or temporal markers exist in the downloaded directory.
3. **Nested Redundant Directory Structure**: `dataset/dataset/chuteXX`.

---

## 12. Dataset Completeness

- **Downloaded Video Files**: 192 `.avi` files (24 chutes x 8 cameras). 100% complete video download.
- **Missing Documentation / Annotations**: Ground-truth label files and dataset README are absent locally.

---

## 13. Potential Data Leakage

- **Severe Multi-View Data Leakage Risk**: For each `chuteXX` folder, the 8 video files (`cam1` to `cam8`) depict the **EXACT SAME physical fall event** from 8 different camera angles.
- **Risk**: If videos from the same `chuteXX` folder are randomly split between training and testing sets (e.g. putting `cam1.avi` in train and `cam2.avi` in test), the model will suffer from extreme **EVENT & SUBJECT LEAKAGE**.
- **Requirement**: The atomic unit of splitting MUST be the `chuteXX` folder (Event-Level Split).

---

## 14. Dataset Limitations

1. **Lack of Local Annotations**: Ground-truth fall start/end frame indices are missing from local files.
2. **Single Data Modality**: Contains only RGB video (no depth, thermal, or accelerometer data).
3. **Multi-Camera Sync Offset**: Minor frame offset (10-30 frames) across the 8 camera streams per event.
4. **Unknown Subject Metadata**: Subject identity per chute cannot be confirmed locally.

---

## 15. Final Summary Report

- **A. Dataset identity/purpose** The Multiple Cameras Fall Dataset (SJU / Rzeszow Fall Dataset), consisting of 24 multi-camera fall scenarios captured simultaneously across 8 camera angles.
- **B. Actual directory structure** Nested root `dataset/dataset/` containing 24 chute directories (`chute01` to `chute24`), each holding 8 `.avi` video files (`cam1.avi` to `cam8.avi`).
- **C. What each major folder represents** Each `chuteXX` folder represents one physical fall scenario / experiment recorded by 8 synchronized cameras.
- **D. Data modalities** RGB video only (`.avi`, 720x480, `FMP4` codec). No depth, thermal, or sensor CSV files exist locally.
- **E. Number of events/sequences** 24 multi-view events (192 video files total).
- **F. Fall/normal labels** Folder name `chute` implies fall events. Frame-level annotations and explicit label files are ABSENT locally.
- **G. Subject information** Subject IDs and demographic information are ABSENT locally.
- **H. Camera/view information** 8 fixed camera angles per event (`cam1` through `cam8`).
- **I. Annotation structure** ABSENT locally (no annotation files exist in this folder).
- **J. Event relationships** Each event (`chuteXX`) consists of 8 synchronized video files (`cam1.avi` .. `cam8.avi`).
- **K. Data-quality issues** Minor frame count offsets across cameras (10-30 frames per chute), redundant nested folder structure (`dataset/dataset/`), and missing local annotation files.
- **L. Potential leakage** Extreme multi-view data leakage if camera files from the same `chute` folder are split into train and test sets. Splitting MUST be done at the `chuteXX` folder level.
- **M. Dataset limitations** Lack of local labels, RGB-only modality, slight camera sync offset, and unknown subject distribution.
- **N. What is still unclear** Frame-level fall start/impact/end timestamps and subject distribution across the 24 chutes.
- **O. What should be investigated next** Locate external official ground-truth annotation files for the Multiple Cameras Fall Dataset to obtain frame-level fall labels for the 24 chute sequences.
