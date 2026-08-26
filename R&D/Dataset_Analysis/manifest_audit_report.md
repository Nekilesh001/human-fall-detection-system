# Master Dataset Manifest Read-Only Audit Report

## 1. Executive Summary

This report presents a thorough, read-only audit of `d:\ONE_DATA\Fall detection\R&D\Dataset_Analysis\dataset_manifest.csv`. Every label, path, metadata field, and event-camera relationship in the manifest was audited against the raw dataset evidence and published dataset documentation.

No files were modified, split, processed, or generated during this audit.

---

## 2. Event-Level vs. Video-Level Inventory

| Dataset | Metric Level | Total Count | FALL | NORMAL | UNKNOWN | Camera Structure |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **URFD** | Event Level | **70** | 30 | 40 | 0 | 30 dual-cam (Fall), 40 single-cam (ADL) |
| **URFD** | Video Level | **100** | 60 | 40 | 0 | 60 Fall videos (cam0+cam1), 40 ADL videos (cam0) |
| **Le2i** | Event Level | **190** | 96 | 31 | 63 | 190 single-cam video events |
| **Le2i** | Video Level | **190** | 96 | 31 | 63 | 96 Fall, 31 Normal, 63 Unknown |
| **MultiCamera**| Event Level | **24** | 22 | 2 | 0 | 24 multi-view scenarios (8 cams each) |
| **MultiCamera**| Video Level | **192** | 176 | 16 | 0 | 176 Fall videos, 16 Normal videos |
| **TOTAL** | **Event Level** | **284** | **148** | **73** | **63** | **284 unique events** |
| **TOTAL** | **Video Level** | **482** | **332** | **87** | **63** | **482 video/observation records** |

---

## 3. URFD Label Audit

- **Verification**:
  - `fall-01` through `fall-30` (30 events, 60 video records) are all verified as `FALL`.
  - Both `cam0` and `cam1` for every fall event share the identical `FALL` label.
  - `adl-01` through `adl-40` (40 events, 40 video records) are all verified as `NORMAL`.
  - ADL events contain only 1 camera (`cam0`), which is correctly reflected in the manifest.

### Complete URFD Event Mapping (70 Events)

| Event ID | Video Count | Camera IDs | Event Label | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| `adl-01` .. `adl-40` (40 events) | 1 video each (40 total) | `['cam0']` | `NORMAL` | **VERIFIED** (Folder prefix `adl-`) |
| `fall-01` .. `fall-30` (30 events) | 2 videos each (60 total) | `['cam0', 'cam1']` | `FALL` | **VERIFIED** (Folder prefix `fall-`) |

---

## 4. Le2i Label Audit

### Label Breakdown (190 Videos)
- **FALL (96 Videos)**: Verified from `.txt` header where `StartFrame > 0` and `EndFrame > 0`.
- **NORMAL (31 Videos)**: Verified from `.txt` header where `StartFrame == 0` and `EndFrame == 0`.
- **UNKNOWN (63 Videos)**: Kept strictly as `UNKNOWN`.

### Audit of UNKNOWN Records (63 Videos)

| Location | Video Count | Reason for UNKNOWN Label | Annotation File Status |
| :--- | :--- | :--- | :--- |
| `Office` | 33 videos | Unannotated in official download | `UNKNOWN` (No `.txt` file exists) |
| `Lecture_room` | 27 videos | Unannotated in official download | `UNKNOWN` (No `.txt` file exists) |
| `Coffee_room_01` | 1 video (`v26`) | Malformed `.txt` header | `.txt` exists, but omits 2-line header |
| `Coffee_room_02` | 2 videos (`v50`, `v52`)| Malformed `.txt` header | `.txt` exists, but omits 2-line header |

> [!IMPORTANT]
> None of the 63 UNKNOWN records were converted into NORMAL or FALL. They remain strictly `UNKNOWN` in accordance with rules.

---

## 5. MultiCamera Label Audit

### Audit of Assigned Labels (176 FALL / 16 NORMAL)
- **Observations**: The local directory `dataset/dataset/` contains 24 `chuteXX` folders with 8 `.avi` videos each (192 videos), but **0 local text annotation files**.
- **Manifest Assignment**:
  - `chute01` through `chute22` (22 events x 8 cams = 176 videos) -> Assigned `FALL`.
  - `chute23` and `chute24` (2 events x 8 cams = 16 videos) -> Assigned `NORMAL`.
- **Source of Labels**: Derived from published **OFFICIAL CREATOR DOCUMENTATION** (Edouard Auvinet et al., 2010, DIRO Technical Report 1350, Université de Montréal).
- **Classification**: `OFFICIAL_DOCUMENTATION` (Not directly observed in local text files, but documented by dataset creators).

### Scenario-by-Scenario Audit (24 Chutes)

| Event ID | Camera Count | Camera IDs | Assigned Label | Label Source | Evidence / Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `chute01` .. `chute22` | 8 cams each (176 total) | `['cam1'..'cam8']` | `FALL` | `OFFICIAL_DOCUMENTATION` | Auvinet et al. 2010 DIRO Report 1350 |
| `chute23` .. `chute24` | 8 cams each (16 total) | `['cam1'..'cam8']` | `NORMAL` | `OFFICIAL_DOCUMENTATION` | Auvinet et al. 2010 DIRO Report 1350 (Confounding ADLs) |

---

## 6. Label Source Audit Summary

| Dataset | Label Value | Source Category | Evidence / Basis |
| :--- | :--- | :--- | :--- |
| **URFD** | `FALL` / `NORMAL` | `LOCAL_FILE` & `OFFICIAL_DOC` | Directory prefix (`fall-` / `adl-`) & Kwolek & Kepski 2014 paper. |
| **Le2i** | `FALL` / `NORMAL` | `LOCAL_FILE` & `OFFICIAL_DOC` | 2-line header in `.txt` annotation files & Charfi et al. 2013 paper. |
| **Le2i** | `UNKNOWN` | `LOCAL_FILE` | Missing `.txt` file (60 videos) or malformed header (3 videos). |
| **MultiCamera** | `FALL` / `NORMAL` | `OFFICIAL_DOCUMENTATION` | Published Auvinet et al. 2010 DIRO Technical Report 1350 (Scenarios 1-22 Falls, 23-24 ADLs). |

---

## 7. Event / Camera Relationship Audit

- **URFD**: Verified that 30 Fall events have 2 cameras (`cam0`, `cam1`) sharing the same `event_id`, and 40 ADL events have 1 camera (`cam0`).
- **Le2i**: Verified that each video observation has a unique `event_id` (`{location}_v{num}`).
- **MultiCamera**: Verified that all 24 chute scenarios have 8 cameras (`cam1`..`cam8`) sharing the same `event_id`.
- **Result**: **0 cross-event camera misassignments detected**.

---

## 8. Video Metadata Audit

- **FPS Audit**:
  - URFD: 30.0 FPS (Standard Kinect v1 stream rate).
  - Le2i: 25.0 FPS (`Coffee_room`, `Lecture_room`, `Office`) and 24.0 FPS (`Home_01`, `Home_02`).
  - MultiCamera: 25.0 FPS (Real-time IP camera rate; header timescale scales to 120 FPS).
- **Resolution Audit**:
  - URFD: 640 x 480 pixels (100% uniform).
  - Le2i: 320 x 240 pixels (165 videos) and 320 x 180 pixels (25 videos in `Home_02`).
  - MultiCamera: 720 x 480 pixels (100% uniform).
- **Discrepancy Flag**: `Home_02` resolution (320x180) differs from other Le2i locations (320x240). Must be handled during spatial preprocessing.

---

## 9. Path Audit

| Path Column | Valid Path Count | Missing Path Count | Unknown / Not Applicable Count | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| `video_path` | **482** | **0** | 0 | **100% VALID** |
| `annotation_path` | **130** | **0** | 352 | **100% VALID** |
| `depth_path` | **100** | **0** | 382 | **100% VALID** |
| `rgb_path` | **100** | **0** | 382 | **100% VALID** |
| `timestamp_path` | **100** | **0** | 382 | **100% VALID** |
| `accelerometer_path` | **60** | **0** | 422 | **100% VALID** |

---

## 10. Data Leakage Audit

- **Atomic Split Unit**: Event ID (`event_id`) is strictly confirmed as the atomic split unit.
- **Multi-Camera Binding**: Confirmed. Cameras belonging to the same physical event share `event_id`.
- **Conflicting Labels**: 0 events have conflicting labels across camera views.
- **Duplicate Records**: 0 duplicate `event_id` / `camera_id` combinations exist.

---

## 11. Final Audit Verdict

### A. What is definitely correct?
- All 482 video record file paths exist on disk.
- All 284 event IDs correctly group multi-camera observations.
- URFD 70 event labels and Le2i 127 annotated video labels are directly supported by local file evidence.

### B. What requires attention / clear reporting?
- MultiCamera labels (`chute01`..`chute22` Fall, `chute23`..`chute24` Normal) are derived from official creator literature (Auvinet 2010), not from local text files.
- Le2i `Home_02` resolution is 320x180 (vs 320x240 in other Le2i locations).

### C. What is ambiguous?
- Le2i 3 malformed annotation files (`Coffee_room_01/v26`, `Coffee_room_02/v50`, `Coffee_room_02/v52` contain bounding boxes but lack 2-line start/end headers).

### D. Which labels are trustworthy enough for supervised training?
- **URFD**: All 100 video records (70 events) for event-level classification.
- **Le2i**: 127 video records (96 Fall, 31 Normal) for frame-level and event-level classification.
- **MultiCamera**: All 192 video records (24 events) for event-level classification based on Auvinet 2010 documentation.

### E. Which records must remain UNKNOWN?
- 63 Le2i video records (60 unannotated in `Lecture_room`/`Office` + 3 malformed header videos).

### F. Is the manifest ready to be used for preprocessing?
- **YES**. The manifest `dataset_manifest.csv` is fully verified, indexed, and structurally sound for group-level data splitting and spatial/temporal preprocessing.
