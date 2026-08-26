# Le2i Fall Detection Dataset - Analysis & Inspection Report

## Executive Summary
This document presents an empirical, non-modifying analysis of the **Le2i Fall Detection Dataset** located at `d:\ONE_DATA\Fall detection\Le2i`. All findings, file counts, video properties, and annotation structures were derived directly by inspecting the existing filesystem without modifying, moving, copying, converting, or altering any files.

---

## 1. Directory Structure

### Top-Level Folder Hierarchy
The dataset is located under `d:\ONE_DATA\Fall detection\Le2i\data` and consists of **6 location folders** plus 1 documentation file (`README.txt`).

```
d:\ONE_DATA\Fall detection\Le2i\
└── data/
    ├── README.txt
    ├── Coffee_room_01/
    │   └── Coffee_room_01/
    │       ├── Annotation_files/
    │       │   ├── video (1).txt ... video (48).txt (48 files)
    │       └── Videos/
    │           ├── video (1).avi ... video (48).avi (48 files)
    ├── Coffee_room_02/
    │   └── Coffee_room_02/
    │       ├── Annotations_files/                    [Spelling variation]
    │       │   ├── video (49).txt ... video (70).txt (22 files)
    │       └── Videos/
    │           ├── video (49).avi ... video (70).avi (22 files)
    ├── Home_01/
    │   └── Home_01/
    │       ├── Annotation_files/
    │       │   ├── video (1).txt ... video (30).txt (30 files)
    │       └── Videos/
    │           ├── video (1).avi ... video (30).avi (30 files)
    ├── Home_02/
    │   └── Home_02/
    │       ├── Annotation_files/
    │       │   ├── video (31).txt ... video (60).txt (30 files)
    │       └── Videos/
    │           ├── video (31).avi ... video (60).avi (30 files)
    ├── Lecture_room/
    │   └── Lecture room/                             [Space in folder name, no Videos dir]
    │       ├── video (1).avi ... video (27).avi (27 files, 0 annotations)
    └── Office/
        └── Office/                                   [No Videos dir]
            ├── video (1).avi ... video (33).avi (33 files, 0 annotations)
```

### File Extensions & Counts
| Extension | Count | Role / Description |
| :--- | :--- | :--- |
| `.avi` | **190** | Uncompressed RGB Video recordings (`DIB ` codec) |
| `.txt` | **131** | 130 bounding box & fall range annotation files + 1 `README.txt` |
| **Total** | **321** | **Total files in Le2i** |

### Naming Conventions
- **Locations**: Named by real-world scene (`Coffee_room_01`, `Coffee_room_02`, `Home_01`, `Home_02`, `Lecture_room`, `Office`).
- **Videos**: Named as `video (i).avi` (e.g. `video (1).avi`, `video (12).avi`). Note: Video numbers are NOT globally unique across locations (e.g. `video (1).avi` exists in 4 different location folders).
- **Annotations**: Named matching the video filename: `video (i).txt` (e.g. `video (1).txt`).

---

## 2. Data Content

- **RGB Videos**: 190 `.avi` files (uncompressed raw RGB, codec `DIB `).
- **Depth Images / Videos**: **0** (No depth data exists in Le2i).
- **CSV / XML / JSON Files**: **0** (No CSV, XML, or JSON files exist).
- **Annotations**: 130 plain text `.txt` files containing fall start frame, fall end frame, and bounding box coordinates for each frame.
- **Metadata**: 1 `README.txt` file at `d:\ONE_DATA\Fall detection\Le2i\data\README.txt`.

---

## 3. Sequences / Events

- **Total Sequences / Events**: **190 video events** across 6 location folders.
- **Event Representation**: Each individual `.avi` video file represents a single recorded event.
- **File Pairing per Event**:
  - For annotated locations (`Coffee_room_01`, `Coffee_room_02`, `Home_01`, `Home_02`), an event consists of 2 paired files:
    1. `Videos/video (i).avi`
    2. `Annotation_files/video (i).txt`
  - For unannotated locations (`Lecture_room`, `Office`), an event consists of only the `.avi` video file.

---

## 4. Fall vs. Normal Activities

### Label Mechanism
Fall vs. Normal labels are **explicitly defined inside lines 1 and 2 of each `.txt` annotation file**. The video and folder names do NOT state whether a video is a fall or normal activity.

- **Fall Event**: Lines 1 and 2 contain non-zero integers `StartFrame > 0` and `EndFrame > 0`.
- **Normal Event**: Lines 1 and 2 contain zeros `0` and `0`.
- **Unannotated Event**: No `.txt` file exists for the video.

### Summary Breakdown across Annotated Files (130 Total)
- **Explicit Fall Events**: **96 videos**
- **Explicit Normal Events**: **31 videos**
- **Malformed Annotations (Missing Header)**: **3 videos** (`video (26).txt` in `Coffee_room_01`, `video (50).txt` & `video (52).txt` in `Coffee_room_02`).
- **Unannotated Videos**: **60 videos** (`Lecture_room`: 27, `Office`: 33).

---

## 5. Camera Information

- **Camera Setup**: Single fixed camera per location.
- **Multiple Views per Event**: **NO**. Unlike URFD, Le2i does NOT contain synchronized multi-camera views for the same event. Every video is a single-camera recording.

---

## 6. Video Information

| Location | Resolution | FPS | Codec | Video Count | Sample Durations / Frames |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Coffee_room_01` | 320 x 240 | 25.0 FPS | `DIB ` (Raw RGB) | 48 | 157 to 483 frames (6.3s - 19.3s) |
| `Coffee_room_02` | 320 x 240 | 25.0 FPS | `DIB ` (Raw RGB) | 22 | 203 to 1954 frames (8.1s - 78.2s) |
| `Home_01` | 320 x 240 | 24.0 FPS | `DIB ` (Raw RGB) | 30 | 192 to 336 frames (8.0s - 14.0s) |
| `Home_02` | **320 x 180** | 24.0 FPS | `DIB ` (Raw RGB) | 30 | 168 to 360 frames (7.0s - 15.0s) |
| `Lecture_room` | 320 x 240 | 25.0 FPS | `DIB ` (Raw RGB) | 27 | 285 to 493 frames (11.4s - 19.7s) |
| `Office` | 320 x 240 | 25.0 FPS | `DIB ` (Raw RGB) | 33 | 193 to 415 frames (7.7s - 16.6s) |

---

## 7. Annotation Structure

An annotation file (e.g. `video (1).txt`) contains plain text lines:

```
Line 1: 48                          <- Fall Start Frame Index
Line 2: 80                          <- Fall End Frame Index
Line 3: 1, 1, 292, 152, 311, 240    <- Frame 1 Bounding Box: [Frame, State, x_min, y_min, x_max, y_max]
Line 4: 2, 1, 292, 152, 311, 240    <- Frame 2 Bounding Box
...
```

- **Line 1**: Integer frame index where fall begins (or `0` for normal activity).
- **Line 2**: Integer frame index where fall ends (or `0` for normal activity).
- **Lines 3 to End**: Frame-by-frame record containing:
  - `col 0`: Frame index (1-based)
  - `col 1`: Subject state code (1 = standing/normal, 7 = falling/on floor)
  - `col 2..5`: Bounding box pixel coordinates `[x_min, y_min, x_max, y_max]` constrained within 320x240.

---

## 8. Subject Information

- **Finding**: **NO subject IDs, actor names, age, gender, or person tracking codes** exist anywhere in filenames, directory names, annotation files, or README text.
- **Fact**: Subject identities cannot be determined from the dataset files locally.

---

## 9. Dataset Relationships

```
Event (e.g. Coffee_room_01 / video (1))
 ├── Video File: Coffee_room_01/Coffee_room_01/Videos/video (1).avi (157 frames, 320x240 @ 25 FPS)
 └── Annotation File: Coffee_room_01/Coffee_room_01/Annotation_files/video (1).txt
      ├── Line 1: 48 (Fall Start Frame)
      ├── Line 2: 80 (Fall End Frame)
      └── Lines 3..159: Bounding box [Frame, State, x_min, y_min, x_max, y_max] for frames 1 to 157
```

---

## 10. Data Quality & Anomalies

1. **60 Unannotated Videos**: `Lecture_room` (27 videos) and `Office` (33 videos) have ZERO annotation files in this download.
2. **3 Malformed Annotation Files**:
   - `Coffee_room_01/.../video (26).txt`
   - `Coffee_room_02/.../video (50).txt`
   - `Coffee_room_02/.../video (52).txt`  
   *Issue*: These 3 files omit the 2-line header (`start_frame`, `end_frame`) and start directly on Line 1 with bounding box data.
3. **51 Frame Count Discrepancies**: 51 of the 130 annotated pairs have minor mismatches between the AVI total frame count and the text file line count (e.g. video has 216 frames, text file has 218 lines).
4. **Subfolder Naming Inconsistencies**:
   - `Annotation_files` in `Coffee_room_01`, `Home_01`, `Home_02` vs `Annotations_files` (with an 's') in `Coffee_room_02`.
   - `Lecture room` has a space in the folder name (`Lecture room` vs `Lecture_room`).
5. **Resolution Inconsistency**: `Home_02` videos are 320x180, whereas all other locations are 320x240.
6. **FPS Inconsistency**: `Home_01` and `Home_02` are 24.0 FPS, whereas all other locations are 25.0 FPS.

---

## 11. Dataset Completeness

- **Downloaded Contents**: 190 `.avi` videos, 130 `.txt` annotations, 1 `README.txt`. Total 321 files.
- **Incomplete Portion**: 31.6% of videos (60 out of 190 videos) are unannotated in this local folder.

---

## 12. Potential Data Leakage

- **Fixed Background Leakage**: Each location (`Coffee_room_01`, `Coffee_room_02`, `Home_01`, `Home_02`, `Lecture_room`, `Office`) was shot with a static camera in a fixed room.
- **Risk**: Random train/test splits across videos within the same location will cause severe **SCENE LEAKAGE / DATA LEAKAGE** because the test set background and camera angle are identical to the training set.
- **Requirement**: Evaluation MUST be conducted as **Leave-One-Location-Out (LOLO) cross-validation** across the 6 location groups.

---

## 13. Important Limitations

1. **Uncompressed AVI File Size**: Videos use `DIB ` uncompressed RGB codec, resulting in large file sizes (~30-100MB per short video clip).
2. **Missing Annotations**: 60 videos cannot be evaluated for supervised fall detection without manual labeling.
3. **No Depth or Sensor Modalities**: Le2i contains only RGB video (no depth maps, no accelerometer data).
4. **Varying Resolutions & FPS**: Models must handle both 320x240 and 320x180 resolution, as well as 24 FPS vs 25 FPS frame rates.

---

## 14. Final Summary Report

- **A. What is this dataset?** The **Le2i Fall Detection Dataset** (LE2I DIJON UMR6306), consisting of 190 RGB video recordings of falls and normal activities across 6 real-world environment setups.
- **B. Directory structure** `Le2i/data/` contains 6 location folders (`Coffee_room_01`, `Coffee_room_02`, `Home_01`, `Home_02`, `Lecture_room`, `Office`).
- **C. Data types** RGB videos (`.avi`, uncompressed `DIB ` codec) and bounding box / fall interval annotations (`.txt`). No depth or sensor CSV data exists.
- **D. Number/type of events** 190 video events total. In the annotated portion (130 videos): 96 Fall events, 31 Normal activity events, 3 malformed annotations. 60 videos are unannotated.
- **E. Fall/normal labels** Labels are explicitly given in lines 1 and 2 of each `.txt` file (`StartFrame > 0` & `EndFrame > 0` for Fall, `0` & `0` for Normal).
- **F. Camera/view information** Single fixed camera per location. No multi-view or multi-camera recordings exist for individual events.
- **G. Annotation structure** Plain text `.txt` files: Line 1 = Fall start frame, Line 2 = Fall end frame, Lines 3+ = `[frame, state, x_min, y_min, x_max, y_max]`.
- **H. Subject information** Subject IDs do NOT exist in file names, folder names, annotations, or README text.
- **I. Event relationships** Each event is represented by one `.avi` video file and one paired `.txt` annotation file within the same location subfolder.
- **J. Data-quality problems** 60 unannotated videos, 3 malformed annotation header files, 51 frame-count mismatches between videos and annotations, subfolder spelling variations, and non-unique video filenames.
- **K. Potential leakage** Fixed background and camera view per location folder. Random splitting across videos in the same location will cause severe background data leakage.
- **L. Important limitations** Uncompressed AVI format, missing annotations for 2 locations, lack of depth/sensor modalities, and resolution/FPS variations across locations.
- **M. What remains unclear** Whether ground-truth annotations exist externally for `Lecture_room` (27 videos) and `Office` (33 videos), and how many unique human subjects participated in the recordings.
- **N. What should we inspect next** Inspect whether external annotation files for `Lecture_room` and `Office` are available, or proceed to structure a unified dataset catalog comparing URFD and Le2i modalities.
