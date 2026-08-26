# Phase 1.2 — Experimental Definition & Evaluation Protocols

## Executive Summary
This document formalizes the **experimental terminology, data hierarchy, label semantics, splitting rules, evaluation metrics, and dataset roles** for our Human Fall Detection System prior to initiating any data preprocessing, windowing, or model design.

All definitions and constraints are strictly derived from the empirical findings established in our dataset inspection and ground-truth verification phase.

---

## 1. EVENT

### Definition
An **EVENT** is defined as one complete, independent, continuous real-world physical performance of an action (either a fall or an activity of daily living) by a human participant in a specific spatial environment.

### Dataset-Specific Event Boundaries
- **URFD**: Each sequence prefixed by `fall-XX` (30 events) or `adl-XX` (40 events) represents 1 independent event. **Total: 70 events**.
- **Le2i**: Each `.avi` video recording within a specific location folder represents 1 independent event. **Total: 190 events**.
- **Multiple Cameras Dataset**: Each `chuteXX` folder (`chute01` to `chute24`) represents 1 independent multi-view scenario event. **Total: 24 events**.

> [!IMPORTANT]
> An Event is the **atomic boundary of real-world reality**. Multiple camera views, image frame sequences, depth maps, and sensor readings derived from the same event belong to that single event.

---

## 2. VIDEO

### Definition
A **VIDEO** is defined as a single visual camera stream observation recording an event from a specific spatial camera position and angle.

### Dataset-Specific Video Units
- **URFD**: `cam0` (front/overhead) or `cam1` (side view) stream for an event (100 MP4 video files / 100 RGB image folders).
- **Le2i**: The single `.avi` video file recorded by the fixed camera in a given room (190 AVI video files).
- **Multiple Cameras Dataset**: One of the 8 synchronized camera streams (`cam1.avi` through `cam8.avi`) recording a `chuteXX` scenario (192 AVI video files).

---

## 3. FRAME

### Definition
A **FRAME** is defined as a single discrete temporal sample point captured at timestamp $t_i$.

### Spatial & Modal Attributes per Frame
- **URFD**: 
  - RGB Frame: $640 \times 480$ pixels, 24-bit color (`RGB`, 8 bits/channel).
  - Depth Frame: $640 \times 480$ pixels, 16-bit single-channel raw depth map (`I;16`, values in mm).
- **Le2i**: RGB Frame: $320 \times 240$ pixels (or $320 \times 180$ for `Home_02`), 24-bit color.
- **Multiple Cameras Dataset**: RGB Frame: $720 \times 480$ pixels, 24-bit color (`FMP4` codec).

---

## 4. TEMPORAL WINDOW

### Definition
A **TEMPORAL WINDOW** is defined as a contiguous sub-sequence of $W$ consecutive frames $[f_t, f_{t+1}, \dots, f_{t+W-1}]$ sampled from a video stream with a stride of $S$ frames.

### Prerequisites to Determine Window Length $W$
Before selecting a fixed numerical value for $W$, the following factors must be experimentally evaluated:
1. **Physical Fall Duration**: Typical human falls last between 1.0 and 2.5 seconds (approx. 25 to 75 frames at 30 FPS).
2. **Sampling Rate Variations**: Frame rates differ across datasets (24 FPS in Le2i `Home`, 25 FPS in Le2i `Coffee_room`, ~30 FPS in URFD).
3. **Temporal Stride $S$**: Stride determines window overlap; overlap must be constrained to prevent train-test window leakage.
4. **Real-Time Latency Limit**: In hospital patient monitoring, the window length $W$ directly impacts detection delay ($\text{Delay} \approx \frac{W}{\text{FPS}}$).

---

## 5. LABEL

### Label Definitions

#### A. FALL
A **FALL** event is defined as an involuntary, uncontrolled loss of posture causing a person to come to rest inadvertently on the floor or a lower surface.
- **URFD**: Formally identified by directory prefix `fall-01` through `fall-30` (30 events).
- **Le2i**: Formally identified in `.txt` annotation header where `StartFrame > 0` and `EndFrame > 0` (96 videos).
- **Multiple Cameras Dataset**: Scenarios `chute01` through `chute22` (documented in Auvinet et al., 2010).

#### B. NORMAL / ADL (Activities of Daily Living)
A **NORMAL / ADL** event is defined as a non-fall human activity (e.g., walking, standing, sitting, crouching, bending, or intentionally lying on a sofa/bed).
- **URFD**: Formally identified by directory prefix `adl-01` through `adl-40` (40 events).
- **Le2i**: Formally identified in `.txt` annotation header where `StartFrame == 0` and `EndFrame == 0` (31 videos).
- **Multiple Cameras Dataset**: Scenarios `chute23` and `chute24` (documented in Auvinet et al., 2010).

> [!WARNING]
> **Strict Labeling Rule**: Labels must NEVER be assigned to unannotated videos (e.g., Le2i `Lecture_room` and `Office` videos) or unverified frames without explicit ground-truth documentation.

---

## 6. FALL BOUNDARY

The availability of temporal fall boundaries across local datasets is as follows:

| Temporal Marker | URFD | Le2i | Multiple Cameras Dataset | Evidence / Source |
| :--- | :--- | :--- | :--- | :--- |
| **Fall Start Frame** | **ABSENT** | **PRESENT** | **ABSENT** | **Le2i**: Line 1 in `.txt`. URFD & MultiCamera lack local start annotations. |
| **Fall End Frame** | **ABSENT** | **PRESENT** | **ABSENT** | **Le2i**: Line 2 in `.txt`. URFD & MultiCamera lack local end annotations. |
| **Impact Marker** | **PARTIAL** | **ABSENT** | **ABSENT** | **URFD**: Peak acceleration magnitude in CSV (Column 2) indicates impact moment. |

---

## 7. SPLITTING RULES

To guarantee valid, leak-free evaluation, the following splitting rules are strictly enforced:

### Rule 1: Event-Level Group Splitting (No Event Leakage)
All frames, camera views, and temporal windows derived from the SAME physical event MUST be assigned exclusively to either the Training set, Validation set, or Test set.

### Rule 2: Multi-Camera Group Splitting (No Multi-Camera Leakage)
In URFD (`cam0` and `cam1`) and Multiple Cameras Dataset (`cam1` through `cam8`), all 2 or 8 camera streams belonging to `fall-XX` or `chuteXX` MUST be placed into the SAME split partition.

### Rule 3: Temporal Window Buffer Zone (No Overlapping-Window Leakage)
When sliding windows overlap by $O$ frames, a temporal buffer zone equal to the window length $W$ must be placed between adjacent windows near partition boundaries to prevent temporal feature leakage.

### Rule 4: Leave-One-Location-Out (LOLO) for Le2i
In Le2i, all videos recorded within a single location (`Coffee_room_01`, `Coffee_room_02`, `Home_01`, `Home_02`, `Lecture_room`, `Office`) share a static background and fixed camera position.
- **Requirement**: Evaluation on Le2i MUST use **Leave-One-Location-Out (LOLO) cross-validation** across the 6 location folders to test scene generalization and prevent background memorization.

---

## 8. SUBJECT SPLITTING

Subject-independent (Leave-One-Subject-Out) cross-validation status:

- **URFD**: **NOT CURRENTLY POSSIBLE** (Subject IDs are unmapped in local sequence filenames, though official literature states 5 subjects participated).
- **Le2i**: **NOT CURRENTLY POSSIBLE** (Subject IDs are unmapped in local sequence filenames, though official literature states 9 subjects participated).
- **Multiple Cameras Dataset**: **NOT POSSIBLE** (All 24 scenarios were recorded by 1 single subject actor).

---

## 9. DATASET ROLES

| Dataset | Proposed Role in R&D Pipeline | Primary Technical Objective |
| :--- | :--- | :--- |
| **URFD** | **Multi-Modal Sensor Fusion Benchmark** | Evaluate multi-modal fusion combining RGB vision, 16-bit raw Depth maps, and wearable Accelerometer sensor data. |
| **Le2i** | **Real-World Scene Generalization & Bounding-Box Detection Benchmark** | Evaluate 2D person detection, posture tracking, and cross-location generalization across 6 real-world room environments. |
| **Multiple Cameras Dataset** | **Multi-View Angle Robustness Benchmark** | Evaluate multi-view vision algorithms and view-invariant representation learning across 8 orthogonal camera angles. |

---

## 10. FIRST EXPERIMENT DEFINITION

### Objective
Establish a clean, leak-free **Binary Event-Level Fall Detection Baseline** (Fall vs. ADL classification) using non-overlapping event-level splits.

### Protocol
- Input: RGB visual streams.
- Task: Classify an entire event sequence as `FALL` or `NORMAL / ADL`.
- Splitting: Event-level GroupKFold cross-validation (LOLO for Le2i).

---

## 11. EVALUATION METRICS

### 1. Classification Metrics
- **Precision**: $\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$
- **Recall / Sensitivity**: $\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$
- **F1 Score**: $\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
- **Specificity**: $\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}$
- **Confusion Matrix**: 2x2 matrix $[\text{TN}, \text{FP}; \text{FN}, \text{TP}]$.

### 2. Temporal & System Performance Metrics
- **Time-to-Detection ($\Delta t$)**:
  $$\Delta t = (f_{\text{detected}} - f_{\text{fall\_start}}) \times \frac{1}{\text{FPS}}$$
  *(Supported on Le2i using annotated Fall Start Line 1)*.
- **FPS (Frames Per Second)**: Pipeline throughput speed in frames per second.
- **Inference Latency**: Time required in milliseconds to process one temporal window.

### 3. Metric Limitation: False Alarms Per Camera-Hour
- **Explanation**: False alarms per camera-hour **cannot be properly measured** on these datasets because they consist of short, pre-trimmed 5-to-30 second event clips rather than continuous, unsegmented multi-hour surveillance footage.

---

## 12. KNOWN DATASET LIMITATIONS

1. **Ground-Truth Label Gaps**: Local URFD and MultiCamera lack temporal fall start/end annotations.
2. **Unannotated Portion**: 31.6% of Le2i videos (`Lecture_room` and `Office`) are unannotated locally.
3. **Single-Subject Bias**: MultiCamera contains only 1 participant.
4. **Unmapped Subject Metadata**: URFD and Le2i do not map subject IDs to sequence filenames.
5. **Fixed Background Leakage Risk**: Static camera positions per room necessitate strict group splitting.

---

## 13. OPEN QUESTIONS BEFORE PREPROCESSING

1. Should external ground-truth annotation tables for MultiCamera (DIRO Report 1350) and URFD be manually transcribed/linked?
2. What unified spatial resolution (e.g., $320 \times 240$ vs. $640 \times 480$) should be adopted for cross-dataset evaluation?
3. What target temporal frame rate (e.g., 25 FPS or 30 FPS) should be standardized via temporal resampling?
4. How should the 60 unannotated Le2i videos (`Lecture_room` and `Office`) be utilized (e.g., unsupervised domain adaptation vs. excluded from supervised evaluation)?
