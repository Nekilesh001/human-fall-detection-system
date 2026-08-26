# Data Splitting & Temporal Sampling Strategy Design

## Executive Summary
This document formalizes the **leakage-safe data splitting rules, deterministic partition assignments, temporal window sampling designs, and evaluation protocols** for our Human Fall Detection System prior to data preprocessing or model training.

All designs adhere to the strict rule that **EVENT is the atomic split unit**, ensuring zero temporal, spatial, or multi-camera data leakage.

---

## PART 1 — SPLITTING STRATEGY

### 1. The Fundamental Rule of Splitting
> [!IMPORTANT]
> **ATOMIC SPLIT UNIT = EVENT (`event_id`)**
> All video recordings, camera views, raw image frames, depth maps, sensor CSVs, and temporal sliding windows derived from a single physical event MUST remain in the **SAME** split partition (Training, Validation, or Testing).

### Why Frame-Level / Window-Level Random Splitting is Invalid
Randomly assigning individual frames or temporal windows to train and test sets leads to catastrophic **data leakage** and invalid research results due to:
1. **Temporal Autocorrelation Leakage**: Frame $t$ and frame $t+1$ in a video sequence are nearly identical. Placing frame $t$ in train and frame $t+1$ in test causes the model to memorize specific frame colors rather than learning human action dynamics.
2. **Multi-Camera Spatial Leakage**: In multi-camera datasets (URFD with 2 cams, MultiCamera with 8 cams), `cam1` and `cam2` record the exact same human pose at the exact same millisecond. Splitting cameras across partitions leaks identical pose representations into the test set.
3. **Scene & Background Memorization**: Fixed cameras capture static room backgrounds. Splitting videos from the same room across train and test allows models to achieve artificially high accuracy by memorizing room geometry rather than detecting falls.

---

### 2. URFD Event-Level Split Design
- **Total Events**: 70 events (30 Fall events: `fall-01`..`fall-30`, 40 Normal ADL events: `adl-01`..`adl-40`).
- **Camera Handling**: Both `cam0` and `cam1` of fall events are bound strictly to the same event ID.
- **Deterministic Seed**: `seed = 42`
- **Target Ratio**: ~70% Train, ~15% Validation, ~15% Test (Stratified by Fall/Normal class).

#### Deterministic Event Partition Assignments (URFD)

```
TRAIN PARTITION (49 Events: 21 Fall, 28 ADL)
├── Fall Events (21): fall-01, fall-02, fall-03, fall-04, fall-05, fall-06, fall-09, fall-10,
│                     fall-12, fall-13, fall-14, fall-16, fall-17, fall-18, fall-22, fall-23,
│                     fall-24, fall-25, fall-26, fall-28, fall-29
└── ADL Events (28):  adl-01, adl-02, adl-05, adl-06, adl-08, adl-10, adl-11, adl-12, adl-13,
                      adl-17, adl-19, adl-20, adl-22, adl-23, adl-24, adl-26, adl-27, adl-29,
                      adl-30, adl-31, adl-32, adl-33, adl-34, adl-35, adl-36, adl-37, adl-38, adl-39

VALIDATION PARTITION (10 Events: 4 Fall, 6 ADL)
├── Fall Events (4):  fall-19, fall-21, fall-27, fall-30
└── ADL Events (6):   adl-04, adl-09, adl-14, adl-18, adl-25, adl-40

TEST PARTITION (11 Events: 5 Fall, 6 ADL)
├── Fall Events (5):  fall-07, fall-08, fall-11, fall-15, fall-20
└── ADL Events (6):   adl-03, adl-07, adl-15, adl-16, adl-21, adl-28
```

#### Class Balance Summary (URFD)
- **Train**: 21/49 Fall (42.9%), 28/49 Normal (57.1%)
- **Validation**: 4/10 Fall (40.0%), 6/10 Normal (60.0%)
- **Test**: 5/11 Fall (45.5%), 6/11 Normal (54.5%)

---

### 3. Le2i Leave-One-Location-Out (LOLO) Split Design

#### Supervised Dataset Scope
- **Usable Labeled Videos**: **127 videos** (96 FALL, 31 NORMAL) across 4 annotated locations.
- **Excluded Records**: **63 UNKNOWN videos** (60 in `Lecture_room` and `Office`, 3 malformed headers) are strictly EXCLUDED from supervised training and validation.

#### LOLO Fold Composition

| LOLO Fold ID | Held-Out Test Location | Test Set Labeled Composition | Training Set Locations | Train Set Labeled Composition |
| :--- | :--- | :--- | :--- | :--- |
| **Fold 1** | `Coffee_room_01` | 47 Fall, 0 Normal (47 videos) | `Coffee_room_02`, `Home_01`, `Home_02` | 49 Fall, 31 Normal (80 videos) |
| **Fold 2** | `Coffee_room_02` | 12 Fall, 8 Normal (20 videos) | `Coffee_room_01`, `Home_01`, `Home_02` | 84 Fall, 23 Normal (107 videos) |
| **Fold 3** | `Home_01` | 30 Fall, 0 Normal (30 videos) | `Coffee_room_01`, `Coffee_room_02`, `Home_02` | 66 Fall, 31 Normal (97 videos) |
| **Fold 4** | `Home_02` | 7 Fall, 23 Normal (30 videos) | `Coffee_room_01`, `Coffee_room_02`, `Home_01` | 89 Fall, 8 Normal (97 videos) |

#### Handling Single-Class Held-Out Locations
- **Observation**: `Coffee_room_01` and `Home_01` contain ONLY fall recordings (0 normal ADL controls).
- **Protocol**: When evaluating Fold 1 or Fold 3 individually, metric calculation must report **Sensitivity / Recall** on the held-out fold, while overall benchmark performance must be computed by aggregating predictions across all 4 LOLO test folds (yielding a combined test pool of 96 Falls and 31 Normals).

---

### 4. Multiple Cameras Dataset Event-Level Split Design
- **Total Events**: 24 `chuteXX` events (`chute01`..`chute24`), each containing 8 synchronized camera streams.
- **Camera Binding**: All 8 camera files of a `chute` folder are strictly bound to the same partition.
- **Deterministic Seed**: `seed = 42`
- **Target Ratio**: ~70% Train (17 chutes), ~12.5% Val (3 chutes), ~16.7% Test (4 chutes).

#### Deterministic Event Partition Assignments (MultiCamera)

```
TRAIN PARTITION (17 Chutes: 16 Fall, 1 Normal)
├── Fall Chutes (16): chute01, chute03, chute05, chute06, chute07, chute08, chute09, chute11,
│                     chute13, chute15, chute17, chute18, chute19, chute20, chute21, chute22
└── Normal Chute (1): chute23

VALIDATION PARTITION (3 Chutes: 3 Fall)
└── Fall Chutes (3):  chute04, chute14, chute16

TEST PARTITION (4 Chutes: 3 Fall, 1 Normal)
├── Fall Chutes (3):  chute02, chute10, chute12
└── Normal Chute (1): chute24
```

---

## PART 2 — TEMPORAL SAMPLING DESIGN

### 1. Sampling Frame Rate Standardization
- **Target Standard FPS**: **25.0 FPS** (Resampling period $\Delta t = 40.0\text{ ms}$).
- **Resampling Rules**:
  - URFD (~30 FPS): Downsampled temporally by factor of $1.2\times$ to 25 FPS.
  - Le2i (`Coffee_room`, `Lecture_room`, `Office` @ 25 FPS): Preserved at 1:1 rate.
  - Le2i (`Home_01`, `Home_02` @ 24 FPS): Nearest-neighbor interpolation to 25 FPS.
  - MultiCamera (25 FPS playback rate): Preserved at 1:1 rate.

### 2. Spatial Resolution Standardization
- **Target Resolution**: **$320 \times 240$ pixels**.
- **Rescaling Rules**:
  - URFD ($640 \times 480$): Downscaled bilinear to $320 \times 240$.
  - Le2i ($320 \times 240$): Maintained at $320 \times 240$.
  - Le2i `Home_02` ($320 \times 180$): Zero-padded vertically (30px top, 30px bottom) to achieve $320 \times 240$ without aspect ratio distortion.
  - MultiCamera ($720 \times 480$): Rescaled bilinear to $320 \times 240$.

---

### 3. Proposed Temporal Window Configurations & Trade-Offs

| Parameter | Configuration A (Short Window) | Configuration B (Medium - Recommended) | Configuration C (Long Window) |
| :--- | :--- | :--- | :--- |
| **Window Length ($W$)** | **25 frames (1.0s)** | **50 frames (2.0s)** | **75 frames (3.0s)** |
| **Frame Stride ($S$)** | **12 frames (0.48s)** | **25 frames (1.0s)** | **25 frames (1.0s)** |
| **Window Overlap** | 52.0% | **50.0%** | 66.7% |
| **Primary Strength** | Minimal detection delay ($\Delta t \approx 1.0\text{s}$) | **Captures full dynamic fall transition ($1-2\text{s}$)** | Includes pre-fall, impact, and post-fall lying |
| **Primary Limitation** | May miss post-fall lying state | **Balanced temporal context** | Higher inference latency ($\Delta t \approx 3.0\text{s}$) |

---

### 4. Window Labeling & Fall Coverage Criteria

#### A. FALL Window Criteria
A temporal window $[f_t, \dots, f_{t+W-1}]$ is labeled `FALL` if and only if **at least 50% of its frames** fall within the annotated fall interval $[f_{\text{start}}, f_{\text{end}}]$:
$$\frac{\text{Count}(f \in [f_{\text{start}}, f_{\text{end}}])}{W} \ge 0.50 \implies \text{Label} = \text{FALL}$$

#### B. NORMAL Window Criteria
A temporal window is labeled `NORMAL` if **100% of its frames** occur before $f_{\text{start}}$ or belong to a normal ADL event sequence:
$$\text{Count}(f \in \text{Normal Frames}) = W \implies \text{Label} = \text{NORMAL}$$

#### C. Transition Window Handling (Ambiguity Exclusion)
Windows spanning $f_{\text{start}}$ or $f_{\text{end}}$ with fall frame coverage between **1% and 49%** are classified as **Transition Windows** and are **EXCLUDED from supervised training** to eliminate label noise.

---

## PART 3 — LE2I SPECIAL TEMPORAL GROUND TRUTH

Because Le2i provides explicit $f_{\text{start}}$ and $f_{\text{end}}$ frame indices in `.txt` annotations, temporal windows can be partitioned into 3 distinct phase classes:

```
Video Timeline:
[1] -------------- [f_start] ================== [f_end] ------------------ [N_total]
  Pre-Fall Phase            Active Fall Phase            Post-Fall / Lying Phase
 (NORMAL Posture)          (DYNAMIC TRANSITION)         (FALLEN / INACTIVE)
```

1. **Pre-Fall Normal Windows**: Frames $[1, f_{\text{start}} - 1]$. Represents normal standing/walking activities prior to loss of balance. Label = `NORMAL`.
2. **Active Fall Windows**: Frames $[f_{\text{start}}, f_{\text{end}}]$. Represents the dynamic loss of posture and impact. Label = `FALL`.
3. **Post-Fall Windows**: Frames $[f_{\text{end}} + 1, N_{\text{total}}]$. Represents the person lying motionless on the floor. Label = `FALLEN / ON-FLOOR`.

---

## PART 4 — FINAL RECOMMENDATION

### A. First Dataset for Baseline
**URFD (RGB Stream)**: Recommended as the first dataset for initial baseline development due to its clean, multi-modal structure, 100% path validity, and balanced class distribution (30 Fall / 40 ADL).

### B. First Split Strategy
**URFD Event-Level Group Split (Seed 42)**: Use the deterministic 49 Train / 10 Val / 11 Test event partition defined in Part 1.

### C. First Temporal Sampling Configuration
**Configuration B (Medium Window)**:
- Window length $W = 50$ frames (2.0s @ 25 FPS)
- Frame stride $S = 25$ frames (1.0s, 50% overlap)

### D. What Should Remain Unchanged
- Raw dataset files on disk (`URFD/`, `Le2i/`, `dataset/`).
- Master manifest `dataset_manifest.csv`.
- Validation report `manifest_validation_report.md`.

### E. What Still Needs an Experiment Before Finalizing
- Empirical evaluation of spatial zero-padding vs. center-cropping for Le2i `Home_02` ($320 \times 180$).
- Empirical evaluation of bilinear downscaling vs. antialiased area downscaling for $640 \times 480 \to 320 \times 240$.
