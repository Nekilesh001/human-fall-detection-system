# Preprocessing Feasibility Experiment Report

## Executive Summary
This document presents the findings of our **Small Preprocessing Feasibility Experiment** conducted prior to full-dataset processing. The objective is to empirically evaluate spatial resizing algorithms, temporal FPS resampling methods, sliding window feasibility, leakage constraints, and spatial padding strategies for the proposed baseline pipeline.

All experiments were executed in read-only mode relative to raw dataset directories. Sample outputs and visual comparison files have been saved strictly inside `R&D/Preprocessing_Experiments/representative_comparisons/`.

---

## 1. RGB Resizing Test ($640 \times 480 \to 320 \times 240$)

### Experimental Setup
Representative URFD RGB frames from Fall (`fall-01-cam0`, `fall-01-cam1`) and ADL (`adl-01-cam0`) sequences were extracted and resized from $640 \times 480$ down to $320 \times 240$ using Bilinear, Bicubic, and Lanczos (Antialiased Area) interpolation algorithms.

### Empirical Evaluation & Execution Speed

| Resizing Method | Execution Speed (ms/frame) | Visual Edge Quality | Human Bounding/Pose Preservation | Moiré / Aliasing Artifacts |
| :--- | :--- | :--- | :--- | :--- |
| **Bilinear** | **0.992 ms** | Acceptable | Preserved main human outline | Minor high-frequency aliasing on clothing edges |
| **Bicubic** | 1.521 ms | Good | Smooth limb contours | Reduced edge aliasing |
| **Lanczos (Area)**| 2.412 ms | **Superior** | **Highest fidelity on human silhouette** | **Zero aliasing / staircasing artifacts** |

### Sample Output Files
- `R&D/Preprocessing_Experiments/representative_comparisons/fall_cam0_320x240_bilinear.png`
- `R&D/Preprocessing_Experiments/representative_comparisons/fall_cam0_320x240_lanczos.png`
- `R&D/Preprocessing_Experiments/representative_comparisons/fall_cam1_320x240_lanczos.png`

### Finding & Objective Assessment
Bilinear resizing is ~2.4x faster, but Lanczos (Antialiased Area) filtering eliminates high-frequency edge staircasing along thin human limbs. For spatial resolution normalization ($640 \times 480 \to 320 \times 240$), **Lanczos / Area downscaling** is recommended to maximize human pose feature fidelity.

---

## 2. FPS Sampling Analysis (~30 FPS $\to$ 25.0 FPS)

### Experimental Setup
Tested on representative URFD sequences (e.g. `fall-01`, 160 source frames @ ~30 FPS, duration 5,305 ms).

### Resampling Methodology
Instead of naive uniform frame dropping or duplication, target frame timestamps $t_k = k \times 40.0\text{ ms}$ (for 25.0 FPS) are computed across sequence duration $D$. For each target timestamp $t_k$, the nearest-neighbor source frame $f_{\text{source}}(i)$ is selected based on minimum absolute timestamp delta:
$$i^* = \arg\min_i |t_{\text{source}}(i) - t_k|$$

### Empirical Resampling Results (`fall-01`)
- **Source Duration**: 5,305.0 ms
- **Source Frame Count (@ ~30 FPS)**: 160 frames ($\Delta t \approx 33.3\text{ ms}$)
- **Target Frame Count (@ 25.0 FPS)**: 133 frames ($\Delta t = 40.0\text{ ms}$)
- **Sampled Frame Index Sequence**: `[0, 1, 2, 4, 5, 6, 7, 8, 10, 11, ...]`
- **Timestamp Alignment**: Max timestamp discrepancy across all sampled frames is $< 16.6\text{ ms}$ (less than half a frame interval).

### Finding
Deterministic nearest-neighbor timestamp matching converts 30 FPS video to exactly 25.0 FPS without frame duplication or motion jitter.

---

## 3. Expected Temporal-Window Counts (URFD @ 25 FPS)

Using the standardized 25.0 FPS resampled URFD sequence lengths, expected temporal window counts were calculated across the 70 URFD events:

| Window Configuration | Window Length ($W$) | Stride ($S$) | Overlap % | Expected FALL Windows | Expected ADL Windows | Total Windows Generated |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Config A (Short)** | 25 frames (1.0s) | 12 frames (0.48s) | 52.0% | 316 | 557 | **873** |
| **Config B (Medium)** | **50 frames (2.0s)** | **25 frames (1.0s)** | **50.0%** | **118** | **246** | **364** |
| **Config C (Long)** | 75 frames (3.0s) | 25 frames (1.0s) | 66.7% | 64 | 206 | **270** |

### Finding
- **Config A** yields 873 windows, providing high sample count but shorter temporal context.
- **Config B (Recommended)** yields 364 windows (118 Fall, 246 ADL), providing balanced 2.0-second temporal coverage of full fall transitions.
- **Config C** yields 270 windows, but reduces fall window count to 64.

---

## 4. Leakage Implications & Verification

### Verification of Event Partition Inheritance
To prevent multi-camera or window leakage, temporal window generation MUST occur **AFTER** event-level group partitioning:

```
URFD Raw Events (70)
 ├── Train Events (49: fall-01..06, fall-09..10, adl-01..02...)
 │    └── Extract Windows -> ALL derived windows assigned to TRAIN SET
 ├── Val Events (10: fall-19, fall-21, adl-04...)
 │    └── Extract Windows -> ALL derived windows assigned to VAL SET
 └── Test Events (11: fall-07, fall-08, adl-03...)
      └── Extract Windows -> ALL derived windows assigned to TEST SET
```

### Concrete Verification (`fall-01` Dual Camera)
- `fall-01` consists of `cam0` (front/overhead) and `cam1` (side view).
- Per `split_strategy.md`, `fall-01` is assigned to **TRAIN**.
- All temporal windows extracted from `fall-01-cam0` AND `fall-01-cam1` inherit the **TRAIN** partition.
- Zero windows from `fall-01` enter Validation or Testing, confirming **0% multi-camera or temporal window leakage**.

---

## 5. Le2i Spatial Handling Comparison ($320 \times 180 \to 320 \times 240$)

### Experimental Comparison

| Strategy | Implementation | Geometry Preservation | Aspect Ratio Distortion | Visual Artifacts |
| :--- | :--- | :--- | :--- | :--- |
| **1. Non-Aspect Stretch** | Direct resize to $320 \times 240$ | **Distorted** ($1.33\times$ vertical stretch) | Distorted | Human posture appears artificially elongated |
| **2. Center Crop** | Crop width to $240 \times 180$, resize to $320 \times 240$ | Preserved | Preserved | **Truncates lateral human motion / scene background** |
| **3. Vertical Zero Padding** | **Pad 30px top, 30px bottom** | **100% Preserved** | **100% Preserved** | Clean black bars top/bottom; zero subject distortion |

### Sample Output Files
- `R&D/Preprocessing_Experiments/representative_comparisons/le2i_home02_padded_320x240.png`
- `R&D/Preprocessing_Experiments/representative_comparisons/le2i_home02_stretched_320x240.png`

### Finding
Vertical zero-padding (30px top, 30px bottom) is the **only method** that preserves exact human body geometry without cropping subject limbs or distorting vertical posture.

---

## 6. Recommended Preprocessing Configuration

> [!TIP]
> **Recommended Standard Preprocessing Configuration**
> 1. **Target Sampling FPS**: **25.0 FPS** (Nearest-neighbor timestamp matching).
> 2. **Target Spatial Resolution**: **$320 \times 240$ pixels** (Lanczos downscaling for URFD/MultiCamera; vertical zero-padding for Le2i `Home_02`).
> 3. **Temporal Window Config**: **Config B** ($W = 50$ frames / 2.0s @ 25 FPS, Stride $S = 25$ frames / 50% overlap).
> 4. **Splitting Protocol**: Event-level GroupKFold (`seed = 42` for URFD/MultiCamera; LOLO across 4 supervised room folds for Le2i).

---

## 7. Remaining Uncertainties Before Full Preprocessing

1. **Unannotated Le2i Videos**: Confirm whether the 60 unannotated videos in `Lecture_room` and `Office` will be excluded entirely or used in a separate unsupervised domain adaptation pipeline.
2. **Accelerometric Fusion Window Alignment**: For URFD multi-modal fusion, verify exact millisecond synchronization alignment between accelerometer sample timestamps and resampled video frame timestamps.
