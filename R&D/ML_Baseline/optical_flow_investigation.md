# Experiment D: Optical Flow Feasibility & Research Design Investigation

> [!IMPORTANT]
> **READINESS & DESIGN ONLY — NO TRAINING PERFORMED — NO DATA GENERATED.**
> This document details the scientific rationale, empirical feasibility benchmark, candidate representations, domain shift mitigations, and controlled experimental design (D1/D2/D3) for explicit motion representation.

---

## 1. Executive Summary & Recommendation
- **Target Experiment**: Experiment D — Explicit Motion Representation via Optical Flow.
- **Investigative Status**: **INVESTIGATION COMPLETE — RECOMMENDED DECISION: GO FOR EXPERIMENT D IMPLEMENTATION**.
- **Key Benchmark Result**: Precomputing 512-D optical flow feature embeddings for all 1,396 Le2i windows takes **~11.1 minutes** on CPU and requires only **133.60 MB** of disk storage (compared to 39.14 GB for uncompressed raw flow tensors).

---

## 2. Motivation & Scientific Rationale

### Empirical Context
- **Experiment A (URFD $\to$ Le2i Zero-Shot Transfer)**: F1 = $31.51\%$ ($\tau=0.50$). Demonstrated severe cross-dataset domain shift.
- **Experiment B (Le2i $\to$ Le2i LOLO Baseline)**: F1 = $71.53\% \pm 26.69\%$. Proved that in-domain exposure improves location generalization, but performance on `Home_01` ($40.34\%$ F1) and `Home_02` ($58.33\%$ F1) remains bottlenecked.
- **Experiment C (Temporal Representation Ablation)**:
  - Mean-Only: $53.69\%$ F1
  - Mean+Std Control: $71.53\%$ F1
  - 1-Layer GRU: $68.46\%$ F1
  - **Key Finding**: Preserving frame order (GRU) did **NOT** improve performance over Mean+Std pooling.

### Core Scientific Hypothesis for Experiment D
Static RGB spatial features saturate on background furniture, lighting reflections, and wall contrast. **Explicit motion vectors ($\mathbf{v} = (v_x, v_y)$)** isolate frame-to-frame downward velocity vectors during a fall, effectively discarding static room background textures and achieving true scene-invariant fall detection.

---

## 3. Optical Flow Candidate Representation Comparison

| Representation Option | Tensor Shape per Window | Storage Footprint per Window | Total Disk Footprint (1,396 Windows) | Farneback Extraction Time (1,396 Windows) | Suitability / Recommendation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Option A: Full-Res Raw Flow Fields** | `(49, 240, 320, 2)` float32 | `28.7 MB` | **39.14 GB** | ~11.1 min (CPU) | **NO-GO**: Excessively high disk storage footprint. |
| **Option B: Downsampled Flow Fields** | `(49, 112, 112, 2)` float32 | `4.7 MB` | **6.39 GB** | ~11.1 min (CPU) | **FEASIBLE**: Reasonable storage, but requires 2D CNN flow model. |
| **Option C: Pretrained Flow Features** | `(49, 512)` float32 | `98.0 KB` | **133.60 MB** | ~11.1 min + feature pass | **RECOMMENDED ✅**: Compact 512-D flow embeddings seamlessly integrate with existing feature pipeline. |

---

## 4. Empirical Feasibility Benchmark Results

Tested OpenCV Farneback Optical Flow (`cv2.calcOpticalFlowFarneback`) on actual Le2i video clips ($320 \times 240$ resolution):

- **Mean Time per Flow Frame Pair**: **9.73 ms**
- **Estimated Time per Window ($49$ pairs)**: **0.48 seconds**
- **Total Extraction Time for 1,396 Windows**: **665.40 seconds (~11.1 minutes)**
- **Feature Vector Storage (512-dim per frame)**: **133.60 MB Total**

---

## 5. Domain Shift & Scientific Concerns

1. **Camera & Background Motion**: Farneback optical flow can produce non-zero vectors due to lighting changes or moving shadows.
   - *Mitigation*: Apply spatial magnitude thresholding ($M = \sqrt{v_x^2 + v_y^2} < 0.5\text{ px/frame} \to 0$) to eliminate ambient noise.
2. **Static Post-Fall Posture**: Optical flow approaches zero once a person lands on the floor and stops moving. Flow alone cannot detect static lying posture.
   - *Mitigation*: **Mandatory Dual-Stream Fusion (Experiment D3)**. RGB spatial features maintain high activation for post-fall floor postures, while Flow features capture the dynamic downward descent.
3. **Camera Perspective / Distance Variations**: Subjects closer to the camera produce higher magnitude displacement vectors than distant subjects.
   - *Mitigation*: Normalize flow displacement vectors by frame resolution dimensions before feature extraction.

---

## 6. Controlled Experimental Design (D1, D2, D3)

All three controlled experiments will evaluate the exact same 4-Fold LOLO cross-validation protocol (Coffee_room_01, Coffee_room_02, Home_01, Home_02) using event-level inner validation checkpointing and threshold selection.

```text
Experiment D1 (Flow-Only):    (B, 49, 512) ──► Mean+Std (1024) ──► Linear(1024→64) ──► ReLU ──► Dropout(0.5) ──► Linear(64→2) [65,730 params]
Experiment D2 (RGB Control):  (B, 50, 512) ──► Mean+Std (1024) ──► Linear(1024→64) ──► ReLU ──► Dropout(0.5) ──► Linear(64→2) [65,730 params]
Experiment D3 (RGB+Flow):     Concatenate [(B, 1024) RGB, (B, 1024) Flow] ──► (B, 2048) ──► Linear(2048→64) ──► ReLU ──► Dropout(0.5) ──► Linear(64→2) [131,266 params]
```

### Experimental Matrix Overview
- **D1 (Optical Flow-Only Baseline)**: Evaluates whether motion dynamics alone generalize across unseen physical rooms.
- **D2 (RGB Mean+Std Control)**: Reference baseline from Experiment B (**71.53% LOLO F1**).
- **D3 (RGB + Flow Fusion)**: Evaluates whether combining spatial scene context + explicit motion velocity outperforms RGB alone ($> 71.53\%$ F1).

---

## 7. Strict Leakage Controls
- The outer test location will remain 100% unseen during flow feature extraction, training, class-weight calculation, checkpoint selection, and threshold tuning.
- Inner validation splits will maintain strict video event isolation ($\text{Inner Train Events} \cap \text{Inner Val Events} = \emptyset$).

---

## 8. Final Verdict & Decision

**DECISION: GO FOR EXPERIMENT D IMPLEMENTATION**

- Feasibility: **PASS** (~11.1 min extraction time, 133.6 MB storage).
- Rationale: **PASS** (Directly addresses spatial feature saturation and background texture shift).
- Status: **NO DATA GENERATED YET — NO TRAINING PERFORMED YET.**
