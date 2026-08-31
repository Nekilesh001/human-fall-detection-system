# Phase F5 Application State Machine Refinement Report

> [!IMPORTANT]
> **MODEL K1 FREEZE VERIFICATION**  
> Checkpoint Path: `checkpoints/final_k1/final_production.pth`  
> SHA256 Checksum (Before & After): `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d` (**100% UNTOUCHED**)  
> Official Operating Threshold: $\tau = 0.3650$ (**UNCHANGED**)  
> Model Architecture: 1D Residual Temporal Convolutional Network (86,434 core parameters, **UNCHANGED**)  
> Training Executed: **0 Epochs (No retraining performed)**

---

## 1. Discovered Current Behavior Prior to Modification

Prior to refinement, `ApplicationStateMachine` required 10 frames ($\approx 0.4\text{s}$) in `GETTING UP / RECOVERY` plus 15 frames ($\approx 0.6\text{s}$) in `RECOVERED — STANDING` (total 25 frames $\approx 1.0\text{s}$ @ 25 FPS) to reset a confirmed fall latch. While safe against noisy single pose frames, this produced a noticeable hysteresis delay during rapid stand-ups post-fall.

---

## 2. Exact Files Modified

1. [`src/final_k1_realtime_inference.py`](file:///d:/ONE_DATA/Fall%20detection/src/final_k1_realtime_inference.py): Refined `getting_up_counter` threshold from 10 to 5 frames and `recovered_counter` threshold from 15 to 5 frames in `ApplicationStateMachine`.
2. [`src/validate_f5_streamlit.py`](file:///d:/ONE_DATA/Fall%20detection/src/validate_f5_streamlit.py): Expanded static validation audit script to include 8 programmatic state machine regression tests.

---

## 3. Application-Level Changes

- Reduced postural recovery counter thresholds to 5 frames each ($\approx 0.4\text{s}$ total recovery hysteresis at 25 FPS).
- Preserved exact 2-stage recovery sequence (`GETTING UP / RECOVERY` $\to$ `RECOVERED — STANDING` $\to$ `NORMAL — STANDING`).
- Kept model inference binary ($P(\text{FALL})$ and raw `NORMAL`/`FALL` decision).

---

## 4. State Transition Diagram

```
[WARMING UP] (N < 50)
     │
     ├── (N >= 50, person_detected=False / conf_sum < 0.5) ──► [NO PERSON DETECTED]
     │
     └── (N >= 50, person_detected=True) ──► [NORMAL / STANDING / WALKING / SITTING]
                                                   │
                                                   │ P(FALL) >= 0.3650 (Window 1 or 2)
                                                   ▼
                                           [FALL SUSPECTED]
                                                   │
                                                   │ P(FALL) >= 0.3650 (Window 3+)
                                                   ▼
                                           [FALL DETECTED] (🚨 Alert Active, has_confirmed_fall = True)
                                                   │
                                                   │ P(FALL) < 0.3650 & is_upright == False
                                                   ▼
                                           [FALLEN — ON FLOOR] (Latched!)
                                                   │
                                                   │ is_upright == True (Upright posture detected)
                                                   ▼
                                           [GETTING UP / RECOVERY] (5 frames)
                                                   │
                                                   │ getting_up_counter >= 5 frames
                                                   ▼
                                           [RECOVERED — STANDING] (5 frames)
                                                   │
                                                   │ recovered_counter >= 5 frames
                                                   ▼
                                   Resets Latch ──► [NORMAL — STANDING]
```

---

## 5. Scenario Explanations & Behavior

- **Sudden Fall Handling**: Requires 3 consecutive windows with $P(\text{FALL}) \ge 0.3650$ ($\approx 0.12\text{s}$ response time). Fires `FALL DETECTED` with active alert and perimeter red flash.
- **Fall $\to$ Floor $\to$ Recovery $\to$ Standing**: Latches post-fall low posture (`FALLEN — ON FLOOR`). Upright posture detection triggers `GETTING UP / RECOVERY` (5 frames) $\to$ `RECOVERED — STANDING` (5 frames) $\to$ `NORMAL — STANDING`.
- **Rapid Stand-Up Behavior**: Smoothly transitions through recovery states within $\approx 0.4\text{s}$ of sustained upright posture, preventing excessive delay while protecting against single noisy frames.
- **Normal Standing / Walking / Sitting**: Posture geometry filters ensure normal ADLs stay classified as `NORMAL — STANDING`, `NORMAL — WALKING`, or `NORMAL — SITTING` without false fall activations.
- **No-Person Handling**: `conf_sum < 0.5` sets `person_detected = False`, bypasses Model K1 ($P=0.0$), and displays `NO PERSON DETECTED`.
- **Partial-Frame Handling**: Boundary proximity evaluation sets `is_partial_person = True`, suppressing new false fall alerts on boundary-clipped persons. **NO user-facing "PARTIAL FALL" state exists.**
- **Bounding-Box Visualization**: Real YOLO person box `[x1, y1, x2, y2]` drawn tightly around person coordinates with attached `PERSON | Pose Detected` tag and smooth 1px cyan skeleton.

---

## 6. Model Safety & Checkpoint Verification

| Safety Audit Criterion | Value / Status |
| :--- | :--- |
| **Model Checkpoint SHA256** | `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d` (**IDENTICAL**) |
| **Official Operating Threshold ($\tau$)** | $0.3650$ (**UNCHANGED**) |
| **Model Architecture** | `ModelK1_SpatialTCN` (**UNCHANGED**) |
| **Input Feature Dimension** | 187-D spatial body geometry (**UNCHANGED**) |
| **Temporal Window** | 50 frames (2.0s @ 25 FPS, **UNCHANGED**) |
| **Alert Stabilization** | 3 consecutive FALL windows (**UNCHANGED**) |
| **Training Executed** | **0 Epochs (No retraining performed)** |
| **Git Write Commands** | **0 Commands Executed** |

---

## 7. Limitations & Scientific Honesty

Model K1 is a binary fall classifier outputting fall probability $P(\text{FALL})$. Standing, walking, sitting, fallen, recovery, and recovered states are application-level state-machine interpretations based on YOLO Pose keypoints, model probability, and temporal tracking. They are **NOT** independent classes learned by the K1 neural network.
