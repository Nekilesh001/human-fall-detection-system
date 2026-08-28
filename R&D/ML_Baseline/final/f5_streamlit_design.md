# Phase F5 Production Streamlit Application Architecture & Design Guide

> [!IMPORTANT]
> **EXPLICIT ARCHITECTURAL SEPARATION OF MODEL OUTPUT VS. APPLICATION STATE MACHINE**  
> Model K1 is a **supervised binary fall detector** predicting $P(\text{FALL}) \in [0, 1]$ and raw decision (`NORMAL` vs `FALL` @ $\tau = 0.3650$).  
> Postures (`NORMAL — STANDING`, `NORMAL — WALKING`, `NORMAL — SITTING`), recovery states (`GETTING UP / RECOVERY`, `RECOVERED — STANDING`), and floor states (`FALLEN — ON FLOOR`) are **application-level derived interpretations** combining YOLO Pose keypoints, Model K1 $P(\text{FALL})$, and state-machine transition logic. They are **NOT** independent classes learned by the K1 neural network.

---

## 1. System Pipeline & Architectural Isolation

```
Raw Video Stream (.mp4 / .avi / .mov)
    │
    ├── 1. YOLO11-Pose Extractor
    │      └── Detects person, bounding box [x1,y1,x2,y2], 17 COCO keypoints (33 landmarks), confidence
    │
    ├── 2. Edge-of-Frame Proximity Evaluator (`is_partial_person` & `edge_reason`)
    │      └── Evaluates boundary proximity (margin < 2%) + keypoint completeness (ankles/head/count)
    │
    ├── 3. 187-D Spatial Feature Derivation
    │      └── 165-D Base (Coords + Velocities) + 22-D Body Geometry (Angles, Heights, Torso Tilt)
    │
    ├── 4. 50-Frame Rolling Temporal Buffer (2.0s Context @ 25 FPS)
    │      └── Warmup Guard (Frames 1–49 = WARMING UP)
    │
    ├── 5. Person Presence Validation Gate (`person_detected` Gate)
    │      └── If person absent / conf < 0.5: Output NO_PERSON_DETECTED, P(FALL)=0.0, freeze alert counters
    │
    ├── 6. FROZEN Model K1 1D Residual TCN (`checkpoints/final_k1/final_production.pth`)
    │      └── Outputs ML Model Prediction: P(FALL) ∈ [0, 1] & Raw Decision (NORMAL/FALL @ τ = 0.3650)
    │
    ├── 7. 3-Consecutive Window Alert Stabilizer
    │      └── 1–2 FALL windows = FALL SUSPECTED | 3+ FALL windows = ALERT ACTIVE
    │
    ├── 8. Application State Machine with Edge Protection (`ApplicationStateMachine`)
    │      └── Blocks NEW fall activations when `is_partial_person == True` while preserving latched falls
    │
    └── 9. Production Streamlit UI (`app.py`) with Non-Obstructive Compact HUD & CSV Logger
```

---

## 2. Problem Diagnoses & Resolutions

### Problem 1: Edge-of-Frame / Partial Person False Alert
- **Root Cause**: When a person moves near/outside camera boundaries (left, right, top, bottom), keypoints become partially truncated (e.g. feet or ankles clipped at bottom border). This distorts 187-D spatial body geometry (joint height ratios, torso-to-leg proportions), causing Model K1 to output $P(\text{FALL}) \approx 0.47\text{--}0.57$.
- **Edge Proximity Geometric Rule**:
  - `margin_x = x1 / w_img` (or `(w_img - x2) / w_img`)
  - `margin_y = y1 / h_img` (or `(h_img - y2) / h_img`)
  - If boundary proximity is within 2% AND critical keypoints are missing (e.g. ankles missing when touching bottom edge): `is_partial_person = True`, `edge_reason = "PARTIAL_PERSON_EDGE_BOTTOM"`.
- **Fall Safety Guarantee**:
  - If `is_partial_person == True` and no prior fall was confirmed: Suppress NEW fall activations (prevents false alarms!).
  - If a fall was **already confirmed** (`has_confirmed_fall == True`): **Retain latched fall state** (`FALLEN — ON FLOOR`). Does NOT clear confirmed falls!
  - Does NOT suppress genuine sudden falls occurring in full view.

### Problem 2: HUD Overlay Obstructing Detected Person
- **Root Cause**: Large solid black background text rectangles drawn over `(x1, y1)` obscured the person's head, face, or upper body.
- **Non-Obstructive Redesign**:
  - Removed all solid black overlay rectangles placed over the person.
  - Draw a thin 2-pixel bounding box `[x1, y1, x2, y2]` around the person.
  - Render compact 1-line label directly above `y1` using crisp text stroke outlining (`cv2.putText` with black outline stroke).
  - Keeps video stream and person 100% unobstructed.

---

## 3. 10 Operational Application States

| State Index | Application State Name | Derived Logic / Trigger Condition | Visual Color Banner |
| :---: | :--- | :--- | :---: |
| 1 | `WARMING UP` | `frames_buffered < 50` | Slate Gray |
| 2 | `NO PERSON DETECTED` | `person_detected == False` or `conf_sum < 0.5` | Dark Slate / Gray |
| 3 | `NORMAL — STANDING` | $P(\text{FALL}) < 0.3650$, upright posture (`spine_angle < 35°`, `nose_y < hip_y`) | Dark Green |
| 4 | `NORMAL — WALKING` | $P(\text{FALL}) < 0.3650$, upright posture + temporal velocity > 0.015 | Green |
| 5 | `NORMAL — SITTING` | $P(\text{FALL}) < 0.3650$, sitting posture (`upper_leg / torso < 0.6`) | Green |
| 6 | `NORMAL` | $P(\text{FALL}) < 0.3650$, generic posture fallback | Emerald |
| 7 | `FALL SUSPECTED` | $P(\text{FALL}) \ge 0.3650$ for 1 or 2 consecutive windows | Amber / Orange |
| 8 | `FALL DETECTED` | $P(\text{FALL}) \ge 0.3650$ for 3+ consecutive windows (Active Alert) | Red Pulsing |
| 9 | `FALLEN — ON FLOOR` | Latched low/lying posture post-fall ($P < 0.3650$ but non-upright) | Purple |
| 10 | `GETTING UP / RECOVERY` | Post-fall upward vertical movement detected (head rising) | Yellow / Gold |
| 11 | `RECOVERED — STANDING` | Sustained upright posture post-recovery before returning to `NORMAL` | Teal |

---

## 4. Mandatory Prediction Logging Fields

Every evaluated window records:
`video_name`, `frame_index`, `win_start_frame`, `win_end_frame`, `timestamp_sec`, `person_detected`, `is_partial_person`, `edge_reason`, `fall_probability`, `decision_threshold`, `raw_prediction`, `raw_decision`, `stabilized_prediction`, `previous_application_state`, `current_application_state`, `state_transition`, `consecutive_fall_count`, `latency_ms`, `processing_fps`.

---

## 5. Scientific Honesty Notice

> **SCIENTIFIC & ENGINEERING HONESTY NOTICE**:  
> Model K1 is a binary supervised fall detector outputting $P(\text{FALL})$. Standing, walking, sitting, fallen, recovery, and recovered states are application-level interpretations derived from YOLO Pose observations, K1 fall probability, and temporal state-machine logic. They are not independent classes learned by the K1 neural network.
