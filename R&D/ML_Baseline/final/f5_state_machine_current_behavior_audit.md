# Phase F5 State Machine Current Behavior Audit

> [!IMPORTANT]
> **READ-ONLY SYSTEM AUDIT**  
> Model K1 Checkpoint SHA256: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d`  
> Official Decision Threshold: $\tau = 0.3650$  
> Model K1 is a binary supervised fall detector. All operational states are derived at the application layer.

---

## 1. Current Signal Determination Architecture

- **Person Detection & Bounding Box**: Extracted in `YOLOPoseExtractor.extract_landmarks()` (`src/infer_final_k1.py`) from YOLO11/YOLOv8 pose prediction results `results[0].boxes.xyxy[0]` (`[x1, y1, x2, y2]`). Validated in `RealtimeFallDetector.process_frame()` (`src/final_k1_realtime_inference.py`) where `conf_sum = np.sum(raw_33[:, 2]) < 0.5` sets `person_detected = False`.
- **Pose / Keypoints**: 17 COCO keypoints mapped to 33 canonical landmarks `raw_33` (normalized $[0, 1]$ coordinates).
- **$P(\text{FALL})$ & Raw Decision**: 187-D spatial window features `(1, 50, 187)` fed into frozen Model K1 (`ModelK1_SpatialTCN`). $P(\text{FALL}) = \text{softmax}(out)[0, 1]$. Decision: `FALL` if $P \ge 0.3650$, else `NORMAL`.
- **Fall Confirmation**: Requires **3 consecutive windows** with $P(\text{FALL}) \ge 0.3650$ (`consecutive_fall_count >= 3`). Sets `alert_active = True` and `has_confirmed_fall = True`.
- **Fallen / On Floor State**: When $P(\text{FALL}) < 0.3650$ post-fall, if `has_confirmed_fall == True` and `is_upright == False` (`spine_angle >= 35°` or `nose_y >= mid_hip_y`), state transitions to `FALLEN — ON FLOOR`.
- **Getting Up / Recovery**: Triggered when `has_confirmed_fall == True`, $P < 0.3650$, and `is_upright == True`. Increments `getting_up_counter` up to 10 frames.
- **Recovered State**: Triggered when `getting_up_counter >= 10`. Increments `recovered_counter` up to 15 frames before resetting `has_confirmed_fall = False` and reverting to `NORMAL — STANDING`.

---

## 2. Current Scenario Behaviors

### Scenario 1: Standing $\to$ Sudden Fall $\to$ Remains on Floor
- **Behavior**: `NORMAL — STANDING` $\to$ `FALL SUSPECTED` (windows 1, 2) $\to$ `FALL DETECTED` (window 3, alert active) $\to$ `FALLEN — ON FLOOR` (latched while lying on floor).

### Scenario 2: Standing $\to$ Sudden Fall $\to$ Immediately Starts Getting Up $\to$ Stands
- **Behavior**: `NORMAL — STANDING` $\to$ `FALL SUSPECTED` $\to$ `FALL DETECTED` $\to$ `GETTING UP / RECOVERY` (10 frames) $\to$ `RECOVERED — STANDING` (15 frames) $\to$ `NORMAL — STANDING`.
- **Latency Observation**: Currently requires 10 frames ($\approx 0.4\text{s}$) in `GETTING UP / RECOVERY` + 15 frames ($\approx 0.6\text{s}$) in `RECOVERED — STANDING` (total 25 frames $\approx 1.0\text{s}$).

### Scenario 3: Standing $\to$ Sits Normally
- **Behavior**: `NORMAL — STANDING` $\to$ `NORMAL — SITTING` (`spine_angle < 45°`, `upper_leg/torso < 0.6`, `mid_hip_y < mid_ankle_y`). Model K1 $P(\text{FALL})$ stays $< 0.3650$. No false fall triggered.

### Scenario 4: Normal Walking
- **Behavior**: `NORMAL — WALKING` (`is_upright == True`, keypoint velocity `vel > 0.015`). Model K1 $P(\text{FALL})$ stays $< 0.3650$.

### Scenario 5: Person Disappears
- **Behavior**: `conf_sum < 0.5` sets `person_detected = False`. Bypasses K1 model ($P=0.0$), resets `consecutive_fall_count = 0`, sets `alert_active = False`, state becomes `NO PERSON DETECTED`.

### Scenario 6: Partially Clipped by Camera Boundary
- **Behavior**: `is_partial_person = True` (margin $< 2\%$ + missing keypoints). Suppresses NEW fall activations (`consecutive_fall_count = 0`, stays `NORMAL`). Does NOT clear latched falls (`has_confirmed_fall`). No "PARTIAL FALL" state exists.

---

## 3. Misclassification Risk Analysis

1. **Rapid Recovery Hysteresis**: 25 frames (10 getting up + 15 recovered) is safe against noisy single frames, but can be slightly tuned to 5 + 5 = 10 frames ($\approx 0.4\text{s}$) for faster response during rapid stand-ups.
2. **Keypoint Occlusion Post-Fall**: If a person falls behind furniture and keypoints drop (`conf_sum < 0.5`), state transitions to `NO PERSON DETECTED`. If keypoints re-appear non-upright, `has_confirmed_fall` restores `FALLEN — ON FLOOR`.
