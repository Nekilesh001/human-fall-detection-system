# Phase G — Production Architecture Audit: Multi-Person Tracking, Continuous Streams & Dataset Readiness

> [!IMPORTANT]
> **IMMUTABLE BASELINE MANDATE & READ-ONLY SAFETY AUDIT**  
> Baseline Model Path: `checkpoints/final_k1/final_production.pth`  
> Checkpoint SHA256: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d` (**IMMUTABLE & FROZEN**)  
> Operating Parameters: $\tau = 0.3650$, 50-frame buffer (2.0s context @ 25 FPS), 187-D spatial features, 3-consecutive-window fall confirmation.  
> Safety Policy: Zero retraining, zero checkpoint modifications, zero dataset modifications, and zero Git write operations permitted during Phase G engineering.

---

## 1. Comprehensive System Audit & Component Analysis

### 1.1 Current Inference Pipeline
The finalized Phase F5/F6 production pipeline processes input frames through a single-person linear sequence:
$$\text{Raw Video Frame} \to \text{YOLO11-Pose} \to \text{Landmark Extractor } (N=1) \to \text{187-D Features} \to \text{50-Frame Buffer} \to \text{Frozen K1 TCN} \to P(\text{FALL}) \to \tau = 0.3650 \to \text{3-Window Stabilizer} \to \text{Global StateMachine} \to \text{HUD Overlay / CSV Log / Phone Alert}$$

### 1.2 Current Person Detection Logic
In [`src/infer_final_k1.py`](file:///d:/ONE_DATA/Fall%20detection/src/infer_final_k1.py), `YOLOPoseExtractor.extract_landmarks()` invokes YOLO11-Pose (`yolov8n-pose.pt` @ conf = 0.25).  
Currently, it explicitly extracts **only index 0** (`results[0].keypoints.data[0]` and `results[0].boxes.xyxy[0]`). If multiple people are detected by YOLO in the camera frame, secondary candidates are ignored and discarded.

### 1.3 Support for Multiple People
**Not Supported**. The existing architecture assumes a single human subject per video frame. Secondary person detections are dropped.

### 1.4 Persistent Person Tracking IDs
**Non-Existent**. Bounding box coordinates `[x1, y1, x2, y2]` are evaluated independently on each frame. No IoU, centroid, or DeepSORT/ByteTrack identity association exists across frames.

### 1.5 State Machine Scoping (Per-Person vs. Global)
**Global Instance**. `ApplicationStateMachine` in [`src/final_k1_realtime_inference.py`](file:///d:/ONE_DATA/Fall%20detection/src/final_k1_realtime_inference.py) maintains a single set of global counters (`consecutive_fall_count`, `has_confirmed_fall`, `getting_up_counter`, `recovered_counter`). If multiple people were processed, their poses would cross-contaminate state transitions.

### 1.6 187-D Spatial Feature Construction
Constructed via [`src/infer_final_k1.py`](file:///d:/ONE_DATA/Fall%20detection/src/infer_final_k1.py):
- **99-D Base Landmarks**: 33 canonical COCO-mapped joints $\times (x, y, v)$.
- **66-D Temporal Velocities**: 33 canonical joints $\times (\Delta x, \Delta y)$ computed over rolling frame step.
- **22-D Spatial Body Geometry**: Torso tilt, spine-to-verticality angle, aspect ratio, joint angle geometry (knees, elbows, hips).

### 1.7 50-Frame Rolling Receptive Field Buffer
Maintained as a single 2D NumPy array `buffer_50 = np.zeros((50, 33, 3), dtype=np.float32)`. Shifted each frame via `np.roll(self.buffer_50, -1, axis=0)`.

### 1.8 Model K1 Inference Execution
`feat_187 = construct_187d_window_features(base_165)` $\to$ reshaped to PyTorch Tensor $(1, 50, 187)$ $\to$ forwarded through frozen `ModelK1_SpatialTCN` $\to$ softmax probability $P(\text{FALL}) \in [0, 1]$ evaluated against $\tau = 0.3650$.

### 1.9 Streamlit Video Pipeline Mechanics
[`app.py`](file:///d:/ONE_DATA/Fall%20detection/app.py) processes uploaded video files frame-by-frame in a `while cap.isOpened()` loop, converting BGR frames to RGB, applying overlays, updating Plotly time-series charts, and saving structured prediction CSV logs.

### 1.10 Visualization Implementation
`draw_yolo_person_overlay()` renders a single person bounding box `[x1, y1, x2, y2]`, 17 COCO pose skeleton lines, joint dots, top outline label, and dynamic corner HUD badge (`compute_hud_corner_position`).

### 1.11 Telemetry & Event Logging
Per-frame prediction dictionaries record timestamp, detection status, fall probability, raw decision, application state, transition status, and SMS/Telegram dispatch fields to CSV logs under `R&D/ML_Baseline/results/final_k1/`.

### 1.12 Webcam & RTSP IP Camera Support
Basic OpenCV webcam index 0 reading exists, but lacks robust RTSP connection resilience, auto-reconnect retry loops, stream health monitoring, or IP camera dropdown configuration.

### 1.13 Current System Limitations
1. **Single-Subject Limitation**: Cannot monitor multi-person environments (e.g. hospital wards, living rooms with multiple occupants).
2. **State Machine Contamination**: Processing multiple detections through a single state machine causes false state transitions.
3. **No ID Persistence**: Temporary occlusions or person movement cause identity loss and buffer resets.
4. **Fragile RTSP Handling**: RTSP connection drops cause unhandled OpenCV read exceptions or freeze the application.

---

## 2. Target Files & Scope for Phase G Modifications

To transition to Phase G without touching Model K1 or `final_production.pth`:

| Target File | Planned Phase G Extension |
| :--- | :--- |
| [`src/sms_alert_manager.py`](file:///d:/ONE_DATA/Fall%20detection/src/sms_alert_manager.py) / [`src/alert_manager.py`](file:///d:/ONE_DATA/Fall%20detection/src/alert_manager.py) | Per-Person ID notification dispatching (e.g. `Person #1 Fall Alert`). |
| [`src/infer_final_k1.py`](file:///d:/ONE_DATA/Fall%20detection/src/infer_final_k1.py) | Multi-person landmark extraction returning list of detected candidates `[{bbox, raw_33, coco_17}, ...]`. |
| [`src/final_k1_realtime_inference.py`](file:///d:/ONE_DATA/Fall%20detection/src/final_k1_realtime_inference.py) | **`PersonTracker` Engine**: IoU + Centroid multi-person tracking layer.<br>**`MultiPersonFallDetector` Engine**: Manages dictionary of active track states `Dict[int, TrackedPersonState]`, each maintaining independent 50-frame buffer, independent `ApplicationStateMachine`, and independent K1 inference. |
| [`app.py`](file:///d:/ONE_DATA/Fall%20detection/app.py) | Multi-person visual overlay renderer, per-ID badges, Streamlit display control toggles (`Show Person ID`, `Show Pose`, `Show BBox`, etc.), and RTSP / Webcam continuous stream manager with auto-reconnect. |
| [`src/validate_phase_g_multi_person.py`](file:///d:/ONE_DATA/Fall%20detection/src/validate_phase_g_multi_person.py) | Comprehensive 14-scenario multi-person, RTSP reconnection, and state machine isolation validation suite. |
