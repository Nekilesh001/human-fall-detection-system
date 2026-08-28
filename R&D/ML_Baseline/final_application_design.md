# Production Application Design: Real-Time Fall Detection System (Experiment #22)

> [!IMPORTANT]
> **PRODUCTION REAL-TIME APPLICATION ARCHITECTURE — MODEL K1 FROZEN**  
> Built a modular real-time fall detection application around the frozen SOTA Model K1 Champion (YOLO Pose + 187-D Spatial Features + 1D Residual TCN). The application features 50-frame rolling keypoint buffering, threshold decisioning ($\tau = 0.4923$), temporal alert stabilization (3-consecutive fall confirmation), live OpenCV HUD telemetry visualization, and support for both recorded video files and live webcam streams.

---

## 1. Modular Application System Architecture

```text
               LIVE CAMERA / WEBCAM OR RECORDED VIDEO FILE
                                   │
                                   ▼
                       YOLO Pose Keypoint Extractor
                     (17 COCO Keypoints per Frame)
                                   │
                                   ▼
                     50-Frame Rolling Keypoint Buffer
                   (33 Canonical Landmarks × 50 Frames)
                                   │
                                   ▼
                 187-D Spatial Feature Construction
             (99-D Coords + 66-D Vel + 22-D Body Angles)
                                   │
                                   ▼
                     ModelK1_SpatialTCN (CUDA/CPU)
                    (89,250 Params, 1D Residual TCN)
                                   │
                                   ▼
                      Posterior Probability P(FALL)
                                   │
                                   ▼
                    Decision Thresholding (tau = 0.4923)
                        Raw Decision: NORMAL / FALL
                                   │
                                   ▼
                     Temporal Alert Stabilizer
               (3-Consecutive Fall Confirmation Window)
                                   │
                                   ▼
                     OpenCV HUD Interface & CSV Logging
          (Skeletal Joints, P(FALL) Bar, FPS, Latency, Telemetry)
```

---

## 2. Component Specifications

### A. 50-Frame Rolling Temporal Buffer
- **Buffer Dimension**: `(50, 33, 3)` float32 array.
- **Warmup Protocol**: Displays `WARMING UP (N/50)` until 50 consecutive valid frames are received. Model inference is bypassed during warmup to prevent false positives from incomplete buffers.

### B. 187-D Spatial Feature Construction
- **Coordinate Normalization**: Normalizes joint positions by torso length $L_{\text{torso}} = \|\text{Hip}_{\text{center}} - \text{Shoulder}_{\text{center}}\|$.
- **Velocities & Angles**: Computes 66-D instantaneous velocities and 22-D body angles (Knee flexions, Hip angles, Spine inclination $\theta_{\text{spine}}$, Bounding Box Aspect Ratio $W/H$, Joint Heights).

### C. Decision & Temporal Alert Stabilization Layer
- **Operating Threshold Policy**: $\tau = 0.4923$ (Derived strictly from Experiment #19 inner validation selection).
- **Raw Decision**: `FALL` if $P(\text{FALL}) \ge 0.4923$, else `NORMAL`.
- **Temporal Stabilization Rule**:
  - Requires **$N = 3$ consecutive `FALL` decision windows** before escalating to an active `ALERT` state.
  - Decrements consecutive count on `NORMAL` frames with a 10-frame cooldown counter to prevent flickering during transient postural changes.

### D. User Interface & Telemetry HUD Overlay
- **Pill Status Header**: Top bounding bar color-coded by system state (Green for `NORMAL`, Yellow for `WARMING UP`, Red for `ALERT`).
- **Telemetry Panel**: Displays Frame index, processing FPS, latency (ms), $P(\text{FALL})$ percentage bar, threshold policy ($0.4923$), and stabilization state.
- **Skeletal Overlay**: Renders COCO keypoints and green bone connection lines over detected human subjects.

---

## 3. Supported Execution Modes

1. **Mode 1: Recorded Video Stream (`--mode video --video_path <path>`)**: Deterministic frame-by-frame evaluation of pre-recorded video files.
2. **Mode 2: Live Camera Input (`--mode webcam`)**: Continuous real-time processing from connected USB/RTSP camera feeds.
3. **Headless Execution (`--headless`)**: Non-interactive logging mode for automated validation testing.
