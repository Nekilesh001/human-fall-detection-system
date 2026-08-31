# Human Fall Detection System — Production K1 Platform

A production-grade, real-time Human Fall Detection System for healthcare and hospital patient monitoring applications. The platform combines deep keypoint pose estimation, spatial body-geometry modeling, 1D Residual Temporal Convolutional Networks (Spatial TCN), and an application-level state machine.

---

## 📌 1. Pipeline Architecture

```
RAW VIDEO STREAM (.mp4 / .avi / .mov)
   │
   ├── 1. YOLO11-Pose Estimator
   │      └── Person Detection + Bounding Box [x1, y1, x2, y2] + 17 COCO Keypoints (33 Canonical Landmarks)
   │
   ├── 2. Edge-of-Frame Proximity Evaluator (`is_partial_person` & `edge_reason`)
   │      └── Evaluates boundary proximity (margin < 2%) + keypoint completeness (ankles/head/count)
   │
   ├── 3. 187-D Spatial Feature Derivation
   │      └── 165-D Base (Coords + Visibilities + Velocities) + 22-D Spatial Body Geometry (Angles, Ratios, Torso Tilt)
   │
   ├── 4. 50-Frame Rolling Temporal Buffer (2.0s Context @ 25 FPS)
   │      └── Warmup Guard (Frames 1–49 = WARMING UP)
   │
   ├── 5. Person Presence Validation Gate (`person_detected` Gate)
   │      └── Bypasses Model K1 on empty frame / keypoint confidence < 0.5 (Output: NO_PERSON_DETECTED, P(FALL)=0.0)
   │
   ├── 6. FROZEN Champion Model K1 1D Residual TCN (`checkpoints/final_k1/final_production.pth`)
   │      └── Outputs ML Model Prediction: P(FALL) ∈ [0, 1] & Raw Decision (NORMAL / FALL @ τ = 0.3650)
   │
   ├── 7. 3-Consecutive Window Alert Stabilizer
   │      └── 1–2 FALL windows = FALL SUSPECTED | 3+ FALL windows = ALERT ACTIVE
   │
   ├── 8. 10-State Application State Machine (`ApplicationStateMachine`)
   │      └── Post-fall floor latching, posture recovery tracking, edge-of-frame protection & transition logging
   │
   └── 9. Production Streamlit UI (`app.py`) & CSV Prediction Logger
          └── Anti-aliased skeleton, tight person bbox, dynamic corner HUD, Plotly probability timeline & metrics table
```

> [!IMPORTANT]
> **EXPLICIT ARCHITECTURAL SEPARATION OF MODEL OUTPUT VS. APPLICATION STATE MACHINE**  
> Champion Model K1 is a **supervised binary fall detector** outputting fall probability $P(\text{FALL}) \in [0, 1]$ and raw decision (`NORMAL` vs `FALL` @ $\tau = 0.3650$).  
> Postures (`NORMAL — STANDING`, `NORMAL — WALKING`, `NORMAL — SITTING`), recovery states (`GETTING UP / RECOVERY`, `RECOVERED — STANDING`), and floor states (`FALLEN — ON FLOOR`) are **application-level derived interpretations** combining YOLO Pose keypoints, Model K1 $P(\text{FALL})$, and state-machine transition logic. They are **NOT** independent classes learned by the K1 neural network.

---

## 🔬 2. Champion Model K1 Specification

Model K1 is a lightweight, high-performance 1D Residual Temporal Convolutional Network designed for low-latency temporal pattern recognition over continuous pose streams.

| Specification Parameter | Value / Implementation Detail |
| :--- | :--- |
| **Model Architecture** | 2 Residual 1D TCN Blocks (`channels=[64, 64]`, `kernel_size=3`, `dilations=[1, 2]`) |
| **Temporal Pooling** | Concatenated Mean Pooling + Max Pooling (128-D Temporal Feature) |
| **Classification Head** | `Linear(128 -> 32)` $\to$ `ReLU` $\to$ `Dropout(p=0.5)` $\to$ `Linear(32 -> 2)` |
| **Parameter Count** | **86,434 Core Trainable** / 89,250 Total Parameters (0.35 MB Checkpoint) |
| **Input Tensor Shape** | `(1, 50, 187)` float32 (Batch Size = 1, Window = 50 Frames, Features = 187-D) |
| **Spatial Features (187-D)**| 99-D Normalized Keypoints + 66-D Velocities + 22-D Spatial Body Geometry |
| **Temporal Context** | 50 Frames = **2.0 Seconds Context @ 25.0 FPS** (Stride = 1 frame) |
| **Production Checkpoint** | `checkpoints/final_k1/final_production.pth` |
| **Official Operating Threshold** | $\tau = 0.3650$ (Leakage-Free Inner-Validation Optimal Threshold) |
| **Alert Stabilization** | **3 Consecutive FALL Windows** ($\approx 0.12\text{ seconds}$ required for confirmed alert) |

---

## 📊 3. Methodological Benchmark & Validation Results

### Phase F2: 4-Fold Location-Observed Location-Out (LOLO) Evaluation
To guarantee complete spatial generalization and zero data leakage across room environments, Model K1 was evaluated using **4-Fold LOLO Cross-Validation** across 4 distinct Le2i location partitions (`Coffee_room_01`, `Home_01`, `Home_02`, `Office_01`). Outer test locations were strictly held out during model training and inner-fold threshold selection.

| Fold | Outer Held-Out Location | Outer Test Windows | Optimal Inner Threshold ($\tau^*_{\text{inner}}$) | LOLO Test F1 (@ $\tau = 0.50$) | LOLO Test F1 (@ $\tau^*_{\text{inner}}$) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **Fold 1** | Coffee_room_01 | 502 windows | 0.4900 | 85.06% | 85.06% |
| **Fold 2** | Home_01 | 247 windows | 0.2800 | 94.02% | 90.77% |
| **Fold 3** | Home_02 | 344 windows | 0.3800 | 79.50% | 76.54% |
| **Fold 4** | Office_01 | 303 windows | 0.3100 | 87.82% | 87.56% |
| **SUMMARY**| **4-Fold LOLO Mean** | **1,396 windows** | **$\bar{\tau}^* = 0.3650$** | **86.60%** | **84.98% $\pm$ 5.81%** |

- **LOLO Mean F1 Score (@ $\tau = 0.50$)**: **86.60%**
- **LOLO Mean Precision / Recall**: **86.72% / 87.21%**
- **Aggregated LOLO Confusion Matrix**: True Normal: 1,012 | False Fall: 53 | False Normal: 47 | True Fall: 284

### Phase F3: Full-Data Production Model Retraining
Following LOLO validation, the final production checkpoint (`final_production.pth`) was trained on **all 1,396 validated windows** using the fixed hyperparameter configuration and saved for deployment.
- **Training Epochs**: 35 Epochs (AdamW optimizer, lr = $1 \times 10^{-3}$, cosine decay, weight decay = $1 \times 10^{-4}$)
- **Loss Convergence**: Training Loss converged cleanly to **0.0608**.

---

## 📁 4. Datasets & Training Data Attribution

### Datasets Available in Project Structure
1. **Le2i Fall Detection Dataset** (`Le2i/`): Real-world room video recordings across multiple camera angles and environment scenarios.
2. **UR Fall Detection Dataset** (`URFD/`): Multi-modal RGB, Depth, and Accelerometer recordings.
3. **Custom Scenario Data** (`data/`): Additional test scenarios.

### Production Training Attribution
> [!NOTE]
> The official production checkpoint (`checkpoints/final_k1/final_production.pth`) was trained on the consolidated **1,396 validated Le2i windows** across `Coffee_room_01`, `Home_01`, `Home_02`, and `Office_01`. URFD and `data/` directories are available in the project structure for multi-dataset experimentation.

---

## ⚙️ 5. Application State Machine (10 Operational States)

The application state machine derives 10 operational states using YOLO Pose geometry, Model K1 $P(\text{FALL})$, temporal history, and persistence/recovery rules:

```
                       ┌───────────────────────────────┐
                       │       NO_PERSON_DETECTED      │ (Keypoints absent / conf_sum < 0.5)
                       └───────────────▲───────────────┘
                                       │
                       ┌───────────────┴───────────────┐
                       │          WARMING UP           │ (Frames 1–49 < 50)
                       └───────────────┬───────────────┘
                                       │
                       ┌───────────────▼───────────────┐
                       │        NORMAL — STANDING      │ (Upright, P(FALL) < 0.3650)
                       └───────────────┬───────────────┘
                                       │ P(FALL) ≥ 0.3650 (1–2 windows)
                       ┌───────────────▼───────────────┐
                       │        FALL SUSPECTED         │
                       └───────────────┬───────────────┘
                                       │ P(FALL) ≥ 0.3650 (3+ windows)
                       ┌───────────────▼───────────────┐
                       │         FALL DETECTED         │ (🚨 Active Alert Fired)
                       └───────────────┬───────────────┘
                                       │ P(FALL) < 0.3650 & Low/Horizontal Posture
                       ┌───────────────▼───────────────┐
                       │       FALLEN — ON FLOOR       │ (Latched Post-Fall Posture)
                       └───────────────┬───────────────┘
                                       │ Upward Motion / Head Rising
                       ┌───────────────▼───────────────┐
                       │     GETTING UP / RECOVERY     │
                       └───────────────┬───────────────┘
                                       │ Sustained Upright Posture (≥ 15 frames)
                       ┌───────────────▼───────────────┐
                       │      RECOVERED — STANDING     │ ──► Reverts to NORMAL — STANDING
                       └───────────────────────────────┘
```

### Operational State Definitions

| State Index | Operational State Name | Derived Logic & Trigger Conditions | Visual Badge Color |
| :---: | :--- | :--- | :---: |
| 1 | `WARMING UP` | `frames_buffered < 50` | Slate Gray |
| 2 | `NO PERSON DETECTED` | `person_detected == False` or `conf_sum < 0.5` (Bypasses K1 model) | Dark Slate / Gray |
| 3 | `NORMAL — STANDING` | $P(\text{FALL}) < 0.3650$, upright posture (`spine_angle < 35°`, `nose_y < hip_y`) | Dark Green |
| 4 | `NORMAL — WALKING` | $P(\text{FALL}) < 0.3650$, upright posture + temporal keypoint velocity > 0.015 | Green |
| 5 | `NORMAL — SITTING` | $P(\text{FALL}) < 0.3650$, sitting posture (`upper_leg / torso < 0.6`) | Green |
| 6 | `NORMAL` | $P(\text{FALL}) < 0.3650$, generic ADL fallback state | Emerald |
| 7 | `FALL SUSPECTED` | $P(\text{FALL}) \ge 0.3650$ for 1 or 2 consecutive windows | Amber / Orange |
| 8 | `FALL DETECTED` | $P(\text{FALL}) \ge 0.3650$ for 3+ consecutive windows (Active Alert Fired) | Red Pulsing |
| 9 | `FALLEN — ON FLOOR` | Latched low/lying posture post-fall ($P < 0.3650$ but non-upright) | Purple |
| 10 | `GETTING UP / RECOVERY` | Upward vertical movement detected post-fall (head height rising) | Yellow / Gold |
| 11 | `RECOVERED — STANDING` | Sustained upright posture post-recovery before returning to `NORMAL` | Teal |

### Edge-of-Frame Safety Protection
When a person moves near or outside image boundaries (margin < 2%), keypoints may become partially truncated. The boundary proximity evaluator sets `is_partial_person = True` and:
- **Suppresses NEW false fall alerts** caused by keypoint clipping when the person is upright.
- **Preserves confirmed latched falls** (`has_confirmed_fall == True`), maintaining `FALLEN — ON FLOOR`.
- **Preserves continuous 50-frame temporal buffer** without flushing or resetting.

---

## 💻 6. Production Streamlit Application (`app.py`)

The Streamlit web application (`app.py`) provides an interactive interface for model evaluation and real-time demonstration.

### Key Visual Features
- **Tight Person Bounding Box**: 2-pixel smooth rectangle drawn tightly around the detected person coordinates `[x1, y1, x2, y2]` with an attached `PERSON | Pose Detected` label tag.
- **High-Quality Anti-Aliased Skeleton**: Thin 1-pixel cyan skeleton lines (`cv2.LINE_AA`) and small 2-pixel red joint dots drawn **only on the detected person**.
- **Dynamic Non-Overlapping Video HUD (`compute_hud_corner_position`)**: Automatically places a compact semi-transparent HUD badge ($240 \times 52\text{ px}$) in a non-overlapping corner based on person location relative to frame center.
- **Full-Video Red Border Flash**: 12-pixel red perimeter flash during active `🚨 FALL DETECTED` alerts.
- **Real-Time Probability Profile**: Plotly line chart plotting $P(\text{FALL})$ vs. decision threshold $\tau = 0.3650$ over video time.
- **Prediction CSV Logger**: Generates per-window prediction CSV logs recording `previous_application_state`, `current_application_state`, `state_transition`, `is_partial_person`, `edge_reason`, `latency_ms`, and `processing_fps`.

---

## 🚀 7. Quick Start & Terminal Commands

### Environment Setup & Cellular SMS Fall Alerts
All project commands execute using Python 3.11:

```powershell
cd "d:\ONE_DATA\Fall detection"

# Set Cellular SMS Environment Variables (Local Powershell or .env)
$env:TWILIO_ACCOUNT_SID="your_twilio_account_sid_here"
$env:TWILIO_AUTH_TOKEN="your_twilio_auth_token_here"
$env:TWILIO_FROM_NUMBER="+1234567890"
$env:FALL_ALERT_TO_NUMBER="+19876543210"
$env:SMS_ALERTS_ENABLED="true"
```

### Command 1: Run Phase F6 Comprehensive Cellular SMS Validation Suite
Validates AST syntax, checkpoint existence, threshold policy $\tau = 0.3650$, 50-frame buffer, 3-window alert stabilization, SMS alert dispatcher (mocked), and zero secret exposure:

```powershell
& "C:\Users\NEKILESH\AppData\Local\Programs\Python\Python311\python.exe" src/validate_f6_sms_alerts.py
```

### Command 2: Launch Production Streamlit Application
Launches the web application interface with Cellular SMS alert telemetry at `http://localhost:8501`:

```powershell
& "C:\Users\NEKILESH\AppData\Local\Programs\Python\Python311\python.exe" -m streamlit run app.py
```

### Command 3: Standalone CLI Independent Video Evaluation
Runs real-time inference on arbitrary user-supplied test videos (.mp4, .avi, .mov) and generates isolated prediction CSVs and summary JSONs:

```powershell
& "C:\Users\NEKILESH\AppData\Local\Programs\Python\Python311\python.exe" src/test_independent_video_k1.py --video_path "Le2i/data/Coffee_room_01/Coffee_room_01/Videos/video (1).avi" --display
```

---

## 📁 8. Repository Structure

```
d:\ONE_DATA\Fall detection
├── app.py                                             # Phase F5 Production Streamlit Web Application
├── requirements.txt                                    # Python dependencies manifest
├── README.md                                          # Project documentation & technical guide
├── checkpoints/
│   └── final_k1/
│       ├── final_production.pth                       # Frozen Champion Production Checkpoint (0.35 MB)
│       └── fold_{1..4}_best.pth                       # Isolated cross-validation fold checkpoints
├── src/
│   ├── infer_final_k1.py                              # YOLO Pose Extractor & 187-D spatial feature builder
│   ├── final_k1_realtime_inference.py                 # RealtimeFallDetector engine & ApplicationStateMachine
│   ├── test_independent_video_k1.py                   # Standalone CLI video testing runner & logger
│   ├── train_final_k1.py                              # Phase F2/F3 training pipeline (Option A + B)
│   └── validate_f5_streamlit.py                       # 16-check static validation audit script
└── R&D/
    └── ML_Baseline/
        ├── final_k1_data_consolidation_readiness_audit.md # Phase F1 data audit report
        ├── final/
        │   ├── f4_false_positive_investigation.md     # Ground-truth investigation report
        │   ├── f5_streamlit_design.md                 # System architecture and Streamlit design guide
        │   ├── final_k1_evaluation_report.md          # Phase F2 LOLO benchmark evaluation report
        │   └── final_k1_training_report.md            # Phase F3 production retraining report
        └── results/
            └── final_k1/                              # Benchmark metrics, threshold JSON & prediction CSVs
```

---

## 📜 9. License & Scientific Citation

This platform is developed as a research and engineering system for human fall detection. All datasets (`Le2i`, `URFD`) belong to their respective academic authors and institutions.
