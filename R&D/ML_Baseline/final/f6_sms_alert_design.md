# Phase F6 Cellular SMS Phone Alert Architecture & Integration Guide

> [!IMPORTANT]
> **STRICT CREDENTIAL SECURITY & DECOUPLED CELLULAR SMS ARCHITECTURE POLICY**  
> All Twilio / SMS provider credentials (`TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER`, `FALL_ALERT_TO_NUMBER`, `SMS_ALERTS_ENABLED`) are loaded exclusively from local environment variables or local `.env`. Secrets are strictly excluded from source code, logs, CSV prediction tables, JSON summaries, and Git commits.

---

## 1. End-to-End Cellular SMS Notification Lifecycle

```
RAW VIDEO STREAM (.mp4 / .avi / .mov)
   │
   ├── 1. YOLO11-Pose Extractor (Person BBox [x1,y1,x2,y2] + 17 COCO Keypoints)
   │
   ├── 2. 187-D Spatial Feature Derivation & 50-Frame Temporal Window
   │
   ├── 3. FROZEN Model K1 1D Residual TCN (checkpoints/final_k1/final_production.pth)
   │      └── Outputs ML Model Prediction: P(FALL) ∈ [0, 1] & Raw Decision (NORMAL/FALL @ τ = 0.3650)
   │
   ├── 4. 3-Consecutive Window Alert Stabilizer (Windows 1 & 2 = FALL SUSPECTED, No SMS sent)
   │
   ├── 5. Application State Machine Transition: FALL DETECTED (Window 3)
   │      └── Authoritative Event 1: Fires EXACTLY ONE Cellular SMS Alert 💬
   │
   ├── 6. Floor Monitoring: FALLEN — ON FLOOR
   │      └── Informational Latched State: Zero duplicate SMS alerts sent
   │
   ├── 7. Recovery Movement: GETTING UP / RECOVERY
   │      └── Posture Tracking State: Zero new fall SMS sent
   │
   ├── 8. Application State Machine Transition: RECOVERED — STANDING
   │      └── Authoritative Event 2: Fires EXACTLY ONE Cellular SMS Recovery Alert 💬
   │
   └── 9. State Machine Latch Reset: NORMAL — STANDING
          └── Resets alert latch, permitting subsequent independent fall SMS alerts (Fall #2)
```

---

## 2. Explicit Architectural Distinction Matrix

| Component Layer | Responsibility | Key Outputs / States | SMS Dispatch Action |
| :--- | :--- | :--- | :--- |
| **ML Model Layer (Model K1)** | Supervised binary fall classification over 50-frame pose tensors | $P(\text{FALL}) \in [0, 1]$, Raw Decision (`NORMAL` / `FALL` @ $\tau = 0.3650$) | **NO DIRECT SMS DISPATCH** |
| **Application State Machine** | Multi-state posture geometry, 3-window alert stabilization, recovery tracking | `WARMING UP`, `NO PERSON DETECTED`, `NORMAL — STANDING`, `NORMAL — WALKING`, `NORMAL — SITTING`, `FALL SUSPECTED`, `FALL DETECTED`, `FALLEN — ON FLOOR`, `GETTING UP / RECOVERY`, `RECOVERED — STANDING` | **AUTHORITATIVE TRIGGER SOURCE** |
| **Cellular SMS Alert Manager** | Asynchronous, single-latch, fault-tolerant SMS delivery via Twilio REST API | `send_fall_alert()`, `send_recovery_alert()`, `send_connectivity_test()` | **TWILIO CELLULAR SMS DISPATCH** |

---

## 3. Cellular SMS Message Templates

### Fall Emergency SMS (`send_fall_alert`)
```text
🚨 FALL DETECTED

Camera: Coffee_room_01 video (1).avi
Time: 4.80s
Fall Probability: 94.2%
Threshold: 36.5%
Confirmation: 3 consecutive windows

Status: FALL DETECTED
```

### Recovery SMS (`send_recovery_alert`)
```text
✅ RECOVERY DETECTED

Camera: Coffee_room_01 video (1).avi
Time: 7.20s

Status: RECOVERED — STANDING
```

### Connectivity Test SMS (`send_connectivity_test`)
```text
🧪 Fall Detection System SMS test successful.
```

---

## 4. Environment Variables & Telemetry Logging Fields

```env
TWILIO_ACCOUNT_SID=your_twilio_account_sid_here
TWILIO_AUTH_TOKEN=your_twilio_auth_token_here
TWILIO_FROM_NUMBER=+1234567890
FALL_ALERT_TO_NUMBER=+19876543210
SMS_ALERTS_ENABLED=true
```

Every evaluated frame records non-sensitive operational fields:
- `sms_alert_enabled`: `true` / `false`
- `sms_alert_sent`: `true` / `false`
- `sms_alert_status`: `"DELIVERED"`, `"SKIPPED_DUPLICATE"`, `"MISSING_CREDENTIALS"`, `"DISABLED"`, `"FAILED"`
- `notification_event_type`: `"FALL_ALERT"`, `"RECOVERY_ALERT"`, `"NONE"`
