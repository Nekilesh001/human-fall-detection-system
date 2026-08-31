# Phase F6B Telegram Phone Alert Architecture & Integration Guide

> [!IMPORTANT]
> **STRICT CREDENTIAL SECURITY & DECOUPLED ARCHITECTURE POLICY**  
> All Telegram Bot API tokens and Chat IDs are loaded exclusively from local environment variables (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `TELEGRAM_ALERTS_ENABLED`) or local `.env`. Secrets are strictly excluded from source code, logs, CSV prediction tables, JSON summaries, and Git commits.

---

## 1. End-to-End Notification Lifecycle

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
   ├── 4. 3-Consecutive Window Alert Stabilizer (Windows 1 & 2 = FALL SUSPECTED, No Telegram alert)
   │
   ├── 5. Application State Machine Transition: FALL DETECTED (Window 3)
   │      └── Authoritative Event 1: Fires EXACTLY ONE Telegram Fall Alert 📱
   │
   ├── 6. Floor Monitoring: FALLEN — ON FLOOR
   │      └── Informational Latched State: Zero duplicate Telegram alerts sent
   │
   ├── 7. Recovery Movement: GETTING UP / RECOVERY
   │      └── Posture Tracking State: Zero new fall alerts sent
   │
   ├── 8. Application State Machine Transition: RECOVERED — STANDING
   │      └── Authoritative Event 2: Fires EXACTLY ONE Telegram Recovery Alert 📱
   │
   └── 9. State Machine Latch Reset: NORMAL — STANDING
          └── Resets alert latch, permitting subsequent independent fall alerts (Fall #2)
```

---

## 2. Explicit Architectural Distinction Matrix

| Component Layer | Responsibility | Key Outputs / States | Telegram Dispatch Action |
| :--- | :--- | :--- | :--- |
| **ML Model Layer (Model K1)** | Supervised binary fall classification over 50-frame pose tensors | $P(\text{FALL}) \in [0, 1]$, Raw Decision (`NORMAL` / `FALL` @ $\tau = 0.3650$) | **NO DIRECT TELEGRAM DISPATCH** |
| **Application State Machine** | Multi-state posture geometry, 3-window alert stabilization, recovery tracking | `WARMING UP`, `NO PERSON DETECTED`, `NORMAL — STANDING`, `NORMAL — WALKING`, `NORMAL — SITTING`, `FALL SUSPECTED`, `FALL DETECTED`, `FALLEN — ON FLOOR`, `GETTING UP / RECOVERY`, `RECOVERED — STANDING` | **AUTHORITATIVE TRIGGER SOURCE** |
| **Telegram Alert Manager** | Asynchronous, single-latch, fault-tolerant notification delivery | `send_fall_alert()`, `send_recovery_alert()`, `send_connectivity_test()` | **TELEGRAM HTTP API DISPATCH** |

---

## 3. Telegram Message Formatting Specifications

### Fall Event Notification (`send_fall_alert`)
```html
🚨 <b>FALL DETECTED</b>

📷 <b>Source</b>: <code>Coffee_room_01 video (1).avi</code>
🕐 <b>Time</b>: <b>4.80s</b> (Frame 120)
📊 <b>Fall Probability</b>: <b>94.2%</b>
🎯 <b>Threshold</b>: <code>0.3650</code>
🔔 <b>Confirmation</b>: 3 consecutive FALL windows
📍 <b>Status</b>: 🚨 <b>FALL DETECTED</b>
```

### Recovery Notification (`send_recovery_alert`)
```html
✅ <b>RECOVERY DETECTED</b>

📷 <b>Source</b>: <code>Coffee_room_01 video (1).avi</code>
🕐 <b>Time</b>: <b>7.20s</b> (Frame 180)
📍 <b>Status</b>: ✅ <b>RECOVERED — STANDING</b>
```

### Connectivity Test Notification (`send_connectivity_test`)
```html
🧪 <b>Fall Detection System</b> — Telegram test successful.
```

---

## 4. Telemetry Logging Fields

Every evaluated frame records non-sensitive operational fields:
- `telegram_alert_enabled`: `true` / `false`
- `telegram_alert_sent`: `true` / `false`
- `telegram_alert_status`: `"DELIVERED"`, `"SKIPPED_DUPLICATE"`, `"MISSING_CREDENTIALS"`, `"DISABLED"`, `"FAILED"`
- `notification_event_type`: `"FALL_ALERT"`, `"RECOVERY_ALERT"`, `"NONE"`
