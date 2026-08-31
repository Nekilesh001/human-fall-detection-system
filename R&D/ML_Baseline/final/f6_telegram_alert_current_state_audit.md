# Phase F6 Telegram Phone Alert Current State Audit

> [!IMPORTANT]
> **READ-ONLY SYSTEM AUDIT & SECURITY NOTICE**  
> Checkpoint Path: `checkpoints/final_k1/final_production.pth`  
> Checkpoint SHA256: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d` (**READ-ONLY & UNTOUCHED**)  
> Official Operating Threshold: $\tau = 0.3650$  
> Security Policy: All Telegram Bot API tokens and Chat IDs are stored strictly in local environment variables (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `TELEGRAM_ALERTS_ENABLED`). Zero secrets shall be committed to source files, logs, CSVs, or Git.

---

## 1. Current State Machine & Transition Mechanics

In [`src/final_k1_realtime_inference.py`](file:///d:/ONE_DATA/Fall%20detection/src/final_k1_realtime_inference.py), `ApplicationStateMachine` manages 11 operational states:
1. `WARMING UP`
2. `NO PERSON DETECTED`
3. `NORMAL — STANDING`
4. `NORMAL — WALKING`
5. `NORMAL — SITTING`
6. `NORMAL`
7. `FALL SUSPECTED`
8. `FALL DETECTED`
9. `FALLEN — ON FLOOR`
10. `GETTING UP / RECOVERY`
11. `RECOVERED — STANDING`

### Authoritative Alert Transitions
- **Authoritative Fall Event**: Occurs when `previous_state != "FALL DETECTED"` and `current_state == "FALL DETECTED"` (triggered on window #3 where $P(\text{FALL}) \ge 0.3650$). This sets `alert_active = True` and `has_confirmed_fall = True`.
- **Authoritative Recovery Event**: Occurs when `previous_state == "GETTING UP / RECOVERY"` and `current_state == "RECOVERED — STANDING"` (triggered after 5 consecutive upright frames post-fall).
- **Fall Latch Reset**: Reaching 5 frames in `RECOVERED — STANDING` resets `has_confirmed_fall = False` and returns to `NORMAL — STANDING`, permitting subsequent fall alerts.

---

## 2. Decoupled Alert Integration Architecture

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
   ├── 4. 3-Consecutive Window Alert Stabilizer
   │
   ├── 5. Application State Machine (`ApplicationStateMachine`)
   │      └── Evaluates state transitions (previous_state -> current_state)
   │
   └── 6. Telegram Alert Manager (`TelegramAlertManager` in src/alert_manager.py)
          ├── If Transition == "FALL DETECTED" & NOT latched: send_fall_alert() -> Telegram API
          ├── If Transition == "RECOVERED — STANDING" & NOT latched: send_recovery_alert() -> Telegram API
          └── Non-blocking, fault-tolerant execution (catches network errors without stopping video processing)
```

---

## 3. Security & Telemetry Compliance

- **`.gitignore` Audit**: Confirmed `.gitignore` excludes `.env`, `.env.*`, `credentials.json`, `*.pem`, `*.key`, and log files.
- **Environment Placeholders**: Created `.env.example` containing placeholder credentials only.
- **Telemetry Isolation**: Prediction logs record non-sensitive boolean flags (`telegram_alert_enabled`, `telegram_alert_sent`, `telegram_alert_status`, `notification_event_type`). Secret tokens are never written to CSV or JSON summary logs.
