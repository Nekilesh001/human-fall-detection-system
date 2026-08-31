"""
Phase F6B Production Telegram Alert Manager Service.

Architectural Design:
1. Environment Variable Loading: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TELEGRAM_ALERTS_ENABLED (Auto-loads .env if present).
2. Single Alert Latching: Prevents duplicate notification spam during continuous FALL DETECTED or FALLEN frames.
3. Recovery Notification: Sends exactly ONE notification when patient successfully recovers to upright posture.
4. Non-Blocking & Fault-Tolerant: Catches network, timeout, or credential errors cleanly without breaking video inference.
5. Zero Secret Exposure: Never reveals secret bot tokens in logs, UI, CSVs, or error outputs. Masking used for UI display.
"""

import os
import sys
import time
import requests

ROOT_DIR = r"d:\ONE_DATA\Fall detection"
sys.path.insert(0, ROOT_DIR)

def load_env_file():
    """Parses local .env file automatically if present in ROOT_DIR without requiring python-dotenv."""
    env_path = os.path.join(ROOT_DIR, ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass

class TelegramAlertManager:
    """
    Decoupled Telegram Alert Notification Service.
    Handles fall alerts, recovery notifications, single-alert latching, connectivity testing, and zero-crash fault tolerance.
    """
    def __init__(self, bot_token=None, chat_id=None, enabled=None):
        # Auto-load local .env file if present
        load_env_file()
        
        # Load from arguments or environment variables
        self.bot_token = (bot_token if bot_token is not None else os.getenv("TELEGRAM_BOT_TOKEN", "")).strip()
        self.chat_id = (chat_id if chat_id is not None else os.getenv("TELEGRAM_CHAT_ID", "")).strip()
        
        env_enabled = os.getenv("TELEGRAM_ALERTS_ENABLED", "true").lower() in ("true", "1", "yes")
        self.enabled = enabled if enabled is not None else env_enabled
        
        # State Latching Flags (Single alert per event)
        self.fall_alert_sent = False
        self.recovery_alert_sent = False
        
        self.last_status = "INITIALIZED" if self.is_configured() else "NOT_CONFIGURED"

    def is_configured(self):
        return bool(self.bot_token and self.chat_id and self.enabled)

    def get_masked_token(self):
        if not self.bot_token:
            return "NOT_CONFIGURED"
        if len(self.bot_token) < 8:
            return "***"
        return f"{self.bot_token[:4]}***{self.bot_token[-4:]}"

    def send_message(self, message_text):
        """Sends an HTML-formatted message via Telegram Bot API with zero-crash fault tolerance."""
        if not self.enabled:
            self.last_status = "DISABLED"
            return {"sent": False, "status": "DISABLED", "reason": "Alerts disabled by configuration"}
            
        if not self.bot_token or not self.chat_id:
            self.last_status = "MISSING_CREDENTIALS"
            return {"sent": False, "status": "MISSING_CREDENTIALS", "reason": "Telegram credentials not provided"}

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message_text,
            "parse_mode": "HTML"
        }

        try:
            response = requests.post(url, json=payload, timeout=5.0)
            if response.status_code == 200:
                self.last_status = "DELIVERED"
                return {"sent": True, "status": "DELIVERED"}
            else:
                self.last_status = f"FAILED_HTTP_{response.status_code}"
                return {"sent": False, "status": "FAILED", "reason": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            self.last_status = "FAILED_NETWORK_ERROR"
            return {"sent": False, "status": "FAILED", "reason": f"Network error: {str(e)}"}

    def send_connectivity_test(self):
        """Sends a manual connectivity test message to verify Telegram bot credentials."""
        msg = "🧪 <b>Fall Detection System</b> — Telegram test successful."
        return self.send_message(msg)

    def send_fall_alert(self, prob_fall, frame_index, timestamp_sec, video_name="Live Camera"):
        """Fires exactly ONE fall emergency notification per fall event when FALL DETECTED is first entered."""
        if self.fall_alert_sent:
            return {"sent": False, "status": "SKIPPED_DUPLICATE", "reason": "Fall alert already sent for current event"}

        msg = (
            f"🚨 <b>FALL DETECTED</b>\n\n"
            f"📷 <b>Source</b>: <code>{video_name}</code>\n"
            f"🕐 <b>Time</b>: <b>{timestamp_sec:.2f}s</b> (Frame {frame_index})\n"
            f"📊 <b>Fall Probability</b>: <b>{prob_fall*100:.1f}%</b>\n"
            f"🎯 <b>Threshold</b>: <code>0.3650</code>\n"
            f"🔔 <b>Confirmation</b>: 3 consecutive FALL windows\n"
            f"📍 <b>Status</b>: 🚨 <b>FALL DETECTED</b>"
        )

        res = self.send_message(msg)
        if res["sent"]:
            self.fall_alert_sent = True
            self.recovery_alert_sent = False
        return res

    def send_recovery_alert(self, frame_index, timestamp_sec, video_name="Live Camera"):
        """Fires exactly ONE recovery notification when patient reaches RECOVERED — STANDING."""
        if not self.fall_alert_sent:
            return {"sent": False, "status": "SKIPPED_NO_PRIOR_FALL", "reason": "No prior fall alert active"}

        if self.recovery_alert_sent:
            return {"sent": False, "status": "SKIPPED_DUPLICATE", "reason": "Recovery alert already sent for current event"}

        msg = (
            f"✅ <b>RECOVERY DETECTED</b>\n\n"
            f"📷 <b>Source</b>: <code>{video_name}</code>\n"
            f"🕐 <b>Time</b>: <b>{timestamp_sec:.2f}s</b> (Frame {frame_index})\n"
            f"📍 <b>Status</b>: ✅ <b>RECOVERED — STANDING</b>"
        )

        res = self.send_message(msg)
        if res["sent"]:
            self.recovery_alert_sent = True
        return res

    def reset_latch(self):
        """Resets latch state when application state machine completely clears a fall event."""
        self.fall_alert_sent = False
        self.recovery_alert_sent = False
