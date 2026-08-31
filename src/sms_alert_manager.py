"""
Phase F6 Production Cellular SMS Alert Manager Service (Twilio / SMS Provider API).

Architectural Design:
1. Environment Variable Loading: TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER, FALL_ALERT_TO_NUMBER, SMS_ALERTS_ENABLED (Auto-loads .env if present).
2. Single Alert Latching: Prevents duplicate notification spam during continuous FALL DETECTED or FALLEN frames.
3. Recovery Notification: Sends exactly ONE notification when patient successfully recovers to upright posture.
4. Non-Blocking & Fault-Tolerant: Catches network, timeout, or credential errors cleanly without breaking video inference.
5. Zero Secret Exposure: Never reveals auth tokens, account secrets, or sensitive credentials in logs, UI, CSVs, or error outputs. Masking used for UI display.
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
                        if k and v:
                            os.environ[k] = v
        except Exception:
            pass

class SMSAlertManager:
    """
    Decoupled Cellular SMS Alert Notification Service (Twilio REST API).
    Handles fall SMS alerts, recovery notifications, single-alert latching, connectivity testing, and zero-crash fault tolerance.
    """
    def __init__(self, account_sid=None, auth_token=None, from_number=None, to_number=None, enabled=None):
        # Auto-load local .env file if present
        load_env_file()
        
        # Load from arguments or environment variables
        self.account_sid = (account_sid if account_sid is not None else os.getenv("TWILIO_ACCOUNT_SID", "")).strip()
        self.auth_token = (auth_token if auth_token is not None else os.getenv("TWILIO_AUTH_TOKEN", "")).strip()
        self.from_number = (from_number if from_number is not None else os.getenv("TWILIO_FROM_NUMBER", "")).strip()
        self.to_number = (to_number if to_number is not None else os.getenv("FALL_ALERT_TO_NUMBER", "")).strip()
        
        env_enabled = os.getenv("SMS_ALERTS_ENABLED", "true").lower() in ("true", "1", "yes")
        self.enabled = enabled if enabled is not None else env_enabled
        
        # State Latching Flags (Single alert per event)
        self.fall_alert_sent = False
        self.recovery_alert_sent = False
        
        self.last_status = "INITIALIZED" if self.is_configured() else "NOT_CONFIGURED"

    def is_configured(self):
        return bool(self.account_sid and self.auth_token and self.from_number and self.to_number and self.enabled)

    def get_masked_account_sid(self):
        if not self.account_sid:
            return "NOT_CONFIGURED"
        if len(self.account_sid) < 8:
            return "***"
        return f"{self.account_sid[:4]}***{self.account_sid[-4:]}"

    def get_masked_to_number(self):
        if not self.to_number:
            return "NOT_CONFIGURED"
        if len(self.to_number) < 6:
            return "***"
        return f"{self.to_number[:3]}*****{self.to_number[-4:]}"

    def send_sms(self, message_text):
        """Sends a plain text cellular SMS via Twilio HTTP REST API with zero-crash fault tolerance."""
        if not self.enabled:
            self.last_status = "DISABLED"
            return {"sent": False, "status": "DISABLED", "reason": "SMS alerts disabled by configuration"}
            
        if not self.account_sid or not self.auth_token or not self.from_number or not self.to_number:
            self.last_status = "MISSING_CREDENTIALS"
            return {"sent": False, "status": "MISSING_CREDENTIALS", "reason": "Twilio SMS credentials not provided"}

        url = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Messages.json"
        data = {
            "From": self.from_number,
            "To": self.to_number,
            "Body": message_text
        }

        try:
            response = requests.post(
                url,
                data=data,
                auth=(self.account_sid, self.auth_token),
                timeout=5.0
            )
            if response.status_code in (200, 201):
                self.last_status = "DELIVERED"
                return {"sent": True, "status": "DELIVERED"}
            else:
                self.last_status = f"FAILED_HTTP_{response.status_code}"
                return {"sent": False, "status": "FAILED", "reason": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            self.last_status = "FAILED_NETWORK_ERROR"
            return {"sent": False, "status": "FAILED", "reason": f"Network error: {str(e)}"}

    def send_connectivity_test(self):
        """Sends a manual connectivity test SMS to verify Twilio credentials."""
        msg = "🧪 Fall Detection System SMS test successful."
        return self.send_sms(msg)

    def send_fall_alert(self, prob_fall, frame_index, timestamp_sec, video_name="Live Camera"):
        """Fires exactly ONE fall emergency SMS per fall event when FALL DETECTED is first entered."""
        if self.fall_alert_sent:
            return {"sent": False, "status": "SKIPPED_DUPLICATE", "reason": "Fall SMS already sent for current event"}

        msg = (
            f"🚨 FALL DETECTED\n\n"
            f"Camera: {video_name}\n"
            f"Time: {timestamp_sec:.2f}s\n"
            f"Fall Probability: {prob_fall*100:.1f}%\n"
            f"Threshold: 36.5%\n"
            f"Confirmation: 3 consecutive windows\n\n"
            f"Status: FALL DETECTED"
        )

        res = self.send_sms(msg)
        if res["sent"]:
            self.fall_alert_sent = True
            self.recovery_alert_sent = False
        return res

    def send_recovery_alert(self, frame_index, timestamp_sec, video_name="Live Camera"):
        """Fires exactly ONE recovery SMS when patient reaches RECOVERED — STANDING."""
        if not self.fall_alert_sent:
            return {"sent": False, "status": "SKIPPED_NO_PRIOR_FALL", "reason": "No prior fall alert active"}

        if self.recovery_alert_sent:
            return {"sent": False, "status": "SKIPPED_DUPLICATE", "reason": "Recovery SMS already sent for current event"}

        msg = (
            f"✅ RECOVERY DETECTED\n\n"
            f"Camera: {video_name}\n"
            f"Time: {timestamp_sec:.2f}s\n\n"
            f"Status: RECOVERED — STANDING"
        )

        res = self.send_sms(msg)
        if res["sent"]:
            self.recovery_alert_sent = True
        return res

    def reset_latch(self):
        """Resets latch state when application state machine completely clears a fall event."""
        self.fall_alert_sent = False
        self.recovery_alert_sent = False
