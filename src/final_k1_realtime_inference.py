"""
Experiment #22: Final Production Real-Time Fall Detection Engine & Alert Stabilization Layer.
Encapsulates frozen Model K1 inference, 50-frame temporal buffering, and temporal alert stabilization.

Architecture:
Video Frame -> YOLO Pose Extractor -> 187-D Spatial Feature Derivation
            -> 50-Frame Rolling Buffer -> Model K1 Spatial TCN -> P(FALL)
            -> Threshold (0.4923) -> Temporal Alert Stabilizer (3-Consecutive Fall Confirmation)
"""

import os
import sys
import time
import numpy as np
import torch
import cv2

ROOT_DIR = r"d:\ONE_DATA\Fall detection"
sys.path.insert(0, ROOT_DIR)

from src.infer_final_k1 import YOLOPoseExtractor, compute_165d_base_features, construct_187d_window_features
from src.train_le2i_yolo_k1_spatial import ModelK1_SpatialTCN

class TemporalAlertStabilizer:
    """Application-level post-processing layer for temporal alert stabilization."""
    def __init__(self, consecutive_required=3, cooldown_frames=10):
        self.consecutive_required = consecutive_required
        self.cooldown_frames = cooldown_frames
        self.consecutive_fall_count = 0
        self.cooldown_counter = 0
        self.alert_active = False

    def update(self, raw_decision):
        # raw_decision: 'FALL' or 'NORMAL'
        if raw_decision == 'FALL':
            self.consecutive_fall_count += 1
            if self.consecutive_fall_count >= self.consecutive_required:
                self.alert_active = True
                self.cooldown_counter = self.cooldown_frames
        else:
            self.consecutive_fall_count = max(0, self.consecutive_fall_count - 1)
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
            if self.consecutive_fall_count == 0 and self.cooldown_counter == 0:
                self.alert_active = False

        status = "ALERT" if self.alert_active else ("WARMUP_RECOVERY" if self.consecutive_fall_count > 0 else "NORMAL")
        return {
            "alert_active": self.alert_active,
            "status": status,
            "consecutive_fall_count": self.consecutive_fall_count,
            "cooldown_counter": self.cooldown_counter
        }

class RealtimeFallDetector:
    def __init__(self, checkpoint_path=None, threshold_policy=0.4923, consecutive_fall_required=3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold_policy = threshold_policy
        self.extractor = YOLOPoseExtractor()
        
        # Load Frozen Model K1
        self.model = ModelK1_SpatialTCN().to(self.device)
        if checkpoint_path is None:
            checkpoint_path = os.path.join(ROOT_DIR, "checkpoints", "le2i_yolo_k1", "fold_1_best.pth")
        
        assert os.path.exists(checkpoint_path), f"Checkpoint missing: {checkpoint_path}"
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()

        # Buffering & Stabilization State
        self.buffer_50 = np.zeros((50, 33, 3), dtype=np.float32)
        self.frames_buffered = 0
        self.stabilizer = TemporalAlertStabilizer(consecutive_required=consecutive_fall_required)
        self.last_latency_ms = 0.0
        self.last_fps = 0.0

    def process_frame(self, frame_bgr):
        t_start = time.perf_counter()
        
        # 1. YOLO Pose Keypoint Extraction
        raw_33, detected = self.extractor.extract_landmarks(frame_bgr)
        
        # 2. Update 50-Frame Rolling Buffer
        self.buffer_50 = np.roll(self.buffer_50, -1, axis=0)
        self.buffer_50[-1] = raw_33
        self.frames_buffered = min(50, self.frames_buffered + 1)

        is_warmed_up = (self.frames_buffered >= 50)
        
        if not is_warmed_up:
            t_end = time.perf_counter()
            self.last_latency_ms = (t_end - t_start) * 1000.0
            self.last_fps = 1000.0 / self.last_latency_ms if self.last_latency_ms > 0 else 0.0
            return {
                "is_warmed_up": False,
                "buffer_status": f"WARMING UP ({self.frames_buffered}/50)",
                "raw_decision": "WARMING UP",
                "prob_fall": 0.0,
                "threshold": self.threshold_policy,
                "alert_state": {"alert_active": False, "status": "WARMING_UP", "consecutive_fall_count": 0, "cooldown_counter": 0},
                "latency_ms": self.last_latency_ms,
                "fps": self.last_fps,
                "raw_33": raw_33
            }

        # 3. Derive 187-D Spatial Features
        base_165 = compute_165d_base_features(self.buffer_50)
        feat_187 = construct_187d_window_features(base_165)

        # 4. Model K1 Forward Pass
        tensor_x = torch.tensor(feat_187, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(tensor_x)
            prob_fall = torch.softmax(out, dim=1)[0, 1].item()

        raw_decision = "FALL" if prob_fall >= self.threshold_policy else "NORMAL"
        
        # 5. Temporal Alert Stabilization
        alert_state = self.stabilizer.update(raw_decision)

        t_end = time.perf_counter()
        self.last_latency_ms = (t_end - t_start) * 1000.0
        self.last_fps = 1000.0 / self.last_latency_ms if self.last_latency_ms > 0 else 0.0

        return {
            "is_warmed_up": True,
            "buffer_status": "50/50 READY",
            "raw_decision": raw_decision,
            "prob_fall": prob_fall,
            "threshold": self.threshold_policy,
            "alert_state": alert_state,
            "latency_ms": self.last_latency_ms,
            "fps": self.last_fps,
            "raw_33": raw_33
        }

    def reset_buffer(self):
        self.buffer_50.fill(0)
        self.frames_buffered = 0
        self.stabilizer = TemporalAlertStabilizer()
