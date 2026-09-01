"""
Phase F5 Production Pipeline & Application State Machine.

ARCHITECTURE SEPARATION:
1. YOLO11-Pose         : Detects person, keypoints (33 landmarks), confidence, bounding box, pose geometry.
2. Frozen Model K1     : Supervised Binary Fall Detector -> Outputs P(FALL) and raw decision (NORMAL/FALL @ tau=0.3650).
3. Application Machine : State Machine combining YOLO Pose geometry, K1 P(FALL), and temporal logic (10 States).

10 APPLICATION STATES:
1. WARMING UP
2. NORMAL — STANDING
3. NORMAL — WALKING
4. NORMAL — SITTING
5. NORMAL (Generic Fallback)
6. FALL SUSPECTED (1-2 consecutive windows @ P >= 0.3650)
7. FALL DETECTED (3+ consecutive windows @ P >= 0.3650)
8. FALLEN — ON FLOOR (Latched post-fall low posture)
9. GETTING UP / RECOVERY (Upward motion / unbending post-fall)
10. RECOVERED — STANDING (Sustained upright posture post-recovery)
11. NO PERSON DETECTED (Empty room / person left frame)

SCIENTIFIC NOTICE:
Model K1 is a binary fall detector. Standing, walking, sitting, fallen, recovery, and recovered
are application-level derived states from YOLO Pose geometry and state-machine logic. They are not
independent classes learned by the K1 neural network.
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
from src.train_final_k1 import ModelK1_SpatialTCN
from src.alert_manager import TelegramAlertManager

class ApplicationStateMachine:
    """
    Application-level state machine representing 10 operational states.
    Derives user-facing state from YOLO Pose geometry, Model K1 P(FALL), and temporal history.
    """
    def __init__(self, consecutive_required=3, threshold=0.3650):
        self.consecutive_required = consecutive_required
        self.threshold = threshold
        
        self.consecutive_fall_count = 0
        self.current_state = "WARMING UP"
        self.previous_state = "WARMING UP"
        self.alert_active = False
        
        self.has_confirmed_fall = False
        self.getting_up_counter = 0
        self.recovered_counter = 0

    def update(self, prob_fall, person_detected, raw_33, buffer_50, is_partial_person=False, edge_reason="FULL_PERSON"):
        self.previous_state = self.current_state
        
        # 1. NO PERSON DETECTED GATE
        if not person_detected:
            self.consecutive_fall_count = 0
            self.alert_active = False
            self.current_state = "NO PERSON DETECTED"
            transition = f"{self.previous_state} -> {self.current_state}"
            return {
                "current_state": self.current_state,
                "previous_state": self.previous_state,
                "state_transition": transition,
                "alert_active": False,
                "consecutive_fall_count": 0,
                "is_partial_person": False,
                "edge_reason": edge_reason
            }

        # 2. Extract Pose Geometry for Sub-Classification & Recovery
        # Keypoints: nose=0, L_shoulder=11, R_shoulder=12, L_hip=23, R_hip=24, L_knee=25, R_knee=26, L_ankle=27, R_ankle=28
        nose_y = raw_33[0, 1]
        mid_shoulder_y = (raw_33[11, 1] + raw_33[12, 1]) / 2.0
        mid_hip_y = (raw_33[23, 1] + raw_33[24, 1]) / 2.0
        mid_knee_y = (raw_33[25, 1] + raw_33[26, 1]) / 2.0
        mid_ankle_y = (raw_33[27, 1] + raw_33[28, 1]) / 2.0
        
        torso_len = abs(mid_hip_y - mid_shoulder_y) + 1e-6
        upper_leg_len = abs(mid_knee_y - mid_hip_y) + 1e-6
        
        # Spine verticality angle
        dx = (raw_33[12, 0] + raw_33[11, 0])/2.0 - (raw_33[24, 0] + raw_33[23, 0])/2.0
        dy = mid_hip_y - mid_shoulder_y
        spine_angle_deg = np.abs(np.degrees(np.arctan2(dx, dy + 1e-6)))
        
        # Posture geometry checks
        is_upright = (spine_angle_deg < 35.0) and (nose_y < mid_hip_y)
        is_sitting = (spine_angle_deg < 45.0) and (upper_leg_len / torso_len < 0.6) and (mid_hip_y < mid_ankle_y)
        
        # Check raw K1 binary prediction against official threshold
        is_fall_window = (prob_fall >= self.threshold)
        
        if is_fall_window:
            # Partial person edge guard: suppress NEW fall activations when keypoints are clipped at frame boundary
            if is_partial_person and not self.has_confirmed_fall:
                pass
            else:
                self.consecutive_fall_count += 1
        else:
            self.consecutive_fall_count = 0

        # State Transition Evaluator
        if self.consecutive_fall_count >= self.consecutive_required:
            self.alert_active = True
            self.has_confirmed_fall = True
            self.current_state = "FALL DETECTED"
            self.getting_up_counter = 0
            self.recovered_counter = 0
        elif self.consecutive_fall_count > 0:
            if not self.has_confirmed_fall:
                self.current_state = "FALL SUSPECTED"
        else:
            # Evaluate posture geometry post-fall or normal
            nose_y = raw_33[0, 1]
            mid_shoulder_y = (raw_33[11, 1] + raw_33[12, 1]) / 2.0
            mid_hip_y = (raw_33[23, 1] + raw_33[24, 1]) / 2.0
            mid_knee_y = (raw_33[25, 1] + raw_33[26, 1]) / 2.0
            mid_ankle_y = (raw_33[27, 1] + raw_33[28, 1]) / 2.0
            
            torso_len = abs(mid_hip_y - mid_shoulder_y) + 1e-6
            upper_leg_len = abs(mid_knee_y - mid_hip_y) + 1e-6
            
            # Spine verticality angle
            dx = (raw_33[12, 0] + raw_33[11, 0])/2.0 - (raw_33[24, 0] + raw_33[23, 0])/2.0
            dy = mid_hip_y - mid_shoulder_y
            spine_angle_deg = np.abs(np.degrees(np.arctan2(dx, dy + 1e-6)))
            
            # Posture geometry checks
            is_upright = (spine_angle_deg < 35.0) and (nose_y < mid_hip_y)
            is_sitting = (spine_angle_deg < 45.0) and (upper_leg_len / torso_len < 0.6) and (mid_hip_y < mid_ankle_y)
            
            # Calculate temporal velocity over buffer
            if len(buffer_50) >= 5:
                prev_pose = buffer_50[-5, :, :2]
                curr_pose = raw_33[:, :2]
                if np.sum(buffer_50[-5, :, 2]) > 0.5:
                    vel = np.mean(np.abs(curr_pose - prev_pose))
                else:
                    vel = np.mean(np.abs(np.diff(buffer_50[-5:, :, 0], axis=0)))
            else:
                vel = 0.0

            is_walking = is_upright and (vel > 0.015)

            if self.has_confirmed_fall:
                if is_upright:
                    self.getting_up_counter += 1
                    if self.getting_up_counter < 5:
                        self.current_state = "GETTING UP / RECOVERY"
                    else:
                        self.recovered_counter += 1
                        if self.recovered_counter < 5:
                            self.current_state = "RECOVERED — STANDING"
                        else:
                            self.current_state = "NORMAL — STANDING"
                            self.has_confirmed_fall = False
                            self.alert_active = False
                            self.getting_up_counter = 0
                            self.recovered_counter = 0
                else:
                    self.current_state = "FALLEN — ON FLOOR"
                    self.getting_up_counter = 0
                    self.recovered_counter = 0
            else:
                self.alert_active = False
                if is_walking:
                    self.current_state = "NORMAL — WALKING"
                elif is_sitting:
                    self.current_state = "NORMAL — SITTING"
                elif is_upright:
                    self.current_state = "NORMAL — STANDING"
                else:
                    self.current_state = "NORMAL"

        return self._build_state_dict()

    def _build_state_dict(self):
        state_trans = f"{self.previous_state} -> {self.current_state}"
        return {
            "current_state": self.current_state,
            "previous_state": self.previous_state,
            "state_transition": state_trans,
            "alert_active": self.alert_active,
            "has_confirmed_fall": self.has_confirmed_fall,
            "consecutive_fall_count": self.consecutive_fall_count
        }


class RealtimeFallDetector:
    """Production Real-Time Fall Detection Engine with Explicit Architecture Separation."""
    def __init__(self, checkpoint_path=None, threshold_policy=0.3650, consecutive_fall_required=3, alert_manager=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold_policy = threshold_policy
        self.extractor = YOLOPoseExtractor()
        
        # Load Frozen Model K1
        self.model = ModelK1_SpatialTCN().to(self.device)
        if checkpoint_path is None:
            checkpoint_path = os.path.join(ROOT_DIR, "checkpoints", "final_k1", "final_production.pth")
            
        assert os.path.exists(checkpoint_path), f"Production checkpoint missing: {checkpoint_path}"
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device, weights_only=True))
        self.model.eval()

        # Buffering, State Machine & SMS Alert Manager
        self.buffer_50 = np.zeros((50, 33, 3), dtype=np.float32)
        self.frames_buffered = 0
        self.frame_count = 0
        self.state_machine = ApplicationStateMachine(
            consecutive_required=consecutive_fall_required,
            threshold=threshold_policy
        )
        self.alert_manager = alert_manager if alert_manager is not None else SMSAlertManager()
        self.last_latency_ms = 0.0
        self.last_fps = 0.0

    def process_frame(self, frame_bgr, video_name="Live Stream", fps_src=25.0):
        t_start = time.perf_counter()
        self.frame_count += 1
        timestamp_sec = self.frame_count / (fps_src if fps_src > 0 else 25.0)
        
        # 1. YOLO11-Pose Keypoint, Bounding Box & Edge-of-Frame Extraction
        raw_33, person_detected, bbox, coco_17_px, is_partial_person, edge_reason = self.extractor.extract_landmarks(frame_bgr)
        
        # Keypoint confidence sum validation
        conf_sum = np.sum(raw_33[:, 2])
        if conf_sum < 0.5:
            person_detected = False
            bbox = None
            coco_17_px = None
            is_partial_person = False
            edge_reason = "NO_PERSON"
            
        # 2. Update 50-Frame Rolling Buffer
        self.buffer_50 = np.roll(self.buffer_50, -1, axis=0)
        self.buffer_50[-1] = raw_33
        self.frames_buffered = min(50, self.frames_buffered + 1)

        is_warmed_up = (self.frames_buffered >= 50)
        
        # 3. NO PERSON DETECTED GATE
        if not person_detected:
            t_end = time.perf_counter()
            self.last_latency_ms = (t_end - t_start) * 1000.0
            self.last_fps = 1000.0 / self.last_latency_ms if self.last_latency_ms > 0 else 0.0
            
            sm_res = self.state_machine.update(
                prob_fall=0.0,
                person_detected=False,
                raw_33=raw_33,
                buffer_50=self.buffer_50,
                is_partial_person=False,
                edge_reason="NO_PERSON"
            )
            
            return {
                "is_warmed_up": is_warmed_up,
                "person_detected": False,
                "is_partial_person": False,
                "edge_reason": "NO_PERSON",
                "buffer_status": f"NO PERSON DETECTED ({self.frames_buffered}/50)",
                "raw_decision": "NO_PERSON",
                "prob_fall": 0.0,
                "threshold": self.threshold_policy,
                "current_state": "NO PERSON DETECTED",
                "previous_state": sm_res["previous_state"],
                "state_transition": sm_res["state_transition"],
                "alert_state": sm_res,
                "sms_alert_enabled": self.alert_manager.enabled,
                "sms_alert_sent": False,
                "sms_alert_status": self.alert_manager.last_status,
                "notification_event_type": "NONE",
                "latency_ms": self.last_latency_ms,
                "fps": self.last_fps,
                "raw_33": raw_33,
                "bbox": None,
                "coco_17_px": None
            }
            
        if not is_warmed_up:
            t_end = time.perf_counter()
            self.last_latency_ms = (t_end - t_start) * 1000.0
            self.last_fps = 1000.0 / self.last_latency_ms if self.last_latency_ms > 0 else 0.0
            
            return {
                "is_warmed_up": False,
                "person_detected": True,
                "is_partial_person": is_partial_person,
                "edge_reason": edge_reason,
                "buffer_status": f"WARMING UP ({self.frames_buffered}/50)",
                "raw_decision": "WARMING UP",
                "prob_fall": 0.0,
                "threshold": self.threshold_policy,
                "current_state": "WARMING UP",
                "previous_state": "WARMING UP",
                "state_transition": "WARMING UP -> WARMING UP",
                "alert_state": {"current_state": "WARMING UP", "previous_state": "WARMING UP", "state_transition": "WARMING UP -> WARMING UP", "alert_active": False, "consecutive_fall_count": 0},
                "sms_alert_enabled": self.alert_manager.enabled,
                "sms_alert_sent": False,
                "sms_alert_status": self.alert_manager.last_status,
                "notification_event_type": "NONE",
                "latency_ms": self.last_latency_ms,
                "fps": self.last_fps,
                "raw_33": raw_33,
                "bbox": bbox,
                "coco_17_px": coco_17_px
            }

        # 4. Derive 187-D Spatial Features & Model K1 Inference
        base_165 = compute_165d_base_features(self.buffer_50)
        feat_187 = construct_187d_window_features(base_165)

        tensor_x = torch.tensor(feat_187, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(tensor_x)
            prob_fall = torch.softmax(out, dim=1)[0, 1].item()

        raw_decision = "FALL" if prob_fall >= self.threshold_policy else "NORMAL"
        
        # 5. Application State Machine Update
        sm_res = self.state_machine.update(
            prob_fall=prob_fall,
            person_detected=True,
            raw_33=raw_33,
            buffer_50=self.buffer_50,
            is_partial_person=is_partial_person,
            edge_reason=edge_reason
        )

        # 6. Cellular SMS Alert Dispatcher (Authoritative State Machine Transitions)
        prev_st = sm_res["previous_state"]
        curr_st = sm_res["current_state"]
        
        sms_sent = False
        sms_event = "NONE"

        if prev_st != "FALL DETECTED" and curr_st == "FALL DETECTED":
            sms_res = self.alert_manager.send_fall_alert(
                prob_fall=prob_fall,
                frame_index=self.frame_count,
                timestamp_sec=timestamp_sec,
                video_name=video_name
            )
            sms_event = "FALL_ALERT"
            sms_sent = sms_res.get("sent", False)
        elif prev_st == "GETTING UP / RECOVERY" and curr_st == "RECOVERED — STANDING":
            sms_res = self.alert_manager.send_recovery_alert(
                frame_index=self.frame_count,
                timestamp_sec=timestamp_sec,
                video_name=video_name
            )
            sms_event = "RECOVERY_ALERT"
            sms_sent = sms_res.get("sent", False)
        elif curr_st == "NORMAL — STANDING" and not self.state_machine.has_confirmed_fall:
            self.alert_manager.reset_latch()

        t_end = time.perf_counter()
        self.last_latency_ms = (t_end - t_start) * 1000.0
        self.last_fps = 1000.0 / self.last_latency_ms if self.last_latency_ms > 0 else 0.0

        return {
            "is_warmed_up": True,
            "person_detected": True,
            "is_partial_person": is_partial_person,
            "edge_reason": edge_reason,
            "buffer_status": "50/50 READY",
            "raw_decision": raw_decision,
            "prob_fall": prob_fall,
            "threshold": self.threshold_policy,
            "current_state": sm_res["current_state"],
            "previous_state": sm_res["previous_state"],
            "state_transition": sm_res["state_transition"],
            "alert_state": sm_res,
            "sms_alert_enabled": self.alert_manager.enabled,
            "sms_alert_sent": sms_sent,
            "sms_alert_status": self.alert_manager.last_status,
            "notification_event_type": sms_event,
            "latency_ms": self.last_latency_ms,
            "fps": self.last_fps,
            "raw_33": raw_33,
            "bbox": bbox,
            "coco_17_px": coco_17_px
        }

    def reset_buffer(self):
        self.buffer_50.fill(0)
        self.frames_buffered = 0
        self.state_machine = ApplicationStateMachine(
            consecutive_required=3,
            threshold=self.threshold_policy
        )

def compute_iou(box1, box2):
    if box1 is None or box2 is None:
        return 0.0
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - intersection + 1e-6
    return float(intersection / union)

class TrackedPersonState:
    def __init__(self, person_id, bbox, raw_33, coco_17_px, is_partial_person, edge_reason, threshold_policy=0.3650):
        self.person_id = person_id
        self.bbox = bbox
        self.raw_33 = raw_33
        self.coco_17_px = coco_17_px
        self.is_partial_person = is_partial_person
        self.edge_reason = edge_reason
        self.threshold_policy = threshold_policy
        
        self.buffer_50 = np.zeros((50, 33, 3), dtype=np.float32)
        self.frames_buffered = 0
        self.state_machine = ApplicationStateMachine(consecutive_required=3, threshold=threshold_policy)
        self.prob_fall = 0.0
        self.raw_decision = "WARMING UP"
        self.alert_state = {}
        self.missed_frames = 0
        self.last_seen_frame = 0

class PersonTracker:
    def __init__(self, max_age=30, iou_threshold=0.30):
        self.tracks = {}  # person_id -> TrackedPersonState
        self.next_id = 1
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.frame_index = 0

    def update(self, candidates, threshold_policy=0.3650):
        self.frame_index += 1
        matched_track_ids = set()
        matched_cand_indices = set()

        track_ids = list(self.tracks.keys())
        
        # Greedy IoU matching
        if len(track_ids) > 0 and len(candidates) > 0:
            iou_matrix = np.zeros((len(track_ids), len(candidates)), dtype=np.float32)
            for i, t_id in enumerate(track_ids):
                for j, cand in enumerate(candidates):
                    iou_matrix[i, j] = compute_iou(self.tracks[t_id].bbox, cand["bbox"])
            
            while True:
                if iou_matrix.size == 0:
                    break
                max_val = np.max(iou_matrix)
                if max_val < self.iou_threshold:
                    break
                i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                t_id = track_ids[i]
                
                if t_id not in matched_track_ids and j not in matched_cand_indices:
                    cand = candidates[j]
                    tr = self.tracks[t_id]
                    tr.bbox = cand["bbox"]
                    tr.raw_33 = cand["raw_33"]
                    tr.coco_17_px = cand["coco_17_px"]
                    tr.is_partial_person = cand["is_partial_person"]
                    tr.edge_reason = cand["edge_reason"]
                    tr.missed_frames = 0
                    tr.last_seen_frame = self.frame_index
                    
                    matched_track_ids.add(t_id)
                    matched_cand_indices.add(j)

                iou_matrix[i, :] = -1.0
                iou_matrix[:, j] = -1.0

        # Unmatched tracks
        for t_id, tr in list(self.tracks.items()):
            if t_id not in matched_track_ids:
                tr.missed_frames += 1
                if tr.missed_frames > self.max_age:
                    del self.tracks[t_id]

        # New candidates -> Spawn new person_id tracks
        for j, cand in enumerate(candidates):
            if j not in matched_cand_indices:
                new_id = self.next_id
                self.next_id += 1
                tr = TrackedPersonState(
                    person_id=new_id,
                    bbox=cand["bbox"],
                    raw_33=cand["raw_33"],
                    coco_17_px=cand["coco_17_px"],
                    is_partial_person=cand["is_partial_person"],
                    edge_reason=cand["edge_reason"],
                    threshold_policy=threshold_policy
                )
                tr.last_seen_frame = self.frame_index
                self.tracks[new_id] = tr

        return list(self.tracks.values())

class MultiPersonFallDetector:
    def __init__(self, checkpoint_path=None, threshold_policy=0.3650, device=None):
        if checkpoint_path is None:
            checkpoint_path = os.path.join(ROOT_DIR, "checkpoints", "final_k1", "final_production.pth")
        
        self.checkpoint_path = checkpoint_path
        self.threshold_policy = threshold_policy
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        from src.infer_final_k1 import YOLOPoseExtractor, compute_165d_base_features, construct_187d_window_features
        from src.train_final_k1 import ModelK1_SpatialTCN
        
        self._compute_base = compute_165d_base_features
        self._construct_187 = construct_187d_window_features
        
        self.extractor = YOLOPoseExtractor()
        self.model = ModelK1_SpatialTCN(in_channels=187, num_classes=2).to(self.device)
        
        state_dict = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        self.tracker = PersonTracker(max_age=30, iou_threshold=0.30)
        self.alert_manager = SMSAlertManager()
        self.frame_count = 0

    def process_frame(self, frame_bgr, video_name="Live Stream", fps_src=25.0):
        t_start = time.perf_counter()
        self.frame_count += 1
        timestamp_sec = self.frame_count / float(fps_src)

        candidates = self.extractor.extract_multi_person_landmarks(frame_bgr)
        active_tracks = self.tracker.update(candidates, threshold_policy=self.threshold_policy)

        track_results = []
        for tr in active_tracks:
            # 1. Roll person's independent 50-frame buffer
            tr.buffer_50 = np.roll(tr.buffer_50, -1, axis=0)
            tr.buffer_50[-1] = tr.raw_33
            tr.frames_buffered = min(50, tr.frames_buffered + 1)

            if tr.frames_buffered < 50:
                sm_res = tr.state_machine.update(
                    prob_fall=0.0,
                    person_detected=True,
                    raw_33=tr.raw_33,
                    buffer_50=tr.buffer_50,
                    is_partial_person=tr.is_partial_person,
                    edge_reason=tr.edge_reason
                )
                track_results.append({
                    "person_id": tr.person_id,
                    "bbox": tr.bbox,
                    "coco_17_px": tr.coco_17_px,
                    "raw_33": tr.raw_33,
                    "is_warmed_up": False,
                    "prob_fall": 0.0,
                    "raw_decision": "WARMING UP",
                    "current_state": sm_res["current_state"],
                    "previous_state": sm_res["previous_state"],
                    "state_transition": sm_res["state_transition"],
                    "is_partial_person": tr.is_partial_person,
                    "edge_reason": tr.edge_reason
                })
                continue

            # 2. Independent K1 forward pass per person
            base_165 = self._compute_base(tr.buffer_50)
            feat_187 = self._construct_187(base_165)
            tensor_x = torch.tensor(feat_187, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                out = self.model(tensor_x)
                prob_fall = torch.softmax(out, dim=1)[0, 1].item()

            tr.prob_fall = prob_fall
            tr.raw_decision = "FALL" if prob_fall >= self.threshold_policy else "NORMAL"

            # 3. Independent ApplicationStateMachine update per person
            sm_res = tr.state_machine.update(
                prob_fall=prob_fall,
                person_detected=True,
                raw_33=tr.raw_33,
                buffer_50=tr.buffer_50,
                is_partial_person=tr.is_partial_person,
                edge_reason=tr.edge_reason
            )

            # 4. Per-Person SMS Alert Dispatcher
            prev_st = sm_res["previous_state"]
            curr_st = sm_res["current_state"]
            if prev_st != "FALL DETECTED" and curr_st == "FALL DETECTED":
                self.alert_manager.send_fall_alert(
                    prob_fall=prob_fall,
                    frame_index=self.frame_count,
                    timestamp_sec=timestamp_sec,
                    video_name=f"{video_name} (Person #{tr.person_id})"
                )
            elif prev_st == "GETTING UP / RECOVERY" and curr_st == "RECOVERED — STANDING":
                self.alert_manager.send_recovery_alert(
                    frame_index=self.frame_count,
                    timestamp_sec=timestamp_sec,
                    video_name=f"{video_name} (Person #{tr.person_id})"
                )

            track_results.append({
                "person_id": tr.person_id,
                "bbox": tr.bbox,
                "coco_17_px": tr.coco_17_px,
                "raw_33": tr.raw_33,
                "is_warmed_up": True,
                "prob_fall": prob_fall,
                "raw_decision": tr.raw_decision,
                "current_state": sm_res["current_state"],
                "previous_state": sm_res["previous_state"],
                "state_transition": sm_res["state_transition"],
                "is_partial_person": tr.is_partial_person,
                "edge_reason": tr.edge_reason
            })

        t_end = time.perf_counter()
        lat_ms = (t_end - t_start) * 1000.0
        fps = 1000.0 / lat_ms if lat_ms > 0 else 0.0

        return {
            "num_people_detected": len(track_results),
            "track_results": track_results,
            "latency_ms": lat_ms,
            "fps": fps
        }
