"""
Final SOTA Production Inference Pipeline for Human Fall Detection (Model K1 Champion).

Pipeline Flow:
  Video (.avi / .mp4) or Precomputed Pose NPZ
         │
         ▼
  YOLO Pose Extractor (17 COCO Keypoints -> 33 Canonical Landmarks)
         │
         ▼
  187-D Spatial Feature Construction (Coordinates + Visibilities + Velocities + 22 Body Angles)
         │
         ▼
  50-Frame Temporal Windowing (2.0s Receptive Field) -> Shape (1, 50, 187)
         │
         ▼
  ModelK1_SpatialTCN (1D Residual TCN, 89,250 params) -> P(FALL)
         │
         ▼
  Leakage-Free Validated Operating Threshold Policy (tau = 0.4923) -> NORMAL / FALL
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cv2

ROOT_DIR = r"d:\ONE_DATA\Fall detection"
sys.path.insert(0, ROOT_DIR)

from src.train_le2i_yolo_k1_spatial import ModelK1_SpatialTCN
from src.precompute_yolo_k1_spatial_features import derive_22_spatial_features

COCO_TO_CANONICAL_33 = {
    0: 0,   # Nose
    1: 2,   # L_Eye
    2: 5,   # R_Eye
    3: 7,   # L_Ear
    4: 8,   # R_Ear
    5: 11,  # L_Shoulder
    6: 12,  # R_Shoulder
    7: 13,  # L_Elbow
    8: 14,  # R_Elbow
    9: 15,  # L_Wrist
    10: 16, # R_Wrist
    11: 23, # L_Hip
    12: 24, # R_Hip
    13: 25, # L_Knee
    14: 26, # R_Knee
    15: 27, # L_Ankle
    16: 28  # R_Ankle
}

class YOLOPoseExtractor:
    def __init__(self):
        from ultralytics import YOLO
        model_path = os.path.join(ROOT_DIR, "models", "yolov8n-pose.pt")
        if not os.path.exists(model_path):
            self.model = YOLO("yolov8n-pose.pt")
        else:
            self.model = YOLO(model_path)

    def extract_landmarks(self, frame_bgr):
        raw_33 = np.zeros((33, 3), dtype=np.float32)
        h_img, w_img = frame_bgr.shape[:2]
        results = self.model.predict(frame_bgr, verbose=False, conf=0.25)
        
        if len(results) > 0 and len(results[0].keypoints) > 0 and len(results[0].keypoints.data) > 0:
            kpts_data = results[0].keypoints.data[0].cpu().numpy() # (17, 3) or (17, 2)
            confs = results[0].keypoints.conf[0].cpu().numpy() if (results[0].keypoints.conf is not None and len(results[0].keypoints.conf) > 0) else np.ones(17)

            coco_17_px = np.zeros((17, 3), dtype=np.float32)
            for coco_idx in range(min(17, len(kpts_data))):
                x_px, y_px = kpts_data[coco_idx][:2]
                conf = float(confs[coco_idx]) if coco_idx < len(confs) else 0.5
                coco_17_px[coco_idx] = [x_px, y_px, conf]
                
                can_idx = COCO_TO_CANONICAL_33.get(coco_idx, None)
                if can_idx is not None:
                    raw_33[can_idx] = [x_px / float(w_img), y_px / float(h_img), conf]

            bbox = None
            is_partial_person = False
            edge_reason = "FULL_PERSON"
            
            if len(results[0].boxes) > 0 and len(results[0].boxes.xyxy) > 0:
                bbox = results[0].boxes.xyxy[0].cpu().numpy() # [x1, y1, x2, y2]
                x1, y1, x2, y2 = bbox
                
                edge_tx = 0.02 * float(w_img)
                edge_ty = 0.02 * float(h_img)
                
                touch_left = (x1 <= edge_tx)
                touch_right = (x2 >= float(w_img) - edge_tx)
                touch_top = (y1 <= edge_ty)
                touch_bottom = (y2 >= float(h_img) - edge_ty)
                
                valid_kpt_count = np.sum(coco_17_px[:, 2] > 0.3)
                ankles_missing = (coco_17_px[15, 2] <= 0.3) and (coco_17_px[16, 2] <= 0.3)
                head_missing = (coco_17_px[0, 2] <= 0.3)
                
                if touch_bottom and (ankles_missing or valid_kpt_count < 12):
                    is_partial_person = True
                    edge_reason = "PARTIAL_PERSON_EDGE_BOTTOM"
                elif touch_top and (head_missing or valid_kpt_count < 12):
                    is_partial_person = True
                    edge_reason = "PARTIAL_PERSON_EDGE_TOP"
                elif (touch_left or touch_right) and valid_kpt_count < 12:
                    is_partial_person = True
                    edge_reason = "PARTIAL_PERSON_EDGE_SIDE"

            return raw_33, True, bbox, coco_17_px, is_partial_person, edge_reason

        return raw_33, False, None, None, False, "NO_PERSON"

def compute_165d_base_features(raw_window_33):
    # raw_window_33: (50, 33, 3)
    T = raw_window_33.shape[0]
    norm_window_99 = np.zeros((T, 99), dtype=np.float32)

    for t in range(T):
        frame_raw = raw_window_33[t]
        vis = frame_raw[:, 2]

        if np.sum(vis > 0) == 0:
            norm_window_99[t] = 0.0
            continue

        hip_center = 0.5 * (frame_raw[23, :2] + frame_raw[24, :2])
        sh_center  = 0.5 * (frame_raw[11, :2] + frame_raw[12, :2])
        torso_len  = np.linalg.norm(sh_center - hip_center)

        if torso_len < 1e-5:
            torso_len = 1.0

        norm_coords = (frame_raw[:, :2] - hip_center) / torso_len
        frame_99 = np.zeros((33, 3), dtype=np.float32)
        frame_99[:, :2] = norm_coords
        frame_99[:, 2]  = vis
        norm_window_99[t] = frame_99.flatten()

    pos_coords_50 = norm_window_99.reshape(T, 33, 3)[:, :, :2]
    vel_50 = np.zeros((T, 33, 2), dtype=np.float32)
    vel_50[1:] = pos_coords_50[1:] - pos_coords_50[:-1]
    vel_66 = vel_50.reshape(T, 66)

    return np.hstack([norm_window_99, vel_66]).astype(np.float32) # (50, 165)

def construct_187d_window_features(base_165d):
    # base_165d: (50, 165)
    spatial_22 = derive_22_spatial_features(base_165d) # (50, 22)
    return np.hstack([base_165d, spatial_22]).astype(np.float32) # (50, 187)

class ModelK1InferenceEngine:
    def __init__(self, checkpoint_path=None, threshold_policy=0.4923):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold_policy = threshold_policy
        self.model = ModelK1_SpatialTCN().to(self.device)
        
        if checkpoint_path is None:
            checkpoint_path = os.path.join(ROOT_DIR, "checkpoints", "le2i_yolo_k1", "fold_1_best.pth")
        
        assert os.path.exists(checkpoint_path), f"Checkpoint missing: {checkpoint_path}"
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()

    def predict_window(self, feat_187d):
        # feat_187d: (50, 187)
        tensor_x = torch.tensor(feat_187d, dtype=torch.float32).unsqueeze(0).to(self.device) # (1, 50, 187)
        with torch.no_grad():
            out = self.model(tensor_x)
            prob_fall = torch.softmax(out, dim=1)[0, 1].item()
        
        prediction = "FALL" if prob_fall >= self.threshold_policy else "NORMAL"
        return {
            "prediction": prediction,
            "prob_fall": prob_fall,
            "prob_normal": 1.0 - prob_fall,
            "threshold_used": self.threshold_policy
        }

def run_inference_demo():
    print("=" * 70)
    print("FINAL SOTA PRODUCTION INFERENCE DEMO (MODEL K1 CHAMPION)")
    print("=" * 70)

    engine = ModelK1InferenceEngine()
    print(f"Loaded ModelK1_SpatialTCN on {engine.device}")
    print(f"Validated Decision Policy Threshold: {engine.threshold_policy}")

    # Load 1 sample window from precomputed 187-D dataset
    sample_npz = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "pose_estimator_features", "yolo_pose_k1", "Le2i_Coffee_room_01_video (1)_w000.npz")
    if os.path.exists(sample_npz):
        with np.load(sample_npz) as d:
            feat_sample = d["features"] # (50, 187)
        res = engine.predict_window(feat_sample)
        print(f"\nInference Result on Sample Window 'w000':")
        print(f"  - Predicted Class : {res['prediction']}")
        print(f"  - Fall Probability: {res['prob_fall']*100:.2f}%")
        print(f"  - Decision Policy : P(FALL) >= {res['threshold_used']:.4f}")
    else:
        print(f"Sample NPZ not found: {sample_npz}")

if __name__ == "__main__":
    run_inference_demo()
