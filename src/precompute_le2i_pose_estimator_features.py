"""
Experiment H1: Precomputes 165-D Pose + Velocity Feature Tensors for H1 (MediaPipe), H2 (YOLO Pose), and H3 (RTMPose).

Input Shape per Window: (50, 165) float32
1,396 Supervised Windows across 127 Le2i Videos (331 FALL, 1,065 NORMAL).

Outputs:
- processed_data/Le2i_baseline/pose_estimator_features/mediapipe/
- processed_data/Le2i_baseline/pose_estimator_features/yolo_pose/
- processed_data/Le2i_baseline/pose_estimator_features/rtmpose/
- R&D/ML_Baseline/results/pose_estimator_precomputation_summary.json
"""

import os
import sys
import time
import json
import glob
import cv2
import numpy as np
import pandas as pd
import torch

torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
if os.path.exists(torch_lib):
    os.add_dll_directory(torch_lib)

ROOT_DIR = r"d:\ONE_DATA\Fall detection"
sys.path.insert(0, ROOT_DIR)

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

# ----------------------------------------------------------------------
# MediaPipe Extractor
# ----------------------------------------------------------------------
class MediaPipeExtractor:
    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        model_path = os.path.join(ROOT_DIR, "models", "pose_landmarker_full.task")
        assert os.path.exists(model_path), f"MediaPipe model asset missing: {model_path}"

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5
        )
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self.mp_image_cls = mp.Image
        self.image_format_cls = mp.ImageFormat

    def extract_landmarks(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = self.mp_image_cls(image_format=self.image_format_cls.SRGB, data=frame_rgb)
        res = self.landmarker.detect(mp_img)

        raw_33 = np.zeros((33, 3), dtype=np.float32)
        if res.pose_landmarks and len(res.pose_landmarks) > 0:
            lms = res.pose_landmarks[0]
            for i, lm in enumerate(lms):
                vis = getattr(lm, 'visibility', 1.0)
                raw_33[i] = [lm.x, lm.y, vis]
            return raw_33, True
        return raw_33, False

# ----------------------------------------------------------------------
# YOLO Pose Extractor
# ----------------------------------------------------------------------
class YOLOPoseExtractor:
    def __init__(self):
        try:
            from ultralytics import YOLO
            model_path = os.path.join(ROOT_DIR, "models", "yolov8n-pose.pt")
            if not os.path.exists(model_path):
                self.model = YOLO("yolov8n-pose.pt")
            else:
                self.model = YOLO(model_path)
            self.has_yolo = True
        except Exception:
            self.has_yolo = False

    def extract_landmarks(self, frame_bgr):
        raw_33 = np.zeros((33, 3), dtype=np.float32)
        if not self.has_yolo:
            return raw_33, False

        h_img, w_img = frame_bgr.shape[:2]
        results = self.model.predict(frame_bgr, verbose=False, conf=0.25)
        if len(results) > 0 and len(results[0].keypoints) > 0 and len(results[0].keypoints.data) > 0:
            kpts_data = results[0].keypoints.data[0].cpu().numpy() # (17, 3) or (17, 2)
            confs = results[0].keypoints.conf[0].cpu().numpy() if results[0].keypoints.conf is not None else np.ones(17)

            for coco_idx, can_idx in COCO_TO_CANONICAL_33.items():
                if coco_idx < len(kpts_data):
                    x_px, y_px = kpts_data[coco_idx][:2]
                    conf = float(confs[coco_idx]) if coco_idx < len(confs) else 0.5
                    raw_33[can_idx] = [x_px / float(w_img), y_px / float(h_img), conf]
            return raw_33, True
        return raw_33, False

# ----------------------------------------------------------------------
# RTMPose Extractor
# ----------------------------------------------------------------------
class RTMPoseExtractor:
    def __init__(self):
        try:
            from rtmlib import Body
            self.body = Body(mode='balanced', to_openpose=False, device='cuda')
            self.has_rtm = True
        except Exception as e:
            self.has_rtm = False

    def extract_landmarks(self, frame_bgr):
        raw_33 = np.zeros((33, 3), dtype=np.float32)
        if not self.has_rtm:
            return raw_33, False

        h_img, w_img = frame_bgr.shape[:2]
        keypoints, scores = self.body(frame_bgr)
        if len(keypoints) > 0 and len(scores) > 0:
            kpts = keypoints[0] # (17, 2)
            scs = scores[0]     # (17,)
            for coco_idx, can_idx in COCO_TO_CANONICAL_33.items():
                if coco_idx < len(kpts):
                    x_px, y_px = kpts[coco_idx]
                    conf = float(scs[coco_idx])
                    raw_33[can_idx] = [x_px / float(w_img), y_px / float(h_img), conf]
            return raw_33, True
        return raw_33, False

# ----------------------------------------------------------------------
# Torso Normalization & Velocity Derivation (Canonical 165-D Vector)
# ----------------------------------------------------------------------
def compute_165d_pose_features(raw_window_33):
    # raw_window_33: (50, 33, 3)
    norm_window_99 = np.zeros((50, 99), dtype=np.float32)

    for t in range(50):
        frame_raw = raw_window_33[t] # (33, 3)
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

    # Velocity Derivation (66-D)
    pos_coords_50 = norm_window_99.reshape(50, 33, 3)[:, :, :2] # (50, 33, 2)
    vel_50 = np.zeros((50, 33, 2), dtype=np.float32)
    vel_50[1:] = pos_coords_50[1:] - pos_coords_50[:-1]
    vel_66 = vel_50.reshape(50, 66)

    # 165-D Feature Vector
    features_165 = np.hstack([norm_window_99, vel_66]) # (50, 165)
    return features_165

def precompute_le2i_pose_estimator_features():
    print("=" * 70)
    print("EXPERIMENT H1: PRECOMPUTING LE2I POSE ESTIMATOR FEATURES")
    print("=" * 70)

    pose_manifest_path = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "processed_pose_features_manifest.csv")
    assert os.path.exists(pose_manifest_path), f"Manifest missing: {pose_manifest_path}"
    df_manifest = pd.read_csv(pose_manifest_path)
    df_manifest = df_manifest.sort_values("window_id").reset_index(drop=True)

    print(f"Total Supervised Windows: {len(df_manifest):,}")

    estimators = {
        "mediapipe": {"name": "H1: MediaPipe Pose", "dir": os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "pose_estimator_features", "mediapipe"), "cls": MediaPipeExtractor},
        "yolo_pose": {"name": "H2: YOLO Pose",     "dir": os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "pose_estimator_features", "yolo_pose"), "cls": YOLOPoseExtractor},
        "rtmpose":   {"name": "H3: RTMPose",       "dir": os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "pose_estimator_features", "rtmpose"),   "cls": RTMPoseExtractor}
    }

    # 1. SMOKE TEST ON 3 SAMPLE WINDOWS
    print("\n1. RUNNING SMOKE TEST ON 3 SAMPLE WINDOWS...")
    sample_rows = df_manifest.iloc[:3]

    for est_key, est_meta in estimators.items():
        extractor = est_meta["cls"]()
        print(f"\n   Testing {est_meta['name']} Extractor...")
        
        for idx, row in sample_rows.iterrows():
            v_rel = str(row["raw_video_path"]).replace("/", os.sep)
            v_abs = os.path.join(ROOT_DIR, v_rel)
            cap = cv2.VideoCapture(v_abs)
            start_f = int(row["win_start_frame"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

            raw_33_win = np.zeros((50, 33, 3), dtype=np.float32)
            det_count = 0

            for f_idx in range(50):
                ret, frame = cap.read()
                if not ret:
                    break
                raw_33, is_det = extractor.extract_landmarks(frame)
                raw_33_win[f_idx] = raw_33
                if is_det:
                    det_count += 1
            cap.release()

            feat_165 = compute_165d_pose_features(raw_33_win)
            assert feat_165.shape == (50, 165), f"Smoke test shape error: {feat_165.shape}"
            assert not np.isnan(feat_165).any(), "Smoke test NaN error!"
            assert not np.isinf(feat_165).any(), "Smoke test Inf error!"
            print(f"     - Window {row['window_id']}: Extracted (50, 165) tensor | Det Frames = {det_count}/50 [PASS]")

    print("\n   SMOKE TEST PASSED 100% — PROCEEDING TO FULL PRECOMPUTATION ✅")

    # 2. FULL PRECOMPUTATION ACROSS ALL 1,396 WINDOWS
    summary_data = {}

    for est_key, est_meta in estimators.items():
        print("\n" + "=" * 70)
        print(f"PRECOMPUTING FULL FEATURE DATASET FOR {est_meta['name'].upper()}")
        print("=" * 70)

        os.makedirs(est_meta["dir"], exist_ok=True)
        extractor = est_meta["cls"]()
        
        t0 = time.perf_counter()
        total_det_frames = 0
        total_frames = 0
        undetected_windows = 0
        partially_detected_windows = 0
        fully_detected_windows = 0

        manifest_updated_rows = []

        for idx, row in df_manifest.iterrows():
            wid = row["window_id"]
            out_file = os.path.join(est_meta["dir"], f"{wid}.npz")
            rel_feat_path = os.path.relpath(out_file, ROOT_DIR).replace("\\", "/")

            is_valid_cache = False
            if os.path.exists(out_file):
                with np.load(out_file) as d:
                    feat_165 = d["features"]
                vis_per_frame = feat_165[:, 2:99:3]
                det_f = int(np.sum((vis_per_frame > 0).any(axis=1)))
                if det_f > 0 or est_key == "mediapipe":
                    is_valid_cache = True
                    total_frames += 50
                    total_det_frames += det_f

            if is_valid_cache:
                pass
            else:
                v_rel = str(row["raw_video_path"]).replace("/", os.sep)
                v_abs = os.path.join(ROOT_DIR, v_rel)
                cap = cv2.VideoCapture(v_abs)
                start_f = int(row["win_start_frame"])
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

                raw_33_win = np.zeros((50, 33, 3), dtype=np.float32)
                det_f = 0

                for f_idx in range(50):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    total_frames += 1
                    raw_33, is_det = extractor.extract_landmarks(frame)
                    raw_33_win[f_idx] = raw_33
                    if is_det:
                        det_f += 1
                cap.release()

                total_det_frames += det_f
                feat_165 = compute_165d_pose_features(raw_33_win)
                np.savez_compressed(out_file, features=feat_165)

            if det_f == 50:
                fully_detected_windows += 1
            elif det_f == 0:
                undetected_windows += 1
            else:
                partially_detected_windows += 1

            if idx % 200 == 0 or idx == len(df_manifest) - 1:
                print(f"   [{est_key}] Processed {idx+1:4d} / {len(df_manifest):4d} windows...")

        t_elapsed = time.perf_counter() - t0
        det_rate = (total_det_frames / total_frames) * 100.0 if total_frames > 0 else 0.0

        print(f"\n   [{est_meta['name']}] Summary:")
        print(f"     - Extraction Time            : {t_elapsed:.2f} seconds ({t_elapsed/60.0:.2f} minutes)")
        print(f"     - Total Frames               : {total_frames:,}")
        print(f"     - Detected Frames            : {total_det_frames:,} ({det_rate:.2f}%)")
        print(f"     - Fully Detected Windows     : {fully_detected_windows:,} ({(fully_detected_windows/1396)*100:.1f}%)")
        print(f"     - Partially Detected Windows : {partially_detected_windows:,} ({(partially_detected_windows/1396)*100:.1f}%)")
        print(f"     - Completely Undetected Wins : {undetected_windows:,} ({(undetected_windows/1396)*100:.1f}%)")

        summary_data[est_key] = {
            "name": est_meta["name"],
            "extraction_time_s": t_elapsed,
            "total_frames": total_frames,
            "detected_frames": total_det_frames,
            "detection_rate_pct": det_rate,
            "fully_detected_windows": fully_detected_windows,
            "partially_detected_windows": partially_detected_windows,
            "undetected_windows": undetected_windows
        }

    # Save Precomputation Summary JSON
    res_dir = os.path.join(ROOT_DIR, "R&D", "ML_Baseline", "results")
    os.makedirs(res_dir, exist_ok=True)
    summary_json_path = os.path.join(res_dir, "pose_estimator_precomputation_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print("\n" + "=" * 70)
    print("PHASE H1 PRECOMPUTATION COMPLETE — SUMMARY SAVED")
    print("=" * 70)

if __name__ == "__main__":
    precompute_le2i_pose_estimator_features()
