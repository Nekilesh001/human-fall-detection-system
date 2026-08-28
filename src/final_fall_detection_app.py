"""
Experiment #22: Final Production Real-Time Fall Detection Application & HUD Interface.

Supports:
- Mode 1: Recorded Video Stream
- Mode 2: Live Camera / Webcam Input
- Headless Validation Logging Mode

Displays Real-Time Overlay:
- Skeletal Joint Connection Lines
- P(FALL) Probability Bar
- Threshold Policy (0.4923)
- System Decision & Stabilized Alert Status
- Processing FPS & Latency (ms)
"""

import os
import sys
import time
import argparse
import csv
import cv2
import numpy as np

ROOT_DIR = r"d:\ONE_DATA\Fall detection"
sys.path.insert(0, ROOT_DIR)

from src.final_k1_realtime_inference import RealtimeFallDetector

# COCO 17 Keypoint Skeletal Connections
COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # Head / Ears
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),# Arms / Shoulders
    (5, 11), (6, 12), (11, 12),            # Torso
    (11, 13), (13, 15), (12, 14), (14, 16) # Legs / Feet
]

CANONICAL_33_TO_COCO_17 = {
    0: 0, 2: 1, 5: 2, 7: 3, 8: 4, 11: 5, 12: 6, 13: 7, 14: 8, 15: 9, 16: 10,
    23: 11, 24: 12, 25: 13, 26: 14, 27: 15, 28: 16
}

def draw_skeleton(frame, raw_33):
    h, w = frame.shape[:2]
    coco_kpts = {}
    for can_idx, coco_idx in CANONICAL_33_TO_COCO_17.items():
        x_norm, y_norm, conf = raw_33[can_idx]
        if conf > 0.2:
            coco_kpts[coco_idx] = (int(x_norm * w), int(y_norm * h))
            cv2.circle(frame, coco_kpts[coco_idx], 4, (0, 255, 255), -1)

    for p1, p2 in COCO_EDGES:
        if p1 in coco_kpts and p2 in coco_kpts:
            cv2.line(frame, coco_kpts[p1], coco_kpts[p2], (0, 255, 0), 2)

def draw_hud(frame, res, frame_idx):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Determine Header Color & Status Text
    alert_active = res["alert_state"]["alert_active"]
    raw_dec = res["raw_decision"]
    p_fall = res["prob_fall"]

    if alert_active:
        header_color = (0, 0, 255) # Bright Red
        header_text = f"!!! ALERT: FALL DETECTED !!! ({res['alert_state']['consecutive_fall_count']}/3)"
    elif not res["is_warmed_up"]:
        header_color = (0, 255, 255) # Yellow
        header_text = f"WARMING UP BUFFER ({res['buffer_status']})"
    elif raw_dec == "FALL":
        header_color = (0, 165, 255) # Orange
        header_text = f"WARNING: TRANSIENT FALL ({res['alert_state']['consecutive_fall_count']}/3)"
    else:
        header_color = (0, 200, 0) # Green
        header_text = "SYSTEM STATUS: NORMAL (ADL)"

    # Header Bar
    cv2.rectangle(overlay, (0, 0), (w, 50), header_color, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, header_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Telemetry Panel (Bottom Left)
    cv2.rectangle(frame, (10, h - 140), (340, h - 10), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, h - 140), (340, h - 10), (255, 255, 255), 1)

    cv2.putText(frame, f"Frame: {frame_idx} | FPS: {res['fps']:.1f}", (20, h - 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Latency: {res['latency_ms']:.1f} ms", (20, h - 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"P(FALL): {p_fall*100:5.1f}% (Tau={res['threshold']:.4f})", (20, h - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Alert State: {res['alert_state']['status']}", (20, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Probability Bar
    bar_width = int(p_fall * 300)
    bar_color = (0, 0, 255) if p_fall >= res['threshold'] else (0, 255, 0)
    cv2.rectangle(frame, (20, h - 35), (20 + bar_width, h - 20), bar_color, -1)
    cv2.rectangle(frame, (20, h - 35), (320, h - 20), (255, 255, 255), 1)

def run_application(mode="video", video_path=None, save_logs=True, headless=False):
    print("=" * 70)
    print("STARTING REAL-TIME FALL DETECTION APPLICATION (EXPERIMENT #22)")
    print("=" * 70)

    detector = RealtimeFallDetector(threshold_policy=0.4923, consecutive_fall_required=3)

    if mode == "webcam":
        src = 0
        print("Input Mode: LIVE CAMERA / WEBCAM")
    else:
        if video_path is None:
            video_path = os.path.join(ROOT_DIR, "Le2i", "data", "Coffee_room_01", "Coffee_room_01", "Videos", "video (1).avi")
        src = video_path
        print(f"Input Mode: RECORDED VIDEO STREAM ({src})")

    cap = cv2.VideoCapture(src)
    assert cap.isOpened(), f"Failed to open video source: {src}"

    res_dir = os.path.join(ROOT_DIR, "R&D", "ML_Baseline", "results")
    os.makedirs(res_dir, exist_ok=True)
    log_csv_path = os.path.join(res_dir, "final_application_test_logs.csv")

    log_rows = []
    frame_idx = 0

    print(f"Processing Video Stream (Headless={headless})...\n")

    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            break

        frame_idx += 1
        res = detector.process_frame(frame_bgr)

        # Log entry
        log_rows.append({
            "timestamp": time.time(),
            "frame_index": frame_idx,
            "is_warmed_up": res["is_warmed_up"],
            "prob_fall": res["prob_fall"],
            "threshold": res["threshold"],
            "raw_decision": res["raw_decision"],
            "alert_status": res["alert_state"]["status"],
            "alert_active": res["alert_state"]["alert_active"],
            "consecutive_fall_count": res["alert_state"]["consecutive_fall_count"],
            "latency_ms": res["latency_ms"],
            "fps": res["fps"]
        })

        if not headless:
            draw_skeleton(frame_bgr, res["raw_33"])
            draw_hud(frame_bgr, res, frame_idx)
            cv2.imshow("Production Real-Time Fall Detection System (Model K1)", frame_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    if not headless:
        cv2.destroyAllWindows()

    if save_logs:
        import pandas as pd
        pd.DataFrame(log_rows).to_csv(log_csv_path, index=False)
        print(f"\nApplication Log Saved: {log_csv_path} ({len(log_rows)} frames logged)")

    # Print Summary Statistics
    warmed_rows = [r for r in log_rows if r["is_warmed_up"]]
    n_warmed = len(warmed_rows)
    n_raw_falls = sum(1 for r in warmed_rows if r["raw_decision"] == "FALL")
    n_alerts = sum(1 for r in warmed_rows if r["alert_active"])
    avg_fps = np.mean([r["fps"] for r in log_rows])
    avg_latency = np.mean([r["latency_ms"] for r in log_rows])

    print("\n" + "=" * 70)
    print("APPLICATION TEST RUN SUMMARY")
    print("=" * 70)
    print(f"  Total Video Frames Processed : {frame_idx}")
    print(f"  Buffered Warmed-Up Windows   : {n_warmed}")
    print(f"  Raw Model FALL Window Count  : {n_raw_falls}")
    print(f"  Stabilized ALERT Window Count: {n_alerts}")
    print(f"  Mean Application Processing FPS: {avg_fps:.1f} FPS")
    print(f"  Mean Application Latency     : {avg_latency:.2f} ms")
    print("=" * 70)

    return {
        "total_frames": frame_idx,
        "warmed_windows": n_warmed,
        "raw_fall_windows": n_raw_falls,
        "stabilized_alert_windows": n_alerts,
        "mean_fps": float(avg_fps),
        "mean_latency_ms": float(avg_latency)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["video", "webcam"], default="video")
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    run_application(mode=args.mode, video_path=args.video_path, headless=args.headless)
