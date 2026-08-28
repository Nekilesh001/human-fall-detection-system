"""
PHASE F4/F5 — STANDALONE INDEPENDENT VIDEO TESTING & PREDICTION LOGGING

Loads the final production K1 model checkpoint (checkpoints/final_k1/final_production.pth)
and official threshold (tau = 0.3650) to run real-time inference on arbitrary user-supplied
test videos (.avi, .mp4, .mov).

Pipeline:
Raw Video -> YOLO Pose Extraction -> 187-D Spatial Features -> 50-Frame Rolling Buffer
          -> Person Presence Validation Gate -> K1 1D TCN Forward Pass -> P(FALL) vs tau (0.3650)
          -> 10-State Application State Machine -> Logged CSV Output & Run Summary
"""

import os
import sys
import time
import argparse
import json
import numpy as np
import pandas as pd
import cv2
import torch

ROOT_DIR = r"d:\ONE_DATA\Fall detection"
sys.path.insert(0, ROOT_DIR)

from src.final_k1_realtime_inference import RealtimeFallDetector
from src.train_final_k1 import ModelK1_SpatialTCN

def run_independent_video_test(video_path, checkpoint_path, threshold, output_dir, display_mode):
    print("=" * 75)
    print("PHASE F4/F5 — STANDALONE INDEPENDENT VIDEO TEST (WITH STATE TRANSITION LOGGING)")
    print("=" * 75)
    
    # 1. Path Safety & Verification
    assert os.path.exists(video_path), f"Video file not found: {video_path}"
    assert os.path.exists(checkpoint_path), f"Production checkpoint missing: {checkpoint_path}"
    
    video_name = os.path.basename(video_path)
    video_stem = os.path.splitext(video_name)[0]
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"  Source Video       : {video_path}")
    print(f"  Model Checkpoint   : {checkpoint_path}")
    print(f"  Decision Threshold : tau = {threshold:.4f}")
    print(f"  Output Directory   : {output_dir}")
    print(f"  Display Mode       : {'ENABLED (OpenCV Window)' if display_mode else 'HEADLESS (No Window)'}")
    
    # 2. Initialize Real-Time Fall Detector Engine
    detector = RealtimeFallDetector(checkpoint_path=checkpoint_path, threshold_policy=threshold)
    
    # Open Video File
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Failed to open video file: {video_path}"
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0
        
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video Metadata     : {total_video_frames} frames | {fps:.2f} FPS | Duration: {total_video_frames/fps:.2f}s")
    
    # 3. Processing State Tracking
    frame_idx = 0
    latencies = []
    log_records = []
    
    raw_fall_count = 0
    stabilized_fall_count = 0
    alert_activations_count = 0
    prev_alert_state = False
    
    first_p_fall_frame = None
    first_raw_fall_frame = None
    first_alert_frame = None
    
    t_test_start = time.time()
    
    print(f"\n  Processing video frames...")
    
    while True:
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            break
            
        frame_idx += 1
        timestamp_sec = frame_idx / fps
        
        # Process Frame through Realtime Engine
        result = detector.process_frame(frame_bgr)
        
        is_warmed_up = result["is_warmed_up"]
        person_detected = result["person_detected"]
        is_partial_person = result.get("is_partial_person", False)
        edge_reason = result.get("edge_reason", "FULL_PERSON")
        prob_fall = result["prob_fall"]
        raw_decision = result["raw_decision"]
        alert_state = result["alert_state"]
        current_state = result["current_state"]
        previous_state = result["previous_state"]
        state_transition = result["state_transition"]
        latency_ms = result["latency_ms"]
        proc_fps = result["fps"]
        
        latencies.append(latency_ms)
        
        raw_pred_binary = 1 if raw_decision == "FALL" else 0
        stabilized_binary = 1 if alert_state["alert_active"] else 0
        
        if raw_pred_binary == 1:
            raw_fall_count += 1
            if first_raw_fall_frame is None:
                first_raw_fall_frame = frame_idx
                
        if prob_fall >= threshold and first_p_fall_frame is None:
            first_p_fall_frame = frame_idx
            
        if alert_state["alert_active"]:
            stabilized_fall_count += 1
            if not prev_alert_state:
                alert_activations_count += 1
                if first_alert_frame is None:
                    first_alert_frame = frame_idx
                    
        prev_alert_state = alert_state["alert_active"]
        
        win_start = max(1, frame_idx - 49)
        win_end = frame_idx
        
        log_records.append({
            "video_name": video_name,
            "frame_index": frame_idx,
            "win_start_frame": win_start,
            "win_end_frame": win_end,
            "timestamp_sec": round(timestamp_sec, 3),
            "person_detected": person_detected,
            "is_partial_person": is_partial_person,
            "edge_reason": edge_reason,
            "fall_probability": round(prob_fall, 4),
            "raw_prediction": raw_pred_binary,
            "raw_decision": raw_decision,
            "decision_threshold": threshold,
            "stabilized_prediction": stabilized_binary,
            "previous_application_state": previous_state,
            "current_application_state": current_state,
            "state_transition": state_transition,
            "consecutive_fall_windows": alert_state["consecutive_fall_count"],
            "processing_fps": round(proc_fps, 2),
            "latency_ms": round(latency_ms, 2)
        })
            
        # 4. Optional HUD Display
        if display_mode:
            disp_frame = frame_bgr.copy()
            h, w = disp_frame.shape[:2]
            
            # Color mapping for system states
            state_colors = {
                "WARMING UP": (200, 200, 200),
                "NO PERSON DETECTED": (150, 150, 150),
                "NORMAL — STANDING": (0, 255, 0),
                "NORMAL — WALKING": (0, 255, 0),
                "NORMAL — SITTING": (0, 255, 0),
                "NORMAL": (0, 255, 0),
                "FALL SUSPECTED": (0, 165, 255),
                "FALL DETECTED": (0, 0, 255),
                "FALLEN — ON FLOOR": (0, 0, 180),
                "GETTING UP / RECOVERY": (255, 165, 0),
                "RECOVERED — STANDING": (0, 200, 100)
            }
            color = state_colors.get(current_state, (255, 255, 255))
            
            # HUD Overlay Box
            cv2.rectangle(disp_frame, (10, 10), (480, 150), (0, 0, 0), -1)
            cv2.rectangle(disp_frame, (10, 10), (480, 150), (255, 255, 255), 1)
            
            status_text = f"STATE: {current_state}"
            cv2.putText(disp_frame, f"Video: {video_name}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(disp_frame, status_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(disp_frame, f"P(FALL): {prob_fall*100:5.1f}% (tau={threshold:.2f})", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(disp_frame, f"FPS: {proc_fps:5.1f} | Latency: {latency_ms:4.1f} ms", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(disp_frame, f"Frame: {frame_idx}/{total_video_frames} | Person: {'YES' if person_detected else 'NO'}", (20, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            
            # Probability Bar
            bar_w = int(prob_fall * 300)
            cv2.rectangle(disp_frame, (20, 98), (320, 105), (50, 50, 50), -1)
            bar_color = (0, 0, 255) if prob_fall >= threshold else (0, 255, 0)
            cv2.rectangle(disp_frame, (20, 98), (20 + bar_w, 105), bar_color, -1)
            
            cv2.imshow("K1 Real-Time Independent Video Test", disp_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("  [User Terminated Display]")
                break

    cap.release()
    if display_mode:
        cv2.destroyAllWindows()
        
    t_test_end = time.time()
    elapsed_sec = t_test_end - t_test_start
    
    # 5. Summarize Metrics & Performance
    eval_windows = len(log_records)
    warmup_count = min(49, frame_idx)
    
    mean_lat = float(np.mean(latencies)) if latencies else 0.0
    p95_lat  = float(np.percentile(latencies, 95)) if latencies else 0.0
    max_lat  = float(np.max(latencies)) if latencies else 0.0
    mean_fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0
    
    windows_to_alert = (first_alert_frame - first_raw_fall_frame + 1) if (first_alert_frame and first_raw_fall_frame) else None
    
    # Save CSV Log
    log_df = pd.DataFrame(log_records)
    csv_filename = f"{video_stem}_log.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    log_df.to_csv(csv_path, index=False)
    
    # Save Summary JSON
    summary_data = {
        "video_name": video_name,
        "video_path": os.path.abspath(video_path),
        "total_video_frames": frame_idx,
        "warmup_frames": warmup_count,
        "evaluated_windows": eval_windows,
        "test_execution_time_sec": round(elapsed_sec, 2),
        "mean_latency_ms": round(mean_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "max_latency_ms": round(max_lat, 2),
        "mean_fps": round(mean_fps, 2),
        "decision_threshold": threshold,
        "raw_fall_windows": raw_fall_count,
        "stabilized_fall_windows": stabilized_fall_count,
        "alert_activations_count": alert_activations_count,
        "first_p_fall_above_tau_frame": first_p_fall_frame,
        "first_raw_fall_frame": first_raw_fall_frame,
        "first_stabilized_alert_frame": first_alert_frame,
        "windows_required_for_alert": windows_to_alert,
        "csv_log_path": os.path.abspath(csv_path)
    }
    
    json_filename = f"{video_stem}_summary.json"
    json_path = os.path.join(output_dir, json_filename)
    with open(json_path, "w") as f:
        json.dump(summary_data, f, indent=2)
        
    print(f"\n" + "=" * 75)
    print("PHASE F4/F5 — INDEPENDENT VIDEO TEST SUMMARY")
    print("=" * 75)
    print(f"  Processed Frames      : {frame_idx} / {total_video_frames}")
    print(f"  Evaluated Windows     : {eval_windows} (Warmup: {warmup_count} frames)")
    print(f"  Throughput & Speed    : {mean_fps:.2f} FPS | Mean Latency: {mean_lat:.2f} ms (P95: {p95_lat:.2f} ms)")
    print(f"  Raw FALL Windows      : {raw_fall_count} / {eval_windows}")
    print(f"  Stabilized FALL Wins  : {stabilized_fall_count} / {eval_windows}")
    print(f"  Alert Activations     : {alert_activations_count} event(s)")
    if first_alert_frame:
        print(f"  First Alert Frame     : Frame {first_alert_frame} (Window delay to alert: {windows_to_alert} windows)")
    print(f"  Saved Prediction Log  : {csv_path}")
    print(f"  Saved Run Summary     : {json_path}")
    print("=" * 75)

def main():
    parser = argparse.ArgumentParser(description="Phase F4/F5: Standalone Independent Video Fall Detection Test")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input test video (.avi, .mp4, .mov)")
    parser.add_argument("--checkpoint_path", type=str, default=r"d:\ONE_DATA\Fall detection\checkpoints\final_k1\final_production.pth", help="Path to trained model checkpoint")
    parser.add_argument("--threshold", type=float, default=0.3650, help="Decision threshold tau")
    parser.add_argument("--output_dir", type=str, default=r"d:\ONE_DATA\Fall detection\R&D\ML_Baseline\results\final_k1\independent_tests", help="Output directory for test logs")
    parser.add_argument("--display", action="store_true", help="Enable OpenCV GUI HUD display")
    
    args = parser.parse_args()
    
    run_independent_video_test(
        video_path=args.video_path,
        checkpoint_path=args.checkpoint_path,
        threshold=args.threshold,
        output_dir=args.output_dir,
        display_mode=args.display
    )

if __name__ == "__main__":
    main()
