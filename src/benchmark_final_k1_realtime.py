"""
Experiment #21: End-to-End Real-Time Latency & Throughput Benchmark for Frozen Model K1.

Benchmarked Components:
1. YOLO Pose Keypoint Extraction Latency & FPS
2. 187-D Spatial Feature Derivation Latency
3. 50-Frame Temporal Buffering Overhead
4. K1 1D Residual TCN Model Inference Latency & FPS
5. Complete End-to-End Latency & System Throughput (FPS)
6. GPU & VRAM Memory Footprint Audit
7. Prediction Stability Audit over Video Stream
"""

import os
import sys
import time
import json
import glob
import cv2
import psutil
import numpy as np
import pandas as pd
import torch

ROOT_DIR = r"d:\ONE_DATA\Fall detection"
sys.path.insert(0, ROOT_DIR)

from src.infer_final_k1 import YOLOPoseExtractor, compute_165d_base_features, construct_187d_window_features, ModelK1InferenceEngine
from src.train_le2i_yolo_k1_spatial import ModelK1_SpatialTCN

def compute_percentiles(arr_ms):
    return {
        "mean_ms": float(np.mean(arr_ms)),
        "median_ms": float(np.median(arr_ms)),
        "p95_ms": float(np.percentile(arr_ms, 95)),
        "p99_ms": float(np.percentile(arr_ms, 99)),
        "min_ms": float(np.min(arr_ms)),
        "max_ms": float(np.max(arr_ms)),
        "std_ms": float(np.std(arr_ms))
    }

def benchmark_final_k1_realtime():
    print("=" * 70)
    print("EXPERIMENT #21: FROZEN K1 REAL-TIME INFERENCE BENCHMARK")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU Model: {gpu_name}")
        print(f"PyTorch CUDA Version: {torch.version.cuda}")
    else:
        gpu_name = "CPU"

    # CPU & Initial GPU Memory
    process = psutil.Process(os.getpid())
    ram_before_mb = process.memory_info().rss / (1024 * 1024)
    vram_before_mb = torch.cuda.memory_allocated(0) / (1024 * 1024) if torch.cuda.is_available() else 0.0

    print(f"Initial Memory Footprint: CPU RAM = {ram_before_mb:.2f} MB | VRAM = {vram_before_mb:.2f} MB")

    # Load Pipeline Modules
    print("\n1. INITIALIZING PIPELINE MODULES & FROZEN MODEL...")
    extractor = YOLOPoseExtractor()
    engine = ModelK1InferenceEngine(threshold_policy=0.4923)

    vram_after_init_mb = torch.cuda.memory_allocated(0) / (1024 * 1024) if torch.cuda.is_available() else 0.0
    print(f"VRAM After Model Load: {vram_after_init_mb:.2f} MB")

    # Load sample video frame for realistic benchmarking
    sample_video_path = os.path.join(ROOT_DIR, "Le2i", "data", "Coffee_room_01", "Coffee_room_01", "Videos", "video (1).avi")
    assert os.path.exists(sample_video_path), f"Sample video missing: {sample_video_path}"

    cap = cv2.VideoCapture(sample_video_path)
    ret, sample_frame = cap.read()
    cap.release()
    assert ret and sample_frame is not None, "Failed to read sample frame"

    # ------------------------------------------------------------------
    # Warmup Phase (50 iterations)
    # ------------------------------------------------------------------
    print("\n2. EXECUTING WARMUP PHASE (50 Iterations)...")
    dummy_raw_window = np.zeros((50, 33, 3), dtype=np.float32)
    for _ in range(50):
        raw_33, _ = extractor.extract_landmarks(sample_frame)
        dummy_raw_window = np.roll(dummy_raw_window, -1, axis=0)
        dummy_raw_window[-1] = raw_33
        base_165 = compute_165d_base_features(dummy_raw_window)
        feat_187 = construct_187d_window_features(base_165)
        _ = engine.predict_window(feat_187)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    vram_after_warmup_mb = torch.cuda.memory_allocated(0) / (1024 * 1024) if torch.cuda.is_available() else 0.0
    vram_peak_mb = torch.cuda.max_memory_allocated(0) / (1024 * 1024) if torch.cuda.is_available() else 0.0

    print(f"Memory Footprint After Warmup: VRAM Allocated = {vram_after_warmup_mb:.2f} MB | VRAM Peak = {vram_peak_mb:.2f} MB")

    # ------------------------------------------------------------------
    # Benchmark Phase (100 Iterations)
    # ------------------------------------------------------------------
    N_ITERS = 100
    print(f"\n3. EXECUTING BENCHMARK PHASE ({N_ITERS} Iterations)...")

    t_yolo_list = []
    t_feat_list = []
    t_buff_list = []
    t_k1_list   = []
    t_e2e_list  = []

    rolling_window_33 = np.zeros((50, 33, 3), dtype=np.float32)

    for i in range(N_ITERS):
        t_start_e2e = time.perf_counter()

        # Step A: YOLO Pose Landmark Extraction
        t0 = time.perf_counter()
        raw_33, _ = extractor.extract_landmarks(sample_frame)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        t_yolo = (t1 - t0) * 1000.0

        # Step B: Temporal Buffering
        t0 = time.perf_counter()
        rolling_window_33 = np.roll(rolling_window_33, -1, axis=0)
        rolling_window_33[-1] = raw_33
        t1 = time.perf_counter()
        t_buff = (t1 - t0) * 1000.0

        # Step C: 187-D Spatial Feature Derivation
        t0 = time.perf_counter()
        base_165 = compute_165d_base_features(rolling_window_33)
        feat_187 = construct_187d_window_features(base_165)
        t1 = time.perf_counter()
        t_feat = (t1 - t0) * 1000.0

        # Step D: K1 TCN Model Inference
        t0 = time.perf_counter()
        _ = engine.predict_window(feat_187)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        t_k1 = (t1 - t0) * 1000.0

        t_end_e2e = time.perf_counter()
        t_e2e = (t_end_e2e - t_start_e2e) * 1000.0

        t_yolo_list.append(t_yolo)
        t_feat_list.append(t_feat)
        t_buff_list.append(t_buff)
        t_k1_list.append(t_k1)
        t_e2e_list.append(t_e2e)

    # Compute Statistics
    stats_yolo = compute_percentiles(t_yolo_list)
    stats_feat = compute_percentiles(t_feat_list)
    stats_buff = compute_percentiles(t_buff_list)
    stats_k1   = compute_percentiles(t_k1_list)
    stats_e2e  = compute_percentiles(t_e2e_list)

    fps_yolo     = 1000.0 / stats_yolo["mean_ms"]
    fps_k1_model = 1000.0 / stats_k1["mean_ms"]
    fps_e2e      = 1000.0 / stats_e2e["mean_ms"]

    realtime_capable = fps_e2e >= 25.0
    status_str = "REAL-TIME CAPABLE ✅" if realtime_capable else "NOT REAL-TIME CAPABLE ❌"

    print("\n" + "=" * 70)
    print("LATENCY & THROUGHPUT BENCHMARK RESULTS")
    print("=" * 70)
    print(f"  YOLO Pose Inference Latency   : Mean={stats_yolo['mean_ms']:.2f} ms | P95={stats_yolo['p95_ms']:.2f} ms | Throughput={fps_yolo:.1f} FPS")
    print(f"  187-D Feature Extraction Time : Mean={stats_feat['mean_ms']:.2f} ms | P95={stats_feat['p95_ms']:.2f} ms")
    print(f"  Temporal Buffering Overhead   : Mean={stats_buff['mean_ms']:.3f} ms | P95={stats_buff['p95_ms']:.3f} ms")
    print(f"  Model K1 TCN Inference Time   : Mean={stats_k1['mean_ms']:.2f} ms | P95={stats_k1['p95_ms']:.2f} ms | Throughput={fps_k1_model:.1f} FPS")
    print("-" * 70)
    print(f"  TOTAL END-TO-END LATENCY      : Mean={stats_e2e['mean_ms']:.2f} ms | P95={stats_e2e['p95_ms']:.2f} ms")
    print(f"  TOTAL END-TO-END THROUGHPUT   : {fps_e2e:.1f} FPS")
    print(f"  REAL-TIME CAPABILITY (25 FPS) : {status_str}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 4: Stream Prediction Stability Audit
    # ------------------------------------------------------------------
    print("\n4. EXECUTING PREDICTION STABILITY AUDIT OVER VIDEO STREAM...")
    cap = cv2.VideoCapture(sample_video_path)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = float(cap.get(cv2.CAP_PROP_FPS))

    stream_window_33 = np.zeros((50, 33, 3), dtype=np.float32)
    stream_probs = []
    stream_decisions = []

    frame_count = 0
    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            break
        raw_33, _ = extractor.extract_landmarks(frame_bgr)
        stream_window_33 = np.roll(stream_window_33, -1, axis=0)
        stream_window_33[-1] = raw_33
        frame_count += 1

        if frame_count >= 50: # Valid 50-frame buffer
            base_165 = compute_165d_base_features(stream_window_33)
            feat_187 = construct_187d_window_features(base_165)
            res = engine.predict_window(feat_187)
            stream_probs.append(res["prob_fall"])
            stream_decisions.append(res["prediction"])

    cap.release()

    n_normal_dec = stream_decisions.count("NORMAL")
    n_fall_dec   = stream_decisions.count("FALL")

    print(f"  Processed Stream Video: {sample_video_path}")
    print(f"  Total Video Frames: {total_video_frames} ({total_video_frames/video_fps:.2f} seconds)")
    print(f"  Evaluated Window Count: {len(stream_decisions)}")
    print(f"  Inference Output Breakdown: NORMAL={n_normal_dec} windows | FALL={n_fall_dec} windows")

    ram_after_mb = process.memory_info().rss / (1024 * 1024)

    # Compile Benchmark Results
    benchmark_data = {
        "device": str(device),
        "gpu_name": gpu_name,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "None",
        "iterations_evaluated": N_ITERS,
        "realtime_capable": realtime_capable,
        "target_fps": 25.0,
        "end_to_end_fps": round(fps_e2e, 2),
        "yolo_pose_fps": round(fps_yolo, 2),
        "k1_model_fps": round(fps_k1_model, 2),
        "memory_footprint_mb": {
            "ram_used_mb": round(ram_after_mb, 2),
            "vram_allocated_mb": round(vram_after_warmup_mb, 2),
            "vram_peak_mb": round(vram_peak_mb, 2)
        },
        "latency_stats_ms": {
            "yolo_pose": stats_yolo,
            "spatial_feature_extraction": stats_feat,
            "temporal_buffering": stats_buff,
            "k1_tcn_model": stats_k1,
            "end_to_end": stats_e2e
        },
        "stream_stability_test": {
            "video_path": sample_video_path,
            "total_frames": total_video_frames,
            "evaluated_windows": len(stream_decisions),
            "threshold_used": 0.4923,
            "normal_windows": n_normal_dec,
            "fall_windows": n_fall_dec
        }
    }

    res_dir = os.path.join(ROOT_DIR, "R&D", "ML_Baseline", "results")
    os.makedirs(res_dir, exist_ok=True)

    json_path = os.path.join(res_dir, "final_k1_realtime_benchmark.json")
    with open(json_path, "w") as f:
        json.dump(benchmark_data, f, indent=2)

    csv_rows = [{
        "gpu_name": gpu_name,
        "target_fps": 25.0,
        "end_to_end_fps": round(fps_e2e, 2),
        "e2e_latency_mean_ms": stats_e2e["mean_ms"],
        "e2e_latency_p95_ms": stats_e2e["p95_ms"],
        "yolo_latency_mean_ms": stats_yolo["mean_ms"],
        "k1_tcn_latency_mean_ms": stats_k1["mean_ms"],
        "vram_peak_mb": vram_peak_mb,
        "realtime_capable": realtime_capable
    }]

    csv_path = os.path.join(res_dir, "final_k1_realtime_benchmark.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    print(f"\nResults Saved: {json_path} & {csv_path}")

if __name__ == "__main__":
    benchmark_final_k1_realtime()
