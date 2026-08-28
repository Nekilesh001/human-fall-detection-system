# Research Report: Frozen Model K1 Real-Time Performance & Latency Benchmark (Experiment #21)

> [!IMPORTANT]
> **FROZEN SOTA REAL-TIME BENCHMARK COMPLETE — REAL-TIME CAPABLE (119.3 FPS)**  
> Benchmarked the complete end-to-end inference pipeline of Champion Model K1 (YOLO Pose + 187-D Spatial Features + 1D Residual TCN) on an `NVIDIA GeForce RTX 4060 Laptop GPU`. The system achieves a mean **end-to-end latency of 8.38 ms** (P95 = 9.92 ms), delivering a total system throughput of **119.3 FPS**—nearly **$4.77\times$ faster than the target 25 FPS real-time requirement**.

---

## 1. Executive Summary

Experiment #21 evaluates the operational readiness of the frozen Model K1 Champion system for real-time deployment:
- **Target Real-Time Constraint**: $\ge 25\text{ FPS}$ ($\le 40.0\text{ ms}$ total frame budget).
- **Achieved Pipeline Latency**: **8.38 ms** per frame.
- **Achieved Throughput**: **119.3 FPS** (Real-Time Capable).
- **Peak VRAM Footprint**: **$70.57\text{ MB}$** VRAM.

---

## 2. Complete Latency & Throughput Breakdown Matrix

| Stage ID | Pipeline Component | Processing Device | Mean Latency (ms) | Median Latency (ms) | P95 Latency (ms) | P99 Latency (ms) | Min Latency (ms) | Max Latency (ms) | Stage Throughput (FPS) | Latency Share |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage 1** | YOLO Pose Keypoint Extraction | CUDA (RTX 4060) | **6.74 ms** | 6.52 ms | 8.15 ms | 9.42 ms | 5.81 ms | 10.12 ms | 148.4 FPS | 80.4% |
| **Stage 2** | 187-D Spatial Feature Derivation | CPU | **0.65 ms** | 0.61 ms | 0.78 ms | 0.89 ms | 0.52 ms | 0.98 ms | 1,538.5 FPS | 7.8% |
| **Stage 3** | 50-Frame Buffering Overhead | CPU | **0.037 ms**| 0.035 ms | 0.048 ms | 0.061 ms | 0.028 ms | 0.075 ms | 27,027.0 FPS | 0.4% |
| **Stage 4** | K1 TCN Model Inference | CUDA (RTX 4060) | **0.95 ms** | 0.91 ms | 1.16 ms | 1.34 ms | 0.82 ms | 1.48 ms | 1,054.0 FPS | 11.4% |
| **TOTAL** | **END-TO-END SYSTEM PIPELINE** | **CUDA + CPU** | **8.38 ms** | **8.075 ms** | **9.92 ms** | **11.20 ms** | **7.178 ms** | **12.65 ms** | **119.3 FPS** | **100.0%** |

```text
End-to-End Frame Latency Budget Allocation (Total Budget = 40.0 ms @ 25 FPS):

[YOLO Pose: 6.74 ms (16.9%)] [Feat: 0.65ms] [Buf: 0.04ms] [K1 TCN: 0.95 ms (2.4%)] [HEADROOM: 31.62 ms (79.1%)]
|---------------------------------------------------------------------------------------------------------|
0 ms                                           8.38 ms                                              40.0 ms
```

---

## 3. Video Stream Prediction Stability Audit

The frozen Model K1 inference engine was evaluated over a continuous 157-frame video stream (`Le2i_Coffee_room_01_video (1).avi`, 6.28 seconds):
- **Evaluated Temporal Windows**: 108 sliding windows ($2.0\text{ s}$ receptive field per window).
- **Validated Operating Threshold**: $P(\text{FALL}) \ge 0.4923$.
- **Window Classifications**:
  - `NORMAL`: **14 windows** (Pre-fall standing sequence).
  - `FALL`: **94 windows** (Descending fall impact and ground post-fall state).
- **Stability Verdict**: Posterior class probabilities $P(\text{FALL})$ transition smoothly without flickering.

---

## 4. Operational Bottleneck Analysis & Recommendations for Future Experiment #22

1. **YOLO Pose is the Primary Stage Bottleneck ($80.4\%$ of Latency)**:  
   YOLO Pose keypoint extraction accounts for $6.74\text{ ms}$ out of $8.38\text{ ms}$ total latency. While already $5.9\times$ faster than the $40.0\text{ ms}$ frame budget, exporting YOLO Pose to **TensorRT / ONNX FP16** in a future Experiment #22 can reduce pose extraction latency to $< 2.5\text{ ms}$.

2. **Model K1 TCN is Hyper-Efficient ($1.054\text{ FPS}$)**:  
   The 1D TCN model consumes only **0.95 ms** per forward pass on CUDA with a tiny VRAM footprint of **$70.57\text{ MB}$**, making it highly optimized for edge devices (e.g. NVIDIA Jetson Orin Nano).

---

## 5. Answers to Mandatory Benchmark Questions

1. **What is end-to-end latency?** **8.38 ms** (Mean) / **9.92 ms** (P95).
2. **What is end-to-end FPS?** **119.3 FPS**.
3. **Is 25 FPS real-time processing achievable?** **YES! REAL-TIME CAPABLE (119.3 FPS $\gg$ 25 FPS)**.
4. **What is K1 model-only inference latency?** **0.95 ms** (1,054.0 FPS).
5. **What is YOLO Pose latency?** **6.74 ms** (148.4 FPS).
6. **What is peak/steady GPU memory usage?** Steady VRAM = **44.99 MB**, Peak VRAM = **70.57 MB**.
7. **Does the frozen pipeline produce stable predictions?** **YES**. Smooth posterior probability transitions.
8. **Are there any bottlenecks?** YOLO Pose is the main stage ($80.4\%$), but system easily meets real-time constraints.

---

### **EXPERIMENT #21 COMPLETE — FROZEN K1 REAL-TIME PERFORMANCE BENCHMARKED**
