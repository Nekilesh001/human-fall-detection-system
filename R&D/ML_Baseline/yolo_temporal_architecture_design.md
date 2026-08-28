# Research Design: YOLO Pose Temporal Architecture Benchmark (Experiment I)

> [!IMPORTANT]
> **DESIGN & READINESS AUDIT ONLY — NO TRAINING PERFORMED YET**  
> This document specifies the controlled experimental design, model architectures, temporal parameterizations, 4-fold LOLO partitions, and evaluation protocols for Experiment I: YOLO Pose Temporal Architecture Benchmark.

---

## 1. Scientific Objective & Research Question

In Experiment H, switching from MediaPipe Pose to **YOLO Pose** achieved a massive performance breakthrough, elevating Leave-One-Location-Out (LOLO) Mean F1 from **$68.48\%$ to $80.46\%$ (+12.0% absolute gain)** under a static 21,314-parameter MLP control head.

Experiment I addresses the next core scientific question:
> *"Given the winning YOLO Pose keypoint representation (165-D), does explicit temporal sequence modeling (GRU, LSTM, TCN, Transformer) improve cross-location fall-detection performance beyond the 80.46% YOLO Pose MLP control baseline?"*

---

## 2. Benchmark Model Architectures & Parameter Specifications

All 5 benchmark models consume input shape **$(B, 50, 165)$ float32** from `processed_data/Le2i_baseline/pose_estimator_features/yolo_pose/`:

### Model I0: YOLO Pose Control MLP (Baseline Control)
- **Architecture**: Mean + Std Temporal Pooling $\to$ Linear(330 $\to$ 64) $\to$ ReLU $\to$ Dropout(0.5) $\to$ Linear(64 $\to$ 2)
- **Trainable Parameters**: **21,314 parameters**
- **Role**: Scientific control benchmark establishing static pooling performance.

### Model I1: YOLO Pose + 1-Layer GRU
- **Architecture**: GRU(`input_size=165, hidden_size=64, num_layers=1`, batch_first=True) $\to$ Final Hidden State $h_T \to$ Linear(64 $\to$ 32) $\to$ ReLU $\to$ Dropout(0.5) $\to$ Linear(32 $\to$ 2)
- **Trainable Parameters**: **46,498 parameters**

### Model I2: YOLO Pose + 1-Layer LSTM
- **Architecture**: LSTM(`input_size=165, hidden_size=64, num_layers=1`, batch_first=True) $\to$ Final Hidden State $h_T \to$ Linear(64 $\to$ 32) $\to$ ReLU $\to$ Dropout(0.5) $\to$ Linear(32 $\to$ 2)
- **Trainable Parameters**: **61,282 parameters**

### Model I3: YOLO Pose + 1D TCN (Temporal Convolutional Network)
- **Architecture**: 2 Residual TCN Blocks (`num_channels=[64, 64], kernel_size=3`, dilations=[1, 2], Dropout=0.2) $\to$ Mean+Max Concatenated Pooling (128-D) $\to$ Linear(128 $\to$ 32) $\to$ ReLU $\to$ Dropout(0.5) $\to$ Linear(32 $\to$ 2)
- **Trainable Parameters**: **83,618 parameters**

### Model I4: YOLO Pose + Transformer Encoder
- **Architecture**: Linear(165 $\to$ 64) + Positional Encoding $\to$ 1-Layer TransformerEncoder (`d_model=64, nhead=4, dim_feedforward=128`, Dropout=0.1, batch_first=True) $\to$ Global Mean Pooling $\to$ Linear(64 $\to$ 32) $\to$ ReLU $\to$ Dropout(0.5) $\to$ Linear(32 $\to$ 2)
- **Trainable Parameters**: **46,242 parameters**

---

## 3. Controlled Experimental Variables

| Controlled Parameter | Specification | Scientific Rationale |
| :--- | :--- | :--- |
| **Input Representation** | YOLO Pose 165-D Tensors | Controlled 99-D Pose Geometry + 66-D Joint Velocity |
| **Window Length** | 50 frames ($2.0\text{ s}$ at 25 FPS) | Fixed temporal receptive field |
| **Outer Evaluation** | 4-Fold Physical LOLO | Zero outer test location data in training |
| **Inner Split** | 80/20 Event-Stratified Split | Zero video event overlap between train and validation |
| **Seed & Determinism** | `set_seed(42)` per fold/model | 100% deterministic reproducibility |
| **Training Execution** | PyTorch GPU CUDA (RTX 4060) | Hardware consistency |
| **Threshold Tuning** | Optimal $\tau^*$ on Inner Val | Frozen application to Outer Test |

---

## 4. 4-Fold LOLO Partitioning Protocol

| Fold | Outer Test Location | Outer Train Windows | Outer Test Windows | Outer Event Overlap | Partition Audit |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Fold 1** | `Coffee_room_01` | 894 | 502 | **0** | **LEAKAGE FREE ✅** |
| **Fold 2** | `Coffee_room_02` | 986 | 410 | **0** | **LEAKAGE FREE ✅** |
| **Fold 3** | `Home_01` | 1,157 | 239 | **0** | **LEAKAGE FREE ✅** |
| **Fold 4** | `Home_02` | 1,151 | 245 | **0** | **LEAKAGE FREE ✅** |
