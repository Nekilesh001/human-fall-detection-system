# Experiment K Phase K3: ST-GCN Readiness Audit & Smoke Test Report

## 1. Executive Summary & Readiness Verdict
- **Audit Target**: Verification of feature reusability, COCO-17 graph topology, input tensor formulation, parameter counts, partition splits, checkpoint isolation, and CUDA forward pass smoke test for Model K3 (Spatial-Temporal Graph Convolutional Network).
- **Readiness Verdict**: **EXPERIMENT K3 READY FOR TRAINING — NO CODE MODIFIED — NO TRAINING EXECUTED**

---

## 2. Programmatic Parameter Count & Architecture Verification

| Model Variant | Input Representation | Architecture Description | Trainable Parameters | Parameter Verification |
| :--- | :---: | :--- | :---: | :---: |
| **K0 Control TCN** | `(B, 50, 165)` | 2 Residual 1D TCN Blocks | **83,618** | **VERIFIED ✅** |
| **K1 Spatial TCN** | `(B, 50, 187)` | 2 Residual 1D TCN Blocks + 22 Body Angles | **86,434** | **VERIFIED ✅** |
| **K3 ST-GCN Graph** | `(B, 5, 50, 17)` | 3 ST-GCN Blocks (COCO-17 Graph) | **107,778** | **VERIFIED ✅** |

---

## 3. CUDA Forward Pass Smoke Test Verification Results

A deterministic CUDA smoke test was executed on the `NVIDIA GeForce RTX 4060 Laptop GPU`:
1. **Model Instantiation**: `ModelK3_STGCN(in_channels=5, num_classes=2)` **[PASS ✅]**
2. **CUDA Transfer**: Successfully loaded on device `cuda:0` **[PASS ✅]**
3. **Real Window Conversion Test**: Converted canonical 165-D YOLO Pose NPZ to `(5, 50, 17)` float32 in memory **[PASS ✅]**
4. **Forward Pass Execution**: `dummy_input = torch.randn(1, 5, 50, 17).cuda()` $\to$ `output = model(dummy_input)` **[PASS ✅]**
5. **Output Tensor Shape**: Exactly `torch.Size([1, 2])` **[PASS ✅]**
6. **Data Integrity**: **0 NaN, 0 Inf** in output logits **[PASS ✅]**

---

## 4. Dataset & Reusability Audit

- **Source Feature Directory**: `processed_data/Le2i_baseline/pose_estimator_features/yolo_pose/`
- **Reusability**: **100% Reusable**. Slicing indices `[0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]` yields the 17 populated COCO keypoints.
- **Precomputation Requirement**: **NONE**. Instant dynamic slicing in `YoloK3Dataset`.

---

## 5. LOLO & Leakage Audit

- **Physical Locations**: `Coffee_room_01`, `Coffee_room_02`, `Home_01`, `Home_02`
- **Outer Train / Outer Test Event Overlap**: **0 overlapping event IDs** across all 4 folds.
- **Inner Validation Split**: 80/20 event-stratified split with zero event overlap.
- **Seed Isolation**: `set_seed(42 + fold_idx)` per fold.

---

## 6. GPU & Computational Cost Estimation

- **Execution Hardware**: NVIDIA GeForce RTX 4060 Laptop GPU.
- **VRAM Usage**: $< 1.1\text{ GB}$ VRAM.
- **Estimated Training Time**: $\sim 45\text{ seconds}$ per fold ($\sim 3.0\text{ minutes}$ total for 4 folds).

---

## 7. Checkpoint & Result Isolation

- **Checkpoint Path**: `checkpoints/le2i_yolo_k3_stgcn/fold_{1..4}_best.pth`
- **Result Paths**:
  - `R&D/ML_Baseline/results/yolo_k3_stgcn_benchmark_results.json`
  - `R&D/ML_Baseline/results/yolo_k3_stgcn_benchmark_results.csv`
- **Existing Experiments A–K1**: **100% Safe and Untouched**.

---

## 8. Audit Verdict & Next Training Command

When ready to launch Experiment K Phase K3 Training across all 4 LOLO partitions, run:

```powershell
cd "d:\ONE_DATA\Fall detection"
& "C:\Users\NEKILESH\AppData\Local\Programs\Python\Python311\python.exe" src/train_le2i_yolo_k3_stgcn.py
```
