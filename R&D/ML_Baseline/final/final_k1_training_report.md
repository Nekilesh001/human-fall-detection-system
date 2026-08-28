# Final K1 Training Report

**Model**: K1 — YOLO Pose + 187-D Spatial Features + 1D Residual TCN
**Phase**: F1 Final Training
**Date**: 2026-08-28

---

## 1. Hyperparameters

| Parameter | Value |
| :--- | :--- |
| Architecture | 2-Block Residual 1D TCN |
| Input Dim | 187-D |
| Channels | [64, 64] |
| Kernel Size | 3 |
| Dilations | [1, 2] |
| Pooling | Mean + Max |
| FC Dim | 32 |
| Dropout | 0.5 (training) |
| Optimizer | Adam |
| Learning Rate | 1e-3 |
| Weight Decay | 1e-4 |
| Epochs | 100 |
| Batch Size | 32 |
| Checkpoint criterion | Max inner-val F1 |

---

## 2. Option A — LOLO Fold Checkpoint Summary

| Fold | Test Location | tau* | Best Inner-Val F1 | Checkpoint |
| :---: | :--- | :---: | :---: | :--- |
| 1 | Coffee_room_01 | 0.3400 | 0.9697 | `checkpoints/final_k1/fold_1_best.pth` |
| 2 | Coffee_room_02 | 0.2000 | 0.9286 | `checkpoints/final_k1/fold_2_best.pth` |
| 3 | Home_01 | 0.4800 | 0.9302 | `checkpoints/final_k1/fold_3_best.pth` |
| 4 | Home_02 | 0.4400 | 0.9259 | `checkpoints/final_k1/fold_4_best.pth` |

---

## 3. Option B — Production Checkpoint

- Trained on all 1,396 windows
- Deployment threshold: tau = 0.3650
- Checkpoint: `d:\ONE_DATA\Fall detection\checkpoints\final_k1\final_production.pth`
- No held-out evaluation. Performance estimated from Option A LOLO benchmark (86.65% LOLO Mean F1).
