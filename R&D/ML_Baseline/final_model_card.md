# Model Card: Champion Model K1 (YOLO Pose + 187-D Spatial TCN)

## 1. Model Overview
- **Model Identifier**: Model K1 Champion (Spatial-Augmented Temporal Convolutional Network)
- **Model Version**: 1.0 (Frozen SOTA Milestone)
- **Release Date**: August 2026
- **Developer**: Advanced Agentic Fall Detection R&D Team
- **Primary Function**: Automated detection of human fall events from video keypoint streams in indoor environments.

> [!CAUTION]
> **DISCLAIMER**: This model is a research prototype for automated video monitoring. It is **NOT** a certified medical device and should not replace emergency response personnel or medical monitoring hardware.

---

## 2. Intended Use & Capabilities
- **Intended Deployment Contexts**: Smart homes, assisted living facilities, hospital rooms, and indoor camera streams.
- **Input Requirements**: Continuous video stream ($\ge 25\text{ FPS}$) or 2D keypoint coordinates extracted by YOLO Pose (`yolov8n-pose.pt`).
- **Output**: Binary classification (`NORMAL` vs `FALL`) and posterior probability $P(\text{FALL})$.
- **Temporal Receptive Field**: 50 frames ($2.0\text{ seconds}$ at 25 FPS).

---

## 3. Training & Evaluation Protocol
- **Dataset**: Le2i Fall Detection Dataset (127 videos, 1,396 supervised 50-frame windows).
- **Validation Protocol**: 4-Fold Leave-One-Location-Out (LOLO) cross-validation across `Coffee_room_01`, `Coffee_room_02`, `Home_01`, `Home_02`.
- **Event-Level Leakage Control**: All outer train/test splits and inner train/val splits are strictly partitioned by `event_id` (0 event overlap).
- **Training Hyperparameters**: Adam optimizer ($\text{lr}=10^{-3}$, $\text{weight\_decay}=10^{-4}$), 100 epochs, batch size = 32, PyTorch CUDA.

---

## 4. Benchmark Performance Metrics

| Metric | Official Un-Cheated Benchmark (@ $\tau^*_{\text{inner}}$) | High-Recall Mode (@ $\tau = 0.35$) | High-Precision Mode (@ $\tau = 0.55$) |
| :--- | :---: | :---: | :---: |
| **LOLO Mean F1** | **$86.65\%$** | $85.77\%$ | **$87.45\%$** |
| **Cross-Room Variance ($\sigma$)** | **$\pm 5.64\%$** | $\pm 5.91\%$ | $\pm 5.46\%$ |
| **Mean Recall / Sensitivity** | **$92.96\%$** | **$96.07\%$** | $91.59\%$ |
| **Mean Specificity** | **$91.91\%$** | $90.76\%$ | **$93.53\%$** |
| **Total False Positives** | **67 / 1,065** | 76 / 1,065 | **55 / 1,065** |
| **Total False Negatives** | **23 / 331** | **17 / 331** | 27 / 331 |

---

## 5. Per-Location Location Performance Breakdown

| Physical Environment | Test Windows | Test Falls | F1 Score (@ $\tau^*_{\text{inner}}$) | Recall | Specificity | True Positives | False Negatives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Coffee_room_01` | 502 | 172 | **`0.9188`** | `0.9535` | `0.9364` | 164 | 8 |
| `Coffee_room_02` | 410 | 47 | **`0.9038`** | **`1.0000`** | `0.9725` | 47 | **0** |
| `Home_01` | 239 | 90 | **`0.7739`** | `0.8556` | `0.7852` | 77 | 13 |
| `Home_02` | 245 | 22 | **`0.8696`** | `0.9091` | `0.9821` | 20 | 2 |

---

## 6. Known Failure Modes & Limitations
1. **Severe Severe Occlusion**: If $> 60\%$ of body keypoints are occluded by heavy furniture (e.g. bed edges or sofas in `Home_01`), keypoint confidence drops, leading to potential false negatives.
2. **Abrupt Intentional Crouching**: Rapid crouching/bending movements near ground level can occasionally trigger false positive alerts due to downward velocity similarity.
3. **Camera Perspective**: Performance is optimized for fixed indoor wall-mounted cameras at $2.0 - 2.5\text{ meters}$ height.

---

## 7. Reproducibility & Verification
To verify checkpoint reproducibility, run:

```powershell
cd "d:\ONE_DATA\Fall detection"
& "C:\Users\NEKILESH\AppData\Local\Programs\Python\Python311\python.exe" src/evaluate_le2i_yolo_k1_spatial.py
```
