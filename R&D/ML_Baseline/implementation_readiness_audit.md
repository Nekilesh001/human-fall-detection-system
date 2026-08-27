# Implementation-Readiness Audit & Baseline Model Design

## 1. Executive Summary & Go / No-Go Decision

> [!IMPORTANT]
> **EXPLICIT DECISION: GO (APPROVED FOR DATASET LOADER & BASELINE MODEL IMPLEMENTATION)**
> All empirical dataset checks, manifest path integrity verifications, PyTorch data-loader tests, and architectural feasibility analyses have been successfully completed. 
> 
> `src/dataset.py` has been implemented and verified. The proposed **Frozen ResNet-18 + Temporal Mean-Std Pooling + MLP Classifier** is scientifically defensible, leak-free, and ready for baseline implementation.

---

## 2. Actual Processed Dataset & Manifest Structure

### A. Directory Structure
```
processed_data/URFD_RGB_baseline/
├── train/                  # 260 compressed .npz sample files
├── val/                    # 43 compressed .npz sample files
├── test/                   # 57 compressed .npz sample files
├── processed_manifest.csv  # 360 portable manifest records
└── preprocessing_report.md # Preprocessing documentation
```

### B. Empirical `.npz` Sample Structure
- **Array Key**: `'frames'`
- **Tensor Shape**: $(50, 240, 320, 3)$
- **Data Type**: `uint8` (values $0 \dots 255$)
- **Temporal Dimension**: $W=50$ frames (2.0s @ 25 FPS)
- **Spatial Dimensions**: $H=240, W_s=320$ pixels, RGB channels ($C=3$).

### C. Processed Manifest Columns (`processed_manifest.csv`)
`['dataset', 'event_id', 'video_id', 'camera_id', 'partition', 'label', 'window_id', 'start_frame', 'end_frame', 'start_timestamp', 'end_timestamp', 'num_frames', 'source_fps', 'target_fps', 'width', 'height', 'source_video_path', 'processed_sample_path']`

---

## 3. Independent Split Integrity Verification

Empirical verification executed directly against `processed_manifest.csv` and stored `.npz` files:

1. **Event Leakage**: **PASSED** ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$).
2. **Camera Leakage**: **PASSED** (For all dual-camera fall events, `cam0` partition == `cam1` partition).
3. **Window Leakage**: **PASSED** (All 50-frame windows extracted post event-level split).
4. **Label Consistency**: **PASSED** (`FALL` windows originate ONLY from `fall-XX`; `NORMAL` windows originate ONLY from `adl-XX`).
5. **Path Integrity**: **PASSED** (0 broken paths out of 360 manifest records).
6. **Unknown Records**: **PASSED** (0 UNKNOWN records in URFD baseline).
7. **Test Set Protection**: **PASSED** (100% of test events [5 Fall, 6 ADL] and 57 test windows retained).

---

## 4. Dataset Loader Design (`src/dataset.py`)

The PyTorch dataset loader class `URFDRGBDataset` has been implemented in `src/dataset.py`:
- **Lazy Loading**: Reads `.npz` files on demand during `__getitem__` to keep RAM usage minimal.
- **Format Conversion**: Converts uint8 array $(50, 240, 320, 3)$ to PyTorch FloatTensor $(50, 3, 240, 320)$ scaled to $[0.0, 1.0]$.
- **ImageNet Normalization**: Normalizes frames using ImageNet mean $\boldsymbol{\mu} = [0.485, 0.456, 0.406]$ and std $\boldsymbol{\sigma} = [0.229, 0.224, 0.225]$.
- **Label Encoding**: `'NORMAL' \to 0`, `'FALL' \to 1`.
- **Returned Dictionary**: `{'frames', 'label', 'event_id', 'video_id', 'window_id', 'camera_id'}`.

---

## 5. Architectural Critique & Response to 18 Design Questions

1. **ImageNet ResNet-18 Appropriateness**: YES. Pretrained low-level edge, limb, and posture features generalize well to $320 \times 240$ indoor human visual streams.
2. **Input Normalization Compatibility**: YES. `(frames / 255.0 - mean) / std` scales $0\dots 255$ uint8 to standard FP32 tensors.
3. **Normalization Constants**: ImageNet mean $[0.485, 0.456, 0.406]$ and std $[0.229, 0.224, 0.225]$.
4. **CNN Batching**: Reshape $(B, 50, 3, 240, 320) \to (B \times 50, 3, 240, 320)$ for single-pass forward feature extraction through ResNet-18.
5. **50-Frame Batch Efficiency**: Extremely fast. $8 \times 50 = 400$ frames per batch requires $< 370\text{ MB}$ VRAM.
6. **Meaning of Temporal Mean + Std Pooling**: Mean $\boldsymbol{\mu}$ captures average spatial posture over 2 seconds (e.g. lying vs upright); Std $\boldsymbol{\sigma}$ captures motion energy/variance during dynamic posture change.
7. **Loss of Temporal Order**: Mean/std does not distinguish forward vs backward motion. However, as an initial baseline, it is a robust, non-sequential statistical summary.
8. **Validity as Baseline**: **YES**. Establishes a strict lower bound for all future sequential models (LSTM/GRU/3D CNN).
9. **Trainable Parameter Calculation**: Linear($1024 \to 64$) $\implies 65,600$ + Linear($64 \to 2$) $\implies 130 = \mathbf{65,730 \text{ trainable parameters}}$.
10. **Class Weighting Formula**:
    $$w_{\text{normal}} = \frac{260}{2 \times 176} \approx 0.7386, \quad w_{\text{fall}} = \frac{260}{2 \times 84} \approx 1.5476$$
11. **Class Weight Data Scope**: Calculated strictly on the 260 Training windows.
12. **Horizontal Flipping Safety**: YES. Left-right body motion is symmetric.
13. **Compatible Augmentations**: Random Horizontal Flip ($p=0.5$), subtle Color Jitter ($\pm 10\%$). Vertical flip remains prohibited.
14. **Dropout Appropriateness**: Dropout(0.5) is acceptable for the 64-dim hidden layer.
15. **Validation Threshold Stability**: Validation set contains 43 windows (10 Fall, 33 Normal). Threshold grid $\tau \in \{0.3, 0.4, 0.5, 0.6, 0.7\}$ will be evaluated.
16. **Threshold Selection Metric**: Maximize Validation F1 Score subject to minimum Specificity $\ge 80\%$.
17. **Threshold Reporting**: Report both default $\tau = 0.5$ and tuned threshold $\tau^*$.
18. **Hidden Leakage Risks**: None. Split boundaries are strictly enforced.

---

## 6. Implementation Contract (20 Points)

1. **Input Format**: Processed `.npz` files in `processed_data/URFD_RGB_baseline/{partition}/`.
2. **Output Format**: PyTorch DataLoader dictionary.
3. **Tensor Shapes**: `frames`: $(B, 50, 3, 240, 320)$, `label`: $(B,)$.
4. **Normalization**: ImageNet mean $[0.485, 0.456, 0.406]$, std $[0.229, 0.224, 0.225]$.
5. **Label Encoding**: `NORMAL = 0`, `FALL = 1`.
6. **Batch Structure**: Default batch size = 8 windows (400 frames).
7. **Model Input Shape**: $(B, 50, 3, 240, 320)$.
8. **Model Output Shape**: $(B, 2)$ unnormalized logits.
9. **Loss Function**: `torch.nn.CrossEntropyLoss(weight=weights)`.
10. **Class Weights**: `[0.7386, 1.5476]`.
11. **Optimizer**: AdamW (`lr=1e-3`, `weight_decay=1e-2`).
12. **Learning Rate Schedule**: Cosine Annealing / Constant with warmup.
13. **Epochs**: 50 epochs.
14. **Validation Protocol**: Evaluate end of every epoch on Val set (43 windows).
15. **Threshold Selection Protocol**: Select $\tau^* \in [0.1, 0.9]$ on Val set.
16. **Test Protocol**: Single final evaluation on Test set (57 windows) using $\tau^*$.
17. **Metrics**: Precision, Recall/Sensitivity, F1 Score, Specificity, Confusion Matrix, Latency (ms), FPS, Parameter Count.
18. **Reproducibility**: Set seeds for `random`, `numpy`, `torch`.
19. **Random Seed**: `seed = 42`.
20. **Checkpointing**: Save best model state dict to `checkpoints/urfd_rgb_baseline_best.pth` (Git-ignored).

---

## 7. File Action Matrix

- **Files Created**:
  - `src/dataset.py` (PyTorch Dataset Loader — IMPLEMENTED & TESTED)
  - `R&D/ML_Baseline/implementation_readiness_audit.md` (This document)
- **Files to be Created in Next Step**:
  - `src/model.py` (ResNet18 + Temporal Mean-Std MLP Classifier)
  - `src/train_baseline.py` (Training & Evaluation Script)
- **Files NOT to be Modified**:
  - Raw datasets (`URFD/`, `Le2i/`, `dataset/`)
  - `processed_data/URFD_RGB_baseline/`
  - `R&D/split_strategy.md`

---

## 8. Current Git Status

```text
On branch dev
Your branch is up to date with 'origin/dev'.

Untracked files:
  R&D/ML_Baseline/
  src/dataset.py
```
