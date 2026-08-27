# Research Design: Temporal Architecture Benchmark for Pose + Velocity Fall Detection (Experiment G)

> [!IMPORTANT]
> **DESIGN ONLY — NO TRAINING PERFORMED YET — NO FEATURE REGENERATION PERFORMED.**
> This document specifies the scientific protocol, model architectures (G0, G1, G2, G3, G4), parameter counts, loss functions, and evaluation metrics for Experiment G: Temporal Architecture Benchmark.

---

## 1. Scientific Objective & Research Question
To determine whether explicit sequence modeling architectures (GRU, LSTM, TCN, Transformer) provide a statistically and practically meaningful improvement over the canonical Experiment E2 Pose + Velocity Mean+Std baseline (**$72.23\%$ LOLO Mean F1**) on unseen physical locations.

### Core Scientific Question
*"Does modeling sequential temporal dynamics over frame-to-frame joint trajectories provide better cross-location fall detection accuracy than simple Mean+Std temporal aggregation ($72.23\%$ F1), or is temporal order uninformative when body keypoint geometry is already frame-normalized?"*

---

## 2. Benchmark Model Architectures & Parameter Audit

All benchmark models process the exact same precomputed E2 Pose + Velocity feature tensors `(B, 50, 165)` (165-D per frame: 99-D Pose Geometry + 66-D Joint Velocity) across the 1,396 temporal windows of the Le2i dataset.

```text
G0 (Control MLP):      (B, 50, 165) ──► Mean+Std (330) ──► Linear(330→64) ──► ReLU ──► Dropout(0.5) ──► Linear(64→2) [21,314 params]
G1 (1-Layer GRU):      (B, 50, 165) ──► GRU(165→64) ──► Final H (64) ──► Linear(64→32) ──► ReLU ──► Dropout(0.5) ──► Linear(32→2) [46,498 params]
G2 (1-Layer LSTM):     (B, 50, 165) ──► LSTM(165→64) ──► Final H (64) ──► Linear(64→32) ──► ReLU ──► Dropout(0.5) ──► Linear(32→2) [61,282 params]
G3 (TCN):              (B, 50, 165) ──► TCN(165→64, d=1,2) ──► Mean+Max (128) ──► Linear(128→32) ──► ReLU ──► Dropout(0.5) ──► Linear(32→2) [83,618 params]
G4 (Transformer):     (B, 50, 165) ──► Proj+PE (64) ──► TransEnc(1-L, 4-H) ──► Mean (64) ──► Linear(64→32) ──► ReLU ──► Dropout(0.5) ──► Linear(32→2) [46,242 params]
```

| Model Variant | Temporal Architecture Specification | Sequence Aggregation Method | Classifier Sub-Network | Trainable Parameters | Benchmark Role |
| :--- | :--- | :---: | :--- | :---: | :--- |
| **G0 (Control)** | Canonical E2 Mean+Std MLP | Temporal Mean + Std Pooling $\to (B, 330)$ | `Linear(330 -> 64) -> ReLU -> Dropout(0.5) -> Linear(64 -> 2)` | **21,314** | Canonical Reference Control |
| **G1 (GRU)** | 1-Layer Gated Recurrent Unit (`hidden_size=64`) | Final Hidden State $h_{50} \to (B, 64)$ | `Linear(64 -> 32) -> ReLU -> Dropout(0.5) -> Linear(32 -> 2)` | **46,498** | Recurrent Sequence Model |
| **G2 (LSTM)** | 1-Layer Long Short-Term Memory (`hidden_size=64`) | Final Hidden State $h_{50} \to (B, 64)$ | `Linear(64 -> 32) -> ReLU -> Dropout(0.5) -> Linear(32 -> 2)` | **61,282** | Recurrent Memory Model |
| **G3 (TCN)** | 1D Temporal Convolutional Network (2 blocks, dilations 1, 2, 64 ch) | Temporal Mean + Max Pooling $\to (B, 128)$ | `Linear(128 -> 32) -> ReLU -> Dropout(0.5) -> Linear(32 -> 2)` | **83,618** | Receptive Field Conv Model |
| **G4 (Transformer)** | 1-Layer Lightweight Transformer Encoder (4 heads, `d_model=64`, PE) | Temporal Mean Pooling $\to (B, 64)$ | `Linear(64 -> 32) -> ReLU -> Dropout(0.5) -> Linear(32 -> 2)` | **46,242** | Self-Attention Sequence Model |

---

## 3. 4-Fold LOLO Experimental Protocol

- **Dataset Scope**: 1,396 windows across 127 verified supervised videos (96 FALL, 31 NORMAL). All 63 UNKNOWN records are strictly excluded.
- **Folds Partitioning**:
  - **Fold 1**: Outer Test = `Coffee_room_01` (894 train wins, 502 test wins)
  - **Fold 2**: Outer Test = `Coffee_room_02` (986 train wins, 410 test wins)
  - **Fold 3**: Outer Test = `Home_01` (1,157 train wins, 239 test wins)
  - **Fold 4**: Outer Test = `Home_02` (1,151 train wins, 245 test wins)

---

## 4. Inner Validation, Class Weighting & Threshold Strategy

- **Inner Event Split**: Inner Train (80% of outer train events) and Inner Validation (20% of outer train events) constructed with zero video event overlap ($\text{Inner Train Events} \cap \text{Inner Val Events} = \emptyset$).
- **Outer Test Isolation**: Outer held-out test location is 100% unseen (0 test windows used for training, class weights, checkpoint selection, or threshold tuning).
- **Class Weights**: Calculated programmatically per fold from outer training location windows.
- **Threshold Selection**: Inner validation threshold $\tau^*_{\text{inner}}$ searched in $[0.05, 0.95]$ to maximize Inner Validation F1.

---

## 5. Seed Isolation & Reproducibility Controls

To prevent CUDA random state cascading between models and folds:
- Call `set_seed(42)` BEFORE EVERY MODEL AND BEFORE EVERY FOLD:
  ```python
  random.seed(42)
  np.random.seed(42)
  torch.manual_seed(42)
  torch.cuda.manual_seed_all(42)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  ```
- Standardize window ordering by sorting `df_manifest` by `window_id` before inner event partitioning.

---

## 6. Expected Output Artifacts & Directories

- Checkpoints path: `checkpoints/le2i_temporal_benchmark/{g0, g1, g2, g3, g4}/fold_{1..4}_best.pth`.
- Results CSV path: `R&D/ML_Baseline/results/le2i_temporal_benchmark/temporal_benchmark_results.csv`.
- Canonical research report artifact: [`R&D/ML_Baseline/temporal_architecture_report.md`](file:///d:/ONE_DATA/Fall%20detection/R&D/ML_Baseline/temporal_architecture_report.md).
