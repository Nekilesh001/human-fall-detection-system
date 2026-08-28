# Research Design: Class Balancing & Oversampling Strategies (Experiment #17)

> [!IMPORTANT]
> **EXPERIMENTAL DESIGN ONLY — READINESS AUDIT PHASE ONLY — NO CODE MODIFIED — NO TRAINING EXECUTED**  
> This document specifies the controlled experimental design, class weighting formulations, oversampling algorithms, balanced sampling protocols, and 4-fold LOLO ablation plans for Experiment #17: Class Balancing & Oversampling Strategies.

---

## 1. Research Motivation & Core Question

In real-world video datasets such as Le2i, normal daily activities (walking, sitting, standing) outnumber fall events:
- Total Supervised Windows: **1,396 windows**
- Normal Windows ($y=0$): **1,065 windows ($76.29\%$)**
- Fall Windows ($y=1$): **331 windows ($23.71\%$)**
- Global Imbalance Ratio: **$3.22 : 1$** (NORMAL : FALL)

While the Champion SOTA Model K1 (187-D Spatial TCN) achieved **$86.60\%$ LOLO Mean F1**, standard unweighted cross-entropy loss tends to bias model gradients towards the majority `NORMAL` class, leading to potential under-prediction of falls in hard physical environments (e.g., `Home_02`).

Experiment #17 addresses the research question:
> *"Do class-balancing techniques (weighted loss, random oversampling, or balanced batch sampling) improve cross-location fall detection beyond the baseline K1 SOTA ($86.60\%$) while keeping model architecture, features, and 4-fold LOLO evaluation protocol strictly identical?"*

---

## 2. Benchmark Variant Formulations

All four variants utilize the exact **ModelK1_SpatialTCN (86,434 params)** architecture and **187-D spatial feature tensors `(50, 187)` float32** from `processed_data/Le2i_baseline/pose_estimator_features/yolo_pose_k1/`:

```text
Experiment #17 Controlled Variant Matrix:

EXP17-A (Control)         : Unweighted Cross-Entropy Loss + Standard DataLoader (Reference Baseline)
EXP17-B (Weighted Loss)   : Class-Weighted Cross-Entropy Loss w_fall = N_norm / N_fall
EXP17-C (Oversampling)    : Random Oversampling of FALL windows within inner training split
EXP17-D (Balanced Sampler): PyTorch WeightedRandomSampler (equal 50/50 class probability per batch)
```

### Variant EXP17-A: K1 Control (Unweighted Reference)
- **Loss Function**: Standard unweighted CrossEntropyLoss:
  $$\mathcal{L}_{\text{CE}} = - \frac{1}{B} \sum_{i=1}^B \log P(y_i \mid X_i)$$
- **Sampling**: Standard random shuffle DataLoader (`batch_size=32`).

### Variant EXP17-B: Class-Weighted Loss
- **Loss Function**: Class-Weighted CrossEntropyLoss:
  $$\mathcal{L}_{\text{WCE}} = - \frac{1}{B} \sum_{i=1}^B w_{y_i} \log P(y_i \mid X_i)$$
- **Weight Calculation**: Computed per fold strictly on the **inner training split**:
  $$w_0 = 1.0, \quad w_1 = \frac{N_{\text{NORMAL, inner\_train}}}{N_{\text{FALL, inner\_train}}}$$

### Variant EXP17-C: Random Oversampling (Inner Train Only)
- **Oversampling Strategy**: Duplicate `FALL` windows strictly inside `inner_train_df` until $N_{\text{FALL, oversampled}} = N_{\text{NORMAL, inner\_train}}$.
- **Isolation Rule**: Inner validation and outer test partitions remain **100% untouched** (original un-oversampled class distribution).

### Variant EXP17-D: Balanced Batch Sampler
- **Sampling Strategy**: `torch.utils.data.WeightedRandomSampler` applied to `inner_train_df`.
- **Sample Weights**: $w_{i, \text{sample}} = 1.0 / N_{y_i, \text{inner\_train}}$.
- **Batch Composition**: Draws `len(inner_train_df)` samples per epoch where each sample has an equal $50\%$ probability of being `FALL` or `NORMAL`.

---

## 3. Scientific Controls & 4-Fold LOLO Protocol

| Protocol Parameter | Specification | Scientific Rationale |
| :--- | :--- | :--- |
| **Model Architecture** | `ModelK1_SpatialTCN` (86,434 params) | 100% architectural control |
| **Input Features** | 187-D Spatial Feature Tensors `(50, 187)` | 100% feature control |
| **Physical Locations** | `Coffee_room_01`, `Coffee_room_02`, `Home_01`, `Home_02` | Identical 4-fold LOLO partitions |
| **Leakage Control** | Event-level grouping (`event_id`) | Zero event overlap across splits |
| **Balancing Scope** | Applied STRICTLY to inner train split | Zero modification of val or test data |
| **Threshold Selection** | Tuned on inner validation F1 | Zero outer-test threshold leakage |
| **Random Seed** | `set_seed(42 + fold_idx)` per fold | Deterministic reproducibility |

---

## 4. Per-Fold Outer Train Class Ratios & Loss Weights

| Fold | Outer Test Location | Outer Train Total | Normal Samples ($y=0$) | Fall Samples ($y=1$) | Imbalance Ratio | EXP17-B Fall Loss Weight ($w_1$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | `Coffee_room_01` | 894 | 735 | 159 | $4.62 : 1$ | **4.6226** |
| **Fold 2** | `Coffee_room_02` | 986 | 702 | 284 | $2.47 : 1$ | **2.4718** |
| **Fold 3** | `Home_01` | 1,157 | 916 | 241 | $3.80 : 1$ | **3.8008** |
| **Fold 4** | `Home_02` | 1,151 | 842 | 309 | $2.72 : 1$ | **2.7249** |

---

## 5. Output Isolation Plan

```text
checkpoints/le2i_exp17_class_balance/
├── control/          (EXP17-A Checkpoints: fold_{1..4}_best.pth)
├── weighted_loss/    (EXP17-B Checkpoints: fold_{1..4}_best.pth)
├── oversampling/     (EXP17-C Checkpoints: fold_{1..4}_best.pth)
└── balanced_sampler/ (EXP17-D Checkpoints: fold_{1..4}_best.pth)
```

- **Benchmark Results JSON**: `R&D/ML_Baseline/results/exp17_class_balance_results.json`
- **Benchmark Results CSV**: `R&D/ML_Baseline/results/exp17_class_balance_results.csv`
- **Existing K1 Artifacts**: **100% Safe, Isolated, and Untouched**.
