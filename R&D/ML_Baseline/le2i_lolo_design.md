# Research Design: Le2i Supervised Leave-One-Location-Out (LOLO) Baseline Protocol

> [!IMPORTANT]
> **DESIGN ONLY — NO TRAINING PERFORMED YET.**
> This document specifies the scientific protocol, fold definitions, inner validation strategy, class weight formula, and evaluation metrics for Experiment B: Supervised In-Domain Le2i Baseline.

---

## 1. Objective
To establish an in-domain baseline for fall detection on the Le2i dataset by training the baseline classifier architecture (`URFDRGBFeatureBaseline`, 65,730 trainable parameters) on 3 physical locations and evaluating generalization performance on the 4th held-out unseen physical location under a 4-Fold Leave-One-Location-Out (LOLO) cross-validation protocol.

---

## 2. Scientific Question
*"When a model is trained directly on Le2i environment variations (holding out one physical scene), how much does in-domain location generalization improve compared to zero-shot transfer from URFD?"*

This experiment separates:
1. **Raw Cross-Dataset Shift** (URFD $\to$ Le2i Zero-Shot) vs.
2. **Unseen Scene In-Domain Generalization** (Le2i $\to$ Le2i LOLO).

---

## 3. Dataset Scope & Supervised Data Selection
- **Total Preprocessed Temporal Windows**: 1,396 windows ($W=50, S=25, 25\text{ FPS}, 320 \times 240$).
- **Supervised Video Events**: 127 verified videos (96 FALL, 31 NORMAL).
- **Excluded Records**: All 63 UNKNOWN records (60 Office/Lecture Room videos + 3 malformed annotation records) remain 100% EXCLUDED.

### Location Distribution
- `Coffee_room_01`: 502 windows (172 FALL, 330 NORMAL) — 47 FALL videos, 0 NORMAL videos.
- `Coffee_room_02`: 410 windows (47 FALL, 363 NORMAL) — 12 FALL videos, 8 NORMAL videos.
- `Home_01`: 239 windows (90 FALL, 149 NORMAL) — 30 FALL videos, 0 NORMAL videos.
- `Home_02`: 245 windows (22 FALL, 223 NORMAL) — 7 FALL videos, 23 NORMAL videos.

---

## 4. 4-Fold LOLO Partition Protocol

```text
               ┌─────────────────────────────────────────────────────────┐
Fold 1         │ Outer Train: Coffee_room_02 + Home_01 + Home_02 (894w)  │ ──► Outer Test: Coffee_room_01 (502w)
               └─────────────────────────────────────────────────────────┘
               ┌─────────────────────────────────────────────────────────┐
Fold 2         │ Outer Train: Coffee_room_01 + Home_01 + Home_02 (986w)  │ ──► Outer Test: Coffee_room_02 (410w)
               └─────────────────────────────────────────────────────────┘
               ┌─────────────────────────────────────────────────────────┐
Fold 3         │ Outer Train: Coffee_room_01 + Coffee_room_02 + Home_02  │ ──► Outer Test: Home_01 (239w)
               └─────────────────────────────────────────────────────────┘
               ┌─────────────────────────────────────────────────────────┐
Fold 4         │ Outer Train: Coffee_room_01 + Coffee_room_02 + Home_01  │ ──► Outer Test: Home_02 (245w)
               └─────────────────────────────────────────────────────────┘
```

| Fold Name | Outer Test Location | Outer Train Locations | Outer Train Windows | Outer Test Windows | Fold Class Weights ($w_{\text{normal}}, w_{\text{fall}}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | `Coffee_room_01` | `Coffee_room_02`, `Home_01`, `Home_02` | 894 (159 F, 735 N) | 502 (172 F, 330 N) | $w_{\text{norm}} = 0.6082, w_{\text{fall}} = 2.8113$ |
| **Fold 2** | `Coffee_room_02` | `Coffee_room_01`, `Home_01`, `Home_02` | 986 (284 F, 702 N) | 410 (47 F, 363 N) | $w_{\text{norm}} = 0.7023, w_{\text{fall}} = 1.7359$ |
| **Fold 3** | `Home_01` | `Coffee_room_01`, `Coffee_room_02`, `Home_02` | 1157 (241 F, 916 N) | 239 (90 F, 149 N) | $w_{\text{norm}} = 0.6316, w_{\text{fall}} = 2.4004$ |
| **Fold 4** | `Home_02` | `Coffee_room_01`, `Coffee_room_02`, `Home_01` | 1151 (309 F, 842 N) | 245 (22 F, 223 N) | $w_{\text{norm}} = 0.6835, w_{\text{fall}} = 1.8625$ |

---

## 5. Critical Leakage Prevention & Isolation Rules
1. **Outer Test Isolation**: The held-out outer test location is 100% unseen until final evaluation. Zero test samples are used for model training, class-weight calculation, epoch selection, or threshold tuning.
2. **Event Boundary Isolation**: Training and inner validation splits are performed at the **video event level**. Windows from the same video/event never appear in both train and validation partitions.
3. **No Window-Level Random Splitting**: Location and event boundaries are strictly preserved.

---

## 6. Inner Validation & Checkpoint Selection Strategy

Since no global Le2i validation location exists, an **Inner Event-Level Validation Split** is constructed within each fold using ONLY the 3 outer training locations:

- **Inner Train Set**: $\approx 80\%$ of outer training events.
- **Inner Validation Set**: $\approx 20\%$ of outer training events.
- **Model Selection Rule**: The best model checkpoint for each fold is saved from the epoch achieving the highest **Inner Validation F1 Score** ($\text{Val F1}$).
- **Threshold Selection Rule**: The decision threshold $\tau^*_{\text{inner}}$ is selected by searching $\tau \in [0.05, 0.95]$ to maximize Inner Validation F1 Score. The outer test set is evaluated once at $\tau=0.50$ and once at $\tau^*_{\text{inner}}$.

---

## 7. Model Architecture & Feature Representation
- **Backbone**: Frozen ImageNet ResNet-18 (512-dim features per frame).
- **Temporal Pooling**: Mean + Standard Deviation Pooling along dim 1 $\to (B, 1024)$.
- **Classifier Head**:
  ```text
  Linear(1024 → 64) ──► ReLU ──► Dropout(p=0.5) ──► Linear(64 → 2)
  ```
- **Trainable Parameters**: **65,730**.
- **Input Feature Source**: Precomputed Le2i features (`processed_data/Le2i_baseline/features/`).

---

## 8. Training Hyperparameters
- **Optimizer**: AdamW ($\text{lr} = 1\text{e-}3$, $\text{weight\_decay} = 1\text{e-}2$)
- **Loss Function**: CrossEntropyLoss with fold-specific class weights $[w_{\text{norm}}, w_{\text{fall}}]$
- **Epochs**: 50 max epochs (with inner validation checkpointing)
- **Batch Size**: 32
- **Random Seed**: `42` (applied to Python `random`, `numpy`, `torch`, and deterministic cuDNN)

---

## 9. Evaluation Metrics Suite
For each LOLO fold and averaged across all 4 folds:
- **Window-Level**: Accuracy, Precision, Recall / Sensitivity, Specificity, F1 Score, Confusion Matrix.
- **Event-Level**: Event Sensitivity (percentage of fall video events with $\ge 1$ FALL alert during active/post-fall phase).
- **Time-to-Detection ($\Delta t$)**: Temporal latency between fall onset and first alert.

---

## 10. Implementation Plan & Execution Order
1. Create `src/train_le2i_lolo.py`: Implements the 4-fold LOLO training loop with inner validation checkpointing and fold-specific class weights.
2. Create `src/evaluate_le2i_lolo.py`: Evaluates frozen LOLO checkpoints on outer held-out test locations.
3. Generate `R&D/ML_Baseline/le2i_lolo_evaluation_report.md`: Summarizes outer test performance across all 4 folds.
