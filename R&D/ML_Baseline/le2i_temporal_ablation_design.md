# Research Design: Le2i Temporal Representation Ablation Protocol (Experiment C)

> [!IMPORTANT]
> **DESIGN ONLY — NO TRAINING PERFORMED YET.**
> This document specifies the scientific protocol, model variants, parameter counts, and evaluation metrics for Experiment C: Le2i Temporal Representation Ablation.

---

## 1. Objective
To evaluate whether explicit temporal sequence modeling improves cross-location generalization on the Le2i dataset by systematically comparing three controlled temporal aggregation/modeling variants under the exact same 4-Fold Leave-One-Location-Out (LOLO) cross-validation protocol.

---

## 2. Scientific Question
*"Does preserving explicit temporal frame order (via a sequential GRU) improve in-domain cross-location fall detection performance on Le2i compared to static temporal pooling (Mean and Mean+Std)?"*

### Primary Scientific Hypotheses
1. **Hypothesis A (Sequence Modeling Benefit)**: Preserving sequential frame dynamics (standing $\to$ descent $\to$ impact $\to$ floor pose) allows the classifier to reject static background structures and non-fall ADLs.
2. **Hypothesis B (Static Feature Sufficiency)**: Mean/Std temporal statistics already capture sufficient feature variance, and sequential modeling provides no additional cross-location benefit.
3. **Hypothesis C (Overfitting Risk)**: Sequential modeling on limited physical events ($N=127$ videos) increases model complexity and overfits to training location dynamics, reducing unseen location test performance.

---

## 3. Controlled Model Variants

All three variants consume the exact same precomputed ResNet-18 feature tensors $(B, 50, 512)$ float32 extracted from the 1,396 preprocessed Le2i windows. ResNet-18 is 100% frozen.

```text
Model A (Mean-Only):    (B, 50, 512) ──► Temporal Mean (512) ──► Linear(512→64) ──► ReLU ──► Dropout(0.5) ──► Linear(64→2)
Model B (Mean+Std):     (B, 50, 512) ──► Temporal Mean+Std (1024) ──► Linear(1024→64) ──► ReLU ──► Dropout(0.5) ──► Linear(64→2)
Model C (1-Layer GRU):  (B, 50, 512) ──► GRU(512→64, 1-layer) ──► Final Hidden (64) ──► Linear(64→32) ──► ReLU ──► Dropout(0.5) ──► Linear(32→2)
```

| Model Variant | Temporal Aggregation / Modeling | Feature Dimension | Classifier Architecture | Trainable Parameters | Role in Ablation |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Model A** | Temporal Mean over 50 frames | $(B, 512)$ | `Linear(512 -> 64) -> ReLU -> Dropout(0.5) -> Linear(64 -> 2)` | **32,962** | Single-Statistic Pooling Baseline |
| **Model B** | Temporal Mean + Standard Deviation | $(B, 1024)$ | `Linear(1024 -> 64) -> ReLU -> Dropout(0.5) -> Linear(64 -> 2)` | **65,730** | Primary Control (Experiment B Baseline) |
| **Model C** | 1-Layer Sequential GRU ($h=64$) | $(B, 64)$ | `Linear(64 -> 32) -> ReLU -> Dropout(0.5) -> Linear(32 -> 2)` | **113,122** | Explicit Sequential Temporal Model |

---

## 4. Dataset Scope & 4-Fold LOLO Protocol

- **Dataset**: 1,396 precomputed feature windows ($W=50, S=25, 25\text{ FPS}, 320 \times 240$).
- **Supervised Data Scope**: 127 verified supervised videos (96 FALL, 31 NORMAL).
- **Excluded Data Scope**: 63 UNKNOWN records remain 100% EXCLUDED.
- **Folds**:
  - Fold 1: Outer Test = `Coffee_room_01` (Train: `Coffee_room_02`, `Home_01`, `Home_02`)
  - Fold 2: Outer Test = `Coffee_room_02` (Train: `Coffee_room_01`, `Home_01`, `Home_02`)
  - Fold 3: Outer Test = `Home_01` (Train: `Coffee_room_01`, `Coffee_room_02`, `Home_02`)
  - Fold 4: Outer Test = `Home_02` (Train: `Coffee_room_01`, `Coffee_room_02`, `Home_01`)

---

## 5. Inner Validation & Checkpoint Selection Strategy

- **Outer Test Isolation**: Outer held-out test location is 100% unseen (0 test windows used for training, class weights, checkpoint selection, or threshold tuning).
- **Inner Event Split**: Inner Train (80% of outer train events) and Inner Validation (20% of outer train events) constructed with zero video event overlap ($\text{Inner Train Events} \cap \text{Inner Val Events} = \emptyset$).
- **Checkpoint Selection**: Best checkpoint for each model/fold saved based on highest Inner Validation F1 score ($\text{Val F1}$).
- **Threshold Selection**: Inner validation threshold $\tau^*_{\text{inner}}$ searched in $[0.05, 0.95]$ to maximize Inner Val F1.

---

## 6. Training Hyperparameters
- **Optimizer**: AdamW ($\text{lr} = 1\text{e-}3$, $\text{weight\_decay} = 1\text{e-}2$)
- **Loss Function**: Weighted CrossEntropyLoss with fold-specific class weights $[w_{\text{norm}}, w_{\text{fall}}]$
- **Epochs**: 50 max epochs (with inner validation checkpointing)
- **Batch Size**: 32
- **Random Seed**: `42` (applied to Python `random`, `numpy`, `torch`, and deterministic cuDNN)

---

## 7. Expected Results Artifacts & Reporting Plan
- Save separate model checkpoints under `checkpoints/le2i_temporal_ablation/{model_type}/fold_{i}_best.pth`.
- Save result metrics CSV under `R&D/ML_Baseline/results/le2i_temporal_ablation/ablation_results.csv`.
- Create canonical research report artifact: [`R&D/ML_Baseline/le2i_temporal_ablation_training_report.md`](file:///d:/ONE_DATA/Fall%20detection/R&D/ML_Baseline/le2i_temporal_ablation_training_report.md).
