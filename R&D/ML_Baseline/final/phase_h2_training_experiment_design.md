# Phase H2 — Research Training Experiment Design (Experiments A–E)

> [!IMPORTANT]
> **RESEARCH SPECIFICATION ONLY — NO MODEL TRAINING EXECUTED**  
> Baseline Model Path: `checkpoints/final_k1/final_production.pth`  
> Checkpoint SHA256: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d` (**IMMUTABLE & FROZEN**)

---

## 1. Experiment Plan Overview

| Experiment ID | Training Dataset | Validation / Test Sets | Research Objective |
| :--- | :--- | :--- | :--- |
| **EXP-A (Baseline)** | Le2i Only (Frozen K1) | Le2i Test Split | Measure baseline Le2i performance ($\text{F1} \approx 86.60\%$). |
| **EXP-B** | Le2i + URFD | Le2i + URFD | Evaluate dual-dataset generalization. |
| **EXP-C** | Le2i + Multicam | Le2i + Multicam | Evaluate multi-camera angle robustness. |
| **EXP-D (Unified)** | Le2i + URFD + Multicam | Grouped 5-Fold Split | Evaluate unified 3-dataset candidate model. |
| **EXP-E (Out-of-Distribution)** | Le2i + URFD | Multicam (Zero-Shot) | Measure cross-environment zero-shot generalization. |

---

## 2. Recommended Training Configuration

- **Script Target**: [`src/train_multi_dataset_k1.py`](file:///d:/ONE_DATA/Fall%20detection/src/train_multi_dataset_k1.py)
- **Output Checkpoint Directory**: `checkpoints/multi_dataset_k1/` (NEVER overwrites `final_production.pth`).
- **Optimizer**: AdamW (`learning_rate = 1e-3`, `weight_decay = 1e-4`).
- **Loss Function**: Weighted Binary Cross-Entropy Loss ($\text{pos\_weight} = 4.0$).
- **Batch Size**: 32
- **Epochs**: 30 (with Early Stopping patience = 7 epochs).
- **Random Seed**: 42 (Deterministic execution).

---

## 3. Evaluation & Promotion Criteria

The candidate multi-dataset model trained under **EXP-D** will be evaluated against frozen baseline K1.

### Baseline K1 Benchmarks
- LOLO Mean F1 @ $\tau = 0.50$: **86.60%**
- Inner-selected threshold Mean F1: **84.98% ± 5.81%**
- Production threshold: $\tau = 0.3650$

### Promotion Criteria for New Production Model
1. **Recall**: Must achieve $\ge 92.0\%$ Fall Recall across unified test sets.
2. **False Negatives**: Must reduce false negatives compared to Le2i-only K1.
3. **Cross-Dataset Robustness**: Must maintain $\ge 80.0\%$ F1 on out-of-distribution environments.
4. **Latency**: Single-person inference latency must remain $< 10.0\text{ ms}$ ($\ge 100\text{ FPS}$).

---

## 🔒 Mandatory Final Confirmation

- **No training was executed during this audit phase.**
- **Production model K1 remains frozen and active.**
