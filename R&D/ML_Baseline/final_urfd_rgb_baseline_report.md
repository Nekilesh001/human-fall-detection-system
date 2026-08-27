# Final URFD RGB Baseline Model Training & Benchmark Report

## 1. Experimental Objective
The objective of this experiment is to establish the first complete, leakage-safe ML baseline for the Human Fall Detection System using the URFD RGB dataset. The baseline evaluates an ImageNet-pretrained ResNet-18 spatial feature extractor combined with temporal mean-std pooling and a 2-layer MLP classifier under strict event-level cross-validation split boundaries.

---

## 2. Dataset Description
- **Dataset**: URFD (University of Rzeszów Fall Detection Dataset)
- **Modality**: RGB Only
- **Effective Scope**: 67 usable events (3 missing events skipped during preprocessing due to < 50 frames duration: `fall-16`, `fall-21`, `fall-22`)
- **Total Processed Windows**: 360 temporal windows ($W=50$ frames, $S=25$ stride, 25 FPS, $320 \times 240$ spatial resolution)
- **Class Breakdown**: 118 FALL windows (32.8%), 242 NORMAL windows (67.2%)

---

## 3. Split Description (Leakage-Safe Event-Level Split)
Partition assignments were derived strictly at the event level with seed 42:
- **Train Partition**: 47 events | 260 windows (84 FALL, 176 NORMAL)
- **Validation Partition**: 9 events | 43 windows (10 FALL, 33 NORMAL)
- **Test Partition**: 11 events | 57 windows (24 FALL, 33 NORMAL)
- **Disjointness Audit**: $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$ (0 event or camera leakage).

---

## 4. Feature Representation
- **Spatial Backbone**: ImageNet-pretrained ResNet-18 (`resnet.fc = nn.Identity()`)
- **Feature Matrix per Window**: $(50, 512)$ float32 embeddings
- **Precomputation Mode**: Two-stage precomputation pipeline saved to `processed_data/URFD_RGB_baseline/features/` (32.26 MB total storage, 91.75 KB/file). Bit-exact numerical equivalence verified ($\Delta_{\text{max}} = 0.0$).

---

## 5. Model Architecture & Parameters
```text
Input Feature Tensor: (B, 50, 512)
        │
Temporal Mean + Std Pooling: (B, 1024)
        │
Linear(1024 → 64) ──► ReLU ──► Dropout(p=0.5) ──► Linear(64 → 2)
```
- **Total Parameters**: 65,730
- **Frozen Parameters**: 0 (Backbone ResNet-18 parameters precomputed)
- **Trainable Parameters**: **65,730**

---

## 6. Training Configuration & Reproducibility
- **Optimizer**: AdamW (`lr=0.001`, `weight_decay=0.01`)
- **Scheduler**: CosineAnnealingLR (`T_max=50`, `eta_min=1e-5`)
- **Epochs**: 50
- **Batch Size**: 8
- **Seed**: 42 (`random`, `numpy`, `torch`, `cudnn.deterministic=True`)

---

## 7. Class Weighting
Calculated strictly from the 260 Train windows:
$$w_{\text{normal}} = \frac{260}{2 \times 176} \approx 0.738636, \quad w_{\text{fall}} = \frac{260}{2 \times 84} \approx 1.547619$$

---

## 8. Checkpoint Selection Rule
Model checkpoints were evaluated strictly on the Validation set at the end of each epoch:
$$\text{Score}_{\text{val}} = \begin{cases} \text{F1}_{\text{val}} & \text{if } \text{Specificity}_{\text{val}} \ge 0.75 \\ \text{F1}_{\text{val}} \times 0.5 & \text{otherwise} \end{cases}$$
- **Best Model Epoch**: Epoch 2 (Validation Score: `1.0000`)
- **Best Checkpoint Saved**: `checkpoints/urfd_rgb_baseline_best.pth`

---

## 9. Validation Threshold Selection ($\tau^*$)
Grid search across $\tau \in [0.10, 0.90]$ (step 0.05) evaluated on Validation set using the best checkpoint:
- **Default Threshold**: $\tau = 0.50$
- **Selected Optimal Threshold**: $\tau^* = 0.10$ (Validation F1 = `1.0000`)

---

## 10. Final Test Evaluation Results

### A. Evaluation @ Default Threshold ($\tau = 0.50$)
- **Accuracy**: `1.0000` (100.0%)
- **Precision**: `1.0000` (100.0%)
- **Recall / Sensitivity**: `1.0000` (100.0%)
- **Specificity**: `1.0000` (100.0%)
- **F1 Score**: `1.0000` (1.000)
- **Confusion Matrix**: `[[33, 0], [0, 24]]` (TN: 33, FP: 0, FN: 0, TP: 24)

### B. Evaluation @ Validation-Selected Threshold ($\tau^* = 0.10$)
- **Accuracy**: `1.0000` (100.0%)
- **Precision**: `1.0000` (100.0%)
- **Recall / Sensitivity**: `1.0000` (100.0%)
- **Specificity**: `1.0000` (100.0%)
- **F1 Score**: `1.0000` (1.000)
- **Confusion Matrix**: `[[33, 0], [0, 24]]` (TN: 33, FP: 0, FN: 0, TP: 24)

---

## 11. Latency & Throughput Performance
- **Inference Latency**: **0.16 ms / window** (50-frame temporal window)
- **Throughput**: **312,550.6 FPS** (equivalent frame processing throughput)

---

## 12. Interpretation & Clinical Deployment Boundaries
> [!WARNING]
> **IMPORTANT NON-CLINICAL DISCLAIMER**: High validation/test accuracy on URFD does **NOT** indicate hospital or clinical deployment readiness.
> 
> 1. **Dataset Limitations**: URFD consists of staged fall events performed by healthy actors in controlled indoor environments under consistent lighting.
> 2. **False Alarms per Camera-Hour**: Continuous real-world ward monitoring requires establishing a False Alarm Rate (FAR) per camera-hour across multi-hour non-fall activities (e.g., tying shoes, picking up items, lying down intentionally). Staged event datasets cannot establish FAR.
> 3. **Modality Constraints**: RGB-only models are sensitive to lighting shifts, occlusions, patient blankets, and privacy concerns in real clinical wards.

---

## 13. Reproducibility & Artifact Index
- **Training Log CSV**: `R&D/ML_Baseline/results/training_history.csv`
- **Validation Threshold Grid CSV**: `R&D/ML_Baseline/results/validation_threshold_analysis.csv`
- **Final Test Metrics JSON**: `R&D/ML_Baseline/results/final_test_metrics.json`
- **Experiment Config JSON**: `R&D/ML_Baseline/results/experiment_config.json`
- **Best Model Checkpoint**: `checkpoints/urfd_rgb_baseline_best.pth`
