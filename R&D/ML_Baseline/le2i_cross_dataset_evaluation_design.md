# Research Design: Le2i Cross-Dataset Generalization Protocol

> [!IMPORTANT]
> **DESIGN ONLY — NO LE2I PREPROCESSING OR TRAINING PERFORMED.**
> This document specifies the scientific protocol, data scope, feature pipeline, and evaluation metrics for assessing zero-shot cross-dataset generalization of the frozen URFD RGB baseline model on the Le2i dataset.

---

## 1. Objective
To evaluate the zero-shot cross-dataset generalization capability of the completed, frozen URFD RGB baseline model (`checkpoints/urfd_rgb_baseline_best.pth`) when tested directly on unseen environments from the Le2i Fall Detection Dataset without retraining, fine-tuning, threshold re-tuning, or target-domain adaptation.

---

## 2. Scientific Question
The core scientific questions addressed in this phase are:

1. **Primary Question**: *"Does a fall detection model trained exclusively on URFD RGB temporal windows generalize to completely unseen real-world environments (Le2i) without retraining or adaptation?"*
2. **Secondary Question**: *"What is the magnitude of performance degradation caused by domain shift (scene background, lighting, camera perspectives, actor variability, and occlusions) when moving from a single laboratory setting (URFD) to diverse residential/office settings (Le2i)?"*

---

## 3. URFD Reference Model (Frozen Baseline Contract)
The reference model is strictly frozen and immutable:

```text
RGB Temporal Window: (B, 50, 3, 240, 320) @ 25 FPS
        │
Frozen Pretrained ResNet-18: (B, 50, 512)
        │
Temporal Mean + Standard Deviation Pooling: (B, 1024)
        │
Linear(1024 → 64) ──► ReLU ──► Dropout(p=0.5) ──► Linear(64 → 2)
```

- **Frozen Weights File**: `checkpoints/urfd_rgb_baseline_best.pth`
- **Trainable Parameters**: 65,730 (0 parameters updated during evaluation)
- **Reference Decision Thresholds**:
  - Default Threshold: $\tau = 0.50$
  - URFD Validation-Selected Threshold: $\tau^* = 0.10$
- **URFD Baseline Performance Benchmark**: $100.0\%$ Accuracy, $100.0\%$ Sensitivity, $100.0\%$ Specificity, $100.0\%$ F1 Score ($\text{Confusion Matrix} = [[33, 0], [0, 24]]$).

---

## 4. Le2i Dataset Scope & Labeled Data Selection
Based on the canonical manifest audit (`R&D/Dataset_Analysis/manifest_audit_report.md`):

- **Total Le2i Video Events**: 190 videos
- **Verified Supervised Labeled Videos**: **127 videos**
  - **FALL Videos**: **96 videos** (with verified `f_start` and `f_end` frame annotations)
  - **NORMAL Videos**: **31 videos** (ADL videos without falls)
- **Location Distribution of Supervised Data**:
  - `Coffee_room_01`: 47 FALL, 0 NORMAL (47 total)
  - `Coffee_room_02`: 12 FALL, 8 NORMAL (20 total)
  - `Home_01`: 30 FALL, 0 NORMAL (30 total)
  - `Home_02`: 7 FALL, 23 NORMAL (30 total)

---

## 5. UNKNOWN Record Handling
- **Excluded UNKNOWN Records**: **63 videos**
  - **Unannotated Office & Lecture Room Videos**: 60 videos (`Office`: 30, `Lecture_room`: 30) lacking bounding-box / temporal ground-truth files.
  - **Malformed Annotation Records**: 3 videos with corrupted annotation files (`Coffee_room_01/v26`, `Coffee_room_02/v50`, `Coffee_room_02/v52`).
- **Strict Rule**: All 63 UNKNOWN records **MUST remain strictly excluded** from evaluation. They must NOT be converted to FALL or NORMAL to prevent ground-truth corruption.

---

## 6. Zero-Shot Protocol vs. Supervised LOLO Protocol

To maintain scientific rigor, two distinct experiments are defined:

### Experiment A: Zero-Shot Cross-Dataset Evaluation (URFD $\to$ Le2i) — **FIRST**
- **Model**: Frozen URFD-trained baseline model (`urfd_rgb_baseline_best.pth`).
- **Training Data**: None (0 Le2i samples used for training or validation).
- **Threshold**: Fixed at URFD $\tau^* = 0.10$ and default $\tau = 0.50$.
- **Evaluation Scope**: All 127 supervised Le2i videos evaluated directly.
- **Purpose**: Measure raw zero-shot cross-dataset domain transfer.

### Experiment B: Supervised In-Domain LOLO Evaluation (Le2i $\to$ Le2i) — **SECOND**
- **Model**: Newly initialized baseline architecture trained on Le2i.
- **Protocol**: 4-Fold Leave-One-Location-Out (LOLO) cross-validation (holding out `Coffee_room_01`, `Coffee_room_02`, `Home_01`, `Home_02` sequentially).
- **Purpose**: Measure in-domain scene generalization when trained on Le2i environment variations.

> [!NOTE]
> Experiment A is a **pure zero-shot evaluation**. LOLO splitting is NOT used for Experiment A because no model training or validation takes place on Le2i. However, Experiment A results will be reported both overall and broken down by Le2i location to analyze location-specific domain shift.

---

## 7. Temporal Windowing & Phase Label Mapping

### Frame-Level Phase Definitions for Labeled Fall Videos
For each verified FALL video with start frame $f_{\text{start}}$ and end frame $f_{\text{end}}$:
1. **Pre-Fall Phase**: Frames $[1, f_{\text{start}} - 1]$ (Normal ADL movement prior to loss of balance).
2. **Active Fall Phase**: Frames $[f_{\text{start}}, f_{\text{end}}]$ (Dynamic fall descent).
3. **Post-Fall Phase**: Frames $[f_{\text{end}} + 1, N_{\text{total}}]$ (Person on floor / collapsed posture).

### 50-Frame Temporal Windowing Parameters
- **Window Length ($W$)**: 50 frames (2.0 seconds at 25 FPS)
- **Stride ($S$)**: 25 frames (1.0 second overlap)

### Deterministic Window Label Mapping Rules
For a window $w = [f_{\text{win\_start}}, f_{\text{win\_end}}]$:

1. **NORMAL Videos**: All windows $\to$ **NORMAL (0)**.
2. **FALL Videos**:
   - **Pre-Fall Windows** ($f_{\text{win\_end}} < f_{\text{start}}$): $\to$ **NORMAL (0)**.
   - **Active Fall Windows** ($w \cap [f_{\text{start}}, f_{\text{end}}] \ge 10 \text{ frames}$): $\to$ **FALL (1)**.
   - **Post-Fall Windows** ($f_{\text{win\_start}} > f_{\text{end}}$): $\to$ **FALL (1)**. (Post-fall lying on floor represents the hazardous outcome state, mapped to FALL class consistent with binary fall monitoring).
   - **Transition Windows** ($f_{\text{win\_start}} < f_{\text{start}} \le f_{\text{win\_end}}$ with $< 10$ active fall frames): Classified as **NORMAL (0)** if $< 5$ fall frames, or excluded from ambiguous boundary evaluation.

---

## 8. Spatial & Temporal Standardization Protocol

To ensure 100% input compatibility with the frozen URFD ResNet-18 model:

1. **Frame Rate Resampling**: Resample all Le2i videos to **25 FPS** (using Lanczos temporal interpolation if source FPS differs).
2. **Spatial Standardization**:
   - `Coffee_room_01`, `Coffee_room_02`, `Home_01` ($320 \times 240$ native): Direct Lanczos spatial resize to $320 \times 240$.
   - `Home_02` ($320 \times 180$ 16:9 native): Apply **Vertical Zero-Padding**:
     $$\text{Top Padding} = 30 \text{ px (black)}, \quad \text{Bottom Padding} = 30 \text{ px (black)}$$
     Achieves exact $320 \times 240$ resolution while preserving true anatomical aspect ratio and human body proportions. **Stretching ($320 \times 180 \to 320 \times 240$) is strictly prohibited.**
3. **Channel & Normalization**: Convert uint8 to float32 $[0.0, 1.0]$, apply standard ImageNet mean $[0.485, 0.456, 0.406]$ and std $[0.229, 0.224, 0.225]$.

---

## 9. Feature Extraction & Forward Inference Path

```text
Le2i Preprocessed Temporal Window: (50, 3, 240, 320) @ 25 FPS
        │
ImageNet Pretrained ResNet-18 (Frozen): (50, 512)
        │
Precomputed / On-the-fly Features: (50, 512) Float32
        │
Temporal Mean + Standard Deviation Pooling: (1024)
        │
Frozen URFD MLP Classifier (checkpoints/urfd_rgb_baseline_best.pth): (2)
        │
Softmax Probability: P(FALL)
```

No adaptation layers, re-calibration factors, or feature transformations are permitted.

---

## 10. Evaluation Metrics Suite

### Window-Level Metrics
- **Accuracy**: Proportion of correctly classified windows.
- **Precision**: $\frac{\text{TP}}{\text{TP} + \text{FP}}$
- **Recall / Sensitivity**: $\frac{\text{TP}}{\text{TP} + \text{FN}}$
- **Specificity**: $\frac{\text{TN}}{\text{TN} + \text{FP}}$
- **F1 Score**: Harmonic mean of Precision and Recall.
- **Confusion Matrix**: $[[\text{TN}, \text{FP}], [\text{FN}, \text{TP}]]$.

### Event-Level & Location-Level Reporting
- **Event Sensitivity**: Percentage of 96 FALL videos where at least one window during the active/post-fall phase is predicted as FALL ($P(\text{FALL}) \ge \tau$).
- **Location Breakdown**: Separate window-level metrics reported for `Coffee_room_01`, `Coffee_room_02`, `Home_01`, and `Home_02` to analyze scene-specific domain shift.
- **Aggregation**: Both **Micro-average** (overall pooled window metrics) and **Macro-average** (unweighted mean across locations) will be reported.

---

## 11. Time-to-Detection ($\Delta t$) Specification

For each verified FALL video, Time-to-Detection ($\Delta t$) measures the temporal latency between actual fall onset and first model alert:

$$\Delta t = \frac{f_{\text{pred\_first}} - f_{\text{start}}}{\text{FPS}} \quad (\text{seconds})$$

where $f_{\text{pred\_first}}$ is the start frame of the earliest temporal window predicting $P(\text{FALL}) \ge \tau^*$.

- $\Delta t \le 0.0\text{s}$: Prompt detection during fall descent.
- $\Delta t > 0.0\text{s}$: Delayed detection post-descent.
- Undetected: No window predicted as FALL during the video ($\text{FN}$ event).

---

## 12. Leakage & Isolation Controls

1. **Zero Weight Updates**: Model weights (`urfd_rgb_baseline_best.pth`) remain 100% read-only.
2. **Zero Threshold Tuning**: Threshold $\tau^* = 0.10$ and default $\tau = 0.50$ are fixed prior to Le2i evaluation.
3. **Zero Parameter Adaptation**: Class weights, normalizations, and network architectures remain unchanged.
4. **Strict UNKNOWN Exclusion**: 63 UNKNOWN records are excluded from evaluation.

---

## 13. Domain Shift Breakdown & Failure Mode Analysis

A degradation in performance moving from URFD (100%) to Le2i zero-shot evaluation is expected due to fundamental domain shifts:

1. **Background & Lighting Shift**: Real home/office environments with shadows, dynamic lighting, furniture, and reflective surfaces vs. URFD's controlled lab setting.
2. **Camera Angles & Height**: Wall-mounted high-angle views in Le2i vs. eye-level tripods in URFD.
3. **Occlusions**: Body parts partially obscured by desks, chairs, beds, or sofas during falls in Le2i.
4. **Actor & Clothing Variability**: Multiple diverse actors wearing varied, loose clothing vs. standard gym attire in URFD.
5. **Complex ADLs**: Confounding normal activities such as tying shoes, picking up items, sitting on floors, or bending over.

---

## 14. Recommended Execution Order

1. **Step 1**: Preprocess the 127 supervised Le2i videos into standard temporal windows ($W=50, S=25, 25\text{ FPS}, 320 \times 240$ with vertical padding for `Home_02`).
2. **Step 2**: Execute **Experiment A (Zero-Shot URFD $\to$ Le2i Cross-Dataset Evaluation)** using frozen model `checkpoints/urfd_rgb_baseline_best.pth`.
3. **Step 3**: Compile zero-shot evaluation report and domain shift breakdown.
4. **Step 4** *(Future Milestone)*: Execute **Experiment B (Le2i Supervised 4-Fold LOLO Training & Evaluation)** to establish in-domain Le2i benchmark.

---

## 15. Implementation Plan

- Create `src/preprocess_le2i.py`: Standardizes 127 supervised videos, applies padding to `Home_02`, extracts 50-frame windows, and generates `processed_data/Le2i_baseline/processed_manifest.csv`.
- Create `src/precompute_le2i_features.py`: Extracts ResNet-18 features for Le2i windows.
- Create `src/evaluate_le2i_zeroshot.py`: Evaluates frozen URFD model on Le2i features and outputs zero-shot report artifact `R&D/ML_Baseline/le2i_zeroshot_evaluation_report.md`.

---

## 16. Stop Conditions

- **NO preprocessing, window generation, feature extraction, or model evaluation will be performed during this turn.**
- Execution will pause for explicit approval of this research protocol design.
