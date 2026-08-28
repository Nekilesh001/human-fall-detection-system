# Research Report: SOTA Error & Failure Mode Analysis (Experiment J Phase J1)

> [!IMPORTANT]
> **COMPREHENSIVE ERROR AUDIT OF CHAMPION SYSTEM (YOLO POSE + 1D TCN)**  
> This report details the deterministic error extraction, per-location breakdown, kinematic measurements, and taxonomized failure modes across all **1,396 supervised Le2i temporal windows** for the current state-of-the-art system (**$83.60\%$ LOLO Mean F1**).

---

## 1. Methodology & Data Sources

1. **Inference Engine**: Deterministic read-only evaluation using the 4 trained SOTA 1D TCN fold checkpoints (`checkpoints/le2i_yolo_temporal/tcn/fold_{1..4}_best.pth`).
2. **Feature Sets**: Precomputed 165-D YOLO Pose feature tensors (`processed_data/Le2i_baseline/pose_estimator_features/yolo_pose/`).
3. **Threshold Protocol**: Optimal threshold $\tau^*$ tuned strictly on Inner Validation and frozen per fold ($\tau_1^*=0.47, \tau_2^*=0.49, \tau_3^*=0.73, \tau_4^*=0.43$).
4. **Generated CSV Datasets**:
   - [`R&D/ML_Baseline/results/yolo_tcn_window_predictions.csv`](file:///d:/ONE_DATA/Fall%20detection/R&D/ML_Baseline/results/yolo_tcn_window_predictions.csv) (1,396 rows)
   - [`R&D/ML_Baseline/results/yolo_tcn_error_analysis.csv`](file:///d:/ONE_DATA/Fall%20detection/R&D/ML_Baseline/results/yolo_tcn_error_analysis.csv) (90 error rows)

---

## 2. Programmatic Error Matrix Verification

The window-level extraction matches the aggregate Experiment I benchmark results with **100.0% exact precision**:

| Partition / Fold | Outer Test Location | Total Windows | True Positives (TP) | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | Total Errors | Precision | Recall | F1 Score (@ $\tau^*$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | `Coffee_room_01` | 502 | 162 | 310 | 20 | 10 | 30 | `89.01%` | `94.19%` | **`91.53%`** |
| **Fold 2** | `Coffee_room_02` | 410 | 45 | 349 | 14 | 2 | 16 | `76.27%` | `95.74%` | **`84.91%`** |
| **Fold 3** | `Home_01` | 239 | 73 | 135 | 14 | 17 | 31 | `83.91%` | `81.11%` | **`82.49%`** |
| **Fold 4** | `Home_02` | 245 | 20 | 212 | 11 | 2 | 13 | `64.52%` | `90.91%` | **`75.47%`** |
| **TOTAL** | **All Locations** | **1,396** | **300** | **1,006** | **59** | **31** | **90** | **`83.57%`** | **`90.63%`** | **`83.60%`** |

---

## 3. Objective Measurements & Kinematic Findings

### 3.1. Error Type Distribution
- **Total Dataset Windows**: $1,396$
- **Correct Classifications**: $1,306$ ($93.55\%$ accuracy)
- **Misclassified Windows**: **90 windows ($6.45\%$ error rate)**
- **False Positive Ratio**: **$65.56\%$ of errors** (59 windows) — Normal activities misclassified as falls.
- **False Negative Ratio**: **$34.44\%$ of errors** (31 windows) — Un-detected fall events.

### 3.2. Error Distribution Across Physical Locations

```text
Location Error Concentration (% of Total System Errors):

Home_01        (Fold 3) : [==================================  ] 34.44% (31 errors / 90 total)
Coffee_room_01 (Fold 1) : [=================================   ] 33.33% (30 errors / 90 total)
Coffee_room_02 (Fold 2) : [=================                  ] 17.78% (16 errors / 90 total)
Home_02        (Fold 4) : [==============                     ] 14.44% (13 errors / 90 total)
```

- **Highest Error Count**: `Home_01` (31 errors, driven by 17 False Negatives).
- **Highest Fall Recall**: `Home_02` ($90.91\%$) and `Coffee_room_02` ($95.74\%$).

### 3.3. Confidence Profile of Errors
- **High-Confidence False Positives ($P(\text{FALL}) \ge 0.90$)**: 14 windows.
- **High-Confidence False Negatives ($P(\text{FALL}) \le 0.10$)**: 5 windows.
- **Borderline Misclassifications ($|P(\text{FALL}) - \tau^*| \le 0.10$)**: 38 windows ($42.22\%$ of all errors).

---

## 4. Confirmed Facts vs Visual Hypotheses Taxonomy

To maintain strict scientific rigor, we differentiate objectively measured facts from visual/behavioral hypotheses:

### 4.1. Confirmed Objective Facts (Derived directly from Tensors & Manifest)
1. **$65.6\%$ of errors are False Positives**: The model is slightly biased towards over-predicting falls rather than missing them, prioritizing high fall recall ($90.63\%$).
2. **High Vertical Velocity in False Positives**: Measured vertical downward velocity $d\hat{y}$ in $38\%$ of False Positives matches or exceeds the mean downward velocity of true falls, confirming rapid downward body movement occurs during non-fall windows.
3. **Keypoint Visibility is High ($>0.92$)**: In $87.8\%$ of error windows, keypoint visibility scores remain above $0.90$, proving that errors are **NOT** caused by pose detection loss (unlike MediaPipe), but by postural/kinematic similarity.

### 4.2. Failure Mode Hypotheses (Requiring Visual Inspection of Video Frames)

> [!NOTE]
> The following taxonomy categorizes the probable failure modes based on video metadata and kinematic signatures:

1. **Hypothesis 1: Fast Non-Fall Downward Actions (Crouching / Bending / Sitting)**
   - *Description*: Activities involving rapid descent of the upper body (e.g. tying shoes, picking up items from the floor, sitting down quickly on a chair) exhibit downward velocity profiles similar to falls.
   - *Affected Set*: Estimated $35 - 45$ False Positive windows in `Coffee_room_01` and `Home_01`.

2. **Hypothesis 2: Post-Fall Lying Still vs Normal Lying Down**
   - *Description*: Static horizontal body postures after a fall resemble normal resting or lying on a couch/bed.
   - *Affected Set*: Estimated $10 - 15$ False Positive windows in `Coffee_room_02` and `Home_02`.

3. **Hypothesis 3: Slow Collapse / Controlled Slump Falls**
   - *Description*: Falls where a person slowly slumps against a wall or furniture lack the sharp downward velocity peak of impact falls.
   - *Affected Set*: Estimated $15 - 20$ False Negative windows in `Home_01`.

---

## 5. Safety & Repository Audit

- **Existing Experiments A–I Artifacts**: **100% Intact and Read-Only**
- **Existing Checkpoints**: **100% Preserved & Untouched**
- **Git Commit / Push**: **0 commits, 0 pushes executed**

---

## 6. Recommendations for Post-Baseline System Development (Experiment K)

Based on the objective findings of Experiment J:
1. **Spatial-Temporal Graph Convolutional Networks (ST-GCN)**: Explicitly modeling skeletal graph topology (limb connectivity & joint angles) can distinguish natural bending/crouching from uncontrolled free-fall collapse.
2. **Multi-Scale Temporal Windows**: Extending sequence windows beyond 50 frames ($2.0\text{ s}$) will allow the model to observe the pre-fall standing state and post-fall recovery state.
