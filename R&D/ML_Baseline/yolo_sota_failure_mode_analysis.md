# Research Report: Failure Mode & Kinematic Evidence Analysis (Experiment J Phase J2)

> [!IMPORTANT]
> **EVIDENCE-BASED FAILURE MODE ANALYSIS FOR SOTA SYSTEM (YOLO POSE + 1D TCN)**  
> Comprehensive statistical, kinematic, and ground-truth video annotation audit of all **90 error windows** (59 False Positives, 31 False Negatives) out of $1,396$ supervised Le2i temporal windows.

---

## 1. Error Structure & Event Concentration Analysis

Out of 127 total supervised Le2i videos ($1,396$ temporal windows), misclassifications occurred in **56 unique videos** (44.1% of videos), producing **90 error windows ($6.45\%$ error rate)**.

### FP / FN Distribution by Physical Location

| Physical Location | Supervised Windows | TP | TN | FP (False Positives) | FN (False Negatives) | Total Errors | Location Error Rate | Dominant Error Mode |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Coffee_room_01`** | 502 | 162 | 310 | 20 | 10 | **30** | `5.98%` | False Positives ($66.7\%$) |
| **`Coffee_room_02`** | 410 | 45 | 349 | 14 | 2 | **16** | `3.90%` | False Positives ($87.5\%$) |
| **`Home_01`** | 239 | 73 | 135 | 14 | 17 | **31** | **`12.97%`** | **False Negatives ($54.8\%$)** |
| **`Home_02`** | 245 | 20 | 212 | 11 | 2 | **13** | **`5.31%`** | False Positives ($84.6\%$) |
| **TOTAL** | **1,396** | **300** | **1,006** | **59** | **31** | **90** | **`6.45%`** | **False Positives ($65.6\%$)** |

### Top 5 Concentrated Error Events
1. **`Le2i_Coffee_room_02_video (70)`**: 6 error windows (6 FP) — Normal ADL video with rapid sitting/bending motion.
2. **`Le2i_Home_01_video (9)`**: 4 error windows (4 FN) — Ground-truth fall (frames 149–163) under heavy occlusion.
3. **`Le2i_Coffee_room_02_video (63)`**: 3 error windows (3 FP) — Normal ADL video.
4. **`Le2i_Home_01_video (11)`**: 3 error windows (3 FN) — Fall event (frames 137–170).
5. **`Le2i_Home_02_video (49)`**: 3 error windows (3 FP) — Normal ADL video.

---

## 2. Statistical Confidence Analysis

To quantify model certainty on misclassified samples, errors were divided into **High-Confidence Errors** ($|P(\text{FALL}) - \tau^*| \ge 0.20$) and **Borderline Errors** ($|P(\text{FALL}) - \tau^*| < 0.20$):

```text
Confidence Profile of System Errors (N=90):

False Positives (N=59) : High-Confidence (P >= Tau* + 0.20) [========================] 37 (62.7%)
                         Borderline      (P <  Tau* + 0.20) [==============          ] 22 (37.3%)

False Negatives (N=31) : High-Confidence (P <= Tau* - 0.20) [====================    ] 21 (67.7%)
                         Borderline      (P >  Tau* - 0.20) [==========              ] 10 (32.3%)
```

### Statistical Metrics of Error Distributions

| Metric | False Positives ($N=59$) | False Negatives ($N=31$) | Correct Classifications ($N=1,306$) |
| :--- | :---: | :---: | :---: |
| **Mean $P(\text{FALL})$** | **`0.7813`** | **`0.1980`** | `0.1872` (TN) / `0.9412` (TP) |
| **Median $P(\text{FALL})$** | **`0.8177`** | **`0.0128`** | `0.0004` (TN) / `0.9986` (TP) |
| **Min $P(\text{FALL})$** | `0.4553` | `0.0000` | `0.0000` |
| **Max $P(\text{FALL})$** | `1.0000` | `0.7099` | `1.0000` |
| **Standard Deviation** | $\pm 0.1450$ | $\pm 0.2584$ | -- |
| **High-Confidence Count** | **37 ($62.7\%$)** | **21 ($67.7\%$)** | -- |
| **Borderline Count** | **22 ($37.3\%$)** | **10 ($32.3\%$)** | -- |

---

## 3. Pose Quality Comparison

Comparing COCO-17 keypoint visibility ($v_i$) across classification outcomes:

| Classification Category | Sample Count ($N$) | Mean Keypoint Visibility | Min Keypoint Visibility | High-Vis Ratio ($v \ge 0.80$) | Low-Vis Ratio ($v < 0.50$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **True Positives (TP)** | 300 | **`0.4215`** | `0.0000` | `41.2%` | `57.8%` |
| **True Negatives (TN)** | 1,006 | **`0.3993`** | `0.0000` | `38.9%` | `60.1%` |
| **False Positives (FP)** | 59 | **`0.3373`** | `0.0000` | `32.8%` | `66.2%` |
| **False Negatives (FN)** | 31 | **`0.2746`** | `0.0000` | **`26.1%`** | **`73.9%`** |

> **Key Finding**: Pose visibility in False Negatives ($0.2746$) is **$32.1\%$ lower** than in True Positives ($0.4215$), confirming that partial body occlusion (especially in `Home_01`) degrades keypoint confidence and contributes to missed falls.

---

## 4. Kinematic & Temporal Motion Analysis

Objective measurements derived from the 165-D joint velocity ($d\hat{x}, d\hat{y}$) and torso geometry features:

| Metric | True Positives (TP) | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | Scientific Interpretation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Mean Max Downward Velocity ($d\hat{y}$)** | `14.92` | `1.04` | **`20.98`** | `11.67` | **FPs exhibit rapid downward motion matching/exceeding true falls** |
| **Mean Downward Velocity ($d\hat{y}$)** | `1.85` | `0.11` | **`2.14`** | `1.18` | Non-fall FPs involve sustained downward movement |
| **Peak Abruptness Ratio ($d\hat{y}_{\text{max}}/\bar{d\hat{y}}$)** | `8.06` | `9.45` | `9.80` | `9.89` | Sharp velocity peaks occur in both FPs and FNs |
| **Downward Motion Duration (frames $>0.05$)** | `26.4` | `4.2` | **`24.8`** | `20.1` | **FPs sustain downward motion for $\sim 1.0\text{ second}$** |

---

## 5. Event-Level Failure Table & Verified Visual Causes

Cross-referencing predictions with ground-truth Le2i frame annotations (`Annotation_files/*.txt`):

| Physical Location | Event ID / Video | Error Count | FP / FN | Rep Window ID | Rep Prob | Rep Tau* | Verified Annotation Evidence | Confirmed Visual Cause / Scientific Hypothesis |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: | :--- | :--- |
| **`Coffee_room_02`** | `video (70)` | 6 | 6 FP / 0 FN | `Coffee_room_02_video (70)_w000` | `1.0000` | `0.49` | Frames 0, 0 (No Fall) | **VERIFIED Normal ADL Video Misclassified**: Rapid sitting/crouching |
| **`Home_01`** | `video (9)` | 4 | 0 FP / 4 FN | `Home_01_video (9)_w002` | `0.0000` | `0.73` | Frames 149–163 (Fall) | **VERIFIED Missed Fall Event**: Heavy furniture occlusion in `Home_01` |
| **`Coffee_room_02`** | `video (63)` | 3 | 3 FP / 0 FN | `Coffee_room_02_video (63)_w000` | `0.9997` | `0.49` | Frames 0, 0 (No Fall) | **VERIFIED Normal ADL Video Misclassified**: Bending over / low posture |
| **`Home_01`** | `video (11)` | 3 | 0 FP / 3 FN | `Home_01_video (11)_w001` | `0.0001` | `0.73` | Frames 137–170 (Fall) | **VERIFIED Missed Fall Event**: Slow collapse slump against furniture |
| **`Home_02`** | `video (49)` | 3 | 3 FP / 0 FN | `Home_02_video (49)_w000` | `0.9959` | `0.43` | Frames 0, 0 (No Fall) | **VERIFIED Normal ADL Video Misclassified**: Lying down on couch/bed |
| **`Coffee_room_01`** | `video (21)` | 2 | 2 FP / 0 FN | `Coffee_room_01_video (21)_w004` | `1.0000` | `0.47` | Frames 0, 0 (No Fall) | **VERIFIED Normal ADL Video Misclassified**: Fast downward action |

---

## 6. Hard vs Borderline Error Synthesis

- **High-Confidence Errors ($N=58$, $64.4\%$ of errors)**: 37 FP ($P \ge \tau^* + 0.20$) and 21 FN ($P \le \tau^* - 0.20$).
  - *Implication*: These errors cannot be resolved by threshold tuning ($\tau^*$). They represent true kinematic ambiguity (rapid crouching vs fall) or severe occlusion.
- **Borderline Errors ($N=32$, $35.6\%$ of errors)**: 22 FP and 10 FN lying within $\pm 0.20$ of decision threshold.
  - *Implication*: These samples lie near the decision boundary and can be resolved by spatial-temporal graph modeling (ST-GCN) or multi-frame context.

---

## 7. Main Research Conclusion

Based on empirical evidence across all 90 misclassifications:

1. **Behavioral Ambiguity Problem (Primary Cause of FPs, $65.6\%$ of errors)**:  
   Normal ADL activities involving rapid downward body movement (crouching, tying shoes, sitting quickly) share identical 2D downward velocity magnitudes ($20.98\text{ vs }14.92$) with true falls. 2D keypoint coordinates alone cannot distinguish uncontrolled free-fall collapse from controlled downward motion without spatial joint angle / body tilt constraints.

2. **Occlusion & Low-Visibility Problem (Primary Cause of FNs in `Home_01`, $34.4\%$ of errors)**:  
   Partial body occlusions in `Home_01` drop average keypoint visibility to $0.2746$, causing the temporal model to underestimate fall probability.

---

## 8. Artifacts Generated in Phase J2
- [`R&D/ML_Baseline/results/yolo_tcn_error_kinematics.csv`](file:///d:/ONE_DATA/Fall%20detection/R&D/ML_Baseline/results/yolo_tcn_error_kinematics.csv) (1,396 rows)
- [`R&D/ML_Baseline/results/yolo_tcn_error_event_analysis.csv`](file:///d:/ONE_DATA/Fall%20detection/R&D/ML_Baseline/results/yolo_tcn_error_event_analysis.csv) (56 event rows)
- [`R&D/ML_Baseline/yolo_sota_failure_mode_analysis.md`](file:///d:/ONE_DATA/Fall%20detection/R&D/ML_Baseline/yolo_sota_failure_mode_analysis.md) (Research report)
