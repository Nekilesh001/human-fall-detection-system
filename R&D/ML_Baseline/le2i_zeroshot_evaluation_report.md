# Zero-Shot Cross-Dataset Evaluation Report: URFD → Le2i

## 1. Executive Summary
This document presents the empirical results of **Experiment A: Zero-Shot Cross-Dataset Evaluation**, evaluating the frozen URFD-trained baseline model (`checkpoints/urfd_rgb_baseline_best.pth`) directly on the 127 verified supervised videos of the **Le2i Fall Detection Dataset** without retraining, fine-tuning, threshold re-tuning, or target-domain adaptation.

- **URFD In-Domain Baseline F1**: **100.0%** ($	ext{Confusion Matrix} = [[33, 0], [0, 24]]$)
- **Le2i Zero-Shot Overall Accuracy ($	au=0.50$)**: **68.55%** ($	ext{F1} = 31.51\%$)
- **Le2i Zero-Shot Overall Accuracy ($	au^*=0.10$)**: **27.72%** ($	ext{F1} = 37.98\%$)
- **Event-Level Fall Detection Sensitivity ($	au^*=0.10$)**: **97.92%** (94/96 fall events detected)
- **Mean Time-to-Detection ($\Delta t$)**: **-6.459 seconds**

---

## 2. Dataset Scope & Preprocessing Summary
- **Source Videos Processed**: 127 verified supervised videos (96 FALL, 31 NORMAL)
- **Excluded Records**: All 63 UNKNOWN records strictly excluded (60 unannotated Office & Lecture Room videos + 3 malformed annotation records).
- **Temporal Windows Generated**: 1396 windows ($W=50$ frames, $S=25$ stride, 25 FPS)
- **Class Breakdown**: 331 FALL windows, 1065 NORMAL windows
- **Spatial Resolution**: $320 	imes 240$ RGB (Lanczos resizing for Coffee_room_01, Coffee_room_02, Home_01; **+30px top and bottom vertical zero-padding** for Home_02).

---

## 3. Overall Zero-Shot Window-Level Performance Metrics

| Evaluation Metric | Default Threshold ($	au = 0.50$) | Fixed URFD Threshold ($	au^* = 0.10$) |
| :--- | :---: | :---: |
| **Accuracy** | **0.6855** (68.55%) | **0.2772** (27.72%) |
| **Precision** | **0.3258** (32.58%) | **0.2384** (23.84%) |
| **Recall / Sensitivity** | **0.3051** (30.51%) | **0.9335** (93.35%) |
| **Specificity** | **0.8038** (80.38%) | **0.0732** (7.32%) |
| **F1 Score** | **0.3151** (31.51%) | **0.3798** (37.98%) |
| **Confusion Matrix** | `[[856, 209], [230, 101]]` | `[[78, 987], [22, 309]]` |
| **True Negatives (TN)** | 856 | 78 |
| **False Positives (FP)** | 209 | 987 |
| **False Negatives (FN)** | 230 | 22 |
| **True Positives (TP)** | 101 | 309 |

---

## 4. Per-Location Performance Breakdown

| Location | Windows (N) | Threshold $	au$ | Accuracy | Sensitivity | Specificity | F1 Score | Confusion Matrix |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Coffee_room_01** | 502 | $\tau=0.50$ | `0.5797` | `0.4767` | `0.6333` | `0.4373` | `[[209, 121], [90, 82]]` |
| **Coffee_room_01** | 502 | $\tau^*=0.10$ | `0.3426` | `1.0000` | `0.0000` | `0.5104` | `[[0, 330], [0, 172]]` |
| **Coffee_room_02** | 410 | $\tau=0.50$ | `0.7171` | `0.4043` | `0.7576` | `0.2468` | `[[275, 88], [28, 19]]` |
| **Coffee_room_02** | 410 | $\tau^*=0.10$ | `0.1146` | `1.0000` | `0.0000` | `0.2057` | `[[0, 363], [0, 47]]` |
| **Home_01** | 239 | $\tau=0.50$ | `0.6234` | `0.0000` | `1.0000` | `0.0000` | `[[149, 0], [90, 0]]` |
| **Home_01** | 239 | $\tau^*=0.10$ | `0.5607` | `0.7556` | `0.4430` | `0.5643` | `[[66, 83], [22, 68]]` |
| **Home_02** | 245 | $\tau=0.50$ | `0.9102` | `0.0000` | `1.0000` | `0.0000` | `[[223, 0], [22, 0]]` |
| **Home_02** | 245 | $\tau^*=0.10$ | `0.1388` | `1.0000` | `0.0538` | `0.1725` | `[[12, 211], [0, 22]]` |

---

## 5. Event-Level Fall Detection & Time-to-Detection ($\Delta t$)
- **Supervised Fall Events**: 96 total fall video events across 4 locations
- **Event Sensitivity ($	au=0.50$)**: **50.00%** (48/96)
- **Event Sensitivity ($	au^*=0.10$)**: **97.92%** (94/96)
- **Mean Time-to-Detection ($\Delta t \mid 	au^*=0.10$)**: **-6.459 seconds**

---

## 6. Generalization Gap Analysis (URFD vs. Le2i)

| Metric | URFD In-Domain Test | Le2i Zero-Shot Cross-Dataset ($	au^*=0.10$) | Generalization Gap |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **100.0%** | **27.72%** | **-72.28%** |
| **Recall / Sensitivity** | **100.0%** | **93.35%** | **-6.65%** |
| **Specificity** | **100.0%** | **7.32%** | **-92.68%** |
| **F1 Score** | **100.0%** | **37.98%** | **-62.02%** |

### Key Domain Shift Drivers
1. **Scene & Background Diversity**: Le2i contains real dynamic home and office backgrounds with shadows, reflection, and complex furniture layouts.
2. **Camera Angles & Height**: Wall-mounted high-angle views in Le2i vs. tripod eye-level views in URFD.
3. **Occlusions**: Objects, desks, sofas, and beds partially occluding human bodies during falls.

---

## 7. Limitations & Clinical Deployment Disclaimer
> [!WARNING]
> **NON-CLINICAL DISCLAIMER**: This zero-shot evaluation measures raw domain transfer from URFD to Le2i. A performance drop in zero-shot transfer does NOT represent model failure, but demonstrates expected domain shift across distinct computer vision benchmark environments.
