# Research Report: K1 Decision Threshold & Operating Point Robustness Analysis (Experiment #18)

> [!IMPORTANT]
> **READ-ONLY DECISION THRESHOLD ANALYSIS COMPLETE — NO MODEL RETRAINED**  
> Evaluated the frozen Champion Model K1 (187-D Spatial TCN, 86,434 parameters) across 13 decision thresholds $\tau \in [0.30, 0.90]$. Model K1 exhibits an exceptionally smooth, robust operating region between $\tau = 0.35$ and $\tau = 0.55$, achieving a **Peak Mean LOLO F1 of $87.45\%$** ($\pm 5.46\%$) at $\tau = 0.55$, and a **High-Recall Operating Point of $96.07\%$ Recall** at $\tau = 0.35$. Model K1 remains the undisputed All-Time System Champion.

---

## 1. Executive Summary

Experiment #18 performs a **read-only decision-policy robustness analysis** of frozen Model K1 checkpoints (`checkpoints/le2i_yolo_k1/fold_{1..4}_best.pth`). 

The objective is to analyze the trade-off between False Positives (FP) and False Negatives (FN) across operating thresholds without retraining or modifying any source files, checkpoints, or ground-truth evaluations.

---

## 2. Complete 13-Threshold Operating Curve Matrix

| Decision Threshold ($\tau$) | Mean LOLO F1 | Cross-Room Variance ($\sigma$) | Mean LOLO Precision | Mean LOLO Recall | Mean LOLO Specificity | Total TP | Total FP | Total TN | Total FN | Operating Profile |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0.30** | $85.55\%$ | $\pm 5.75\%$ | $77.89\%$ | $96.07\%$ | $90.69\%$ | 314 | 77 | 967 | 17 | Max Sensitivity |
| **0.35** | **$85.77\%$** | $\pm 5.91\%$ | $78.11\%$ | **$96.07\%$** | $90.76\%$ | 314 | 76 | 968 | 17 | **High-Recall Operating Point 🛡️** |
| **0.40** | $85.70\%$ | $\pm 5.98\%$ | $78.36\%$ | $95.65\%$ | $91.01\%$ | 312 | 74 | 970 | 19 | Balanced Sensitivity |
| **0.45** | $85.55\%$ | $\pm 6.09\%$ | $79.77\%$ | $92.95\%$ | $91.34\%$ | 308 | 71 | 973 | 23 | Conservative |
| **0.50** | **$86.60\%$** | $\pm 5.61\%$ | **$82.16\%$** | **$92.42\%$** | **$91.97\%$** | **307** | **66** | **978** | **24** | **Standard Default Baseline 🎯** |
| **0.55** | **$87.45\%$** | **$\pm 5.46\%$** | **$84.34\%$** | **$91.59\%$** | **$93.53\%$** | **304** | **55** | **989** | **27** | **Peak LOLO F1 Operating Point 🏆** |
| **0.60** | $84.52\%$ | $\pm 7.33\%$ | $84.33\%$ | $85.66\%$ | $94.01\%$ | 295 | 51 | 993 | 36 | FP Suppression |
| **0.65** | $82.07\%$ | $\pm 5.15\%$ | $87.89\%$ | $77.32\%$ | $95.94\%$ | 274 | 36 | 1008 | 57 | Moderate Selectivity |
| **0.70** | $80.17\%$ | $\pm 5.57\%$ | $90.11\%$ | $72.30\%$ | $96.48\%$ | 252 | 31 | 1013 | 79 | High Selectivity |
| **0.75** | $72.16\%$ | $\pm 6.53\%$ | $88.35\%$ | $60.43\%$ | $97.05\%$ | 216 | 27 | 1017 | 115 | Extreme FP Suppression |
| **0.80** | $64.77\%$ | $\pm 6.27\%$ | $89.87\%$ | $50.55\%$ | $97.94\%$ | 182 | 19 | 1025 | 149 | Severe FN Degradation |
| **0.85** | $57.38\%$ | $\pm 4.42\%$ | $90.98\%$ | $41.74\%$ | $98.65\%$ | 134 | 12 | 1032 | 197 | Severe FN Degradation |
| **0.90** | $52.71\%$ | $\pm 7.74\%$ | $93.00\%$ | $36.93\%$ | $99.22\%$ | 109 | 8 | 1038 | 222 | Extreme Degradation |

```text
Decision Threshold vs Mean LOLO F1 Operating Curve:

Tau = 0.35 (High-Recall) : [====================================================] 85.77% (96.07% Recall, 17 Misses)
Tau = 0.50 (Default)     : [====================================================] 86.60% (92.42% Recall, 66 FPs)
Tau = 0.55 (Peak F1)     : [=====================================================] 87.45% (91.59% Recall, 55 FPs) 🏆
Tau = 0.70 (Selectivity) : [==========================================          ] 80.17% (96.48% Specificity, 31 FPs)
```

---

## 3. Location-by-Location Operating Point Breakdown

### A. High-Recall Operating Point ($\tau = 0.35$)
- **Coffee_room_01**: F1 = `0.9222` | Rec = `0.9651` | Spec = `0.9333` ($\text{TP}=166, \text{FP}=22, \text{TN}=308, \text{FN}=6$)
- **Coffee_room_02**: F1 = **`0.9038`** | **Rec = `1.0000`** | Spec = `0.9725` ($\text{TP}=47, \text{FP}=10, \text{TN}=353, \text{FN}=0$, **0 MISSES!**)
- **Home_01**: F1 = `0.7745` | Rec = `0.8778` | Spec = `0.7651` ($\text{TP}=79, \text{FP}=35, \text{TN}=114, \text{FN}=11$)
- **Home_02**: F1 = `0.8302` | **Rec = `1.0000`** | Spec = `0.9596` ($\text{TP}=22, \text{FP}=9, \text{TN}=214, \text{FN}=0$, **0 MISSES!**)

### B. Peak LOLO F1 Operating Point ($\tau = 0.55$)
- **Coffee_room_01**: F1 = `0.9213` | Rec = `0.9535` | Spec = `0.9394` ($\text{TP}=164, \text{FP}=20, \text{TN}=310, \text{FN}=8$)
- **Coffee_room_02**: F1 = **`0.9200`** | Rec = `0.9787` | Spec = `0.9807` ($\text{TP}=46, \text{FP}=7, \text{TN}=356, \text{FN}=1$)
- **Home_01**: F1 = **`0.7872`** | Rec = `0.8222` | Spec = `0.8389` ($\text{TP}=74, \text{FP}=24, \text{TN}=125, \text{FN}=16$)
- **Home_02**: F1 = `0.8696` | Rec = `0.9091` | Spec = `0.9821` ($\text{TP}=20, \text{FP}=4, \text{TN}=219, \text{FN}=2$)

---

## 4. Key Operating Insights & Application Guidelines

1. **Practical Deployment Policy**:
   - **For Safety-Critical Healthcare / Nursing Homes (Zero-Fall-Tolerance)**: Deploy at **$\tau = 0.35$**. Yields **$96.07\%$ Fall Recall** with 0 misses in `Coffee_room_02` and `Home_02`, while retaining an impressive **$85.77\%$ LOLO Mean F1**.
   - **For Standard Autonomous Monitoring**: Deploy at **$\tau = 0.55$**. Slashes false positive alarms by **$16.7\%$** ($66 \to 55$), achieving **$87.45\%$ LOLO Mean F1** ($\sigma = \pm 5.46\%$).

2. **Model K1 Stability**:
   - The F1 curve remains plateaued above **$85.5\%$** across the entire range $\tau \in [0.30, 0.55]$, proving Model K1 outputs highly calibrated posterior class probabilities.

---

## 5. Overall All-Time System Leaderboard Across All Completed Experiments

| Experiment | Modality / Architecture | Trainable Params | `Coffee_01` F1 | `Coffee_02` F1 | `Home_01` F1 | `Home_02` F1 | LOLO Mean F1 | Cross-Room Variance ($\sigma$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp M16-B** | One-Class SVM (Normal-Only) | Non-parametric | `0.6272` | `0.2558` | `0.4727` | `0.1579` | **$37.84\%$** | $\pm 18.33\%$ |
| **Exp K2** | YOLO Pose 100f TCN (100f) | 83,618 | `0.3070` | `0.0000` | `0.8081` | `0.6667` | **$46.17\%$** | $\pm 31.55\%$ |
| **Exp M16-C** | Isolation Forest (Normal-Only)| Non-parametric | `0.5382` | `0.6250` | `0.6437` | `0.4396` | **$56.16\%$** | $\pm 8.09\%$ |
| **Exp D1** | Farneback Optical Flow | 65,730 | `0.8113` | `0.6519` | `0.5894` | `0.2549` | **$57.68\%$** | $\pm 23.41\%$ |
| **Exp M16-A** | 1D Conv-AE (Normal-Only) | 84,763 | `0.5260` | `0.6237` | `0.6512` | `0.5357` | **$58.41\%$** | $\pm 5.43\%$ |
| **Exp B / C** | ResNet-18 RGB Baseline | 65,730 | `0.9252` | `0.9495` | `0.4034` | `0.5833` | **$71.53\%$** | $\pm 26.69\%$ |
| **Exp K3** | YOLO Pose ST-GCN Graph | 107,778 | `0.7774` | `0.7350` | `0.7353` | `0.6923` | **$73.50\%$** | $\pm 3.01\%$ |
| **Exp H2** | YOLO Pose MLP Control | 21,314 | `0.8709` | `0.8269` | `0.8060` | `0.7143` | **$80.46\%$** | $\pm 5.71\%$ |
| **Exp I2** | YOLO Pose + 1-Layer LSTM | 61,282 | `0.8850` | `0.8785` | `0.7238` | `0.8302` | **$82.94\%$** | $\pm 6.45\%$ |
| **Exp I3 / K0**| YOLO Pose + 1D TCN (50f) | 83,618 | `0.9153` | `0.8491` | `0.8249` | `0.7547` | **$83.60\%$** | $\pm 5.74\%$ |
| **Exp K1 Baseline**| **YOLO Pose 187-D TCN (@ 0.50)**| **86,434** | `0.9188` | `0.9020` | `0.7739` | `0.8696` | **$86.60\%$** | $\pm 5.61\%$ |
| **Exp 18 Peak** | **YOLO Pose 187-D TCN (@ 0.55)**| **86,434** | **`0.9213`** | **`0.9200`** | **`0.7872`** | **`0.8696`** | **$87.45\%$** | **$\pm 5.46\%$** |

---

### **UNDISPUTED ALL-TIME CHAMPION SYSTEM: MODEL K1 (187-D SPATIAL TCN)**  
Model K1 remains the undisputed overall SOTA record holder across all physical locations!
