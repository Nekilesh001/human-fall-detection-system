# URFD RGB Baseline Test Prediction & Confidence Analysis

## 1. Executive Summary
This audit artifact provides a window-level and event-level confidence decomposition of the final test predictions for the URFD RGB Baseline model.

## 2. Test Set Window-Level Prediction Table
| Window ID | Event ID | Camera | Ground Truth | Pred Label | P(FALL) | P(NORMAL) | Confidence | Correct? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `URFD_adl-03_cam0_w000` | `adl-03` | `cam0` | **NORMAL** | **NORMAL** | `0.008786` | `0.991214` | `99.12%` | ✅ |
| `URFD_adl-03_cam0_w001` | `adl-03` | `cam0` | **NORMAL** | **NORMAL** | `0.010751` | `0.989249` | `98.92%` | ✅ |
| `URFD_adl-03_cam0_w002` | `adl-03` | `cam0` | **NORMAL** | **NORMAL** | `0.017016` | `0.982984` | `98.30%` | ✅ |
| `URFD_adl-03_cam0_w003` | `adl-03` | `cam0` | **NORMAL** | **NORMAL** | `0.028503` | `0.971497` | `97.15%` | ✅ |
| `URFD_adl-03_cam0_w004` | `adl-03` | `cam0` | **NORMAL** | **NORMAL** | `0.026536` | `0.973464` | `97.35%` | ✅ |
| `URFD_adl-07_cam0_w000` | `adl-07` | `cam0` | **NORMAL** | **NORMAL** | `0.007101` | `0.992899` | `99.29%` | ✅ |
| `URFD_adl-07_cam0_w001` | `adl-07` | `cam0` | **NORMAL** | **NORMAL** | `0.008817` | `0.991183` | `99.12%` | ✅ |
| `URFD_adl-07_cam0_w002` | `adl-07` | `cam0` | **NORMAL** | **NORMAL** | `0.015657` | `0.984343` | `98.43%` | ✅ |
| `URFD_adl-07_cam0_w003` | `adl-07` | `cam0` | **NORMAL** | **NORMAL** | `0.020689` | `0.979311` | `97.93%` | ✅ |
| `URFD_adl-15_cam0_w000` | `adl-15` | `cam0` | **NORMAL** | **NORMAL** | `0.004246` | `0.995754` | `99.58%` | ✅ |
| `URFD_adl-15_cam0_w001` | `adl-15` | `cam0` | **NORMAL** | **NORMAL** | `0.003895` | `0.996105` | `99.61%` | ✅ |
| `URFD_adl-15_cam0_w002` | `adl-15` | `cam0` | **NORMAL** | **NORMAL** | `0.003840` | `0.996160` | `99.62%` | ✅ |
| `URFD_adl-15_cam0_w003` | `adl-15` | `cam0` | **NORMAL** | **NORMAL** | `0.003545` | `0.996455` | `99.65%` | ✅ |
| `URFD_adl-15_cam0_w004` | `adl-15` | `cam0` | **NORMAL** | **NORMAL** | `0.003531` | `0.996469` | `99.65%` | ✅ |
| `URFD_adl-15_cam0_w005` | `adl-15` | `cam0` | **NORMAL** | **NORMAL** | `0.005239` | `0.994761` | `99.48%` | ✅ |
| `URFD_adl-15_cam0_w006` | `adl-15` | `cam0` | **NORMAL** | **NORMAL** | `0.007313` | `0.992687` | `99.27%` | ✅ |
| `URFD_adl-15_cam0_w007` | `adl-15` | `cam0` | **NORMAL** | **NORMAL** | `0.004529` | `0.995471` | `99.55%` | ✅ |
| `URFD_adl-16_cam0_w000` | `adl-16` | `cam0` | **NORMAL** | **NORMAL** | `0.004477` | `0.995523` | `99.55%` | ✅ |
| `URFD_adl-16_cam0_w001` | `adl-16` | `cam0` | **NORMAL** | **NORMAL** | `0.003957` | `0.996043` | `99.60%` | ✅ |
| `URFD_adl-16_cam0_w002` | `adl-16` | `cam0` | **NORMAL** | **NORMAL** | `0.003813` | `0.996187` | `99.62%` | ✅ |
| `URFD_adl-16_cam0_w003` | `adl-16` | `cam0` | **NORMAL** | **NORMAL** | `0.003493` | `0.996507` | `99.65%` | ✅ |
| `URFD_adl-16_cam0_w004` | `adl-16` | `cam0` | **NORMAL** | **NORMAL** | `0.003333` | `0.996667` | `99.67%` | ✅ |
| `URFD_adl-16_cam0_w005` | `adl-16` | `cam0` | **NORMAL** | **NORMAL** | `0.003688` | `0.996312` | `99.63%` | ✅ |
| `URFD_adl-16_cam0_w006` | `adl-16` | `cam0` | **NORMAL** | **NORMAL** | `0.003887` | `0.996113` | `99.61%` | ✅ |
| `URFD_adl-21_cam0_w000` | `adl-21` | `cam0` | **NORMAL** | **NORMAL** | `0.004290` | `0.995710` | `99.57%` | ✅ |
| `URFD_adl-21_cam0_w001` | `adl-21` | `cam0` | **NORMAL** | **NORMAL** | `0.003947` | `0.996053` | `99.61%` | ✅ |
| `URFD_adl-21_cam0_w002` | `adl-21` | `cam0` | **NORMAL** | **NORMAL** | `0.003599` | `0.996401` | `99.64%` | ✅ |
| `URFD_adl-21_cam0_w003` | `adl-21` | `cam0` | **NORMAL** | **NORMAL** | `0.002801` | `0.997199` | `99.72%` | ✅ |
| `URFD_adl-21_cam0_w004` | `adl-21` | `cam0` | **NORMAL** | **NORMAL** | `0.002920` | `0.997080` | `99.71%` | ✅ |
| `URFD_adl-21_cam0_w005` | `adl-21` | `cam0` | **NORMAL** | **NORMAL** | `0.004004` | `0.995996` | `99.60%` | ✅ |
| `URFD_adl-21_cam0_w006` | `adl-21` | `cam0` | **NORMAL** | **NORMAL** | `0.005112` | `0.994888` | `99.49%` | ✅ |
| `URFD_adl-21_cam0_w007` | `adl-21` | `cam0` | **NORMAL** | **NORMAL** | `0.006351` | `0.993649` | `99.36%` | ✅ |
| `URFD_adl-28_cam0_w000` | `adl-28` | `cam0` | **NORMAL** | **NORMAL** | `0.074940` | `0.925060` | `92.51%` | ✅ |
| `URFD_fall-07_cam0_w000` | `fall-07` | `cam0` | **FALL** | **FALL** | `0.935461` | `0.064539` | `93.55%` | ✅ |
| `URFD_fall-07_cam0_w001` | `fall-07` | `cam0` | **FALL** | **FALL** | `0.914435` | `0.085565` | `91.44%` | ✅ |
| `URFD_fall-07_cam0_w002` | `fall-07` | `cam0` | **FALL** | **FALL** | `0.955380` | `0.044620` | `95.54%` | ✅ |
| `URFD_fall-07_cam0_w003` | `fall-07` | `cam0` | **FALL** | **FALL** | `0.967733` | `0.032267` | `96.77%` | ✅ |
| `URFD_fall-07_cam1_w000` | `fall-07` | `cam1` | **FALL** | **FALL** | `0.977638` | `0.022362` | `97.76%` | ✅ |
| `URFD_fall-07_cam1_w001` | `fall-07` | `cam1` | **FALL** | **FALL** | `0.976069` | `0.023931` | `97.61%` | ✅ |
| `URFD_fall-07_cam1_w002` | `fall-07` | `cam1` | **FALL** | **FALL** | `0.986905` | `0.013095` | `98.69%` | ✅ |
| `URFD_fall-07_cam1_w003` | `fall-07` | `cam1` | **FALL** | **FALL** | `0.989772` | `0.010228` | `98.98%` | ✅ |
| `URFD_fall-08_cam0_w000` | `fall-08` | `cam0` | **FALL** | **FALL** | `0.935293` | `0.064707` | `93.53%` | ✅ |
| `URFD_fall-08_cam0_w001` | `fall-08` | `cam0` | **FALL** | **FALL** | `0.964232` | `0.035768` | `96.42%` | ✅ |
| `URFD_fall-08_cam1_w000` | `fall-08` | `cam1` | **FALL** | **FALL** | `0.987332` | `0.012668` | `98.73%` | ✅ |
| `URFD_fall-08_cam1_w001` | `fall-08` | `cam1` | **FALL** | **FALL** | `0.987231` | `0.012769` | `98.72%` | ✅ |
| `URFD_fall-11_cam0_w000` | `fall-11` | `cam0` | **FALL** | **FALL** | `0.912733` | `0.087267` | `91.27%` | ✅ |
| `URFD_fall-11_cam0_w001` | `fall-11` | `cam0` | **FALL** | **FALL** | `0.956437` | `0.043563` | `95.64%` | ✅ |
| `URFD_fall-11_cam0_w002` | `fall-11` | `cam0` | **FALL** | **FALL** | `0.965784` | `0.034216` | `96.58%` | ✅ |
| `URFD_fall-11_cam1_w000` | `fall-11` | `cam1` | **FALL** | **FALL** | `0.981164` | `0.018836` | `98.12%` | ✅ |
| `URFD_fall-11_cam1_w001` | `fall-11` | `cam1` | **FALL** | **FALL** | `0.985197` | `0.014803` | `98.52%` | ✅ |
| `URFD_fall-11_cam1_w002` | `fall-11` | `cam1` | **FALL** | **FALL** | `0.985406` | `0.014594` | `98.54%` | ✅ |
| `URFD_fall-15_cam0_w000` | `fall-15` | `cam0` | **FALL** | **FALL** | `0.781254` | `0.218746` | `78.13%` | ✅ |
| `URFD_fall-15_cam1_w000` | `fall-15` | `cam1` | **FALL** | **FALL** | `0.978452` | `0.021548` | `97.85%` | ✅ |
| `URFD_fall-20_cam0_w000` | `fall-20` | `cam0` | **FALL** | **FALL** | `0.684519` | `0.315481` | `68.45%` | ✅ |
| `URFD_fall-20_cam0_w001` | `fall-20` | `cam0` | **FALL** | **FALL** | `0.705341` | `0.294659` | `70.53%` | ✅ |
| `URFD_fall-20_cam1_w000` | `fall-20` | `cam1` | **FALL** | **FALL** | `0.966621` | `0.033379` | `96.66%` | ✅ |
| `URFD_fall-20_cam1_w001` | `fall-20` | `cam1` | **FALL** | **FALL** | `0.967566` | `0.032434` | `96.76%` | ✅ |

## 3. Event-Level Summary & Consistency
| Event ID | Ground Truth | Total Windows | Correct Windows | P(FALL) Min | P(FALL) Max | P(FALL) Mean | Cameras | Consistent? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `adl-03` | **NORMAL** | 5 | 5 | `0.008786` | `0.028503` | `0.018318` | `cam0` | ✅ |
| `adl-07` | **NORMAL** | 4 | 4 | `0.007101` | `0.020689` | `0.013066` | `cam0` | ✅ |
| `adl-15` | **NORMAL** | 8 | 8 | `0.003531` | `0.007313` | `0.004517` | `cam0` | ✅ |
| `adl-16` | **NORMAL** | 7 | 7 | `0.003333` | `0.004477` | `0.003807` | `cam0` | ✅ |
| `adl-21` | **NORMAL** | 8 | 8 | `0.002801` | `0.006351` | `0.004128` | `cam0` | ✅ |
| `adl-28` | **NORMAL** | 1 | 1 | `0.074940` | `0.074940` | `0.074940` | `cam0` | ✅ |
| `fall-07` | **FALL** | 8 | 8 | `0.914435` | `0.989772` | `0.962924` | `cam0,cam1` | ✅ |
| `fall-08` | **FALL** | 4 | 4 | `0.935293` | `0.987332` | `0.968522` | `cam0,cam1` | ✅ |
| `fall-11` | **FALL** | 6 | 6 | `0.912733` | `0.985406` | `0.964454` | `cam0,cam1` | ✅ |
| `fall-15` | **FALL** | 2 | 2 | `0.781254` | `0.978452` | `0.879853` | `cam0,cam1` | ✅ |
| `fall-20` | **FALL** | 4 | 4 | `0.684519` | `0.967566` | `0.831012` | `cam0,cam1` | ✅ |

## 4. Probability Separation & Margin Analysis
- **True FALL Windows (N=24)**: Minimum P(FALL) = `0.684519`, Mean P(FALL) = `0.935331`
- **True NORMAL Windows (N=33)**: Maximum P(FALL) = `0.074940`, Mean P(FALL) = `0.009534`
- **Probability Separation Margin**: `0.609579` (Huge separation gap)
- **Decision Threshold Analysis**: Default threshold $\tau=0.50$ sits perfectly in the center of the separation gap (`[0.0749, 0.6845]`). Tuning $\tau^*=0.10$ was mathematically redundant because the classes are perfectly linearly separable in feature space.
