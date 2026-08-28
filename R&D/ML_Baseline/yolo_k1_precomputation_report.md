# Research Report: Spatial-Augmented Feature Precomputation (Experiment K Phase K1)

> [!IMPORTANT]
> **PRECOMPUTATION & VALIDATION GATE PASSED — NO TRAINING PERFORMED YET**  
> Precomputed 187-D spatial body-geometry augmented feature tensors across all **1,396 supervised Le2i temporal windows** for Experiment K1. All base 165 features are bit-for-bit identical to original YOLO Pose features.

---

## 1. Precomputation Summary

- **Source Feature Directory**: `processed_data/Le2i_baseline/pose_estimator_features/yolo_pose/` (`(50, 165)` float32)
- **Target Feature Directory**: `processed_data/Le2i_baseline/pose_estimator_features/yolo_pose_k1/` (`(50, 187)` float32)
- **Total Windows Extracted**: **1,396 / 1,396 files ($100.0\%$)**
- **Total Disk Storage**: **28.08 MB**
- **Precomputation Execution Time**: **3.52 seconds**
- **Machine-Readable Summary JSON**: [`R&D/ML_Baseline/results/yolo_k1_precomputation_summary.json`](file:///d:/ONE_DATA/Fall%20detection/R&D/ML_Baseline/results/yolo_k1_precomputation_summary.json)
- **Validation Statistics JSON**: [`R&D/ML_Baseline/results/yolo_k1_feature_validation.json`](file:///d:/ONE_DATA/Fall%20detection/R&D/ML_Baseline/results/yolo_k1_feature_validation.json)

---

## 2. Validation Gate Results

1. **File Count & Alignment**: Exactly 1,396 files match 1-to-1 with canonical manifest `processed_pose_features_manifest.csv` **[PASS ✅]**
2. **Tensor Shape & Dtype**: `(50, 187)` float32 for all 1,396 files **[PASS ✅]**
3. **Data Quality**: **0 NaN, 0 Inf** **[PASS ✅]**
4. **Source Equality**: First 165 dimensions match the original YOLO Pose tensors bit-for-bit (`np.array_equal`) **[PASS ✅]**
5. **Source Safety**: Original `yolo_pose/` feature files remain **100% untouched** **[PASS ✅]**

---

## 3. Derived 22 Spatial Body-Geometry Feature Distributions

| Index | Feature Name | Min | Max | Mean | Std Dev | Physical / Kinematic Meaning |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **[ 0]** | Left Knee Angle | `0.0034` | `3.1401` | `2.6346` | `0.5974` | Radians ($0 \to \pi$) |
| **[ 1]** | Right Knee Angle | `0.0036` | `3.1400` | `2.6564` | `0.6107` | Radians ($0 \to \pi$) |
| **[ 2]** | Left Hip Angle | `0.0028` | `3.1398` | `2.6139` | `0.6019` | Radians ($0 \to \pi$) |
| **[ 3]** | Right Hip Angle | `0.0021` | `3.1397` | `2.6260` | `0.6163` | Radians ($0 \to \pi$) |
| **[ 4]** | Left Elbow Angle | `0.0037` | `3.1397` | `2.4169` | `0.6715` | Radians ($0 \to \pi$) |
| **[ 5]** | Right Elbow Angle | `0.0019` | `3.1408` | `2.3838` | `0.6845` | Radians ($0 \to \pi$) |
| **[ 6]** | Left Shoulder Angle | `0.0021` | `3.1383` | `0.4913` | `0.5009` | Radians ($0 \to \pi$) |
| **[ 7]** | Right Shoulder Angle | `0.0018` | `3.1400` | `0.5069` | `0.5162` | Radians ($0 \to \pi$) |
| **[ 8]** | **Spine Inclination** | `0.0014` | `3.1361` | **`0.4652`** | `0.6709` | **Torso tilt from vertical** |
| **[ 9]** | Neck Angle | `0.0025` | `3.1394` | `2.3653` | `0.5804` | Radians ($0 \to \pi$) |
| **[10]** | Left Leg Vertical | `0.0008` | `3.1398` | `0.4357` | `0.6379` | Radians ($0 \to \pi$) |
| **[11]** | Right Leg Vertical | `0.0005` | `3.1409` | `0.4431` | `0.6409` | Radians ($0 \to \pi$) |
| **[12]** | BBox Aspect Ratio | `0.0000` | `9.2716` | `0.5151` | `0.6833` | $W_{\text{bbox}} / H_{\text{bbox}}$ |
| **[13]** | BBox Area Ratio | `0.0000` | `4260.10` | `4.1329` | `37.633` | $W_{\text{bbox}} \times H_{\text{bbox}}$ |
| **[14]** | Norm Head Height | `-18.1481` | `21.9138` | `-0.9317` | `0.8820` | Head $Y$ rel to hip / $L_{\text{torso}}$ |
| **[15]** | Norm L Wrist Height | `-67.1088` | `87.7504` | `-0.1229` | `2.6951` | L Wrist $Y$ rel to hip / $L_{\text{torso}}$ |
| **[16]** | Norm R Wrist Height | `-70.4533` | `84.9982` | `-0.1287` | `2.7757` | R Wrist $Y$ rel to hip / $L_{\text{torso}}$ |
| **[17]** | Norm Ankle Height | `-30.0398` | `86.1272` | `1.1249` | `1.6769` | Ankle $Y$ rel to hip / $L_{\text{torso}}$ |
| **[18]** | Torso Scale Ratio | `0.0000` | `49.9951` | `0.9950` | `0.4919` | $L_{\text{torso}}(t) / \bar{L}_{\text{torso}}$ |
| **[19]** | Shoulder Tilt Angle | `-3.1416` | `3.1416` | `-0.1324` | `2.2663` | Radians ($-\pi \to \pi$) |
| **[20]** | Hip Tilt Angle | `-3.1416` | `3.1416` | `0.0059` | `2.2514` | Radians ($-\pi \to \pi$) |
| **[21]** | Torso Aspect Ratio | `0.0000` | `35.4616` | `0.4220` | `0.5134` | $W_{\text{shoulder}} / L_{\text{torso}}$ |

---

## 4. Next Step Recommendation

With K1 feature precomputation complete and validated 100%, the workspace is ready for Phase K1 Training under the 4-fold LOLO benchmark pipeline:

```powershell
cd "d:\ONE_DATA\Fall detection"
& "C:\Users\NEKILESH\AppData\Local\Programs\Python\Python311\python.exe" src/train_le2i_yolo_k1_spatial.py
```
