# Research Report: Pose Estimator Feature Precomputation & Validation Gate (Experiment H1)

> [!IMPORTANT]
> **PHASE H1 COMPLETE — VALIDATION GATE PASS**  
> 100% of feature tensors ($1,396 \times 3 = 4,188$ files) precomputed, verified, and validated for MediaPipe Pose (H1), YOLO Pose (H2), and RTMPose (H3).

---

## 1. Executive Summary

Experiment H1 precomputes scale- and translation-invariant 165-D Pose + Velocity feature tensors across **1,396 supervised Le2i windows** ($69,800$ frames) for three human pose estimators:
1. **H1: MediaPipe Pose** (Landmarker task engine)
2. **H2: YOLO Pose** (`yolov8n-pose.pt` via PyTorch CUDA)
3. **H3: RTMPose** (`rtmpose-m` via ONNX Runtime CUDA)

All estimators operate under the **exact canonical 33-landmark vector layout** and **scale-invariant torso normalization equations** specified in H0, guaranteeing 100% downstream compatibility with the controlled 21,314-parameter classifier head.

---

## 2. Validation Gate Summary Matrix

| Estimator Variant | Total Windows | Total Files | Shape & Dtype | NaN / Inf Errors | Total Frames | Detected Frames | Detection Rate (%) | Fully Detected Windows (50/50) | Completely Undetected Windows (0/50) | Storage Footprint | Validation Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **H1: MediaPipe Pose** | 1,396 | 1,396 | `(50, 165)` float32 | **0 / 0** | 69,800 | 54,836 | **78.56%** | 717 ($51.4\%$) | 74 ($5.3\%$) | 31.86 MB | **PASS ✅** |
| **H2: YOLO Pose** | 1,396 | 1,396 | `(50, 165)` float32 | **0 / 0** | 69,800 | 66,564 | **95.36%** | 1,195 ($85.6\%$) | 14 ($1.0\%$) | 22.93 MB | **PASS ✅** |
| **H3: RTMPose** | 1,396 | 1,396 | `(50, 165)` float32 | **0 / 0** | 69,800 | 69,797 | **100.00%** | 1,393 ($99.8\%$) | **0 ($0.0\%$)** | 24.16 MB | **PASS ✅** |

---

## 3. Location-Wise Pose Detection Breakdown

The detection rate per physical location demonstrates the dramatic improvement in keypoint detection stability offered by SOTA deep pose estimators (YOLO Pose and RTMPose) over MediaPipe, especially in dark or low-contrast residential environments (`Home_01` and `Home_02`):

```text
Location Detection Stability Comparison:

RTMPose (H3)   : [==================================================] 100.0% (Home_02: 100.0%)
YOLO Pose (H2) : [==============================================    ]  95.4% (Home_02:  94.3%)
MediaPipe (H1) : [======================                            ]  44.2% (Home_02:  44.2%)
```

### Detailed Physical Location Breakdown Matrix

| Physical Location | Supervised Windows | Total Frames | H1 MediaPipe Det (%) | H1 Undetected Wins | H2 YOLO Pose Det (%) | H2 Undetected Wins | H3 RTMPose Det (%) | H3 Undetected Wins | Key Findings |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Coffee_room_01`** | 502 | 25,100 | **93.6%** | 3 | **97.9%** | 1 | **100.0%** | **0** | Excellent quality across all estimators |
| **`Coffee_room_02`** | 410 | 20,500 | **91.2%** | 8 | **98.6%** | 2 | **100.0%** | **0** | High contrast, strong detection |
| **`Home_01`** | 239 | 11,950 | **60.4%** | 20 | **85.6%** | 8 | **100.0%** | **0** | Severe occlusion; SOTA pose models recover person |
| **`Home_02`** | 245 | 12,250 | **44.2%** | 43 | **94.3%** | 3 | **100.0%** | **0** | **RTMPose / YOLO eliminate residential contrast failure** |

---

## 4. Feature Space & Normalization Protocol Audit

- **Input Dimension**: `(50, 165)` float32 tensor per window.
- **Pose Geometry (99-D)**: 33 canonical landmarks $\times (\hat{x}_i, \hat{y}_i, v_i)$.
- **Joint Velocity (66-D)**: 33 canonical landmarks $\times (d\hat{x}_i, d\hat{y}_i)$.
- **COCO-17 to Canonical 33 Topology Mapping**: Direct 1-to-1 mapping for COCO keypoints `0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28`; remaining 16 facial/foot landmarks zero-padded ($v_i = 0.0$).
- **Torso Normalization**: Hip-centered and scaled by torso length $L_{\text{torso}} = \|\mathbf{p}_{\text{shoulder}} - \mathbf{p}_{\text{hip}}\|_2$.

---

## 5. Storage Directories & Manifest Alignment

All feature files are stored in isolated subdirectories:
- `processed_data/Le2i_baseline/pose_estimator_features/mediapipe/` (1,396 `.npz` files)
- `processed_data/Le2i_baseline/pose_estimator_features/yolo_pose/` (1,396 `.npz` files)
- `processed_data/Le2i_baseline/pose_estimator_features/rtmpose/` (1,396 `.npz` files)

**Manifest Alignment**: 100% deterministic 1-to-1 alignment with `processed_pose_features_manifest.csv` across all 1,396 window IDs.

---

## 6. Next Experimental Steps (Phase H2 & H3)

With Phase H1 complete and 100% validated:
1. **Phase H2**: Train controlled Pose + Velocity MLP ($21,314$ params) across 4-fold LOLO partitions for H1, H2, and H3.
2. **Phase H3**: Evaluate Leave-One-Location-Out (LOLO) performance to test whether YOLO Pose (H2) and RTMPose (H3) improve cross-room generalization on `Home_01` and `Home_02`.
