# Research Report: 100-Frame Feature Precomputation (Experiment K Phase K2)

> [!IMPORTANT]
> **PRECOMPUTATION & VALIDATION GATE PASSED — NO TRAINING PERFORMED YET**  
> Precomputed 100-frame ($4.0\text{ seconds}$ at 25 FPS) 165-D YOLO Pose feature tensors and manifest across all **1,142 supervised 100-frame temporal windows**. All 1,142 files passed bit-level shape, dtype, non-NaN/Inf, and 1-to-1 manifest validation.

---

## 1. Precomputation Summary

- **Source Dataset**: 127 Le2i videos across 4 physical locations (`Coffee_room_01`, `Coffee_room_02`, `Home_01`, `Home_02`)
- **Sequence Length**: 100 frames ($4.0\text{ s}$ at 25 FPS)
- **Stride**: 25 frames ($1.0\text{ s}$ overlap)
- **Target Manifest**: [`processed_data/Le2i_baseline/processed_pose_100f_manifest.csv`](file:///d:/ONE_DATA/Fall%20detection/processed_data/Le2i_baseline/processed_pose_100f_manifest.csv)
- **Target Feature Directory**: `processed_data/Le2i_baseline/pose_estimator_features/yolo_pose_100f/`
- **Total Windows Extracted**: **1,142 / 1,142 files ($100.0\%$)**
- **Total Disk Storage**: **37.72 MB**
- **Precomputation Execution Time**: **373.21 seconds**
- **Machine-Readable Summary JSON**: [`R&D/ML_Baseline/results/yolo_k2_100f_precomputation_summary.json`](file:///d:/ONE_DATA/Fall%20detection/R&D/ML_Baseline/results/yolo_k2_100f_precomputation_summary.json)

---

## 2. Validation Gate Results

1. **Window & File Count Check**: Exactly 1,142 windows in manifest and 1,142 NPZ files **[EXACT MATCH PASS ✅]**
2. **Window ID Uniqueness**: 0 duplicate window IDs across all 1,142 entries **[PASS ✅]**
3. **Tensor Shape & Dtype**: `(100, 165)` float32 for all 1,142 files **[PASS ✅]**
4. **Data Quality**: **0 NaN, 0 Inf** across all 114,200 feature frames **[PASS ✅]**
5. **Location Distribution Match**:
   - `Coffee_room_01`: **408 / 408** **[EXACT MATCH PASS ✅]**
   - `Coffee_room_02`: **370 / 370** **[EXACT MATCH PASS ✅]**
   - `Home_01`: **179 / 179** **[EXACT MATCH PASS ✅]**
   - `Home_02`: **185 / 185** **[EXACT MATCH PASS ✅]**
6. **Label Distribution Match**:
   - `NORMAL`: **838 / 838** ($73.38\%$) **[EXACT MATCH PASS ✅]**
   - `FALL`: **304 / 304** ($26.62\%$) **[EXACT MATCH PASS ✅]**
7. **Dataset Safety Audit**:
   - Canonical 50-frame manifest (`processed_pose_features_manifest.csv`): **1,396 rows intact**
   - Canonical 50-frame `yolo_pose/` feature tensors: **1,396 files intact**
   - K1 187-D `yolo_pose_k1/` feature tensors: **1,396 files intact**

---

## 3. Next Step Recommendation

With 100-frame feature precomputation complete and validated 100%, the workspace is ready for Phase K2 Training under the 4-fold LOLO benchmark pipeline:

```powershell
cd "d:\ONE_DATA\Fall detection"
& "C:\Users\NEKILESH\AppData\Local\Programs\Python\Python311\python.exe" src/train_le2i_yolo_k2_100f.py
```
