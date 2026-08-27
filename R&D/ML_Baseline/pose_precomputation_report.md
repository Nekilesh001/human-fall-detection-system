# Experiment E: MediaPipe Pose Feature Precomputation & Validation Report

## 1. Executive Summary & Phase 1 Decision Gate
- **Precomputation Status**: **COMPLETED & VALIDATED**
- **Phase 1 Gate Decision**: **PASS (PROCEED TO PHASE 2 LOLO TRAINING)**
- **Windows Represented**: **1,396 / 1,396 temporal windows** ($331$ FALL, $1,065$ NORMAL).
- **Supervised Videos**: **127 / 127 videos** ($0$ UNKNOWN records included).
- **Tensor Integrity**: **0 missing files, 0 invalid shapes, 0 NaN/Inf errors**.

---

## 2. Location Pose Detection Quality Breakdown

| Location | Total Windows | Total Frames | Detected Frames | Pose Detection Rate | Completely Undetected Windows | Quality Rating |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Coffee_room_01`** | 502 | 25,100 | 23502 | **93.6%** | 3 / 502 | **EXCELLENT ✅** |
| **`Coffee_room_02`** | 410 | 20,500 | 18691 | **91.2%** | 8 / 410 | **EXCELLENT ✅** |
| **`Home_01`** | 239 | 11,950 | 5871 | **49.1%** | 33 / 239 | **GOOD ✅** |
| **`Home_02`** | 245 | 12,250 | 4665 | **38.1%** | 57 / 245 | **FAIR (Low Illumination) ⚠️** |
| **TOTAL** | **1,396** | **69,800** | **52729** | **75.5%** | **101 / 1,396** | **PASSED ✅** |

- **Home_02 Specific Inspection**: Low residential illumination and furniture occlusion resulted in a lower detection rate ($33.9\%$). Zero-vector padding with visibility $=0.0$ preserves temporal alignment for classifier training.

---

## 3. Feature Tensor Specifications

| Model Variant | Feature Tensor Shape per Window | Window Aggregation Shape | Storage Footprint (1,396 wins) |
| :--- | :---: | :---: | :---: |
| **Model E1 (Pose Geometry)** | `(50, 99)` float32 | Mean+Std `(198)` | **26.36 MB** |
| **Model E2 (Pose + Velocity)** | `(50, 165)` float32 | Mean+Std `(330)` | **43.93 MB** |
| **Model E3 (Pose Motion Geometry)** | `(50, 173)` float32 | Mean+Std `(346)` | **46.06 MB** |

---

## 4. Phase 1 Verification Checklist
- [x] Exactly 1,396 windows processed
- [x] Exactly 331 FALL and 1,065 NORMAL windows
- [x] Exactly 127 supervised videos represented
- [x] 0 UNKNOWN records included
- [x] 0 NaN / Inf errors
- [x] All tensor shapes verified: E1 `(50, 99)`, E2 `(50, 165)`, E3 `(50, 173)`
- [x] Precomputation summary JSON generated
