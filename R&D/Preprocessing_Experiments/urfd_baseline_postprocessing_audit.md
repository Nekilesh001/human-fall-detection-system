# URFD RGB Baseline Post-Preprocessing Audit Report

## 1. Original 70-Event Experimental Design
Per `R&D/split_strategy.md` and `Phase12_Experimental_Definition.md`, the intended URFD experiment design specified:
- **Total Intended Scope**: 70 events (30 Fall events, 40 ADL events), 100 video observations.
- **Intended Event Partition**:
  - **Train**: 49 events (21 Fall, 28 ADL)
  - **Validation**: 10 events (4 Fall, 6 ADL)
  - **Test**: 11 events (5 Fall, 6 ADL)

---

## 2. Actual 67-Event Processed Result
The completed preprocessing pipeline generated **360 temporal windows** ($W=50$ frames @ 25 FPS) across **67 events (94 video streams)**:
- **Train**: 47 events (19 Fall, 28 ADL) $\to$ **260 windows** (84 Fall, 176 ADL)
- **Validation**: 9 events (3 Fall, 6 ADL) $\to$ **43 windows** (10 Fall, 33 ADL)
- **Test**: 11 events (5 Fall, 6 ADL) $\to$ **57 windows** (24 Fall, 33 ADL)

---

## 3. Exact Skipped Events & Source Statistics

Three Fall events (6 video records) were cleanly skipped during windowing:

| Skipped Event ID | Camera IDs | Source Frame Count (PNGs) | Source FPS | Source Duration | Resampled 25 FPS Frames | Resampled Target Duration | Assigned Partition | Shortfall vs. $W=50$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `fall-16` | `cam0`, `cam1` | 55 frames | 30.0 FPS | 1.83 s | 46 frames | 1.84 s | **`train`** | 4 frames short |
| `fall-21` | `cam0`, `cam1` | 55 frames | 30.0 FPS | 1.83 s | 46 frames | 1.84 s | **`val`** | 4 frames short |
| `fall-22` | `cam0`, `cam1` | 56 frames | 30.0 FPS | 1.87 s | 47 frames | 1.88 s | **`train`** | 3 frames short |

---

## 4. Why They Were Skipped
The baseline preprocessing specification enforced a strict temporal window constraint:
$$\text{Window Length } W = 50 \text{ frames (2.0 seconds at 25 FPS)}$$
$$\text{Constraint: } \text{Num Resampled Frames} \ge W \implies \text{Generate Complete Window}$$

Because `fall-16`, `fall-21`, and `fall-22` contain 55–56 raw 30 FPS frames (approx. 1.83s–1.87s duration), their resampled 25 FPS sequences yield only 46–47 frames. In accordance with the rule prohibiting partial window creation, these 3 sequences were cleanly excluded. Both `cam0` and `cam1` for each affected event were treated identically, preserving 100% camera symmetry.

---

## 5. Split Impact Analysis

### Event & Window Breakdown Comparison

| Partition Metric | Intended 70-Event Experiment | Actual 67-Event Processed Experiment | Difference / Impact |
| :--- | :--- | :--- | :--- |
| **Train Events** | 49 (21 Fall, 28 ADL) | **47 (19 Fall, 28 ADL)** | -2 Fall events (`fall-16`, `fall-22`) |
| **Validation Events** | 10 (4 Fall, 6 ADL) | **9 (3 Fall, 6 ADL)** | -1 Fall event (`fall-21`) |
| **Test Events** | 11 (5 Fall, 6 ADL) | **11 (5 Fall, 6 ADL)** | **0 Difference (100% Test Retained)** |
| **Train Windows** | N/A | **260 windows** (84 Fall, 176 ADL) | 32.3% Fall / 67.7% ADL |
| **Validation Windows**| N/A | **43 windows** (10 Fall, 33 ADL) | 23.3% Fall / 76.7% ADL |
| **Test Windows** | N/A | **57 windows** (24 Fall, 33 ADL) | **42.1% Fall / 57.9% ADL** |
| **Total Windows** | N/A | **360 windows** (118 Fall, 242 ADL) | 32.8% Fall / 67.2% ADL |

### Key Impact Conclusion
- **Test Set Protection**: The Test partition retained **100% of its intended events** (5 Fall, 6 ADL) and **100% of its test windows** (24 Fall, 33 ADL). Benchmark evaluation integrity is completely unaffected.
- **Train & Val Shift**: Train lost 2 Fall events and Val lost 1 Fall event.

---

## 6. Temporal Window Configuration Comparison

Calculated window generation across options without modifying the processed dataset:

| Parameter / Metric | Option A ($W=25, S=12$) | Option B ($W=50, S=25$) (Current) | Option C ($W=75, S=25$) |
| :--- | :--- | :--- | :--- |
| **Window Duration** | 1.0 second | **2.0 seconds** | 3.0 seconds |
| **Usable Events** | **70 / 70 (100%)** | **67 / 70 (95.7%)** | **55 / 70 (78.6%)** |
| **Skipped Events** | **0 events** | **3 events** (`fall-16`, `21`, `22`) | **15 events** (13 Fall, 2 ADL) |
| **`fall-16` Status** | **USABLE** (2 wins/cam) | **SKIPPED** | **SKIPPED** |
| **`fall-21` Status** | **USABLE** (2 wins/cam) | **SKIPPED** | **SKIPPED** |
| **`fall-22` Status** | **USABLE** (2 wins/cam) | **SKIPPED** | **SKIPPED** |
| **Total Windows** | **873 windows** (316 Fall, 557 ADL)| **360 windows** (118 Fall, 242 ADL)| **270 windows** (64 Fall, 206 ADL) |

---

## 7. Short-Sequence Handling Evaluation & Recommendation

### Comparative Evaluation of Approaches

1. **Approach 1: Exclude Sequences $< W$ (Current Method)**
   - *Data Retention*: 95.7% of events (67/70).
   - *Data Purity*: 100% pure real-world human dynamics. Zero artificial noise or synthetic frames.
   - *Suitability*: **HIGHEST FOR FIRST BASELINE (RECOMMENDED)**.

2. **Approach 2: Reduce Window Size $W$ (e.g. $W=25$)**
   - *Data Retention*: 100% (70/70 events).
   - *Trade-off*: 1.0s window captures shorter posture context. Recommended as an ablation experiment.

3. **Approach 3: Temporal Edge Padding (3–4 Frames)**
   - *Data Retention*: 100% (70/70 events).
   - *Trade-off*: Introduces 6%–8% synthetic static frames at movement boundaries. Not recommended for baseline.

4. **Approach 4: Variable-Length Sequences**
   - *Data Retention*: 100%.
   - *Trade-off*: High batching complexity; reserved for future RNN/Transformer architectures.

---

## 8. Storage Efficiency Analysis

- **Processed Samples**: 360 `.npz` files
- **Total Storage Size**: **2,899.29 MB (~2.90 GB)**
- **Average Storage Per Window**: **8.05 MB per window**
- **Theoretical Uncompressed Size**: $50 \times 240 \times 320 \times 3 \text{ bytes} = 11.52 \text{ MB per window}$
- **Compression Ratio**: $11.52 / 8.05 = \mathbf{1.43\times}$

### Scaling Projections

| Target Window Count | Estimated Storage (Single `.npz` per window) | Scalability Assessment |
| :--- | :--- | :--- |
| **1,000 windows** | ~8.05 GB | Manageable for local SSD |
| **5,000 windows** | ~40.25 GB | Moderate; high file descriptor count |
| **10,000 windows** | ~80.50 GB | Requires binary container formats (HDF5 / LMDB) |
| **50,000 windows** | ~402.50 GB | Requires binary container formats (HDF5 / LMDB) |

#### Recommendation
`.npz`-per-window storage is **appropriate and optimal for this 360-window URFD baseline**. For larger future datasets (Le2i / MultiCamera > 5,000 windows), transition to HDF5 binary containers (`.h5`).

---

## 9. Retain Current Dataset Status
**RECOMMENDATION**: **RETAIN THE CURRENT 360-WINDOW PROCESSED DATASET**.
The 360 processed `.npz` sample files and `processed_manifest.csv` are 100% structurally valid, leak-free, and preserve 100% of the intended test set.

---

## 10. Re-preprocessing Status
**RECOMMENDATION**: **DO NOT RERUN PREPROCESSING AT THIS STAGE**.
The dataset is verified, documented, and ready for baseline model development.

---

## 11. Recommended Next Step Before Model Training
With the post-preprocessing audit complete and documented:
1. Review the final git status (confirm `processed_data/` remains ignored).
2. Stage and commit the new source scripts (`src/config.py`, `src/preprocess_urfd.py`, `src/validate_preprocessing.py`) and audit documentation onto the **`dev`** branch.
3. Proceed to the first ML baseline model development on branch **`dev`**.
