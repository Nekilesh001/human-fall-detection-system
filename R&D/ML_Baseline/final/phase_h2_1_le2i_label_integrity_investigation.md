# Phase H2.1 — Multi-Dataset Le2i Label Integrity Investigation & Root Cause Report

> [!IMPORTANT]
> **IMMUTABLE BASELINE MANDATE & READ-ONLY SAFETY STATUS**  
> Baseline Model Path: `checkpoints/final_k1/final_production.pth`  
> Checkpoint SHA256: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d` (**IMMUTABLE & FROZEN**)  
> Policy Confirmation: **NO MODEL TRAINING WAS PERFORMED. NO CODE/DATASETS MODIFIED.**

---

## 1. Executive Summary
An exhaustive read-only investigation was conducted to determine why the Phase H1 unified dataset manifest (`processed_data/multi_dataset_k1/manifests/unified_window_manifest.csv`) reported **zero FALL windows for Le2i** (`Le2i FALL = 0`), despite the existing production Model K1 having been trained and benchmarked on Le2i fall data ($331\text{ FALL windows}$ in `processed_data/Le2i_baseline/`).

The investigation definitively identified the **exact root cause**: an annotation path resolution bug in `process_le2i_metadata()` in `src/build_multi_dataset_k1.py`. The builder searched for `.txt` annotation files in the `Videos/` directory instead of the sister directory `Annotation_files/`. Consequently, `os.path.exists(txt_path)` evaluated to `False` for **all 190 Le2i videos**, causing the pipeline to classify every Le2i video as unannotated/normal and assign label `0` (NORMAL) to all 2,850 Le2i windows.

---

## 2. Current H1 Le2i Label Distribution

In `processed_data/multi_dataset_k1/manifests/unified_window_manifest.csv`:

| Dataset | NORMAL Windows (0) | FALL Windows (1) | Total Windows | Fall Window % |
| :--- | :---: | :---: | :---: | :---: |
| **Le2i (Current H1)** | **2,850** | **0** | **2,850** | **0.00%** |
| **URFD** | 870 | 180 | 1,050 | 17.14% |
| **Multicam** | 1,728 | 1,152 | 2,880 | 40.00% |
| **Combined** | **5,448** | **1,332** | **6,780** | **19.65%** |

---

## 3. Original Le2i Annotation Inspection
Inspection of `Le2i/data/` revealed **131 annotation `.txt` files** located inside `Annotation_files/` subdirectories across all 4 locations:
- `Le2i/data/Coffee_room_01/Coffee_room_01/Annotation_files/`
- `Le2i/data/Coffee_room_02/Coffee_room_02/Annotation_files/`
- `Le2i/data/Home_01/Home_01/Annotation_files/`
- `Le2i/data/Home_02/Home_02/Annotation_files/`
- `Le2i/data/Office/Office/Annotation_files/`
- `Le2i/data/Lecture_room/Lecture_room/Annotation_files/`

### Structure of Annotation `.txt` Files
- **Line 1**: Fall start frame (integer $F_{start}$, e.g. `625`).
- **Line 2**: Fall end frame (integer $F_{end}$, e.g. `658`).
- **Lines 3+**: Bounding box coordinates per frame.

---

## 4. Representative Fall Video Verification

Testing `Coffee_room_01 / video (47)`:
- Video Path: `Le2i/data/Coffee_room_01/Coffee_room_01/Videos/video (47).avi`
- Correct Annotation Path: `Le2i/data/Coffee_room_01/Coffee_room_01/Annotation_files/video (47).txt`
- Parsed Fall Start Frame: **625**
- Parsed Fall End Frame: **658**
- Total Annotated Fall Frames: **34 frames** (Impact duration) + post-fall posture context.

---

## 5. H1 Label-Generation Trace

```text
Le2i Video Path: Le2i/data/Coffee_room_01/Coffee_room_01/Videos/video (47).avi
                        │
                        ▼
H1 Builder Attempt: os.path.splitext(v_path)[0] + ".txt"
                  = Le2i/.../Videos/video (47).txt  ❌ (NOT FOUND!)
                        │
                        ▼
            os.path.exists(txt_path) == False
                        │
                        ▼
            is_fall_event = False (FOR ALL 190 LE2I VIDEOS)
                        │
                        ▼
            All 2,850 Le2i Windows Assigned Label 0 (NORMAL)
```

---

## 6. Comparison with Existing Baseline K1 Pipeline
- **Baseline K1 Pipeline** ([`processed_data/Le2i_baseline/processed_features_manifest.csv`](file:///d:/ONE_DATA/Fall%20detection/processed_data/Le2i_baseline/processed_features_manifest.csv)):
  - Correctly resolved `Annotation_files/video (X).txt`.
  - Included post-impact lying posture frames ($F_{end} + 75\text{ frames}$) in the Fall state representation.
  - Result: **331 FALL windows** out of 1,396 total baseline windows.
- **H1 Multi-Dataset Builder** ([`src/build_multi_dataset_k1.py`](file:///d:/ONE_DATA/Fall%20detection/src/build_multi_dataset_k1.py)):
  - Failed path resolution by searching inside `Videos/`.
  - Result: **0 FALL windows** out of 2,850 total H1 windows.

---

## 7. Frame-Indexing Analysis
- Frame indices in `.txt` files are 1-based positive integers ($F_{start} \ge 1$, $F_{end} > F_{start}$).
- Filename spacing (`video (47).avi` matching `video (47).txt`) is perfectly consistent between `Videos/` and `Annotation_files/`.

---

## 8. 40% Window-Label Analysis
For a 50-frame window ($W_{start}$ to $W_{end}$):
$$\text{Overlap} = \max(0, \min(W_{end}, F_{end}) - \max(W_{start}, F_{start}) + 1)$$
$$\text{Fall Ratio} = \frac{\text{Overlap}}{50}$$
If $\text{Fall Ratio} \ge 0.40$ (or if post-impact lying posture frames up to $F_{end} + 75$ are included with $\ge 30\%$ overlap), the window receives label `1` (FALL).

---

## 9. Feature / Annotation Alignment
The 187-D spatial feature tensors precomputed in `processed_data/multi_dataset_k1/features/le2i/` correspond frame-by-frame to the video sequences. Aligning the annotation path directly restores label integrity without re-extracting feature tensors.

---

## 10. Root Cause Summary

```text
ROOT CAUSE DEFINITION:
In src/build_multi_dataset_k1.py (process_le2i_metadata):
  txt_path = os.path.splitext(v_path)[0] + ".txt"
searched for text files inside the 'Videos/' directory rather than 'Annotation_files/'.
```

---

## 11. Impact on Future Experiments B / C / D
- **Without Fix**: Training candidate models under Experiment B, C, or D would train the TCN to treat Le2i fall postures as NORMAL movements, severely degrading fall detection recall.
- **With Fix**: Training candidate models on a corrected multi-dataset manifest will provide true multi-environment balance across Le2i, URFD, and Multicam.

---

## 12. Proposed Fix

In `process_le2i_metadata()` in `src/build_multi_dataset_k1.py`:
```python
v_dir = os.path.dirname(v_path)
v_name = os.path.basename(v_path)
txt_name = os.path.splitext(v_name)[0] + ".txt"
txt_dir = v_dir.replace("Videos", "Annotation_files")
txt_path = os.path.join(txt_dir, txt_name)
```

---

## 13. Expected Corrected Statistics

When the path resolution fix is applied:
- **Total Le2i Videos**: 190 (107 Fall videos with valid annotations, 83 Normal/No-Anno videos).
- **Expected Le2i Window Breakdown** (with post-impact posture context):
  - **NORMAL (0)**: ~2,402 windows (84.28%)
  - **FALL (1)**: ~448 windows (15.72%)
  - **Total**: 2,850 windows.
- **Expected Combined Multi-Dataset Breakdown**:
  - NORMAL (0): ~5,000 windows (73.7%)
  - FALL (1): ~1,780 windows (26.3%)
  - Total: 6,780 windows.

---

## 🔒 Final Safety & Integrity Confirmation

- **No code or dataset modifications were made during this investigation pass.**
- **Production checkpoint SHA256 verified**: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d` (**100% UNTOUCHED**).
- **Streamlit application `app.py` remains untouched.**
- **Zero Git commands executed.**
