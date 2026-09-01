# Phase H1 — Validation Check 14 Investigation & Correction Report

> [!IMPORTANT]
> **IMMUTABLE BASELINE MANDATE & READ-ONLY SAFETY STATUS**  
> Baseline Model Path: `checkpoints/final_k1/final_production.pth`  
> Checkpoint SHA256: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d` (**IMMUTABLE & FROZEN**)  
> Policy Confirmation: **NO MODEL TRAINING WAS PERFORMED.**

---

## 1. Original Check 14 Failure Analysis
During initial validation of the generated Phase H1 unified dataset, Check 14 failed with the following assertion:

```text
AssertionError: Check 14 Fail: Duplicate sequences
```

---

## 2. Root Cause Investigation & Findings

### Exact Duplicate Sequence IDs Identified
Upon inspecting `processed_data/multi_dataset_k1/splits/grouping_metadata.csv`, 180 duplicate `sequence_id` entries were identified, occurring **exclusively within the Le2i dataset**.  
Examples:
- `video (1).avi` (occurred 4 times: Coffee_room_01, Home_01, Lecture_room, Office_01)
- `video (2).avi` (occurred 4 times)
- ...
- `video (60).avi` (occurred 2 times)

### Why the Duplicates Occurred
In `process_le2i_metadata()`, `video_id` was extracted as `os.path.basename(v_path)` (e.g. `video (1).avi`).  
`loc_id` was naively set to `parts[1]`. For path `Le2i/data/Coffee_room_01/Videos/video (1).avi`, `parts[1]` evaluated to the literal folder name `"data"`, causing `loc_id` to be `"data"` for **all** Le2i videos!  
As a result, `sequence_id = f"Le2i_{video_id}"` generated identical sequence IDs (`Le2i_video (1).avi`) across different room locations.

### Classification of Duplicates
This was an **actual parsing implementation bug** in extracting `loc_id` for Le2i videos.

---

## 3. Preprocessing & Grouping Key Fixes

### A. Location Parsing Correction
Updated `process_le2i_metadata()` in `src/build_multi_dataset_k1.py` to extract `parts[2]` (`Coffee_room_01`, `Home_01`, `Lecture_room`, `Office_01`) when `parts[1] == "data"`:
```python
loc_id = parts[2] if (len(parts) > 2 and parts[1] == "data") else (parts[1] if len(parts) > 1 else "Unknown")
sequence_id = f"Le2i_{loc_id}_{video_id}"
```
Result: All 190 Le2i videos now have **100% UNIQUE `sequence_id` values** (`Le2i_Coffee_room_01_video (1).avi`, `Le2i_Home_01_video (1).avi`, etc.).

### B. Canonical Group ID Design
Constructed explicit `group_id` metadata across all datasets:
- **Le2i**: `group_id = f"Le2i_{loc_id}_{video_id}"`
- **URFD**: `group_id = f"URFD_{seq_name}"` (groups synchronized camera views of same sequence)
- **Multicam**: `group_id = f"Multicam_{ch_name}"` (groups all 8 cameras `cam1..cam8` of `chuteXX` together into the same split group!).

---

## 4. Validator Modifications
Updated `src/validate_phase_h1_multi_dataset.py`:
1. Verified `df_grp["sequence_id"].is_unique` across all 452 video streams.
2. Verified `group_id` exists and is non-null for all 452 records.
3. Verified zero exact duplicate source video records (`dataset + video_path`).
4. Verified that all 8 camera streams for each of the 24 Multicam chute scenarios share the **EXACT SAME `group_id`**, guaranteeing **zero cross-camera physical-event leakage**.

---

## 5. Dataset Statistics After Correction

```text
===========================================================================
UNIFIED DATASET SUMMARY STATISTICS
===========================================================================
  Total Source Videos : 452
  Total Windows       : 6,780
  NORMAL Windows (0)  : 5,448 (80.35%)
  FALL Windows (1)    : 1,332 (19.65%)
  Le2i Windows        : 2,850 (42.0%)
  URFD Windows        : 1,050 (15.5%)
  Multicam Windows    : 2,880 (42.5%)
===========================================================================
```

---

## 6. Complete 25-Check Validation Suite Results

```text
===========================================================================
PHASE H1 — MULTI-DATASET DATASET VALIDATION AUDIT (25 CHECKS)
===========================================================================
  [PASS 1/25] Dataset Coverage                          : Le2i, URFD & Multicam Represented
  [PASS 2/25] Source Datasets Untouched                 : Raw Directories Preserved Intact
  [PASS 3/25] Production Checkpoint SHA256               : a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d
  [PASS 4/25] Feature Dimension                         : 187-D Validated
  [PASS 5/25] Feature Tensor Dtype                      : float32 Validated
  [PASS 6-7/25] Tensor Numeric Integrity                 : Zero NaN / Zero Inf Values
  [PASS 8-9/25] Temporal Windowing Parameters            : 50 Frames Length, 25 Frames Stride
  [PASS 10/25] Target Temporal Resolution                : Standardized 25.0 FPS
  [PASS 11-12/25] Sequence & Event Boundary Isolation     : Zero Cross-Video Window Leaks
  [PASS 13-14/25] Window & Grouping Deduplication         : Unique Sequence IDs & Group IDs Verified
  [PASS 15/25] URFD Duplicate Exclusion                 : fall-11-data (1).csv Successfully Excluded
  [PASS 16-17/25] Multicam Camera Grouping & Zero Leak    : All 8 Cameras Grouped per Chute Scenario (group_id)
  [PASS 18-19/25] Label Conventions & Threshold Policy   : Binary (0/1) @ 40% Fall Window Rule
  [PASS 20-21/25] Summary Statistics & Grouping Metadata  : Verified Generated Json & CSV Artifacts
  [PASS 22/25] Feature File Existence                   : 100% Referenced Files Exist on Disk
  [PASS 23/25] Feature Tensor Shape                     : (50, 187) Verified
  [PASS 24/25] Manifest Identifier Completeness          : Zero Null Identifiers
  [PASS 25/25] Production Application Integrity          : app.py & Streamlit Behavior Untouched
===========================================================================
PHASE H1 — ALL 25 VALIDATION CHECKS PASSED SUCCESSFULLY
===========================================================================
```

---

## 🔒 Mandatory Final Section

### A. VERIFIED FACTS
- Check 14 failure was caused by Le2i location parsing capturing `"data"` instead of room subdirectories.
- Total unique source video sequences: **452 sequences** (190 Le2i, 70 URFD, 192 Multicam).
- Production checkpoint SHA256: `a1ed0c9f028f8b039d5d0adc95aa8d3f7d86a4a32e516f6e0b06aebe207f735d` (**100% UNTOUCHED**).

### B. FIXES MADE
- Corrected Le2i location ID extraction in `src/build_multi_dataset_k1.py`.
- Added canonical `group_id` attribute to `grouping_metadata.csv` and `unified_window_manifest.csv`.
- Updated `src/validate_phase_h1_multi_dataset.py` to enforce `group_id` integrity and zero cross-camera scenario leakage.

### C. REMAINING RISKS
- Class imbalance: $80.35\%$ Normal / $19.65\%$ Fall windows. Recommend weighted BCE loss ($\text{pos\_weight} \approx 4.0$) during future candidate training.

### D. NEXT STEP
- Await user review and explicit approval before initiating Phase H2 candidate model training.
