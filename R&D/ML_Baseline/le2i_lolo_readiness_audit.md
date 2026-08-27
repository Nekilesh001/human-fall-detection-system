# Experiment B: Le2i Supervised LOLO Baseline Readiness Audit Report

## 1. Executive Summary & Audit Decision
- **Audit Target**: Implementation Readiness for Experiment B: Supervised In-Domain Le2i Baseline (4-Fold LOLO).
- **Audit Decision**: **READY FOR EXPERIMENT B TRAINING — NO TRAINING PERFORMED YET**
- **Audit Scope**: Non-modifying read-only verification of precomputed feature files, dataset manifests, fold partition boundaries, event isolation, class weight formulas, and model compatibility.

---

## 2. Dataset & Feature Integrity Audit

| Metric | Empirical Count | Target Count | Verification Status |
| :--- | :---: | :---: | :---: |
| **Supervised Videos Processed** | **127** | **127** | **PASS ✅** |
| **Total Precomputed Feature Windows** | **1,396** | **1,396** | **PASS ✅** |
| **Feature Tensor Shape** | `(50, 512)` | `(50, 512)` float32 | **PASS ✅** |
| **Missing Feature Files** | **0** | **0** | **PASS ✅** |
| **Excluded UNKNOWN Records** | **63** | **63** | **PASS ✅ (100% Excluded)** |

---

## 3. 4-Fold LOLO Partition & Class Weight Verification

| Fold Name | Outer Test Location | Outer Train Locations | Outer Train Windows | Outer Test Windows | Fold Class Weights ($w_{\text{normal}}, w_{\text{fall}}$) | Partition Leakage Check |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | `Coffee_room_01` | `Coffee_room_02`, `Home_01`, `Home_02` | 894 (159 F, 735 N) | 502 (172 F, 330 N) | $w_{\text{norm}} = 0.6082, w_{\text{fall}} = 2.8113$ | **PASSED (0 Leakage) ✅** |
| **Fold 2** | `Coffee_room_02` | `Coffee_room_01`, `Home_01`, `Home_02` | 986 (284 F, 702 N) | 410 (47 F, 363 N) | $w_{\text{norm}} = 0.7023, w_{\text{fall}} = 1.7359$ | **PASSED (0 Leakage) ✅** |
| **Fold 3** | `Home_01` | `Coffee_room_01`, `Coffee_room_02`, `Home_02` | 1,157 (241 F, 916 N) | 239 (90 F, 149 N) | $w_{\text{norm}} = 0.6316, w_{\text{fall}} = 2.4004$ | **PASSED (0 Leakage) ✅** |
| **Fold 4** | `Home_02` | `Coffee_room_01`, `Coffee_room_02`, `Home_01` | 1,151 (309 F, 842 N) | 245 (22 F, 223 N) | $w_{\text{norm}} = 0.6835, w_{\text{fall}} = 1.8625$ | **PASSED (0 Leakage) ✅** |

---

## 4. Inner Validation Split Strategy Audit

Within each fold, an **Inner Event-Level Validation Split** is constructed from outer training events (80% Inner Train, 20% Inner Validation):

- **Fold 1**: Inner Train = 64 events (648 windows) | Inner Val = 16 events (246 windows)
- **Fold 2**: Inner Train = 86 events (807 windows) | Inner Val = 21 events (179 windows)
- **Fold 3**: Inner Train = 78 events (920 windows) | Inner Val = 19 events (237 windows)
- **Fold 4**: Inner Train = 78 events (916 windows) | Inner Val = 19 events (235 windows)

- **Isolation Audit**: $\text{Inner Train Events} \cap \text{Inner Val Events} = \emptyset$ and $\text{Inner Events} \cap \text{Outer Test Events} = \emptyset$. **Zero Event Leakage Verified ✅**.

---

## 5. Model Architecture & Checkpoint Safety Audit
- **Classifier Head**: `URFDRGBFeatureBaseline`
- **Trainable Parameters**: **65,730** (Exact match)
- **Frozen Backbone**: ResNet-18 feature extraction path is 100% frozen.
- **Reference Model Safety**: URFD baseline model (`checkpoints/urfd_rgb_baseline_best.pth`) and URFD datasets remain 100% read-only and untouched.

---

## 6. Files Created for Design Phase
1. [`R&D/ML_Baseline/le2i_lolo_design.md`](file:///d:/ONE_DATA/Fall%20detection/R&D/ML_Baseline/le2i_lolo_design.md): Research design document.
2. [`R&D/ML_Baseline/le2i_lolo_readiness_audit.md`](file:///d:/ONE_DATA/Fall%20detection/R&D/ML_Baseline/le2i_lolo_readiness_audit.md): Readiness audit report artifact.
3. `scratch/audit_le2i_lolo_readiness.py`: Empirical readiness verification script.

---

## 7. Final Git Status Audit (`dev` branch)

```text
Current Branch: dev
Tracking Status: Up to date with 'origin/dev'

Untracked files:
  R&D/ML_Baseline/
  src/dataset.py
  src/evaluate_le2i_zeroshot.py
  src/model.py
  src/precompute_features.py
  src/precompute_le2i_features.py
  src/preprocess_le2i.py
  src/train_baseline.py
  src/validate_feature_precomputation.py
  src/validate_le2i_features.py
  src/validate_le2i_preprocessing.py

nothing added to commit but untracked files present
```

- **Branch**: `dev` (`main` untouched).
- **Git Operations**: **No commits or pushes performed.**

---

## 8. Final Verdict
**READY FOR EXPERIMENT B TRAINING — NO TRAINING PERFORMED YET.**
