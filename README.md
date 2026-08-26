# Human Fall Detection System

A research and engineering platform for multi-modal human fall detection intended for hospital patient monitoring applications.

---

## 📌 Project Overview

This repository houses the R&D documentation, dataset manifests, preprocessing specifications, splitting rules, and machine learning pipelines for our Human Fall Detection System.

The project evaluates human fall detection performance across three benchmark datasets:
1. **URFD (UR Fall Detection Dataset)** — Multi-modal RGB, 16-bit Depth, and Accelerometer data.
2. **Le2i Fall Detection Dataset** — Real-world video recordings with frame-level bounding boxes across 6 room environments.
3. **Multiple Cameras Fall Dataset (MCFD)** — Synchronized 8-camera multi-view scenario recordings.

---

## 📁 Repository Structure

```
.
├── .gitignore                      # Strictly excludes raw dataset files, models, and environments
├── README.md                       # Project overview and reproducibility documentation
├── requirements.txt                # Python package dependencies
├── src/                            # Reproducible research & utility scripts
│   ├── generate_manifest.py        # Programmatically indexes raw dataset metadata
│   ├── validate_manifest.py        # Read-only integrity & data leakage checker
│   └── design_splits.py            # Deterministic event-level split generator (Seed 42)
└── R&D/                            # Complete research & analysis documentation
    ├── split_strategy.md           # Leakage-safe event-level data splitting rules
    ├── Phase12_Experimental_Definition.md # Terminology, labels, and metrics definition
    ├── Dataset_Analysis/
    │   ├── URFD_analysis.md        # Detailed structural inspection of URFD
    │   ├── Le2i_analysis.md        # Detailed structural inspection of Le2i
    │   ├── MultiCamera_analysis.md # Detailed structural inspection of MCFD
    │   ├── Dataset_Comparison.md   # Comparative synthesis matrix across all 3 datasets
    │   ├── Ground_Truth_Verification.md # Ground-truth availability & evaluation capability
    │   ├── dataset_manifest.csv    # Authoritative dataset index (482 video records)
    │   ├── manifest_validation_report.md # Read-only validation & leakage check report
    │   └── manifest_audit_report.md     # Read-only manifest audit report
    └── Preprocessing_Experiments/
        ├── preprocessing_feasibility_report.md # Spatial & temporal standardization experiments
        └── representative_comparisons/         # Visual comparison sample artifacts
```

---

## 🛡️ Strict Dataset & Git Rules

Raw dataset directories (`URFD/`, `Le2i/`, `dataset/`), raw video files (`.mp4`, `.avi`), extracted frames, large processed data, model checkpoints, `.venv`, and secret key files **are strictly excluded from Git tracking via `.gitignore`**.

Local datasets remain stored locally outside version control to maintain a clean, lightweight, and reproducible code repository.

---

## 🔬 Initial Experimental Baseline Target

- **Dataset**: URFD (RGB Visual Stream)
- **Task**: Binary Event-Level Fall Detection (Fall vs. ADL)
- **Spatial Normalization**: $320 \times 240$ pixels (Lanczos antialiased area downscaling)
- **Temporal Normalization**: 25.0 FPS (Nearest-neighbor timestamp matching)
- **Temporal Window**: $W = 50$ frames (2.0s), Stride $S = 25$ frames (1.0s, 50% overlap)
- **Splitting**: Event-Level Group Split (`seed = 42`): 49 Train events, 10 Val events, 11 Test events.

---

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/Nekilesh001/human-fall-detection-system.git
   cd human-fall-detection-system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Generate or validate dataset manifests locally:
   ```bash
   python src/generate_manifest.py
   python src/validate_manifest.py
   ```
