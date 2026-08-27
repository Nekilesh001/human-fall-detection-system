# URFD RGB Baseline Feature Precomputation Report

## 1. Executive Summary & Motivation
During initial baseline training, execution was observed taking ~9.7 minutes per epoch (~8.12 hours for 50 epochs) even with GPU acceleration available. Diagnostic analysis revealed that:
1. **Redundant ResNet-18 Evaluation**: The ImageNet-pretrained ResNet-18 spatial backbone is strictly frozen (`requires_grad=False`). Re-evaluating 13,000 RGB frames ($260 \text{ windows} \times 50 \text{ frames}$) through ResNet-18 on every single epoch was wasteful.
2. **PCIe & I/O Overhead**: Pushing ~184 MB of raw float32 image tensors per batch over PCIe from CPU RAM to GPU VRAM accounted for **48.5% of total batch time**, and disk I/O accounted for **11.7%**.

By precomputing the 512-dimensional ResNet-18 embeddings for all 360 temporal windows ONCE and storing them as compressed `.npz` feature arrays (`(50, 512)` float32), we decoupled feature extraction from MLP classifier training without altering a single aspect of the scientific baseline contract.

---

## 2. Diagnostic & Profiling Findings
The initial execution diagnostic on the `NVIDIA GeForce RTX 4060 Laptop GPU` measured the following per-batch breakdown ($B=4$, 200 frames/batch):

| Pipeline Stage | Avg Time per Batch ($B=4$) | Share of Batch Time |
| :--- | :---: | :---: |
| **Data Loading (Disk I/O)** | 1,052.4 ms | 11.7% |
| **CPU $\to$ GPU PCIe Transfer** | 4,359.2 ms | 48.5% |
| **ResNet-18 Forward Pass** | 3,482.4 ms | 38.7% |
| **Temporal Pooling (Mean+Std)** | 5.6 ms | < 0.1% |
| **Classifier Forward Pass** | 41.1 ms | 0.5% |
| **Backward Pass** | 32.2 ms | 0.4% |
| **Optimizer Step** | 19.7 ms | 0.2% |
| **TOTAL Batch Execution Time** | **8,992.5 ms (~9.0s)** | **100.0%** |

- **ResNet-18 Fraction**: ResNet-18 pass + CPU-to-GPU transfer accounted for **87.2%** of execution latency.
- **Backbone Freezing Verification**: ResNet-18 backbone has 60 parameter tensors, **0 trainable (`requires_grad=False`)**. Classifier head has 4 parameter tensors, **4 trainable (`requires_grad=True`)**, total **65,730 trainable parameters**.

---

## 3. Implementation Details
The two-stage feature precomputation pipeline was implemented across the codebase:
1. **Extraction Script**: `src/precompute_features.py` reads `processed_data/URFD_RGB_baseline/processed_manifest.csv` (360 samples).
2. **Image Preprocessing**: Converts uint8 frames `(50, 240, 320, 3)` to float32 scaled to $[0, 1]$, applies standard ImageNet mean $[0.485, 0.456, 0.406]$ and std $[0.229, 0.224, 0.225]$.
3. **Extraction Mode**: Runs ResNet-18 inside `torch.no_grad()` in `eval()` mode on `cuda:0`.
4. **Feature Storage**: Saves per-window feature arrays of shape `(50, 512)` float32 to `processed_data/URFD_RGB_baseline/features/{partition}/{window_id}_features.npz`.
5. **Feature Manifest**: Generates `processed_data/URFD_RGB_baseline/processed_features_manifest.csv` containing sample metadata (`window_id`, `event_id`, `video_id`, `camera_id`, `partition`, `label`, `processed_feature_path`).
6. **Feature Dataset Loader**: Implemented `URFDRGBFeatureDataset` in `src/dataset.py`.
7. **Feature Classifier Model**: Implemented `URFDRGBFeatureBaseline` in `src/model.py` (65,730 trainable params).

---

## 4. Feature Representation & Storage
- **Feature Tensor Dimensions**: $(50, 512)$ float32 array per window.
- **Total Processed Samples**: 360 windows (18,000 total frame vectors).
- **Total Feature Dataset Size**: **32.26 MB** (down from 2.9 GB raw RGB dataset).
- **Average Feature File Size**: **91.75 KB** per `.npz` file.
- **One-Time Extraction Time**: **113.05 seconds** (314.0 ms/window).

---

## 5. Numerical Equivalence Verification
To ensure mathematical equivalence between on-the-fly ResNet-18 feature extraction and precomputed feature loading, 5 temporal windows across Train, Val, and Test partitions were sampled and compared via `src/validate_feature_precomputation.py`:

$$\Delta_{\text{max}} = \max | \mathbf{F}_{\text{live}} - \mathbf{F}_{\text{precomputed}} |$$

- **Maximum Absolute Difference**: `0.00000000` ($0.0$)
- **Mean Absolute Difference**: `0.00000000` ($0.0$)
- **`np.allclose(atol=1e-5)` Result**: **PASS (`True`)**
- **Conclusion**: Precomputed features are 100% bit-exact identical to on-the-fly extraction.

---

## 6. Leakage Validation & Partition Audit
A strict leakage audit was conducted via `src/validate_feature_precomputation.py`:
- **Record Counts**: 360 total records (**Train = 260**, **Val = 43**, **Test = 57**).
- **Event-Level Partition Disjointness**:
  - $\text{Train Events} \cap \text{Val Events} = \emptyset$ (0 overlap)
  - $\text{Train Events} \cap \text{Test Events} = \emptyset$ (0 overlap)
  - $\text{Val Events} \cap \text{Test Events} = \emptyset$ (0 overlap)
- **Camera Consistency**: Preserved 100%.
- **Original RGB Data Integrity**: 360 raw `.npz` files remain unchanged and intact.

---

## 7. Pipeline Performance Comparison (Old vs New Measured)

| Metric | OLD Pipeline (Raw RGB Frames) | NEW Pipeline (Precomputed Features) | Measured Improvement |
| :--- | :---: | :---: | :---: |
| **Per-Batch Input Payload** | ~184.3 MB | ~0.4 MB | **460x reduction** |
| **ResNet-18 Execution** | Every epoch (65 batches/epoch) | ONCE (113.05s total extraction) | **Eliminated from training** |
| **Data Load + Transfer Time** | 5,411.6 ms / batch | < 2.0 ms / batch | **> 2,500x faster I/O** |
| **1-Epoch Training Time** | ~584.5 seconds (~9.7 min) | **< 0.2 seconds** | **> 2,900x speedup** |
| **Inference Latency** | ~891 ms / window | **0.56 ms / window** | **> 1,500x faster inference** |
| **50-Epoch Total Time** | ~29,226 seconds (**~8.12 hours**) | **< 8.0 seconds total** (+ 113s one-time) | **Reduced from 8.1 hrs to ~2 min total** |

---

## 8. 1-Epoch Feature Benchmark Validation Results
- **Training Epoch 1**: Train Loss = `0.4649`, Train Acc = `0.7962` | Val Loss = `0.1409`, Val Acc = `0.9767`, Val F1 = `0.9474`, Val Spec = `1.0000`
- **Validation Threshold Selection ($\tau^*$)**: $\tau^* = 0.25$ (Val F1 = `1.0000`)
- **Test Evaluation @ Default ($\tau = 0.50$)**:
  - Accuracy: `1.0000` | Precision: `1.0000` | Sensitivity: `1.0000` | Specificity: `1.0000` | F1: `1.0000`
  - Confusion Matrix: `[[33, 0], [0, 24]]`
- **Test Evaluation @ Threshold ($\tau^* = 0.25$)**:
  - Accuracy: `0.9825` | Precision: `0.9600` | Sensitivity: `1.0000` | Specificity: `0.9697` | F1: `0.9796`
  - Confusion Matrix: `[[32, 1], [0, 24]]`
- **Inference Speed**: **0.56 ms / window** (Equivalent to 89,286 FPS throughput).

---

## 9. Limitations & Scope Guidelines
- **Frozen Backbone Only**: Feature precomputation is valid *only* when the spatial backbone is strictly frozen (`requires_grad=False`). If fine-tuning ResNet-18 is explored in future milestones, features must be re-extracted or computed on-the-fly.
- **Fixed Preprocessing**: If input resolution or normalization parameters change in future experiments, feature precomputation must be re-run.

---

## 10. Baseline Contract Integrity Confirmation
- **Event Split**: Frozen & Unchanged (47 Train, 9 Val, 11 Test).
- **Temporal Window**: $W=50$ frames, $S=25$ stride, 25 FPS.
- **Model Architecture**: Frozen ResNet-18 + Temporal Mean/Std Pooling + 2-layer MLP (65,730 trainable params).
- **Class Weights**: Calculated strictly from Train split ($w_{\text{normal}} = 0.7386, w_{\text{fall}} = 1.5476$).
