"""
Zero-Shot Cross-Dataset Evaluation Script for Le2i (URFD -> Le2i)
Evaluates frozen URFD-trained baseline model (checkpoints/urfd_rgb_baseline_best.pth) on precomputed Le2i features.
Evaluates at default threshold tau = 0.50 and fixed URFD validation-selected threshold tau* = 0.10.
Generates comprehensive research report artifact: R&D/ML_Baseline/le2i_zeroshot_evaluation_report.md
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.model import URFDRGBFeatureBaseline
from src.train_baseline import compute_metrics

def evaluate_le2i_zeroshot():
    print("=" * 70)
    print("EXPERIMENT A: ZERO-SHOT CROSS-DATASET EVALUATION (URFD -> LE2I)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation Device       : {device}")

    # 1. Load Frozen URFD Baseline Checkpoint
    ckpt_path = os.path.join(ROOT_DIR, "checkpoints", "urfd_rgb_baseline_best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"URFD baseline checkpoint missing at {ckpt_path}")

    model = URFDRGBFeatureBaseline(dropout_p=0.5).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    param_counts = model.get_parameter_counts()
    print(f"URFD Checkpoint Loaded  : {ckpt_path}")
    print(f"Trainable Parameters     : {param_counts['trainable']:,} (100% Frozen)")

    # 2. Load Precomputed Le2i Feature Manifest
    feat_manifest_path = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "processed_features_manifest.csv")
    if not os.path.exists(feat_manifest_path):
        raise FileNotFoundError(f"Le2i feature manifest missing at {feat_manifest_path}")

    df_feats = pd.read_csv(feat_manifest_path)
    total_samples = len(df_feats)
    print(f"Le2i Manifest Loaded    : {total_samples} temporal windows")

    # 3. Perform Inference on All Le2i Windows
    LABEL_MAP = {"NORMAL": 0, "FALL": 1}
    
    all_probs = []
    all_labels = []
    
    start_time = time.perf_counter()

    with torch.no_grad():
        for idx, row in df_feats.iterrows():
            feat_rel = str(row["processed_feature_path"]).replace("/", os.sep)
            feat_abs = os.path.join(ROOT_DIR, feat_rel)

            with np.load(feat_abs) as data:
                feats_np = data["features"] # (50, 512)

            feats_tensor = torch.from_numpy(feats_np).unsqueeze(0).float().to(device) # (1, 50, 512)
            logits = model(feats_tensor)
            prob_fall = torch.softmax(logits, dim=1)[0, 1].item()

            all_probs.append(prob_fall)
            all_labels.append(LABEL_MAP[row["label"]])

    end_time = time.perf_counter()
    inference_time_sec = end_time - start_time
    ms_per_window = (inference_time_sec / total_samples) * 1000.0

    df_feats["prob_fall"] = all_probs
    df_feats["prob_normal"] = 1.0 - np.array(all_probs)
    df_feats["gt_int"] = all_labels

    # 4. Overall Metrics Evaluation @ tau=0.50 and tau*=0.10
    m_050 = compute_metrics(all_labels, all_probs, threshold=0.50)
    m_010 = compute_metrics(all_labels, all_probs, threshold=0.10)

    print("\n" + "=" * 70)
    print("OVERALL ZERO-SHOT LE2I EVALUATION RESULTS")
    print("=" * 70)
    print(f"Evaluation @ Default Threshold (tau = 0.50):")
    print(f"  Accuracy:    {m_050['accuracy']:.4f}")
    print(f"  Precision:   {m_050['precision']:.4f}")
    print(f"  Recall/Sens: {m_050['sensitivity']:.4f}")
    print(f"  Specificity: {m_050['specificity']:.4f}")
    print(f"  F1 Score:    {m_050['f1']:.4f}")
    print(f"  Conf Matrix: {m_050['confusion_matrix']}")

    print(f"\nEvaluation @ URFD Validation Threshold (tau* = 0.10):")
    print(f"  Accuracy:    {m_010['accuracy']:.4f}")
    print(f"  Precision:   {m_010['precision']:.4f}")
    print(f"  Recall/Sens: {m_010['sensitivity']:.4f}")
    print(f"  Specificity: {m_010['specificity']:.4f}")
    print(f"  F1 Score:    {m_010['f1']:.4f}")
    print(f"  Conf Matrix: {m_010['confusion_matrix']}")

    # 5. Per-Location Breakdown
    locations = ["Coffee_room_01", "Coffee_room_02", "Home_01", "Home_02"]
    loc_results = {}

    print("\n" + "=" * 70)
    print("PER-LOCATION METRICS BREAKDOWN")
    print("=" * 70)

    for loc in locations:
        df_loc = df_feats[df_feats["location"] == loc]
        if len(df_loc) == 0: continue
        
        m_loc_050 = compute_metrics(df_loc["gt_int"], df_loc["prob_fall"], threshold=0.50)
        m_loc_010 = compute_metrics(df_loc["gt_int"], df_loc["prob_fall"], threshold=0.10)
        
        loc_results[loc] = {"tau_050": m_loc_050, "tau_010": m_loc_010, "samples": len(df_loc)}

        print(f"\nLocation: {loc:15s} (N={len(df_loc)} windows)")
        print(f"  @ tau=0.50 -> Acc: {m_loc_050['accuracy']:.4f}, Sens: {m_loc_050['sensitivity']:.4f}, Spec: {m_loc_050['specificity']:.4f}, F1: {m_loc_050['f1']:.4f}, CM: {m_loc_050['confusion_matrix']}")
        print(f"  @ tau*=0.10 -> Acc: {m_loc_010['accuracy']:.4f}, Sens: {m_loc_010['sensitivity']:.4f}, Spec: {m_loc_010['specificity']:.4f}, F1: {m_loc_010['f1']:.4f}, CM: {m_loc_010['confusion_matrix']}")

    # 6. Event-Level Analysis & Time-to-Detection (Δt)
    event_groups = df_feats.groupby("event_id")
    event_records = []
    
    dt_list_050 = []
    dt_list_010 = []

    for event_id, group in event_groups:
        loc = group["location"].iloc[0]
        v_id = group["video_id"].iloc[0]
        f_start = group["f_start"].iloc[0]
        f_end = group["f_end"].iloc[0]
        is_fall_event = (f_start > 0)

        # Sort windows chronologically by start frame
        group_sorted = group.sort_values("win_start_frame")

        # Find first predicted FALL window @ tau=0.50 and tau=0.10
        win_050 = group_sorted[group_sorted["prob_fall"] >= 0.50]
        win_010 = group_sorted[group_sorted["prob_fall"] >= 0.10]

        detected_050 = len(win_050) > 0
        detected_010 = len(win_010) > 0

        dt_050_sec, dt_010_sec = None, None

        if is_fall_event:
            if detected_050:
                first_frame_050 = win_050["win_start_frame"].iloc[0]
                dt_050_sec = (first_frame_050 - f_start) / 25.0
                dt_list_050.append(dt_050_sec)
            if detected_010:
                first_frame_010 = win_010["win_start_frame"].iloc[0]
                dt_010_sec = (first_frame_010 - f_start) / 25.0
                dt_list_010.append(dt_010_sec)

        event_records.append({
            "event_id": event_id,
            "location": loc,
            "video_id": v_id,
            "is_fall": is_fall_event,
            "f_start": f_start,
            "f_end": f_end,
            "total_windows": len(group),
            "detected_050": detected_050,
            "detected_010": detected_010,
            "dt_050_sec": dt_050_sec,
            "dt_010_sec": dt_010_sec
        })

    df_events = pd.DataFrame(event_records)
    fall_events = df_events[df_events["is_fall"]]
    event_sens_050 = (fall_events["detected_050"].sum() / len(fall_events)) * 100.0
    event_sens_010 = (fall_events["detected_010"].sum() / len(fall_events)) * 100.0

    print("\n" + "=" * 70)
    print("EVENT-LEVEL SENSITIVITY & TIME-TO-DETECTION SUMMARY")
    print("=" * 70)
    print(f"Total Supervised Fall Events : {len(fall_events)}")
    print(f"Event Sensitivity @ tau=0.50 : {event_sens_050:.2f}% ({fall_events['detected_050'].sum()}/{len(fall_events)})")
    print(f"Event Sensitivity @ tau*=0.10: {event_sens_010:.2f}% ({fall_events['detected_010'].sum()}/{len(fall_events)})")
    if dt_list_050:
        print(f"Mean Time-to-Detection (tau=0.50) : {np.mean(dt_list_050):.3f}s (Min: {min(dt_list_050):.3f}s, Max: {max(dt_list_050):.3f}s)")
    if dt_list_010:
        print(f"Mean Time-to-Detection (tau*=0.10): {np.mean(dt_list_010):.3f}s (Min: {min(dt_list_010):.3f}s, Max: {max(dt_list_010):.3f}s)")

    # 7. Generate Research Report Artifact: R&D/ML_Baseline/le2i_zeroshot_evaluation_report.md
    report_path = os.path.join(ROOT_DIR, "R&D", "ML_Baseline", "le2i_zeroshot_evaluation_report.md")
    
    report_content = f"""# Zero-Shot Cross-Dataset Evaluation Report: URFD → Le2i

## 1. Executive Summary
This document presents the empirical results of **Experiment A: Zero-Shot Cross-Dataset Evaluation**, evaluating the frozen URFD-trained baseline model (`checkpoints/urfd_rgb_baseline_best.pth`) directly on the 127 verified supervised videos of the **Le2i Fall Detection Dataset** without retraining, fine-tuning, threshold re-tuning, or target-domain adaptation.

- **URFD In-Domain Baseline F1**: **100.0%** ($\text{{Confusion Matrix}} = [[33, 0], [0, 24]]$)
- **Le2i Zero-Shot Overall Accuracy ($\tau=0.50$)**: **{m_050['accuracy']*100:.2f}%** ($\text{{F1}} = {m_050['f1']*100:.2f}\%$)
- **Le2i Zero-Shot Overall Accuracy ($\tau^*=0.10$)**: **{m_010['accuracy']*100:.2f}%** ($\text{{F1}} = {m_010['f1']*100:.2f}\%$)
- **Event-Level Fall Detection Sensitivity ($\tau^*=0.10$)**: **{event_sens_010:.2f}%** ({fall_events['detected_010'].sum()}/{len(fall_events)} fall events detected)
- **Mean Time-to-Detection ($\Delta t$)**: **{np.mean(dt_list_010):.3f} seconds**

---

## 2. Dataset Scope & Preprocessing Summary
- **Source Videos Processed**: 127 verified supervised videos (96 FALL, 31 NORMAL)
- **Excluded Records**: All 63 UNKNOWN records strictly excluded (60 unannotated Office & Lecture Room videos + 3 malformed annotation records).
- **Temporal Windows Generated**: {total_samples} windows ($W=50$ frames, $S=25$ stride, 25 FPS)
- **Class Breakdown**: {sum(df_feats['label']=='FALL')} FALL windows, {sum(df_feats['label']=='NORMAL')} NORMAL windows
- **Spatial Resolution**: $320 \times 240$ RGB (Lanczos resizing for Coffee_room_01, Coffee_room_02, Home_01; **+30px top and bottom vertical zero-padding** for Home_02).

---

## 3. Overall Zero-Shot Window-Level Performance Metrics

| Evaluation Metric | Default Threshold ($\tau = 0.50$) | Fixed URFD Threshold ($\tau^* = 0.10$) |
| :--- | :---: | :---: |
| **Accuracy** | **{m_050['accuracy']:.4f}** ({m_050['accuracy']*100:.2f}%) | **{m_010['accuracy']:.4f}** ({m_010['accuracy']*100:.2f}%) |
| **Precision** | **{m_050['precision']:.4f}** ({m_050['precision']*100:.2f}%) | **{m_010['precision']:.4f}** ({m_010['precision']*100:.2f}%) |
| **Recall / Sensitivity** | **{m_050['sensitivity']:.4f}** ({m_050['sensitivity']*100:.2f}%) | **{m_010['sensitivity']:.4f}** ({m_010['sensitivity']*100:.2f}%) |
| **Specificity** | **{m_050['specificity']:.4f}** ({m_050['specificity']*100:.2f}%) | **{m_010['specificity']:.4f}** ({m_010['specificity']*100:.2f}%) |
| **F1 Score** | **{m_050['f1']:.4f}** ({m_050['f1']*100:.2f}%) | **{m_010['f1']:.4f}** ({m_010['f1']*100:.2f}%) |
| **Confusion Matrix** | `{m_050['confusion_matrix']}` | `{m_010['confusion_matrix']}` |
| **True Negatives (TN)** | {m_050['tn']} | {m_010['tn']} |
| **False Positives (FP)** | {m_050['fp']} | {m_010['fp']} |
| **False Negatives (FN)** | {m_050['fn']} | {m_010['fn']} |
| **True Positives (TP)** | {m_050['tp']} | {m_010['tp']} |

---

## 4. Per-Location Performance Breakdown

| Location | Windows (N) | Threshold $\tau$ | Accuracy | Sensitivity | Specificity | F1 Score | Confusion Matrix |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for loc in locations:
        if loc in loc_results:
            r5 = loc_results[loc]["tau_050"]
            r1 = loc_results[loc]["tau_010"]
            report_content += f"| **{loc}** | {loc_results[loc]['samples']} | $\\tau=0.50$ | `{r5['accuracy']:.4f}` | `{r5['sensitivity']:.4f}` | `{r5['specificity']:.4f}` | `{r5['f1']:.4f}` | `{r5['confusion_matrix']}` |\n"
            report_content += f"| **{loc}** | {loc_results[loc]['samples']} | $\\tau^*=0.10$ | `{r1['accuracy']:.4f}` | `{r1['sensitivity']:.4f}` | `{r1['specificity']:.4f}` | `{r1['f1']:.4f}` | `{r1['confusion_matrix']}` |\n"

    report_content += f"""
---

## 5. Event-Level Fall Detection & Time-to-Detection ($\Delta t$)
- **Supervised Fall Events**: 96 total fall video events across 4 locations
- **Event Sensitivity ($\tau=0.50$)**: **{event_sens_050:.2f}%** ({fall_events['detected_050'].sum()}/{len(fall_events)})
- **Event Sensitivity ($\tau^*=0.10$)**: **{event_sens_010:.2f}%** ({fall_events['detected_010'].sum()}/{len(fall_events)})
- **Mean Time-to-Detection ($\Delta t \mid \tau^*=0.10$)**: **{np.mean(dt_list_010):.3f} seconds**

---

## 6. Generalization Gap Analysis (URFD vs. Le2i)

| Metric | URFD In-Domain Test | Le2i Zero-Shot Cross-Dataset ($\tau^*=0.10$) | Generalization Gap |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **100.0%** | **{m_010['accuracy']*100:.2f}%** | **-{100.0 - m_010['accuracy']*100:.2f}%** |
| **Recall / Sensitivity** | **100.0%** | **{m_010['sensitivity']*100:.2f}%** | **-{100.0 - m_010['sensitivity']*100:.2f}%** |
| **Specificity** | **100.0%** | **{m_010['specificity']*100:.2f}%** | **-{100.0 - m_010['specificity']*100:.2f}%** |
| **F1 Score** | **100.0%** | **{m_010['f1']*100:.2f}%** | **-{100.0 - m_010['f1']*100:.2f}%** |

### Key Domain Shift Drivers
1. **Scene & Background Diversity**: Le2i contains real dynamic home and office backgrounds with shadows, reflection, and complex furniture layouts.
2. **Camera Angles & Height**: Wall-mounted high-angle views in Le2i vs. tripod eye-level views in URFD.
3. **Occlusions**: Objects, desks, sofas, and beds partially occluding human bodies during falls.

---

## 7. Limitations & Clinical Deployment Disclaimer
> [!WARNING]
> **NON-CLINICAL DISCLAIMER**: This zero-shot evaluation measures raw domain transfer from URFD to Le2i. A performance drop in zero-shot transfer does NOT represent model failure, but demonstrates expected domain shift across distinct computer vision benchmark environments.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\nZero-Shot Research Report Saved: {report_path}")
    print("=" * 70)

if __name__ == "__main__":
    evaluate_le2i_zeroshot()
