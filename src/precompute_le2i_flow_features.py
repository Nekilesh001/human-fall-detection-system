"""
Precompute Le2i Optical Flow Features using Polar Component Encoding
Extracts Farneback optical flow for all 1,396 temporal windows of the 127 supervised Le2i videos.
Applies magnitude thresholding (M < 0.5 px/frame -> 0) and Polar 3-Channel Mapping [dx_norm, dy_norm, mag_norm].
Extracts 512-dim ResNet-18 features per flow frame -> (49, 512) float32 per window.
"""

import os
import sys
import time
import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

def encode_polar_flow(flow, mag_thresh=0.5):
    """
    Polar Component Mapping:
    Channel 0: dx_norm = dx / 320.0 + 0.5
    Channel 1: dy_norm = dy / 240.0 + 0.5
    Channel 2: mag_norm = mag / 20.0
    flow: (H, W, 2) float32
    Returns: (3, H, W) float32 tensor in [0, 1] range
    """
    dx = flow[:, :, 0]
    dy = flow[:, :, 1]
    mag = np.sqrt(dx**2 + dy**2)
    
    # Apply magnitude noise thresholding
    mag[mag < mag_thresh] = 0.0
    dx[mag < mag_thresh] = 0.0
    dy[mag < mag_thresh] = 0.0

    # Normalize by frame dimensions (320x240)
    dx_norm = np.clip((dx / 320.0 + 0.5), 0.0, 1.0)
    dy_norm = np.clip((dy / 240.0 + 0.5), 0.0, 1.0)
    mag_norm = np.clip(mag / 20.0, 0.0, 1.0)

    tensor_3c = np.stack([dx_norm, dy_norm, mag_norm], axis=0).astype(np.float32)
    return torch.from_numpy(tensor_3c)

def precompute_flow_features():
    print("=" * 70)
    print("PRECOMPUTING LE2I OPTICAL FLOW FEATURES (POLAR MAPPING)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device}")

    # Load frozen ImageNet ResNet-18
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.fc = nn.Identity()
    resnet.eval().to(device)
    for p in resnet.parameters():
        p.requires_grad = False

    normalize_transform = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    manifest_path = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "processed_features_manifest.csv")
    assert os.path.exists(manifest_path), f"Processed manifest missing at {manifest_path}"
    df_manifest = pd.read_csv(manifest_path)

    out_flow_feat_dir = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "flow_features")
    os.makedirs(out_flow_feat_dir, exist_ok=True)

    flow_manifest_rows = []
    start_time = time.perf_counter()

    grouped = df_manifest.groupby("video_id")
    total_videos = len(grouped)
    total_windows_processed = 0

    print(f"Processing {total_videos} supervised videos ({len(df_manifest)} total window records)...")

    for vid_idx, (video_id, grp) in enumerate(grouped, start=1):
        sample_row = grp.iloc[0]
        raw_vid_rel = str(sample_row["raw_video_path"]).replace("/", os.sep)
        raw_vid_abs = os.path.join(ROOT_DIR, raw_vid_rel)

        assert os.path.exists(raw_vid_abs), f"Video missing: {raw_vid_abs}"

        cap = cv2.VideoCapture(raw_vid_abs)
        ret, prev_frame = cap.read()
        assert ret, f"Failed to read first frame of {video_id}"

        # Standardize spatial resolution to 320x240
        prev_gray = cv2.cvtColor(cv2.resize(prev_frame, (320, 240)), cv2.COLOR_BGR2GRAY)
        frame_buffer = [prev_gray]

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(cv2.resize(frame, (320, 240)), cv2.COLOR_BGR2GRAY)
            frame_buffer.append(gray)
        cap.release()

        # Compute window flow features
        for _, win_row in grp.iterrows():
            win_id = win_row["window_id"]
            start_f = int(win_row["win_start_frame"]) - 1 # 0-indexed
            end_f   = int(win_row["win_end_frame"])       # 50 frames

            # Extract window flow pairs (49 pairs)
            win_flow_tensors = []
            for f_idx in range(start_f, end_f - 1):
                idx0 = max(0, min(f_idx, len(frame_buffer) - 1))
                idx1 = max(0, min(f_idx + 1, len(frame_buffer) - 1))

                g0 = frame_buffer[idx0]
                g1 = frame_buffer[idx1]

                flow = cv2.calcOpticalFlowFarneback(
                    g0, g1, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )

                t_polar = encode_polar_flow(flow, mag_thresh=0.5)
                t_norm = normalize_transform(t_polar)
                win_flow_tensors.append(t_norm)

            # Stack into (49, 3, 240, 320)
            batch_flow = torch.stack(win_flow_tensors, dim=0).to(device)

            with torch.no_grad():
                feats_flow = resnet(batch_flow) # (49, 512)

            feats_np = feats_flow.cpu().numpy().astype(np.float32)

            # Check NaNs / Infs
            assert not np.isnan(feats_np).any(), f"NaN found in flow features for {win_id}"
            assert not np.isinf(feats_np).any(), f"Inf found in flow features for {win_id}"
            assert feats_np.shape == (49, 512), f"Invalid flow shape {feats_np.shape} for {win_id}"

            # Save NPZ
            feat_filename = f"{win_id}_flow_features.npz"
            feat_abs_path = os.path.join(out_flow_feat_dir, feat_filename)
            np.savez_compressed(feat_abs_path, features=feats_np)

            rel_feat_path = os.path.relpath(feat_abs_path, ROOT_DIR).replace("\\", "/")

            win_dict = win_row.to_dict()
            win_dict["flow_feature_path"] = rel_feat_path
            flow_manifest_rows.append(win_dict)
            total_windows_processed += 1

        if vid_idx % 25 == 0 or vid_idx == total_videos:
            elapsed = time.perf_counter() - start_time
            print(f"  Processed {vid_idx}/{total_videos} videos ({total_windows_processed}/1396 flow windows) in {elapsed:.1f}s")

    # Save manifest
    df_flow_manifest = pd.DataFrame(flow_manifest_rows)
    out_manifest_path = os.path.join(ROOT_DIR, "processed_data", "Le2i_baseline", "processed_flow_features_manifest.csv")
    df_flow_manifest.to_csv(out_manifest_path, index=False)

    total_time = time.perf_counter() - start_time
    print("\n" + "=" * 70)
    print("LE2I OPTICAL FLOW FEATURE PRECOMPUTATION COMPLETE")
    print("=" * 70)
    print(f"Total Extraction Time   : {total_time:.2f} seconds")
    print(f"Flow Windows Saved      : {total_windows_processed} NPZ files")
    print(f"Flow Manifest Saved     : {out_manifest_path}")

if __name__ == "__main__":
    precompute_flow_features()
