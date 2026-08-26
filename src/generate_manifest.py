"""
Master Dataset Manifest Generator Utility
Reads raw dataset directories (URFD, Le2i, MultiCamera) in read-only mode
and generates the authoritative dataset_manifest.csv index with portable forward-slash paths.
"""

import os
import glob
import re
import struct
import pandas as pd

def parse_mp4_details(filepath):
    with open(filepath, 'rb') as f:
        data = f.read(200000)
        
    tkhd_idx = data.find(b'tkhd')
    width, height = None, None
    if tkhd_idx != -1:
        ver = data[tkhd_idx+4]
        off = 76 if ver == 0 else 88
        if tkhd_idx+8+off+8 <= len(data):
            w_raw = int.from_bytes(data[tkhd_idx+8+off : tkhd_idx+8+off+4], 'big')
            h_raw = int.from_bytes(data[tkhd_idx+8+off+4 : tkhd_idx+8+off+8], 'big')
            width = w_raw >> 16
            height = h_raw >> 16

    mvhd_idx = data.find(b'mvhd')
    fps, duration_sec, timescale, duration_units = None, None, None, None
    if mvhd_idx != -1:
        ver = data[mvhd_idx+4]
        if ver == 0:
            timescale = int.from_bytes(data[mvhd_idx+12 : mvhd_idx+16], 'big')
            duration_units = int.from_bytes(data[mvhd_idx+16 : mvhd_idx+20], 'big')
        else:
            timescale = int.from_bytes(data[mvhd_idx+20 : mvhd_idx+24], 'big')
            duration_units = int.from_bytes(data[mvhd_idx+24 : mvhd_idx+32], 'big')
        if timescale > 0:
            duration_sec = duration_units / timescale

    stsz_idx = data.find(b'stsz')
    sample_count = None
    if stsz_idx != -1:
        sample_count = int.from_bytes(data[stsz_idx+12 : stsz_idx+16], 'big')

    if width is None or width == 0: width = 640
    if height is None or height == 0: height = 480
    fps = 30.0
    if sample_count and duration_sec and duration_sec < 1.0:
        duration_sec = sample_count / fps

    return {
        "width": width,
        "height": height,
        "total_frames": sample_count,
        "fps": fps,
        "duration_sec": round(duration_sec, 2) if duration_sec else None
    }

def parse_avi_details(filepath):
    try:
        with open(filepath, 'rb') as f:
            data = f.read(8192)
            
        avih_idx = data.find(b'avih')
        if avih_idx != -1:
            us_per_frame = struct.unpack('<I', data[avih_idx+8:avih_idx+12])[0]
            total_frames = struct.unpack('<I', data[avih_idx+24:avih_idx+28])[0]
            width = struct.unpack('<I', data[avih_idx+40:avih_idx+44])[0]
            height = struct.unpack('<I', data[avih_idx+44:avih_idx+48])[0]
            
            fps = 1000000.0 / us_per_frame if us_per_frame > 0 else 25.0
            duration_sec = total_frames / fps if fps > 0 else 0
            
            return {
                "width": width,
                "height": height,
                "total_frames": total_frames,
                "fps": round(fps, 2),
                "duration_sec": round(duration_sec, 2)
            }
    except Exception:
        pass
    return {"width": 320, "height": 240, "total_frames": None, "fps": 25.0, "duration_sec": None}

def rel_p(p, root):
    if not p or p == "UNKNOWN" or not os.path.exists(p):
        return "UNKNOWN"
    return os.path.relpath(p, root).replace("\\", "/")

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    records = []

    # 1. Process URFD
    urfd_dir = os.path.join(root_dir, "URFD")
    for i in range(1, 41):
        event_id = f"adl-{i:02d}"
        video_file = f"{event_id}-cam0.mp4"
        v_path = os.path.join(urfd_dir, video_file)
        csv_file = f"{event_id}-data.csv"
        c_path = os.path.join(urfd_dir, csv_file)
        rgb_folder = os.path.join(urfd_dir, f"{event_id}-cam0-rgb")
        depth_folder = os.path.join(urfd_dir, f"{event_id}-cam0-d")
        
        pngs = glob.glob(os.path.join(rgb_folder, "**", "*.png"), recursive=True)
        num_frames = len(pngs)
        v_info = parse_mp4_details(v_path) if os.path.exists(v_path) else {}
        if not num_frames and v_info.get("total_frames"): num_frames = v_info["total_frames"]
            
        records.append({
            "dataset": "URFD",
            "event_id": event_id,
            "video_id": f"URFD_{event_id}_cam0",
            "location": "Lab",
            "camera_id": "cam0",
            "label": "NORMAL",
            "video_path": rel_p(v_path, root_dir),
            "annotation_path": "UNKNOWN",
            "start_frame": "UNKNOWN",
            "end_frame": "UNKNOWN",
            "num_frames": num_frames if num_frames else "UNKNOWN",
            "fps": 30.0,
            "width": 640,
            "height": 480,
            "duration_seconds": round(num_frames / 30.0, 2) if num_frames else "UNKNOWN",
            "subject_id": "UNKNOWN",
            "depth_path": rel_p(depth_folder, root_dir),
            "rgb_path": rel_p(rgb_folder, root_dir),
            "timestamp_path": rel_p(c_path, root_dir),
            "accelerometer_path": "UNKNOWN",
            "annotation_format": "CSV_TIMESTAMPS"
        })

    for i in range(1, 31):
        event_id = f"fall-{i:02d}"
        csv_file = f"{event_id}-data.csv"
        c_path = os.path.join(urfd_dir, csv_file)
        
        for cam in ["cam0", "cam1"]:
            video_file = f"{event_id}-{cam}.mp4"
            v_path = os.path.join(urfd_dir, video_file)
            rgb_folder = os.path.join(urfd_dir, f"{event_id}-{cam}-rgb")
            depth_folder = os.path.join(urfd_dir, f"{event_id}-{cam}-d")
            
            pngs = glob.glob(os.path.join(rgb_folder, "**", "*.png"), recursive=True)
            num_frames = len(pngs)
            v_info = parse_mp4_details(v_path) if os.path.exists(v_path) else {}
            if not num_frames and v_info.get("total_frames"): num_frames = v_info["total_frames"]

            records.append({
                "dataset": "URFD",
                "event_id": event_id,
                "video_id": f"URFD_{event_id}_{cam}",
                "location": "Lab",
                "camera_id": cam,
                "label": "FALL",
                "video_path": rel_p(v_path, root_dir),
                "annotation_path": "UNKNOWN",
                "start_frame": "UNKNOWN",
                "end_frame": "UNKNOWN",
                "num_frames": num_frames if num_frames else "UNKNOWN",
                "fps": 30.0,
                "width": 640,
                "height": 480,
                "duration_seconds": round(num_frames / 30.0, 2) if num_frames else "UNKNOWN",
                "subject_id": "UNKNOWN",
                "depth_path": rel_p(depth_folder, root_dir),
                "rgb_path": rel_p(rgb_folder, root_dir),
                "timestamp_path": rel_p(c_path, root_dir),
                "accelerometer_path": rel_p(c_path, root_dir),
                "annotation_format": "CSV_TIMESTAMPS_ACCEL"
            })

    # 2. Process Le2i
    le2i_dir = os.path.join(root_dir, "Le2i", "data")
    locations = ["Coffee_room_01", "Coffee_room_02", "Home_01", "Home_02", "Lecture_room", "Office"]

    for loc in locations:
        loc_path = os.path.join(le2i_dir, loc)
        avi_files = sorted(glob.glob(os.path.join(loc_path, "**", "*.avi"), recursive=True))
        
        for avi_path in avi_files:
            avi_name = os.path.basename(avi_path)
            video_num_str = re.search(r"video\s*\((\d+)\)", avi_name)
            v_num = video_num_str.group(1) if video_num_str else avi_name
            
            event_id = f"{loc}_v{v_num}"
            video_id = f"Le2i_{loc}_v{v_num}"
            
            txt_name = avi_name.replace(".avi", ".txt")
            ann_files = glob.glob(os.path.join(loc_path, "**", txt_name), recursive=True)
            ann_path = ann_files[0] if ann_files else None
            
            v_info = parse_avi_details(avi_path)
            num_frames = v_info["total_frames"]
            
            label, start_frame, end_frame, ann_format = "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN"
            
            if ann_path and os.path.exists(ann_path):
                ann_format = "TXT_BBOX_FALL_RANGE"
                with open(ann_path, "r") as fp:
                    lines = [l.strip() for l in fp.readlines() if l.strip()]
                if len(lines) >= 2 and lines[0].isdigit() and lines[1].isdigit():
                    s_f, e_f = int(lines[0]), int(lines[1])
                    if s_f > 0 and e_f > 0:
                        label, start_frame, end_frame = "FALL", s_f, e_f
                    elif s_f == 0 and e_f == 0:
                        label, start_frame, end_frame = "NORMAL", 0, 0

            records.append({
                "dataset": "Le2i",
                "event_id": event_id,
                "video_id": video_id,
                "location": loc,
                "camera_id": f"{loc}_cam0",
                "label": label,
                "video_path": rel_p(avi_path, root_dir),
                "annotation_path": rel_p(ann_path, root_dir),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "num_frames": num_frames if num_frames else "UNKNOWN",
                "fps": v_info["fps"],
                "width": v_info["width"],
                "height": v_info["height"],
                "duration_seconds": v_info["duration_sec"] if v_info["duration_sec"] else "UNKNOWN",
                "subject_id": "UNKNOWN",
                "depth_path": "UNKNOWN",
                "rgb_path": "UNKNOWN",
                "timestamp_path": "UNKNOWN",
                "accelerometer_path": "UNKNOWN",
                "annotation_format": ann_format
            })

    # 3. Process MultiCamera
    multicam_dir = os.path.join(root_dir, "dataset", "dataset")
    chute_folders = sorted([d for d in os.listdir(multicam_dir) if os.path.isdir(os.path.join(multicam_dir, d))])

    for chute in chute_folders:
        chute_path = os.path.join(multicam_dir, chute)
        chute_num = int(chute.replace("chute", ""))
        chute_label = "FALL" if chute_num <= 22 else "NORMAL"
            
        for cam_idx in range(1, 9):
            avi_path = os.path.join(chute_path, f"cam{cam_idx}.avi")
            if os.path.exists(avi_path):
                v_info = parse_avi_details(avi_path)
                num_frames = v_info["total_frames"]
                records.append({
                    "dataset": "MultiCamera",
                    "event_id": chute,
                    "video_id": f"MultiCamera_{chute}_cam{cam_idx}",
                    "location": "Lab",
                    "camera_id": f"cam{cam_idx}",
                    "label": chute_label,
                    "video_path": rel_p(avi_path, root_dir),
                    "annotation_path": "UNKNOWN",
                    "start_frame": "UNKNOWN",
                    "end_frame": "UNKNOWN",
                    "num_frames": num_frames if num_frames else "UNKNOWN",
                    "fps": 25.0,
                    "width": v_info["width"],
                    "height": v_info["height"],
                    "duration_seconds": round(num_frames / 25.0, 2) if num_frames else "UNKNOWN",
                    "subject_id": "subject_1",
                    "depth_path": "UNKNOWN",
                    "rgb_path": "UNKNOWN",
                    "timestamp_path": "UNKNOWN",
                    "accelerometer_path": "UNKNOWN",
                    "annotation_format": "UNKNOWN"
                })

    df_manifest = pd.DataFrame(records)
    out_dir = os.path.join(root_dir, "R&D", "Dataset_Analysis")
    os.makedirs(out_dir, exist_ok=True)
    csv_out = os.path.join(out_dir, "dataset_manifest.csv")
    df_manifest.to_csv(csv_out, index=False)
    print(f"Manifest created successfully at: {csv_out} ({len(df_manifest)} records)")

if __name__ == "__main__":
    main()
