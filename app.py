"""
HUMAN FALL DETECTION SYSTEM — STAGE F5 PRODUCTION STREAMLIT WEB APPLICATION

Phase F5 Architectural Update: Explicit Separation of Model K1 Output vs. Application State Machine
+ Dynamic Non-Overlapping Video HUD & High-Quality Anti-Aliased Skeletal Visualization
+ Independent Display & Visual Quality Control Settings

Model K1 Output:
- fall_probability = P(FALL)
- raw_model_prediction = NORMAL or FALL @ tau=0.3650

Application State Machine (11 Derived States):
1. WARMING UP
2. NORMAL — STANDING
3. NORMAL — WALKING
4. NORMAL — SITTING
5. NORMAL (Generic Fallback)
6. FALL SUSPECTED (1-2 consecutive windows @ P >= 0.3650)
7. FALL DETECTED (3+ consecutive windows @ P >= 0.3650)
8. FALLEN — ON FLOOR (Latched post-fall low posture)
9. GETTING UP / RECOVERY (Upward motion / unbending post-fall)
10. RECOVERED — STANDING (Sustained upright posture post-recovery)
11. NO PERSON DETECTED (Empty room / person left frame)
"""

import os
import sys
import time
import json
import tempfile
import numpy as np
import pandas as pd
import cv2
import torch
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT_DIR = r"d:\ONE_DATA\Fall detection"
sys.path.insert(0, ROOT_DIR)

from src.final_k1_realtime_inference import RealtimeFallDetector, ApplicationStateMachine
from src.sms_alert_manager import SMSAlertManager

COCO_SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),      # Facial connections
    (5, 6),                             # Shoulders
    (5, 7), (7, 9),                     # Left Arm
    (6, 8), (8, 10),                    # Right Arm
    (5, 11), (6, 12), (11, 12),         # Torso & Hips
    (11, 13), (13, 15),                 # Left Leg
    (12, 14), (14, 16)                  # Right Leg
]

def compute_hud_corner_position(w_img, h_img, bbox, hud_w=240, hud_h=55):
    """
    Determines optimal non-overlapping corner placement for the video HUD relative to detected person bounding box.
    """
    if bbox is None:
        return 15, 15  # Default Top-Left when no person detected
        
    x1, y1, x2, y2 = bbox
    person_cx = (x1 + x2) / 2.0
    person_cy = (y1 + y2) / 2.0
    
    corners = {
        "top_right": (w_img - hud_w - 15, 15),
        "top_left": (15, 15),
        "bottom_right": (w_img - hud_w - 15, h_img - hud_h - 15),
        "bottom_left": (15, h_img - hud_h - 15)
    }
    
    # Priority order based on person location
    if person_cx < w_img / 2.0:
        preferred = ["top_right", "bottom_right", "bottom_left", "top_left"]
    else:
        preferred = ["top_left", "bottom_left", "bottom_right", "top_right"]
        
    for c_name in preferred:
        cx, cy = corners[c_name]
        # Check collision with person bbox
        overlap = not (cx + hud_w < x1 or cx > x2 or cy + hud_h < y1 or cy > y2)
        if not overlap:
            return cx, cy
            
    # Fallback to top priority corner if screen is crowded
    return corners[preferred[0]]


def draw_yolo_person_overlay(
    img_bgr, bbox, coco_17_px, current_state, prob_fall, threshold, raw_decision,
    is_partial_person=False,
    show_bbox=True,
    show_skeleton=True,
    show_keypoints=True,
    show_status_text=True,
    show_ml_info=True,
    bbox_thickness=2,
    skeleton_thickness=1,
    keypoint_size=2,
    text_scale=0.45
):
    """
    Renders clean tight YOLO person bounding box, high-quality anti-aliased 17-keypoint skeleton,
    and dynamically positioned non-overlapping video HUD with independent display toggles.
    """
    h_img, w_img = img_bgr.shape[:2]
    
    # BGR Color Mapping
    state_bgr = {
        "WARMING UP": (200, 200, 200),
        "NO PERSON DETECTED": (120, 120, 120),
        "NORMAL — STANDING": (0, 255, 0),
        "NORMAL — WALKING": (0, 255, 0),
        "NORMAL — SITTING": (0, 255, 0),
        "NORMAL": (0, 255, 0),
        "FALL SUSPECTED": (0, 165, 255),
        "FALL DETECTED": (0, 0, 255),
        "FALLEN — ON FLOOR": (180, 0, 180),
        "GETTING UP / RECOVERY": (0, 215, 255),
        "RECOVERED — STANDING": (100, 200, 0)
    }
    box_color = state_bgr.get(current_state, (0, 255, 0))

    # 1. Person Bounding Box & High-Quality Anti-Aliased Skeleton (On Detected Person Only)
    if bbox is not None and len(bbox) == 4 and current_state != "NO PERSON DETECTED":
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
        # Tight Person Bounding Box (Only around detected person)
        if show_bbox:
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), box_color, bbox_thickness, cv2.LINE_AA)
            
            # Person Box Identification Label Tag (Top-Left of Person Box)
            tag_text = "PERSON | Pose Detected" if not is_partial_person else "PERSON | Edge Clipped"
            lbl_y = max(15, y1 - 6)
            cv2.putText(img_bgr, tag_text, (x1, lbl_y), cv2.FONT_HERSHEY_SIMPLEX, text_scale * 0.9, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img_bgr, tag_text, (x1, lbl_y), cv2.FONT_HERSHEY_SIMPLEX, text_scale * 0.9, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Draw 17 COCO Keypoints & Skeleton Lines
        if coco_17_px is not None and len(coco_17_px) == 17:
            # Thin, smooth cyan skeleton lines
            if show_skeleton:
                for p1_idx, p2_idx in COCO_SKELETON_PAIRS:
                    if p1_idx < len(coco_17_px) and p2_idx < len(coco_17_px):
                        x_a, y_a, conf_a = coco_17_px[p1_idx]
                        x_b, y_b, conf_b = coco_17_px[p2_idx]
                        if conf_a > 0.3 and conf_b > 0.3:
                            cv2.line(img_bgr, (int(x_a), int(y_a)), (int(x_b), int(y_b)), (255, 255, 0), skeleton_thickness, cv2.LINE_AA)
            
            # Small, crisp red joint dots
            if show_keypoints:
                for kp in coco_17_px:
                    x_k, y_k, conf_k = kp
                    if conf_k > 0.3:
                        cv2.circle(img_bgr, (int(x_k), int(y_k)), keypoint_size, (0, 0, 255), -1, cv2.LINE_AA)

    # 2. Dynamic Non-Overlapping Video HUD Overlay Badge
    if show_status_text or show_ml_info:
        hud_scale_ratio = text_scale / 0.45
        hud_w, hud_h = int(240 * hud_scale_ratio), int(52 * hud_scale_ratio)
        hx, hy = compute_hud_corner_position(w_img, h_img, bbox if current_state != "NO PERSON DETECTED" else None, hud_w=hud_w, hud_h=hud_h)
        
        # Semi-transparent HUD overlay background
        overlay = img_bgr.copy()
        cv2.rectangle(overlay, (hx, hy), (hx + hud_w, hy + hud_h), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.75, img_bgr, 0.25, 0, img_bgr)
        cv2.rectangle(img_bgr, (hx, hy), (hx + hud_w, hy + hud_h), box_color, 1, cv2.LINE_AA)
        
        # Video HUD Text Inside Overlay Box
        edge_str = " [EDGE]" if is_partial_person else ""
        hud_line1 = f"{current_state}{edge_str}" if show_status_text else ""
        hud_line2 = (f"P(FALL): {prob_fall*100:4.1f}% | ML: {raw_decision}" if current_state != "NO PERSON DETECTED" else "No Reliable Pose Available") if show_ml_info else ""
        
        if show_status_text and show_ml_info:
            cv2.putText(img_bgr, hud_line1, (hx + 8, hy + int(20 * hud_scale_ratio)), cv2.FONT_HERSHEY_SIMPLEX, text_scale, box_color, 1, cv2.LINE_AA)
            cv2.putText(img_bgr, hud_line2, (hx + 8, hy + int(42 * hud_scale_ratio)), cv2.FONT_HERSHEY_SIMPLEX, text_scale * 0.93, (220, 220, 220), 1, cv2.LINE_AA)
        elif show_status_text:
            cv2.putText(img_bgr, hud_line1, (hx + 8, hy + int(32 * hud_scale_ratio)), cv2.FONT_HERSHEY_SIMPLEX, text_scale * 1.1, box_color, 1, cv2.LINE_AA)
        elif show_ml_info:
            cv2.putText(img_bgr, hud_line2, (hx + 8, hy + int(32 * hud_scale_ratio)), cv2.FONT_HERSHEY_SIMPLEX, text_scale * 0.95, (220, 220, 220), 1, cv2.LINE_AA)

    return img_bgr

# ---------------------------------------------------------------------------
# Page Configuration & Modern Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Human Fall Detection System — Production K1 Demo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .model-output-card {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
    .status-normal {
        background-color: #064e3b;
        border: 1px solid #059669;
        color: #34d399;
        padding: 12px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.2rem;
        text-align: center;
    }
    .status-suspected {
        background-color: #78350f;
        border: 1px solid #d97706;
        color: #fbbf24;
        padding: 12px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.2rem;
        text-align: center;
    }
    .status-alert {
        background-color: #7f1d1d;
        border: 1px solid #dc2626;
        color: #fca5a5;
        padding: 12px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.3rem;
        text-align: center;
    }
    .status-fallen {
        background-color: #581c87;
        border: 1px solid #9333ea;
        color: #e9d5ff;
        padding: 12px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.2rem;
        text-align: center;
    }
    .status-recovering {
        background-color: #854d0e;
        border: 1px solid #eab308;
        color: #fef08a;
        padding: 12px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.2rem;
        text-align: center;
    }
    .status-recovered {
        background-color: #065f46;
        border: 1px solid #10b981;
        color: #a7f3d0;
        padding: 12px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.2rem;
        text-align: center;
    }
    .status-noperson {
        background-color: #334155;
        border: 1px solid #64748b;
        color: #94a3b8;
        padding: 12px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.2rem;
        text-align: center;
    }
    .status-warmup {
        background-color: #1e293b;
        border: 1px solid #475569;
        color: #cbd5e1;
        padding: 12px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.2rem;
        text-align: center;
    }
    .disclaimer-box {
        background-color: #0f172a;
        border-left: 4px solid #3b82f6;
        padding: 10px 15px;
        margin-top: 15px;
        margin-bottom: 15px;
        font-size: 0.85rem;
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants & System Configuration
# ---------------------------------------------------------------------------
PRODUCTION_CHECKPOINT = os.path.join(ROOT_DIR, "checkpoints", "final_k1", "final_production.pth")
PRODUCTION_THRESHOLD = 0.3650
STREAMLIT_OUTPUT_DIR = os.path.join(ROOT_DIR, "R&D", "ML_Baseline", "results", "final_k1", "streamlit_tests")

DEVICE_NAME = "cuda (NVIDIA RTX 4060 GPU)" if torch.cuda.is_available() else "CPU Execution"

# ---------------------------------------------------------------------------
# Session State Initialization & Reset Handler
# ---------------------------------------------------------------------------
def reset_display_settings():
    st.session_state.show_bbox = True
    st.session_state.show_skeleton = True
    st.session_state.show_keypoints = True
    st.session_state.show_person_id = True
    st.session_state.show_state_label = True
    st.session_state.show_prob = True
    st.session_state.show_status_text = True
    st.session_state.show_ml_info = True
    st.session_state.show_alert_perimeter = True
    st.session_state.render_quality = "High"
    st.session_state.bbox_thickness = 2
    st.session_state.skeleton_thickness = 2
    st.session_state.keypoint_size = 3
    st.session_state.text_scale = 0.50

defaults = [
    ("show_bbox", True),
    ("show_skeleton", True),
    ("show_keypoints", True),
    ("show_person_id", True),
    ("show_state_label", True),
    ("show_prob", True),
    ("show_status_text", True),
    ("show_ml_info", True),
    ("show_alert_perimeter", True),
    ("render_quality", "High"),
    ("bbox_thickness", 2),
    ("skeleton_thickness", 2),
    ("keypoint_size", 3),
    ("text_scale", 0.50)
]
for k, v in defaults:
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# Sidebar UI Layout — Display Controls, Quality Settings & Telemetry
# ---------------------------------------------------------------------------
st.sidebar.title("🛡️ System Control & Telemetry")
st.sidebar.markdown("---")

st.sidebar.markdown("### 🎨 DISPLAY CONTROLS")
st.sidebar.checkbox("Show Person Bounding Box", key="show_bbox")
st.sidebar.checkbox("Show Pose Skeleton", key="show_skeleton")
st.sidebar.checkbox("Show Keypoints", key="show_keypoints")
st.sidebar.checkbox("Show Person ID Badge", key="show_person_id")
st.sidebar.checkbox("Show State Label", key="show_state_label")
st.sidebar.checkbox("Show Fall Probability", key="show_prob")
st.sidebar.checkbox("Show Application Status Text", key="show_status_text")
st.sidebar.checkbox("Show ML Information", key="show_ml_info")
st.sidebar.checkbox("Show Alert Perimeter", key="show_alert_perimeter")

st.sidebar.button("🔄 Reset Display Settings", on_click=reset_display_settings, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ VISUAL QUALITY")
st.sidebar.selectbox("Render Quality", ["Low", "Medium", "High"], key="render_quality")

st.sidebar.slider("Bounding Box Thickness", 1, 5, key="bbox_thickness")
st.sidebar.slider("Skeleton Thickness", 1, 5, key="skeleton_thickness")
st.sidebar.slider("Keypoint Size (px)", 1, 6, key="keypoint_size")
st.sidebar.slider("Text Scale", 0.30, 0.80, key="text_scale", step=0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 ML MODEL K1 SPECIFICATION")
st.sidebar.text_input("Checkpoint Path", value=PRODUCTION_CHECKPOINT, disabled=True)
st.sidebar.number_input("Decision Threshold (τ)", value=PRODUCTION_THRESHOLD, format="%.4f", disabled=True,
                         help="Official validated leakage-free production decision threshold (tau = 0.3650)")
st.sidebar.text_input("Model Output Type", value="Binary P(FALL) Probability", disabled=True)
st.sidebar.text_input("Temporal Window", value="50 Frames (2.0s Context)", disabled=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 APPLICATION STATE ENGINE")
st.sidebar.text_input("State Machine Engine", value="11 Derived Application States", disabled=True)
st.sidebar.text_input("Alert Stabilization", value="3 Consecutive FALL Windows", disabled=True)
st.sidebar.text_input("Edge-of-Frame Protection", value="Active Boundary Guard", disabled=True)
st.sidebar.text_input("Execution Device", value=DEVICE_NAME, disabled=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📱 CELLULAR SMS PHONE ALERTS")
sms_mgr_side = SMSAlertManager()
st.sidebar.text_input("Twilio Account SID (Masked)", value=sms_mgr_side.get_masked_account_sid(), disabled=True)
st.sidebar.text_input("Destination Number (Masked)", value=sms_mgr_side.get_masked_to_number(), disabled=True)
st.sidebar.text_input("SMS Service Status", value="CONFIGURED & READY" if sms_mgr_side.is_configured() else ("DISABLED" if not sms_mgr_side.enabled else "NOT_CONFIGURED"), disabled=True)

st.sidebar.markdown("---")
st.sidebar.caption("Human Fall Detection System — Research & Engineering Demonstration Platform")

# ---------------------------------------------------------------------------
# Main Page Header & Disclaimer
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">Human Fall Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Phase F5 Production Streamlit Testing Application | 11-State Application Machine</div>', unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-box">
  ℹ️ <b>SCIENTIFIC & ENGINEERING HONESTY NOTICE</b><br>
  Model K1 is a binary supervised fall detector outputting P(FALL). Standing, walking, sitting, fallen, recovery, and recovered states are application-level interpretations derived from YOLO Pose observations, K1 fall probability, and temporal state-machine logic. They are not independent classes learned by the K1 neural network.
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# File Uploader
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Test Video File (.mp4, .avi, .mov)",
    type=["mp4", "avi", "mov"],
    help="Select a video file to run production K1 fall detection inference"
)

if uploaded_file is not None:
    # Verify production checkpoint availability
    if not os.path.exists(PRODUCTION_CHECKPOINT):
        st.error(f"❌ CRITICAL ERROR: Production checkpoint not found at `{PRODUCTION_CHECKPOINT}`. Please ensure Phase F3 training has completed.")
        st.stop()

    os.makedirs(STREAMLIT_OUTPUT_DIR, exist_ok=True)
    
    # Save upload to temporary file safely
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    video_name = uploaded_file.name
    video_stem = os.path.splitext(video_name)[0]

    # Pre-open Video to Read Metadata
    cap_meta = cv2.VideoCapture(video_path)
    if not cap_meta.isOpened():
        st.error(f"❌ ERROR: Failed to open or decode video file `{video_name}`. File may be corrupted or unsupported.")
        st.stop()

    total_frames = int(cap_meta.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_src = cap_meta.get(cv2.CAP_PROP_FPS)
    if fps_src <= 0 or np.isnan(fps_src):
        fps_src = 25.0
    cap_meta.release()

    duration_sec = total_frames / fps_src if fps_src > 0 else 0.0

    st.success(f"Loaded Video: **{video_name}** ({total_frames} frames | {fps_src:.2f} FPS | {duration_sec:.2f} seconds)")

    if total_frames < 50:
        st.warning(f"⚠️ WARMING UP — Insufficient frames for K1 inference ({total_frames}/50 frames). At least 50 frames are required to form a full temporal window.")

    # UI Layout Split
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### 🎥 Video Stream & Skeletal Overlay")
        frame_placeholder = st.empty()

    with col_right:
        st.markdown("### 🧠 ML Model Output & Application State Machine")
        model_out_placeholder = st.empty()
        status_placeholder = st.empty()
        
        m1, m2, m3 = st.columns(3)
        with m1:
            fps_metric = st.empty()
        with m2:
            lat_metric = st.empty()
        with m3:
            win_metric = st.empty()

    st.markdown("---")
    st.markdown("### 📈 Real-Time Fall Probability Profile")
    chart_placeholder = st.empty()

    run_btn = st.button("🚀 Run Production K1 Fall Detection", use_container_width=True)

    if run_btn:
        # Initialize Production Fall Detector Engine
        try:
            detector = RealtimeFallDetector(
                checkpoint_path=PRODUCTION_CHECKPOINT,
                threshold_policy=PRODUCTION_THRESHOLD,
                consecutive_fall_required=3
            )
        except Exception as e:
            st.error(f"❌ Failed to load Model K1: {e}")
            st.stop()

        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        latencies = []
        log_records = []
        df_timeline = []

        raw_fall_windows = 0
        stabilized_fall_windows = 0
        alert_events = 0
        prev_alert_active = False

        first_alert_frame = None
        first_alert_time = None

        progress_bar = st.progress(0.0)
        t_start_app = time.time()

        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                break

            frame_idx += 1
            timestamp_sec = frame_idx / fps_src
            progress_bar.progress(min(1.0, frame_idx / max(1, total_frames)))

            # Execute Engine Inference (ML & State Machine unaltered)
            result = detector.process_frame(frame_bgr)

            is_warmed = result["is_warmed_up"]
            person_detected = result["person_detected"]
            is_partial_person = result.get("is_partial_person", False)
            edge_reason = result.get("edge_reason", "FULL_PERSON")
            prob_fall = result["prob_fall"]
            raw_decision = result["raw_decision"]
            alert_state = result["alert_state"]
            current_state = result["current_state"]
            previous_state = result["previous_state"]
            state_transition = result["state_transition"]
            lat_ms = result["latency_ms"]
            proc_fps = result["fps"]
            bbox = result.get("bbox", None)
            coco_17_px = result.get("coco_17_px", None)

            # Render Overlay based on User Display Toggles & Quality Settings
            annotated_bgr = frame_bgr.copy()
            annotated_bgr = draw_yolo_person_overlay(
                annotated_bgr, bbox, coco_17_px, current_state, prob_fall, PRODUCTION_THRESHOLD, raw_decision,
                is_partial_person=is_partial_person,
                show_bbox=st.session_state.show_bbox,
                show_skeleton=st.session_state.show_skeleton,
                show_keypoints=st.session_state.show_keypoints,
                show_status_text=st.session_state.show_status_text,
                show_ml_info=st.session_state.show_ml_info,
                bbox_thickness=st.session_state.bbox_thickness,
                skeleton_thickness=st.session_state.skeleton_thickness,
                keypoint_size=st.session_state.keypoint_size,
                text_scale=st.session_state.text_scale
            )

            # Convert to RGB for Streamlit Display
            disp_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            h_f, w_f = disp_rgb.shape[:2]

            # Render ML Model Output Card
            model_html = f"""
            <div class="model-output-card">
              <b>ML MODEL K1 OUTPUT</b>: P(FALL) = <b>{prob_fall*100:5.1f}%</b> | ML Decision: <b>{raw_decision}</b> (τ = {PRODUCTION_THRESHOLD:.4f})
            </div>
            """
            model_out_placeholder.markdown(model_html, unsafe_allow_html=True)

            # Render Status Indicator Banner for all 11 states
            if current_state == "NO PERSON DETECTED":
                status_html = '<div class="status-noperson">👤 NO PERSON DETECTED IN FRAME</div>'
            elif current_state == "WARMING UP":
                status_html = f'<div class="status-warmup">⏳ WARMING UP ({detector.frames_buffered}/50 frames)</div>'
            elif current_state == "FALL DETECTED":
                status_html = '<div class="status-alert">🚨 ALERT ACTIVE — FALL DETECTED!</div>'
            elif current_state == "FALL SUSPECTED":
                status_html = f'<div class="status-suspected">⚠️ FALL SUSPECTED ({alert_state["consecutive_fall_count"]}/3 windows)</div>'
            elif current_state == "FALLEN — ON FLOOR":
                status_html = '<div class="status-fallen">⚠️ PERSON FALLEN — ON FLOOR</div>'
            elif current_state == "GETTING UP / RECOVERY":
                status_html = '<div class="status-recovering">🔄 PERSON GETTING UP / RECOVERY</div>'
            elif current_state == "RECOVERED — STANDING":
                status_html = '<div class="status-recovered">✅ PERSON RECOVERED — STANDING</div>'
            elif current_state == "NORMAL — STANDING":
                status_html = '<div class="status-normal">✅ NORMAL — STANDING</div>'
            elif current_state == "NORMAL — WALKING":
                status_html = '<div class="status-normal">🚶 NORMAL — WALKING</div>'
            elif current_state == "NORMAL — SITTING":
                status_html = '<div class="status-normal">🪑 NORMAL — SITTING</div>'
            else:
                status_html = '<div class="status-normal">✅ NORMAL ADL POSTURE</div>'

            status_placeholder.markdown(status_html, unsafe_allow_html=True)

            # Full-Video Red Border Flash on Active Alert (Controlled by show_alert_perimeter)
            if is_warmed and alert_state["alert_active"] and st.session_state.show_alert_perimeter:
                cv2.rectangle(disp_rgb, (0, 0), (w_f, h_f), (255, 0, 0), 12)

            frame_placeholder.image(disp_rgb, channels="RGB", use_container_width=True)

            # Update Metric Counters
            fps_metric.metric("Processing Speed", f"{proc_fps:.1f} FPS")
            lat_metric.metric("Latency", f"{lat_ms:.1f} ms")
            win_metric.metric("Frame / Total", f"{frame_idx} / {total_frames}")

            # Collect Log & Predictions
            latencies.append(lat_ms)

            raw_bin = 1 if raw_decision == "FALL" else 0
            stab_bin = 1 if alert_state["alert_active"] else 0

            if raw_bin == 1:
                raw_fall_windows += 1
            if stab_bin == 1:
                stabilized_fall_windows += 1
                if not prev_alert_active:
                    alert_events += 1
                    if first_alert_frame is None:
                        first_alert_frame = frame_idx
                        first_alert_time = timestamp_sec

            prev_alert_active = alert_state["alert_active"]

            log_records.append({
                "video_name": video_name,
                "frame_index": frame_idx,
                "win_start_frame": max(1, frame_idx - 49),
                "win_end_frame": frame_idx,
                "timestamp_sec": round(timestamp_sec, 3),
                "person_detected": person_detected,
                "is_partial_person": is_partial_person,
                "edge_reason": edge_reason,
                "fall_probability": round(prob_fall, 4),
                "decision_threshold": PRODUCTION_THRESHOLD,
                "raw_prediction": raw_bin,
                "raw_decision": raw_decision,
                "stabilized_prediction": stab_bin,
                "previous_application_state": previous_state,
                "current_application_state": current_state,
                "state_transition": state_transition,
                "consecutive_fall_count": alert_state["consecutive_fall_count"],
                "sms_alert_enabled": result.get("sms_alert_enabled", False),
                "sms_alert_sent": result.get("sms_alert_sent", False),
                "sms_alert_status": result.get("sms_alert_status", "DISABLED"),
                "notification_event_type": result.get("notification_event_type", "NONE"),
                "latency_ms": round(lat_ms, 2),
                "processing_fps": round(proc_fps, 2)
            })

            df_timeline.append({
                "Time (s)": timestamp_sec,
                "P(FALL)": prob_fall,
                "Threshold (0.3650)": PRODUCTION_THRESHOLD
            })

            # Periodically update line chart
            if len(df_timeline) % 15 == 0 or frame_idx == total_frames:
                pdf = pd.DataFrame(df_timeline)
                fig = px.line(
                    pdf, x="Time (s)", y=["P(FALL)", "Threshold (0.3650)"],
                    range_y=[0, 1.05],
                    color_discrete_map={"P(FALL)": "#ef4444", "Threshold (0.3650)": "#3b82f6"},
                    labels={"value": "Probability", "variable": "Metric"}
                )
                fig.update_layout(
                    margin=dict(l=20, r=20, t=30, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                chart_placeholder.plotly_chart(fig, use_container_width=True)

        cap.release()
        progress_bar.empty()
        t_end_app = time.time()

        st.success(f"✅ Production Inference Finished in {t_end_app - t_start_app:.2f} seconds!")

        # ---------------------------------------------------------------------------
        # Results Summary & Output Logging
        # ---------------------------------------------------------------------------
        if log_records:
            df_log = pd.DataFrame(log_records)
            eval_wins = len(df_log)

            mean_lat = float(np.mean(latencies)) if latencies else 0.0
            p95_lat  = float(np.percentile(latencies, 95)) if latencies else 0.0
            mean_fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

            # Isolated Logging Paths
            csv_path = os.path.join(STREAMLIT_OUTPUT_DIR, f"{video_stem}_streamlit_log.csv")
            df_log.to_csv(csv_path, index=False)

            summary_dict = {
                "video_name": video_name,
                "total_frames": total_frames,
                "evaluated_windows": eval_wins,
                "warmup_frames": min(49, total_frames),
                "duration_sec": round(duration_sec, 2),
                "mean_fps": round(mean_fps, 2),
                "mean_latency_ms": round(mean_lat, 2),
                "p95_latency_ms": round(p95_lat, 2),
                "raw_fall_windows": raw_fall_windows,
                "stabilized_fall_windows": stabilized_fall_windows,
                "alert_events": alert_events,
                "first_alert_frame": first_alert_frame,
                "first_alert_time_sec": round(first_alert_time, 2) if first_alert_time else None,
                "production_threshold": PRODUCTION_THRESHOLD,
                "saved_csv_log": os.path.abspath(csv_path)
            }
            json_path = os.path.join(STREAMLIT_OUTPUT_DIR, f"{video_stem}_streamlit_summary.json")
            with open(json_path, "w") as f:
                json.dump(summary_dict, f, indent=2)

            st.markdown("### 📋 Final Detection Summary")

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Evaluated Windows", f"{eval_wins}")
            s2.metric("Raw FALL Windows", f"{raw_fall_windows}")
            s3.metric("Stabilized ALERT Windows", f"{stabilized_fall_windows}")
            s4.metric("ALERT Events Fired", f"{alert_events}")

            s5, s6, s7, s8 = st.columns(4)
            s5.metric("Mean Throughput", f"{mean_fps:.1f} FPS")
            s6.metric("Mean Latency", f"{mean_lat:.2f} ms")
            s7.metric("P95 Latency", f"{p95_lat:.2f} ms")
            s8.metric("First Alert Timing", f"{first_alert_time:.2f}s (Frame {first_alert_frame})" if first_alert_frame else "None")

            st.markdown("### 📑 Per-Window Prediction & Transition Data Table")
            st.dataframe(df_log, use_container_width=True)

            csv_data = df_log.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Per-Window Prediction CSV Log",
                data=csv_data,
                file_name=f"{video_stem}_k1_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )

    # Cleanup temporary uploaded file
    try:
        os.remove(video_path)
    except Exception:
        pass

else:
    st.info("👆 Please upload a test video file (.mp4, .avi, .mov) using the file uploader above to begin real-time K1 inference.")
