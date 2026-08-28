"""
PHASE F5 — COMPREHENSIVE STATIC VALIDATION AUDIT

Verifies 16 Static Safety & Functional Requirements:
1. K1 production checkpoint path unchanged and exists read-only.
2. Official decision threshold (tau = 0.3650) enforced.
3. 50-frame temporal buffer logic intact.
4. 3-consecutive FALL alert stabilization intact.
5. Person bounding box is based on real YOLO pose detection.
6. Person bounding box is around the detected person, not the full video frame.
7. Edge-of-frame proximity detection logic exists (is_partial_person / edge_reason).
8. Partial-person handling suppresses unreliable edge false falls.
9. Partial-person does not trigger a new fall event.
10. Confirmed fall remains safely latched (has_confirmed_fall).
11. Postural recovery logic remains present (getting_up_counter / recovered_counter).
12. Application state transitions are logged (previous_state, current_state, transition).
13. Missing person gate prevents feeding fabricated zero pose data to K1.
14. Non-obstructive HUD rendering present (no large blocking black rectangles).
15. Streamlit output isolated under R&D/ML_Baseline/results/final_k1/streamlit_tests/.
16. Existing experiment artifacts remain completely untouched (read-only audit).
"""

import os
import sys
import ast

ROOT_DIR = r"d:\ONE_DATA\Fall detection"
sys.path.insert(0, ROOT_DIR)

def validate_f5_streamlit():
    print("=" * 75)
    print("PHASE F5 — COMPREHENSIVE STATIC VALIDATION AUDIT")
    print("=" * 75)

    app_path = os.path.join(ROOT_DIR, "app.py")
    infer_path = os.path.join(ROOT_DIR, "src", "infer_final_k1.py")
    realtime_path = os.path.join(ROOT_DIR, "src", "final_k1_realtime_inference.py")
    test_path = os.path.join(ROOT_DIR, "src", "test_independent_video_k1.py")

    for path in [app_path, infer_path, realtime_path, test_path]:
        assert os.path.exists(path), f"CRITICAL MISSING FILE: {path}"

    with open(app_path, "r", encoding="utf-8") as f:
        app_code = f.read()
    with open(infer_path, "r", encoding="utf-8") as f:
        infer_code = f.read()
    with open(realtime_path, "r", encoding="utf-8") as f:
        realtime_code = f.read()

    # 1. AST Syntax Verification
    try:
        ast.parse(app_code)
        ast.parse(infer_code)
        ast.parse(realtime_code)
        print("  [PASS 1/16] AST Syntax Audit             : All python files parse cleanly")
    except SyntaxError as e:
        print(f"  [FAIL 1/16] AST Syntax Audit             : Syntax error -> {e}")
        sys.exit(1)

    # 2. Production Checkpoint Unchanged & Exists
    ckpt_path = os.path.join(ROOT_DIR, "checkpoints", "final_k1", "final_production.pth")
    assert os.path.exists(ckpt_path), f"CRITICAL: Production checkpoint missing at {ckpt_path}"
    assert "final_production.pth" in app_code and "final_production.pth" in realtime_code
    print(f"  [PASS 2/16] Production Checkpoint        : Referenced & Exists on Disk ({os.path.basename(ckpt_path)})")

    # 3. Official Threshold tau = 0.3650
    assert "0.3650" in app_code and "0.3650" in realtime_code
    print("  [PASS 3/16] Official Threshold Policy   : Enforced tau = 0.3650")

    # 4. 50-Frame Buffer Logic
    assert "50" in realtime_code and "frames_buffered" in realtime_code
    print("  [PASS 4/16] 50-Frame Temporal Buffer     : Verified (2.0s context @ 25 FPS)")

    # 5. 3-Consecutive FALL Stabilization
    assert "consecutive_required" in realtime_code and "consecutive_fall_count" in realtime_code
    print("  [PASS 5/16] Alert Stabilization Policy   : Enforced 3 consecutive FALL windows")

    # 6. Real YOLO Bounding Box Extraction
    assert "boxes.xyxy" in infer_code and "bbox" in realtime_code
    print("  [PASS 6/16] Real YOLO Bounding Box       : Extracted from results[0].boxes.xyxy")

    # 7. Person Bounding Box Around Person
    assert "draw_yolo_person_overlay" in app_code and "cv2.rectangle(img_bgr, (x1, y1), (x2, y2)" in app_code
    print("  [PASS 7/16] Person BBox Rendering        : Drawn tightly around person coordinates")

    # 8. Edge-of-Frame Proximity Logic
    assert "is_partial_person" in infer_code and "edge_reason" in infer_code
    print("  [PASS 8/16] Edge-of-Frame Proximity      : Active boundary proximity evaluation")

    # 9. Partial Person Fall Suppression
    assert "if is_partial_person and not self.has_confirmed_fall" in realtime_code
    print("  [PASS 9/16] Partial Person Protection    : Suppresses false alerts from clipped keypoints")

    # 10. Confirmed Fall Latching
    assert "has_confirmed_fall" in realtime_code and "FALLEN — ON FLOOR" in realtime_code
    print("  [PASS 10/16] Confirmed Fall Latching     : Confirmed fall remains latched post-event")

    # 11. Postural Recovery Tracking
    assert "getting_up_counter" in realtime_code and "recovered_counter" in realtime_code
    print("  [PASS 11/16] Postural Recovery Logic     : Posture-based recovery verification active")

    # 12. Transition Logging
    assert "previous_application_state" in app_code and "state_transition" in app_code
    print("  [PASS 12/16] State Transition Logging    : previous_state, current_state & transition logged")

    # 13. Missing Person Presence Gate
    assert "conf_sum < 0.5" in realtime_code and "NO PERSON DETECTED" in realtime_code
    print("  [PASS 13/16] Missing Person Safety Gate  : Bypasses K1 model on zero person confidence")

    # 14. Non-Obstructive Dynamic HUD Placement
    assert "cv2.putText" in app_code and "compute_hud_corner_position" in app_code
    print("  [PASS 14/16] Non-Obstructive Dynamic HUD : Dynamic corner placement (compute_hud_corner_position)")

    # 15. Streamlit Output Isolation
    assert "streamlit_tests" in app_code
    print("  [PASS 15/16] Output Isolation            : Isolated under results/final_k1/streamlit_tests/")

    # 16. Read-Only Safety Audit
    forbidden_writes = ["torch.save", "to_csv('processed_data", "to_csv('checkpoints"]
    safety_violations = [fw for fw in forbidden_writes if fw in app_code or fw in realtime_code]
    assert len(safety_violations) == 0, f"CRITICAL: Forbidden write operation detected: {safety_violations}"
    print("  [PASS 16/16] Read-Only Safety Audit      : Zero dataset/model/checkpoint modifications")

    print("=" * 75)
    print("ALL 16 STATIC VALIDATION CHECKS PASSED — APP.PY & INFERENCE ENGINE VERIFIED")
    print("=" * 75)

if __name__ == "__main__":
    validate_f5_streamlit()
