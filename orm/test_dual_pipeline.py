"""
Kiểm thử luồng dữ liệu kép (Dual-Stream OMR Pipeline)

Sử dụng:
    cd /home/runner/work/do_an/do_an
    python orm/test_dual_pipeline.py [path_to_image]

Ảnh đầu vào mặc định: img_test/test0.png

Các file xuất ra trong thư mục out_dual/:
    <base>_staff_prob.png      — heatmap xác suất dòng kẻ (stream 1, kênh 1)
    <base>_notehead_prob.png   — heatmap xác suất notehead (stream 2, kênh 1)
    <base>_symbol_mask.png     — mask tổng hợp tất cả ký hiệu nhạc (stream 2)
    <base>_staff_overlay.png   — ảnh gốc với dòng kẻ được vẽ đè (stream 1)
    <base>_notehead_overlay.png— ảnh gốc với bounding box notehead (stream 2)
    <base>_combined.png        — ảnh gốc với cả staff + notehead visualize
"""

import os
import sys
import time

import cv2
import numpy as np

# Ensure the package is importable when run directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orm.model_inference import (
    StafflineSegmentationModel,
    DetailedSemanticModel,
    run_dual_pipeline,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "img_test/test0.png"
OUTPUT_DIR = "out_dual"
os.makedirs(OUTPUT_DIR, exist_ok=True)

base_name = os.path.splitext(os.path.basename(INPUT_PATH))[0]

# ---------------------------------------------------------------------------
# Load image
# ---------------------------------------------------------------------------
img = cv2.imread(INPUT_PATH)
if img is None:
    raise FileNotFoundError(f"Không đọc được ảnh: {INPUT_PATH}")
print(f"[INFO] Ảnh đầu vào: {INPUT_PATH}  shape={img.shape}")

# ---------------------------------------------------------------------------
# Load models once (reuse across calls for efficiency)
# ---------------------------------------------------------------------------
print("[INFO] Đang tải mô hình…")
t0 = time.time()
staffline_model = StafflineSegmentationModel()
semantic_model = DetailedSemanticModel()
print(f"[INFO] Tải mô hình hoàn tất ({time.time() - t0:.1f}s)")

# ---------------------------------------------------------------------------
# Run dual pipeline
# ---------------------------------------------------------------------------
print("[INFO] Bắt đầu chạy luồng kép…")
t1 = time.time()
result = run_dual_pipeline(
    img,
    staffline_model=staffline_model,
    semantic_model=semantic_model,
    staff_conf_thresh=0.3,
    note_conf_thresh=0.4,
    overlap=64,
)
elapsed = time.time() - t1
print(f"[INFO] Pipeline hoàn tất ({elapsed:.1f}s)")

staff_lines = result["staff_lines"]
noteheads   = result["noteheads"]
staff_prob  = result["staff_prob_map"]   # (H, W, 3) float32
sem_map     = result["semantic_map"]     # (H, W, 4) float32
sym_mask    = result["symbol_mask"]      # (H, W) uint8

print(f"[STREAM 1] Phát hiện {len(staff_lines)} staff system(s)")
for i, st in enumerate(staff_lines):
    print(f"  Staff {i+1}: {st}")

print(f"[STREAM 2] Phát hiện {len(noteheads)} notehead(s)")

# ---------------------------------------------------------------------------
# Helper: float probability map → colourised heatmap (BGR)
# ---------------------------------------------------------------------------
def _prob_to_heatmap(prob_ch: np.ndarray) -> np.ndarray:
    """Convert a (H, W) float32 probability channel to a BGR heatmap uint8."""
    norm = np.clip(prob_ch, 0.0, 1.0)
    gray = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

# 1. Staff-line probability heatmap (channel 1 of stream 1)
staff_heatmap = _prob_to_heatmap(staff_prob[:, :, 1])
cv2.imwrite(f"{OUTPUT_DIR}/{base_name}_staff_prob.png", staff_heatmap)
print(f"✔ {OUTPUT_DIR}/{base_name}_staff_prob.png")

# 2. Notehead probability heatmap (channel 1 of stream 2)
note_heatmap = _prob_to_heatmap(sem_map[:, :, 2])
cv2.imwrite(f"{OUTPUT_DIR}/{base_name}_notehead_prob.png", note_heatmap)
print(f"✔ {OUTPUT_DIR}/{base_name}_notehead_prob.png")

# 3. Symbol mask (all non-background from stream 2)
cv2.imwrite(f"{OUTPUT_DIR}/{base_name}_symbol_mask.png", sym_mask)
print(f"✔ {OUTPUT_DIR}/{base_name}_symbol_mask.png")

# 4. Staff overlay — draw detected staff lines on original image
img_staff_vis = img.copy()
for staff in staff_lines:
    for y in staff:
        cv2.line(img_staff_vis, (0, y), (img_staff_vis.shape[1], y), (0, 0, 255), 2)
cv2.imwrite(f"{OUTPUT_DIR}/{base_name}_staff_overlay.png", img_staff_vis)
print(f"✔ {OUTPUT_DIR}/{base_name}_staff_overlay.png  ({len(staff_lines)} staff system(s))")

# 5. Notehead overlay — draw bounding boxes on original image
img_note_vis = img.copy()
for (x, y, w, h, cx, cy) in noteheads:
    cv2.rectangle(img_note_vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.circle(img_note_vis, (cx, cy), 3, (255, 0, 0), -1)
cv2.imwrite(f"{OUTPUT_DIR}/{base_name}_notehead_overlay.png", img_note_vis)
print(f"✔ {OUTPUT_DIR}/{base_name}_notehead_overlay.png  ({len(noteheads)} notehead(s))")

# 6. Combined visualisation — staff lines (red) + noteheads (green box)
img_combined = img.copy()
for staff in staff_lines:
    for y in staff:
        cv2.line(img_combined, (0, y), (img_combined.shape[1], y), (0, 0, 255), 1)
for (x, y, w, h, cx, cy) in noteheads:
    cv2.rectangle(img_combined, (x, y), (x + w, y + h), (0, 200, 0), 2)
    cv2.circle(img_combined, (cx, cy), 3, (255, 128, 0), -1)
cv2.imwrite(f"{OUTPUT_DIR}/{base_name}_combined.png", img_combined)
print(f"✔ {OUTPUT_DIR}/{base_name}_combined.png")

print(
    f"\n✔ Hoàn tất. {len(staff_lines)} staff system(s), {len(noteheads)} notehead(s). "
    f"Kết quả trong: {OUTPUT_DIR}/"
)
