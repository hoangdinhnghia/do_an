"""
Checkpoint: Kiểm tra Notehead Detection Pipeline

Chạy:
    cd /home/runner/work/do_an/do_an
    python orm/test_notehead_detection.py

Output (lưu trong out_notehead/):
    <base>_staff_<n>_no_staff.png   — crop staff sau khi xóa dòng kẻ
    <base>_staff_<n>_notehead.png   — crop staff với notehead annotated (box đỏ, tâm xanh)
    <base>_summary.png              — ghép tất cả annotated crop thành một ảnh dọc
"""

import os
import cv2
import numpy as np

from orm.preprocess import (
    preprocess_image,
    adaptive_binarize,
    remove_noise,
    sharpen,
    enhance_contrast,
)
from orm.staff_detection import detect_and_refine_staff_lines
from orm.staff_removal import staff_removal_pipeline
from orm.notehead_detection import notehead_detection_pipeline

# ========================= CẤU HÌNH =========================
INPUT_PATH = "img_test/test0.png"
OUTPUT_DIR = "out_notehead"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================= ĐỌC ẢNH =========================
img = cv2.imread(INPUT_PATH)
if img is None:
    raise FileNotFoundError(f"Không đọc được ảnh: {INPUT_PATH}")

base_name = os.path.splitext(os.path.basename(INPUT_PATH))[0]
print(f"[INFO] Đọc ảnh: {INPUT_PATH}, shape={img.shape}")

# ========================= PREPROCESS =========================
img_gray = preprocess_image(img)
img_gray = enhance_contrast(img_gray)
img_sharp = sharpen(img_gray)
img_bin = adaptive_binarize(img_sharp)
img_bin = remove_noise(img_bin)
img_bin_255 = (img_bin * 255).astype(np.uint8)

# ========================= DETECT STAFF LINES =========================
staff_lines = detect_and_refine_staff_lines(img_bin_255)
print(f"[INFO] Phát hiện {len(staff_lines)} staff")

if not staff_lines:
    print("[WARN] Không phát hiện được staff line nào — kiểm tra lại ảnh đầu vào.")
    exit(0)

# ========================= STAFF REMOVAL =========================
img_no_staff = staff_removal_pipeline(
    img_bin_255,
    staff_lines,
    thickness_margin=1,
    min_run_ratio=0.04,
    repair=True,
)
print(f"[INFO] Staff removal hoàn tất, shape={img_no_staff.shape}")

# ========================= NOTEHEAD DETECTION =========================
results = notehead_detection_pipeline(img_no_staff, staff_lines, expand=20)
print(f"[INFO] Notehead detection hoàn tất trên {len(results)} staff")

# ========================= LƯU KẾT QUẢ =========================
annotated_crops = []
for idx, staff_y, noteheads, annotated_crop in results:
    # Crop chỉ có ảnh sau staff removal (không annotate)
    h_img = img_no_staff.shape[0]
    y0 = max(0, staff_y[0] - 20)
    y1 = min(h_img, staff_y[-1] + 20)
    raw_crop = img_no_staff[y0:y1, :]

    path_raw = f"{OUTPUT_DIR}/{base_name}_staff_{idx+1}_no_staff.png"
    path_ann = f"{OUTPUT_DIR}/{base_name}_staff_{idx+1}_notehead.png"

    cv2.imwrite(path_raw, raw_crop)
    cv2.imwrite(path_ann, annotated_crop)

    annotated_crops.append(annotated_crop)
    print(
        f"  Staff {idx+1}: {len(noteheads)} notehead(s) | "
        f"staff_y=[{staff_y[0]}..{staff_y[-1]}] | {path_ann}"
    )

# ========================= ẢNH TỔNG HỢP =========================
if annotated_crops:
    max_w = max(c.shape[1] for c in annotated_crops)
    padded = []
    for c in annotated_crops:
        if c.ndim == 2:
            c = cv2.cvtColor(c, cv2.COLOR_GRAY2BGR)
        if c.shape[1] < max_w:
            pad = np.zeros((c.shape[0], max_w - c.shape[1], 3), dtype=np.uint8)
            c = np.hstack([c, pad])
        padded.append(c)

    summary = np.vstack(padded)
    summary_path = f"{OUTPUT_DIR}/{base_name}_summary.png"
    cv2.imwrite(summary_path, summary)
    print(f"[INFO] Ảnh tổng hợp: {summary_path}")

total_notes = sum(len(r[2]) for r in results)
print(f"\n✔ Hoàn tất. Tổng cộng {total_notes} notehead trên {len(results)} staff.")
print(f"  Kết quả lưu trong: {OUTPUT_DIR}/")
