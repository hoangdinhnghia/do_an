"""
Checkpoint: Kiểm tra Staff Removal Pipeline

Chạy:
    cd /home/runner/work/do_an/do_an
    python orm/test_staff_removal.py

Output (lưu trong out_staff_removal/):
    <base>_original.png       — ảnh gốc với dòng kẻ được vẽ overlay
    <base>_binary.png         — ảnh nhị phân trước khi xóa dòng kẻ
    <base>_removed.png        — ảnh nhị phân SAU khi xóa dòng kẻ
    <base>_comparison.png     — so sánh before/after cạnh nhau
    <base>_staff_<n>_before.png / _after.png — từng staff crop
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
from orm.staff_detection import detect_and_refine_staff_lines, crop_staffs
from orm.staff_removal import staff_removal_pipeline, visualize_staff_removal

# ========================= CẤU HÌNH =========================
INPUT_PATH = "img_test/test0.png"
OUTPUT_DIR = "out_staff_removal"
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

# img_bin hiện là 0/1 uint8 — convert sang 0/255 để dễ xử lý
img_bin_255 = (img_bin * 255).astype(np.uint8)

# ========================= DETECT STAFF LINES =========================
staff_lines = detect_and_refine_staff_lines(img_bin_255)
print(f"[INFO] Phát hiện {len(staff_lines)} staff, tọa độ y:")
for i, st in enumerate(staff_lines):
    print(f"  Staff {i+1}: {st}")

if not staff_lines:
    print("[WARN] Không phát hiện được staff line nào — kiểm tra lại ảnh đầu vào.")
    exit(0)

# ========================= STAFF REMOVAL =========================
img_removed = staff_removal_pipeline(
    img_bin_255,
    staff_lines,
    thickness_margin=1,
    min_run_ratio=0.04,
    repair=True,
)
print(f"[INFO] Staff removal hoàn tất, shape={img_removed.shape}")

# ========================= LƯU KẾT QUẢ TỔNG =========================
# 1. Ảnh nhị phân trước khi xóa
cv2.imwrite(f"{OUTPUT_DIR}/{base_name}_binary.png", img_bin_255)

# 2. Ảnh sau khi xóa dòng kẻ
cv2.imwrite(f"{OUTPUT_DIR}/{base_name}_removed.png", img_removed)

# 3. Ảnh so sánh before/after
comparison = visualize_staff_removal(img, img_removed, staff_lines)
cv2.imwrite(f"{OUTPUT_DIR}/{base_name}_comparison.png", comparison)
print(f"[INFO] Lưu: {OUTPUT_DIR}/{base_name}_comparison.png")

# ========================= LƯU TỪNG STAFF CROP =========================
crops_before = crop_staffs(img_bin_255, staff_lines, expand=20)
crops_after = crop_staffs(img_removed, staff_lines, expand=20)

for idx, (cb, ca) in enumerate(zip(crops_before, crops_after)):
    path_b = f"{OUTPUT_DIR}/{base_name}_staff_{idx+1}_before.png"
    path_a = f"{OUTPUT_DIR}/{base_name}_staff_{idx+1}_after.png"
    cv2.imwrite(path_b, cb)
    cv2.imwrite(path_a, ca)
    # Thống kê số pixel còn lại sau removal
    px_before = int(np.sum(cb > 0))
    px_after = int(np.sum(ca > 0))
    removed_pct = 100.0 * (px_before - px_after) / max(px_before, 1)
    print(
        f"  Staff {idx+1}: {px_before} px → {px_after} px "
        f"(xóa {removed_pct:.1f}%) | {path_a}"
    )

print(f"\n✔ Hoàn tất. Kết quả lưu trong: {OUTPUT_DIR}/")
print(
    "  Kiểm tra _comparison.png: dòng kẻ bên trái (đỏ), "
    "nốt nhạc còn lại bên phải (trắng)."
)
