import os
import cv2
import numpy as np

from orm.staff_detection import detect_and_refine_staff_lines, crop_staffs
from notehead_detection import detect_notehead_contour, annotate_noteheads

# ==== Đầu vào: chỉ định file hoặc thư mục ảnh staff lớn ====
INPUT_PATH = 'out_staff_detect/'  # có thể là ảnh gốc, hoặc ảnh staff crop riêng lẻ
OUTPUT_DIR = 'out_notehead'
os.makedirs(OUTPUT_DIR, exist_ok=True)

if INPUT_PATH.endswith('.png') or INPUT_PATH.endswith('.jpg'):
    # 1. Đọc ảnh gốc hoặc ảnh staff lớn
    img = cv2.imread(INPUT_PATH)
    assert img is not None, f"Lỗi đọc file {INPUT_PATH}"

    # 2. Nếu là ảnh gốc lớn: phát hiện staff lines và crop staff
    staff_lines = detect_and_refine_staff_lines(img)
    staff_crops = crop_staffs(img, staff_lines, expand=10)

    # Nếu đã là staff crop nhỏ (vd: bạn có nhiều file staff_1.png, staff_2.png...), 
    # có thể bỏ qua bước này và duyệt từng ảnh luôn!

    # 3. Detect notehead & annotate từng staff crop
    for idx, crop in enumerate(staff_crops):
        noteheads = detect_notehead_contour(crop)
        crop_vis = annotate_noteheads(crop, noteheads)
        cv2.imwrite(f"{OUTPUT_DIR}/staff_{idx+1}_notehead.png", crop_vis)
        print(f"Staff {idx+1}: Detected {len(noteheads)} noteheads, output {OUTPUT_DIR}/staff_{idx+1}_notehead.png")
else:
    # 2. Nếu INPUT_PATH là thư mục chứa ảnh staff crop nhỏ
    crop_files = sorted([f for f in os.listdir(INPUT_PATH) if any(f.lower().endswith(ext) for ext in ['.png', '.jpg'])])
    for fidx, fname in enumerate(crop_files):
        crop = cv2.imread(os.path.join(INPUT_PATH, fname))
        noteheads = detect_notehead_contour(crop)
        crop_vis = annotate_noteheads(crop, noteheads)
        out_path = os.path.join(OUTPUT_DIR, f"notehead_{fname}")
        cv2.imwrite(out_path, crop_vis)
        print(f"{fname}: Detected {len(noteheads)} noteheads, output {out_path}")

print("Đã hoàn thành nhận diện notehead trên tất cả staff!")