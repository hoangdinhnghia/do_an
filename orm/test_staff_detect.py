import cv2
import os
import numpy as np
from orm.preprocess import (
    preprocess_image,
    adaptive_binarize,
    remove_noise,
    sharpen,
    enhance_contrast
)
from orm.staff_detection import(
    detect_and_refine_staff_lines, 
    crop_staffs, 
    group_grand_staff,
    detect_vertical_connectors,
)


def load_input_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f'Không đọc được file đầu vào: {path}')
    return img


def load_pdf_pages(path: str, dpi: int = 300) -> list[np.ndarray]:
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz

    pages = []
    doc = fitz.open(path)
    try:
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        for i in range(len(doc)):
            pix = doc[i].get_pixmap(matrix=mat, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            pages.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    finally:
        doc.close()

    return pages


def process_one_image(img: np.ndarray):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # ==== PREPROCESS ====
    img_gray = preprocess_image(img)
    img_gray = enhance_contrast(img_gray)
    img_sharp = sharpen(img_gray)
    img_bin = adaptive_binarize(img_sharp)
    img_bin = remove_noise(img_bin)

    # ==== DETECT STAFF ====
    staff_lines = detect_and_refine_staff_lines(img_bin)

    # ==== GROUP GRAND STAFF ====
    grand_staffs = group_grand_staff(staff_lines, img_bin=img_bin)
    connectors = detect_vertical_connectors(img_bin, min_height=40, x_merge_tol=8)
    print(f"[INFO] Staff detect: {len(staff_lines)} | Grand-staff ghép: {len(grand_staffs)}")
    print(f"[INFO] Vertical connectors detected: {len(connectors)}")

    # ==== CROP STAFF ====
    # Balanced mode: still tall enough for observation, but avoid spilling into next staff.
    crops = crop_staffs(img, staff_lines, expand=20, clip_to_neighbors=True)

    # ==== VISUALIZE STAFF ====
    img_vis = img.copy()
    for staff in staff_lines:
        for y in staff:
            cv2.line(img_vis, (0, y), (img_vis.shape[1], y), (0, 0, 255), 1)

    # ==== VISUALIZE GRAND STAFF ====
    img_grand = img.copy()
    colors = [
        (255, 0, 0),     # Xanh dương
        (0, 255, 0),     # Xanh lá
        (0, 0, 255),     # Đỏ
        (255, 255, 0),   # Cyan
        (255, 0, 255),   # Magenta
        (0, 255, 255),   # Vàng
        (128, 0, 255),   # Tím đậm
        (0, 128, 255),   # Cam đậm
        (255, 128, 0),   # Xanh dương sáng
        (128, 255, 0),   # Xanh lá sáng
        (255, 0, 128),   # Hồng đậm
        (0, 255, 128),   # Xanh ngọc
    ]

    grand_crops = []

    # Draw detected vertical connectors (orange) for debugging.
    for x1, y1, x2, y2 in connectors:
        cv2.line(img_grand, (x1, y1), (x2, y2), (0, 165, 255), 2)

    for idx, (upper, lower) in enumerate(grand_staffs):
        color = colors[idx % len(colors)]

        # vẽ upper
        for y in upper:
            cv2.line(img_grand, (0, y), (img_grand.shape[1], y), color, 2)

        # vẽ lower
        for y in lower:
            cv2.line(img_grand, (0, y), (img_grand.shape[1], y), color, 2)

        # label
        mid_y = (upper[-1] + lower[0]) // 2
        cv2.putText(img_grand, f'GS {idx+1}', (10, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # crop grand staff
        top = max(0, upper[0] - 15)
        bottom = min(img.shape[0], lower[-1] + 15)
        # Crop from colored visualization so the output clearly shows grouping.
        crop = img_grand[top:bottom, :]
        grand_crops.append(crop)

        print(f"[GS {idx+1}] Upper: {upper}, Lower: {lower}, Crop shape: {crop.shape}")

    return img_vis, crops, img_grand, grand_crops


# ========================= MAIN =========================

INPUT_PATH = 'img_test/test0.png'
os.makedirs('out', exist_ok=True)

ext = os.path.splitext(INPUT_PATH)[1].lower()
base_name = os.path.splitext(os.path.basename(INPUT_PATH))[0]

if ext == '.pdf':
    pages = load_pdf_pages(INPUT_PATH, dpi=300)

    for i, page_img in enumerate(pages):
        img_vis, crops, img_grand, grand_crops = process_one_image(page_img)

        # ===== xuất staff detect =====
        cv2.imwrite(f'out/{base_name}_page_{i+1:03d}_staff_detected.png', img_vis)

        # ===== xuất từng staff =====
        for idx, crop in enumerate(crops):
            cv2.imwrite(f'out/{base_name}_page_{i+1:03d}_staff_{idx+1}.png', crop)

        # ===== xuất grand staff =====
        cv2.imwrite(f'out/{base_name}_page_{i+1:03d}_grand_staff.png', img_grand)

        for idx, crop in enumerate(grand_crops):
            cv2.imwrite(f'out/{base_name}_page_{i+1:03d}_grand_{idx+1}.png', crop)

        print(f'✔ Trang {i+1}: {len(crops)} staff | {len(grand_crops)} grand staff')

else:
    img = load_input_image(INPUT_PATH)
    img_vis, crops, img_grand, grand_crops = process_one_image(img)

    # ===== staff detect =====
    cv2.imwrite(f'out_staff_detect/{base_name}_staff_detected.png', img_vis)

    # ===== từng staff =====
    for idx, crop in enumerate(crops):
        out_crop = f'out_staff_detect/{base_name}_staff_{idx+1}.png'
        cv2.imwrite(out_crop, crop)
        print(f'✔ {out_crop} | shape={crop.shape}')

    # ===== grand staff =====
    cv2.imwrite(f'out/{base_name}_grand_staff.png', img_grand)

    for idx, crop in enumerate(grand_crops):
        out_crop = f'out/{base_name}_grand_{idx+1}.png'
        cv2.imwrite(out_crop, crop)
        print(f'✔ {out_crop} | shape={crop.shape}')

    print(f'✔ Hoàn tất: {base_name}')