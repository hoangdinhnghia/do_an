import cv2
import numpy as np
from typing import List, Optional, Tuple

# Định nghĩa NoteheadBBox: (x, y, w, h, cx, cy)
NoteheadBBox = Tuple[int, int, int, int, int, int]

def detect_notehead_contour(
    staff_crop_img: np.ndarray,
    staff_y: Optional[List[int]] = None,
    min_area: int = 18,
    max_area: int = 1200,
    aspect_ratio: Tuple[float, float] = (0.45, 1.8),
) -> List[NoteheadBBox]:
    """
    Detect noteheads from a cropped staff image with robust morphological and shape filtering.
    Args:
        staff_crop_img: Ảnh crop staff (grayscale, binary hoặc BGR)
        staff_y: List 5 dòng staff (giúp lọc note ngoài staff)
        min_area, max_area: Kích thước diện tích contour note hợp lệ
        aspect_ratio: Tỉ lệ width/height chấp nhận của notehead
    Returns:
        List tuple (x, y, w, h, cx, cy) cho từng notehead phát hiện được
    """
    # 1. Chuyển về gray nếu là RGB
    if staff_crop_img.ndim == 3:
        gray = cv2.cvtColor(staff_crop_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = staff_crop_img.copy()

    # 2. Stretch gray về uint8, nhị phân hóa: foreground (note+staff) sẽ là trắng
    if gray.max() <= 1.0:
        gray = (gray * 255).astype(np.uint8)
    else:
        gray = gray.astype(np.uint8)
    _, fg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. Loại net staffline bằng morphological open ngang
    line_kernel_w = max(15, staff_crop_img.shape[1] // 18)
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_kernel_w, 1))
    staff_mask = cv2.morphologyEx(fg, cv2.MORPH_OPEN, hor_kernel)
    note_img = cv2.subtract(fg, staff_mask)

    # 4. Làm kín nhanh để vá các hole nhỏ trong notehead
    note_img = cv2.morphologyEx(note_img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    contours, _ = cv2.findContours(note_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    notehead_list: List[NoteheadBBox] = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        ar = (w / h) if h else 0.0

        if area < min_area or area > max_area:
            continue
        if ar < aspect_ratio[0] or ar > aspect_ratio[1]:
            continue

        # Thêm lọc compactness/circularity loại bỏ stem, slur, ký hiệu phụ
        perimeter = cv2.arcLength(c, True)
        if perimeter <= 0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.18:
            continue

        cx, cy = x + w // 2, y + h // 2

        # Nếu có list staff_y (5 dòng y), chỉ nhận notehead gần vùng đó
        if staff_y is not None and len(staff_y) > 0:
            y0 = min(staff_y)
            y4 = max(staff_y)
            pad = max(14, int((y4 - y0) * 0.9))
            if not (y0 - pad <= cy <= y4 + pad):
                continue

        notehead_list.append((x, y, w, h, cx, cy))

    return notehead_list

def annotate_noteheads(
    img: np.ndarray,
    noteheads: List[NoteheadBBox],
) -> np.ndarray:
    """
    Vẽ bounding box và center các notehead lên ảnh crop staff để debug hoặc visualize.
    Args:
        img: input image (crop staff)
        noteheads: list box notehead
    Returns:
        img_vis: ảnh đã annotate box/circle lên
    """
    img_vis = img.copy()
    for (x, y, w, h, cx, cy) in noteheads:
        cv2.rectangle(img_vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.circle(img_vis, (cx, cy), 2, (255, 0, 0), -1)
    return img_vis