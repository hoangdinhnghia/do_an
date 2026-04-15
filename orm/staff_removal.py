import cv2
import numpy as np
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Staff Removal Module
# ---------------------------------------------------------------------------
# Xóa dòng kẻ (staff lines) khỏi ảnh nhị phân sau khi đã detect được tọa độ y.
#
# Pipeline:
#   1. estimate_line_thickness()  — ước lượng độ dày thực tế của dòng kẻ
#   2. remove_staff_lines()       — xóa dòng kẻ bằng run-length filtering
#   3. repair_noteheads()         — vá các blob bị cắt đôi bởi dòng kẻ
#   4. staff_removal_pipeline()   — hàm tổng hợp toàn bộ 3 bước trên
# ---------------------------------------------------------------------------

# Tỉ lệ spacing của staff dùng để ước lượng độ dày dòng kẻ.
# Mỗi dòng kẻ thường dày ~18% khoảng cách giữa 2 dòng liên tiếp.
_LINE_THICKNESS_RATIO = 0.18

# Tỉ lệ chiều rộng ảnh để phân biệt một hàng là dòng kẻ (khi >10% pixel sáng).
_STAFF_ROW_WIDTH_RATIO = 0.10


def estimate_line_thickness(
    img_bin: np.ndarray,
    staff_ys: List[int],
    window: int = 4,
) -> int:
    """Ước lượng độ dày dòng kẻ (tính bằng pixel) từ horizontal projection.
    Args:
        img_bin: Ảnh nhị phân (foreground=255 hoặc 1).
        staff_ys: Danh sách tọa độ y của các dòng kẻ (1 staff, 5 phần tử).
        window: Cửa sổ tìm kiếm mỗi phía xung quanh y.
    Returns:
        Độ dày ước lượng (số nguyên, tối thiểu 1).
    """
    if img_bin.max() <= 1:
        bin_img = (img_bin * 255).astype(np.uint8)
    else:
        bin_img = img_bin.astype(np.uint8)

    h, w = bin_img.shape
    profile = np.sum(bin_img > 0, axis=1).astype(float)
    thresh_px = max(1, w * _STAFF_ROW_WIDTH_RATIO)

    thicknesses = []
    for y in staff_ys:
        y0 = max(0, y - window)
        y1 = min(h, y + window + 1)
        local = profile[y0:y1]
        count = int(np.sum(local > thresh_px))
        thicknesses.append(max(1, count))

    return int(np.median(thicknesses)) if thicknesses else 2


def remove_staff_lines(
    img_bin: np.ndarray,
    staff_lines: List[List[int]],
    thickness_margin: int = 1,
    min_run_ratio: float = 0.04,
) -> np.ndarray:
    """Xóa dòng kẻ khỏi ảnh nhị phân bằng run-length filtering.
    Chỉ xóa các "horizontal runs" (đoạn pixel trắng nằm ngang liên tục) đủ
    dài — tức là đoạn thuộc dòng kẻ. Các pixel đơn lẻ hay cụm pixel nhỏ
    (notehead, stem, ký hiệu...) được giữ lại.
    Args:
        img_bin: Ảnh nhị phân (foreground=255 hoặc 1), dtype uint8.
        staff_lines: Output từ detect_and_refine_staff_lines() —
            list các staff, mỗi staff là list 5 tọa độ y.
        thickness_margin: Số pixel dư thêm mỗi phía khi quét quanh y.
        min_run_ratio: Tỉ lệ chiều rộng tối thiểu của run bị xóa.
            Run ngắn hơn ngưỡng này được giữ lại (có thể là nốt nhạc).
            Mặc định 0.04 = 4% chiều rộng.
    Returns:
        Ảnh nhị phân sau khi xóa dòng kẻ, cùng shape và dtype với đầu vào.
    """
    if img_bin.max() <= 1:
        result = (img_bin * 255).astype(np.uint8)
    else:
        result = img_bin.copy().astype(np.uint8)

    h, w = result.shape
    min_run_len = max(3, int(w * min_run_ratio))

    for staff in staff_lines:
        # Ước lượng độ dày dòng kẻ từ spacing các line trong staff
        sorted_staff = sorted(staff)
        spacings = np.diff(sorted_staff)
        if len(spacings) > 0:
            line_thickness = max(1, int(np.median(spacings) * _LINE_THICKNESS_RATIO)) + thickness_margin
        else:
            line_thickness = 1 + thickness_margin

        for y in sorted_staff:
            y0 = max(0, y - line_thickness)
            y1 = min(h, y + line_thickness + 1)

            for row in range(y0, y1):
                pixels = result[row, :]
                # Run-length encoding thủ công trên hàng này
                runs = _find_runs(pixels)
                for (x_s, x_e) in runs:
                    if (x_e - x_s + 1) >= min_run_len:
                        result[row, x_s : x_e + 1] = 0  # xóa run dài (dòng kẻ)
                # Run ngắn (< min_run_len) giữ nguyên — có thể là notehead nhỏ

    return result


def _find_runs(row: np.ndarray) -> List[Tuple[int, int]]:
    """Trả về list (x_start, x_end) của các đoạn pixel > 0 liên tục."""
    runs: List[Tuple[int, int]] = []
    in_run = False
    run_start = 0
    for x in range(len(row)):
        if row[x] > 0:
            if not in_run:
                in_run = True
                run_start = x
        else:
            if in_run:
                runs.append((run_start, x - 1))
                in_run = False
    if in_run:
        runs.append((run_start, len(row) - 1))
    return runs


def repair_noteheads(
    img_removed: np.ndarray,
    thickness: int = 3,
) -> np.ndarray:
    """Vá lỗ hổng trong nốt nhạc bị cắt đôi bởi dòng kẻ.
    Sau khi xóa dòng kẻ, một số noteheads bị tách thành 2 mảnh trên/dưới.
    Bước này dùng morphological closing theo chiều dọc để hợp nhất lại.
    Args:
        img_removed: Ảnh nhị phân sau bước remove_staff_lines().
        thickness: Chiều cao kernel closing dọc (bằng khoảng xóa ~line_thickness*2).
    Returns:
        Ảnh sau khi vá.
    """
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, thickness * 2 + 1))
    repaired = cv2.morphologyEx(img_removed, cv2.MORPH_CLOSE, kernel_v)

    # Cleanup nhẹ để loại nhiễu còn sót
    kernel_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    repaired = cv2.morphologyEx(repaired, cv2.MORPH_CLOSE, kernel_e)

    return repaired


def staff_removal_pipeline(
    img_bin: np.ndarray,
    staff_lines: List[List[int]],
    thickness_margin: int = 1,
    min_run_ratio: float = 0.04,
    repair: bool = True,
) -> np.ndarray:
    """Pipeline hoàn chỉnh: xóa dòng kẻ và vá nốt.
    Args:
        img_bin: Ảnh nhị phân đầu vào (foreground=255 hoặc 1).
        staff_lines: Output từ detect_and_refine_staff_lines().
        thickness_margin: Pixel dư mỗi phía khi quét quanh y.
        min_run_ratio: Tỉ lệ chiều rộng tối thiểu của run bị xóa (0.04 = 4%).
        repair: Có chạy bước repair_noteheads() sau khi xóa không.
    Returns:
        Ảnh nhị phân đã loại bỏ dòng kẻ, foreground=255.
    """
    removed = remove_staff_lines(
        img_bin,
        staff_lines,
        thickness_margin=thickness_margin,
        min_run_ratio=min_run_ratio,
    )

    if repair and staff_lines:
        # Ước lượng độ dày từ staff đầu tiên để dùng làm tham số repair
        sorted_st = sorted(staff_lines[0])
        spacings = np.diff(sorted_st)
        thickness = max(2, int(np.median(spacings) * _LINE_THICKNESS_RATIO) + thickness_margin) if len(spacings) else 2
        removed = repair_noteheads(removed, thickness=thickness)

    return removed


def visualize_staff_removal(
    img_original: np.ndarray,
    img_removed: np.ndarray,
    staff_lines: Optional[List[List[int]]] = None,
) -> np.ndarray:
    """Tạo ảnh so sánh before/after và vẽ lại tọa độ dòng kẻ để debug.
    Args:
        img_original: Ảnh gốc (BGR hoặc grayscale).
        img_removed: Ảnh sau staff removal (nhị phân 0/255).
        staff_lines: Danh sách staff để vẽ lại (tùy chọn).
    Returns:
        Ảnh BGR ghép ngang: [ảnh gốc | ảnh sau removal].
    """
    # Chuyển cả hai về BGR để ghép
    if img_original.ndim == 2:
        vis_ori = cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR)
    else:
        vis_ori = img_original.copy()
        if vis_ori.dtype != np.uint8:
            vis_ori = (vis_ori * 255).astype(np.uint8)

    if img_removed.ndim == 2:
        vis_rem = cv2.cvtColor(img_removed, cv2.COLOR_GRAY2BGR)
    else:
        vis_rem = img_removed.copy()

    # Vẽ tọa độ dòng kẻ lên ảnh gốc bằng màu đỏ
    if staff_lines:
        for staff in staff_lines:
            for y in staff:
                cv2.line(vis_ori, (0, y), (vis_ori.shape[1], y), (0, 0, 255), 1)

    # Đảm bảo cùng chiều cao trước khi ghép
    h1, w1 = vis_ori.shape[:2]
    h2, w2 = vis_rem.shape[:2]
    if h1 != h2:
        h_target = max(h1, h2)
        if h1 < h_target:
            pad = np.full((h_target - h1, w1, 3), 255, dtype=np.uint8)
            vis_ori = np.vstack([vis_ori, pad])
        else:
            pad = np.full((h_target - h2, w2, 3), 255, dtype=np.uint8)
            vis_rem = np.vstack([vis_rem, pad])

    return np.hstack([vis_ori, vis_rem])