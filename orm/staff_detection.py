import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d
from typing import List, Tuple, Optional, Union

def find_peaks_profile(
    profile: np.ndarray,
    smooth_sigma: float = 2,
    min_dist: int = 6,
    min_len: int = 2,
    thresh_ratio: float = 0.6
) -> List[int]:
    """Tìm các peak staff dựa trên profile histogram đã làm mượt."""
    smooth = gaussian_filter1d(profile.astype(float), sigma=smooth_sigma)
    thresh = thresh_ratio * np.max(smooth)
    ys = []
    in_peak = False
    buf = []
    for idx, val in enumerate(smooth):
        if val >= thresh:
            in_peak = True
            buf.append(idx)
        else:
            if in_peak and len(buf) >= min_len:
                ys.append(int(np.median(buf)))
            buf = []
            in_peak = False
    if in_peak and len(buf) >= min_len:
        ys.append(int(np.median(buf)))
    # Loại peak cách nhau quá gần
    result = []
    for y in ys:
        if not result or y - result[-1] > min_dist:
            result.append(y)
    return result

def group_peaks_to_staffs(
    ys: List[int],
    spacing_tol: float = 0.22,
    min_staff_line_gap: int = 6
) -> List[List[int]]:
    """
    Gom peak y thành từng staff (5 dòng), dựa trên độ đều spacing của lines.
    """
    if not ys:
        return []
    ys = sorted(ys)
    staffs = []
    cur = [ys[0]]
    for y in ys[1:]:
        gap = y - cur[-1]
        if len(cur) < 5 or (abs(gap - np.median(np.diff(cur))) < spacing_tol * np.median(np.diff(cur))):
            cur.append(y)
        else:
            if len(cur) >= 3:
                staffs.append(cur)
            cur = [y]
    if len(cur) >= 3:
        staffs.append(cur)
    # Fix mỗi staff đủ 5 dòng
    out = []
    for st in staffs:
        st = sorted(st)
        if len(st) == 5:
            out.append(st)
        elif 3 <= len(st) < 5:
            gaps = np.diff(st)
            avg = int(np.mean(gaps)) if len(gaps) else 10
            while len(st) < 5:
                st.append(int(st[-1] + avg))
            out.append(st)
        elif len(st) > 5:
            out.append(st[:5])
    return out

def refine_staff_lines(
    staffs: List[List[int]],
    img_bin: np.ndarray,
    window: int = 5
) -> List[List[int]]:
    """Tối ưu vị trí từng dòng staff: tìm vị trí hàng y profile mạnh nhất quanh từng dòng ban đầu."""
    h = img_bin.shape[0]
    profile = np.sum(img_bin == 255, axis=1)
    result = []
    for staff in staffs:
        staff_refined = []
        for y in staff:
            y_min = max(0, y-window)
            y_max = min(h, y+window+1)
            window_profile = profile[y_min:y_max]
            if len(window_profile)>2:
                top2_idx = np.argsort(window_profile)[-2:]
                y_real = int(np.median([y_min+idx for idx in top2_idx]))
            else:
                y_real = int(np.argmax(window_profile)+y_min)
            staff_refined.append(y_real)
        result.append(staff_refined)
    return result

def detect_and_refine_staff_lines(
    img_bin: np.ndarray,
    profile_smooth: float = 2,
    peak_thresh: float = 0.6,
    refine_window: int = 5
) -> List[List[int]]:
    """
    Đầu vào: ảnh nhị phân (staff trắng), đầu ra: List các staff (mỗi staff là 5 y)
    """
    # Accept color input from callers and convert to single channel first.
    if img_bin.ndim == 3:
        img_bin = cv2.cvtColor(img_bin, cv2.COLOR_BGR2GRAY)

    # Normalize
    if img_bin.max() <= 1.0:
        img_bin = (img_bin * 255).astype(np.uint8)
    else:
        img_bin = img_bin.astype(np.uint8)

    # If input is not binary yet, binarize with Otsu before staff projection.
    uniq = np.unique(img_bin)
    if len(uniq) > 4:
        _, img_bin = cv2.threshold(img_bin, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.mean(img_bin) > 127:
        img_bin = 255 - img_bin

    # Projection + Find peak
    profile = np.sum(img_bin == 255, axis=1)
    ys = find_peaks_profile(profile, smooth_sigma=profile_smooth, thresh_ratio=peak_thresh)
    # Group và refine
    staffs = group_peaks_to_staffs(ys, spacing_tol=0.22)
    staffs_refined = refine_staff_lines(staffs, img_bin, window=refine_window)
    return staffs_refined

def crop_staffs(
    img: np.ndarray,
    staffs: List[List[int]],
    expand: int = 10,
    clip_to_neighbors: bool = True
) -> List[np.ndarray]:
    """
    Cắt từng staff thành ảnh nhỏ, mở rộng theo spacing staff.
    Nếu clip_to_neighbors=False thì sẽ ưu tiên lấy đủ cao để quan sát,
    chấp nhận có thể chồng lấn với staff lân cận.
    """
    h, w = img.shape[:2]
    crops = []
    centers = [np.mean(st) for st in staffs]
    for i, staff in enumerate(staffs):
        st = sorted(staff)
        unit = float(np.median(np.diff(st))) if len(st) >= 2 else float(expand)
        pad_top = int(max(expand, round(3.5 * unit)))
        pad_bottom = int(max(expand, round(3.0 * unit)))

        ymin = int(max(0, st[0] - pad_top))
        ymax = int(min(h, st[-1] + pad_bottom))

        # Tinh chỉnh crop bằng midpoint để tránh trùng lấn (nếu cần).
        if clip_to_neighbors:
            if i > 0:
                prev_mid = int((centers[i - 1] + centers[i]) / 2)
                ymin = max(ymin, prev_mid - int(pad_top * 0.3))
            if i < len(staffs) - 1:
                next_mid = int((centers[i] + centers[i + 1]) / 2)
                ymax = min(ymax, next_mid)
        crop = img[ymin:ymax + 1, :]
        crops.append(crop)
    return crops

def crop_staffs_advanced(
    img: np.ndarray,
    staffs: List[List[int]],
    expand_y: int = 10,
    expand_x: int = 10,
    return_bbox: bool = True,
    return_mask: bool = False
) -> List[Union[np.ndarray, Tuple[np.ndarray, Tuple[int, int, int, int]], Tuple[np.ndarray, Tuple[int, int, int, int], np.ndarray]]]:
    """Crop staff, bao gồm padding và mask cho vùng staff."""
    h, w = img.shape[:2]
    results = []
    # resize/staff cột hoàn toàn nếu muốn (ở đây mặc định lấy hết ngang ảnh)
    x_left = 0 + expand_x
    x_right = w - expand_x
    for staff in staffs:
        ymin = int(max(0, min(staff) - expand_y))
        ymax = int(min(h, max(staff) + expand_y))
        crop = img[ymin:ymax, x_left:x_right]
        if return_mask:
            mask = np.zeros((ymax - ymin, x_right - x_left), dtype="uint8")
            mask_staff_start = int(min(staff) - ymin)
            mask_staff_end = int(max(staff) - ymin)
            mask[mask_staff_start:mask_staff_end + 1, :] = 255
            if return_bbox:
                results.append((crop, (ymin, ymax, x_left, x_right), mask))
            else:
                results.append((crop, mask))
        elif return_bbox:
            results.append((crop, (ymin, ymax, x_left, x_right)))
        else:
            results.append(crop)
    return results


def detect_vertical_connectors(
    img_bin: np.ndarray,
    min_height: int = 40,
    x_merge_tol: int = 8,
) -> List[Tuple[int, int, int, int]]:
    """Detect vertical connector candidates (e.g., piano barlines) from a binary image."""
    if img_bin.max() <= 1.0:
        bin_img = (img_bin * 255).astype(np.uint8)
    else:
        bin_img = img_bin.astype(np.uint8)
    if np.mean(bin_img) > 127:
        bin_img = 255 - bin_img

    # Keep vertical strokes and bridge small breaks.
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 19))
    vertical_map = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, vertical_kernel)
    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))
    vertical_map = cv2.morphologyEx(vertical_map, cv2.MORPH_CLOSE, bridge_kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats((vertical_map > 0).astype(np.uint8), 8)
    segments: List[Tuple[int, int, int, int]] = []
    for i in range(1, n_labels):
        x, y, w, h, _ = stats[i]
        if h < min_height:
            continue
        if h < max(20, 3 * w):
            continue
        x_center = int(x + w // 2)
        segments.append((x_center, int(y), x_center, int(y + h - 1)))

    if not segments:
        return []

    # Merge broken segments that belong to the same vertical column.
    segments = sorted(segments, key=lambda s: (s[0], s[1]))
    merged: List[List[int]] = []
    for x1, y1, _, y2 in segments:
        if not merged:
            merged.append([x1, y1, y2])
            continue
        last_x, last_y1, last_y2 = merged[-1]
        if abs(x1 - last_x) <= x_merge_tol:
            merged[-1][0] = int(round((last_x + x1) / 2))
            merged[-1][1] = min(last_y1, y1)
            merged[-1][2] = max(last_y2, y2)
        else:
            merged.append([x1, y1, y2])

    return [(x, y1, x, y2) for x, y1, y2 in merged]

def group_grand_staff(
    staffs: List[List[int]], 
    img_bin: Optional[np.ndarray] = None,
    max_staff_gap_ratio: Optional[float] = None
) -> List[Tuple[List[int], List[int]]]:
    """
    Group grand staff (piano) theo hai tiêu chí: 
    - Ratio spacing, hoặc 
    - Có barline lớn nối liền giữa hai staff
    """
    staffs = sorted(staffs, key=lambda st: np.mean(st))
    if len(staffs) < 2:
        return []

    ratios = []
    pair_meta = []
    for i in range(len(staffs) - 1):
        upper_staff = staffs[i]
        lower_staff = staffs[i + 1]
        gap = float(min(lower_staff) - max(upper_staff))
        spacing_up = float(np.median(np.diff(upper_staff)))
        spacing_lo = float(np.median(np.diff(lower_staff)))
        mean_spacing = max((spacing_up + spacing_lo) / 2.0, 1e-6)
        ratio = gap / mean_spacing
        ratios.append(ratio)
        pair_meta.append((i, upper_staff, lower_staff, gap, mean_spacing, ratio))

    if max_staff_gap_ratio is None:
        if len(ratios) >= 2:
            q1 = float(np.percentile(ratios, 25))
            q3 = float(np.percentile(ratios, 75))
            max_staff_gap_ratio = (q1 + q3) / 2.0
        else:
            max_staff_gap_ratio = 8.0
        max_staff_gap_ratio = float(np.clip(max_staff_gap_ratio, 5.5, 12.0))

    barlines: List[Tuple[int, int, int, int]] = []
    if img_bin is not None:
        barlines = detect_vertical_connectors(img_bin, min_height=40, x_merge_tol=8)

    grand_staffs = []
    i = 0
    while i < len(staffs) - 1:
        upper_staff = staffs[i]
        lower_staff = staffs[i + 1]
        gap = float(min(lower_staff) - max(upper_staff))
        spacing_up = float(np.median(np.diff(upper_staff)))
        spacing_lo = float(np.median(np.diff(lower_staff)))
        mean_spacing = max((spacing_up + spacing_lo) / 2.0, 1e-6)
        ratio = gap / mean_spacing

        # Ưu tiên barline dọc nối giữa hai staff (nếu có)
        has_barline = False
        if barlines:
            upper_y1 = min(upper_staff)
            upper_y2 = max(upper_staff)
            lower_y1 = min(lower_staff)
            lower_y2 = max(lower_staff)
            span_top = upper_y1 - int(1.5 * mean_spacing)
            span_bottom = lower_y2 + int(1.5 * mean_spacing)
            for x1, y1, x2, y2 in barlines:
                x_span = abs(x2 - x1)
                y_span = abs(y2 - y1)
                if x_span > max(6, int(0.25 * mean_spacing)):
                    continue
                if y_span < int(2.0 * mean_spacing):
                    continue
                # Allow partial coverage, because the connector can be broken by symbols/noise.
                top = min(y1, y2)
                bottom = max(y1, y2)
                overlap = max(0, min(bottom, span_bottom) - max(top, span_top))
                required = int(0.55 * max(1, span_bottom - span_top))
                if overlap >= required:
                    has_barline = True
                    break

        if has_barline or ratio < max_staff_gap_ratio:
            grand_staffs.append((upper_staff, lower_staff))
            i += 2
        else:
            i += 1

    return grand_staffs