from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    from .bbox import merge_nearby_bbox, rm_merge_overlap_bbox
except ImportError:
    from bbox import merge_nearby_bbox, rm_merge_overlap_bbox


# NoteheadBBox: (x, y, w, h, cx, cy)
NoteheadBBox = Tuple[int, int, int, int, int, int]
# Per-staff result: (staff_index, staff_y_lines, notehead_boxes, annotated_crop)
NoteheadStaffResult = Tuple[int, List[int], List[NoteheadBBox], np.ndarray]


def _to_binary_foreground(gray: np.ndarray) -> np.ndarray:
    """Convert input to a binary map where foreground symbols are 255."""
    if gray.max() <= 1.0:
        gray_u8 = (gray * 255).astype(np.uint8)
    else:
        gray_u8 = gray.astype(np.uint8)

    uniq = np.unique(gray_u8)
    if len(uniq) <= 3:
        bin_map = (gray_u8 > 0).astype(np.uint8) * 255
    else:
        _, bw = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        white_ratio = float(np.count_nonzero(bw)) / float(bw.size)
        # Keep sparse class as foreground when background dominates.
        bin_map = bw if white_ratio < 0.5 else (255 - bw)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bin_map = cv2.morphologyEx(bin_map, cv2.MORPH_CLOSE, close_kernel)
    return bin_map


def _estimate_staff_unit(staff_y: Optional[List[int]], crop_h: int) -> float:
    if staff_y is not None and len(staff_y) >= 2:
        gaps = np.diff(sorted(staff_y))
        if len(gaps) > 0:
            unit = float(np.median(gaps))
            if unit > 1:
                return unit
    return max(8.0, crop_h / 12.0)


def _crop_staffs_with_bounds(
    img: np.ndarray,
    staffs: List[List[int]],
    expand: int = 20,
) -> List[Tuple[np.ndarray, int, int]]:
    """Crop staffs and return (crop, y_min, y_max) using neighbor-aware clipping."""
    h, _ = img.shape[:2]
    if not staffs:
        return []

    centers = [float(np.mean(st)) for st in staffs]
    out: List[Tuple[np.ndarray, int, int]] = []
    for i, staff in enumerate(staffs):
        st = sorted(staff)
        unit = float(np.median(np.diff(st))) if len(st) >= 2 else float(expand)
        pad_top = int(max(expand, round(3.5 * unit)))
        pad_bottom = int(max(expand, round(3.0 * unit)))

        y_min = int(max(0, st[0] - pad_top))
        y_max = int(min(h - 1, st[-1] + pad_bottom))

        if i > 0:
            prev_mid = int((centers[i - 1] + centers[i]) / 2)
            y_min = max(y_min, prev_mid - int(pad_top * 0.3))
        if i < len(staffs) - 1:
            next_mid = int((centers[i] + centers[i + 1]) / 2)
            y_max = min(y_max, next_mid)

        crop = img[y_min : y_max + 1, :]
        out.append((crop, y_min, y_max))

    return out


def _deduplicate_noteheads(noteheads: List[NoteheadBBox], iou_thr: float = 0.4) -> List[NoteheadBBox]:
    """Merge/remove overlapping notehead boxes with bbox utilities and IoU fallback."""
    if not noteheads:
        return []

    def _merge_close_boxes(
        box_list: List[Tuple[int, int, int, int]],
        gap_limit: int,
        overlap_limit: float,
    ) -> List[Tuple[int, int, int, int]]:
        if not box_list:
            return []

        merged = sorted(box_list, key=lambda b: (b[0], b[1]))
        changed = True
        while changed:
            changed = False
            next_boxes: List[Tuple[int, int, int, int]] = []
            current = merged[0]
            for box in merged[1:]:
                cx1, cy1, cx2, cy2 = current
                bx1, by1, bx2, by2 = box
                horizontal_gap = max(0, max(bx1 - cx2, cx1 - bx2))
                top = max(cy1, by1)
                bottom = min(cy2, by2)
                vertical_overlap = max(0, bottom - top)
                current_h = max(1, cy2 - cy1)
                box_h = max(1, by2 - by1)
                overlap_ratio = vertical_overlap / float(min(current_h, box_h))

                if horizontal_gap <= gap_limit and overlap_ratio >= overlap_limit:
                    current = (min(cx1, bx1), min(cy1, by1), max(cx2, bx2), max(cy2, by2))
                    changed = True
                else:
                    next_boxes.append(current)
                    current = box
            next_boxes.append(current)
            merged = next_boxes
        return merged

    boxes = [(x, y, x + w, y + h) for (x, y, w, h, _, _) in noteheads]
    sizes = [max(1, min(w, h)) for (_, _, w, h, _, _) in noteheads]
    merge_dist = max(3, int(np.median(sizes) * 0.75))
    try:
        boxes = merge_nearby_bbox(boxes, distance=merge_dist, x_factor=1, y_factor=1)
        boxes = rm_merge_overlap_bbox(boxes, mode="merge", overlap_ratio=0.40)
        boxes = rm_merge_overlap_bbox(boxes, mode="remove", overlap_ratio=0.75)
    except Exception:
        pass

    fragment_gap = max(3, int(round(np.median(sizes) * 1.25)))
    boxes = _merge_close_boxes(boxes, gap_limit=fragment_gap, overlap_limit=0.20)

    restored: List[NoteheadBBox] = []
    for x1, y1, x2, y2 in boxes:
        w = max(1, int(x2 - x1))
        h = max(1, int(y2 - y1))
        cx = int(x1 + w // 2)
        cy = int(y1 + h // 2)
        restored.append((int(x1), int(y1), w, h, cx, cy))

    ordered = sorted(restored, key=lambda b: b[2] * b[3], reverse=True)
    kept: List[NoteheadBBox] = []

    def _iou(a: NoteheadBBox, b: NoteheadBBox) -> float:
        ax1, ay1, aw, ah, _, _ = a
        bx1, by1, bw, bh, _, _ = b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        union = aw * ah + bw * bh - inter
        return float(inter) / float(union) if union > 0 else 0.0

    for cand in ordered:
        if all(_iou(cand, old) < iou_thr for old in kept):
            kept.append(cand)

    kept.sort(key=lambda b: (b[1], b[0]))
    return kept


def _longest_run_1d(values: np.ndarray) -> Tuple[int, int, int]:
    """Return (start, end, length) of the longest positive run in a 1D array."""
    best_start = best_end = best_len = 0
    run_start = None
    for idx, val in enumerate(values):
        if val > 0:
            if run_start is None:
                run_start = idx
        elif run_start is not None:
            run_end = idx - 1
            run_len = run_end - run_start + 1
            if run_len > best_len:
                best_start, best_end, best_len = run_start, run_end, run_len
            run_start = None
    if run_start is not None:
        run_end = len(values) - 1
        run_len = run_end - run_start + 1
        if run_len > best_len:
            best_start, best_end, best_len = run_start, run_end, run_len
    return best_start, best_end, best_len


def _has_attached_stem(bin_img: np.ndarray, bbox: Tuple[int, int, int, int], unit: float) -> bool:
    """Check whether a candidate notehead has a nearby vertical stem in binary image."""
    x, y, w, h = bbox
    img_h, img_w = bin_img.shape[:2]
    search_margin = max(3, int(round(0.40 * unit)))
    stem_min_len = max(5, int(round(0.65 * unit)))
    overlap_min = max(1, int(round(0.05 * unit)))
    y0 = max(0, y - int(round(0.35 * unit)))
    y1 = min(img_h, y + h + int(round(0.35 * unit)))

    bands = [
        (max(0, x - search_margin), x),
        (min(img_w, x + w), min(img_w, x + w + search_margin)),
    ]

    for x0, x1 in bands:
        if x1 <= x0:
            continue
        strip = bin_img[y0:y1, x0:x1]
        if strip.size == 0:
            continue
        for col in range(strip.shape[1]):
            top, bottom, run_len = _longest_run_1d(strip[:, col])
            if run_len < stem_min_len:
                continue
            abs_top = y0 + top
            abs_bottom = y0 + bottom
            overlap = max(0, min(y + h, abs_bottom) - max(y, abs_top) + 1)
            if overlap >= overlap_min:
                return True
    return False


def detect_notehead_contour(
    staff_crop_img: np.ndarray,
    staff_y: Optional[List[int]] = None,
    min_area: int = 12,
    max_area: int = 2000,
    aspect_ratio: Tuple[float, float] = (0.35, 2.0),
) -> List[NoteheadBBox]:
    """Detect noteheads from a staff crop with morphology and shape filters."""
    if staff_crop_img.ndim == 3:
        gray = cv2.cvtColor(staff_crop_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = staff_crop_img.copy()

    note_img = _to_binary_foreground(gray)
    unit = _estimate_staff_unit(staff_y, crop_h=note_img.shape[0])

    core_size = max(3, int(round(0.34 * unit)))
    core_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (core_size, core_size))
    note_core = cv2.morphologyEx(note_img, cv2.MORPH_OPEN, core_kernel)
    note_core = cv2.morphologyEx(
        note_core,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )

    min_area_eff = max(min_area, int(0.06 * unit * unit))
    max_area_eff = min(max_area, int(0.95 * unit * unit))
    min_w = max(2, int(round(0.22 * unit)))
    max_w = max(min_w + 1, int(round(1.75 * unit)))
    min_h = max(2, int(round(0.18 * unit)))
    max_h = max(min_h + 1, int(round(1.80 * unit)))

    contours, _ = cv2.findContours(note_core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    notehead_list: List[NoteheadBBox] = []
    left_margin = max(260, int(round(staff_crop_img.shape[1] * 0.050)))

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        ar = (w / h) if h else 0.0

        if area < min_area_eff or area > max_area_eff:
            continue
        if w < min_w or w > max_w or h < min_h or h > max_h:
            continue
        if ar < aspect_ratio[0] or ar > aspect_ratio[1]:
            continue

        perimeter = cv2.arcLength(c, True)
        if perimeter <= 0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.22:
            continue

        hull_area = cv2.contourArea(cv2.convexHull(c))
        solidity = area / float(hull_area) if hull_area > 0 else 0.0
        extent = area / float(max(1, w * h))
        if solidity < 0.68 or extent < 0.32:
            continue

        cx, cy = x + w // 2, y + h // 2

        if x + w <= left_margin:
            continue

        if staff_y is not None and len(staff_y) > 0:
            y0 = min(staff_y)
            y4 = max(staff_y)
            pad = max(int(1.15 * unit), int((y4 - y0) * 0.25))
            if not (y0 - pad <= cy <= y4 + pad):
                continue

        if not _has_attached_stem(note_img, (x, y, w, h), unit):
            if not (
                ar >= 0.85
                and ar <= 1.20
                and circularity >= 0.70
                and solidity >= 0.75
                and extent >= 0.40
                and area >= 0.28 * unit * unit
            ):
                continue

        notehead_list.append((x, y, w, h, cx, cy))

    return _deduplicate_noteheads(notehead_list)


def annotate_noteheads(
    img: np.ndarray,
    noteheads: List[NoteheadBBox],
) -> np.ndarray:
    """Draw notehead bounding boxes and centers for visualization."""
    img_vis = img.copy()
    if img_vis.ndim == 2:
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h, cx, cy) in noteheads:
        cv2.rectangle(img_vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.circle(img_vis, (cx, cy), 2, (255, 0, 0), -1)
    return img_vis


def notehead_detection_pipeline(
    img_no_staff: np.ndarray,
    staff_line: List[List[int]],
    expand: int = 20,
    min_area: int = 18,
    max_area: int = 1200,
    aspect_ratio: Tuple[float, float] = (0.45, 1.8),
) -> List[NoteheadStaffResult]:
    """Detect noteheads per staff from a no-staff image."""
    crop_entries = _crop_staffs_with_bounds(img_no_staff, staff_line, expand=expand)
    results: List[NoteheadStaffResult] = []

    for idx, (staff_y, crop_entry) in enumerate(zip(staff_line, crop_entries)):
        crop, y_min, _ = crop_entry
        staff_y_local = [int(y - y_min) for y in staff_y]
        noteheads = detect_notehead_contour(
            crop,
            staff_y=staff_y_local,
            min_area=min_area,
            max_area=max_area,
            aspect_ratio=aspect_ratio,
        )
        annotated = annotate_noteheads(crop, noteheads)
        results.append((idx, staff_y, noteheads, annotated))

    return results
