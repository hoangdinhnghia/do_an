import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from .bbox import merge_nearby_bbox, rm_merge_overlap_bbox
from .constant import M2_CH_NOTEHEAD, NOTE_CONF_THRESH
from .logger import get_logger

logger = get_logger(__name__)

# Định nghĩa NoteheadBBox: (x, y, w, h, cx, cy)
NoteheadBBox = Tuple[int, int, int, int, int, int]

#kết quả trả về của staff detect sẽ là:
NoteheadStaffResult = Tuple[int, List[int], List[NoteheadBBox], np.ndarray]  # (staff_index, staff_y, list notehead box, annotated_crop)


def _to_binary_foreground(gray: np.ndarray) -> np.ndarray:
    """Convert input to clean binary map where foreground symbols are 255."""
    if gray.max() <= 1.0:
        gray_u8 = (gray * 255).astype(np.uint8)
    else:
        gray_u8 = gray.astype(np.uint8)

    uniq = np.unique(gray_u8)
    if len(uniq) <= 3:
        bin_map = (gray_u8 > 0).astype(np.uint8) * 255
    else:
        _, bw = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Symbols occupy less pixels than background, keep sparse class as foreground.
        white_ratio = float(np.count_nonzero(bw)) / float(bw.size)
        bin_map = bw if white_ratio < 0.5 else (255 - bw)

    # Bridge small gaps inside noteheads while preserving thin components.
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
    """Post-process notehead boxes using bbox utilities, then IoU dedup fallback."""
    if not noteheads:
        return []

    def _merge_close_boxes(box_list: List[Tuple[int, int, int, int]], gap_limit: int, overlap_limit: float) -> List[Tuple[int, int, int, int]]:
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

    # 1) Use bbox utilities to merge close boxes and handle heavy overlap.
    # Convert (x, y, w, h, cx, cy) -> (x1, y1, x2, y2)
    boxes = [(x, y, x + w, y + h) for (x, y, w, h, _, _) in noteheads]
    sizes = [max(1, min(w, h)) for (_, _, w, h, _, _) in noteheads]
    merge_dist = max(3, int(np.median(sizes) * 0.75))
    try:
        boxes = merge_nearby_bbox(boxes, distance=merge_dist, x_factor=1, y_factor=1)
        boxes = rm_merge_overlap_bbox(boxes, mode="merge", overlap_ratio=0.40)
        boxes = rm_merge_overlap_bbox(boxes, mode="remove", overlap_ratio=0.75)
    except Exception:
        # Keep pipeline robust when bbox post-processing cannot be applied.
        pass

    # 2) Merge small fragments that belong to the same notehead-like symbol.
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
    """Check whether a candidate notehead has a nearby vertical stem in the original binary image."""
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

    # 2. Chuẩn hóa foreground=255 và ước lượng đơn vị khoảng cách dòng staff.
    note_img = _to_binary_foreground(gray)
    unit = _estimate_staff_unit(staff_y, crop_h=note_img.shape[0])

    # 2.1. Trích blob gần kích thước notehead bằng opening ellipse để bỏ stem mảnh.
    core_size = max(3, int(round(0.34 * unit)))
    core_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (core_size, core_size))
    note_core = cv2.morphologyEx(note_img, cv2.MORPH_OPEN, core_kernel)
    note_core = cv2.morphologyEx(
        note_core,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )

    # 3. Scale-invariant constraints by staff unit.
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

        # Thêm lọc compactness/circularity loại bỏ clef, rest, ký hiệu phụ
        perimeter = cv2.arcLength(c, True)
        if perimeter <= 0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.22:
            continue

        solidity = area / float(cv2.contourArea(cv2.convexHull(c))) if cv2.contourArea(cv2.convexHull(c)) > 0 else 0.0
        extent = area / float(max(1, w * h))
        if solidity < 0.68 or extent < 0.32:
            continue

        cx, cy = x + w // 2, y + h // 2

        # Loại vùng khóa nhạc / time signature ở mé trái, nơi thường sinh false positive.
        if x + w <= left_margin:
            continue

        # Nếu có list staff_y (5 dòng y), chỉ nhận notehead gần vùng đó
        if staff_y is not None and len(staff_y) > 0:
            y0 = min(staff_y)
            y4 = max(staff_y)
            pad = max(int(1.15 * unit), int((y4 - y0) * 0.25))
            if not (y0 - pad <= cy <= y4 + pad):
                continue

        if not _has_attached_stem(note_img, (x, y, w, h), unit):
            # Whole notes/rare cases can still pass if the blob is sufficiently round and central.
            if not (ar >= 0.85 and ar <= 1.20 and circularity >= 0.70 and solidity >= 0.75 and extent >= 0.40 and area >= 0.28 * unit * unit):
                continue

        notehead_list.append((x, y, w, h, cx, cy))

    return _deduplicate_noteheads(notehead_list)

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
    """
    Pipeline tổng hợp: từ ảnh gốc đã loại staff line, cắt từng staff, detect notehead, trả về kết quả.
    Args:
        img_no_staff: Ảnh đã loại staff line (grayscale hoặc BGR)
        staff_line: List các staff line (mỗi staff là list 5 y)
        expand: Số pixel mở rộng vùng cắt trên/dưới so với y0/y4 của staff
        min_area, max_area, aspect_ratio: Tham số lọc contour notehead
    Returns:
        List kết quả cho từng staff: (staff_index, list notehead box, staff_crop_img)
    """
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


# ---------------------------------------------------------------------------
# Semantic-map-based notehead extraction
# ---------------------------------------------------------------------------

def extract_noteheads_from_semantic_map(
    semantic_prob: np.ndarray,
    staff_lines: Optional[List[List[int]]] = None,
    conf_thresh: float = NOTE_CONF_THRESH,
    min_area: int = 12,
    max_area: int = 2000,
    aspect_ratio: Tuple[float, float] = (0.35, 2.0),
    min_circularity: float = 0.15,
) -> List[NoteheadBBox]:
    """Extract notehead bounding boxes directly from the semantic probability map.

    This function uses the model's notehead channel (ch 1) instead of the
    binary-image-after-staff-removal approach used by
    ``detect_notehead_contour``.  It gives cleaner results on pages where the
    CV-based staff removal leaves artefacts.

    Parameters
    ----------
    semantic_prob:
        (H, W, 4) float32 probability map from the semantic model.
    staff_lines:
        Optional list of staff systems.  When provided the results are filtered
        to noteheads whose centre falls within a staff region.
    conf_thresh:
        Binarisation threshold for the notehead channel.
    min_area, max_area:
        Blob area filter (pixels).
    aspect_ratio:
        Allowed (w/h) range.
    min_circularity:
        Minimum 4πA/P² circularity to reject elongated stems/beams.

    Returns
    -------
    List of ``(x, y, w, h, cx, cy)`` tuples.
    """
    mask = (semantic_prob[:, :, M2_CH_NOTEHEAD] >= conf_thresh).astype(np.uint8) * 255

    # Morphological closing to unite small gaps within one notehead blob
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results: List[NoteheadBBox] = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        ar = (w / h) if h else 0.0
        if ar < aspect_ratio[0] or ar > aspect_ratio[1]:
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter <= 0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity < min_circularity:
            continue
        cx, cy = x + w // 2, y + h // 2
        results.append((x, y, w, h, cx, cy))

    # Optional: filter to staff regions
    if staff_lines:
        results = _filter_to_staff_regions(results, staff_lines)

    return _deduplicate_noteheads(results)


def _filter_to_staff_regions(
    noteheads: List[NoteheadBBox],
    staff_lines: List[List[int]],
) -> List[NoteheadBBox]:
    """Keep only noteheads whose centre falls near a staff system."""
    kept: List[NoteheadBBox] = []
    for bbox in noteheads:
        x, y, w, h, cx, cy = bbox
        accepted = False
        for staff_y in staff_lines:
            sorted_y = sorted(staff_y)
            unit = float(np.median(np.diff(sorted_y))) if len(sorted_y) >= 2 else 10.0
            pad = max(int(1.5 * unit), int((sorted_y[-1] - sorted_y[0]) * 0.25))
            if sorted_y[0] - pad <= cy <= sorted_y[-1] + pad:
                accepted = True
                break
        if accepted:
            kept.append(bbox)
    return kept


# ---------------------------------------------------------------------------
# Fused pipeline: combine CV + model-driven detections
# ---------------------------------------------------------------------------

def fused_notehead_detection_pipeline(
    img_no_staff: np.ndarray,
    staff_lines: List[List[int]],
    semantic_prob: Optional[np.ndarray] = None,
    expand: int = 20,
    note_conf_thresh: float = NOTE_CONF_THRESH,
    min_area: int = 18,
    max_area: int = 1200,
    aspect_ratio: Tuple[float, float] = (0.45, 1.8),
) -> List[NoteheadStaffResult]:
    """Fused notehead detection combining classic CV and semantic-model detections.

    If *semantic_prob* is provided this function:
    1. Runs the classic CV pipeline (``notehead_detection_pipeline``).
    2. Runs the model-driven extraction (``extract_noteheads_from_semantic_map``).
    3. Merges the two result sets per staff by deduplication (IoU-based NMS).

    When *semantic_prob* is None this falls back to the classic CV pipeline.

    Parameters
    ----------
    img_no_staff:
        Binary image (uint8, foreground = 255) after staff removal.
    staff_lines:
        Detected staff systems.
    semantic_prob:
        (H, W, 4) float32 from the semantic model (optional).
    expand:
        Crop margin for classic CV pipeline.
    note_conf_thresh:
        Binarisation threshold for semantic notehead channel.
    min_area, max_area, aspect_ratio:
        Classic CV contour filters.

    Returns
    -------
    List of ``(staff_idx, staff_y, noteheads, annotated)`` tuples.
    """
    # --- Classic CV detections (per-staff crops) ---
    cv_results: List[NoteheadStaffResult] = notehead_detection_pipeline(
        img_no_staff, staff_lines, expand=expand,
        min_area=min_area, max_area=max_area, aspect_ratio=aspect_ratio,
    )

    if semantic_prob is None:
        return cv_results

    # --- Model-driven detections (full image) ---
    model_noteheads_all: List[NoteheadBBox] = extract_noteheads_from_semantic_map(
        semantic_prob,
        staff_lines=staff_lines,
        conf_thresh=note_conf_thresh,
        min_area=min_area,
        max_area=max_area,
        aspect_ratio=aspect_ratio,
    )

    # Assign model-driven noteheads to their nearest staff system
    model_per_staff: Dict[int, List[NoteheadBBox]] = {
        i: [] for i in range(len(staff_lines))
    }
    for bbox in model_noteheads_all:
        x, y, w, h, cx, cy = bbox
        best_idx, best_dist = 0, float("inf")
        for si, staff_y in enumerate(staff_lines):
            sc = float(np.mean(staff_y))
            dist = abs(cy - sc)
            if dist < best_dist:
                best_dist, best_idx = dist, si
        model_per_staff[best_idx].append(bbox)

    # --- Merge CV + model per staff ---
    fused_results: List[NoteheadStaffResult] = []
    for cv_entry in cv_results:
        idx, staff_y, cv_noteheads, _ = cv_entry
        model_noteheads = model_per_staff.get(idx, [])
        merged = _deduplicate_noteheads(cv_noteheads + model_noteheads, iou_thr=0.35)
        merged.sort(key=lambda b: (b[1], b[0]))   # sort top-left to bottom-right

        # Build annotated crop from original (use whole-image view for annotating)
        h_img, w_img = img_no_staff.shape[:2]
        sorted_y_s = sorted(staff_y)
        unit = float(np.median(np.diff(sorted_y_s))) if len(sorted_y_s) >= 2 else 10.0
        pad = max(expand, int(round(3.5 * unit)))
        y_min = max(0, sorted_y_s[0] - pad)
        y_max = min(h_img - 1, sorted_y_s[-1] + pad)
        crop_ann = img_no_staff[y_min : y_max + 1, :]
        if crop_ann.ndim == 2:
            crop_ann = cv2.cvtColor(crop_ann, cv2.COLOR_GRAY2BGR)

        # Translate global bbox to crop coords for annotation
        local_noteheads = [
            (x, y - y_min, w, h, cx, cy - y_min)
            for (x, y, w, h, cx, cy) in merged
        ]
        annotated = annotate_noteheads(crop_ann, local_noteheads)

        fused_results.append((idx, staff_y, merged, annotated))
        logger.debug("Staff %d: CV=%d model=%d fused=%d",
                     idx, len(cv_noteheads), len(model_noteheads), len(merged))

    return fused_results