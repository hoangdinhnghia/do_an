"""Notehead detection from the Detailed Semantic U-Net (Stream 2).

The Detailed Semantic Model (2nd_model.onnx) produces a (H, W, 4) float32
probability map where, for each pixel:

    channel 0 = background
    channel 1 = stem / beam
    channel 2 = notehead   ← primary channel used here
    channel 3 = other symbol (clef, accidental, rest …)

Public API
----------
    extract_noteheads_from_prob_map()  — threshold + filter blobs from a semantic map
    annotate_noteheads()               — draw bounding boxes on an image for visualisation
    notehead_detection_pipeline()      — end-to-end per-staff detection via U-Net
    _assign_noteheads_to_staves()      — assign global noteheads to nearest staff
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import List, Optional, Tuple

from .staff_detection import crop_staffs

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# Notehead bounding box: (x, y, w, h, cx, cy)
NoteheadBBox = Tuple[int, int, int, int, int, int]

# Per-staff result: (staff_index, staff_y_lines, noteheads, annotated_crop)
NoteheadStaffResult = Tuple[int, List[int], List[NoteheadBBox], np.ndarray]

# Default notehead channel in 2nd_model.onnx (verified empirically)
_NOTEHEAD_CHANNEL = 2

# ---------------------------------------------------------------------------
# Core detection function
# ---------------------------------------------------------------------------

def extract_noteheads_from_prob_map(
    prob_map: np.ndarray,
    notehead_channel: int = _NOTEHEAD_CHANNEL,
    conf_thresh: float = 0.4,
    min_area: int = 12,
    max_area: int = 2000,
    aspect_ratio: Tuple[float, float] = (0.35, 2.0),
) -> List[NoteheadBBox]:
    """Detect noteheads from a U-Net semantic probability map.

    Thresholds the notehead channel of *prob_map*, closes small intra-blob
    gaps with a morphological close, then filters connected components by
    area, aspect ratio, and circularity to retain only notehead-shaped blobs.

    Args:
        prob_map         : (H, W, C) float32 output of
                           ``DetailedSemanticModel.predict_full()``.
        notehead_channel : Index of the notehead probability channel
                           (default 2 for 2nd_model.onnx).
        conf_thresh      : Minimum notehead probability for a pixel to be "on".
        min_area         : Minimum blob area in pixels.
        max_area         : Maximum blob area in pixels.
        aspect_ratio     : Acceptable (min, max) width/height ratio.

    Returns:
        List of ``(x, y, w, h, cx, cy)`` tuples in image coordinates.
    """
    mask = (prob_map[:, :, notehead_channel] >= conf_thresh).astype(np.uint8) * 255

    # Close small intra-blob gaps
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

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
        if circularity < 0.15:
            continue
        results.append((x, y, w, h, x + w // 2, y + h // 2))
    return results


# ---------------------------------------------------------------------------
# Visualisation helper
# ---------------------------------------------------------------------------

def annotate_noteheads(
    img: np.ndarray,
    noteheads: List[NoteheadBBox],
) -> np.ndarray:
    """Draw bounding boxes and center dots on an image for visualisation.

    Args:
        img       : Source image (BGR or grayscale).
        noteheads : List of ``(x, y, w, h, cx, cy)`` bounding boxes.

    Returns:
        Copy of *img* with each notehead marked with a red bounding box
        and a blue center dot.
    """
    img_vis = img.copy()
    if img_vis.ndim == 2:
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h, cx, cy) in noteheads:
        cv2.rectangle(img_vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.circle(img_vis, (cx, cy), 2, (255, 0, 0), -1)
    return img_vis


# ---------------------------------------------------------------------------
# Staff assignment helper
# ---------------------------------------------------------------------------

def _assign_noteheads_to_staves(
    noteheads: List[NoteheadBBox],
    staff_lines: List[List[int]],
) -> List[List[NoteheadBBox]]:
    """Assign each global notehead to its nearest staff by y-centroid proximity.

    For each notehead centroid ``cy`` the vertical distance to the centre of
    every staff is computed and the notehead is assigned to the closest staff.
    Notes that fall outside a staff bounding box (e.g. ledger lines) are still
    assigned to the nearest staff, letting the pitch module handle them.

    Args:
        noteheads   : List of ``(x, y, w, h, cx, cy)`` in global image coords.
        staff_lines : List of staves; each staff is a list of 5 y-coords.

    Returns:
        List of the same length as *staff_lines*, where each element is the
        sub-list of noteheads assigned to that staff.
    """
    per_staff: List[List[NoteheadBBox]] = [[] for _ in staff_lines]
    if not staff_lines or not noteheads:
        return per_staff

    staff_centers = [float(sum(s)) / len(s) for s in staff_lines]
    for nh in noteheads:
        cy = nh[5]
        best_idx = int(
            min(range(len(staff_centers)), key=lambda i: abs(staff_centers[i] - cy))
        )
        per_staff[best_idx].append(nh)
    return per_staff


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def notehead_detection_pipeline(
    img_bgr: np.ndarray,
    staff_lines: List[List[int]],
    semantic_model=None,
    conf_thresh: float = 0.4,
    overlap: int = 64,
    max_side: Optional[int] = 2048,
) -> List[NoteheadStaffResult]:
    """Detect and assign noteheads to staves using the Detailed Semantic U-Net.

    Runs the Detailed Semantic Model (2nd_model.onnx) on the full image to
    produce a semantic probability map, extracts all noteheads via
    :func:`extract_noteheads_from_prob_map`, then assigns each notehead to
    its nearest staff using :func:`_assign_noteheads_to_staves`.

    Args:
        img_bgr        : Full input image (BGR uint8).
        staff_lines    : List of staves; each staff is a list of 5 y-coords
                         (from
                         :class:`~orm.model_inference.StafflineSegmentationModel`).
        semantic_model : Pre-loaded
                         :class:`~orm.model_inference.DetailedSemanticModel`
                         instance. Created automatically if *None*.
        conf_thresh    : Notehead confidence threshold.
        overlap        : Tile-and-stitch overlap in pixels.
        max_side       : Maximum image side before auto-downscaling.

    Returns:
        List of ``(staff_index, staff_y_lines, noteheads, annotated_crop)``
        tuples, one per staff.  Noteheads are in *global* image coordinates.
        The annotated crop is the staff region extracted from *img_bgr* with
        all detected noteheads drawn on top.
    """
    # Lazy import avoids a circular dependency at module load time.
    from .model_inference import DetailedSemanticModel

    if semantic_model is None:
        semantic_model = DetailedSemanticModel()

    prob_map = semantic_model.predict_full(img_bgr, overlap=overlap, max_side=max_side)
    all_noteheads = extract_noteheads_from_prob_map(prob_map, conf_thresh=conf_thresh)
    per_staff = _assign_noteheads_to_staves(all_noteheads, staff_lines)

    # Annotate all noteheads on a copy of the full image, then crop per staff.
    # Drawing on the full image before cropping avoids any coordinate-offset
    # arithmetic while still producing correctly positioned bounding boxes.
    img_annotated = annotate_noteheads(img_bgr, all_noteheads)
    annotated_crops = crop_staffs(img_annotated, staff_lines)

    results: List[NoteheadStaffResult] = []
    for idx, (staff_y, noteheads, crop_ann) in enumerate(
        zip(staff_lines, per_staff, annotated_crops)
    ):
        results.append((idx, staff_y, noteheads, crop_ann))
    return results