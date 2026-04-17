"""Barline detection module.

Detects vertical barlines in a music score image.

Algorithm
---------
1. From the binary image (foreground = 255, staff lines already removed) keep
   only vertical strokes using a tall morphological structuring element.
2. Apply connected-component analysis; keep components whose height is at least
   ``BARLINE_MIN_HEIGHT_RATIO × staff_unit`` and whose width is at most
   ``BARLINE_MAX_WIDTH_PX``.
3. Merge fragments at the same x position (within ``BARLINE_MERGE_TOL_PX``).
4. Assign each barline to the nearest staff system.

Because barlines cross *all* staff lines in a system they are typically among
the tallest thin vertical objects in the image after staff removal.

Public API
----------
    detect_barlines_for_staff()  — barlines for one staff
    detect_all_barlines()        — all barlines grouped by staff
    extract()                    — pipeline entry point
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from orm import layers
from orm.constant import (
    BARLINE_MAX_WIDTH_PX,
    BARLINE_MERGE_TOL_PX,
    BARLINE_MIN_HEIGHT_RATIO,
)
from orm.logger import get_logger

logger = get_logger(__name__)

# Type alias for a barline: (x_center, y_top, y_bottom)
BarlineTuple = Tuple[int, int, int]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _staff_unit(staff_y: List[int]) -> float:
    sorted_y = sorted(staff_y)
    if len(sorted_y) >= 2:
        return float(np.median(np.diff(sorted_y)))
    return 10.0


def _vertical_open_run(
    bin_img: np.ndarray,
    min_height: int,
    max_width: int,
) -> List[Tuple[int, int, int, int]]:
    """Return a list of (x, y, w, h) blobs that look like vertical strokes.

    Filters by height >= *min_height* and width <= *max_width*.
    """
    # Emphasise vertical strokes with a tall open kernel
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(7, min_height // 3)))
    vert_map = cv2.morphologyEx(
        bin_img.astype(np.uint8), cv2.MORPH_OPEN, vert_kernel
    )
    # Bridge small horizontal gaps so a broken barline is one component
    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    vert_map = cv2.morphologyEx(vert_map, cv2.MORPH_CLOSE, bridge_kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (vert_map > 0).astype(np.uint8), connectivity=8
    )
    blobs: List[Tuple[int, int, int, int]] = []
    for i in range(1, n):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if h >= min_height and w <= max_width:
            blobs.append((x, y, w, h))
    return blobs


def _merge_barline_fragments(
    blobs: List[Tuple[int, int, int, int]],
    tol: int,
) -> List[BarlineTuple]:
    """Merge fragments that share the same x-column (within *tol* pixels).

    Returns list of (x_center, y_top, y_bottom).
    """
    if not blobs:
        return []
    blobs_sorted = sorted(blobs, key=lambda b: b[0])
    merged: List[List] = []  # each entry: [x_sum, count, y_top, y_bottom]
    for x, y, w, h in blobs_sorted:
        xc = x + w // 2
        if not merged or abs(xc - (merged[-1][0] // merged[-1][1])) > tol:
            merged.append([xc, 1, y, y + h])
        else:
            last = merged[-1]
            last[0] += xc
            last[1] += 1
            last[2] = min(last[2], y)
            last[3] = max(last[3], y + h)
    result: List[BarlineTuple] = []
    for s, cnt, y_top, y_bot in merged:
        result.append((s // cnt, y_top, y_bot))
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_barlines_for_staff(
    bin_img: np.ndarray,
    staff_y: List[int],
    conf_thresh: float = 0.5,
) -> List[BarlineTuple]:
    """Detect barlines within the row-extent of a single staff system.

    Parameters
    ----------
    bin_img:
        Binary image (uint8, foreground = 255) after staff removal.
    staff_y:
        5 y-coordinates of this staff.

    Returns
    -------
    List of ``(x_center, y_top, y_bottom)`` tuples, sorted by x.
    """
    unit = _staff_unit(staff_y)
    sorted_y = sorted(staff_y)
    min_h = max(5, int(round(BARLINE_MIN_HEIGHT_RATIO * unit)))

    # Crop to the staff height + a small margin
    h_img = bin_img.shape[0]
    margin = int(round(1.0 * unit))
    y_min = max(0, sorted_y[0] - margin)
    y_max = min(h_img - 1, sorted_y[-1] + margin)
    strip = bin_img[y_min : y_max + 1, :]

    blobs = _vertical_open_run(strip, min_height=min_h, max_width=BARLINE_MAX_WIDTH_PX)

    # Translate y coords back to full-image space
    blobs_abs = [(x, y + y_min, w, h) for (x, y, w, h) in blobs]

    barlines = _merge_barline_fragments(blobs_abs, tol=BARLINE_MERGE_TOL_PX)
    barlines_sorted = sorted(barlines, key=lambda b: b[0])
    return barlines_sorted


def detect_all_barlines(
    bin_img: np.ndarray,
    staff_lines: List[List[int]],
) -> Dict[int, List[BarlineTuple]]:
    """Detect barlines for every staff system.

    Parameters
    ----------
    bin_img:
        Binary image (uint8, foreground = 255) after staff removal.
    staff_lines:
        All detected staff systems.

    Returns
    -------
    ``{ staff_idx: [BarlineTuple, ...], ... }``
    """
    result: Dict[int, List[BarlineTuple]] = {}
    for idx, staff_y in enumerate(staff_lines):
        barlines = detect_barlines_for_staff(bin_img, staff_y)
        result[idx] = barlines
        logger.info("Staff %d: detected %d barline(s)", idx, len(barlines))
    return result


def extract() -> Dict[int, List[BarlineTuple]]:
    """Detect barlines for all staff systems.

    Reads
    -----
    ``img_no_staff``   : binary image after staff removal
    ``staff_lines``    : list of staff systems

    Registers
    ---------
    ``barline_results`` : Dict[int, List[BarlineTuple]]

    Returns
    -------
    ``barline_results``
    """
    bin_img: np.ndarray = layers.get_layer("img_no_staff")
    staff_lines: List[List[int]] = layers.get_layer("staff_lines")

    barline_results = detect_all_barlines(bin_img, staff_lines)

    total = sum(len(v) for v in barline_results.values())
    logger.info("Barline detection complete: %d barline(s) across %d staff(s)",
                total, len(staff_lines))

    layers.register_layer("barline_results", barline_results)
    return barline_results
