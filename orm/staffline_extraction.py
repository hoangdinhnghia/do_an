"""Staffline extraction step.

Reads the staffline probability map from the layer registry, detects staff
lines using the projection-based algorithm, and registers the result.

This module mirrors the structure of oemer's ``staffline_extraction.py``:
each pipeline step exposes an ``extract()`` function that reads its inputs
from :mod:`orm.layers` and writes its outputs back to the registry.

Public API
----------
    extract(conf_thresh, smooth_sigma, peak_thresh) -> List[List[int]]
"""
from typing import List, Optional

import numpy as np

from orm import layers
from orm.constant import M1_CH_STAFF, STAFF_CONF_THRESH
from orm.exceptions import StafflineNotDetected
from orm.logger import get_logger
from orm.staff_detection import (
    find_peaks_profile,
    group_peaks_to_staffs,
    refine_staff_lines,
)

logger = get_logger(__name__)


def extract(
    conf_thresh: float = STAFF_CONF_THRESH,
    smooth_sigma: float = 2.0,
    peak_thresh: float = 0.5,
) -> List[List[int]]:
    """Detect staff lines from the registered staffline probability map.

    Reads
    -----
    ``staff_prob_map`` : (H, W, 3) float32 — registered by the inference step.

    Registers
    ---------
    ``staff_lines`` : List[List[int]] — each inner list contains the 5 y-coords
                      of one staff system.

    Returns
    -------
    List of staff systems, each a list of 5 y-coordinates.

    Raises
    ------
    StafflineNotDetected
        When no staff lines can be found with the given thresholds.
    """
    prob_map: np.ndarray = layers.get_layer("staff_prob_map")

    # Binarise the staff channel
    staff_mask = (prob_map[:, :, M1_CH_STAFF] >= conf_thresh).astype(np.uint8)

    # Horizontal projection → find y-peaks
    profile = staff_mask.sum(axis=1).astype(float)
    ys = find_peaks_profile(profile, smooth_sigma=smooth_sigma, thresh_ratio=peak_thresh)
    logger.info("Found %d raw staff-line peaks", len(ys))

    # Group peaks into staffs (5 lines each) and refine positions
    staffs = group_peaks_to_staffs(ys)
    staffs = refine_staff_lines(staffs, staff_mask * 255)

    if not staffs:
        raise StafflineNotDetected(
            "No staff lines detected. Try lowering conf_thresh or peak_thresh."
        )

    logger.info("Extracted %d staff system(s)", len(staffs))
    layers.register_layer("staff_lines", staffs)
    return staffs


def get_staff_unit(staff_lines: Optional[List[List[int]]] = None) -> float:
    """Return the average spacing (in pixels) between adjacent lines of the first staff.

    Uses the registered ``staff_lines`` layer when *staff_lines* is *None*.
    Falls back to 10 px when no staffs are available.
    """
    if staff_lines is None:
        try:
            staff_lines = layers.get_layer("staff_lines")
        except KeyError:
            return 10.0

    if not staff_lines:
        return 10.0

    first = sorted(staff_lines[0])
    if len(first) < 2:
        return 10.0
    gaps = np.diff(first)
    return float(np.median(gaps))
