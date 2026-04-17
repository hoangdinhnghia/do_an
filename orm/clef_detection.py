"""Clef detection module.

Determines the clef type for each staff system by analysing the leftmost
region of the semantic probability map (channel 3 = "other symbol") where
the clef glyph is always found.

Algorithm
---------
1. Crop the left ``CLEF_REGION_RATIO`` fraction of the staff strip from the
   semantic map.
2. Binarise the "other symbol" channel at ``SYMBOL_CONF_THRESH``.
3. Find the largest connected component in that region.
4. Classify by the vertical centre of mass relative to the staff span:
   - Treble clef: CoM tends to sit in the **upper third** of the staff
     (the spiral wraps around the G4 line which is line 4 from the top)
   - Bass clef:  CoM sits near the **upper quarter** but with a much shorter
     blob
   - If no confident glyph is found, fall back to CLEF_UNKNOWN → treated as
     treble everywhere downstream.

Public API
----------
    detect_clef_for_staff()  — detect clef for one staff strip
    extract()                — detect clef for all staves, update layer registry
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from orm import layers
from orm.constant import (
    CLEF_ALTO,
    CLEF_BASS,
    CLEF_REGION_RATIO,
    CLEF_TENOR,
    CLEF_TREBLE,
    CLEF_UNKNOWN,
    M2_CH_SYMBOL,
    SYMBOL_CONF_THRESH,
)
from orm.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _crop_staff_strip(
    semantic_prob: np.ndarray,
    staff_y: List[int],
    expand: int = 0,
) -> Tuple[np.ndarray, int, int]:
    """Return (strip, y_min, y_max) for *one* staff system.

    ``expand`` adds extra rows above/below the outermost staff lines so that
    glyphs like the treble-clef curl that extends below the staff are included.
    """
    h = semantic_prob.shape[0]
    sorted_y = sorted(staff_y)
    unit = float(np.median(np.diff(sorted_y))) if len(sorted_y) >= 2 else 10.0
    pad = max(expand, int(round(1.5 * unit)))
    y_min = max(0, sorted_y[0] - pad)
    y_max = min(h - 1, sorted_y[-1] + pad)
    strip = semantic_prob[y_min : y_max + 1, :, :]
    return strip, y_min, y_max


def _largest_component_bbox(
    bin_mask: np.ndarray,
) -> Optional[Tuple[int, int, int, int]]:
    """Return (x, y, w, h) of the largest connected component, or None."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        bin_mask.astype(np.uint8), connectivity=8
    )
    if n <= 1:
        return None
    # skip background (label 0)
    largest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    x = int(stats[largest, cv2.CC_STAT_LEFT])
    y = int(stats[largest, cv2.CC_STAT_TOP])
    w = int(stats[largest, cv2.CC_STAT_WIDTH])
    h = int(stats[largest, cv2.CC_STAT_HEIGHT])
    return x, y, w, h


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def detect_clef_for_staff(
    semantic_prob: np.ndarray,
    staff_y: List[int],
    conf_thresh: float = SYMBOL_CONF_THRESH,
    clef_region_ratio: float = CLEF_REGION_RATIO,
) -> str:
    """Detect the clef type for a single staff system.

    Parameters
    ----------
    semantic_prob:
        Full-image (H, W, 4) float32 probability map from the semantic model.
        Channel 3 is the "other symbol" channel that contains clef glyphs.
    staff_y:
        5 y-coordinates of the staff lines (unsorted OK).
    conf_thresh:
        Binarisation threshold for the symbol channel.
    clef_region_ratio:
        Fraction of image width to examine (left side only).

    Returns
    -------
    One of ``CLEF_TREBLE``, ``CLEF_BASS``, ``CLEF_ALTO``, ``CLEF_TENOR``,
    or ``CLEF_UNKNOWN`` when confidence is too low.
    """
    sorted_y = sorted(staff_y)
    staff_span = sorted_y[-1] - sorted_y[0]
    unit = float(np.median(np.diff(sorted_y))) if len(sorted_y) >= 2 else 10.0

    # --- 1. Crop the staff strip from the full image ---
    strip, y_min, _ = _crop_staff_strip(semantic_prob, sorted_y, expand=int(2 * unit))
    if strip.shape[0] == 0 or strip.shape[1] == 0:
        return CLEF_UNKNOWN

    # --- 2. Restrict to the left clef region ---
    clef_w = max(10, int(round(strip.shape[1] * clef_region_ratio)))
    clef_strip = strip[:, :clef_w, :]

    # --- 3. Binarise the "other symbol" channel ---
    sym_ch = clef_strip[:, :, M2_CH_SYMBOL]
    bin_mask = (sym_ch >= conf_thresh).astype(np.uint8) * 255

    # Morphological closing to unite nearby fragments of the same glyph
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)

    # --- 4. Find the largest blob in the clef region ---
    bbox = _largest_component_bbox(bin_mask)
    if bbox is None:
        logger.debug("Clef detection: no symbol blob found — defaulting to treble")
        return CLEF_UNKNOWN

    _, blob_y, _, blob_h = bbox

    # --- 5. Classify by blob geometry ---
    # Translate blob_y into the coordinate system of the full staff strip
    # (blob_y is relative to clef_strip which is already a cropped strip).
    # We want: where does the blob sit relative to the staff lines?

    # Map blob centre (in strip coords) back to absolute image coords
    blob_cy_abs = y_min + blob_y + blob_h // 2

    # Relative position of blob centre within the staff (0 = top line, 1 = bottom line)
    rel_pos = (blob_cy_abs - sorted_y[0]) / max(1.0, float(staff_span))

    # Height of blob relative to staff span
    rel_h = blob_h / max(1.0, float(strip.shape[0]))

    # --- Classification heuristics (calibrated on common scores) ---
    # Treble: large blob (tall glyph), CoM near lines 2–4 (rel_pos ≈ 0.35–0.70)
    # Bass:   smaller blob (dot + two-arc), CoM near lines 1–2 (rel_pos ≈ 0.0–0.35)
    # Alto/Tenor: medium blob centred on middle line

    if rel_h >= 0.55:
        # Tall glyph → almost certainly treble
        return CLEF_TREBLE

    if rel_pos < 0.20:
        # Blob sits near the very top of the staff → bass clef dots region
        return CLEF_BASS

    if 0.20 <= rel_pos < 0.50 and rel_h < 0.40:
        # Shorter blob in upper half → bass clef body
        return CLEF_BASS

    if 0.40 <= rel_pos <= 0.65 and 0.30 <= rel_h < 0.60:
        # Medium blob centred on the middle or 4th line → alto/tenor
        # Distinguish by absolute position of centre line:
        # - Alto clef: CoM ≈ middle staff line (line 3)
        # - Tenor clef: CoM ≈ 4th line from bottom (line 4)
        middle_line_y = sorted_y[2]
        fourth_line_y = sorted_y[3]
        dist_middle = abs(blob_cy_abs - middle_line_y)
        dist_fourth = abs(blob_cy_abs - fourth_line_y)
        if dist_middle <= dist_fourth:
            return CLEF_ALTO
        return CLEF_TENOR

    if rel_pos >= 0.50:
        return CLEF_TREBLE   # tall curl often falls below mid-staff

    return CLEF_UNKNOWN


# ---------------------------------------------------------------------------
# Pipeline entry point (reads / writes layer registry)
# ---------------------------------------------------------------------------

def extract(
    conf_thresh: float = SYMBOL_CONF_THRESH,
    clef_region_ratio: float = CLEF_REGION_RATIO,
) -> List[str]:
    """Detect clef type for every staff system.

    Reads
    -----
    ``semantic_map``  : (H, W, 4) float32
    ``staff_lines``   : List[List[int]]

    Registers
    ---------
    ``clef_list``     : List[str]  — one clef type per staff

    Returns
    -------
    List of clef strings, one per staff.
    """
    semantic_prob: np.ndarray = layers.get_layer("semantic_map")
    staff_lines: List[List[int]] = layers.get_layer("staff_lines")

    clef_list: List[str] = []
    for i, staff_y in enumerate(staff_lines):
        clef = detect_clef_for_staff(
            semantic_prob,
            staff_y,
            conf_thresh=conf_thresh,
            clef_region_ratio=clef_region_ratio,
        )
        # When confidence is low fall back to treble (most common)
        if clef == CLEF_UNKNOWN:
            clef = CLEF_TREBLE
        clef_list.append(clef)
        logger.info("Staff %d: detected clef = %s", i, clef)

    layers.register_layer("clef_list", clef_list)
    return clef_list
