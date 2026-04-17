"""Note duration detection module.

Classifies every detected notehead into a rhythmic duration (whole, half,
quarter, eighth, …) by combining two independent signals:

1. **Open / filled head** — an *open* (hollow) notehead is half or whole; a
   *filled* (black) notehead is quarter or shorter.

2. **Stem + flag/beam count** — presence of a stem rules out whole notes; the
   number of flags/beams attached to the stem determines the subdivision.

Both signals are derived from the *binary image after staff removal* and
optionally from the semantic-map *stem/beam channel* (channel 2) which gives
cleaner stem detections than pure morphology.

Algorithm
---------
Step 1 — Open / closed classification
    Erode the notehead blob with a small ellipse to remove the outer ring, then
    measure the ratio of foreground pixels inside the eroded region.  An open
    head has few interior pixels (< OPEN_HEAD_FILL_RATIO); a closed head has
    many (> CLOSED_HEAD_FILL_RATIO).

Step 2 — Stem detection
    Search narrow vertical bands on the left and right of the bounding box.
    A stem is confirmed when a continuous vertical run of at least
    ``STEM_MIN_LENGTH_RATIO × unit`` pixels is found.

Step 3 — Flag / beam count
    Once the stem x-column is identified, scan horizontally away from the stem
    at regular intervals along the stem.  Each short horizontal protrusion
    that is taller than ``FLAG_MIN_HEIGHT_RATIO × unit`` counts as one flag.

Step 4 — Duration synthesis
    open + no stem  → whole
    open + stem     → half
    closed + 0 flag → quarter
    closed + 1 flag → eighth
    closed + 2 flag → 16th
    closed + 3 flag → 32nd

Public API
----------
    classify_head_type()       — open vs. filled for a single notehead
    detect_stem()              — find stem column + direction
    count_flags()              — count flags/beams on a stem
    assign_duration()          — synthesize duration string
    assign_durations()         — bulk, for all notehead results
    extract()                  — pipeline entry point
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from orm import layers
from orm.constant import (
    CLOSED_HEAD_FILL_RATIO,
    DUR_EIGHTH,
    DUR_HALF,
    DUR_QUARTER,
    DUR_SIXTEENTH,
    DUR_32ND,
    DUR_WHOLE,
    FLAG_MIN_HEIGHT_RATIO,
    M2_CH_STEM,
    OPEN_HEAD_FILL_RATIO,
    STEM_MIN_LENGTH_RATIO,
    STEM_SEARCH_MARGIN_RATIO,
)
from orm.logger import get_logger

logger = get_logger(__name__)

# Type alias
NoteheadBBox = Tuple[int, int, int, int, int, int]


# ---------------------------------------------------------------------------
# Step 1 — Open / closed head classification
# ---------------------------------------------------------------------------

def classify_head_type(
    bin_img: np.ndarray,
    bbox: NoteheadBBox,
    unit: float,
) -> str:
    """Return ``'open'`` or ``'filled'`` for a single notehead bounding box.

    Strategy
    --------
    * Crop to the bounding box.
    * Sample the *inner* central region (inner 50% × 50%) to avoid the border
      ring which is present in both open and closed heads.
    * Compute the fill ratio of foreground pixels in that inner region.
    * Open heads have a hollow centre → low fill ratio.
    * Filled heads are solid → high fill ratio.

    Parameters
    ----------
    bin_img:
        Binary image (uint8, foreground = 255, background = 0).  Should be the
        output of staff_removal_pipeline — i.e. the notehead pixels are bright.
    bbox:
        ``(x, y, w, h, cx, cy)`` of the notehead.
    unit:
        Staff unit (inter-line spacing) in pixels.

    Returns
    -------
    ``'open'`` or ``'filled'``
    """
    x, y, w, h, cx, cy = bbox
    h_img, w_img = bin_img.shape[:2]

    # Clamp to image bounds
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_img, x + w), min(h_img, y + h)
    if x2 <= x1 or y2 <= y1:
        return "filled"   # degenerate box → assume filled

    crop = bin_img[y1:y2, x1:x2].astype(np.uint8)
    crop_h, crop_w = crop.shape

    # Inner central region (central 50% in each dimension)
    margin_x = max(1, int(crop_w * 0.25))
    margin_y = max(1, int(crop_h * 0.25))
    inner = crop[margin_y : crop_h - margin_y, margin_x : crop_w - margin_x]

    if inner.size == 0:
        # Very small crop — fall back to checking the full crop center pixel
        inner = crop

    total = inner.size
    filled_ratio = float(np.count_nonzero(inner)) / float(total)

    if filled_ratio < OPEN_HEAD_FILL_RATIO:
        return "open"
    if filled_ratio > CLOSED_HEAD_FILL_RATIO:
        return "filled"
    # Ambiguous — use aspect ratio heuristic: round open heads tend to be wider
    ar = w / max(1, h)
    return "open" if ar >= 1.0 else "filled"


# ---------------------------------------------------------------------------
# Step 2 — Stem detection
# ---------------------------------------------------------------------------

def detect_stem(
    bin_img: np.ndarray,
    bbox: NoteheadBBox,
    unit: float,
    semantic_stem_mask: Optional[np.ndarray] = None,
) -> Tuple[bool, str, int]:
    """Detect the stem of a notehead.

    Searches vertical bands to the left and right of the bounding box for a
    continuous dark run at least ``STEM_MIN_LENGTH_RATIO × unit`` pixels tall.

    Parameters
    ----------
    bin_img:
        Binary image (uint8, foreground = 255).
    bbox:
        ``(x, y, w, h, cx, cy)``.
    unit:
        Staff unit in pixels.
    semantic_stem_mask:
        Optional (H, W) uint8 mask from ``semantic_prob[:,:,M2_CH_STEM]``
        binarised at a suitable threshold.  When provided it is ANDed with the
        local search strip to reduce false positives.

    Returns
    -------
    (has_stem, direction, stem_x)
        ``has_stem``  — True if a stem was found
        ``direction`` — ``'up'`` (stem goes above notehead), ``'down'``, or ``''``
        ``stem_x``    — column of the stem, or ``-1``
    """
    x, y, w, h, cx, cy = bbox
    h_img, w_img = bin_img.shape[:2]
    min_stem_len = max(5, int(round(STEM_MIN_LENGTH_RATIO * unit)))
    margin = max(3, int(round(STEM_SEARCH_MARGIN_RATIO * unit)))

    # Search strips: narrow columns just outside the notehead box
    search_bands = [
        ("right", min(w_img, x + w), min(w_img, x + w + margin)),
        ("left",  max(0, x - margin), max(0, x)),
    ]

    # Extend search vertically: stem may go up to 3× staff units above/below
    vert_extend = max(h, int(round(3.0 * unit)))
    y0_search = max(0, y - vert_extend)
    y1_search = min(h_img, y + h + vert_extend)

    best: Tuple[bool, str, int] = (False, "", -1)
    best_len = min_stem_len - 1

    for side, xs, xe in search_bands:
        if xe <= xs:
            continue
        band = bin_img[y0_search:y1_search, xs:xe].copy()
        if semantic_stem_mask is not None:
            sm_band = semantic_stem_mask[y0_search:y1_search, xs:xe]
            band = cv2.bitwise_and(band, band, mask=(sm_band > 0).astype(np.uint8))

        for col_offset in range(xe - xs):
            col = band[:, col_offset]
            # Find the longest continuous run of foreground pixels
            run_len, run_top = _longest_run(col)
            if run_len > best_len:
                best_len = run_len
                abs_col = xs + col_offset
                # Direction: does the run go above or below the notehead centre?
                run_cy = y0_search + run_top + run_len // 2
                direction = "up" if run_cy < cy else "down"
                best = (True, direction, abs_col)

    return best


def _longest_run(arr: np.ndarray) -> Tuple[int, int]:
    """Return (length, start_index) of the longest positive run in *arr*."""
    best_len = 0
    best_start = 0
    cur_start = 0
    cur_len = 0
    for i, v in enumerate(arr):
        if v > 0:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
        else:
            cur_len = 0
    return best_len, best_start


# ---------------------------------------------------------------------------
# Step 3 — Flag / beam count
# ---------------------------------------------------------------------------

def count_flags(
    bin_img: np.ndarray,
    stem_x: int,
    stem_top: int,
    stem_bottom: int,
    unit: float,
    direction: str = "up",
) -> int:
    """Count the number of flags (or beams) attached to a stem.

    Scans horizontally outward from *stem_x* at evenly spaced positions along
    the free part of the stem.  Each horizontal protrusion taller than
    ``FLAG_MIN_HEIGHT_RATIO × unit`` counts as one flag.

    Parameters
    ----------
    bin_img:
        Binary image (uint8, foreground = 255).
    stem_x:
        Column of the stem in the image.
    stem_top, stem_bottom:
        Row range of the visible stem.
    unit:
        Staff unit in pixels.
    direction:
        ``'up'`` means the free (flagged) end is at *stem_top*;
        ``'down'`` means it is at *stem_bottom*.

    Returns
    -------
    Number of flags (0–3).
    """
    h_img, w_img = bin_img.shape[:2]
    min_flag_h = max(2, int(round(FLAG_MIN_HEIGHT_RATIO * unit)))
    flag_spacing = max(2, int(round(0.50 * unit)))

    # The free end of the stem is where flags attach
    if direction == "up":
        free_end = stem_top
        scan_ys = [max(0, free_end + i * flag_spacing) for i in range(5)]
    else:
        free_end = stem_bottom
        scan_ys = [min(h_img - 1, free_end - i * flag_spacing) for i in range(5)]

    # Search width on both sides of the stem for flag protrusions
    search_w = max(4, int(round(1.5 * unit)))
    x0 = max(0, stem_x - search_w)
    x1 = min(w_img, stem_x + search_w + 1)

    n_flags = 0
    for row in scan_ys:
        if row < 0 or row >= h_img:
            continue
        strip = bin_img[row : row + min_flag_h, x0:x1]
        if strip.shape[0] == 0 or strip.shape[1] == 0:
            continue
        # A flag shows up as a run of foreground pixels in this horizontal strip
        col_counts = (strip > 0).sum(axis=0)
        # Count columns that have at least 1 foreground pixel
        active_cols = int(np.sum(col_counts > 0))
        if active_cols >= max(3, int(round(0.3 * search_w))):
            n_flags += 1
        else:
            break  # flags are contiguous from the free end

    return min(n_flags, 3)  # cap at 3 (32nd note)


# ---------------------------------------------------------------------------
# Step 4 — Duration synthesis
# ---------------------------------------------------------------------------

def assign_duration(head_type: str, has_stem: bool, n_flags: int) -> str:
    """Synthesise a duration string from head type + stem/flag info.

    Parameters
    ----------
    head_type:  ``'open'`` or ``'filled'``
    has_stem:   whether a stem was detected
    n_flags:    number of flags/beams (0, 1, 2, 3)

    Returns
    -------
    One of the DUR_* constants defined in ``orm.constant``.
    """
    if head_type == "open":
        if has_stem:
            return DUR_HALF
        return DUR_WHOLE

    # Filled head
    flag_to_dur = {0: DUR_QUARTER, 1: DUR_EIGHTH, 2: DUR_SIXTEENTH, 3: DUR_32ND}
    return flag_to_dur.get(n_flags, DUR_QUARTER)


# ---------------------------------------------------------------------------
# Bulk assignment
# ---------------------------------------------------------------------------

def assign_durations(
    notehead_results: list,
    bin_img: np.ndarray,
    staff_lines: List[List[int]],
    semantic_prob: Optional[np.ndarray] = None,
    stem_conf_thresh: float = 0.40,
) -> list:
    """Assign duration to every notehead across all staff systems.

    Parameters
    ----------
    notehead_results:
        Output of ``notehead_detection_pipeline()`` (or ``assign_pitches()``).
        Expected shape per entry: ``(staff_idx, staff_y, noteheads, annotated, ...)``.
    bin_img:
        Binary image after staff removal (uint8, foreground = 255).
    staff_lines:
        Full list of staff systems (to derive staff unit per staff).
    semantic_prob:
        Optional (H, W, 4) float32 from the semantic model.  When provided the
        stem/beam channel is used to refine stem detection.
    stem_conf_thresh:
        Binarisation threshold for the stem channel.

    Returns
    -------
    ``duration_results`` — same structure as *notehead_results* with an
    appended element: ``List[str]`` of duration strings per staff.

    If the input entries already have a 5th element (pitch labels) they are
    preserved.
    """
    # Optionally binarise the semantic stem/beam channel
    stem_mask: Optional[np.ndarray] = None
    if semantic_prob is not None and semantic_prob.shape[2] > M2_CH_STEM:
        stem_mask = (semantic_prob[:, :, M2_CH_STEM] >= stem_conf_thresh).astype(
            np.uint8
        ) * 255

    duration_results = []
    for entry in notehead_results:
        idx = entry[0]
        staff_y: List[int] = entry[1]
        noteheads: List[NoteheadBBox] = entry[2]
        rest = entry[3:]   # may contain annotated image, pitch labels, ...

        # Staff unit for this system
        sorted_y = sorted(staff_y)
        unit = float(np.median(np.diff(sorted_y))) if len(sorted_y) >= 2 else 10.0

        dur_labels: List[str] = []
        for bbox in noteheads:
            x, y, w, h, cx, cy = bbox

            # 1. Head type
            head_type = classify_head_type(bin_img, bbox, unit)

            # 2. Stem
            has_stem, direction, stem_x = detect_stem(
                bin_img, bbox, unit, semantic_stem_mask=stem_mask
            )

            # 3. Flags — only when stem found
            n_flags = 0
            if has_stem and stem_x >= 0:
                # Rough stem extent: notehead height + vert_extend (same as detect_stem)
                vert_extend = max(h, int(round(3.0 * unit)))
                h_img = bin_img.shape[0]
                if direction == "up":
                    stem_top = max(0, cy - vert_extend)
                    stem_bottom = cy
                else:
                    stem_top = cy
                    stem_bottom = min(h_img - 1, cy + vert_extend)
                n_flags = count_flags(
                    bin_img, stem_x, stem_top, stem_bottom, unit, direction
                )

            # 4. Synthesise
            dur = assign_duration(head_type, has_stem, n_flags)
            dur_labels.append(dur)

        duration_results.append((idx, staff_y, noteheads, *rest, dur_labels))
        logger.debug("Staff %d: %d durations assigned", idx, len(dur_labels))

    return duration_results


# ---------------------------------------------------------------------------
# Pipeline entry point (reads / writes layer registry)
# ---------------------------------------------------------------------------

def extract(stem_conf_thresh: float = 0.40) -> list:
    """Assign durations to all detected noteheads.

    Reads
    -----
    ``pitch_results`` (preferred) or ``notehead_results``
    ``img_no_staff``
    ``staff_lines``
    ``semantic_map``  (optional, used for stem channel)

    Registers
    ---------
    ``duration_results``

    Returns
    -------
    ``duration_results`` list
    """
    # Prefer pitch_results (which already has pitch labels attached)
    try:
        notehead_results = layers.get_layer("pitch_results")
    except KeyError:
        notehead_results = layers.get_layer("notehead_results")
        if hasattr(notehead_results, "tolist"):
            notehead_results = list(notehead_results)

    bin_img: np.ndarray = layers.get_layer("img_no_staff")
    staff_lines: List[List[int]] = layers.get_layer("staff_lines")

    try:
        semantic_prob: Optional[np.ndarray] = layers.get_layer("semantic_map")
    except KeyError:
        semantic_prob = None

    duration_results = assign_durations(
        notehead_results,
        bin_img,
        staff_lines,
        semantic_prob=semantic_prob,
        stem_conf_thresh=stem_conf_thresh,
    )

    total = sum(len(r[2]) for r in duration_results)
    logger.info("Duration assignment complete: %d noteheads across %d staff(s)",
                total, len(duration_results))

    layers.register_layer("duration_results", duration_results)
    return duration_results
