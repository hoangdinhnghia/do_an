"""Pitch assignment module.

Maps each detected notehead to a musical pitch (e.g. "G4", "C#5") based on:
  1. The notehead's vertical centre (cy) in image coordinates.
  2. The 5 y-coordinates of the enclosing staff system.
  3. The clef type of that staff.

Algorithm
---------
Staff lines divide the image into alternating *lines* and *spaces*.  Each
line or space corresponds to exactly one diatonic pitch.  In a treble-clef
staff the bottom line is E4; moving upward (decreasing y) each step adds one
diatonic degree.

Steps
-----
1. Compute the staff *unit* = median inter-line spacing.
2. The half-unit (unit / 2) equals one diatonic step (line → space or vice versa).
3. Derive the number of half-steps the note is above the bottom staff line:
       steps = round((bottom_line_y - cy) / half_unit)
4. Look up the base pitch from the clef lookup table and add ``steps``.
5. Wrap pitch name across octave boundaries.

Public API
----------
    assign_pitch_to_notehead()  — single notehead
    assign_pitches()            — bulk, for the full list of notehead results
    extract()                   — reads layer registry, writes back
"""

import math
from typing import List, Optional, Tuple

import numpy as np

from orm import layers
from orm.constant import (
    CLEF_BOTTOM_LINE_PITCH,
    CLEF_TREBLE,
    CLEF_UNKNOWN,
    PITCH_NAMES,
)
from orm.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Diatonic arithmetic helpers
# ---------------------------------------------------------------------------

def _pitch_to_diatonic_index(name: str, octave: int) -> int:
    """Convert (name, octave) to a single integer diatonic index.

    C0 → 0, D0 → 1, …, B0 → 6, C1 → 7, …
    """
    return PITCH_NAMES.index(name) + octave * 7


def _diatonic_index_to_pitch(idx: int) -> Tuple[str, int]:
    """Convert a diatonic index back to (name, octave)."""
    octave, degree = divmod(idx, 7)
    return PITCH_NAMES[degree], octave


# ---------------------------------------------------------------------------
# Core pitch-assignment logic
# ---------------------------------------------------------------------------

def assign_pitch_to_notehead(
    cy: int,
    staff_y: List[int],
    clef: str = CLEF_TREBLE,
) -> Tuple[str, int, str]:
    """Return the musical pitch of a notehead at vertical position *cy*.

    Parameters
    ----------
    cy:
        Vertical centre of the notehead in image pixels.
    staff_y:
        5 y-coordinates of the staff lines (order does not matter).
    clef:
        Clef type string from ``orm.constant`` (CLEF_TREBLE, CLEF_BASS, …).

    Returns
    -------
    (name, octave, label)
        ``name``  — pitch letter, e.g. ``"G"``
        ``octave`` — integer octave number, e.g. ``4``
        ``label`` — combined string, e.g. ``"G4"``
    """
    sorted_y = sorted(staff_y)

    # Staff unit = median inter-line spacing
    if len(sorted_y) >= 2:
        unit = float(np.median(np.diff(sorted_y)))
    else:
        unit = 10.0

    half_unit = max(1.0, unit / 2.0)

    # Number of diatonic steps above the bottom staff line (positive = above)
    bottom_line_y = sorted_y[-1]
    steps = int(round((bottom_line_y - cy) / half_unit))

    # Base pitch at the bottom staff line for this clef
    clef_key = clef if clef in CLEF_BOTTOM_LINE_PITCH else CLEF_UNKNOWN
    base_name_idx, base_octave = CLEF_BOTTOM_LINE_PITCH[clef_key]
    base_diatonic = _pitch_to_diatonic_index(PITCH_NAMES[base_name_idx], base_octave)

    # Final diatonic index
    final_diatonic = base_diatonic + steps
    name, octave = _diatonic_index_to_pitch(final_diatonic)
    label = f"{name}{octave}"
    return name, octave, label


# ---------------------------------------------------------------------------
# Bulk assignment
# ---------------------------------------------------------------------------

def assign_pitches(
    notehead_results: list,
    clef_list: Optional[List[str]] = None,
) -> list:
    """Assign a pitch label to every notehead in *notehead_results*.

    Parameters
    ----------
    notehead_results:
        Output of ``notehead_detection_pipeline()`` — a list of
        ``(staff_idx, staff_y, noteheads, annotated)`` tuples.
    clef_list:
        One clef string per staff.  When *None* every staff defaults to treble.

    Returns
    -------
    ``pitch_results`` — same structure as ``notehead_results`` but with an
    extra 5th element per tuple: ``List[str]`` of pitch labels, one per
    notehead in that staff.

    Shape: ``List[(staff_idx, staff_y, noteheads, annotated, pitch_labels)]``
    """
    pitch_results = []
    for entry in notehead_results:
        idx, staff_y, noteheads, annotated = entry[:4]
        clef = (clef_list[idx] if clef_list and idx < len(clef_list) else CLEF_TREBLE)
        labels: List[str] = []
        for (x, y, w, h, cx, cy) in noteheads:
            _, _, label = assign_pitch_to_notehead(cy, staff_y, clef)
            labels.append(label)
        pitch_results.append((idx, staff_y, noteheads, annotated, labels))
        logger.debug("Staff %d: %d pitches assigned", idx, len(labels))
    return pitch_results


# ---------------------------------------------------------------------------
# Pipeline entry point (reads / writes layer registry)
# ---------------------------------------------------------------------------

def extract() -> list:
    """Assign pitches to all detected noteheads.

    Reads
    -----
    ``notehead_results`` : output of notehead_detection step
    ``clef_list``        : optional — falls back to treble if absent

    Registers
    ---------
    ``pitch_results``    : extended notehead result with pitch labels

    Returns
    -------
    ``pitch_results`` list
    """
    notehead_results = layers.get_layer("notehead_results")
    # notehead_results may be stored as numpy object array; convert to list
    if hasattr(notehead_results, "tolist"):
        notehead_results = list(notehead_results)

    try:
        clef_list: Optional[List[str]] = layers.get_layer("clef_list")
    except KeyError:
        clef_list = None
        logger.warning("clef_list not found in layer registry — defaulting to treble clef")

    pitch_results = assign_pitches(notehead_results, clef_list)

    total_notes = sum(len(r[2]) for r in pitch_results)
    logger.info("Pitch assignment complete: %d noteheads across %d staff(s)",
                total_notes, len(pitch_results))

    layers.register_layer("pitch_results", pitch_results)
    return pitch_results
