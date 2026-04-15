"""Pitch assignment for detected noteheads.

Given the five staff-line y-coordinates of a staff system and the centroid of
a notehead, this module computes the Western musical pitch (e.g. "E4", "G#5")
by measuring how many diatonic steps the note lies above or below the bottom
staff line.

Supported clefs
---------------
treble  (G-clef)  — bottom line = E4
bass    (F-clef)  — bottom line = G2
alto    (C-clef)  — bottom line = F3
tenor   (C-clef)  — bottom line = D3

Usage example
-------------
    from orm.pitch import assign_pitch, assign_pitches_to_staff

    staff_y = [100, 115, 130, 145, 160]   # 5 sorted y-coords
    noteheads = [(x, y, w, h, cx, cy), ...]

    notes = assign_pitches_to_staff(noteheads, staff_y, clef='treble')
    # → [{'notehead': (...), 'pitch': 'E4', 'step': 'E', 'octave': 4, 'position': 0}, ...]
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
NoteheadBBox = Tuple[int, int, int, int, int, int]  # (x, y, w, h, cx, cy)

Clef = Literal["treble", "bass", "alto", "tenor"]

# ---------------------------------------------------------------------------
# Diatonic pitch sequence (no accidentals, ascending)
# ---------------------------------------------------------------------------
_DIATONIC = ["C", "D", "E", "F", "G", "A", "B"]

# ---------------------------------------------------------------------------
# Bottom-line pitch reference for each clef.
# "position 0" = bottom staff line (ys[0]).
# Each diatonic step upward increases the position by 1.
# ---------------------------------------------------------------------------
#   Treble clef  — bottom line (line 1) = E4 → diatonic index 2 (E), octave 4
#   Bass clef    — bottom line (line 1) = G2 → diatonic index 4 (G), octave 2
#   Alto clef    — bottom line (line 1) = F3 → diatonic index 5 (F), octave 3
#   Tenor clef   — bottom line (line 1) = D3 → diatonic index 1 (D), octave 3
_CLEF_BOTTOM: Dict[str, Tuple[int, int]] = {
    "treble": (2, 4),   # E4  — index in _DIATONIC, octave
    "bass":   (4, 2),   # G2
    "alto":   (3, 3),   # F3  (_DIATONIC[3] = "F")
    "tenor":  (1, 3),   # D3  (_DIATONIC[1] = "D")
}

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def unit_size(staff_y_lines: Sequence[int]) -> float:
    """Return the half-space unit used for pitch counting.

    In standard notation the distance between two adjacent lines is two
    diatonic steps ("one space").  Half that distance is therefore one
    diatonic step — the *unit size*.

    Args:
        staff_y_lines: Five sorted y-coordinates of a staff system.

    Returns:
        Average half-spacing in pixels (float > 0).
    """
    ys = sorted(staff_y_lines)
    if len(ys) < 2:
        return 1.0
    gaps = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
    return float(np.mean(gaps)) / 2.0


def position_from_bottom(note_cy: int, staff_y_lines: Sequence[int]) -> int:
    """Compute the diatonic position of a notehead relative to the bottom line.

    Position 0  = bottom staff line (line 1).
    Position 1  = first space (between lines 1 and 2).
    Position 2  = line 2.
    …
    Position 8  = top staff line (line 5).
    Ledger lines extend the range below (negative) and above (> 8).

    Args:
        note_cy: Vertical centroid of the notehead (pixels, y increases downward).
        staff_y_lines: Five sorted y-coordinates of the staff system.

    Returns:
        Integer position (can be negative for notes below the staff).
    """
    ys = sorted(staff_y_lines)
    u = unit_size(ys)
    if u <= 0:
        return 0
    # Positive direction in music is *upward*, but y increases downward in images.
    # delta_y = bottom_line_y - note_cy  (positive when note is above bottom line)
    delta_y = ys[0] - note_cy
    position = round(delta_y / u)
    return int(position)


def assign_pitch(
    note_cy: int,
    staff_y_lines: Sequence[int],
    clef: Clef = "treble",
    accidental: Optional[str] = None,
) -> Dict[str, object]:
    """Assign a Western pitch name to a single notehead.

    Args:
        note_cy: Vertical centroid (pixel y-coordinate) of the notehead.
        staff_y_lines: Five sorted y-coordinates of the staff system.
        clef: One of ``"treble"``, ``"bass"``, ``"alto"``, ``"tenor"``.
        accidental: Optional accidental to attach (``"#"``, ``"b"``, ``""``).
            If *None* the accidental is omitted from the pitch string.

    Returns:
        A dict with keys:
            ``'position'``  — integer diatonic position from bottom line.
            ``'step'``      — note letter ("C" … "B").
            ``'octave'``    — octave number.
            ``'pitch'``     — full pitch string (e.g. ``"G#4"``).
    """
    clef = clef.lower()  # type: ignore[assignment]
    if clef not in _CLEF_BOTTOM:
        raise ValueError(f"Unknown clef: {clef!r}. Choose from {list(_CLEF_BOTTOM)}")

    ref_index, ref_octave = _CLEF_BOTTOM[clef]
    pos = position_from_bottom(note_cy, staff_y_lines)

    # Walk up/down the diatonic scale from the reference
    abs_index = ref_index + pos  # absolute diatonic step from C0
    step_index = abs_index % 7
    octave_shift = abs_index // 7
    step = _DIATONIC[step_index]
    octave = ref_octave + octave_shift

    acc_str = accidental if accidental is not None else ""
    pitch = f"{step}{acc_str}{octave}"

    return {
        "position": pos,
        "step": step,
        "octave": octave,
        "pitch": pitch,
    }


def assign_pitches_to_staff(
    noteheads: List[NoteheadBBox],
    staff_y_lines: Sequence[int],
    clef: Clef = "treble",
) -> List[Dict[str, object]]:
    """Assign pitches to all noteheads detected on a single staff.

    Args:
        noteheads: List of ``(x, y, w, h, cx, cy)`` tuples (output of
            :func:`orm.notehead_detection.detect_notehead_contour` or the
            U-Net-based detector).
        staff_y_lines: Five sorted y-coordinates of the staff system.
        clef: Clef for this staff (default ``"treble"``).

    Returns:
        List of dicts, one per notehead, each containing:
            ``'notehead'`` — original bbox tuple.
            ``'position'`` — diatonic position from bottom line.
            ``'step'``     — note letter.
            ``'octave'``   — octave number.
            ``'pitch'``    — full pitch string (e.g. ``"E4"``).
        Sorted left-to-right by the notehead centroid x-coordinate.
    """
    results = []
    for nh in noteheads:
        _x, _y, _w, _h, cx, cy = nh
        info = assign_pitch(cy, staff_y_lines, clef=clef)
        results.append({
            "notehead": nh,
            "position": info["position"],
            "step": info["step"],
            "octave": info["octave"],
            "pitch": info["pitch"],
        })
    # Sort by x centroid for left-to-right reading order
    results.sort(key=lambda d: d["notehead"][4])
    return results


def assign_pitches_all_staffs(
    staff_noteheads: List[Tuple[List[int], List[NoteheadBBox]]],
    clefs: Optional[List[Clef]] = None,
) -> List[List[Dict[str, object]]]:
    """Assign pitches for all staves in a system.

    Args:
        staff_noteheads: List of ``(staff_y_lines, noteheads)`` tuples — one
            entry per detected staff.
        clefs: Optional list of clef strings (same length as *staff_noteheads*).
            Defaults to ``"treble"`` for every staff.

    Returns:
        List of pitch-info lists, one per staff (same order as input).
    """
    if clefs is None:
        clefs = ["treble"] * len(staff_noteheads)

    if len(clefs) != len(staff_noteheads):
        raise ValueError(
            f"Length of clefs ({len(clefs)}) must match staff_noteheads ({len(staff_noteheads)})"
        )

    return [
        assign_pitches_to_staff(noteheads, staff_y, clef=clef)
        for (staff_y, noteheads), clef in zip(staff_noteheads, clefs)
    ]
