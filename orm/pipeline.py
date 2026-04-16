"""End-to-end Optical Music Recognition (OMR) pipeline.

Orchestrates the full recognition flow using the dual U-Net stream approach
inspired by the oemer project (https://github.com/BreezeWhite/oemer):

    1. Stream 1 (U-Net) — staffline segmentation → staff system y-coordinates.
    2. Stream 2 (U-Net) — detailed semantic segmentation → notehead detections.
    3. Assign noteheads to staves — map each notehead to its nearest staff using
       the staff y-coordinates from stream 1.
    4. Pitch assign — map each notehead to a pitch using staff geometry.

Usage
-----
    from orm.pipeline import OMRPipeline

    pipe = OMRPipeline()                   # loads ONNX models once
    result = pipe.run(img_bgr)             # numpy BGR image (uint8)

    for staff in result['staves']:
        print("Staff lines:", staff['staff_y'])
        for note in staff['notes']:
            print(note['pitch'], note['notehead'])

Result dict keys
----------------
    'staves'            : list of per-staff dicts (see below).
    'staff_lines'       : raw list[list[int]] — five y-coords per staff.
    'noteheads_global'  : all noteheads in image coordinates (x,y,w,h,cx,cy).
    'staff_prob_map'    : (H,W,3) float32 — stream-1 probability map.
    'semantic_map'      : (H,W,4) float32 — stream-2 probability map.
    'symbol_mask'       : (H,W) uint8 — combined foreground mask from stream 2.

Per-staff dict keys
-------------------
    'staff_index'   : int
    'staff_y'       : list[int] — five y-coords of staff lines.
    'clef'          : str — clef used for pitch assignment.
    'notes'         : list of note dicts.
    'staff_crop'    : (H',W',3) uint8 — crop of the original image for this staff.

Per-note dict keys
------------------
    'notehead'  : (x, y, w, h, cx, cy) in global image coordinates.
    'pitch'     : str   — e.g. "E4"
    'step'      : str   — note letter, e.g. "E"
    'octave'    : int   — octave number
    'position'  : int   — diatonic position from bottom staff line
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .model_inference import (
    DetailedSemanticModel,
    StafflineSegmentationModel,
    run_dual_pipeline,
)
from .pitch import Clef, assign_pitches_to_staff
from .staff_detection import crop_staffs

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default inference parameters
# ---------------------------------------------------------------------------
_DEFAULT_STAFF_THRESH = 0.30
_DEFAULT_NOTE_THRESH = 0.40
_DEFAULT_OVERLAP = 64
_DEFAULT_MAX_SIDE = 2048

# Half-height of a staff system (in units of median inter-line spacing) used
# when deciding which staff "owns" a notehead that falls between two staves.
_STAFF_OWNERSHIP_HALF_SPAN = 4.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assign_noteheads_to_staves(
    noteheads: List[Tuple[int, int, int, int, int, int]],
    staff_lines: List[List[int]],
) -> List[List[Tuple[int, int, int, int, int, int]]]:
    """Assign each global notehead to its nearest staff.

    For each notehead centroid cy we compute the vertical distance to the
    centre of every staff and assign the notehead to the closest one.  A note
    falls outside the staff "bounding box" if its cy is far enough from the
    staff centre; in that case it is still assigned to the nearest staff,
    which lets the pitch module handle ledger lines correctly.

    Args:
        noteheads   : List of (x, y, w, h, cx, cy) in image coordinates.
        staff_lines : List of staves; each staff is a list of 5 y-coords.

    Returns:
        List of the same length as *staff_lines*, where each element is the
        sub-list of noteheads assigned to that staff.
    """
    per_staff: List[List[Tuple[int, int, int, int, int, int]]] = [
        [] for _ in staff_lines
    ]
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
# OMRPipeline
# ---------------------------------------------------------------------------

class OMRPipeline:
    """End-to-end OMR pipeline backed by dual U-Net inference.

    Args:
        staffline_model: Pre-loaded :class:`~orm.model_inference.StafflineSegmentationModel`.
            Created automatically on first use if *None*.
        semantic_model:  Pre-loaded :class:`~orm.model_inference.DetailedSemanticModel`.
            Created automatically on first use if *None*.
        staff_conf_thresh: Confidence threshold for stream-1 staff-line pixels.
        note_conf_thresh:  Confidence threshold for stream-2 notehead pixels
                           (applied to the notehead channel of the semantic map).
        overlap: Tile-and-stitch overlap in pixels used by both models.
        max_side: Maximum image side length before auto-downscaling.
        default_clef: Clef assumed for every staff when no clef detector is
            available.  One of ``"treble"``, ``"bass"``, ``"alto"``, ``"tenor"``.
        staff_expand: Pixels to expand each side when cropping staff regions for
            the per-staff image in the result dict.
    """

    def __init__(
        self,
        staffline_model: Optional[StafflineSegmentationModel] = None,
        semantic_model: Optional[DetailedSemanticModel] = None,
        *,
        staff_conf_thresh: float = _DEFAULT_STAFF_THRESH,
        note_conf_thresh: float = _DEFAULT_NOTE_THRESH,
        overlap: int = _DEFAULT_OVERLAP,
        max_side: int = _DEFAULT_MAX_SIDE,
        default_clef: Clef = "treble",
        staff_expand: int = 20,
    ) -> None:
        self._staffline_model = staffline_model
        self._semantic_model = semantic_model
        self.staff_conf_thresh = staff_conf_thresh
        self.note_conf_thresh = note_conf_thresh
        self.overlap = overlap
        self.max_side = max_side
        self.default_clef = default_clef
        self.staff_expand = staff_expand

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------
    @property
    def staffline_model(self) -> StafflineSegmentationModel:
        if self._staffline_model is None:
            log.info("Loading StafflineSegmentationModel…")
            self._staffline_model = StafflineSegmentationModel()
        return self._staffline_model

    @property
    def semantic_model(self) -> DetailedSemanticModel:
        if self._semantic_model is None:
            log.info("Loading DetailedSemanticModel…")
            self._semantic_model = DetailedSemanticModel()
        return self._semantic_model

    # ------------------------------------------------------------------
    # Main pipeline entry point
    # ------------------------------------------------------------------
    def run(
        self,
        img_bgr: np.ndarray,
        clefs: Optional[List[Clef]] = None,
    ) -> Dict:
        """Run the full OMR pipeline on a BGR image.

        Args:
            img_bgr: Input image (uint8, any resolution).
            clefs: Optional list of clef strings, one per detected staff.
                   Defaults to :attr:`default_clef` for every staff.

        Returns:
            A structured result dict — see module docstring for key descriptions.
        """
        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("img_bgr is empty")

        # ----------------------------------------------------------------
        # Step 1 & 2: Dual U-Net inference (staffline + semantic)
        # ----------------------------------------------------------------
        log.debug("Running dual U-Net inference…")
        dual = run_dual_pipeline(
            img_bgr,
            staffline_model=self.staffline_model,
            semantic_model=self.semantic_model,
            staff_conf_thresh=self.staff_conf_thresh,
            note_conf_thresh=self.note_conf_thresh,
            overlap=self.overlap,
            max_side=self.max_side,
        )

        staff_lines: List[List[int]] = dual["staff_lines"]
        noteheads_global = dual["noteheads"]           # from stream-2, notehead channel
        staff_prob_map: np.ndarray = dual["staff_prob_map"]
        semantic_map: np.ndarray = dual["semantic_map"]
        symbol_mask: np.ndarray = dual["symbol_mask"]

        log.debug("Found %d staff system(s), %d notehead(s).",
                  len(staff_lines), len(noteheads_global))

        if not staff_lines:
            log.warning("No staff systems detected — returning empty result.")
            return self._empty_result(staff_prob_map, semantic_map, symbol_mask)

        # ----------------------------------------------------------------
        # Step 3: Assign noteheads to staves (by y-proximity)
        # ----------------------------------------------------------------
        per_staff_noteheads = _assign_noteheads_to_staves(noteheads_global, staff_lines)

        # Resolve clef list
        if clefs is None:
            clefs_resolved: List[Clef] = [self.default_clef] * len(staff_lines)
        else:
            if len(clefs) != len(staff_lines):
                raise ValueError(
                    f"len(clefs)={len(clefs)} must equal number of detected staves "
                    f"({len(staff_lines)})"
                )
            clefs_resolved = list(clefs)

        # Crop original image once per staff (for the result dict)
        crops = crop_staffs(img_bgr, staff_lines, expand=self.staff_expand)

        # ----------------------------------------------------------------
        # Step 4: Pitch assignment per staff
        # ----------------------------------------------------------------
        staves = []
        for idx, (staff_y, noteheads, crop_bgr, clef) in enumerate(
            zip(staff_lines, per_staff_noteheads, crops, clefs_resolved)
        ):
            notes = assign_pitches_to_staff(noteheads, staff_y, clef=clef)
            staves.append({
                "staff_index": idx,
                "staff_y": staff_y,
                "clef": clef,
                "notes": notes,
                "staff_crop": crop_bgr,
            })

        return {
            "staves": staves,
            "staff_lines": staff_lines,
            "noteheads_global": noteheads_global,
            "staff_prob_map": staff_prob_map,
            "semantic_map": semantic_map,
            "symbol_mask": symbol_mask,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _empty_result(
        staff_prob_map: np.ndarray,
        semantic_map: np.ndarray,
        symbol_mask: np.ndarray,
    ) -> Dict:
        return {
            "staves": [],
            "staff_lines": [],
            "noteheads_global": [],
            "staff_prob_map": staff_prob_map,
            "semantic_map": semantic_map,
            "symbol_mask": symbol_mask,
        }
