"""End-to-end Optical Music Recognition (OMR) pipeline.

Orchestrates the full recognition flow using the dual U-Net stream approach
inspired by the oemer project (https://github.com/BreezeWhite/oemer):

    1. Pre-process      — grayscale, adaptive binarise.
    2. Stream 1 (U-Net) — staffline segmentation → staff system y-coordinates.
    3. Stream 2 (U-Net) — detailed semantic segmentation → symbol probability map.
    4. Staff removal    — erase staff lines from the binary image.
    5. Notehead detect  — detect noteheads using the symbol mask + contour analysis.
    6. Pitch assign     — map each notehead to a pitch using staff geometry.

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
    'notehead'  : (x, y, w, h, cx, cy) in crop coordinates.
    'pitch'     : str   — e.g. "E4"
    'step'      : str   — note letter, e.g. "E"
    'octave'    : int   — octave number
    'position'  : int   — diatonic position from bottom staff line
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from .model_inference import (
    DetailedSemanticModel,
    StafflineSegmentationModel,
    run_dual_pipeline,
)
from .notehead_detection import detect_notehead_contour
from .pitch import Clef, assign_pitches_to_staff
from .preprocess import adaptive_binarize, preprocess_image
from .staff_detection import crop_staffs
from .staff_removal import staff_removal_pipeline

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default inference parameters
# ---------------------------------------------------------------------------
_DEFAULT_STAFF_THRESH = 0.30
_DEFAULT_NOTE_THRESH = 0.40
_DEFAULT_OVERLAP = 64
_DEFAULT_MAX_SIDE = 2048

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
        note_conf_thresh:  Confidence threshold for stream-2 notehead pixels.
        overlap: Tile-and-stitch overlap in pixels used by both models.
        max_side: Maximum image side length before auto-downscaling.
        default_clef: Clef assumed for every staff when no clef detector is
            available.  One of ``"treble"``, ``"bass"``, ``"alto"``, ``"tenor"``.
        notehead_min_area: Minimum contour area for notehead candidates.
        notehead_max_area: Maximum contour area for notehead candidates.
        staff_expand: Pixels to expand each side when cropping staff regions.
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
        notehead_min_area: int = 18,
        notehead_max_area: int = 1200,
        staff_expand: int = 20,
    ) -> None:
        self._staffline_model = staffline_model
        self._semantic_model = semantic_model
        self.staff_conf_thresh = staff_conf_thresh
        self.note_conf_thresh = note_conf_thresh
        self.overlap = overlap
        self.max_side = max_side
        self.default_clef = default_clef
        self.notehead_min_area = notehead_min_area
        self.notehead_max_area = notehead_max_area
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
        # Step 1 & 2: Dual U-Net inference
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
        staff_prob_map: np.ndarray = dual["staff_prob_map"]
        semantic_map: np.ndarray = dual["semantic_map"]
        symbol_mask: np.ndarray = dual["symbol_mask"]

        log.debug("Found %d staff system(s).", len(staff_lines))

        if not staff_lines:
            log.warning("No staff systems detected — returning empty result.")
            return self._empty_result(staff_prob_map, semantic_map, symbol_mask)

        # ----------------------------------------------------------------
        # Step 3: Binarise original image for staff removal & notehead detect
        # ----------------------------------------------------------------
        gray_norm = preprocess_image(img_bgr)             # float32 [0,1]
        img_bin = adaptive_binarize(gray_norm).astype(np.uint8)  # 0/1 uint8

        # ----------------------------------------------------------------
        # Step 4: Staff removal
        # ----------------------------------------------------------------
        log.debug("Removing staff lines…")
        img_no_staff = staff_removal_pipeline(img_bin, staff_lines)

        # ----------------------------------------------------------------
        # Step 5: Notehead detection per staff using symbol mask guidance
        # ----------------------------------------------------------------
        # Use the U-Net symbol_mask (from stream 2) as the primary input
        # for notehead detection — it's cleaner than the raw binary image.
        # Fall back to img_no_staff when the mask is empty.
        if symbol_mask is not None and symbol_mask.max() > 0:
            detect_src = symbol_mask
        else:
            detect_src = (img_no_staff * 255).astype(np.uint8)

        crops = crop_staffs(img_bgr, staff_lines, expand=self.staff_expand)
        crops_bin = crop_staffs(detect_src, staff_lines, expand=self.staff_expand)

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

        staves = []
        all_noteheads_global: List = []

        for idx, (staff_y, crop_bgr, crop_bin, clef) in enumerate(
            zip(staff_lines, crops, crops_bin, clefs_resolved)
        ):
            noteheads_local = detect_notehead_contour(
                crop_bin,
                staff_y=None,   # crop already bounded to this staff
                min_area=self.notehead_min_area,
                max_area=self.notehead_max_area,
            )

            # ----------------------------------------------------------------
            # Step 6: Pitch assignment
            # ----------------------------------------------------------------
            notes = assign_pitches_to_staff(noteheads_local, staff_y, clef=clef)

            # Convert notehead coords back to global image space
            top_y = max(0, min(staff_y) - self.staff_expand)
            for note in notes:
                nx, ny, nw, nh, ncx, ncy = note["notehead"]
                all_noteheads_global.append(
                    (nx, ny + top_y, nw, nh, ncx, ncy + top_y)
                )

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
            "noteheads_global": all_noteheads_global,
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
