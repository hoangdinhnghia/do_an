"""
=============================================================
BƯỚC 5 — PHÁT HIỆN ĐẦU NỐT NHẠC (NOTEHEAD DETECTION)
=============================================================

Mục tiêu
--------
Từ **semantic map** (H×W×4 float32) của stream 2, trích xuất vị trí
bounding-box của từng đầu nốt nhạc, gán nó vào đúng staff system.

Hai nhánh xử lý
---------------
A. **Model-driven** (dùng ``orm.inference.SemanticModel``):
   ``semantic_map[:, :, 1]`` (kênh notehead) → binarise theo
   ``NOTE_CONF_THRESH`` → findContours → lọc theo area + aspect ratio
   → trả về list ``(x, y, w, h, cx, cy)``.

B. **Classic CV** (dùng ``orm.notehead_detection``):
   Ảnh binary đã xóa dòng kẻ → findContours → ``merge_nearby_bbox`` →
   ``rm_merge_overlap_bbox`` → crop staff-by-staff → annotate.
   Đây là luồng hiện tại của ``notehead_detection_pipeline``.

Cả hai nhánh đều được kiểm thử để dễ hoán đổi sau này.

Các hàm được kiểm thử
---------------------
- ``orm.notehead_detection.detect_notehead_contour``
- ``orm.notehead_detection.notehead_detection_pipeline``
- ``orm.model_inference.DetailedSemanticModel.extract_noteheads``
- Các hàm từ ``orm.staff_removal``
- ``orm.bbox.merge_nearby_bbox``
- ``orm.bbox.rm_merge_overlap_bbox``

Cách chạy
---------
::

    cd <repo_root>
    pytest tests/test_step5_notehead_detection.py -v
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from orm.bbox import merge_nearby_bbox, rm_merge_overlap_bbox, get_bbox
from orm.notehead_detection import detect_notehead_contour, notehead_detection_pipeline
from orm.staff_removal import staff_removal_pipeline
from orm.constant import NOTE_CONF_THRESH, M2_CH_NOTEHEAD


# ─────────────────────────────────────────────────────────────────────────────
# 5a. BBox utilities
# ─────────────────────────────────────────────────────────────────────────────

class TestBboxUtils:
    """Unit tests for bbox helper functions."""

    def test_get_bbox_finds_blobs(self):
        """get_bbox should find the bounding box of the single blob."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:30, 30:45] = 255
        bboxes = get_bbox(mask)
        assert len(bboxes) >= 1
        x1, y1, x2, y2 = bboxes[0]
        assert x1 <= 30 and x2 >= 44
        assert y1 <= 20 and y2 >= 29

    def test_merge_nearby_bbox_reduces_count(self):
        """Two boxes whose centres are closer than ``distance`` should be merged.

        Centers: (20, 20) and (42, 20) → Euclidean distance ≈ 22 px.
        Setting distance=30 makes them merge into one.
        """
        boxes = [(10, 10, 30, 30), (32, 10, 52, 30)]   # centres 22 px apart
        merged = merge_nearby_bbox(boxes, distance=30)
        assert len(merged) < len(boxes), "Boxes within distance threshold should be merged"

    def test_merge_distant_bbox_keeps_separate(self):
        """Boxes far apart should remain separate."""
        boxes = [(10, 10, 30, 30), (200, 200, 220, 220)]   # far apart
        merged = merge_nearby_bbox(boxes, distance=10)
        assert len(merged) == 2, "Distant boxes should remain separate"

    def test_rm_merge_overlap_removes_small_inside_large(self):
        """A small box fully inside a large box should be removed in 'remove' mode."""
        large = (0, 0, 100, 100)
        small = (20, 20, 40, 40)
        result = rm_merge_overlap_bbox([large, small], mode="remove", overlap_ratio=0.5)
        # small should be removed (it's fully contained)
        assert len(result) == 1

    def test_rm_merge_overlap_merge_mode(self):
        """Overlapping boxes whose overlap ratio exceeds the threshold should be merged.

        b1=(0,0,60,60), b2=(20,20,80,80).
        Intersection is [20,60]×[20,60] = 40×40 = 1600 px.
        b2 area = 60×60 = 3600.  overlap_ratio = 1600/3600 ≈ 0.44 > 0.3 → merge.
        """
        b1 = (0, 0, 60, 60)
        b2 = (20, 20, 80, 80)   # 44% of b2 overlaps b1
        result = rm_merge_overlap_bbox([b1, b2], mode="merge", overlap_ratio=0.3)
        assert len(result) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 5b. staff_removal_pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestStaffRemoval:
    """Unit tests for the staff-line removal step."""

    def test_removes_staff_lines(self):
        """After removal, pixels at staff-line rows should be ≈ 0."""
        h, w = 200, 400
        binary = np.zeros((h, w), dtype=np.uint8)
        staff_lines = [[50, 60, 70, 80, 90]]
        for y in staff_lines[0]:
            binary[y, :] = 255   # horizontal staff line

        result = staff_removal_pipeline(binary, staff_lines)
        assert result is not None
        assert result.shape == binary.shape

        # The average intensity at staff rows should be lower than original
        for y in staff_lines[0]:
            orig_density = binary[y, :].mean()
            res_density  = result[y, :].mean()
            assert res_density <= orig_density + 5.0, \
                f"Row y={y}: staff not removed (orig={orig_density:.1f}, after={res_density:.1f})"

    def test_preserves_noteheads(self, synthetic_binary_no_staff):
        """Noteheads far from staff lines must survive removal."""
        staff_lines = [[80, 90, 100, 110, 120], [200, 210, 220, 230, 240]]
        # Notehead at (300, 85) — right on staff region, but thick
        result = staff_removal_pipeline(synthetic_binary_no_staff, staff_lines)
        assert result is not None

    def test_returns_same_shape(self):
        binary = np.zeros((100, 200), dtype=np.uint8)
        result = staff_removal_pipeline(binary, [])
        assert result.shape == binary.shape


# ─────────────────────────────────────────────────────────────────────────────
# 5c. detect_notehead_contour
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectNoteheadContour:
    """Unit tests for the classic-CV notehead detector.

    ``detect_notehead_contour`` applies strict shape filters (circularity,
    solidity, aspect ratio) and requires either an attached stem *or* a very
    round blob (whole-note criteria).

    Input format: **grayscale binary image with symbols = 255 (foreground)
    and background = 0**.  This matches the output of ``staff_removal_pipeline``.

    Stem placement: the stem must be just outside the bounding box of the
    notehead so that ``_has_attached_stem`` finds it in the search band.
    """

    # Staff lines spaced 15 px apart → unit = 15
    _STAFF_Y = [70, 85, 100, 115, 130]

    def _make_notehead_image(self, positions, w=1500, h=250):
        """Return a (h, w) uint8 binary image (foreground=255, background=0).

        Each position is (cx, cy).  Ellipse axes are (7, 5).
        A 30-px vertical stem is placed 12 px right of the centre so it falls
        safely outside the bounding box and inside the right search band of
        ``_has_attached_stem`` for both unit≈15 and unit≈20.
        """
        img = np.zeros((h, w), dtype=np.uint8)
        for cx, cy in positions:
            cv2.ellipse(img, (cx, cy), (7, 5), 0, 0, 360, 255, -1)
            # stem_x = cx+12 is outside bbox edge (≈ cx+8..10) and inside band
            stem_x = cx + 12
            cv2.line(img, (stem_x, cy - 15), (stem_x, cy + 15), 255, 1)
        return img

    def test_detects_single_notehead(self):
        """Single notehead + stem must be detected."""
        img = self._make_notehead_image([(500, 100)])
        noteheads = detect_notehead_contour(img, staff_y=self._STAFF_Y)
        assert len(noteheads) >= 1, "Should detect at least 1 notehead"

    def test_detects_multiple_noteheads(self):
        """Three well-separated noteheads + stems should all be detected."""
        img = self._make_notehead_image([(400, 100), (700, 100), (1000, 100)])
        noteheads = detect_notehead_contour(img, staff_y=self._STAFF_Y)
        assert len(noteheads) >= 2, \
            f"Should detect multiple noteheads, got {len(noteheads)}"

    def test_notehead_bbox_has_six_fields(self):
        """Each result tuple must contain exactly 6 integers: (x, y, w, h, cx, cy)."""
        img = self._make_notehead_image([(500, 100)])
        noteheads = detect_notehead_contour(img, staff_y=self._STAFF_Y)
        assert len(noteheads) >= 1
        assert len(noteheads[0]) == 6, "NoteheadBBox must have 6 fields: (x, y, w, h, cx, cy)"

    def test_center_coords_inside_bbox(self):
        """The center (cx, cy) must lie within the bounding box (x..x+w, y..y+h)."""
        img = self._make_notehead_image([(500, 100)])
        noteheads = detect_notehead_contour(img, staff_y=self._STAFF_Y)
        for (x, y, w, h, cx, cy) in noteheads:
            assert x <= cx <= x + w, f"cx={cx} outside bbox x=[{x},{x+w}]"
            assert y <= cy <= y + h, f"cy={cy} outside bbox y=[{y},{y+h}]"

    def test_empty_image_returns_no_noteheads(self):
        img = np.zeros((100, 1500), dtype=np.uint8)
        noteheads = detect_notehead_contour(img)
        assert len(noteheads) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 5d. Notehead extraction from semantic probability map
# ─────────────────────────────────────────────────────────────────────────────

class TestNoteheadFromSemanticMap:
    """Unit tests for extracting noteheads from a semantic probability map."""

    def _extract_from_semantic_map(self, semantic_map: np.ndarray,
                                    conf_thresh: float = NOTE_CONF_THRESH):
        """Standalone helper that replicates DetailedSemanticModel.extract_noteheads."""
        mask = (semantic_map[:, :, M2_CH_NOTEHEAD] >= conf_thresh).astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 12 or area > 2000:
                continue
            x, y, w, h = cv2.boundingRect(c)
            ar = w / h if h else 0.0
            if ar < 0.35 or ar > 2.0:
                continue
            cx, cy = x + w // 2, y + h // 2
            results.append((x, y, w, h, cx, cy))
        return results

    def test_detects_four_noteheads_from_synthetic_map(self, synthetic_semantic_map):
        notes = self._extract_from_semantic_map(synthetic_semantic_map, conf_thresh=0.5)
        assert len(notes) == 4, \
            f"Expected 4 noteheads from synthetic map, got {len(notes)}"

    def test_center_near_expected_positions(self, synthetic_semantic_map):
        notes = self._extract_from_semantic_map(synthetic_semantic_map, conf_thresh=0.5)
        expected_cx = {300, 400, 500, 700}
        detected_cx = {n[4] for n in notes}
        for exp_x in expected_cx:
            nearest = min(detected_cx, key=lambda x: abs(x - exp_x))
            assert abs(nearest - exp_x) <= 10, \
                f"No notehead near expected cx={exp_x}"


# ─────────────────────────────────────────────────────────────────────────────
# 5e. notehead_detection_pipeline  (full integration)
# ─────────────────────────────────────────────────────────────────────────────

class TestNoteheadDetectionPipeline:
    """Tests for the staff-aware notehead detection pipeline."""

    def test_returns_one_result_per_staff(self, synthetic_binary_no_staff,
                                           two_staff_lines):
        """Pipeline must return one entry per staff system."""
        results = notehead_detection_pipeline(
            synthetic_binary_no_staff, two_staff_lines, expand=20
        )
        assert len(results) == len(two_staff_lines), \
            f"Expected {len(two_staff_lines)} results, got {len(results)}"

    def test_result_tuple_has_four_fields(self, synthetic_binary_no_staff,
                                           two_staff_lines):
        results = notehead_detection_pipeline(
            synthetic_binary_no_staff, two_staff_lines, expand=20
        )
        for r in results:
            assert len(r) == 4, "Each result must be (staff_idx, staff_y, noteheads, annotated)"

    def test_annotated_image_is_color(self, synthetic_binary_no_staff,
                                       two_staff_lines):
        """The annotated crop image must be a 3-channel BGR image."""
        results = notehead_detection_pipeline(
            synthetic_binary_no_staff, two_staff_lines, expand=20
        )
        for _, _, _, annotated in results:
            assert annotated.ndim == 3
            assert annotated.shape[2] == 3

    def test_detects_noteheads_in_synthetic_image(self):
        """Pipeline must find at least one notehead in a realistic synthetic image.

        We build a 1500×400 binary image (uint8, foreground=255) with:
        - Two staff systems (spacing 20 px)
        - 2 noteheads per staff: filled ellipses with adjacent stems
        The image is wide enough to pass the 260-px left-margin filter.
        """
        h, w = 400, 1500
        img = np.zeros((h, w), dtype=np.uint8)

        staff_lines = [
            [60, 80, 100, 120, 140],    # staff 1 — spacing 20 px → unit 20
            [220, 240, 260, 280, 300],  # staff 2
        ]

        def _add_notehead(img, cx, cy):
            cv2.ellipse(img, (cx, cy), (8, 6), 0, 0, 360, 255, -1)
            # stem_x = cx+12 ensures it falls outside bbox and inside right search band
            stem_x = cx + 12
            cv2.line(img, (stem_x, cy - 12), (stem_x, cy + 12), 255, 1)

        # Staff 1 noteheads at cx=500 and cx=800
        for cx in [500, 800]:
            _add_notehead(img, cx, 100)   # on staff 1 area

        # Staff 2 noteheads at cx=500 and cx=800
        for cx in [500, 800]:
            _add_notehead(img, cx, 260)   # on staff 2 area

        results = notehead_detection_pipeline(img, staff_lines, expand=25)
        total = sum(len(r[2]) for r in results)
        assert total >= 1, \
            f"Expected at least 1 notehead in realistic synthetic image, got {total}"


# ─────────────────────────────────────────────────────────────────────────────
# 5f. Integration — DetailedSemanticModel.extract_noteheads (needs real model)
# ─────────────────────────────────────────────────────────────────────────────

class TestDetailedSemanticExtractNoteheads:
    """Integration test: extract noteheads from a real image via SemanticModel."""

    def test_finds_noteheads_on_real_image(self, semantic_model, real_image_path):
        """On a real sheet-music image the model must find at least 1 notehead."""
        import cv2
        from orm.model_inference import DetailedSemanticModel
        img = cv2.imread(str(real_image_path))
        assert img is not None

        # Use the model_inference wrapper (DetailedSemanticModel) which has
        # extract_noteheads() method
        model = DetailedSemanticModel()
        prob = model.predict_full(img, max_side=1024)
        noteheads = model.extract_noteheads(prob, conf_thresh=NOTE_CONF_THRESH)
        assert len(noteheads) > 0, \
            f"Expected at least 1 notehead on {real_image_path.name}, got 0"
