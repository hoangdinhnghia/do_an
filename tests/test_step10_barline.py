"""
=============================================================
BƯỚC 10 — PHÁT HIỆN VẠCH NHỊP (BARLINE DETECTION)
=============================================================

Kiểm tra module orm.barline:
- detect_barlines_for_staff() — barline cho 1 staff
- detect_all_barlines()       — barline cho tất cả staff
- extract()                   — layer registry

Không cần model ONNX.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from orm import layers
from orm.barline import detect_barlines_for_staff, detect_all_barlines, extract as barline_extract


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STAFF_Y = [80, 95, 110, 125, 140]   # unit = 15 px


def _make_barline_image(w=1200, h=250, bar_xs=None, staff_y=None):
    """Return (h, w) uint8 binary image with vertical barlines at *bar_xs*.

    Barlines span from the first to the last staff line plus a margin.
    """
    img = np.zeros((h, w), dtype=np.uint8)
    if staff_y is None:
        staff_y = STAFF_Y
    sorted_y = sorted(staff_y)
    y_top = max(0, sorted_y[0] - 5)
    y_bot = min(h - 1, sorted_y[-1] + 5)
    if bar_xs:
        for bx in bar_xs:
            cv2.line(img, (bx, y_top), (bx, y_bot), 255, 2)
    return img


# ---------------------------------------------------------------------------
# detect_barlines_for_staff
# ---------------------------------------------------------------------------

class TestDetectBarlineForStaff:

    def test_no_barlines_on_empty_image(self):
        img = np.zeros((250, 1200), dtype=np.uint8)
        result = detect_barlines_for_staff(img, STAFF_Y)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_detects_single_barline(self):
        img = _make_barline_image(bar_xs=[400])
        result = detect_barlines_for_staff(img, STAFF_Y)
        assert len(result) >= 1, "Should detect the single barline"

    def test_detects_multiple_barlines(self):
        img = _make_barline_image(bar_xs=[300, 600, 900])
        result = detect_barlines_for_staff(img, STAFF_Y)
        assert len(result) >= 2, f"Should detect at least 2 barlines, got {len(result)}"

    def test_barlines_sorted_by_x(self):
        """Barline tuples must be sorted left-to-right."""
        img = _make_barline_image(bar_xs=[600, 300, 900])
        result = detect_barlines_for_staff(img, STAFF_Y)
        xs = [r[0] for r in result]
        assert xs == sorted(xs), "Barlines must be sorted by x position"

    def test_barline_tuple_has_three_fields(self):
        """Each barline must be (x_center, y_top, y_bottom)."""
        img = _make_barline_image(bar_xs=[400])
        result = detect_barlines_for_staff(img, STAFF_Y)
        for bar in result:
            assert len(bar) == 3, f"Barline tuple must have 3 fields, got {len(bar)}: {bar}"

    def test_barline_y_span_covers_staff(self):
        """The y span of each detected barline must cover the staff lines."""
        img = _make_barline_image(bar_xs=[400])
        result = detect_barlines_for_staff(img, STAFF_Y)
        if result:
            bx, y_top, y_bot = result[0]
            assert y_top <= min(STAFF_Y), f"Barline top {y_top} should be at or above staff top {min(STAFF_Y)}"
            assert y_bot >= max(STAFF_Y), f"Barline bottom {y_bot} should be at or below staff bottom {max(STAFF_Y)}"

    def test_horizontal_lines_not_detected(self):
        """Horizontal lines (staff lines) must not be detected as barlines."""
        img = np.zeros((250, 1200), dtype=np.uint8)
        # Draw horizontal staff lines only
        for y in STAFF_Y:
            cv2.line(img, (0, y), (img.shape[1] - 1, y), 255, 2)
        result = detect_barlines_for_staff(img, STAFF_Y)
        # Staff lines are horizontal and very wide → should not pass barline filter
        # (they are removed by the height filter since their height ≪ barline_min_height)
        assert len(result) == 0, \
            f"Horizontal staff lines should not be detected as barlines, got {len(result)}"

    def test_short_blob_not_barline(self):
        """A very short vertical blob (< min_height) must not be detected."""
        img = np.zeros((250, 1200), dtype=np.uint8)
        # Draw a short blob spanning only 3 pixels vertically
        cv2.line(img, (400, 110), (400, 113), 255, 2)
        result = detect_barlines_for_staff(img, STAFF_Y)
        assert len(result) == 0, "Short blob should not be detected as barline"

    def test_x_position_is_close_to_drawn_barline(self):
        """Detected barline x must be within ±10 px of the drawn position."""
        drawn_x = 500
        img = _make_barline_image(bar_xs=[drawn_x])
        result = detect_barlines_for_staff(img, STAFF_Y)
        if result:
            detected_x = result[0][0]
            assert abs(detected_x - drawn_x) <= 10, \
                f"Detected x={detected_x} differs from drawn x={drawn_x} by more than 10 px"


# ---------------------------------------------------------------------------
# detect_all_barlines
# ---------------------------------------------------------------------------

class TestDetectAllBarlines:

    def test_returns_dict_keyed_by_staff_index(self):
        img = _make_barline_image(bar_xs=[400, 800])
        staff_lines = [STAFF_Y]
        result = detect_all_barlines(img, staff_lines)
        assert isinstance(result, dict)
        assert 0 in result

    def test_one_entry_per_staff(self):
        staff2 = [180, 195, 210, 225, 240]
        img = np.zeros((350, 1200), dtype=np.uint8)
        # Draw barlines spanning both staffs
        for bx in [300, 600]:
            cv2.line(img, (bx, 75), (bx, 245), 255, 2)
        staff_lines = [STAFF_Y, staff2]
        result = detect_all_barlines(img, staff_lines)
        assert len(result) == 2

    def test_empty_image_gives_empty_lists(self):
        img = np.zeros((350, 1200), dtype=np.uint8)
        result = detect_all_barlines(img, [STAFF_Y])
        assert result[0] == []


# ---------------------------------------------------------------------------
# Layer-registry integration
# ---------------------------------------------------------------------------

class TestBarlineExtract:

    def setup_method(self):
        layers.clear()

    def _populate_layers(self, bar_xs=None):
        img = _make_barline_image(bar_xs=bar_xs or [])
        layers.register_layer("img_no_staff", img)
        layers.register_layer("staff_lines", [STAFF_Y])

    def test_extract_registers_barline_results(self):
        self._populate_layers(bar_xs=[400, 800])
        barline_extract()
        assert "barline_results" in layers.list_layers()

    def test_extract_returns_dict(self):
        self._populate_layers(bar_xs=[400, 800])
        result = barline_extract()
        assert isinstance(result, dict)

    def test_extract_detects_barlines(self):
        self._populate_layers(bar_xs=[400, 800])
        result = barline_extract()
        total = sum(len(v) for v in result.values())
        assert total >= 1, f"Expected at least 1 barline, got {total}"

    def test_extract_empty_image_no_barlines(self):
        self._populate_layers(bar_xs=[])
        result = barline_extract()
        total = sum(len(v) for v in result.values())
        assert total == 0
