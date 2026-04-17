"""
=============================================================
BƯỚC 9 — PHÁT HIỆN KHÓA NHẠC (CLEF DETECTION)
=============================================================

Kiểm tra module orm.clef_detection:
- detect_clef_for_staff() — phân loại khóa cho 1 staff
- extract()               — đọc/ghi layer registry

Không cần model ONNX; dùng dữ liệu semantic map tổng hợp.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from orm import layers
from orm.clef_detection import detect_clef_for_staff, extract as clef_extract
from orm.constant import (
    CLEF_TREBLE,
    CLEF_BASS,
    CLEF_ALTO,
    CLEF_TENOR,
    CLEF_UNKNOWN,
    M2_CH_SYMBOL,
    SYMBOL_CONF_THRESH,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STAFF_Y_1 = [80, 92, 104, 116, 128]   # unit = 12
STAFF_Y_2 = [200, 215, 230, 245, 260]  # unit = 15


def _make_semantic_prob(h, w, staff_y, glyph_y_top, glyph_h, glyph_x_end=60, ch3_val=0.9):
    """Return (H, W, 4) float32 with a synthetic glyph in channel 3.

    The glyph is drawn as a vertical rectangle in the leftmost portion
    to simulate a clef symbol blob.
    """
    prob = np.zeros((h, w, 4), dtype=np.float32)
    prob[:, :, 0] = 1.0  # all background
    # Place glyph blob in symbol channel (ch 3)
    y0 = max(0, glyph_y_top)
    y1 = min(h, glyph_y_top + glyph_h)
    prob[y0:y1, 0:glyph_x_end, 3] = ch3_val
    # Also lower background in that region so sum stays ≤ 1
    prob[y0:y1, 0:glyph_x_end, 0] = 0.0
    return prob


# ---------------------------------------------------------------------------
# detect_clef_for_staff — basic interface
# ---------------------------------------------------------------------------

class TestDetectClefInterface:

    def test_returns_string(self):
        """Function must return a non-empty string."""
        h, w = 300, 1200
        prob = np.zeros((h, w, 4), dtype=np.float32)
        prob[:, :, 0] = 1.0
        result = detect_clef_for_staff(prob, STAFF_Y_1)
        assert isinstance(result, str) and len(result) > 0

    def test_returns_known_clef_or_unknown(self):
        """Result must be one of the defined clef constants."""
        h, w = 300, 1200
        prob = np.zeros((h, w, 4), dtype=np.float32)
        prob[:, :, 0] = 1.0
        result = detect_clef_for_staff(prob, STAFF_Y_1)
        valid = {CLEF_TREBLE, CLEF_BASS, CLEF_ALTO, CLEF_TENOR, CLEF_UNKNOWN}
        assert result in valid, f"Unexpected clef value: '{result}'"

    def test_no_symbol_returns_unknown(self):
        """When the semantic map has no symbol pixels, must return CLEF_UNKNOWN."""
        h, w = 300, 1200
        prob = np.zeros((h, w, 4), dtype=np.float32)
        prob[:, :, 0] = 1.0   # pure background
        result = detect_clef_for_staff(prob, STAFF_Y_1)
        assert result == CLEF_UNKNOWN

    def test_tall_blob_detects_treble(self):
        """A tall glyph spanning > 55% of the strip height → treble."""
        h, w = 300, 1200
        # staff unit ≈ 12, strip height ≈ 12*3 (expansion) + staff span ≈ 72 px → ~120 px
        # A blob from row 50 to row 170 is ~120 px / 120 px = 100% of strip → treble
        sorted_y = sorted(STAFF_Y_1)
        glyph_top = sorted_y[0] - 30
        glyph_h = (sorted_y[-1] - sorted_y[0]) + 60  # spans full staff + margin
        prob = _make_semantic_prob(h, w, STAFF_Y_1, glyph_top, glyph_h)
        result = detect_clef_for_staff(prob, STAFF_Y_1)
        assert result == CLEF_TREBLE, f"Tall blob should give treble, got '{result}'"

    def test_small_blob_near_top_detects_bass(self):
        """A small blob near the very top of the staff → bass clef."""
        h, w = 300, 1200
        sorted_y = sorted(STAFF_Y_1)
        unit = float(np.median(np.diff(sorted_y)))
        # Bass clef dots sit near the top of the staff (rel_pos < 0.20)
        glyph_top = sorted_y[0] - int(0.5 * unit)
        glyph_h = int(1.5 * unit)   # short blob
        prob = _make_semantic_prob(h, w, STAFF_Y_1, glyph_top, glyph_h)
        result = detect_clef_for_staff(prob, STAFF_Y_1)
        assert result == CLEF_BASS, f"Small top blob should give bass, got '{result}'"


# ---------------------------------------------------------------------------
# Layer-registry integration
# ---------------------------------------------------------------------------

class TestClefExtract:

    def setup_method(self):
        layers.clear()

    def _populate_layers(self, n_staffs=2):
        h, w = 400, 1200
        prob = np.zeros((h, w, 4), dtype=np.float32)
        prob[:, :, 0] = 1.0
        layers.register_layer("semantic_map", prob)
        staff_lines = [STAFF_Y_1, STAFF_Y_2][:n_staffs]
        layers.register_layer("staff_lines", staff_lines)

    def test_extract_registers_clef_list(self):
        self._populate_layers()
        clef_extract()
        assert "clef_list" in layers.list_layers()

    def test_extract_returns_list_of_correct_length(self):
        self._populate_layers(n_staffs=2)
        result = clef_extract()
        assert isinstance(result, list)
        assert len(result) == 2

    def test_extract_defaults_unknown_to_treble(self):
        """When no clef blob is found, extract() must substitute treble."""
        self._populate_layers(n_staffs=1)
        result = clef_extract()
        # The fallback treble replaces CLEF_UNKNOWN
        assert result[0] == CLEF_TREBLE

    def test_extract_all_values_are_valid_clefs(self):
        self._populate_layers(n_staffs=2)
        result = clef_extract()
        valid = {CLEF_TREBLE, CLEF_BASS, CLEF_ALTO, CLEF_TENOR}
        for clef in result:
            assert clef in valid, f"Invalid clef '{clef}' returned by extract()"
