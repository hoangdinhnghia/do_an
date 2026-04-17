"""
=============================================================
BƯỚC 8 — XÁC ĐỊNH TRƯỜNG ĐỘ (DURATION DETECTION)
=============================================================

Kiểm tra module orm.duration:
- classify_head_type() — open vs filled
- detect_stem()        — has stem, direction, stem column
- count_flags()        — number of flags/beams
- assign_duration()    — duration string synthesis
- assign_durations()   — bulk processing
- extract()            — layer registry integration

Không cần model ONNX.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from orm import layers
from orm.duration import (
    assign_duration,
    assign_durations,
    classify_head_type,
    count_flags,
    detect_stem,
    extract as duration_extract,
)
from orm.constant import (
    DUR_EIGHTH,
    DUR_HALF,
    DUR_QUARTER,
    DUR_WHOLE,
    DUR_SIXTEENTH,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_closed_head(w=700, h=200, cx=350, cy=100):
    """Filled (black) ellipse notehead with a vertical stem to the right."""
    img = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(img, (cx, cy), (9, 7), 0, 0, 360, 255, -1)   # solid fill
    stem_x = cx + 13
    cv2.line(img, (stem_x, cy - 30), (stem_x, cy + 5), 255, 2)   # up-stem
    return img


def _make_open_head(w=700, h=200, cx=350, cy=100):
    """Open (hollow) ellipse notehead — ring only, no stem."""
    img = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(img, (cx, cy), (9, 7), 0, 0, 360, 255, 2)   # outline only
    return img


def _make_open_head_with_stem(w=700, h=200, cx=350, cy=100):
    """Open notehead with a stem → half note."""
    img = _make_open_head(w, h, cx, cy)
    stem_x = cx + 11
    cv2.line(img, (stem_x, cy - 35), (stem_x, cy + 5), 255, 2)
    return img


STAFF_Y = [60, 75, 90, 105, 120]   # unit = 15 px


# ---------------------------------------------------------------------------
# classify_head_type
# ---------------------------------------------------------------------------

class TestClassifyHeadType:

    def test_filled_head(self):
        img = _make_closed_head()
        bbox = (341, 93, 18, 14, 350, 100)
        result = classify_head_type(img, bbox, unit=15.0)
        assert result == "filled", f"Expected 'filled', got '{result}'"

    def test_open_head(self):
        img = _make_open_head()
        bbox = (341, 93, 18, 14, 350, 100)
        result = classify_head_type(img, bbox, unit=15.0)
        assert result == "open", f"Expected 'open', got '{result}'"

    def test_returns_string(self):
        img = np.zeros((100, 200), dtype=np.uint8)
        bbox = (10, 10, 20, 15, 20, 17)
        result = classify_head_type(img, bbox, unit=15.0)
        assert result in ("open", "filled")

    def test_degenerate_bbox_returns_filled(self):
        """A bbox with zero area should return 'filled' gracefully."""
        img = np.zeros((100, 100), dtype=np.uint8)
        bbox = (0, 0, 0, 0, 0, 0)
        result = classify_head_type(img, bbox, unit=10.0)
        assert result in ("open", "filled")


# ---------------------------------------------------------------------------
# detect_stem
# ---------------------------------------------------------------------------

class TestDetectStem:

    def test_detects_stem_on_closed_head(self):
        img = _make_closed_head()
        bbox = (341, 93, 18, 14, 350, 100)
        has_stem, direction, stem_x = detect_stem(img, bbox, unit=15.0)
        assert has_stem, "Stem should be detected on a filled notehead with stem"

    def test_stem_direction_up(self):
        """Stem goes upward (above notehead centre)."""
        img = _make_closed_head()
        bbox = (341, 93, 18, 14, 350, 100)
        has_stem, direction, stem_x = detect_stem(img, bbox, unit=15.0)
        if has_stem:
            assert direction == "up", f"Expected stem direction 'up', got '{direction}'"

    def test_no_stem_on_empty_image(self):
        img = np.zeros((200, 700), dtype=np.uint8)
        bbox = (341, 93, 18, 14, 350, 100)
        has_stem, direction, stem_x = detect_stem(img, bbox, unit=15.0)
        assert not has_stem, "No stem should be detected on empty image"

    def test_returns_three_tuple(self):
        img = _make_closed_head()
        bbox = (341, 93, 18, 14, 350, 100)
        result = detect_stem(img, bbox, unit=15.0)
        assert len(result) == 3
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)
        assert isinstance(result[2], int)


# ---------------------------------------------------------------------------
# count_flags
# ---------------------------------------------------------------------------

class TestCountFlags:

    def test_no_flags_on_plain_stem(self):
        """A plain stem with no flags should return 0."""
        img = np.zeros((200, 700), dtype=np.uint8)
        # Draw a plain vertical stem
        cv2.line(img, (350, 50), (350, 100), 255, 2)
        n = count_flags(img, stem_x=350, stem_top=50, stem_bottom=100, unit=15.0)
        assert n == 0, f"Expected 0 flags, got {n}"

    def test_count_non_negative(self):
        img = np.zeros((200, 200), dtype=np.uint8)
        n = count_flags(img, stem_x=100, stem_top=30, stem_bottom=100, unit=15.0)
        assert n >= 0

    def test_count_capped_at_3(self):
        """count_flags must return at most 3."""
        img = np.ones((200, 700), dtype=np.uint8) * 255   # all white
        n = count_flags(img, stem_x=350, stem_top=20, stem_bottom=100, unit=15.0)
        assert n <= 3


# ---------------------------------------------------------------------------
# assign_duration
# ---------------------------------------------------------------------------

class TestAssignDuration:

    def test_open_no_stem_is_whole(self):
        assert assign_duration("open", has_stem=False, n_flags=0) == DUR_WHOLE

    def test_open_with_stem_is_half(self):
        assert assign_duration("open", has_stem=True, n_flags=0) == DUR_HALF

    def test_filled_no_flags_is_quarter(self):
        assert assign_duration("filled", has_stem=True, n_flags=0) == DUR_QUARTER

    def test_filled_one_flag_is_eighth(self):
        assert assign_duration("filled", has_stem=True, n_flags=1) == DUR_EIGHTH

    def test_filled_two_flags_is_sixteenth(self):
        assert assign_duration("filled", has_stem=True, n_flags=2) == DUR_SIXTEENTH

    def test_all_returned_values_are_strings(self):
        for head in ("open", "filled"):
            for stem in (True, False):
                for flags in (0, 1, 2, 3):
                    dur = assign_duration(head, stem, flags)
                    assert isinstance(dur, str) and len(dur) > 0


# ---------------------------------------------------------------------------
# assign_durations — batch
# ---------------------------------------------------------------------------

class TestAssignDurationsBatch:

    def _make_results(self, staff_y, positions):
        """Build a minimal notehead result list."""
        noteheads = [(cx - 9, cy - 7, 18, 14, cx, cy) for (cx, cy) in positions]
        annotated = np.zeros((80, 700, 3), dtype=np.uint8)
        return [(0, staff_y, noteheads, annotated)]

    def test_returns_one_result_per_staff(self):
        # Create image with closed noteheads + stems
        img = _make_closed_head(w=700, h=200, cx=350, cy=100)
        results = self._make_results(STAFF_Y, [(350, 100)])
        dr = assign_durations(results, bin_img=img, staff_lines=[STAFF_Y])
        assert len(dr) == 1

    def test_duration_labels_count_matches_noteheads(self):
        img = _make_closed_head(w=700, h=200, cx=350, cy=100)
        noteheads = [(330, 93, 18, 14, 350, 100), (500, 93, 18, 14, 509, 100)]
        annotated = np.zeros((200, 700, 3), dtype=np.uint8)
        results = [(0, STAFF_Y, noteheads, annotated)]
        dr = assign_durations(results, bin_img=img, staff_lines=[STAFF_Y])
        dur_labels = dr[0][-1]
        assert len(dur_labels) == 2, f"Expected 2 duration labels, got {len(dur_labels)}"

    def test_all_labels_are_valid_duration_strings(self):
        from orm.constant import DUR_BEATS
        img = _make_closed_head()
        noteheads = [(341, 93, 18, 14, 350, 100)]
        annotated = np.zeros((200, 700, 3), dtype=np.uint8)
        results = [(0, STAFF_Y, noteheads, annotated)]
        dr = assign_durations(results, bin_img=img, staff_lines=[STAFF_Y])
        for lbl in dr[0][-1]:
            assert lbl in DUR_BEATS, f"Unknown duration: '{lbl}'"


# ---------------------------------------------------------------------------
# Layer-registry integration
# ---------------------------------------------------------------------------

class TestDurationExtract:

    def setup_method(self):
        layers.clear()

    def _populate_layers(self):
        staff_y = STAFF_Y
        cx, cy = 350, 90
        noteheads = [(cx - 9, cy - 7, 18, 14, cx, cy)]
        annotated = np.zeros((200, 700, 3), dtype=np.uint8)
        pitch_labels = ["B4"]
        # pitch_results format: (idx, staff_y, noteheads, annotated, pitch_labels)
        pitch_results = [(0, staff_y, noteheads, annotated, pitch_labels)]
        layers.register_layer("pitch_results", pitch_results)

        img = _make_closed_head()
        layers.register_layer("img_no_staff", img)
        layers.register_layer("staff_lines", [staff_y])

        sem = np.zeros((200, 700, 4), dtype=np.float32)
        sem[:, :, 0] = 1.0
        layers.register_layer("semantic_map", sem)

    def test_extract_registers_duration_results(self):
        self._populate_layers()
        duration_extract()
        assert "duration_results" in layers.list_layers()

    def test_extract_returns_list(self):
        self._populate_layers()
        result = duration_extract()
        assert isinstance(result, list)
        assert len(result) == 1
