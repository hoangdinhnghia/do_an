"""
=============================================================
BƯỚC 7 — GÁN CAO ĐỘ (PITCH ASSIGNMENT)
=============================================================

Kiểm tra module orm.pitch:
- assign_pitch_to_notehead() đúng với treble và bass clef
- assign_pitches() xử lý batch
- extract() đọc/ghi layer registry

Không cần model ONNX; tất cả là unit tests trên dữ liệu tổng hợp.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from orm import layers
from orm.pitch import assign_pitch_to_notehead, assign_pitches, extract as pitch_extract
from orm.constant import CLEF_TREBLE, CLEF_BASS, PITCH_NAMES


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

TREBLE_STAFF_Y = [100, 110, 120, 130, 140]   # spacing = 10 px, unit = 10
BASS_STAFF_Y   = [200, 212, 224, 236, 248]   # spacing = 12 px, unit = 12


# ---------------------------------------------------------------------------
# assign_pitch_to_notehead — treble clef
# ---------------------------------------------------------------------------

class TestPitchTreble:
    """Pitch assignment with treble clef."""

    def test_bottom_line_is_e4(self):
        """Treble clef: cy exactly on bottom staff line → E4."""
        name, octave, label = assign_pitch_to_notehead(
            cy=TREBLE_STAFF_Y[-1], staff_y=TREBLE_STAFF_Y, clef=CLEF_TREBLE
        )
        assert name == "E", f"Expected E4 at bottom line, got {label}"
        assert octave == 4

    def test_fourth_line_is_g4(self):
        """Treble clef: 4th line from bottom (= index 1 from bottom) → G4."""
        cy = TREBLE_STAFF_Y[-2]   # second from bottom
        name, octave, label = assign_pitch_to_notehead(cy, TREBLE_STAFF_Y, CLEF_TREBLE)
        assert name == "G", f"Expected G4 at 4th line, got {label}"
        assert octave == 4

    def test_middle_line_is_b4(self):
        """Treble clef: middle line (line 3) → B4."""
        cy = TREBLE_STAFF_Y[2]
        name, octave, label = assign_pitch_to_notehead(cy, TREBLE_STAFF_Y, CLEF_TREBLE)
        assert name == "B", f"Expected B4 at middle line, got {label}"
        assert octave == 4

    def test_top_line_is_f5(self):
        """Treble clef: top line → F5."""
        cy = TREBLE_STAFF_Y[0]
        name, octave, label = assign_pitch_to_notehead(cy, TREBLE_STAFF_Y, CLEF_TREBLE)
        assert name == "F", f"Expected F5 at top line, got {label}"
        assert octave == 5

    def test_pitch_rises_upward(self):
        """Pitch labels must be in ascending order from bottom to top."""
        # Sample 9 positions from bottom line upward (every half-unit)
        unit = float(np.median(np.diff(sorted(TREBLE_STAFF_Y))))
        half = unit / 2.0
        bottom = TREBLE_STAFF_Y[-1]
        positions = [int(bottom - i * half) for i in range(9)]
        labels = [
            assign_pitch_to_notehead(cy, TREBLE_STAFF_Y, CLEF_TREBLE)[2]
            for cy in positions
        ]
        # Convert labels to diatonic index for comparison
        def _idx(lbl):
            name = lbl[0]
            oct_ = int(lbl[1])
            return PITCH_NAMES.index(name) + oct_ * 7

        indices = [_idx(lbl) for lbl in labels]
        assert indices == sorted(indices), f"Pitches should rise upward: {labels}"

    def test_label_format(self):
        """Label must be letter + digit, e.g. 'G4'."""
        _, _, label = assign_pitch_to_notehead(
            cy=TREBLE_STAFF_Y[1], staff_y=TREBLE_STAFF_Y, clef=CLEF_TREBLE
        )
        assert len(label) >= 2
        assert label[0] in PITCH_NAMES
        assert label[1:].lstrip("-").isdigit()

    def test_above_staff_higher_pitch(self):
        """A note above the top staff line must have a higher pitch than F5."""
        unit = float(np.median(np.diff(sorted(TREBLE_STAFF_Y))))
        cy_above = TREBLE_STAFF_Y[0] - int(unit)
        _, _, label_above = assign_pitch_to_notehead(cy_above, TREBLE_STAFF_Y, CLEF_TREBLE)
        _, _, label_top = assign_pitch_to_notehead(TREBLE_STAFF_Y[0], TREBLE_STAFF_Y, CLEF_TREBLE)

        def _idx(lbl):
            return PITCH_NAMES.index(lbl[0]) + int(lbl[1:]) * 7

        assert _idx(label_above) > _idx(label_top)

    def test_below_staff_lower_pitch(self):
        """A note below the bottom staff line must have a lower pitch than E4."""
        unit = float(np.median(np.diff(sorted(TREBLE_STAFF_Y))))
        cy_below = TREBLE_STAFF_Y[-1] + int(unit)
        _, _, label_below = assign_pitch_to_notehead(cy_below, TREBLE_STAFF_Y, CLEF_TREBLE)
        _, _, label_bot = assign_pitch_to_notehead(TREBLE_STAFF_Y[-1], TREBLE_STAFF_Y, CLEF_TREBLE)

        def _idx(lbl):
            return PITCH_NAMES.index(lbl[0]) + int(lbl[1:]) * 7

        assert _idx(label_below) < _idx(label_bot)


# ---------------------------------------------------------------------------
# assign_pitch_to_notehead — bass clef
# ---------------------------------------------------------------------------

class TestPitchBass:
    """Pitch assignment with bass clef."""

    def test_bottom_line_is_g2(self):
        """Bass clef: bottom line → G2."""
        name, octave, label = assign_pitch_to_notehead(
            cy=BASS_STAFF_Y[-1], staff_y=BASS_STAFF_Y, clef=CLEF_BASS
        )
        assert name == "G", f"Expected G2 at bottom line, got {label}"
        assert octave == 2

    def test_top_line_is_a3(self):
        """Bass clef: top line → A3."""
        name, octave, label = assign_pitch_to_notehead(
            cy=BASS_STAFF_Y[0], staff_y=BASS_STAFF_Y, clef=CLEF_BASS
        )
        assert name == "A", f"Expected A3 at top line, got {label}"
        assert octave == 3

    def test_pitch_strictly_higher_than_treble_at_same_position(self):
        """The same y position maps to different pitches for treble vs bass."""
        staff = [100, 110, 120, 130, 140]
        _, _, t_label = assign_pitch_to_notehead(120, staff, CLEF_TREBLE)
        _, _, b_label = assign_pitch_to_notehead(120, staff, CLEF_BASS)
        assert t_label != b_label, "Treble and bass pitches at same position must differ"


# ---------------------------------------------------------------------------
# assign_pitches — batch
# ---------------------------------------------------------------------------

class TestAssignPitchesBatch:

    def _make_notehead_results(self, staff_y, cys):
        """Build a minimal notehead_results structure."""
        noteheads = [(100, cy - 4, 10, 8, 105, cy) for cy in cys]
        annotated = np.zeros((50, 200, 3), dtype=np.uint8)
        return [(0, staff_y, noteheads, annotated)]

    def test_returns_one_entry_per_staff(self):
        results = self._make_notehead_results(TREBLE_STAFF_Y, [120, 130])
        pr = assign_pitches(results, clef_list=[CLEF_TREBLE])
        assert len(pr) == 1

    def test_pitch_labels_count_matches_noteheads(self):
        results = self._make_notehead_results(TREBLE_STAFF_Y, [115, 125, 135])
        pr = assign_pitches(results, clef_list=[CLEF_TREBLE])
        assert len(pr[0][4]) == 3, "Number of pitch labels must match notehead count"

    def test_no_clef_list_defaults_to_treble(self):
        results = self._make_notehead_results(TREBLE_STAFF_Y, [140])
        pr_treble = assign_pitches(results, clef_list=[CLEF_TREBLE])
        pr_none   = assign_pitches(results, clef_list=None)
        assert pr_treble[0][4] == pr_none[0][4]

    def test_all_labels_are_non_empty_strings(self):
        results = self._make_notehead_results(TREBLE_STAFF_Y, [100, 115, 130])
        pr = assign_pitches(results)
        for lbl in pr[0][4]:
            assert isinstance(lbl, str) and len(lbl) >= 2


# ---------------------------------------------------------------------------
# Layer-registry integration
# ---------------------------------------------------------------------------

class TestPitchExtract:

    def setup_method(self):
        layers.clear()

    def _populate_layers(self):
        staff_y = [100, 110, 120, 130, 140]
        noteheads = [(300, cy - 4, 10, 8, 305, cy) for cy in [120, 130, 140]]
        annotated = np.zeros((80, 500, 3), dtype=np.uint8)
        notehead_results = np.array(
            [(0, staff_y, noteheads, annotated)], dtype=object
        )
        layers.register_layer("notehead_results", notehead_results)
        layers.register_layer("clef_list", [CLEF_TREBLE])

    def test_extract_registers_pitch_results(self):
        self._populate_layers()
        pitch_extract()
        assert "pitch_results" in layers.list_layers()

    def test_extract_returns_list(self):
        self._populate_layers()
        result = pitch_extract()
        assert isinstance(result, list)
        assert len(result) == 1

    def test_extract_pitch_labels_present(self):
        self._populate_layers()
        result = pitch_extract()
        assert len(result[0]) >= 5
        labels = result[0][4]
        assert len(labels) == 3
