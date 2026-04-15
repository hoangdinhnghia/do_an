"""Unit tests for orm/pitch.py."""

import pytest

from orm.pitch import (
    assign_pitch,
    assign_pitches_all_staffs,
    assign_pitches_to_staff,
    position_from_bottom,
    unit_size,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# A standard staff with 10-pixel spacing between lines (unit = 5 px)
STAFF_Y = [100, 110, 120, 130, 140]   # 5 lines, spacing = 10 px → unit = 5 px


# ---------------------------------------------------------------------------
# unit_size
# ---------------------------------------------------------------------------

class TestUnitSize:
    def test_regular_spacing(self):
        assert unit_size(STAFF_Y) == pytest.approx(5.0)

    def test_irregular_spacing_returns_average(self):
        # gaps: 8, 12, 8, 12 → mean = 10 → unit = 5
        ys = [0, 8, 20, 28, 40]
        assert unit_size(ys) == pytest.approx(5.0)

    def test_single_element(self):
        assert unit_size([50]) == pytest.approx(1.0)

    def test_unsorted_input_handled(self):
        ys = [140, 100, 120, 130, 110]
        assert unit_size(ys) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# position_from_bottom
# ---------------------------------------------------------------------------

class TestPositionFromBottom:
    def test_on_bottom_line(self):
        # centroid exactly at bottom line → position 0
        assert position_from_bottom(100, STAFF_Y) == 0

    def test_on_top_line(self):
        # centroid at top line (y=140) but y increases downward,
        # so top line is *lower* in pixel space than bottom.
        # Top line y = 140, bottom line y = 100
        # delta_y = 100 - 140 = -40, unit = 5 → position = -40/5 = -8
        assert position_from_bottom(140, STAFF_Y) == -8

    def test_first_space(self):
        # half-way between line 1 (y=100) and line 2 (y=110) → y=105
        # delta_y = 100 - 105 = -5, unit = 5 → position = -1
        assert position_from_bottom(105, STAFF_Y) == -1

    def test_above_staff(self):
        # note above top line in music (lower y in image)
        # y = 90, delta_y = 100 - 90 = 10, unit = 5 → position = 2
        assert position_from_bottom(90, STAFF_Y) == 2

    def test_ledger_below(self):
        # note below bottom line (higher y in image)
        # y = 160, delta_y = 100 - 160 = -60, unit = 5 → position = -12
        assert position_from_bottom(160, STAFF_Y) == -12


# ---------------------------------------------------------------------------
# assign_pitch — treble clef
# ---------------------------------------------------------------------------

class TestAssignPitchTreble:
    """Treble clef reference: bottom line (position 0) = E4."""

    def test_bottom_line_is_E4(self):
        result = assign_pitch(100, STAFF_Y, clef="treble")
        assert result["pitch"] == "E4"
        assert result["step"] == "E"
        assert result["octave"] == 4
        assert result["position"] == 0

    def test_first_space_is_F4(self):
        # position 1 above bottom line → F4
        # 1 unit above y=100 → y=95
        result = assign_pitch(95, STAFF_Y, clef="treble")
        assert result["step"] == "F"
        assert result["octave"] == 4

    def test_second_line_is_G4(self):
        # position 2 → G4, y = 100 - 10 = 90
        result = assign_pitch(90, STAFF_Y, clef="treble")
        assert result["step"] == "G"
        assert result["octave"] == 4

    def test_third_line_is_B4(self):
        # position 4 → B4, y = 100 - 20 = 80
        result = assign_pitch(80, STAFF_Y, clef="treble")
        assert result["step"] == "B"
        assert result["octave"] == 4

    def test_top_line_is_F5(self):
        # position 8 → F5, y = 100 - 40 = 60
        result = assign_pitch(60, STAFF_Y, clef="treble")
        assert result["step"] == "F"
        assert result["octave"] == 5

    def test_ledger_above_C6(self):
        # position 12 above E4 → abs_index = 2+12=14 → 14%7=0 (C), 14//7=2 → C6
        # y = 100 - 12*5 = 40
        result = assign_pitch(40, STAFF_Y, clef="treble")
        assert result["step"] == "C"
        assert result["octave"] == 6

    def test_accidental_sharp(self):
        result = assign_pitch(100, STAFF_Y, clef="treble", accidental="#")
        assert result["pitch"] == "E#4"

    def test_accidental_flat(self):
        result = assign_pitch(100, STAFF_Y, clef="treble", accidental="b")
        assert result["pitch"] == "Eb4"

    def test_no_accidental_omitted(self):
        result = assign_pitch(100, STAFF_Y, clef="treble", accidental=None)
        assert "#" not in result["pitch"]
        assert "b" not in result["pitch"]


# ---------------------------------------------------------------------------
# assign_pitch — bass clef
# ---------------------------------------------------------------------------

class TestAssignPitchBass:
    """Bass clef reference: bottom line (position 0) = G2."""

    def test_bottom_line_is_G2(self):
        result = assign_pitch(100, STAFF_Y, clef="bass")
        assert result["step"] == "G"
        assert result["octave"] == 2

    def test_fourth_line_is_F3(self):
        # position 6 above G2 → G→A→B→C→D→E→F = F3
        # y = 100 - 30 = 70
        result = assign_pitch(70, STAFF_Y, clef="bass")
        assert result["step"] == "F"
        assert result["octave"] == 3


# ---------------------------------------------------------------------------
# assign_pitch — alto clef
# ---------------------------------------------------------------------------

class TestAssignPitchAlto:
    def test_bottom_line_is_F3(self):
        result = assign_pitch(100, STAFF_Y, clef="alto")
        assert result["step"] == "F"
        assert result["octave"] == 3


# ---------------------------------------------------------------------------
# assign_pitch — tenor clef
# ---------------------------------------------------------------------------

class TestAssignPitchTenor:
    def test_bottom_line_is_D3(self):
        result = assign_pitch(100, STAFF_Y, clef="tenor")
        assert result["step"] == "D"
        assert result["octave"] == 3


# ---------------------------------------------------------------------------
# assign_pitch — invalid clef
# ---------------------------------------------------------------------------

class TestAssignPitchInvalidClef:
    def test_raises_on_unknown_clef(self):
        with pytest.raises(ValueError, match="Unknown clef"):
            assign_pitch(100, STAFF_Y, clef="percussion")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# assign_pitches_to_staff
# ---------------------------------------------------------------------------

class TestAssignPitchesToStaff:
    def _make_noteheads(self, cys):
        # (x, y, w, h, cx, cy) — give different x so sort is deterministic
        return [(i * 20, cy - 5, 10, 10, i * 20 + 5, cy) for i, cy in enumerate(cys)]

    def test_empty_noteheads(self):
        result = assign_pitches_to_staff([], STAFF_Y)
        assert result == []

    def test_single_notehead(self):
        nh = self._make_noteheads([100])
        result = assign_pitches_to_staff(nh, STAFF_Y)
        assert len(result) == 1
        assert result[0]["pitch"] == "E4"

    def test_sorted_left_to_right(self):
        # Create noteheads in reversed x order
        noteheads = [(40, 95, 10, 10, 45, 95), (10, 95, 10, 10, 15, 100)]
        result = assign_pitches_to_staff(noteheads, STAFF_Y)
        # Should be sorted by cx: 15, then 45
        assert result[0]["notehead"][4] == 15
        assert result[1]["notehead"][4] == 45

    def test_multiple_notes_correct_pitches(self):
        # Bottom line (E4), second line (G4), middle line (B4)
        cys = [100, 90, 80]  # positions 0, 2, 4 → E4, G4, B4
        nh = self._make_noteheads(cys)
        result = assign_pitches_to_staff(nh, STAFF_Y)
        pitches = [r["pitch"] for r in result]
        assert "E4" in pitches
        assert "G4" in pitches
        assert "B4" in pitches

    def test_result_dict_has_all_keys(self):
        nh = self._make_noteheads([100])
        result = assign_pitches_to_staff(nh, STAFF_Y)
        keys = result[0].keys()
        assert "notehead" in keys
        assert "pitch" in keys
        assert "step" in keys
        assert "octave" in keys
        assert "position" in keys


# ---------------------------------------------------------------------------
# assign_pitches_all_staffs
# ---------------------------------------------------------------------------

class TestAssignPitchesAllStaffs:
    def test_two_staves(self):
        staff1_y = [100, 110, 120, 130, 140]
        staff2_y = [200, 210, 220, 230, 240]
        nh1 = [(0, 95, 10, 10, 5, 100)]
        nh2 = [(0, 195, 10, 10, 5, 200)]
        result = assign_pitches_all_staffs(
            [(staff1_y, nh1), (staff2_y, nh2)],
            clefs=["treble", "bass"],
        )
        assert len(result) == 2
        assert result[0][0]["pitch"] == "E4"
        assert result[1][0]["step"] == "G"

    def test_mismatched_clefs_raises(self):
        with pytest.raises(ValueError, match="Length of clefs"):
            assign_pitches_all_staffs(
                [([100, 110, 120, 130, 140], [])],
                clefs=["treble", "bass"],
            )

    def test_default_clef_is_treble(self):
        staff_y = [100, 110, 120, 130, 140]
        nh = [(0, 95, 10, 10, 5, 100)]
        result = assign_pitches_all_staffs([(staff_y, nh)])
        assert result[0][0]["pitch"] == "E4"
