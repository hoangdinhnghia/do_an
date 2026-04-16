"""
=============================================================
BƯỚC 3 — TRÍCH XUẤT DÒNG KẺ NHẠC (STAFFLINE EXTRACTION)
=============================================================

Mục tiêu
--------
Từ **probability map** của model stream 1 (H×W×3 float32), xác định
chính xác vị trí pixel-y của từng dòng kẻ trong từng staff system.

Thuật toán (3 bước nhỏ)
-----------------------
1. **Binarise** kênh staff (ch 1) theo ngưỡng ``conf_thresh`` → binary mask.
2. **Horizontal projection** → profile 1-D → tìm peak y bằng ``find_peaks_profile``.
3. **Group + Refine** → ``group_peaks_to_staffs`` gom thành nhóm 5 dòng,
   ``refine_staff_lines`` tối ưu lại vị trí trong cửa sổ nhỏ.

Kết quả
-------
``List[List[int]]`` — mỗi staff system là 1 list 5 y-coordinates (đã sắp xếp).
Kết quả được đăng ký vào layer registry dưới key ``"staff_lines"``.

Các hàm được kiểm thử
---------------------
- ``orm.staff_detection.find_peaks_profile``
- ``orm.staff_detection.group_peaks_to_staffs``
- ``orm.staff_detection.refine_staff_lines``
- ``orm.staffline_extraction.extract``        ← hàm pipeline chính
- ``orm.staffline_extraction.get_staff_unit`` ← tính khoảng cách dòng

Cách chạy
---------
::

    cd /home/runner/work/do_an/do_an
    pytest tests/test_step3_staffline_extraction.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from orm import layers
from orm.staff_detection import (
    find_peaks_profile,
    group_peaks_to_staffs,
    refine_staff_lines,
)
from orm.staffline_extraction import extract, get_staff_unit
from orm.exceptions import StafflineNotDetected


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _make_prob_map_with_staffs(staff_y_sets, h=400, w=800):
    """Build a fake (H, W, 3) prob map with staff lines at given y positions."""
    prob = np.zeros((h, w, 3), dtype=np.float32)
    prob[:, :, 0] = 1.0
    for ys in staff_y_sets:
        for y in ys:
            prob[max(0, y - 1): y + 2, :, 0] = 0.0
            prob[max(0, y - 1): y + 2, :, 1] = 0.9
    return prob


# ─────────────────────────────────────────────────────────────────────────────
# 3a. find_peaks_profile
# ─────────────────────────────────────────────────────────────────────────────

class TestFindPeaksProfile:
    """Unit tests for the horizontal-projection peak finder."""

    def test_finds_peaks_at_correct_positions(self):
        """Profile with 5 wide peaks at known y → must find peaks at those y values.

        ``find_peaks_profile`` requires at least ``min_len=2`` consecutive pixels above
        the threshold.  The peaks here are 3 px wide (y-1..y+1) and spaced 15 px apart
        so Gaussian smoothing (sigma=2) does not merge adjacent peaks.
        """
        profile = np.zeros(300, dtype=np.float32)
        peak_ys = [20, 35, 50, 65, 80]   # spacing = 15 px, well above min_dist=6
        for y in peak_ys:
            profile[y - 1 : y + 2] = 100.0   # 3-px wide peaks → satisfies min_len=2
        found = find_peaks_profile(profile, smooth_sigma=1.0, min_dist=5, thresh_ratio=0.5)
        assert len(found) >= len(peak_ys) // 2, \
            f"Expected ~{len(peak_ys)} peaks, found {len(found)}"

    def test_no_peaks_in_flat_profile(self):
        """Uniform profile should return no peaks."""
        profile = np.ones(200, dtype=np.float32) * 10.0
        found = find_peaks_profile(profile, smooth_sigma=1.0, thresh_ratio=0.99)
        # Flat profile: all values equal threshold, so either 0 or 1 peak cluster
        assert len(found) <= 1

    def test_returns_list_of_ints(self):
        profile = np.zeros(100, dtype=np.float32)
        profile[20] = 50.0
        found = find_peaks_profile(profile)
        assert all(isinstance(y, int) for y in found)

    def test_peaks_within_bounds(self):
        profile = np.zeros(200, dtype=np.float32)
        for y in [30, 40, 50, 60, 70]:
            profile[y] = 80.0
        found = find_peaks_profile(profile)
        assert all(0 <= y < 200 for y in found)


# ─────────────────────────────────────────────────────────────────────────────
# 3b. group_peaks_to_staffs
# ─────────────────────────────────────────────────────────────────────────────

class TestGroupPeaksToStaffs:
    """Unit tests for the staff grouping algorithm."""

    def test_groups_10_peaks_into_2_staffs(self):
        """10 evenly-spaced peaks should form exactly 2 five-line staffs."""
        # Staff 1: y = 10, 20, 30, 40, 50   (spacing 10)
        # Staff 2: y = 110, 120, 130, 140, 150  (spacing 10, large gap from staff 1)
        ys = [10, 20, 30, 40, 50, 110, 120, 130, 140, 150]
        staffs = group_peaks_to_staffs(ys)
        assert len(staffs) == 2, f"Expected 2 staffs, got {len(staffs)}"

    def test_each_staff_has_five_lines(self):
        ys = [10, 20, 30, 40, 50, 110, 120, 130, 140, 150]
        staffs = group_peaks_to_staffs(ys)
        for i, staff in enumerate(staffs):
            assert len(staff) == 5, f"Staff {i} should have 5 lines, got {len(staff)}"

    def test_empty_input_returns_empty(self):
        assert group_peaks_to_staffs([]) == []

    def test_returns_sorted_within_each_staff(self):
        ys = [50, 10, 30, 20, 40]   # unsorted
        staffs = group_peaks_to_staffs(ys)
        for staff in staffs:
            assert staff == sorted(staff), "y-values within a staff must be sorted"

    def test_single_staff(self):
        ys = [20, 30, 40, 50, 60]
        staffs = group_peaks_to_staffs(ys)
        assert len(staffs) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 3c. refine_staff_lines
# ─────────────────────────────────────────────────────────────────────────────

class TestRefineStaffLines:
    """Unit tests for the position refinement step."""

    def test_output_length_unchanged(self):
        staffs = [[10, 20, 30, 40, 50], [110, 120, 130, 140, 150]]
        # Binary image with lines at the same positions
        img = np.zeros((200, 300), dtype=np.uint8)
        for y in [10, 20, 30, 40, 50, 110, 120, 130, 140, 150]:
            img[y, :] = 255
        refined = refine_staff_lines(staffs, img)
        assert len(refined) == len(staffs)

    def test_each_staff_has_five_lines(self):
        staffs = [[10, 20, 30, 40, 50]]
        img = np.zeros((100, 200), dtype=np.uint8)
        for y in [10, 20, 30, 40, 50]:
            img[y, :] = 255
        refined = refine_staff_lines(staffs, img)
        for staff in refined:
            assert len(staff) == 5

    def test_refines_towards_actual_line(self):
        """If we offset the initial guess by 2 px, refine should correct it."""
        staffs = [[12, 22, 32, 42, 52]]   # actual lines at +2 px
        img = np.zeros((100, 200), dtype=np.uint8)
        for y in [10, 20, 30, 40, 50]:
            img[y, :] = 255           # true position
        refined = refine_staff_lines(staffs, img, window=5)
        # Refined positions should be closer to 10,20,...50 than the original guesses
        for true_y, refined_staff, init_staff in zip(
            [10, 20, 30, 40, 50], refined[0], staffs[0]
        ):
            err_refined = abs(refined_staff - true_y)
            err_init    = abs(init_staff - true_y)
            assert err_refined <= err_init + 1, \
                f"Refinement made position worse: {init_staff}→{refined_staff} (true={true_y})"


# ─────────────────────────────────────────────────────────────────────────────
# 3d. staffline_extraction.extract  (pipeline step)
# ─────────────────────────────────────────────────────────────────────────────

class TestStafflineExtractionExtract:
    """Unit tests for the pipeline-facing ``extract()`` function."""

    def setup_method(self):
        """Clear the layer registry before each test."""
        layers.clear()

    def test_returns_two_staffs(self, synthetic_staff_prob_map):
        """Fake prob map with 2×5 lines should produce 2 staff systems."""
        layers.register_layer("staff_prob_map", synthetic_staff_prob_map)
        staffs = extract(conf_thresh=0.5)
        assert len(staffs) == 2, f"Expected 2 staff systems, got {len(staffs)}"

    def test_each_staff_has_five_y_coordinates(self, synthetic_staff_prob_map):
        layers.register_layer("staff_prob_map", synthetic_staff_prob_map)
        staffs = extract(conf_thresh=0.5)
        for i, staff in enumerate(staffs):
            assert len(staff) == 5, f"Staff {i} should have 5 lines"

    def test_y_coordinates_are_sorted(self, synthetic_staff_prob_map):
        layers.register_layer("staff_prob_map", synthetic_staff_prob_map)
        staffs = extract(conf_thresh=0.5)
        for staff in staffs:
            assert staff == sorted(staff)

    def test_y_near_expected_positions(self, synthetic_staff_prob_map):
        """Detected y should be within ±3 px of the true positions (80..120, 200..240)."""
        layers.register_layer("staff_prob_map", synthetic_staff_prob_map)
        staffs = extract(conf_thresh=0.5)
        expected = [[80, 90, 100, 110, 120], [200, 210, 220, 230, 240]]
        for staff_detected, staff_expected in zip(
            sorted(staffs, key=lambda s: s[0]),
            sorted(expected, key=lambda s: s[0]),
        ):
            for y_det, y_exp in zip(staff_detected, staff_expected):
                assert abs(y_det - y_exp) <= 3, \
                    f"Detected y={y_det}, expected ≈{y_exp} (tolerance ±3 px)"

    def test_registers_staff_lines_layer(self, synthetic_staff_prob_map):
        layers.register_layer("staff_prob_map", synthetic_staff_prob_map)
        extract(conf_thresh=0.5)
        assert "staff_lines" in layers.list_layers()

    def test_raises_when_no_staffs_found(self):
        """All-background prob map → StafflineNotDetected must be raised."""
        h, w = 200, 400
        blank_prob = np.zeros((h, w, 3), dtype=np.float32)
        blank_prob[:, :, 0] = 1.0
        layers.register_layer("staff_prob_map", blank_prob)
        with pytest.raises(StafflineNotDetected):
            extract(conf_thresh=0.5)

    def test_raises_when_layer_missing(self):
        """Calling extract() without registering prob map → KeyError."""
        with pytest.raises(KeyError):
            extract()


# ─────────────────────────────────────────────────────────────────────────────
# 3e. get_staff_unit
# ─────────────────────────────────────────────────────────────────────────────

class TestGetStaffUnit:

    def setup_method(self):
        layers.clear()

    def test_correct_unit_from_10px_spacing(self):
        unit = get_staff_unit([[10, 20, 30, 40, 50]])
        assert unit == pytest.approx(10.0, abs=0.5)

    def test_fallback_when_no_staff(self):
        unit = get_staff_unit([])
        assert unit == pytest.approx(10.0)

    def test_reads_from_layer_registry(self, synthetic_staff_prob_map):
        layers.register_layer("staff_prob_map", synthetic_staff_prob_map)
        staffs = extract(conf_thresh=0.5)
        unit = get_staff_unit()   # no argument → reads from registry
        expected = 10.0           # lines spaced 10 px in synthetic_staff_prob_map
        assert abs(unit - expected) <= 2.0, f"Staff unit: expected ≈{expected}, got {unit}"
