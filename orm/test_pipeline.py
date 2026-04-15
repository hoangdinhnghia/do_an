"""Smoke tests for orm/pipeline.py.

These tests do NOT require the ONNX checkpoints to be present and do NOT
perform real model inference.  Instead they patch the two model classes so
that the full code path (staff removal, notehead detection, pitch assignment,
result structure) is exercised with synthetic data.
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from orm.pipeline import OMRPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_img(h: int = 300, w: int = 400) -> np.ndarray:
    """Return a blank BGR image."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_staff_prob_map(h: int, w: int) -> np.ndarray:
    """Return a zeroed (H, W, 3) float32 map — no staff probability."""
    return np.zeros((h, w, 3), dtype=np.float32)


def _make_semantic_map(h: int, w: int) -> np.ndarray:
    """Return a zeroed (H, W, 4) float32 map — no symbol probability."""
    return np.zeros((h, w, 4), dtype=np.float32)


def _make_symbol_mask(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def _dual_result_no_staffs(img_bgr, **kwargs):
    """Fake run_dual_pipeline returning no staff lines."""
    h, w = img_bgr.shape[:2]
    return {
        "staff_lines": [],
        "noteheads": [],
        "staff_prob_map": _make_staff_prob_map(h, w),
        "semantic_map": _make_semantic_map(h, w),
        "symbol_mask": _make_symbol_mask(h, w),
    }


def _dual_result_one_staff(img_bgr, **kwargs):
    """Fake run_dual_pipeline returning one staff (5 lines at y=50..90)."""
    h, w = img_bgr.shape[:2]
    staff_lines = [[50, 60, 70, 80, 90]]
    return {
        "staff_lines": staff_lines,
        "noteheads": [(10, 45, 8, 8, 14, 50)],
        "staff_prob_map": _make_staff_prob_map(h, w),
        "semantic_map": _make_semantic_map(h, w),
        "symbol_mask": _make_symbol_mask(h, w),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOMRPipelineNoStaffs:
    """When no staves are detected the pipeline should return an empty result."""

    @patch("orm.pipeline.run_dual_pipeline", side_effect=_dual_result_no_staffs)
    def test_empty_staves_list(self, _mock):
        pipe = OMRPipeline(
            staffline_model=MagicMock(),
            semantic_model=MagicMock(),
        )
        result = pipe.run(_make_img())
        assert result["staves"] == []
        assert result["staff_lines"] == []
        assert result["noteheads_global"] == []

    @patch("orm.pipeline.run_dual_pipeline", side_effect=_dual_result_no_staffs)
    def test_maps_present_even_without_staves(self, _mock):
        pipe = OMRPipeline(
            staffline_model=MagicMock(),
            semantic_model=MagicMock(),
        )
        result = pipe.run(_make_img())
        assert result["staff_prob_map"] is not None
        assert result["semantic_map"] is not None
        assert result["symbol_mask"] is not None

    def test_raises_on_empty_image(self):
        pipe = OMRPipeline(
            staffline_model=MagicMock(),
            semantic_model=MagicMock(),
        )
        with pytest.raises(ValueError, match="empty"):
            pipe.run(np.array([]))


class TestOMRPipelineOneStaff:
    """When one staff is detected the pipeline should return structured note data."""

    @patch("orm.pipeline.run_dual_pipeline", side_effect=_dual_result_one_staff)
    def test_one_staff_in_result(self, _mock):
        pipe = OMRPipeline(
            staffline_model=MagicMock(),
            semantic_model=MagicMock(),
        )
        result = pipe.run(_make_img())
        assert len(result["staves"]) == 1

    @patch("orm.pipeline.run_dual_pipeline", side_effect=_dual_result_one_staff)
    def test_staff_dict_has_required_keys(self, _mock):
        pipe = OMRPipeline(
            staffline_model=MagicMock(),
            semantic_model=MagicMock(),
        )
        result = pipe.run(_make_img())
        staff = result["staves"][0]
        for key in ("staff_index", "staff_y", "clef", "notes", "staff_crop"):
            assert key in staff, f"Missing key: {key}"

    @patch("orm.pipeline.run_dual_pipeline", side_effect=_dual_result_one_staff)
    def test_default_clef_is_treble(self, _mock):
        pipe = OMRPipeline(
            staffline_model=MagicMock(),
            semantic_model=MagicMock(),
        )
        result = pipe.run(_make_img())
        assert result["staves"][0]["clef"] == "treble"

    @patch("orm.pipeline.run_dual_pipeline", side_effect=_dual_result_one_staff)
    def test_custom_clef_applied(self, _mock):
        pipe = OMRPipeline(
            staffline_model=MagicMock(),
            semantic_model=MagicMock(),
            default_clef="bass",
        )
        result = pipe.run(_make_img())
        assert result["staves"][0]["clef"] == "bass"

    @patch("orm.pipeline.run_dual_pipeline", side_effect=_dual_result_one_staff)
    def test_explicit_clef_list(self, _mock):
        pipe = OMRPipeline(
            staffline_model=MagicMock(),
            semantic_model=MagicMock(),
        )
        result = pipe.run(_make_img(), clefs=["bass"])
        assert result["staves"][0]["clef"] == "bass"

    @patch("orm.pipeline.run_dual_pipeline", side_effect=_dual_result_one_staff)
    def test_clef_list_length_mismatch_raises(self, _mock):
        pipe = OMRPipeline(
            staffline_model=MagicMock(),
            semantic_model=MagicMock(),
        )
        with pytest.raises(ValueError, match="len\\(clefs\\)"):
            pipe.run(_make_img(), clefs=["treble", "bass"])  # 2 clefs, 1 staff

    @patch("orm.pipeline.run_dual_pipeline", side_effect=_dual_result_one_staff)
    def test_note_dicts_have_required_keys(self, _mock):
        """Every note dict must expose pitch, step, octave, position, notehead."""
        pipe = OMRPipeline(
            staffline_model=MagicMock(),
            semantic_model=MagicMock(),
        )
        result = pipe.run(_make_img())
        for note in result["staves"][0]["notes"]:
            for key in ("pitch", "step", "octave", "position", "notehead"):
                assert key in note, f"Missing key in note dict: {key}"


class TestOMRPipelineLazyLoading:
    """Models are loaded lazily — passing None should not crash at construction."""

    def test_construction_without_models_does_not_raise(self):
        # OMRPipeline() with no args must not fail at __init__ time.
        pipe = OMRPipeline()
        assert pipe._staffline_model is None
        assert pipe._semantic_model is None

    def test_injected_mock_models_are_used(self):
        mock_staffline = MagicMock(spec=["predict_full", "extract_staff_lines"])
        mock_semantic = MagicMock(spec=["predict_full", "extract_noteheads", "extract_symbol_mask"])
        pipe = OMRPipeline(
            staffline_model=mock_staffline,
            semantic_model=mock_semantic,
        )
        assert pipe.staffline_model is mock_staffline
        assert pipe.semantic_model is mock_semantic
