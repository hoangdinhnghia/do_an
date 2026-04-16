"""
=============================================================
BƯỚC 2 — STAFFLINE SEGMENTATION MODEL (STREAM 1)
=============================================================

Mục tiêu
--------
Sử dụng model ``1st_model.onnx`` (unet_big) để **phân loại từng pixel**
trong ảnh nhạc thành một trong 3 lớp:

    • 0 — nền (background)
    • 1 — dòng kẻ nhạc (staff line)       ← **chúng ta cần lớp này**
    • 2 — ký hiệu nhạc (music symbol)

Kiến trúc inference
-------------------
Model nhận từng patch 256×256 (uint8 BGR), trả về probability map
256×256×3 (float32 softmax).  Để xử lý ảnh kích thước tuỳ ý, các
patch được tạo ra bằng cách **tile-and-stitch** với cửa sổ Hann để
tránh viền giữa các patch.

Các đối tượng được kiểm thử
----------------------------
- ``orm.inference.StafflineModel``
  - ``predict_patch``  — inference trên 1 patch 256×256
  - ``predict_full``   — tile-and-stitch trên ảnh đầy đủ
- ``orm.inference._tile_and_stitch``  — hàm helper nội bộ

Unit tests (không cần model)
-----------------------------
Dùng dữ liệu giả (fake probability maps) để kiểm tra shape, dtype,
và logic blending mà không cần tải model thật.

Integration tests (cần model)
------------------------------
Được đánh dấu bằng fixture ``staffline_model``; tự động bỏ qua nếu
checkpoint ``1st_model.onnx`` không có.

Cách chạy
---------
::

    cd <repo_root>
    # Chỉ unit tests (không cần model):
    pytest tests/test_step2_staffline_model.py -v -k "not Integration"

    # Tất cả (bao gồm integration, cần model):
    pytest tests/test_step2_staffline_model.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from orm.inference import _hann_window_2d, _tile_and_stitch
from orm.constant import STAFFLINE_PATCH_SIZE, M1_CH_STAFF


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _make_fake_predict(out_channels: int = 3, patch_size: int = 256):
    """Return a callable that mimics model.predict_patch: returns zero-filled map."""
    def fake_predict(patch: np.ndarray) -> np.ndarray:
        out = np.zeros((patch_size, patch_size, out_channels), dtype=np.float32)
        out[:, :, 0] = 1.0   # all background
        return out
    return fake_predict


# ─────────────────────────────────────────────────────────────────────────────
# 2a. Hann window
# ─────────────────────────────────────────────────────────────────────────────

class TestHannWindow:
    """Unit tests for the 2-D Hann blending window."""

    def test_shape(self):
        win = _hann_window_2d(256)
        assert win.shape == (256, 256)

    def test_dtype_float32(self):
        win = _hann_window_2d(128)
        assert win.dtype == np.float32

    def test_zero_at_border(self):
        win = _hann_window_2d(64)
        assert win[0, 0] == pytest.approx(0.0, abs=1e-6)
        assert win[0, -1] == pytest.approx(0.0, abs=1e-6)
        assert win[-1, 0] == pytest.approx(0.0, abs=1e-6)
        assert win[-1, -1] == pytest.approx(0.0, abs=1e-6)

    def test_peak_at_center(self):
        size = 64
        win = _hann_window_2d(size)
        mid = size // 2
        center_val = win[mid, mid]
        # Center should be close to max
        assert center_val == pytest.approx(win.max(), abs=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# 2b. Tile-and-stitch helper
# ─────────────────────────────────────────────────────────────────────────────

class TestTileAndStitch:
    """Unit tests for ``_tile_and_stitch``."""

    def test_output_shape_matches_input(self):
        """Output (H, W, C) must match the input (H, W)."""
        patch_size = 64
        img = np.zeros((150, 200, 3), dtype=np.uint8)
        pred = _tile_and_stitch(img, patch_size, overlap=16,
                                predict_fn=_make_fake_predict(3, patch_size))
        assert pred.shape == (150, 200, 3)

    def test_output_dtype_float32(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        pred = _tile_and_stitch(img, 64, overlap=8,
                                predict_fn=_make_fake_predict(3, 64))
        assert pred.dtype == np.float32

    def test_output_values_in_0_1(self):
        """Probability values must stay in [0, 1]."""
        img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        pred = _tile_and_stitch(img, 64, overlap=16,
                                predict_fn=_make_fake_predict(3, 64))
        assert pred.min() >= 0.0
        assert pred.max() <= 1.0 + 1e-6

    def test_raises_when_overlap_ge_patch_size(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            _tile_and_stitch(img, 64, overlap=64,
                             predict_fn=_make_fake_predict(3, 64))

    def test_image_smaller_than_patch_size(self):
        """Images smaller than patch_size should still work (padded internally)."""
        img = np.zeros((30, 40, 3), dtype=np.uint8)
        pred = _tile_and_stitch(img, 64, overlap=8,
                                predict_fn=_make_fake_predict(3, 64))
        assert pred.shape == (30, 40, 3)

    def test_background_channel_is_dominant(self):
        """With all-background fake predictions, channel 0 should be highest."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        pred = _tile_and_stitch(img, 64, overlap=8,
                                predict_fn=_make_fake_predict(3, 64))
        # Channel 0 (background) should win
        assert np.all(pred[:, :, 0] >= pred[:, :, 1])


# ─────────────────────────────────────────────────────────────────────────────
# 2c. StafflineModel.predict_patch  (Integration — needs real model)
# ─────────────────────────────────────────────────────────────────────────────

class TestStafflineModelPredictPatch:
    """Integration tests for StafflineModel.predict_patch."""

    def test_output_shape(self, staffline_model):
        """predict_patch(256×256 patch) must return shape (256, 256, 3)."""
        patch = np.zeros((STAFFLINE_PATCH_SIZE, STAFFLINE_PATCH_SIZE, 3), dtype=np.uint8)
        out = staffline_model.predict_patch(patch)
        assert out.shape == (STAFFLINE_PATCH_SIZE, STAFFLINE_PATCH_SIZE, 3)

    def test_output_is_probability(self, staffline_model):
        """Each pixel's channel values must sum to ≈ 1 (softmax output)."""
        patch = np.random.randint(0, 255, (STAFFLINE_PATCH_SIZE, STAFFLINE_PATCH_SIZE, 3),
                                  dtype=np.uint8)
        out = staffline_model.predict_patch(patch)
        channel_sums = out.sum(axis=2)   # (256, 256)
        assert np.allclose(channel_sums, 1.0, atol=1e-4), \
            "Softmax channels must sum to 1.0 per pixel"

    def test_output_values_nonnegative(self, staffline_model):
        patch = np.zeros((STAFFLINE_PATCH_SIZE, STAFFLINE_PATCH_SIZE, 3), dtype=np.uint8)
        out = staffline_model.predict_patch(patch)
        assert out.min() >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 2d. StafflineModel.predict_full  (Integration — needs real model + image)
# ─────────────────────────────────────────────────────────────────────────────

class TestStafflineModelPredictFull:
    """Integration tests for StafflineModel.predict_full."""

    def test_output_shape_matches_input(self, staffline_model, real_image_path):
        """predict_full must return (H, W, 3) matching the input image."""
        img = cv2.imread(str(real_image_path))
        assert img is not None
        prob = staffline_model.predict_full(img, max_side=512)
        h, w = img.shape[:2]
        assert prob.shape == (h, w, 3), \
            f"Expected ({h}, {w}, 3), got {prob.shape}"

    def test_output_dtype(self, staffline_model, real_image_path):
        img = cv2.imread(str(real_image_path))
        prob = staffline_model.predict_full(img, max_side=512)
        assert prob.dtype == np.float32

    def test_staff_channel_activates_on_staff_region(self, staffline_model,
                                                      synthetic_music_page):
        """Channel 1 (staff) should have higher values near staff line rows."""
        prob = staffline_model.predict_full(synthetic_music_page, max_side=None)
        # Channel 1 near known staff rows (y=80..120) vs far from staff (y=300)
        staff_region = prob[78:123, :, M1_CH_STAFF].mean()
        blank_region = prob[300:350, :, M1_CH_STAFF].mean()
        assert staff_region > blank_region, \
            "Staff channel should be higher near actual staff lines"
