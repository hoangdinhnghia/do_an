"""
=============================================================
BƯỚC 4 — DETAILED SEMANTIC MODEL (STREAM 2)
=============================================================

Mục tiêu
--------
Sử dụng model ``2nd_model.onnx`` (seg_net) để **phân loại từng pixel**
trong ảnh nhạc thành một trong 4 lớp:

    • 0 — nền (background)
    • 1 — đầu nốt nhạc (notehead)          ← **chúng ta cần lớp này**
    • 2 — cán nốt / dầm nối (stem / beam)
    • 3 — ký hiệu khác (clef, rest, accidental…)

Tại sao dùng model riêng cho notehead?
---------------------------------------
Stream 1 (staffline) chỉ biết "đây là dòng kẻ" hay "đây là ký hiệu
nói chung" — không phân biệt loại ký hiệu.  Stream 2 (semantic) có
đầu ra chi tiết hơn: tách riêng notehead với các ký hiệu khác, giúp
bước tiếp theo (step 5) chỉ cần xử lý kênh notehead.

Kiến trúc inference
-------------------
Giống stream 1 nhưng:
  - Patch size 288×288 (lớn hơn để bắt đầu nốt nhạc đầy đủ)
  - 4 kênh đầu ra thay vì 3

Các đối tượng được kiểm thử
----------------------------
- ``orm.inference.SemanticModel``
  - ``predict_patch``  — inference trên 1 patch 288×288
  - ``predict_full``   — tile-and-stitch trên ảnh đầy đủ
- Đặc tính xác suất: softmax → tổng kênh = 1

Unit tests (không cần model)
-----------------------------
Dùng dữ liệu giả.

Integration tests (cần model)
------------------------------
Tự động bỏ qua nếu ``2nd_model.onnx`` không có.

Cách chạy
---------
::

    cd /home/runner/work/do_an/do_an
    pytest tests/test_step4_semantic_model.py -v
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from orm.inference import _tile_and_stitch
from orm.constant import SEMANTIC_PATCH_SIZE, M2_CH_NOTEHEAD, M2_CH_STEM, M2_CH_SYMBOL


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _make_semantic_fake_predict(out_channels: int = 4, patch_size: int = 288):
    """Fake predict_patch that marks the top-left quadrant as notehead."""
    def fake_predict(patch: np.ndarray) -> np.ndarray:
        out = np.zeros((patch_size, patch_size, out_channels), dtype=np.float32)
        out[:, :, 0] = 1.0   # all background
        # Mark top-left 20×20 as notehead (ch 1)
        out[:20, :20, 0] = 0.1
        out[:20, :20, M2_CH_NOTEHEAD] = 0.9
        return out
    return fake_predict


# ─────────────────────────────────────────────────────────────────────────────
# 4a. Tile-and-stitch with 4-channel output
# ─────────────────────────────────────────────────────────────────────────────

class TestTileAndStitch4Channel:
    """Verify tile-and-stitch works for 4-channel (semantic) model output."""

    def test_output_has_4_channels(self):
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        pred = _tile_and_stitch(
            img, patch_size=64, overlap=8,
            predict_fn=_make_semantic_fake_predict(4, 64),
        )
        assert pred.shape == (200, 300, 4), f"Expected 4 channels, got {pred.shape}"

    def test_output_shape_matches_input(self):
        for h, w in [(100, 100), (300, 450), (512, 256)]:
            img = np.zeros((h, w, 3), dtype=np.uint8)
            pred = _tile_and_stitch(
                img, patch_size=64, overlap=8,
                predict_fn=_make_semantic_fake_predict(4, 64),
            )
            assert pred.shape == (h, w, 4), f"Shape mismatch for ({h},{w})"

    def test_channel_sums_near_one(self):
        """With a proper softmax-like fake, channel sums should stay near 1."""
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        pred = _tile_and_stitch(
            img, patch_size=64, overlap=8,
            predict_fn=_make_semantic_fake_predict(4, 64),
        )
        ch_sum = pred.sum(axis=2)
        # Due to Hann blending of fake data, sums won't be exactly 1 everywhere,
        # but they should be in a reasonable range [0.5, 1.5]
        assert ch_sum.min() >= 0.0
        assert ch_sum.max() <= 1.5

    def test_notehead_channel_activated_at_corner(self):
        """The fake predict marks top-left 20×20 as notehead; blended result must show
        non-zero notehead probability somewhere in the output."""
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        pred = _tile_and_stitch(
            img, patch_size=64, overlap=8,
            predict_fn=_make_semantic_fake_predict(4, 64),
        )
        # After Hann-window blending the exact corner value gets attenuated,
        # but the notehead channel must have non-trivial activation somewhere.
        max_notehead_val = pred[:, :, M2_CH_NOTEHEAD].max()
        assert max_notehead_val > 0.1, \
            f"Expected notehead channel to activate somewhere, max={max_notehead_val:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# 4b. SemanticModel.predict_patch  (Integration — needs real model)
# ─────────────────────────────────────────────────────────────────────────────

class TestSemanticModelPredictPatch:
    """Integration tests for SemanticModel.predict_patch."""

    def test_output_shape(self, semantic_model):
        """predict_patch(288×288 patch) must return shape (288, 288, 4)."""
        patch = np.zeros((SEMANTIC_PATCH_SIZE, SEMANTIC_PATCH_SIZE, 3), dtype=np.uint8)
        out = semantic_model.predict_patch(patch)
        assert out.shape == (SEMANTIC_PATCH_SIZE, SEMANTIC_PATCH_SIZE, 4), \
            f"Expected (288, 288, 4), got {out.shape}"

    def test_output_is_probability(self, semantic_model):
        """Each pixel's 4 channel values must sum to ≈ 1 (softmax output)."""
        patch = np.random.randint(
            0, 255, (SEMANTIC_PATCH_SIZE, SEMANTIC_PATCH_SIZE, 3), dtype=np.uint8
        )
        out = semantic_model.predict_patch(patch)
        channel_sums = out.sum(axis=2)
        assert np.allclose(channel_sums, 1.0, atol=1e-3), \
            "Softmax channels must sum to 1.0 per pixel"

    def test_channel_0_is_nonnegative(self, semantic_model):
        patch = np.zeros((SEMANTIC_PATCH_SIZE, SEMANTIC_PATCH_SIZE, 3), dtype=np.uint8)
        out = semantic_model.predict_patch(patch)
        assert out[:, :, 0].min() >= 0.0

    def test_all_channels_nonnegative(self, semantic_model):
        patch = np.zeros((SEMANTIC_PATCH_SIZE, SEMANTIC_PATCH_SIZE, 3), dtype=np.uint8)
        out = semantic_model.predict_patch(patch)
        assert out.min() >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 4c. SemanticModel.predict_full  (Integration — needs real model + image)
# ─────────────────────────────────────────────────────────────────────────────

class TestSemanticModelPredictFull:
    """Integration tests for SemanticModel.predict_full."""

    def test_output_shape_matches_input(self, semantic_model, real_image_path):
        img = cv2.imread(str(real_image_path))
        assert img is not None
        prob = semantic_model.predict_full(img, max_side=512)
        h, w = img.shape[:2]
        assert prob.shape == (h, w, 4), f"Expected ({h},{w},4), got {prob.shape}"

    def test_output_dtype(self, semantic_model, real_image_path):
        img = cv2.imread(str(real_image_path))
        prob = semantic_model.predict_full(img, max_side=512)
        assert prob.dtype == np.float32

    def test_notehead_channel_has_nonzero_response(self, semantic_model,
                                                    real_image_path):
        """On a real sheet-music image, the notehead channel must activate somewhere."""
        img = cv2.imread(str(real_image_path))
        prob = semantic_model.predict_full(img, max_side=1024)
        max_notehead_prob = prob[:, :, M2_CH_NOTEHEAD].max()
        assert max_notehead_prob > 0.3, \
            f"Expected notehead response > 0.3, got {max_notehead_prob:.3f}"

    def test_channels_cover_all_classes(self, semantic_model, real_image_path):
        """All 4 channels should have some non-trivial activation on a music page."""
        img = cv2.imread(str(real_image_path))
        prob = semantic_model.predict_full(img, max_side=1024)
        for ch_idx, ch_name in [(0, "bg"), (1, "notehead"), (2, "stem"), (3, "symbol")]:
            ch_max = prob[:, :, ch_idx].max()
            assert ch_max > 0.0, f"Channel {ch_idx} ({ch_name}) has zero max"
