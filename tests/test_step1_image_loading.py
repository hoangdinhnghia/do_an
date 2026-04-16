"""
=============================================================
BƯỚC 1 — TẢI VÀ TIỀN XỬ LÝ ẢNH ĐẦU VÀO
=============================================================

Mục tiêu
--------
Đọc ảnh nhạc từ file, chuyển về định dạng chuẩn (BGR uint8) để
đưa vào pipeline.  Bước này **không** dùng model; nó chỉ dùng
OpenCV và các hàm preprocess đã có.

Các hàm được kiểm thử
---------------------
- ``orm.ete.load_image``            — đọc file → BGR uint8
- ``orm.preprocess.preprocess_image`` — BGR → float32 grayscale [0,1]
- ``orm.preprocess.adaptive_binarize`` — float32 grayscale → binary 0/1
- ``orm.preprocess.remove_noise``   — loại blob nhiễu nhỏ
- ``orm.exceptions.ImageLoadError``  — raise khi file không tồn tại

Cách chạy
---------
::

    cd <repo_root>
    pytest tests/test_step1_image_loading.py -v
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from orm.ete import load_image
from orm.exceptions import ImageLoadError
from orm.preprocess import adaptive_binarize, preprocess_image, remove_noise, sharpen


# ─────────────────────────────────────────────────────────────────────────────
# 1a. load_image()
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadImage:
    """Tests for ``orm.ete.load_image``."""

    def test_returns_bgr_uint8(self, real_image_path):
        """load_image must return a 3-channel uint8 ndarray."""
        img = load_image(str(real_image_path))
        assert isinstance(img, np.ndarray), "load_image should return ndarray"
        assert img.ndim == 3,               "image must be 3-dimensional (H, W, C)"
        assert img.shape[2] == 3,           "image must have 3 channels (BGR)"
        assert img.dtype == np.uint8,       "dtype must be uint8"

    def test_nonzero_dimensions(self, real_image_path):
        """Loaded image must have positive height and width."""
        img = load_image(str(real_image_path))
        h, w = img.shape[:2]
        assert h > 0 and w > 0, f"Expected positive size, got ({h}, {w})"

    def test_raises_on_missing_file(self):
        """load_image must raise ImageLoadError for a non-existent path."""
        with pytest.raises(ImageLoadError):
            load_image("/non/existent/image.png")

    def test_raises_on_invalid_extension(self, tmp_path):
        """load_image must raise ImageLoadError when file cannot be decoded as image."""
        bad_file = tmp_path / "not_an_image.png"
        bad_file.write_bytes(b"not image data at all")
        with pytest.raises(ImageLoadError):
            load_image(str(bad_file))


# ─────────────────────────────────────────────────────────────────────────────
# 1b. preprocess_image()
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessImage:
    """Tests for ``orm.preprocess.preprocess_image``."""

    def test_converts_bgr_to_float_grayscale(self, synthetic_bgr_image):
        result = preprocess_image(synthetic_bgr_image)
        assert result.ndim == 2,                 "output must be 2-D grayscale"
        assert result.dtype == np.float32,        "dtype must be float32"
        assert result.min() >= 0.0,               "values must be ≥ 0"
        assert result.max() <= 1.0,               "values must be ≤ 1"

    def test_shape_preserved(self, synthetic_bgr_image):
        h, w = synthetic_bgr_image.shape[:2]
        result = preprocess_image(synthetic_bgr_image)
        assert result.shape == (h, w), f"expected ({h},{w}), got {result.shape}"

    def test_all_white_image_becomes_ones(self):
        """A pure white BGR image should map to 1.0 after normalisation."""
        white = np.full((50, 50, 3), 255, dtype=np.uint8)
        result = preprocess_image(white)
        assert np.allclose(result, 1.0), "white image should produce all-1.0 output"

    def test_all_black_image_becomes_zeros(self):
        """A pure black BGR image should map to 0.0."""
        black = np.zeros((50, 50, 3), dtype=np.uint8)
        result = preprocess_image(black)
        assert np.allclose(result, 0.0), "black image should produce all-0.0 output"


# ─────────────────────────────────────────────────────────────────────────────
# 1c. adaptive_binarize()
# ─────────────────────────────────────────────────────────────────────────────

class TestAdaptiveBinarize:
    """Tests for ``orm.preprocess.adaptive_binarize``."""

    def test_output_is_binary(self, synthetic_bgr_image):
        gray = preprocess_image(synthetic_bgr_image)
        bw = adaptive_binarize(gray)
        unique = np.unique(bw)
        assert set(unique).issubset({0, 1}), f"Expected only 0/1, got {unique}"

    def test_output_dtype_is_uint8(self, synthetic_bgr_image):
        gray = preprocess_image(synthetic_bgr_image)
        bw = adaptive_binarize(gray)
        assert bw.dtype == np.uint8

    def test_shape_unchanged(self, synthetic_bgr_image):
        gray = preprocess_image(synthetic_bgr_image)
        bw = adaptive_binarize(gray)
        assert bw.shape == gray.shape

    def test_accepts_float32_and_uint8_input(self):
        """adaptive_binarize should work for both [0,1] float and [0,255] uint8."""
        gray_f = np.random.rand(100, 100).astype(np.float32)
        gray_u = (gray_f * 255).astype(np.uint8)
        out_f = adaptive_binarize(gray_f)
        out_u = adaptive_binarize(gray_u)
        assert out_f.shape == (100, 100)
        assert out_u.shape == (100, 100)


# ─────────────────────────────────────────────────────────────────────────────
# 1d. remove_noise()
# ─────────────────────────────────────────────────────────────────────────────

class TestRemoveNoise:
    """Tests for ``orm.preprocess.remove_noise``."""

    def test_removes_single_pixel_blobs(self):
        """Isolated single pixels should be removed."""
        img = np.zeros((50, 50), dtype=np.uint8)
        img[10, 10] = 255   # isolated pixel
        img[25, 25] = 255   # another isolated pixel
        result = remove_noise(img, min_area=5)
        assert result[10, 10] == 0, "isolated pixel should be removed"
        assert result[25, 25] == 0, "isolated pixel should be removed"

    def test_preserves_large_blobs(self):
        """Blobs larger than min_area should be preserved."""
        img = np.zeros((100, 100), dtype=np.uint8)
        img[30:40, 30:40] = 255   # 10×10 blob = 100 px
        result = remove_noise(img, min_area=10)
        assert result[35, 35] > 0, "large blob must survive remove_noise"

    def test_output_is_binary(self):
        img = np.random.randint(0, 2, (50, 50), dtype=np.uint8) * 255
        result = remove_noise(img)
        assert set(np.unique(result)).issubset({0, 1, 255})


# ─────────────────────────────────────────────────────────────────────────────
# 1e. Smoke test: full preprocess chain on a real image
# ─────────────────────────────────────────────────────────────────────────────

class TestFullPreprocessChain:
    """End-to-end smoke test for the preprocess pipeline."""

    def test_chain_produces_binary_map(self, real_image_path):
        """Complete preprocess chain must return a binary image with >0 foreground pixels."""
        img = cv2.imread(str(real_image_path))
        assert img is not None

        gray = preprocess_image(img)
        sharp = sharpen(gray)
        bw = adaptive_binarize(sharp)
        cleaned = remove_noise(bw)

        # Must be 2-D
        assert cleaned.ndim == 2

        # Must have both foreground (ink) and background pixels
        unique = np.unique(cleaned)
        assert len(unique) >= 2, "binary image must contain foreground pixels"
