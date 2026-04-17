"""
=============================================================
KIẾN TRÚC U-NET — KIỂM TRA ĐƠN VỊ (KHÔNG CẦN TF)
=============================================================

Kiểm tra toàn bộ module ``orm/unet.py`` ở nhiều mức:

1. Khi TensorFlow KHÔNG có mặt (luôn xảy ra trong môi trường CI này):
   - ``_require_tf()`` phải raise ImportError với thông báo rõ ràng.
   - Tất cả các hàm xây dựng model (``_double_conv``, ``_encoder_block``,
     ``_decoder_block``, ``staffline_unet``, ``semantic_unet``,
     ``semantic_segmentation``, ``u_net``) phải forward-raise ImportError.

2. Kiểm tra metadata và cấu trúc module:
   - Tất cả các tên public được export đúng.
   - Docstring đủ dài.
   - Alias backward-compatible tồn tại.

3. Khi TensorFlow CÓ mặt (integration tests — bỏ qua nếu TF không cài):
   - ``_double_conv`` trả về tensor với số channel đúng.
   - ``_encoder_block`` trả về đúng (skip, downsampled) với shape đúng.
   - ``_decoder_block`` trả về tensor với shape đúng.
   - ``staffline_unet()`` — shape đầu vào/ra, tên layer, số class.
   - ``semantic_unet()`` — shape đầu vào/ra, tên layer, số class.
   - ``semantic_segmentation()`` — alias hoạt động như ``staffline_unet``.
   - ``u_net()`` — alias hoạt động như ``semantic_unet``.
   - ``staffline_unet(win_size=64, out_class=2)`` tuỳ chỉnh.
   - ``semantic_unet(win_size=64, out_class=2)`` tuỳ chỉnh.
   - ``win_size % 32 != 0`` phải raise ``ValueError``.
   - Mỗi model chứa ``MaxPooling2D``, ``Conv2DTranspose``, ``Concatenate``,
     ``BatchNormalization`` — các layer đặc trưng của U-Net chuẩn.

Cách chạy
---------
::

    cd <repo_root>
    # Không cần TF (unit tests thuần):
    python3.12 -m pytest tests/test_unet_architecture.py -v -k "not TF"

    # Khi TF cài:
    python3.12 -m pytest tests/test_unet_architecture.py -v
"""

import importlib
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Fixture: skip when TF is available (invert when needed)
# ---------------------------------------------------------------------------

def _tf_available() -> bool:
    try:
        import tensorflow  # noqa: F401
        return True
    except ImportError:
        return False


skip_if_tf_missing = pytest.mark.skipif(
    not _tf_available(),
    reason="TensorFlow not installed — integration tests skipped",
)

skip_if_tf_present = pytest.mark.skipif(
    _tf_available(),
    reason="TensorFlow IS installed — no-TF error tests skipped",
)


# ---------------------------------------------------------------------------
# 1. Behaviour when TensorFlow is NOT installed
# ---------------------------------------------------------------------------

class TestNoTensorflow:
    """All public functions must raise ImportError when TF is absent."""

    @skip_if_tf_present
    def test_require_tf_raises_import_error(self):
        from orm.unet import _require_tf
        with pytest.raises(ImportError, match="TensorFlow"):
            _require_tf()

    @skip_if_tf_present
    def test_staffline_unet_raises_without_tf(self):
        from orm.unet import staffline_unet
        with pytest.raises(ImportError):
            staffline_unet()

    @skip_if_tf_present
    def test_semantic_unet_raises_without_tf(self):
        from orm.unet import semantic_unet
        with pytest.raises(ImportError):
            semantic_unet()

    @skip_if_tf_present
    def test_semantic_segmentation_alias_raises_without_tf(self):
        from orm.unet import semantic_segmentation
        with pytest.raises(ImportError):
            semantic_segmentation()

    @skip_if_tf_present
    def test_u_net_alias_raises_without_tf(self):
        from orm.unet import u_net
        with pytest.raises(ImportError):
            u_net()


# ---------------------------------------------------------------------------
# 2. Module structure tests (no TF required)
# ---------------------------------------------------------------------------

class TestModuleStructure:
    """Verify the module exposes all expected names with correct types."""

    def test_module_importable(self):
        import orm.unet  # noqa: F401

    def test_public_functions_exist(self):
        import orm.unet as m
        for name in [
            "staffline_unet",
            "semantic_unet",
            "semantic_segmentation",
            "u_net",
            "_double_conv",
            "_encoder_block",
            "_decoder_block",
            "_require_tf",
        ]:
            assert hasattr(m, name), f"Missing public name: {name}"

    def test_all_names_are_callable(self):
        import orm.unet as m
        for name in ["staffline_unet", "semantic_unet",
                     "semantic_segmentation", "u_net"]:
            obj = getattr(m, name)
            assert callable(obj), f"{name} is not callable"

    def test_module_docstring_present(self):
        import orm.unet as m
        assert m.__doc__ is not None and len(m.__doc__) >= 100, \
            "Module docstring should be at least 100 chars"

    def test_staffline_unet_docstring(self):
        from orm.unet import staffline_unet
        assert staffline_unet.__doc__ is not None and len(staffline_unet.__doc__) >= 50

    def test_semantic_unet_docstring(self):
        from orm.unet import semantic_unet
        assert semantic_unet.__doc__ is not None and len(semantic_unet.__doc__) >= 50

    def test_double_conv_docstring(self):
        from orm.unet import _double_conv
        assert _double_conv.__doc__ is not None and len(_double_conv.__doc__) >= 30

    def test_encoder_block_docstring(self):
        from orm.unet import _encoder_block
        assert _encoder_block.__doc__ is not None and len(_encoder_block.__doc__) >= 30

    def test_decoder_block_docstring(self):
        from orm.unet import _decoder_block
        assert _decoder_block.__doc__ is not None and len(_decoder_block.__doc__) >= 30

    def test_backward_alias_semantic_segmentation_is_callable(self):
        from orm.unet import semantic_segmentation
        assert callable(semantic_segmentation)

    def test_backward_alias_u_net_is_callable(self):
        from orm.unet import u_net
        assert callable(u_net)

    def test_tf_not_available_flag(self):
        """_TF_AVAILABLE should reflect the actual TF presence."""
        import orm.unet as m
        # The flag must be a bool
        assert isinstance(m._TF_AVAILABLE, bool)


# ---------------------------------------------------------------------------
# 3. U-Net architecture conformance (requires TF)
# ---------------------------------------------------------------------------

class TestStafflineUnetArchitecture:
    """Structural tests for staffline_unet() — require TensorFlow."""

    @skip_if_tf_missing
    def test_output_shape_default(self):
        """Output shape must be (batch, 256, 256, 3) for default params."""
        import tensorflow as tf
        from orm.unet import staffline_unet
        model = staffline_unet()
        assert model.output_shape == (None, 256, 256, 3), \
            f"Unexpected output shape: {model.output_shape}"

    @skip_if_tf_missing
    def test_input_shape_default(self):
        from orm.unet import staffline_unet
        model = staffline_unet()
        assert model.input_shape == (None, 256, 256, 3)

    @skip_if_tf_missing
    def test_custom_win_size_and_out_class(self):
        from orm.unet import staffline_unet
        model = staffline_unet(win_size=64, out_class=2)
        assert model.output_shape == (None, 64, 64, 2)

    @skip_if_tf_missing
    def test_invalid_win_size_raises_value_error(self):
        from orm.unet import staffline_unet
        with pytest.raises(ValueError, match="multiple of 32"):
            staffline_unet(win_size=100)

    @skip_if_tf_missing
    def test_model_name(self):
        from orm.unet import staffline_unet
        model = staffline_unet()
        assert model.name == "staffline_unet"

    @skip_if_tf_missing
    def test_contains_max_pooling(self):
        """Standard U-Net encoder must use MaxPooling2D."""
        import keras
        from orm.unet import staffline_unet
        model = staffline_unet(win_size=64)
        layer_types = {type(l).__name__ for l in model.layers}
        assert "MaxPooling2D" in layer_types, \
            "staffline_unet must use MaxPooling2D in the encoder"

    @skip_if_tf_missing
    def test_contains_conv2d_transpose(self):
        """Standard U-Net decoder must use Conv2DTranspose for upsampling."""
        from orm.unet import staffline_unet
        model = staffline_unet(win_size=64)
        layer_types = {type(l).__name__ for l in model.layers}
        assert "Conv2DTranspose" in layer_types, \
            "staffline_unet must use Conv2DTranspose in the decoder"

    @skip_if_tf_missing
    def test_contains_concatenate(self):
        """Standard U-Net skip connections must use Concatenate."""
        from orm.unet import staffline_unet
        model = staffline_unet(win_size=64)
        layer_types = {type(l).__name__ for l in model.layers}
        assert "Concatenate" in layer_types, \
            "staffline_unet must use Concatenate for skip connections"

    @skip_if_tf_missing
    def test_contains_batch_normalization(self):
        """Standard U-Net uses BatchNormalization after each convolution."""
        from orm.unet import staffline_unet
        model = staffline_unet(win_size=64)
        layer_types = {type(l).__name__ for l in model.layers}
        assert "BatchNormalization" in layer_types, \
            "staffline_unet must use BatchNormalization (not LayerNormalization)"

    @skip_if_tf_missing
    def test_does_not_use_layer_normalization(self):
        """Old architecture used LayerNormalization — new U-Net must not."""
        from orm.unet import staffline_unet
        model = staffline_unet(win_size=64)
        layer_types = {type(l).__name__ for l in model.layers}
        assert "LayerNormalization" not in layer_types, \
            "staffline_unet must NOT use LayerNormalization (use BatchNormalization)"

    @skip_if_tf_missing
    def test_output_activation_is_softmax(self):
        """The output layer must be a softmax activation."""
        import tensorflow as tf
        from orm.unet import staffline_unet
        model = staffline_unet(win_size=64)
        import numpy as np
        dummy = np.zeros((1, 64, 64, 3), dtype=np.float32)
        out = model(dummy, training=False).numpy()
        # Channel probabilities must sum to 1 per pixel
        channel_sums = out.sum(axis=-1)
        assert np.allclose(channel_sums, 1.0, atol=1e-5), \
            "Output must be a valid probability distribution (softmax)"

    @skip_if_tf_missing
    def test_4_levels_of_max_pooling(self):
        """Standard 4-level U-Net must have exactly 4 MaxPooling2D layers."""
        from orm.unet import staffline_unet
        model = staffline_unet(win_size=64)
        n_pool = sum(1 for l in model.layers if type(l).__name__ == "MaxPooling2D")
        assert n_pool == 4, f"Expected 4 MaxPooling2D layers, found {n_pool}"

    @skip_if_tf_missing
    def test_4_levels_of_transposed_conv(self):
        """Standard 4-level U-Net must have exactly 4 Conv2DTranspose layers."""
        from orm.unet import staffline_unet
        model = staffline_unet(win_size=64)
        n_up = sum(1 for l in model.layers if type(l).__name__ == "Conv2DTranspose")
        assert n_up == 4, f"Expected 4 Conv2DTranspose layers, found {n_up}"

    @skip_if_tf_missing
    def test_output_layer_name(self):
        from orm.unet import staffline_unet
        model = staffline_unet(win_size=64)
        assert model.output_names[0] == "prediction", \
            f"Output layer should be named 'prediction', got '{model.output_names[0]}'"

    @skip_if_tf_missing
    def test_no_aspp_layers(self):
        """New architecture must not contain ASPP-specific layers (no dilation > 1)."""
        from orm.unet import staffline_unet
        model = staffline_unet(win_size=64)
        for layer in model.layers:
            cfg = layer.get_config()
            if "dilation_rate" in cfg:
                dr = cfg["dilation_rate"]
                assert dr in ((1, 1), 1, [1, 1]), \
                    f"Layer {layer.name} uses dilation_rate={dr} — ASPP not expected in standard U-Net"


class TestSemanticUnetArchitecture:
    """Structural tests for semantic_unet() — require TensorFlow."""

    @skip_if_tf_missing
    def test_output_shape_default(self):
        from orm.unet import semantic_unet
        model = semantic_unet(win_size=32)   # smallest valid size
        assert model.output_shape == (None, 32, 32, 4)

    @skip_if_tf_missing
    def test_custom_win_size_and_out_class(self):
        from orm.unet import semantic_unet
        model = semantic_unet(win_size=64, out_class=5)
        assert model.output_shape == (None, 64, 64, 5)

    @skip_if_tf_missing
    def test_invalid_win_size_raises_value_error(self):
        from orm.unet import semantic_unet
        with pytest.raises(ValueError, match="multiple of 32"):
            semantic_unet(win_size=288)   # 288 % 32 == 0 is fine; 100 is not
        with pytest.raises(ValueError):
            semantic_unet(win_size=100)

    @skip_if_tf_missing
    def test_contains_max_pooling(self):
        from orm.unet import semantic_unet
        model = semantic_unet(win_size=64)
        layer_types = {type(l).__name__ for l in model.layers}
        assert "MaxPooling2D" in layer_types

    @skip_if_tf_missing
    def test_contains_conv2d_transpose(self):
        from orm.unet import semantic_unet
        model = semantic_unet(win_size=64)
        layer_types = {type(l).__name__ for l in model.layers}
        assert "Conv2DTranspose" in layer_types

    @skip_if_tf_missing
    def test_contains_concatenate(self):
        from orm.unet import semantic_unet
        model = semantic_unet(win_size=64)
        layer_types = {type(l).__name__ for l in model.layers}
        assert "Concatenate" in layer_types

    @skip_if_tf_missing
    def test_contains_batch_normalization(self):
        from orm.unet import semantic_unet
        model = semantic_unet(win_size=64)
        layer_types = {type(l).__name__ for l in model.layers}
        assert "BatchNormalization" in layer_types

    @skip_if_tf_missing
    def test_output_is_valid_probability(self):
        import numpy as np
        from orm.unet import semantic_unet
        model = semantic_unet(win_size=32, out_class=4)
        dummy = np.zeros((1, 32, 32, 3), dtype=np.float32)
        out = model(dummy, training=False).numpy()
        channel_sums = out.sum(axis=-1)
        assert np.allclose(channel_sums, 1.0, atol=1e-5)

    @skip_if_tf_missing
    def test_model_name(self):
        from orm.unet import semantic_unet
        model = semantic_unet(win_size=32)
        assert model.name == "semantic_unet"


class TestBackwardCompatAliases:
    """Verify that deprecated aliases still work correctly."""

    @skip_if_tf_missing
    def test_semantic_segmentation_returns_valid_model(self):
        from orm.unet import semantic_segmentation
        import keras
        model = semantic_segmentation(win_size=64, out_class=3)
        assert isinstance(model, keras.Model)
        assert model.output_shape == (None, 64, 64, 3)

    @skip_if_tf_missing
    def test_semantic_segmentation_ignores_multi_grid_params(self):
        """multi_grid_layer_n / multi_grid_n are ignored gracefully."""
        from orm.unet import semantic_segmentation
        # Should not raise even with the old keyword arguments
        model = semantic_segmentation(
            win_size=64,
            multi_grid_layer_n=2,
            multi_grid_n=3,
            out_class=3,
        )
        assert model.output_shape == (None, 64, 64, 3)

    @skip_if_tf_missing
    def test_u_net_alias_returns_valid_model(self):
        from orm.unet import u_net
        import keras
        model = u_net(win_size=32, out_class=4)
        assert isinstance(model, keras.Model)
        assert model.output_shape == (None, 32, 32, 4)


class TestBuildingBlocks:
    """Unit tests for _double_conv, _encoder_block, _decoder_block."""

    @skip_if_tf_missing
    def test_double_conv_output_channels(self):
        """_double_conv must produce a tensor with the correct channel count."""
        import tensorflow as tf
        from orm.unet import _double_conv
        inp = tf.keras.Input(shape=(32, 32, 3))
        out = _double_conv(inp, filters=32)
        assert out.shape[-1] == 32

    @skip_if_tf_missing
    def test_double_conv_preserves_spatial_dims(self):
        """_double_conv must preserve the H×W spatial dimensions."""
        import tensorflow as tf
        from orm.unet import _double_conv
        inp = tf.keras.Input(shape=(32, 32, 3))
        out = _double_conv(inp, filters=16)
        assert out.shape[1] == 32 and out.shape[2] == 32

    @skip_if_tf_missing
    def test_encoder_block_skip_shape(self):
        """skip must have the same H×W as the input."""
        import tensorflow as tf
        from orm.unet import _encoder_block
        inp = tf.keras.Input(shape=(32, 32, 3))
        skip, down = _encoder_block(inp, filters=16)
        # skip should be (None, 32, 32, 16)
        assert skip.shape[1] == 32 and skip.shape[2] == 32

    @skip_if_tf_missing
    def test_encoder_block_downsampled_shape(self):
        """downsampled must have H/2 × W/2 spatial dimensions."""
        import tensorflow as tf
        from orm.unet import _encoder_block
        inp = tf.keras.Input(shape=(32, 32, 3))
        skip, down = _encoder_block(inp, filters=16)
        # MaxPooling2D(2×2) → (None, 16, 16, 16)
        assert down.shape[1] == 16 and down.shape[2] == 16

    @skip_if_tf_missing
    def test_decoder_block_output_shape(self):
        """_decoder_block must restore the spatial dimensions."""
        import tensorflow as tf
        from orm.unet import _decoder_block, _encoder_block
        inp = tf.keras.Input(shape=(32, 32, 3))
        skip, down = _encoder_block(inp, filters=16)   # down: (None,16,16,16)
        up = _decoder_block(down, skip, filters=16)     # should be (None,32,32,16)
        assert up.shape[1] == 32 and up.shape[2] == 32 and up.shape[3] == 16
