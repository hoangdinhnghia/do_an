"""U-Net architecture definitions for staff-line and symbol segmentation.

Two model functions are exposed, both following the **standard U-Net**
architecture proposed in Ronneberger et al. (2015):

    "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    https://arxiv.org/abs/1505.04597

Standard U-Net structure (per level)
-------------------------------------
::

    Input
      │
      ▼
    ╔══════════════════════════════════════════╗
    ║  ENCODER (contracting path)              ║
    ║  Level 1: DoubleConv(64)  → MaxPool(2×2) ║ ← skip s1
    ║  Level 2: DoubleConv(128) → MaxPool(2×2) ║ ← skip s2
    ║  Level 3: DoubleConv(256) → MaxPool(2×2) ║ ← skip s3
    ║  Level 4: DoubleConv(512) → MaxPool(2×2) ║ ← skip s4
    ╚══════════════════════════════════════════╝
      │
      ▼
    ╔══════════════════════════════════════════╗
    ║  BOTTLENECK                              ║
    ║  DoubleConv(1024)                        ║
    ╚══════════════════════════════════════════╝
      │
      ▼
    ╔══════════════════════════════════════════╗
    ║  DECODER (expansive path)               ║
    ║  Level 4: Up(2×2) + Cat(s4) + DoubleConv(512)  ║
    ║  Level 3: Up(2×2) + Cat(s3) + DoubleConv(256)  ║
    ║  Level 2: Up(2×2) + Cat(s2) + DoubleConv(128)  ║
    ║  Level 1: Up(2×2) + Cat(s1) + DoubleConv(64)   ║
    ╚══════════════════════════════════════════╝
      │
      ▼
    ╔══════════════════════════════════════════╗
    ║  OUTPUT HEAD                             ║
    ║  Conv2D(out_class, 1×1) + Softmax        ║
    ╚══════════════════════════════════════════╝

Where **DoubleConv** is:
    Conv2D(C, 3×3, padding=same) → BatchNormalization → ReLU
    Conv2D(C, 3×3, padding=same) → BatchNormalization → ReLU

The key deviations from the *original* paper that are applied here:
  * Padding = "same" (not "valid") → output has the same spatial size as
    input, removing the need to crop skip tensors.
  * Batch Normalization after each convolution instead of no normalisation.
  * Dropout added to the bottleneck and optionally to encoder/decoder.
  * Input is *not* restricted to a fixed size (works with any multiple of 32).

Public API
----------
    staffline_unet(win_size, out_class, base_filters, dropout)
        Staff-line segmentation U-Net (replaces ``semantic_segmentation``).

    semantic_unet(win_size, out_class, base_filters, dropout)
        Detailed symbol segmentation U-Net (replaces ``u_net``).

    semantic_segmentation(...)   — compatibility alias for staffline_unet
    u_net(...)                   — compatibility alias for semantic_unet

TensorFlow / Keras is an *optional* dependency required only for training.
At inference time the ONNX checkpoints are used directly (no TF required).
"""

# TensorFlow / Keras is an *optional* dependency used for training only.
# At inference time the ONNX checkpoints are used directly (no TF required).
try:
    import tensorflow as tf
    from keras import Input, Model
    import keras.layers as L
    _TF_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TF_AVAILABLE = False


def _require_tf() -> None:
    if not _TF_AVAILABLE:
        raise ImportError(
            "TensorFlow / Keras is required to build or train the U-Net models. "
            "Install it with:  pip install tensorflow\n"
            "For inference only, use the ONNX runtime via orm.inference."
        )


# ---------------------------------------------------------------------------
# Standard U-Net building blocks
# ---------------------------------------------------------------------------

def _double_conv(tensor, filters: int, dropout: float = 0.0):
    """Standard U-Net double-convolution block.

    Applies two consecutive ``Conv2D(filters, 3×3, same) → BN → ReLU``
    sequences.  Dropout (spatial) is inserted between the two convolutions
    when ``dropout > 0``.

    This block is the fundamental building block of the U-Net and is used at
    every level of both the encoder and the decoder.

    Parameters
    ----------
    tensor:  Input feature-map tensor.
    filters: Number of output feature maps (channels).
    dropout: Spatial dropout rate applied between the two convolutions.
             0 = no dropout.

    Returns
    -------
    Output feature-map tensor with shape (..., H, W, filters).
    """
    _require_tf()
    # First convolution
    x = L.Conv2D(filters, (3, 3), padding="same", use_bias=False)(tensor)
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)
    # Optional spatial dropout between the two convolutions
    if dropout > 0.0:
        x = L.SpatialDropout2D(dropout)(x)
    # Second convolution
    x = L.Conv2D(filters, (3, 3), padding="same", use_bias=False)(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)
    return x


def _encoder_block(tensor, filters: int, dropout: float = 0.0):
    """One level of the U-Net contracting (encoder) path.

    Applies ``_double_conv`` then ``MaxPooling2D(2×2)`` for downsampling.

    Returns
    -------
    (skip, downsampled)
        ``skip``        — feature map *before* pooling (used as skip connection)
        ``downsampled`` — feature map *after* pooling (passed to next level)
    """
    _require_tf()
    skip = _double_conv(tensor, filters, dropout=dropout)
    downsampled = L.MaxPooling2D((2, 2))(skip)
    return skip, downsampled


def _decoder_block(tensor, skip, filters: int, dropout: float = 0.0):
    """One level of the U-Net expansive (decoder) path.

    1. ``Conv2DTranspose(filters, 2×2, strides=2)`` — upsample by 2×.
    2. ``Concatenate([upsampled, skip])`` — reintroduce encoder detail.
    3. ``_double_conv(filters)`` — fuse features.

    Parameters
    ----------
    tensor: Feature map from the previous (deeper) decoder level.
    skip:   Skip-connection feature map from the corresponding encoder level.
    filters: Number of output channels after the double-conv.
    dropout: Dropout rate inside the double-conv.

    Returns
    -------
    Output feature-map tensor with shape (..., H, W, filters).
    """
    _require_tf()
    # Up-convolution (transposed convolution for learnable upsampling)
    x = L.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding="same")(tensor)
    # Skip connection — concatenate along the channel axis
    x = L.Concatenate()([x, skip])
    # Double conv to fuse encoder + decoder features
    x = _double_conv(x, filters, dropout=dropout)
    return x


# ---------------------------------------------------------------------------
# Model 1: Staffline Segmentation U-Net  (unet_big / 1st_model.onnx)
# ---------------------------------------------------------------------------

def staffline_unet(
    win_size: int = 256,
    out_class: int = 3,
    base_filters: int = 64,
    dropout: float = 0.3,
) -> "Model":
    """Standard U-Net for staffline segmentation (stream 1).

    Classifies each pixel into one of *out_class* classes:
        0 — background
        1 — staff line
        2 — music symbol (other foreground)

    Architecture
    ------------
    ::

        Input (win_size × win_size × 3)
          │
          ├─ Encoder L1:  DoubleConv(base_filters)       → MaxPool  → skip1
          ├─ Encoder L2:  DoubleConv(base_filters×2)     → MaxPool  → skip2
          ├─ Encoder L3:  DoubleConv(base_filters×4)     → MaxPool  → skip3
          ├─ Encoder L4:  DoubleConv(base_filters×8)     → MaxPool  → skip4
          │
          ├─ Bottleneck:  DoubleConv(base_filters×16) + SpatialDropout
          │
          ├─ Decoder L4:  Up + Cat(skip4) + DoubleConv(base_filters×8)
          ├─ Decoder L3:  Up + Cat(skip3) + DoubleConv(base_filters×4)
          ├─ Decoder L2:  Up + Cat(skip2) + DoubleConv(base_filters×2)
          ├─ Decoder L1:  Up + Cat(skip1) + DoubleConv(base_filters)
          │
          └─ Output:      Conv2D(out_class, 1×1) + Softmax

    Parameters
    ----------
    win_size:     Input patch size (width = height). Must be a multiple of 32
                  so that the 4× max-pooling can be reversed exactly.
    out_class:    Number of output segmentation classes (default 3).
    base_filters: Number of feature maps in the shallowest encoder level
                  (default 64).  Each deeper level doubles the count.
    dropout:      SpatialDropout2D rate applied between the two convolutions
                  in every double-conv block (0 = disabled).

    Returns
    -------
    A compiled-ready ``keras.Model`` instance.
    """
    _require_tf()

    if win_size % 32 != 0:
        raise ValueError(
            f"win_size must be a multiple of 32 (got {win_size}). "
            "The 4× max-pooling requires the spatial dimensions to be "
            "divisible by 16; a multiple of 32 provides a safe margin."
        )

    f1  = base_filters          #  64
    f2  = base_filters * 2      # 128
    f3  = base_filters * 4      # 256
    f4  = base_filters * 8      # 512
    f_b = base_filters * 16     # 1024  (bottleneck)

    inp = Input(shape=(win_size, win_size, 3), name="input")

    # ── Encoder (contracting path) ──────────────────────────────────────────
    skip1, x = _encoder_block(inp, f1, dropout=dropout)
    skip2, x = _encoder_block(x,   f2, dropout=dropout)
    skip3, x = _encoder_block(x,   f3, dropout=dropout)
    skip4, x = _encoder_block(x,   f4, dropout=dropout)

    # ── Bottleneck ───────────────────────────────────────────────────────────
    # Increased dropout at the bottleneck to prevent over-fitting on the most
    # abstract representations.
    x = _double_conv(x, f_b, dropout=min(dropout + 0.1, 0.5))

    # ── Decoder (expansive path) ─────────────────────────────────────────────
    x = _decoder_block(x, skip4, f4, dropout=dropout)
    x = _decoder_block(x, skip3, f3, dropout=dropout)
    x = _decoder_block(x, skip2, f2, dropout=dropout)
    x = _decoder_block(x, skip1, f1, dropout=dropout)

    # ── Output head ──────────────────────────────────────────────────────────
    out = L.Conv2D(
        out_class, (1, 1),
        activation="softmax",
        padding="same",
        name="prediction",
    )(x)

    return Model(inputs=inp, outputs=out, name="staffline_unet")


# ---------------------------------------------------------------------------
# Model 2: Detailed Semantic Segmentation U-Net  (seg_net / 2nd_model.onnx)
# ---------------------------------------------------------------------------

def semantic_unet(
    win_size: int = 288,
    out_class: int = 4,
    base_filters: int = 32,
    dropout: float = 0.25,
) -> "Model":
    """Standard U-Net for detailed symbol segmentation (stream 2).

    Classifies each pixel into one of *out_class* classes:
        0 — background
        1 — notehead
        2 — stem / beam
        3 — other symbol (clef, accidental, rest, …)

    Architecture
    ------------
    Identical U-Net structure to ``staffline_unet`` but with a smaller
    *base_filters* (32 instead of 64) to keep the parameter count
    manageable when run at inference time on 288×288 patches.

    ::

        Input (win_size × win_size × 3)
          │
          ├─ Encoder L1:  DoubleConv(32)   → MaxPool  → skip1
          ├─ Encoder L2:  DoubleConv(64)   → MaxPool  → skip2
          ├─ Encoder L3:  DoubleConv(128)  → MaxPool  → skip3
          ├─ Encoder L4:  DoubleConv(256)  → MaxPool  → skip4
          │
          ├─ Bottleneck:  DoubleConv(512) + SpatialDropout
          │
          ├─ Decoder L4:  Up + Cat(skip4) + DoubleConv(256)
          ├─ Decoder L3:  Up + Cat(skip3) + DoubleConv(128)
          ├─ Decoder L2:  Up + Cat(skip2) + DoubleConv(64)
          ├─ Decoder L1:  Up + Cat(skip1) + DoubleConv(32)
          │
          └─ Output:      Conv2D(out_class, 1×1) + Softmax

    Parameters
    ----------
    win_size:     Input patch size. Must be a multiple of 32.
    out_class:    Number of output classes (default 4).
    base_filters: Feature maps at the shallowest level (default 32).
    dropout:      SpatialDropout2D rate (default 0.25).

    Returns
    -------
    A compiled-ready ``keras.Model`` instance.
    """
    _require_tf()

    if win_size % 32 != 0:
        raise ValueError(
            f"win_size must be a multiple of 32 (got {win_size}). "
        )

    f1  = base_filters          #  32
    f2  = base_filters * 2      #  64
    f3  = base_filters * 4      # 128
    f4  = base_filters * 8      # 256
    f_b = base_filters * 16     # 512  (bottleneck)

    inp = Input(shape=(win_size, win_size, 3), name="input")

    # ── Encoder (contracting path) ──────────────────────────────────────────
    skip1, x = _encoder_block(inp, f1, dropout=dropout)
    skip2, x = _encoder_block(x,   f2, dropout=dropout)
    skip3, x = _encoder_block(x,   f3, dropout=dropout)
    skip4, x = _encoder_block(x,   f4, dropout=dropout)

    # ── Bottleneck ───────────────────────────────────────────────────────────
    x = _double_conv(x, f_b, dropout=min(dropout + 0.1, 0.5))

    # ── Decoder (expansive path) ─────────────────────────────────────────────
    x = _decoder_block(x, skip4, f4, dropout=dropout)
    x = _decoder_block(x, skip3, f3, dropout=dropout)
    x = _decoder_block(x, skip2, f2, dropout=dropout)
    x = _decoder_block(x, skip1, f1, dropout=dropout)

    # ── Output head ──────────────────────────────────────────────────────────
    out = L.Conv2D(
        out_class, (1, 1),
        activation="softmax",
        padding="same",
        name="prediction",
    )(x)

    return Model(inputs=inp, outputs=out, name="semantic_unet")


# ---------------------------------------------------------------------------
# Backward-compatibility aliases
# ---------------------------------------------------------------------------
# The original names ``semantic_segmentation`` and ``u_net`` are kept so that
# any training scripts that imported them continue to work.  They now delegate
# to the new standard-U-Net implementations.
#
# Call signature changes:
#   semantic_segmentation used to accept (win_size, multi_grid_layer_n,
#     multi_grid_n, out_class, dropout) — the multi_grid_* parameters are
#     ignored because the ASPP bottleneck has been replaced by the standard
#     double-conv bottleneck.
#   u_net used to accept (win_size, out_class) — still works as before.

def semantic_segmentation(
    win_size: int = 256,
    multi_grid_layer_n: int = 1,   # ignored (kept for API compatibility)
    multi_grid_n: int = 5,          # ignored (kept for API compatibility)
    out_class: int = 3,
    dropout: float = 0.3,
):
    """Backward-compatible alias for :func:`staffline_unet`.

    The *multi_grid_layer_n* and *multi_grid_n* parameters that controlled the
    ASPP bottleneck are accepted but **ignored** — the architecture now uses the
    standard U-Net double-conv bottleneck instead of ASPP.

    .. deprecated::
        Use :func:`staffline_unet` for new code.
    """
    return staffline_unet(win_size=win_size, out_class=out_class, dropout=dropout)


def u_net(win_size: int = 288, out_class: int = 4):
    """Backward-compatible alias for :func:`semantic_unet`.

    .. deprecated::
        Use :func:`semantic_unet` for new code.
    """
    return semantic_unet(win_size=win_size, out_class=out_class)

