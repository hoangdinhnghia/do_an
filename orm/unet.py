"""U-Net architecture definitions for Optical Music Recognition (OMR).

These architectures are the backbone of the dual-stream OMR pipeline.
They are ported directly from the oemer project
(https://github.com/BreezeWhite/oemer, MIT License) and adapted for use
in this repository.

Two U-Net variants are provided:

``semantic_segmentation(win_size=256, out_class=3)``
    **Staffline Segmentation U-Net** — used as Stream 1.

    An improved U-Net with an Atrous Spatial Pyramid Pooling (ASPP) bottleneck.

    The encoder path has four downsampling stages (stride-2 convolutions).
    Skip connections from each encoder stage are concatenated into the
    corresponding decoder stage.  The bottleneck applies ASPP with five
    parallel dilated convolutions at rates 1, 2, 4, 8, 16 before upsampling.

    Output softmax classes:
        0 = background
        1 = staff line
        2 = music symbol

    Compiled ONNX checkpoint:  ``orm/checkpoints/unet_big/1st_model.onnx``

``u_net(win_size=288, out_class=4)``
    **Detailed Semantic U-Net** — used as Stream 2.

    A lightweight U-Net with depthwise separable convolutions, multi-scale
    skip concatenation, and a four-branch ASPP-style dilated bottleneck.
    Shallower than ``semantic_segmentation`` to reduce latency while still
    capturing fine-grained symbol boundaries.

    Output softmax classes:
        0 = background
        1 = notehead
        2 = stem / beam
        3 = other symbol (clef, accidental, rest, …)

    Compiled ONNX checkpoint:  ``orm/checkpoints/seg_net/2nd_model.onnx``

Inference at runtime
--------------------
At inference time the pre-compiled ONNX checkpoints are loaded via
``onnxruntime`` (see :mod:`orm.model_inference`).  The Keras/TensorFlow
definitions below are provided so that:

    * the U-Net backbone is clearly documented in source code,
    * the models can be retrained or fine-tuned when TensorFlow is available.

To build and summarise the models::

    # requires tensorflow >= 2.x
    from orm.unet import semantic_segmentation, u_net
    m1 = semantic_segmentation()   # staffline U-Net
    m1.summary()
    m2 = u_net()                   # semantic U-Net
    m2.summary()
"""

# TensorFlow / Keras is an *optional* dependency used for training only.
# At inference time the ONNX checkpoints are used directly (no TF required).
try:
    import tensorflow as tf
    import tensorflow.keras.layers as L
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import (
        Activation,
        Add,
        Concatenate,
        Conv2D,
        Conv2DTranspose,
        Dropout,
        LayerNormalization,
    )
    _TF_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TF_AVAILABLE = False


def _require_tf() -> None:
    if not _TF_AVAILABLE:
        raise ImportError(
            "TensorFlow is required to build or train the U-Net models. "
            "Install it with:  pip install tensorflow\n"
            "For inference only, use the ONNX runtime via orm.model_inference."
        )


# ---------------------------------------------------------------------------
# Building blocks shared by both U-Net variants
# ---------------------------------------------------------------------------

def _conv_block(tensor, channels, kernel_size, strides=(2, 2), dilation_rate=1, dropout=0.4):
    """Residual convolutional encoder block (used in ``semantic_segmentation``).

    Applies two Conv2D operations with LayerNorm + ReLU pre-activation and a
    residual/skip connection.  When ``strides != (1, 1)`` the skip is adjusted
    with a 1×1 strided convolution so dimensions match.
    """
    _require_tf()
    skip = tensor
    tensor = LayerNormalization()(Activation("relu")(tensor))
    tensor = Dropout(dropout)(tensor)
    tensor = Conv2D(channels, kernel_size, strides=strides, dilation_rate=dilation_rate, padding="same")(tensor)
    tensor = LayerNormalization()(Activation("relu")(tensor))
    tensor = Dropout(dropout)(tensor)
    tensor = Conv2D(channels, kernel_size, strides=(1, 1), dilation_rate=dilation_rate, padding="same")(tensor)
    if strides != (1, 1):
        skip = Conv2D(channels, (1, 1), strides=strides, padding="same")(skip)
    return Add()([tensor, skip])


def _transpose_conv_block(tensor, channels, kernel_size, strides=(2, 2), dropout=0.4):
    """Residual transposed-convolution decoder block (used in ``semantic_segmentation``).

    Mirrors ``_conv_block`` but uses ``Conv2DTranspose`` for upsampling.
    """
    _require_tf()
    skip = tensor
    tensor = LayerNormalization()(Activation("relu")(tensor))
    tensor = Dropout(dropout)(tensor)
    tensor = Conv2D(channels, kernel_size, strides=(1, 1), padding="same")(tensor)
    tensor = LayerNormalization()(Activation("relu")(tensor))
    tensor = Dropout(dropout)(tensor)
    tensor = Conv2DTranspose(channels, kernel_size, strides=strides, padding="same")(tensor)
    if strides != (1, 1):
        skip = Conv2DTranspose(channels, (1, 1), strides=strides, padding="same")(skip)
    return Add()([tensor, skip])


def _small_conv_block(tensor, channels, kernel_size=(3, 3), strides=(1, 1)):
    """Lightweight residual block with a single Conv2D (used in ``u_net``).

    Applies one Conv2D followed by ReLU + LayerNorm and adds back the
    (possibly strided-adjusted) input as a residual.
    """
    _require_tf()
    tensor = L.Conv2D(channels, kernel_size, strides=strides, padding="same", dtype=tf.float32)(tensor)
    out = L.Activation("relu")(L.LayerNormalization()(tensor))
    out = L.Dropout(0.3)(out)
    out = L.Add()([tensor, out])
    return L.Activation("relu")(L.LayerNormalization()(out))


def _trans_conv_block(tensor, channels, kernel_size=(3, 3), strides=(1, 1)):
    """Lightweight residual transposed-conv block (used in ``u_net``)."""
    _require_tf()
    tensor = L.Conv2DTranspose(channels, kernel_size, strides=strides, padding="same", dtype=tf.float32)(tensor)
    out = L.Conv2D(channels, kernel_size, padding="same", dtype=tf.float32)(tensor)
    out = L.Activation("relu")(L.LayerNormalization()(out))
    out = L.Dropout(0.3)(out)
    out = L.Add()([tensor, out])
    return L.Activation("relu")(L.LayerNormalization()(out))


# ---------------------------------------------------------------------------
# Model 1: Staffline Segmentation U-Net  (unet_big / 1st_model.onnx)
# ---------------------------------------------------------------------------

def semantic_segmentation(
    win_size: int = 256,
    multi_grid_layer_n: int = 1,
    multi_grid_n: int = 5,
    out_class: int = 3,
    dropout: float = 0.4,
):
    """Build the Staffline Segmentation U-Net.

    This is a standard U-Net encoder-decoder augmented with an Atrous Spatial
    Pyramid Pooling (ASPP) bottleneck.  The encoder compresses the input through
    four stride-2 blocks; the ASPP block aggregates multi-scale context at the
    bottleneck; the decoder symmetrically expands with skip-connection
    concatenation.

    Default configuration::

        Input   : (batch, 256, 256, 3)   — uint8 RGB / BGR patch
        Encoder : 4 × stride-2 conv blocks  →  spatial 16×16
        Bottleneck: ASPP with 5 parallel dilated convolutions
        Decoder : 4 × transpose-conv blocks with skip concat
        Output  : (batch, 256, 256, 3)   — softmax over 3 classes

    Args:
        win_size           : Square patch side length (default 256).
        multi_grid_layer_n : Number of stacked ASPP layers (default 1).
        multi_grid_n       : Number of parallel dilated branches in ASPP
                             (default 5, dilation rates 1, 2, 4, 8, 16).
        out_class          : Number of output classes (default 3).
        dropout            : Dropout rate used in all blocks (default 0.4).

    Returns:
        A ``tf.keras.Model`` with input name ``"input"`` and output name
        ``"prediction"``.

    Note:
        The pre-compiled ONNX checkpoint is
        ``orm/checkpoints/unet_big/1st_model.onnx``.  Use it for inference
        instead of rebuilding this model from scratch.
    """
    _require_tf()

    inp = Input(shape=(win_size, win_size, 3), name="input")
    en = Conv2D(2**7, (7, 7), strides=(1, 1), padding="same")(inp)

    # Encoder
    en_l1 = _conv_block(en,    2**7, (3, 3), strides=(2, 2), dropout=dropout)
    en_l1 = _conv_block(en_l1, 2**7, (3, 3), strides=(1, 1), dropout=dropout)

    en_l2 = _conv_block(en_l1, 2**7, (3, 3), strides=(2, 2), dropout=dropout)
    en_l2 = _conv_block(en_l2, 2**7, (3, 3), strides=(1, 1), dropout=dropout)
    en_l2 = _conv_block(en_l2, 2**7, (3, 3), strides=(1, 1), dropout=dropout)

    en_l3 = _conv_block(en_l2, 2**7, (3, 3), strides=(2, 2), dropout=dropout)
    en_l3 = _conv_block(en_l3, 2**7, (3, 3), strides=(1, 1), dropout=dropout)
    en_l3 = _conv_block(en_l3, 2**7, (3, 3), strides=(1, 1), dropout=dropout)
    en_l3 = _conv_block(en_l3, 2**7, (3, 3), strides=(1, 1), dropout=dropout)

    en_l4 = _conv_block(en_l3, 2**8, (3, 3), strides=(2, 2), dropout=dropout)
    en_l4 = _conv_block(en_l4, 2**8, (3, 3), strides=(1, 1), dropout=dropout)
    en_l4 = _conv_block(en_l4, 2**8, (3, 3), strides=(1, 1), dropout=dropout)
    en_l4 = _conv_block(en_l4, 2**8, (3, 3), strides=(1, 1), dropout=dropout)
    en_l4 = _conv_block(en_l4, 2**8, (3, 3), strides=(1, 1), dropout=dropout)

    # ASPP bottleneck
    feature = en_l4
    for _ in range(multi_grid_layer_n):
        feature = LayerNormalization()(Activation("relu")(feature))
        feature = Dropout(dropout)(feature)
        m = LayerNormalization()(
            Conv2D(2**9, (1, 1), strides=(1, 1), padding="same", activation="relu")(feature)
        )
        multi_grid = m
        for ii in range(multi_grid_n):
            m = LayerNormalization()(
                Conv2D(
                    2**9, (3, 3), strides=(1, 1),
                    dilation_rate=2**ii, padding="same", activation="relu",
                )(feature)
            )
            multi_grid = Concatenate()([multi_grid, m])
        multi_grid = Dropout(dropout)(multi_grid)
        feature = Conv2D(2**9, (1, 1), strides=(1, 1), padding="same")(multi_grid)

    feature = LayerNormalization()(Activation("relu")(feature))
    feature = Conv2D(2**8, (1, 1), strides=(1, 1), padding="same")(feature)
    feature = Add()([feature, en_l4])

    # Decoder with skip connections
    de_l1 = _transpose_conv_block(feature, 2**7, (3, 3), strides=(2, 2), dropout=dropout)
    skip = de_l1
    de_l1 = LayerNormalization()(Activation("relu")(de_l1))
    de_l1 = Concatenate()([de_l1, LayerNormalization()(Activation("relu")(en_l3))])
    de_l1 = Dropout(dropout)(de_l1)
    de_l1 = Conv2D(2**7, (1, 1), strides=(1, 1), padding="same")(de_l1)
    de_l1 = Add()([de_l1, skip])

    de_l2 = _transpose_conv_block(de_l1, 2**7, (3, 3), strides=(2, 2), dropout=dropout)
    skip = de_l2
    de_l2 = LayerNormalization()(Activation("relu")(de_l2))
    de_l2 = Concatenate()([de_l2, LayerNormalization()(Activation("relu")(en_l2))])
    de_l2 = Dropout(dropout)(de_l2)
    de_l2 = Conv2D(2**7, (1, 1), strides=(1, 1), padding="same")(de_l2)
    de_l2 = Add()([de_l2, skip])

    de_l3 = _transpose_conv_block(de_l2, 2**7, (3, 3), strides=(2, 2), dropout=dropout)
    skip = de_l3
    de_l3 = LayerNormalization()(Activation("relu")(de_l3))
    de_l3 = Concatenate()([de_l3, LayerNormalization()(Activation("relu")(en_l1))])
    de_l3 = Dropout(dropout)(de_l3)
    de_l3 = Conv2D(2**7, (1, 1), strides=(1, 1), padding="same")(de_l3)
    de_l3 = Add()([de_l3, skip])

    de_l4 = _transpose_conv_block(de_l3, 2**7, (3, 3), strides=(2, 2), dropout=dropout)
    de_l4 = LayerNormalization()(Activation("relu")(de_l4))
    de_l4 = Dropout(dropout)(de_l4)

    out = Conv2D(
        out_class, (1, 1), strides=(1, 1),
        activation="softmax", padding="same", name="prediction",
    )(de_l4)

    return Model(inputs=inp, outputs=out)


# ---------------------------------------------------------------------------
# Model 2: Detailed Semantic U-Net  (seg_net / 2nd_model.onnx)
# ---------------------------------------------------------------------------

def u_net(win_size: int = 288, out_class: int = 4):
    """Build the Detailed Semantic U-Net.

    A lightweight U-Net with depthwise separable convolutions and a four-branch
    ASPP-style dilated bottleneck.  The encoder uses three stages (two with
    stride-2) and accumulates multi-scale feature maps through skip
    concatenation.  The bottleneck applies four separable convolutions with
    dilation rates 1, 2, 6, 12, and fuses them via 1×1 conv.  The decoder
    reconstructs spatial resolution with transposed convolutions and skip
    concatenation from the encoder.

    Default configuration::

        Input   : (batch, 288, 288, 3)   — uint8 RGB / BGR patch
        Encoder : SeparableConv → 3 blocks (two stride-2)
        Bottleneck: 4-branch dilated ASPP (rates 1, 2, 6, 12)
        Decoder : 2 × upsampling stages with skip concat
        Output  : (batch, 288, 288, 4)   — softmax over 4 classes

    Args:
        win_size  : Square patch side length (default 288).
        out_class : Number of output classes (default 4).

    Returns:
        A ``tf.keras.Model``.

    Note:
        The pre-compiled ONNX checkpoint is
        ``orm/checkpoints/seg_net/2nd_model.onnx``.  Use it for inference
        instead of rebuilding this model from scratch.
    """
    _require_tf()

    inp = L.Input(shape=(win_size, win_size, 3))
    tensor = L.SeparableConv2D(128, (3, 3), activation="relu", padding="same")(inp)

    # Encoder
    l1 = _small_conv_block(tensor, 64,  (3, 3), strides=(2, 2))
    l1 = _small_conv_block(l1,     64,  (3, 3))
    l1 = _small_conv_block(l1,     64,  (3, 3))

    skip_l2 = _small_conv_block(l1,    128, (3, 3), strides=(2, 2))
    l2 = _small_conv_block(skip_l2,    128, (3, 3))
    l2 = _small_conv_block(l2,         128, (3, 3))
    l2 = _small_conv_block(l2,         128, (3, 3))
    l2 = _small_conv_block(l2,         128, (3, 3))
    l2 = L.Concatenate()([skip_l2, l2])

    l3 = _small_conv_block(l2, 256, (3, 3))
    l3 = _small_conv_block(l3, 256, (3, 3))
    l3 = _small_conv_block(l3, 256, (3, 3))
    l3 = _small_conv_block(l3, 256, (3, 3))
    l3 = _small_conv_block(l3, 256, (3, 3))
    l3 = L.Concatenate()([l2, l3])

    # ASPP bottleneck (stride-2 → bottleneck → stride-2 up)
    bot = _small_conv_block(l3, 256, (3, 3), strides=(2, 2))
    st1 = L.Activation("relu")(L.LayerNormalization()(
        L.SeparableConv2D(256, (3, 3), padding="same", dtype=tf.float32)(bot)
    ))
    st2 = L.Activation("relu")(L.LayerNormalization()(
        L.SeparableConv2D(256, (3, 3), dilation_rate=(2, 2), padding="same", dtype=tf.float32)(bot)
    ))
    st3 = L.Activation("relu")(L.LayerNormalization()(
        L.SeparableConv2D(256, (3, 3), dilation_rate=(6, 6), padding="same", dtype=tf.float32)(bot)
    ))
    st4 = L.Activation("relu")(L.LayerNormalization()(
        L.SeparableConv2D(256, (3, 3), dilation_rate=(12, 12), padding="same", dtype=tf.float32)(bot)
    ))
    st = L.Concatenate()([st1, st2, st3, st4])
    st = L.Conv2D(256, (1, 1), padding="same", dtype=tf.float32)(st)
    norm = L.Activation("relu")(L.LayerNormalization()(st))
    bot = _trans_conv_block(norm, 256, (3, 3), strides=(2, 2))

    # Decoder with skip connections
    tl3 = L.Conv2D(128, (3, 3), padding="same", dtype=tf.float32)(bot)
    tl3 = L.Activation("relu")(L.LayerNormalization()(tl3))
    tl3 = L.Concatenate()([tl3, l3])
    tl3 = _small_conv_block(tl3, 128, (3, 3))
    tl3 = _trans_conv_block(tl3, 128, (3, 3))

    tl2 = L.Conv2D(128, (3, 3), padding="same", dtype=tf.float32)(tl3)
    tl2 = L.Activation("relu")(L.LayerNormalization()(tl2))
    tl2 = L.Concatenate()([tl2, l2])
    tl2 = _small_conv_block(tl2, 128, (3, 3))
    tl2 = _trans_conv_block(tl2, 128, (3, 3), strides=(2, 2))

    tl1 = L.Conv2D(128, (3, 3), padding="same", dtype=tf.float32)(tl2)
    tl1 = L.Activation("relu")(L.LayerNormalization()(tl1))
    tl1 = L.Concatenate()([tl1, l1])
    tl1 = _small_conv_block(tl1, 128, (3, 3))
    tl1 = _trans_conv_block(tl1, 128, (3, 3), strides=(2, 2))

    out = L.Conv2D(
        out_class, (1, 1), activation="softmax", padding="same", dtype=tf.float32,
    )(tl1)

    return tf.keras.Model(inputs=inp, outputs=out)
