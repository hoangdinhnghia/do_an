
# TensorFlow / Keras is an *optional* dependency used for training only.
# At inference time the ONNX checkpoints are used directly (no TF required).
try:
    import tensorflow as tf
    import keras.layers as L
    from keras import Input, Model
    from keras.layers import (
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
