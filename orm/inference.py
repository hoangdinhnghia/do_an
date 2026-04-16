"""Model inference module.

Wraps both ONNX models for tile-and-stitch full-image inference.

Each model is accessed through a thin wrapper class that mirrors oemer's
``inference.py`` API while retaining the Hann-window blending already
implemented in this project's ``model_inference.py``.

Public API
----------
    StafflineModel  — stream-1 wrapper  (unet_big / 1st_model.onnx)
    SemanticModel   — stream-2 wrapper  (seg_net  / 2nd_model.onnx)
    run_inference() — convenience function that runs both streams at once
"""
import os
from typing import Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from orm import MODULE_PATH
from orm.constant import (
    STAFFLINE_PATCH_SIZE,
    SEMANTIC_PATCH_SIZE,
    DEFAULT_OVERLAP,
    DEFAULT_MAX_SIDE,
    M1_CH_STAFF,
    M2_CH_NOTEHEAD,
    M2_CH_STEM,
    M2_CH_SYMBOL,
)
from orm.exceptions import ModelNotFoundError
from orm.logger import get_logger

logger = get_logger(__name__)

# ── Default checkpoint locations ──────────────────────────────────────────────
_CHECKPOINT_DIR = os.path.join(MODULE_PATH, "checkpoints")
_STAFFLINE_MODEL_PATH = os.path.join(_CHECKPOINT_DIR, "unet_big", "1st_model.onnx")
_SEMANTIC_MODEL_PATH = os.path.join(_CHECKPOINT_DIR, "seg_net", "2nd_model.onnx")


# ── Tile-and-stitch helper ────────────────────────────────────────────────────

def _hann_window_2d(size: int) -> np.ndarray:
    """2-D Hann window of shape *(size, size)* for patch blending."""
    w1 = np.hanning(size).astype(np.float32)
    return np.outer(w1, w1)


def _tile_and_stitch(
    img_bgr: np.ndarray,
    patch_size: int,
    overlap: int,
    predict_fn,
) -> np.ndarray:
    """Tile a full image into overlapping patches, run *predict_fn* on each,
    and stitch the results back with Hann-window blending.

    Parameters
    ----------
    img_bgr:    (H, W, 3) uint8 input image.
    patch_size: Square patch side length expected by the model.
    overlap:    Overlap between adjacent patches (pixels). Must be < patch_size.
    predict_fn: Callable (patch_size, patch_size, 3) uint8 → (patch_size, patch_size, C) float32.

    Returns
    -------
    (H_orig, W_orig, C) float32 probability map.
    """
    if overlap >= patch_size:
        raise ValueError("overlap must be smaller than patch_size")

    h_orig, w_orig = img_bgr.shape[:2]
    stride = patch_size - overlap

    pad_h = max(0, patch_size - h_orig)
    pad_w = max(0, patch_size - w_orig)
    extra_h = (stride - (h_orig + pad_h - patch_size) % stride) % stride
    extra_w = (stride - (w_orig + pad_w - patch_size) % stride) % stride
    img_padded = cv2.copyMakeBorder(
        img_bgr, 0, pad_h + extra_h, 0, pad_w + extra_w, cv2.BORDER_REFLECT
    )
    H, W = img_padded.shape[:2]

    win = _hann_window_2d(patch_size)
    first_out = predict_fn(img_padded[:patch_size, :patch_size])
    out_ch = first_out.shape[2] if first_out.ndim == 3 else 1

    acc = np.zeros((H, W, out_ch), dtype=np.float64)
    wgt = np.zeros((H, W), dtype=np.float64)
    row_starts = list(range(0, H - patch_size + 1, stride))
    col_starts = list(range(0, W - patch_size + 1, stride))

    for ri, r in enumerate(row_starts):
        for ci, c in enumerate(col_starts):
            patch = img_padded[r : r + patch_size, c : c + patch_size]
            out = first_out if (ri == 0 and ci == 0) else predict_fn(patch)
            w2d = win[:, :, np.newaxis]
            acc[r : r + patch_size, c : c + patch_size] += out * w2d
            wgt[r : r + patch_size, c : c + patch_size] += win

    wgt = np.maximum(wgt, 1e-8)
    result = (acc / wgt[:, :, np.newaxis]).astype(np.float32)
    return result[:h_orig, :w_orig]


# ── Stream-1 wrapper ──────────────────────────────────────────────────────────

class StafflineModel:
    """Wraps the staffline segmentation ONNX model (stream 1).

    Input  : (N, 256, 256, 3) uint8 BGR patches
    Output : (N, 256, 256, 3) float32 — softmax probabilities
             ch 0 = background | ch 1 = staff line | ch 2 = music symbol
    """

    PATCH_SIZE: int = STAFFLINE_PATCH_SIZE
    STAFF_CHANNEL: int = M1_CH_STAFF

    def __init__(self, model_path: Optional[str] = None) -> None:
        path = model_path or _STAFFLINE_MODEL_PATH
        if not os.path.exists(path):
            raise ModelNotFoundError(
                f"Staffline model not found: {path}\n"
                "Download checkpoints and place '1st_model.onnx' in orm/checkpoints/unet_big/"
            )
        self._session = ort.InferenceSession(path)
        self._in_name = self._session.get_inputs()[0].name
        self._out_name = self._session.get_outputs()[0].name
        logger.info("Loaded staffline model: %s", path)

    def predict_patch(self, patch: np.ndarray) -> np.ndarray:
        """(256, 256, 3) uint8 → (256, 256, 3) float32 probability map."""
        inp = patch[np.newaxis].astype(np.uint8)
        return self._session.run([self._out_name], {self._in_name: inp})[0][0]

    def predict_full(
        self,
        img_bgr: np.ndarray,
        overlap: int = DEFAULT_OVERLAP,
        max_side: Optional[int] = DEFAULT_MAX_SIDE,
    ) -> np.ndarray:
        """Tile-and-stitch inference on a full-resolution image.

        Returns (H, W, 3) float32 probability map.
        """
        h_orig, w_orig = img_bgr.shape[:2]
        scale = 1.0
        if max_side is not None:
            longest = max(h_orig, w_orig)
            if longest > max_side:
                scale = max_side / longest
                new_w = max(1, int(round(w_orig * scale)))
                new_h = max(1, int(round(h_orig * scale)))
                img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        prob = _tile_and_stitch(img_bgr, self.PATCH_SIZE, overlap, self.predict_patch)
        if scale != 1.0:
            prob = cv2.resize(prob, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        return prob


# ── Stream-2 wrapper ──────────────────────────────────────────────────────────

class SemanticModel:
    """Wraps the detailed semantic segmentation ONNX model (stream 2).

    Input  : (N, 288, 288, 3) uint8 BGR patches
    Output : (N, 288, 288, 4) float32 — softmax probabilities
             ch 0 = background | ch 1 = notehead | ch 2 = stem/beam | ch 3 = other symbol
    """

    PATCH_SIZE: int = SEMANTIC_PATCH_SIZE
    NOTEHEAD_CHANNEL: int = M2_CH_NOTEHEAD
    STEM_CHANNEL: int = M2_CH_STEM
    SYMBOL_CHANNEL: int = M2_CH_SYMBOL

    def __init__(self, model_path: Optional[str] = None) -> None:
        path = model_path or _SEMANTIC_MODEL_PATH
        if not os.path.exists(path):
            raise ModelNotFoundError(
                f"Semantic model not found: {path}\n"
                "Download checkpoints and place '2nd_model.onnx' in orm/checkpoints/seg_net/"
            )
        self._session = ort.InferenceSession(path)
        self._in_name = self._session.get_inputs()[0].name
        self._out_name = self._session.get_outputs()[0].name
        logger.info("Loaded semantic model: %s", path)

    def predict_patch(self, patch: np.ndarray) -> np.ndarray:
        """(288, 288, 3) uint8 → (288, 288, 4) float32 probability map."""
        inp = patch[np.newaxis].astype(np.uint8)
        return self._session.run([self._out_name], {self._in_name: inp})[0][0]

    def predict_full(
        self,
        img_bgr: np.ndarray,
        overlap: int = DEFAULT_OVERLAP,
        max_side: Optional[int] = DEFAULT_MAX_SIDE,
    ) -> np.ndarray:
        """Tile-and-stitch inference on a full-resolution image.

        Returns (H, W, 4) float32 probability map.
        """
        h_orig, w_orig = img_bgr.shape[:2]
        scale = 1.0
        if max_side is not None:
            longest = max(h_orig, w_orig)
            if longest > max_side:
                scale = max_side / longest
                new_w = max(1, int(round(w_orig * scale)))
                new_h = max(1, int(round(h_orig * scale)))
                img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        prob = _tile_and_stitch(img_bgr, self.PATCH_SIZE, overlap, self.predict_patch)
        if scale != 1.0:
            prob = cv2.resize(prob, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        return prob


# ── Convenience function ──────────────────────────────────────────────────────

def run_inference(
    img_bgr: np.ndarray,
    staffline_model: Optional[StafflineModel] = None,
    semantic_model: Optional[SemanticModel] = None,
    overlap: int = DEFAULT_OVERLAP,
    max_side: Optional[int] = DEFAULT_MAX_SIDE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run both inference streams on *img_bgr*.

    Parameters
    ----------
    img_bgr:         (H, W, 3) uint8 input image.
    staffline_model: Pre-loaded StafflineModel (created if *None*).
    semantic_model:  Pre-loaded SemanticModel (created if *None*).
    overlap:         Tile overlap in pixels.
    max_side:        Cap longest image dimension before tiling.

    Returns
    -------
    staff_prob  : (H, W, 3) float32 — staffline probability map.
    semantic_prob: (H, W, 4) float32 — semantic probability map.
    """
    if staffline_model is None:
        staffline_model = StafflineModel()
    if semantic_model is None:
        semantic_model = SemanticModel()

    logger.info("Running stream 1 — staffline segmentation")
    staff_prob = staffline_model.predict_full(img_bgr, overlap=overlap, max_side=max_side)

    logger.info("Running stream 2 — semantic segmentation")
    semantic_prob = semantic_model.predict_full(img_bgr, overlap=overlap, max_side=max_side)

    return staff_prob, semantic_prob
