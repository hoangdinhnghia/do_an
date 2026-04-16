"""Dual-stream OMR pipeline using two ONNX models.
Stream 1 — Staffline Segmentation Model (unet_big / 1st_model.onnx)
    Input : (N, 256, 256, 3) uint8 BGR patches
    Output: (N, 256, 256, 3) float32 — softmax probabilities
            channel 0 = background
            channel 1 = staff line
            channel 2 = music symbol
    Role  : Locate the five staff lines of every staff system.
Stream 2 — Detailed Semantic Model (seg_net / 2nd_model.onnx)
    Input : (N, 288, 288, 3) uint8 BGR patches
    Output: (N, 288, 288, 4) float32 — softmax probabilities
            channel 0 = background
            channel 1 = notehead
            channel 2 = stem / beam
            channel 3 = other symbol (clef, accidental, rest…)
    Role  : Segment individual music symbols (noteheads, stems, etc.).
Public API
----------
    StafflineSegmentationModel  — wraps stream 1
    DetailedSemanticModel       — wraps stream 2
    run_dual_pipeline()         — full end-to-end dual-stream function
"""
import os
from typing import Callable, List, Optional, Tuple
import cv2
import numpy as np
import onnxruntime as ort
from .staff_detection import find_peaks_profile, group_peaks_to_staffs, refine_staff_lines
# ---------------------------------------------------------------------------
# Checkpoint paths
# ---------------------------------------------------------------------------
_CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
# ---------------------------------------------------------------------------
# Channel-index constants
# ---------------------------------------------------------------------------
_M1_CH_BG = 0
_M1_CH_STAFF = 1
_M1_CH_SYMBOL = 2
_M2_CH_BG = 0
_M2_CH_NOTEHEAD = 1
_M2_CH_STEM = 2
_M2_CH_SYMBOL = 3
# ---------------------------------------------------------------------------
# Tile-and-stitch helper
# ---------------------------------------------------------------------------
def _hann_window_2d(size: int) -> np.ndarray:
    """Return a 2-D Hann window of shape (size, size) for blending patches."""
    w1 = np.hanning(size).astype(np.float32)
    return np.outer(w1, w1)
def _tile_and_stitch(
    img_bgr: np.ndarray,
    patch_size: int,
    overlap: int,
    predict_fn: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Tile a full image into overlapping patches, run *predict_fn* on each,
    and stitch the results back with Hann-window blending.
    Args:
        img_bgr   : Input image (H, W, 3) uint8.
        patch_size: Square patch side length expected by the model.
        overlap   : Overlap between adjacent patches (pixels). Must be < patch_size.
        predict_fn: Callable that takes a (patch_size, patch_size, 3) uint8 array
                    and returns a (patch_size, patch_size, C) float32 probability map.
    Returns:
        Stitched probability map (H_orig, W_orig, C) float32 in the same spatial
        resolution as *img_bgr*.
    """
    if overlap >= patch_size:
        raise ValueError("overlap must be smaller than patch_size")
    h_orig, w_orig = img_bgr.shape[:2]
    stride = patch_size - overlap
    # Pad so that every region of the original image is fully covered.
    pad_h = max(0, patch_size - h_orig)
    pad_w = max(0, patch_size - w_orig)
    extra_h = (stride - (h_orig + pad_h - patch_size) % stride) % stride
    extra_w = (stride - (w_orig + pad_w - patch_size) % stride) % stride
    img_padded = cv2.copyMakeBorder(
        img_bgr, 0, pad_h + extra_h, 0, pad_w + extra_w, cv2.BORDER_REFLECT
    )
    H, W = img_padded.shape[:2]
    win = _hann_window_2d(patch_size)  # (patch_size, patch_size)
    # Determine output channel count from the first patch.
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
# ---------------------------------------------------------------------------
# Stream 1: Staffline Segmentation Model
# ---------------------------------------------------------------------------
class StafflineSegmentationModel:
    """Wraps the Staffline Segmentation ONNX model (unet_big / 1st_model.onnx).
    Each (256×256) BGR patch is classified at the pixel level into three
    classes: background (0), staff line (1), and music symbol (2).
    """
    PATCH_SIZE: int = 256
    STAFF_CHANNEL: int = _M1_CH_STAFF
    def __init__(self, model_path: Optional[str] = None) -> None:
        if model_path is None:
            model_path = os.path.join(_CHECKPOINT_DIR, "unet_big", "1st_model.onnx")
        self._session = ort.InferenceSession(model_path)
        self._in_name = self._session.get_inputs()[0].name
        self._out_name = self._session.get_outputs()[0].name
    def predict_patch(self, patch: np.ndarray) -> np.ndarray:
        """Run inference on a single (256, 256, 3) uint8 BGR patch.
        Returns:
            (256, 256, 3) float32 probability map.
        """
        inp = patch[np.newaxis].astype(np.uint8)
        return self._session.run([self._out_name], {self._in_name: inp})[0][0]
    def predict_full(
        self,
        img_bgr: np.ndarray,
        overlap: int = 64,
        max_side: Optional[int] = 2048,
    ) -> np.ndarray:
        """Tile-and-stitch inference on a full-resolution image.
        Args:
            img_bgr : Input image (H, W, 3) uint8.
            overlap : Tile overlap in pixels (default 64).
            max_side: Optional longest-side cap before tiling to reduce memory use.
        Returns:
            (H, W, 3) float32 probability map.
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
    def extract_staff_lines(
        self,
        prob_map: np.ndarray,
        conf_thresh: float = 0.3,
        smooth_sigma: float = 2.0,
        peak_thresh: float = 0.5,
    ) -> List[List[int]]:

        staff_mask = (prob_map[:, :, self.STAFF_CHANNEL] >= conf_thresh).astype(
            np.uint8
        )
        profile = staff_mask.sum(axis=1).astype(float)
        ys = find_peaks_profile(
            profile, smooth_sigma=smooth_sigma, thresh_ratio=peak_thresh
        )
        staffs = group_peaks_to_staffs(ys)
        staffs = refine_staff_lines(staffs, staff_mask * 255)
        return staffs
# ---------------------------------------------------------------------------
# Stream 2: Detailed Semantic Model
# ---------------------------------------------------------------------------
class DetailedSemanticModel:
    """Wraps the Detailed Semantic ONNX model (seg_net / 2nd_model.onnx).
    Each (288×288) BGR patch is classified at the pixel level into four
    classes: background (0), notehead (1), stem/beam (2), other symbol (3).
    """
    PATCH_SIZE: int = 288
    NOTEHEAD_CHANNEL: int = _M2_CH_NOTEHEAD
    STEM_CHANNEL: int = _M2_CH_STEM
    SYMBOL_CHANNEL: int = _M2_CH_SYMBOL
    def __init__(self, model_path: Optional[str] = None) -> None:
        if model_path is None:
            model_path = os.path.join(_CHECKPOINT_DIR, "seg_net", "2nd_model.onnx")
        self._session = ort.InferenceSession(model_path)
        self._in_name = self._session.get_inputs()[0].name
        self._out_name = self._session.get_outputs()[0].name
    def predict_patch(self, patch: np.ndarray) -> np.ndarray:
        """Run inference on a single (288, 288, 3) uint8 BGR patch.
        Returns:
            (288, 288, 4) float32 probability map.
        """
        inp = patch[np.newaxis].astype(np.uint8)
        return self._session.run([self._out_name], {self._in_name: inp})[0][0]
    def predict_full(
        self,
        img_bgr: np.ndarray,
        overlap: int = 64,
        max_side: Optional[int] = 2048,
    ) -> np.ndarray:
        """Tile-and-stitch inference on a full-resolution image.
        Args:
            img_bgr : Input image (H, W, 3) uint8.
            overlap : Tile overlap in pixels (default 64).
        Returns:
            (H, W, 4) float32 probability map.
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
    
    def extract_noteheads(
        self,
        prob_map: np.ndarray,
        conf_thresh: float = 0.4,
        min_area: int = 12,
        max_area: int = 2000,
        aspect_ratio: Tuple[float, float] = (0.35, 2.0),
    ) -> List[Tuple[int, int, int, int, int, int]]:

        mask = (
            prob_map[:, :, self.NOTEHEAD_CHANNEL] >= conf_thresh
        ).astype(np.uint8) * 255
        # Close small intra-blob gaps
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results: List[Tuple[int, int, int, int, int, int]] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area or area > max_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            ar = (w / h) if h else 0.0
            if ar < aspect_ratio[0] or ar > aspect_ratio[1]:
                continue
            perimeter = cv2.arcLength(c, True)
            if perimeter <= 0:
                continue
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.15:
                continue
            results.append((x, y, w, h, x + w // 2, y + h // 2))
        return results
    def extract_symbol_mask(
        self,
        prob_map: np.ndarray,
        conf_thresh: float = 0.35,
    ) -> np.ndarray:

        non_bg = prob_map[:, :, 1:].max(axis=2)
        return (non_bg >= conf_thresh).astype(np.uint8) * 255
# ---------------------------------------------------------------------------
# Dual pipeline entry point
# ---------------------------------------------------------------------------
def run_dual_pipeline(
    img_bgr: np.ndarray,
    staffline_model: Optional[StafflineSegmentationModel] = None,
    semantic_model: Optional[DetailedSemanticModel] = None,
    staff_conf_thresh: float = 0.3,
    note_conf_thresh: float = 0.4,
    overlap: int = 64,
    max_side: Optional[int] = 2048,
) -> dict:

    if staffline_model is None:
        staffline_model = StafflineSegmentationModel()
    if semantic_model is None:
        semantic_model = DetailedSemanticModel()
    # --- Stream 1: Staffline Segmentation ---
    staff_prob_map = staffline_model.predict_full(img_bgr, overlap=overlap, max_side=max_side)
    staff_lines = staffline_model.extract_staff_lines(
        staff_prob_map, conf_thresh=staff_conf_thresh
    )
    # --- Stream 2: Detailed Semantic Segmentation ---
    semantic_map = semantic_model.predict_full(img_bgr, overlap=overlap, max_side=max_side)
    noteheads = semantic_model.extract_noteheads(
        semantic_map, conf_thresh=note_conf_thresh
    )
    symbol_mask = semantic_model.extract_symbol_mask(semantic_map)
    return {
        "staff_lines": staff_lines,
        "noteheads": noteheads,
        "staff_prob_map": staff_prob_map,
        "semantic_map": semantic_map,
        "symbol_mask": symbol_mask,
    }