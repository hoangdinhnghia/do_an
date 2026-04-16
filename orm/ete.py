"""End-to-end OMR pipeline.

This module is the main entry point of the system, inspired by oemer's
``ete.py``.  It orchestrates every processing step, logs progress at each
stage, and writes annotated visualisations to an output directory.

Pipeline steps
--------------
1. Load and pre-process input image
2. Run dual-stream model inference
3. Extract staff lines
4. Remove staff lines from the binary image
5. Extract noteheads per staff
6. Save visualisation outputs

Usage
-----
::

    # Command-line
    python -m orm.ete img_test/test0.png
    python -m orm.ete img_test/test0.png -o results/
    python -m orm.ete img_test/test0.png --save-cache

    # Python API
    from orm.ete import run
    run("img_test/test0.png", output_path="out/")
"""

import argparse
import os
import pickle
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from orm import MODULE_PATH
from orm import layers
from orm.constant import (
    DEFAULT_MAX_SIDE,
    DEFAULT_OVERLAP,
    NOTE_CONF_THRESH,
    STAFF_CONF_THRESH,
    SYMBOL_CONF_THRESH,
)
from orm.exceptions import ImageLoadError
from orm.inference import StafflineModel, SemanticModel, run_inference
from orm.logger import get_logger
from orm import staffline_extraction
from orm.staff_removal import staff_removal_pipeline
from orm.notehead_detection import notehead_detection_pipeline

logger = get_logger(__name__)


# ── Helper: probability map → colourised heatmap ─────────────────────────────

def _prob_to_heatmap(prob_ch: np.ndarray) -> np.ndarray:
    """Convert a (H, W) float32 probability channel to a BGR heatmap."""
    norm = np.clip(prob_ch, 0.0, 1.0)
    gray = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)


# ── Step 1: load image ────────────────────────────────────────────────────────

def load_image(img_path: str) -> np.ndarray:
    """Load *img_path* as a BGR uint8 array.

    Raises :class:`~orm.exceptions.ImageLoadError` when the file cannot be
    read by OpenCV.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ImageLoadError(f"Cannot read image: {img_path}")
    return img


# ── Step 2: run model inference ───────────────────────────────────────────────

def generate_pred(
    img_bgr: np.ndarray,
    staffline_model: Optional[StafflineModel] = None,
    semantic_model: Optional[SemanticModel] = None,
    overlap: int = DEFAULT_OVERLAP,
    max_side: Optional[int] = DEFAULT_MAX_SIDE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run both ONNX models and register the probability maps in the layer store.

    Returns
    -------
    staff_prob   : (H, W, 3) float32
    semantic_prob: (H, W, 4) float32
    """
    staff_prob, semantic_prob = run_inference(
        img_bgr,
        staffline_model=staffline_model,
        semantic_model=semantic_model,
        overlap=overlap,
        max_side=max_side,
    )
    layers.register_layer("staff_prob_map", staff_prob)
    layers.register_layer("semantic_map", semantic_prob)
    return staff_prob, semantic_prob


# ── Main pipeline ─────────────────────────────────────────────────────────────

def extract(args: Namespace) -> str:
    """Run the full OMR pipeline described by *args*.

    Returns the path of the output directory.
    """
    img_path = Path(args.img_path)
    out_dir = Path(args.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = img_path.stem

    # ---- 1. Load image ---- #
    logger.info("Loading image: %s", img_path)
    img_bgr = load_image(str(img_path))
    logger.info("Image shape: %s", img_bgr.shape)
    layers.register_layer("original_image", img_bgr)

    # ---- 2. Model inference ---- #
    pkl_path = img_path.parent / f"{base}.pkl"
    if pkl_path.exists():
        logger.info("Loading cached predictions from %s", pkl_path)
        with open(pkl_path, "rb") as f:
            cache = pickle.load(f)
        staff_prob = cache["staff_prob"]
        semantic_prob = cache["semantic_prob"]
        layers.register_layer("staff_prob_map", staff_prob)
        layers.register_layer("semantic_map", semantic_prob)
    else:
        logger.info("Loading models")
        t0 = time.time()
        staffline_model = StafflineModel()
        semantic_model = SemanticModel()
        logger.info("Models loaded (%.1fs)", time.time() - t0)

        logger.info("Running inference")
        t1 = time.time()
        staff_prob, semantic_prob = generate_pred(
            img_bgr,
            staffline_model=staffline_model,
            semantic_model=semantic_model,
            overlap=DEFAULT_OVERLAP,
            max_side=DEFAULT_MAX_SIDE,
        )
        logger.info("Inference complete (%.1fs)", time.time() - t1)

        if args.save_cache:
            with open(pkl_path, "wb") as f:
                pickle.dump(
                    {"staff_prob": staff_prob, "semantic_prob": semantic_prob},
                    f,
                )
            logger.info("Predictions cached to %s", pkl_path)

    # ---- 3. Extract staff lines ---- #
    logger.info("Extracting staff lines")
    staff_lines: List[List[int]] = staffline_extraction.extract(
        conf_thresh=STAFF_CONF_THRESH
    )
    logger.info("Found %d staff system(s)", len(staff_lines))

    # ---- 4. Staff removal ---- #
    logger.info("Removing staff lines from binary image")
    staff_prob = layers.get_layer("staff_prob_map")
    staff_mask = (staff_prob[:, :, 1] >= STAFF_CONF_THRESH).astype(np.uint8) * 255
    img_no_staff = staff_removal_pipeline(staff_mask, staff_lines)
    layers.register_layer("img_no_staff", img_no_staff)

    # ---- 5. Extract noteheads ---- #
    logger.info("Extracting noteheads")
    notehead_results = notehead_detection_pipeline(img_no_staff, staff_lines, expand=20)
    total_notes = sum(len(r[2]) for r in notehead_results)
    logger.info(
        "Found %d notehead(s) across %d staff(s)", total_notes, len(notehead_results)
    )
    layers.register_layer("notehead_results", np.array(notehead_results, dtype=object))

    # ---- 6. Save visualisations ---- #
    logger.info("Saving outputs to %s/", out_dir)
    _save_outputs(img_bgr, staff_lines, notehead_results, staff_prob, semantic_prob, out_dir, base)

    return str(out_dir)


def _save_outputs(
    img_bgr: np.ndarray,
    staff_lines: List[List[int]],
    notehead_results: list,
    staff_prob: np.ndarray,
    semantic_prob: np.ndarray,
    out_dir: Path,
    base: str,
) -> None:
    """Write all visualisation files to *out_dir*."""

    # 1. Staff-line probability heatmap
    staff_heatmap = _prob_to_heatmap(staff_prob[:, :, 1])
    p = str(out_dir / f"{base}_staff_prob.png")
    cv2.imwrite(p, staff_heatmap)
    logger.info("✔ %s", p)

    # 2. Notehead probability heatmap
    note_heatmap = _prob_to_heatmap(semantic_prob[:, :, 1])
    p = str(out_dir / f"{base}_notehead_prob.png")
    cv2.imwrite(p, note_heatmap)
    logger.info("✔ %s", p)

    # 3. Symbol mask (all non-background from stream 2)
    non_bg = semantic_prob[:, :, 1:].max(axis=2)
    sym_mask = (non_bg >= SYMBOL_CONF_THRESH).astype(np.uint8) * 255
    p = str(out_dir / f"{base}_symbol_mask.png")
    cv2.imwrite(p, sym_mask)
    logger.info("✔ %s", p)

    # 4. Staff overlay
    img_staff_vis = img_bgr.copy()
    for staff in staff_lines:
        for y in staff:
            cv2.line(img_staff_vis, (0, y), (img_staff_vis.shape[1], y), (0, 0, 255), 2)
    p = str(out_dir / f"{base}_staff_overlay.png")
    cv2.imwrite(p, img_staff_vis)
    logger.info("✔ %s  (%d staff system(s))", p, len(staff_lines))

    # 5. Notehead overlay
    img_note_vis = img_bgr.copy()
    for _idx, _staff_y, noteheads, _ann in notehead_results:
        for (x, y, w, h, cx, cy) in noteheads:
            cv2.rectangle(img_note_vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.circle(img_note_vis, (cx, cy), 3, (255, 0, 0), -1)
    total = sum(len(r[2]) for r in notehead_results)
    p = str(out_dir / f"{base}_notehead_overlay.png")
    cv2.imwrite(p, img_note_vis)
    logger.info("✔ %s  (%d notehead(s))", p, total)

    # 6. Combined visualisation
    img_combined = img_bgr.copy()
    for staff in staff_lines:
        for y in staff:
            cv2.line(img_combined, (0, y), (img_combined.shape[1], y), (0, 0, 255), 1)
    for _idx, _staff_y, noteheads, _ann in notehead_results:
        for (x, y, w, h, cx, cy) in noteheads:
            cv2.rectangle(img_combined, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv2.circle(img_combined, (cx, cy), 3, (255, 128, 0), -1)
    p = str(out_dir / f"{base}_combined.png")
    cv2.imwrite(p, img_combined)
    logger.info("✔ %s", p)


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_parser() -> ArgumentParser:
    parser = argparse.ArgumentParser(
        "orm",
        description=(
            "End-to-end OMR pipeline.  Receives an image and writes annotated "
            "visualisation files to an output directory."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("img_path", help="Path to the input image.", type=str)
    parser.add_argument(
        "-o",
        "--output-path",
        help="Directory to write output files.",
        type=str,
        default="./out_ete",
    )
    parser.add_argument(
        "--save-cache",
        help="Cache model predictions so subsequent runs skip inference.",
        action="store_true",
    )
    return parser


def run(
    img_path: str,
    output_path: str = "./out_ete",
    save_cache: bool = False,
) -> str:
    """Programmatic entry point — equivalent to calling the CLI.

    Returns the output directory path.
    """
    layers.clear()
    args = argparse.Namespace(
        img_path=img_path,
        output_path=output_path,
        save_cache=save_cache,
    )
    return extract(args)


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()

    if not os.path.exists(args.img_path):
        raise FileNotFoundError(f"Image not found: {args.img_path}")

    layers.clear()
    out_dir = extract(args)
    logger.info("Pipeline complete.  Results in: %s", out_dir)


if __name__ == "__main__":
    main()
