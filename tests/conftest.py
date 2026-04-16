"""Shared pytest fixtures for the ORM test suite.

Fixtures fall into three categories:

1. **Synthetic data** — no files, no models required.  Used in unit tests for
   every processing step so the whole suite can run on any machine.

2. **Real image paths** — path to a sample image from ``img_test/``.  Tests
   that use these are still *unit* tests (they only load an image, not a model).

3. **Real model fixtures** — load the actual ONNX checkpoints.  These are
   created lazily and skipped automatically when the checkpoint files are
   absent.  Mark a test with ``@pytest.mark.real_model`` to be explicit, but
   the skip logic is handled here via ``skipif`` inside the fixture itself.
"""

from pathlib import Path
from typing import List

import numpy as np
import pytest

# ── Repository root so tests can be run from any working directory ────────────
REPO_ROOT = Path(__file__).parent.parent          # do_an/
ORM_ROOT  = REPO_ROOT / "orm"
IMG_DIR   = REPO_ROOT / "img_test"
CHECKPOINT_STAFFLINE = ORM_ROOT / "checkpoints" / "unet_big" / "1st_model.onnx"
CHECKPOINT_SEMANTIC  = ORM_ROOT / "checkpoints" / "seg_net"  / "2nd_model.onnx"


# ── Markers ───────────────────────────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_model: mark test as requiring real ONNX checkpoint files",
    )


# ── Synthetic image fixtures ──────────────────────────────────────────────────

@pytest.fixture
def synthetic_bgr_image() -> np.ndarray:
    """A small (400 × 600) white BGR image — fast to create, no file I/O."""
    img = np.full((400, 600, 3), 255, dtype=np.uint8)
    return img


@pytest.fixture
def synthetic_music_page() -> np.ndarray:
    """BGR image (600 × 1200) with 2 synthetic staff systems (black lines on white).

    Staff 1: y = [80, 90, 100, 110, 120]
    Staff 2: y = [200, 210, 220, 230, 240]
    Each line is 2 px thick.
    """
    img = np.full((400, 1200, 3), 255, dtype=np.uint8)
    for y in [80, 90, 100, 110, 120, 200, 210, 220, 230, 240]:
        img[y : y + 2, :] = 0
    # Add a few black blobs that look like noteheads (10×10 ovals)
    for cx, cy in [(300, 85), (400, 105), (500, 215), (700, 225)]:
        h, w = img.shape[:2]
        y0, y1 = max(0, cy - 5), min(h, cy + 5)
        x0, x1 = max(0, cx - 6), min(w, cx + 6)
        img[y0:y1, x0:x1] = 0
    return img


# ── Synthetic probability map fixtures ───────────────────────────────────────

@pytest.fixture
def synthetic_staff_prob_map() -> np.ndarray:
    """(400, 1200, 3) float32 probability map as if output by StafflineModel.

    Channel 0 = background, Channel 1 = staff-line, Channel 2 = symbol.
    Staff lines are placed at y = {80,90,100,110,120} and {200,210,220,230,240}.
    """
    h, w = 400, 1200
    prob = np.zeros((h, w, 3), dtype=np.float32)
    prob[:, :, 0] = 1.0   # everything is background initially
    for y in [80, 90, 100, 110, 120, 200, 210, 220, 230, 240]:
        prob[max(0, y - 1) : y + 2, :, 0] = 0.0
        prob[max(0, y - 1) : y + 2, :, 1] = 0.9   # staff channel
    return prob


@pytest.fixture
def synthetic_semantic_map() -> np.ndarray:
    """(400, 1200, 4) float32 probability map as if output by SemanticModel.

    Channel 0 = background, Channel 1 = notehead, Channel 2 = stem, Channel 3 = symbol.
    Noteheads placed at (cx, cy) = (300, 85), (400, 105), (500, 215), (700, 225).
    """
    h, w = 400, 1200
    prob = np.zeros((h, w, 4), dtype=np.float32)
    prob[:, :, 0] = 1.0
    for cx, cy in [(300, 85), (400, 105), (500, 215), (700, 225)]:
        y0, y1 = max(0, cy - 5), min(h, cy + 6)
        x0, x1 = max(0, cx - 6), min(w, cx + 7)
        prob[y0:y1, x0:x1, 0] = 0.0
        prob[y0:y1, x0:x1, 1] = 0.85   # notehead channel
    return prob


@pytest.fixture
def synthetic_binary_no_staff() -> np.ndarray:
    """(400, 1200) uint8 binary image *after* staff removal.

    Black noteheads (255) placed at approximate notehead positions.
    """
    img = np.zeros((400, 1200), dtype=np.uint8)
    for cx, cy in [(300, 85), (400, 105), (500, 215), (700, 225)]:
        h, w = img.shape
        y0, y1 = max(0, cy - 5), min(h, cy + 6)
        x0, x1 = max(0, cx - 6), min(w, cx + 7)
        img[y0:y1, x0:x1] = 255
    return img


@pytest.fixture
def two_staff_lines() -> List[List[int]]:
    """Two synthetic staff systems, each with 5 y-coordinates."""
    return [
        [80, 90, 100, 110, 120],
        [200, 210, 220, 230, 240],
    ]


# ── Real image fixture ────────────────────────────────────────────────────────

@pytest.fixture
def real_image_path() -> Path:
    """Path to the first available test image (skips if none found)."""
    for name in ["test0.png", "test1.png", "test.png"]:
        p = IMG_DIR / name
        if p.exists():
            return p
    pytest.skip("No test image found in img_test/")


# ── Real model fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def staffline_model():
    """Load the real StafflineModel; skip if checkpoint is missing."""
    if not CHECKPOINT_STAFFLINE.exists():
        pytest.skip(f"Staffline checkpoint not found: {CHECKPOINT_STAFFLINE}")
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from orm.inference import StafflineModel
    return StafflineModel()


@pytest.fixture
def semantic_model():
    """Load the real SemanticModel; skip if checkpoint is missing."""
    if not CHECKPOINT_SEMANTIC.exists():
        pytest.skip(f"Semantic checkpoint not found: {CHECKPOINT_SEMANTIC}")
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from orm.inference import SemanticModel
    return SemanticModel()
