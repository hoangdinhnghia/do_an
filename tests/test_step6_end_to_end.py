"""
=============================================================
BƯỚC 6 — CHẠY PIPELINE ĐẦU-ĐẾN-CUỐI (END-TO-END)
=============================================================

Mục tiêu
--------
Kiểm tra toàn bộ pipeline từ ảnh đầu vào đến kết quả cuối cùng,
đảm bảo tất cả các bước khớp với nhau đúng:

    Ảnh gốc (BGR)
        ↓ [Stream 1] StafflineModel.predict_full
    staff_prob_map (H×W×3)
        ↓ staffline_extraction.extract
    staff_lines (List[List[int]])
        ↓ staff_removal_pipeline
    img_no_staff (H×W binary)
        ↓ [Stream 2] SemanticModel.predict_full
    semantic_map (H×W×4)
        ↓ extract_noteheads / notehead_detection_pipeline
    notehead_results
        ↓ _save_outputs / visualise
    Ảnh kết quả + overlay

Kết quả được kiểm tra
---------------------
1. Layer registry chứa đúng các key sau khi chạy
2. staff_lines: số lượng staff đúng, mỗi staff có 5 dòng
3. notehead_results: mỗi staff system có kết quả, có notehead
4. Ảnh output PNG được ghi ra đĩa với kích thước hợp lệ
5. run() API trả về đúng output directory
6. --save-cache hoạt động: file .pkl được ghi ra

Cách chạy
---------
::

    cd /home/runner/work/do_an/do_an
    # Unit tests (không cần model):
    pytest tests/test_step6_end_to_end.py -v -k "Unit"

    # Integration tests (cần 2 model ONNX):
    pytest tests/test_step6_end_to_end.py -v

    # Toàn bộ test suite từ bước 1 đến 6:
    pytest tests/ -v
"""

import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from orm import layers
from orm.ete import run, load_image, generate_pred, _save_outputs
from orm.constant import STAFF_CONF_THRESH, SYMBOL_CONF_THRESH
from orm import staffline_extraction
from orm.staff_removal import staff_removal_pipeline
from orm.notehead_detection import notehead_detection_pipeline


# ─────────────────────────────────────────────────────────────────────────────
# 6a. Unit tests (synthetic data, no models required)
# ─────────────────────────────────────────────────────────────────────────────

class TestUnitLayerRegistry:
    """Ensure the layer registry correctly propagates data between steps."""

    def setup_method(self):
        layers.clear()

    def test_register_and_retrieve(self, synthetic_staff_prob_map):
        layers.register_layer("staff_prob_map", synthetic_staff_prob_map)
        retrieved = layers.get_layer("staff_prob_map")
        assert np.array_equal(retrieved, synthetic_staff_prob_map)

    def test_extract_populates_staff_lines(self, synthetic_staff_prob_map):
        layers.register_layer("staff_prob_map", synthetic_staff_prob_map)
        staffs = staffline_extraction.extract(conf_thresh=0.5)
        assert "staff_lines" in layers.list_layers()
        assert len(staffs) == 2

    def test_all_pipeline_layers_present_after_synthetic_run(
        self, synthetic_staff_prob_map, synthetic_semantic_map,
        synthetic_binary_no_staff, two_staff_lines
    ):
        """After a synthetic run all expected layer keys must be registered."""
        layers.register_layer("staff_prob_map", synthetic_staff_prob_map)
        layers.register_layer("semantic_map", synthetic_semantic_map)

        staffline_extraction.extract(conf_thresh=0.5)
        staff_prob = layers.get_layer("staff_prob_map")
        staff_mask = (staff_prob[:, :, 1] >= STAFF_CONF_THRESH).astype(np.uint8) * 255
        img_no_staff = staff_removal_pipeline(staff_mask, two_staff_lines)
        layers.register_layer("img_no_staff", img_no_staff)

        notehead_results = notehead_detection_pipeline(
            img_no_staff, two_staff_lines, expand=20
        )
        layers.register_layer(
            "notehead_results", np.array(notehead_results, dtype=object)
        )

        expected_keys = {"staff_prob_map", "semantic_map", "staff_lines",
                         "img_no_staff", "notehead_results"}
        present = set(layers.list_layers())
        missing = expected_keys - present
        assert not missing, f"Missing layer keys: {missing}"


class TestUnitSaveOutputs:
    """Tests for _save_outputs without running models."""

    def test_creates_output_files(
        self, tmp_path, synthetic_bgr_image,
        synthetic_staff_prob_map, synthetic_semantic_map
    ):
        """_save_outputs must create all expected PNG files."""
        staff_lines = [[80, 90, 100, 110, 120]]
        notehead_results = [
            (0, [80, 90, 100, 110, 120], [(100, 80, 12, 10, 106, 85)], synthetic_bgr_image)
        ]
        _save_outputs(
            synthetic_bgr_image,
            staff_lines,
            notehead_results,
            synthetic_staff_prob_map,
            synthetic_semantic_map,
            tmp_path,
            base="test_img",
        )
        expected_files = [
            "test_img_staff_prob.png",
            "test_img_notehead_prob.png",
            "test_img_symbol_mask.png",
            "test_img_staff_overlay.png",
            "test_img_notehead_overlay.png",
            "test_img_combined.png",
        ]
        for fname in expected_files:
            fpath = tmp_path / fname
            assert fpath.exists(), f"Missing output file: {fname}"
            # File must be non-empty and readable as an image
            img = cv2.imread(str(fpath))
            assert img is not None, f"Output file is not a valid image: {fname}"
            assert img.shape[0] > 0 and img.shape[1] > 0


class TestUnitLoadImage:
    """Tests for the load_image step (no model needed)."""

    def test_raises_for_missing_path(self):
        from orm.exceptions import ImageLoadError
        with pytest.raises(ImageLoadError):
            load_image("/does/not/exist.png")

    def test_loads_real_image_successfully(self, real_image_path):
        img = load_image(str(real_image_path))
        assert img is not None
        assert img.ndim == 3


# ─────────────────────────────────────────────────────────────────────────────
# 6b. Integration tests (requires both ONNX models)
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegrationFullPipeline:
    """End-to-end integration tests using real models and a real image."""

    @pytest.fixture(autouse=True)
    def _need_both_models(self, staffline_model, semantic_model):
        """Ensure both models are available (auto-skip otherwise)."""
        pass   # fixtures already skip if models are missing

    def test_run_returns_existing_directory(self, real_image_path, tmp_path):
        """orm.ete.run() must return a path to an existing directory."""
        out = run(str(real_image_path), output_path=str(tmp_path))
        assert Path(out).is_dir(), f"run() returned non-existent directory: {out}"

    def test_run_creates_all_output_files(self, real_image_path, tmp_path):
        """All 6 visualisation PNGs must be created by run()."""
        run(str(real_image_path), output_path=str(tmp_path))
        base = real_image_path.stem
        expected = [
            f"{base}_staff_prob.png",
            f"{base}_notehead_prob.png",
            f"{base}_symbol_mask.png",
            f"{base}_staff_overlay.png",
            f"{base}_notehead_overlay.png",
            f"{base}_combined.png",
        ]
        for fname in expected:
            fpath = tmp_path / fname
            assert fpath.exists(), f"Expected output file not found: {fname}"

    def test_output_images_are_valid_and_nonzero(self, real_image_path, tmp_path):
        """Every output PNG must load cleanly and have positive dimensions."""
        run(str(real_image_path), output_path=str(tmp_path))
        for png in tmp_path.glob("*.png"):
            img = cv2.imread(str(png))
            assert img is not None, f"Output PNG is corrupt: {png.name}"
            h, w = img.shape[:2]
            assert h > 0 and w > 0, f"PNG has zero dimension: {png.name}"

    def test_staff_lines_detected(self, real_image_path, tmp_path):
        """At least 1 staff system must be detected in a real sheet-music image."""
        layers.clear()
        from orm.inference import StafflineModel
        img = cv2.imread(str(real_image_path))
        m = StafflineModel()
        staff_prob = m.predict_full(img, max_side=1024)
        layers.register_layer("staff_prob_map", staff_prob)
        staffs = staffline_extraction.extract(conf_thresh=STAFF_CONF_THRESH)
        assert len(staffs) >= 1, "At least 1 staff system expected"
        for staff in staffs:
            assert len(staff) == 5, f"Each staff must have 5 lines, got {len(staff)}"

    def test_noteheads_detected(self, real_image_path, tmp_path):
        """At least 1 notehead must be detected in a real sheet-music image."""
        from orm.model_inference import DetailedSemanticModel
        img = cv2.imread(str(real_image_path))
        m = DetailedSemanticModel()
        prob = m.predict_full(img, max_side=1024)
        noteheads = m.extract_noteheads(prob)
        assert len(noteheads) >= 1, \
            f"Expected at least 1 notehead on {real_image_path.name}, got 0"

    def test_save_cache_creates_pkl(self, real_image_path, tmp_path):
        """When --save-cache is used, a .pkl file must appear next to the input."""
        import shutil
        # Copy test image into a writable tmp dir
        img_copy = tmp_path / real_image_path.name
        shutil.copy(str(real_image_path), str(img_copy))
        run(str(img_copy), output_path=str(tmp_path / "out"), save_cache=True)
        pkl = tmp_path / f"{real_image_path.stem}.pkl"
        assert pkl.exists(), f"Cache file not found: {pkl}"
        # Verify it can be loaded back
        cache = pickle.load(open(pkl, "rb"))
        assert "staff_prob" in cache
        assert "semantic_prob" in cache

    def test_run_clears_layer_registry_between_calls(self, real_image_path, tmp_path):
        """Calling run() twice must not raise 'layer already registered' errors."""
        out1_dir = tmp_path / "run1"
        out2_dir = tmp_path / "run2"
        run(str(real_image_path), output_path=str(out1_dir))
        run(str(real_image_path), output_path=str(out2_dir))  # should not raise


# ─────────────────────────────────────────────────────────────────────────────
# 6c. Command-line interface smoke test
# ─────────────────────────────────────────────────────────────────────────────

class TestCLISmoke:
    """Test that the CLI entry point works correctly."""

    def test_cli_help_exits_cleanly(self):
        """python -m orm --help must exit with code 0."""
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-m", "orm", "--help"],
            cwd=str(Path(__file__).parent.parent),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"--help exited with {result.returncode}"
        assert "img_path" in result.stdout.lower() or "img_path" in result.stderr.lower()

    def test_cli_missing_image_gives_error(self):
        """CLI must exit non-zero for a non-existent image."""
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-m", "orm", "/no/such/image.png"],
            cwd=str(Path(__file__).parent.parent),
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0, "Expected non-zero exit for missing image"
