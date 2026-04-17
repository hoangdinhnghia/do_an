"""
Tests and visualisation script for U-Net-based notehead detection.

Pytest unit tests cover the core functions in orm.notehead_detection.
The script section (run directly) produces annotated output images.

Run unit tests:
    cd /home/runner/work/do_an/do_an
    python -m pytest orm/test_notehead_detection.py -v

Run visualisation script:
    cd /home/runner/work/do_an/do_an
    python orm/test_notehead_detection.py [path_to_image]

Output images (saved in out_notehead/):
    <base>_staff_prob.png          — stream-1 staff-line heatmap
    <base>_notehead_prob.png       — stream-2 notehead probability heatmap
    <base>_staff_<n>_notehead.png  — per-staff crop with notehead bounding boxes
    <base>_summary.png             — all annotated crops stacked vertically
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pytest

from orm.notehead_detection import (
    NoteheadBBox,
    _assign_noteheads_to_staves,
    annotate_noteheads,
    extract_noteheads_from_prob_map,
    notehead_detection_pipeline,
)


# ===========================================================================
# Pytest unit tests
# ===========================================================================

class TestExtractNoteheadsFromProbMap:
    """Unit tests for extract_noteheads_from_prob_map."""

    def test_empty_prob_map_returns_no_noteheads(self):
        prob_map = np.zeros((100, 200, 4), dtype=np.float32)
        result = extract_noteheads_from_prob_map(prob_map)
        assert result == []

    def test_high_prob_notehead_blob_detected(self):
        """A square blob at the notehead channel should be detected."""
        prob_map = np.zeros((100, 200, 4), dtype=np.float32)
        # 15×13 blob at channel 2, aspect ratio ≈ 0.87 — should pass
        prob_map[30:45, 50:63, 2] = 0.9
        result = extract_noteheads_from_prob_map(prob_map, conf_thresh=0.5)
        assert len(result) == 1
        x, y, w, h, cx, cy = result[0]
        assert 49 <= cx <= 64
        assert 29 <= cy <= 46

    def test_thin_vertical_blob_rejected_by_aspect_ratio(self):
        """A stem-like blob (ar ≈ 0.06) should be rejected."""
        prob_map = np.zeros((100, 200, 4), dtype=np.float32)
        prob_map[10:60, 50:53, 2] = 0.9   # 50×3 → ar ≈ 0.06
        result = extract_noteheads_from_prob_map(prob_map)
        assert result == []

    def test_low_prob_blob_not_detected(self):
        """Blob below conf_thresh should not appear."""
        prob_map = np.zeros((100, 200, 4), dtype=np.float32)
        prob_map[30:45, 50:63, 2] = 0.2   # below default threshold 0.4
        result = extract_noteheads_from_prob_map(prob_map, conf_thresh=0.4)
        assert result == []

    def test_wrong_channel_not_detected(self):
        """Blob at channel 1 (stem) must not fire when querying channel 2."""
        prob_map = np.zeros((100, 200, 4), dtype=np.float32)
        prob_map[30:45, 50:63, 1] = 0.9   # stem channel
        result = extract_noteheads_from_prob_map(prob_map, notehead_channel=2)
        assert result == []

    def test_result_tuple_has_six_fields(self):
        prob_map = np.zeros((100, 200, 4), dtype=np.float32)
        prob_map[30:45, 50:63, 2] = 0.9
        result = extract_noteheads_from_prob_map(prob_map, conf_thresh=0.5)
        assert len(result) == 1
        assert len(result[0]) == 6


class TestAssignNoteheadsToStaves:
    """Unit tests for _assign_noteheads_to_staves."""

    def test_empty_noteheads(self):
        staffs = [[10, 20, 30, 40, 50], [100, 110, 120, 130, 140]]
        result = _assign_noteheads_to_staves([], staffs)
        assert result == [[], []]

    def test_empty_staves(self):
        result = _assign_noteheads_to_staves([(0, 0, 5, 5, 2, 30)], [])
        assert result == []

    def test_single_staff_all_noteheads_assigned(self):
        staff = [10, 20, 30, 40, 50]
        noteheads = [(0, 0, 5, 5, 2, 25), (10, 0, 5, 5, 12, 35)]
        result = _assign_noteheads_to_staves(noteheads, [staff])
        assert len(result) == 1
        assert len(result[0]) == 2

    def test_two_staves_noteheads_routed_by_proximity(self):
        staff0 = [10, 20, 30, 40, 50]        # center = 30
        staff1 = [110, 120, 130, 140, 150]   # center = 130
        nh0 = (0, 0, 5, 5, 2, 25)    # cy=25 → closer to staff0
        nh1 = (0, 0, 5, 5, 2, 135)   # cy=135 → closer to staff1
        result = _assign_noteheads_to_staves([nh0, nh1], [staff0, staff1])
        assert len(result[0]) == 1
        assert len(result[1]) == 1
        assert result[0][0][5] == 25
        assert result[1][0][5] == 135

    def test_ledger_note_assigned_to_nearest_staff(self):
        staff0 = [50, 60, 70, 80, 90]       # center = 70
        staff1 = [200, 210, 220, 230, 240]  # center = 220
        nh_ledger = (0, 0, 5, 5, 2, 30)    # cy=30 — above staff0 but closer to it
        result = _assign_noteheads_to_staves([nh_ledger], [staff0, staff1])
        assert len(result[0]) == 1
        assert len(result[1]) == 0


class TestAnnotateNoteheads:
    """Basic smoke tests for annotate_noteheads."""

    def test_returns_bgr_from_bgr(self):
        img = np.zeros((50, 100, 3), dtype=np.uint8)
        out = annotate_noteheads(img, [(10, 10, 8, 8, 14, 14)])
        assert out.shape == (50, 100, 3)

    def test_returns_bgr_from_gray(self):
        img = np.zeros((50, 100), dtype=np.uint8)
        out = annotate_noteheads(img, [(10, 10, 8, 8, 14, 14)])
        assert out.ndim == 3
        assert out.shape[2] == 3

    def test_does_not_modify_input(self):
        img = np.zeros((50, 100, 3), dtype=np.uint8)
        original = img.copy()
        annotate_noteheads(img, [(10, 10, 8, 8, 14, 14)])
        np.testing.assert_array_equal(img, original)


# ===========================================================================
# Visualisation script
# ===========================================================================

def _run_script(input_path: str) -> None:
    from orm.model_inference import StafflineSegmentationModel

    OUTPUT_DIR = "out_notehead"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {input_path}")

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    print(f"[INFO] Ảnh đầu vào: {input_path}  shape={img.shape}")

    # --- Stream 1: staff-line detection ---
    print("[INFO] Đang tải mô hình…")
    staffline_model = StafflineSegmentationModel()
    staff_prob = staffline_model.predict_full(img)
    staff_lines = staffline_model.extract_staff_lines(staff_prob)
    print(f"[INFO] Phát hiện {len(staff_lines)} staff system(s)")

    # Save staff heatmap
    staff_heatmap = cv2.applyColorMap(
        (np.clip(staff_prob[:, :, 1], 0, 1) * 255).astype(np.uint8),
        cv2.COLORMAP_JET,
    )
    cv2.imwrite(f"{OUTPUT_DIR}/{base_name}_staff_prob.png", staff_heatmap)

    if not staff_lines:
        print("[WARN] Không phát hiện được staff — kiểm tra lại ảnh đầu vào.")
        return

    # --- Stream 2: notehead detection ---
    from orm.model_inference import DetailedSemanticModel

    semantic_model = DetailedSemanticModel()
    results = notehead_detection_pipeline(
        img, staff_lines, semantic_model=semantic_model
    )
    print(f"[INFO] Notehead detection hoàn tất trên {len(results)} staff(s)")

    # Save notehead probability heatmap
    sem_map = semantic_model.predict_full(img)
    note_heatmap = cv2.applyColorMap(
        (np.clip(sem_map[:, :, 2], 0, 1) * 255).astype(np.uint8),
        cv2.COLORMAP_JET,
    )
    cv2.imwrite(f"{OUTPUT_DIR}/{base_name}_notehead_prob.png", note_heatmap)

    # Save per-staff annotated crops
    annotated_crops = []
    for idx, staff_y, noteheads, annotated_crop in results:
        path_ann = f"{OUTPUT_DIR}/{base_name}_staff_{idx + 1}_notehead.png"
        cv2.imwrite(path_ann, annotated_crop)
        annotated_crops.append(annotated_crop)
        print(
            f"  Staff {idx + 1}: {len(noteheads)} notehead(s) | "
            f"staff_y=[{staff_y[0]}..{staff_y[-1]}] | {path_ann}"
        )

    # Save summary image (all crops stacked vertically)
    if annotated_crops:
        max_w = max(c.shape[1] for c in annotated_crops)
        padded = []
        for c in annotated_crops:
            if c.ndim == 2:
                c = cv2.cvtColor(c, cv2.COLOR_GRAY2BGR)
            if c.shape[1] < max_w:
                pad = np.zeros((c.shape[0], max_w - c.shape[1], 3), dtype=np.uint8)
                c = np.hstack([c, pad])
            padded.append(c)
        summary_path = f"{OUTPUT_DIR}/{base_name}_summary.png"
        cv2.imwrite(summary_path, np.vstack(padded))
        print(f"[INFO] Ảnh tổng hợp: {summary_path}")

    total = sum(len(r[2]) for r in results)
    print(f"\n✔ Hoàn tất. {total} notehead(s) trên {len(results)} staff(s).")
    print(f"  Kết quả lưu trong: {OUTPUT_DIR}/")


if __name__ == "__main__":
    _run_script(sys.argv[1] if len(sys.argv) > 1 else "img_test/test0.png")
