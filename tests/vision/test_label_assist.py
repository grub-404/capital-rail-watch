"""Offline assist map generation for region grow."""

from __future__ import annotations

import pytest

cv2 = pytest.importorskip("cv2")
import numpy as np

from vision.label_assist import ASSIST_EDGE_SUFFIX, ASSIST_FG_SUFFIX, compute_assist_maps


def test_compute_assist_maps_no_background_fg_all_high() -> None:
    bgr = np.zeros((4, 6, 3), dtype=np.uint8)
    bgr[:, :] = (40, 80, 120)
    fg, edge = compute_assist_maps(bgr, None)
    assert fg.shape == (4, 6)
    assert edge.shape == (4, 6)
    assert fg.min() == 255 and fg.max() == 255
    assert edge.dtype == np.uint8


def test_assist_suffixes_match_label_app() -> None:
    assert ASSIST_FG_SUFFIX == "_assist_fg.png"
    assert ASSIST_EDGE_SUFFIX == "_assist_edge.png"
