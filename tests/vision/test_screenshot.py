"""Slice 3: screenshot writer + sidecars."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
from vision.schema import DetectionBox, DetectionResult
from vision.screenshot import ScreenshotWriter, filesystem_timestamp, slugify_source


def _det_present(*, fid: str = "0") -> DetectionResult:
    return DetectionResult(
        present=True,
        count=1,
        boxes=(DetectionBox(0, 0, 1, 1, 0.9, "train"),),
        frame_id=fid,
        source="/tmp/x.mp4",
        timestamp_utc="2026-03-25T12:00:00Z",
    )


def test_filesystem_timestamp() -> None:
    assert ":" not in filesystem_timestamp("2026-03-25T12:00:00Z")


def test_slugify_source() -> None:
    assert slugify_source("z7_seg00") == "z7_seg00"
    assert " " not in slugify_source("a b c")


def test_screenshot_writer_respects_min_interval(tmp_path) -> None:
    w = ScreenshotWriter(tmp_path, min_interval_sec=10.0)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    det = _det_present()
    with patch("time.monotonic", side_effect=[0.0, 5.0, 20.0]):
        a = w.maybe_save(frame, det, source_slug="vid", frame_index=5)
        assert a is not None
        png, js = a
        assert png.exists() and js.exists()
        assert w.maybe_save(frame, det, source_slug="vid", frame_index=6) is None
        assert w.maybe_save(frame, det, source_slug="vid", frame_index=7) is not None


def test_screenshot_writer_skips_not_present(tmp_path) -> None:
    w = ScreenshotWriter(tmp_path, min_interval_sec=0.0)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    det = DetectionResult(
        present=False,
        count=0,
        boxes=(),
        frame_id="0",
        source="x",
        timestamp_utc="2026-03-25T12:00:00Z",
    )
    assert w.maybe_save(frame, det, source_slug="v", frame_index=0) is None
