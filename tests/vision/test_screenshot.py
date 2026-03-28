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


def test_screenshot_writer_wall_only_per_slug(tmp_path) -> None:
    w = ScreenshotWriter(tmp_path, min_video_sec=0.0, min_wall_sec=10.0)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    det = _det_present()
    with patch("time.monotonic", side_effect=[0.0, 5.0, 20.0]):
        assert w.maybe_save(
            frame, det, source_slug="vid", frame_index=5, video_time_sec=0.0
        )
        assert (
            w.maybe_save(
                frame, det, source_slug="vid", frame_index=6, video_time_sec=1.0
            )
            is None
        )
        assert w.maybe_save(
            frame, det, source_slug="vid", frame_index=7, video_time_sec=2.0
        )


def test_screenshot_writer_video_spacing(tmp_path) -> None:
    w = ScreenshotWriter(tmp_path, min_video_sec=5.0, min_wall_sec=0.0)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    det = _det_present()
    with patch("time.monotonic", return_value=0.0):
        assert w.maybe_save(
            frame, det, source_slug="v", frame_index=0, video_time_sec=0.0
        )
        assert (
            w.maybe_save(
                frame, det, source_slug="v", frame_index=30, video_time_sec=2.0
            )
            is None
        )
        assert w.maybe_save(
            frame, det, source_slug="v", frame_index=200, video_time_sec=6.0
        )


def test_screenshot_writer_independent_slugs(tmp_path) -> None:
    w = ScreenshotWriter(tmp_path, min_video_sec=10.0, min_wall_sec=0.0)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    det = _det_present()
    with patch("time.monotonic", return_value=0.0):
        assert w.maybe_save(
            frame, det, source_slug="a", frame_index=0, video_time_sec=0.0
        )
        assert w.maybe_save(
            frame, det, source_slug="b", frame_index=0, video_time_sec=0.0
        )


def test_filename_includes_vidt_ms(tmp_path) -> None:
    w = ScreenshotWriter(tmp_path, min_video_sec=0.0, min_wall_sec=0.0)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    det = _det_present()
    with patch("time.monotonic", return_value=0.0):
        pair = w.maybe_save(
            frame, det, source_slug="z7_seg00", frame_index=150, video_time_sec=5.0
        )
    assert pair is not None
    png, _js = pair
    assert "vidt00005000ms" in png.name
    assert "_f000150" in png.name


def test_screenshot_writer_skips_not_present(tmp_path) -> None:
    w = ScreenshotWriter(tmp_path, min_video_sec=0.0, min_wall_sec=0.0)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    det = DetectionResult(
        present=False,
        count=0,
        boxes=(),
        frame_id="0",
        source="x",
        timestamp_utc="2026-03-25T12:00:00Z",
    )
    assert (
        w.maybe_save(frame, det, source_slug="v", frame_index=0, video_time_sec=0.0)
        is None
    )
