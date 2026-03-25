"""Vision env config (slice 0)."""

from __future__ import annotations

import pytest

from vision.config import load_vision_config


def test_defaults(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VISION_MODEL_PATH", raising=False)
    monkeypatch.delenv("VISION_CONF_THRESHOLD", raising=False)
    monkeypatch.delenv("VISION_SCREENSHOT_DIR", raising=False)
    monkeypatch.delenv("YTDLP_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("VISION_VIDEO_SAMPLE_INTERVAL_SEC", raising=False)
    monkeypatch.delenv("VISION_SCREENSHOT_MIN_INTERVAL_SEC", raising=False)

    c = load_vision_config(project_root=tmp_path)
    assert c.model_path is None
    assert c.conf_threshold == 0.5
    assert c.screenshot_dir == tmp_path / "data" / "vision_screenshots"
    assert c.ytdlp_output_dir == tmp_path / "data" / "clips"
    assert c.video_sample_interval_sec == 1.0
    assert c.screenshot_min_interval_sec == 10.0


def test_custom_env(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VISION_MODEL_PATH", "/models/yolo.pt")
    monkeypatch.setenv("VISION_CONF_THRESHOLD", "0.35")
    monkeypatch.setenv("VISION_SCREENSHOT_DIR", str(tmp_path / "shots"))
    monkeypatch.setenv("YTDLP_OUTPUT_DIR", str(tmp_path / "yt"))
    monkeypatch.setenv("VISION_VIDEO_SAMPLE_INTERVAL_SEC", "2.5")
    monkeypatch.setenv("VISION_SCREENSHOT_MIN_INTERVAL_SEC", "15")

    c = load_vision_config(project_root=tmp_path)
    assert c.model_path == "/models/yolo.pt"
    assert c.conf_threshold == 0.35
    assert c.screenshot_dir == tmp_path / "shots"
    assert c.ytdlp_output_dir == tmp_path / "yt"
    assert c.video_sample_interval_sec == 2.5
    assert c.screenshot_min_interval_sec == 15.0
