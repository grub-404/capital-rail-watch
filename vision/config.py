"""Load vision-related settings from environment (optional ``python-dotenv``)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Repository root (parent of ``vision/``).
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class VisionConfig:
    """Paths and thresholds for inference, screenshots, and yt-dlp corpus."""

    model_path: str | None
    conf_threshold: float
    screenshot_dir: Path
    ytdlp_output_dir: Path
    # Video: approximate time between YOLO runs (uses fps × interval → Ultralytics vid_stride).
    video_sample_interval_sec: float
    # Min real time between screenshot PNG+JSON writes while train is present.
    screenshot_min_interval_sec: float


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    return float(raw)


def _env_path(key: str, default: Path) -> Path:
    raw = os.environ.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    return Path(raw).expanduser()


def load_vision_config(*, project_root: Path | None = None) -> VisionConfig:
    """
    Read ``VISION_*`` and ``YTDLP_OUTPUT_DIR`` from the environment.

    If ``python-dotenv`` is installed, loads ``.env`` from ``project_root`` first
    (same layout as ``backend/server.py``).
    """
    root = project_root if project_root is not None else PROJECT_ROOT
    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env")
    except ImportError:
        pass

    model_raw = os.environ.get("VISION_MODEL_PATH", "").strip()
    return VisionConfig(
        model_path=model_raw or None,
        conf_threshold=_env_float("VISION_CONF_THRESHOLD", 0.5),
        screenshot_dir=_env_path("VISION_SCREENSHOT_DIR", root / "data" / "vision_screenshots"),
        ytdlp_output_dir=_env_path("YTDLP_OUTPUT_DIR", root / "data" / "clips"),
        video_sample_interval_sec=_env_float("VISION_VIDEO_SAMPLE_INTERVAL_SEC", 1.0),
        screenshot_min_interval_sec=_env_float("VISION_SCREENSHOT_MIN_INTERVAL_SEC", 10.0),
    )
