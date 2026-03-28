"""
Save PNG frames plus ``DetectionResult`` JSON sidecars (slice 3).

Filenames include **video timeline** (milliseconds) and frame index so passes are obvious:
``{timestamp}_{source}_vidt{ms}ms_f{frame}.png``
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from vision.schema import DetectionResult, detection_result_to_json_dict


def slugify_source(name: str, *, max_len: int = 64) -> str:
    s = re.sub(r"[^\w.\-]+", "_", str(name).strip(), flags=re.UNICODE).strip("._")
    if not s:
        s = "source"
    return s[:max_len]


def filesystem_timestamp(iso_utc: str) -> str:
    """Turn schema timestamp into something safe for filenames."""
    if not iso_utc:
        return "unknown"
    return (
        iso_utc.replace(":", "-")
        .replace("+00:00", "Z")
        .replace("/", "-")
    )


class ScreenshotWriter:
    """
    Save when ``present``, spaced in **source video time** (primary) and optionally wall clock.

    - ``min_video_sec``: minimum **seconds along the same file** since the last save for this
      slug. This spreads shots across a train pass instead of bunching at frame 0.
    - ``min_wall_sec``: optional extra cap on **real time** between saves per slug (0 = off).

    Spacing is **per ``source_slug``** (e.g. ``z7_seg00``), so each clip has its own timeline.
    """

    def __init__(
        self,
        out_dir: Path,
        *,
        min_video_sec: float = 0.0,
        min_wall_sec: float = 0.0,
    ) -> None:
        self.out_dir = Path(out_dir).expanduser().resolve()
        self.min_video_sec = max(0.0, float(min_video_sec))
        self.min_wall_sec = max(0.0, float(min_wall_sec))
        self._last_wall_mono_by_slug: dict[str, float] = {}
        self._last_video_t_by_slug: dict[str, float] = {}

    def maybe_save(
        self,
        frame_bgr: Any,
        det: DetectionResult,
        *,
        source_slug: str,
        frame_index: int,
        video_time_sec: float,
    ) -> tuple[Path, Path] | None:
        """
        ``video_time_sec`` is ``frame_index / fps`` for video; use ``0`` for still images.
        """
        if frame_bgr is None or not det.present:
            return None

        slug = slugify_source(source_slug)
        now = time.monotonic()

        if self.min_wall_sec > 0:
            lw = self._last_wall_mono_by_slug.get(slug)
            if lw is not None and (now - lw) < self.min_wall_sec:
                return None

        if self.min_video_sec > 0:
            lv = self._last_video_t_by_slug.get(slug)
            if lv is not None and (float(video_time_sec) - lv) < self.min_video_sec - 1e-6:
                return None

        self._last_wall_mono_by_slug[slug] = now
        self._last_video_t_by_slug[slug] = float(video_time_sec)

        import cv2

        ts = filesystem_timestamp(det.timestamp_utc)
        vid_ms = int(round(float(video_time_sec) * 1000))
        stem = f"{ts}_{slug}_vidt{vid_ms:08d}ms_f{frame_index:06d}"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        png_path = self.out_dir / f"{stem}.png"
        json_path = self.out_dir / f"{stem}.json"

        ok = cv2.imwrite(str(png_path), frame_bgr)
        if not ok:
            raise OSError(f"cv2.imwrite failed: {png_path}")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(detection_result_to_json_dict(det), f, indent=2)
            f.write("\n")

        return png_path, json_path
