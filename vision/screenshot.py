"""
Save PNG frames plus ``DetectionResult`` JSON sidecars (slice 3).

Filenames: ``{timestamp}_{source}_{frame_index}.png`` / ``.json`` with a filesystem-safe timestamp.
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
    """Write screenshots only when ``present`` and min wall-clock spacing has elapsed."""

    def __init__(self, out_dir: Path, *, min_interval_sec: float) -> None:
        self.out_dir = Path(out_dir).expanduser().resolve()
        self.min_interval_sec = max(0.0, float(min_interval_sec))
        self._last_save_mono: float | None = None

    def maybe_save(
        self,
        frame_bgr: Any,
        det: DetectionResult,
        *,
        source_slug: str,
        frame_index: int,
    ) -> tuple[Path, Path] | None:
        """
        If ``det.present`` and interval allows, write PNG + JSON next to each other.

        ``frame_bgr`` is a BGR ``uint8`` array (e.g. ``Results.orig_img``).
        """
        if frame_bgr is None or not det.present:
            return None

        now = time.monotonic()
        if self._last_save_mono is not None:
            if (now - self._last_save_mono) < self.min_interval_sec:
                return None
        self._last_save_mono = now

        import cv2

        slug = slugify_source(source_slug)
        ts = filesystem_timestamp(det.timestamp_utc)
        stem = f"{ts}_{slug}_{frame_index:06d}"
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
