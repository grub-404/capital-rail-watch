"""
Detection payload exchanged by vision inference, HTTP APIs, and screenshot sidecars.

Box coordinates (x, y, w, h) are floats in **pixel space** relative to the source frame:
top-left (x, y), width w, height h. (Slice 1 may also write normalized coords; if so, bump
schema ``version`` and document the convention here.)
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Bump when breaking the JSON shape.
SCHEMA_VERSION = 1

_ISO_UTC = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$"
)


@dataclass(frozen=True)
class DetectionBox:
    x: float
    y: float
    w: float
    h: float
    conf: float
    label: str

    def __post_init__(self) -> None:
        if self.w <= 0 or self.h <= 0:
            raise ValueError("DetectionBox w and h must be positive")
        if not 0.0 <= self.conf <= 1.0:
            raise ValueError("DetectionBox conf must be in [0, 1]")
        if not str(self.label).strip():
            raise ValueError("DetectionBox label must be non-empty")


@dataclass(frozen=True)
class DetectionResult:
    """Train-focused detection summary for one frame or still image."""

    present: bool
    count: int
    boxes: tuple[DetectionBox, ...] = field(default_factory=tuple)
    frame_id: str = ""
    source: str = ""
    timestamp_utc: str = ""
    version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError("count must be >= 0")
        if len(self.boxes) != self.count:
            raise ValueError("count must equal len(boxes)")
        if self.present != (self.count > 0):
            raise ValueError("present must be True iff count > 0")
        if self.version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema version {self.version}")
        if self.timestamp_utc and not _ISO_UTC.match(self.timestamp_utc):
            raise ValueError("timestamp_utc must be ISO 8601 with timezone or Z")
        if self.timestamp_utc:
            try:
                datetime.fromisoformat(self.timestamp_utc.replace("Z", "+00:00"))
            except ValueError as e:
                raise ValueError("timestamp_utc is not parseable as datetime") from e


def _box_from_mapping(m: Mapping[str, Any]) -> DetectionBox:
    if not isinstance(m, dict):
        raise TypeError("each box must be an object")
    return DetectionBox(
        x=float(m["x"]),
        y=float(m["y"]),
        w=float(m["w"]),
        h=float(m["h"]),
        conf=float(m["conf"]),
        label=str(m["label"]),
    )


def parse_detection_result(data: dict[str, Any]) -> DetectionResult:
    """Validate and construct a ``DetectionResult`` from a JSON-like dict."""
    if not isinstance(data, dict):
        raise TypeError("payload must be a dict")
    version = int(data.get("version", SCHEMA_VERSION))
    raw_boxes = data.get("boxes", [])
    if not isinstance(raw_boxes, list):
        raise TypeError("boxes must be a list")
    boxes = tuple(_box_from_mapping(b) for b in raw_boxes)
    count = int(data["count"])
    present = bool(data["present"])
    frame_id = str(data.get("frame_id", "") or "")
    source = str(data.get("source", "") or "")
    timestamp_utc = str(data.get("timestamp_utc", "") or "")
    return DetectionResult(
        present=present,
        count=count,
        boxes=boxes,
        frame_id=frame_id,
        source=source,
        timestamp_utc=timestamp_utc,
        version=version,
    )


def detection_result_to_json_dict(r: DetectionResult) -> dict[str, Any]:
    """Serialize for JSON APIs and sidecar files."""
    d = {
        "version": r.version,
        "present": r.present,
        "count": r.count,
        "boxes": [asdict(b) for b in r.boxes],
        "frame_id": r.frame_id,
        "source": r.source,
        "timestamp_utc": r.timestamp_utc,
    }
    return d


def load_detection_fixture(path: Path | str) -> DetectionResult:
    p = Path(path)
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return parse_detection_result(data)
