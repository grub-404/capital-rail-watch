"""
Run Ultralytics YOLO on images or video; map raw class names to train-like ``DetectionResult``.

Default weights ``yolov8n.pt`` are COCO-pretrained; COCO includes a **train** class, which is
enough for slice-1 stills. Swap ``VISION_MODEL_PATH`` for a custom checkpoint when needed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from vision.labels import is_train_like_label
from vision.schema import DetectionBox, DetectionResult

# Shipped with Ultralytics; downloads on first use if not cached.
DEFAULT_YOLO_WEIGHTS = "yolov8n.pt"

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
# yt-dlp / browsers often use these while a file is still growing or fragmented.
_INCOMPLETE_SUFFIXES = frozenset({".part", ".ytdl", ".tmp", ".temp"})


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def yolo_result_to_detection(
    result: Any,
    class_names: dict[int, str],
    *,
    source: str,
    frame_id: str,
    timestamp_utc: str | None = None,
) -> DetectionResult:
    """Convert one Ultralytics ``Results`` row to a train-filtered ``DetectionResult``."""
    ts = timestamp_utc or _utc_now_iso()
    boxes_out: list[DetectionBox] = []

    det = getattr(result, "boxes", None)
    if det is not None and len(det) > 0:
        xyxy = det.xyxy.cpu().numpy()
        confs = det.conf.cpu().numpy()
        clss = det.cls.cpu().numpy().astype(int)
        for i in range(len(det)):
            cls_id = int(clss[i])
            raw = class_names.get(cls_id, str(cls_id))
            if not is_train_like_label(raw):
                continue
            x1, y1, x2, y2 = xyxy[i]
            w = float(x2 - x1)
            h = float(y2 - y1)
            if w <= 0 or h <= 0:
                continue
            boxes_out.append(
                DetectionBox(
                    x=float(x1),
                    y=float(y1),
                    w=w,
                    h=h,
                    conf=_clamp01(float(confs[i])),
                    label=str(raw),
                )
            )

    n = len(boxes_out)
    return DetectionResult(
        present=n > 0,
        count=n,
        boxes=tuple(boxes_out),
        frame_id=frame_id,
        source=source,
        timestamp_utc=ts,
    )


def load_yolo_model(weights_path: str | None = None) -> Any:
    from ultralytics import YOLO

    path = (weights_path or "").strip() or DEFAULT_YOLO_WEIGHTS
    return YOLO(path)


def infer_image(
    image_path: Path | str,
    model: Any,
    *,
    conf_threshold: float = 0.5,
    frame_id: str | None = None,
) -> DetectionResult:
    """Run detection on a single image file."""
    path = Path(image_path).resolve()
    fid = frame_id if frame_id is not None else path.stem
    results = model.predict(
        source=str(path),
        conf=conf_threshold,
        verbose=False,
    )
    if not results:
        raise RuntimeError("YOLO returned no results")
    return yolo_result_to_detection(
        results[0],
        model.names,
        source=str(path),
        frame_id=fid,
    )


def infer_video_frames(
    video_path: Path | str,
    model: Any,
    *,
    conf_threshold: float = 0.5,
) -> Iterator[tuple[int, DetectionResult]]:
    """Yield ``(frame_index, DetectionResult)`` for each decoded frame."""
    path = Path(video_path).resolve()
    stream = model.predict(
        source=str(path),
        conf=conf_threshold,
        stream=True,
        verbose=False,
    )
    for i, r in enumerate(stream):
        yield i, yolo_result_to_detection(
            r,
            model.names,
            source=str(path),
            frame_id=str(i),
        )


def is_image_path(path: Path) -> bool:
    return path.suffix.lower() in _IMAGE_SUFFIXES


def is_video_path(path: Path) -> bool:
    """True for playable video extensions; false for ``.mp4.part`` and similar in-progress names."""
    lower = [s.lower() for s in path.suffixes]
    if any(s in _INCOMPLETE_SUFFIXES for s in lower):
        return False
    return path.suffix.lower() in _VIDEO_SUFFIXES


def list_image_paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(directory.iterdir()):
        if p.is_file() and is_image_path(p):
            out.append(p)
    return out


def list_video_paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(directory.iterdir()):
        if p.is_file() and is_video_path(p):
            out.append(p)
    return out
