"""Slice 1: pretrained YOLO on committed test stills + short synthetic video."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vision.schema import parse_detection_result
from vision.yolo_infer import infer_image, infer_video_frames, is_image_path, is_video_path

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_STILLS = REPO_ROOT / "vision" / "test_stills"
_EXPECTATIONS_PATH = TEST_STILLS / "expectations.json"


def _without_train_max_allowed(name: str) -> int:
    if not _EXPECTATIONS_PATH.is_file():
        return 0
    data = json.loads(_EXPECTATIONS_PATH.read_text(encoding="utf-8"))
    entry = (data.get("without_train") or {}).get(name) or {}
    return int(entry.get("max_train_detections", 0))

# COCO “train” can be shy on distant crops; slightly lower than 0.5 for these stills.
_CONF = 0.45


def _image_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and is_image_path(p))


def test_with_train_stills_detect_train(yolo_model) -> None:
    paths = _image_files(TEST_STILLS / "with_train")
    assert len(paths) >= 1, "expected vision/test_stills/with_train/*.png"
    for p in paths:
        r = infer_image(p, yolo_model, conf_threshold=_CONF)
        assert r.present, f"expected train in {p.name}"
        assert r.count >= 1
        parse_detection_result(
            {
                "present": r.present,
                "count": r.count,
                "boxes": [
                    {"x": b.x, "y": b.y, "w": b.w, "h": b.h, "conf": b.conf, "label": b.label}
                    for b in r.boxes
                ],
            }
        )


def test_without_train_stills_no_train(yolo_model) -> None:
    paths = _image_files(TEST_STILLS / "without_train")
    assert len(paths) >= 1, "expected vision/test_stills/without_train/*.png"
    for p in paths:
        r = infer_image(p, yolo_model, conf_threshold=_CONF)
        cap = _without_train_max_allowed(p.name)
        assert r.count <= cap, (
            f"expected at most {cap} train-like box(es) in {p.name}, got {r.count}"
        )
        assert r.present == (r.count > 0)


def test_is_video_path_ignores_ytdlp_part_names(tmp_path: Path) -> None:
    (tmp_path / "done.mp4").touch()
    (tmp_path / "wip.mp4.part").touch()
    (tmp_path / "frag.ytdl").touch()
    assert is_video_path(tmp_path / "done.mp4") is True
    assert is_video_path(tmp_path / "wip.mp4.part") is False
    assert is_video_path(tmp_path / "frag.ytdl") is False


def test_video_smoke_inmemory_clip(yolo_model, tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    stills = _image_files(TEST_STILLS / "with_train")
    assert stills, "need at least one with_train still for video smoke"
    img = cv2.imread(str(stills[0]))
    assert img is not None
    h, w = img.shape[:2]
    out_mp4 = tmp_path / "smoke.mp4"
    writer = cv2.VideoWriter(
        str(out_mp4),
        cv2.VideoWriter_fourcc(*"mp4v"),
        2.0,
        (w, h),
    )
    assert writer.isOpened()
    for _ in range(6):
        writer.write(img)
    writer.release()

    frames = list(infer_video_frames(out_mp4, yolo_model, conf_threshold=_CONF))
    assert len(frames) >= 1
    for _idx, res in frames:
        assert res.count >= 0
        assert res.present == (res.count > 0)
