"""Detection JSON schema validation (slice 0)."""

from __future__ import annotations

from pathlib import Path

import pytest

from vision.schema import (
    SCHEMA_VERSION,
    DetectionBox,
    DetectionResult,
    detection_result_to_json_dict,
    load_detection_fixture,
    parse_detection_result,
)

FIXTURE = Path(__file__).resolve().parents[2] / "vision" / "fixtures" / "detection_example.json"


def test_load_example_fixture() -> None:
    r = load_detection_fixture(FIXTURE)
    assert r.version == SCHEMA_VERSION
    assert r.present is True
    assert r.count == 2
    assert len(r.boxes) == 2
    assert r.boxes[0].label == "locomotive"
    assert r.source == "fixture"


def test_round_trip_json_dict() -> None:
    r = load_detection_fixture(FIXTURE)
    d = detection_result_to_json_dict(r)
    r2 = parse_detection_result(d)
    assert r2 == r


def test_parse_minimal_empty() -> None:
    r = parse_detection_result(
        {
            "present": False,
            "count": 0,
            "boxes": [],
        }
    )
    assert r.present is False
    assert r.count == 0
    assert r.boxes == ()


def test_reject_count_box_mismatch() -> None:
    with pytest.raises(ValueError, match="count must equal len"):
        parse_detection_result(
            {
                "present": True,
                "count": 2,
                "boxes": [
                    {"x": 0, "y": 0, "w": 1, "h": 1, "conf": 0.9, "label": "train"},
                ],
            }
        )


def test_reject_present_inconsistent() -> None:
    with pytest.raises(ValueError, match="present must be True"):
        parse_detection_result(
            {
                "present": False,
                "count": 1,
                "boxes": [{"x": 0, "y": 0, "w": 1, "h": 1, "conf": 0.9, "label": "train"}],
            }
        )


def test_reject_invalid_conf() -> None:
    with pytest.raises(ValueError, match="conf must be in"):
        DetectionBox(x=0, y=0, w=1, h=1, conf=1.5, label="train")


def test_reject_non_positive_wh() -> None:
    with pytest.raises(ValueError, match="w and h must be positive"):
        DetectionBox(x=0, y=0, w=0, h=1, conf=0.5, label="train")


def test_reject_bad_timestamp() -> None:
    with pytest.raises(ValueError, match="ISO 8601"):
        DetectionResult(
            present=True,
            count=1,
            boxes=(DetectionBox(0, 0, 1, 1, 0.5, "train"),),
            timestamp_utc="not-a-date",
        )


def test_reject_non_dict_payload() -> None:
    with pytest.raises(TypeError):
        parse_detection_result([])  # type: ignore[arg-type]
