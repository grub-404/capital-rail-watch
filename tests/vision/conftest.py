"""Shared fixtures for vision integration tests (YOLO weights download on first run)."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def yolo_model():
    pytest.importorskip("ultralytics")
    from vision.yolo_infer import load_yolo_model

    return load_yolo_model(None)
