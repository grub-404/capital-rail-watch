"""Capital Rail Watch — vision / ML pipeline (see PLAN.md)."""

from .config import VisionConfig, load_vision_config
from .labels import TRAIN_LIKE_LABELS, count_train_like_labels, is_train_like_label, normalize_label
from .schema import (
    SCHEMA_VERSION,
    DetectionBox,
    DetectionResult,
    detection_result_to_json_dict,
    load_detection_fixture,
    parse_detection_result,
)

__all__ = [
    "SCHEMA_VERSION",
    "TRAIN_LIKE_LABELS",
    "DetectionBox",
    "DetectionResult",
    "VisionConfig",
    "count_train_like_labels",
    "detection_result_to_json_dict",
    "is_train_like_label",
    "load_detection_fixture",
    "load_vision_config",
    "normalize_label",
    "parse_detection_result",
]
