"""
Map pretrained YOLO class names to a single logical “rail vehicle” for presence and counting.

Names are compared case-insensitively after strip(). Extend TRAIN_LIKE_LABELS as you adopt
new weight files or custom fine-tunes.
"""

from __future__ import annotations

# Normalized (lowercase) tokens that count as a train for yes/no and count.
TRAIN_LIKE_LABELS: frozenset[str] = frozenset(
    {
        "train",
        "locomotive",
        "railcar",
        "rail car",
        "passenger train",
        "freight train",
    }
)


def normalize_label(name: str) -> str:
    return str(name).strip().lower()


def is_train_like_label(name: str) -> bool:
    return normalize_label(name) in TRAIN_LIKE_LABELS


def count_train_like_labels(labels: list[str]) -> int:
    return sum(1 for lab in labels if is_train_like_label(lab))
