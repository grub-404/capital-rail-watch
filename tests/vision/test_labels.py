"""Train-like label mapping (slice 0)."""

from __future__ import annotations

from vision.labels import count_train_like_labels, is_train_like_label, normalize_label


def test_locomotive_and_railcar_train_like() -> None:
    assert is_train_like_label("locomotive") is True
    assert is_train_like_label("RailCar") is True
    assert is_train_like_label("  train  ") is True


def test_non_train_labels() -> None:
    assert is_train_like_label("person") is False
    assert is_train_like_label("truck") is False


def test_normalize_label() -> None:
    assert normalize_label("  Locomotive ") == "locomotive"


def test_count_train_like_labels() -> None:
    n = count_train_like_labels(
        ["locomotive", "person", "railcar", "train", "bus"]
    )
    assert n == 3
