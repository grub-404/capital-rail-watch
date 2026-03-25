"""Slice 5: labeling Flask app."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
import numpy as np

from vision.label_app import create_app
from vision.label_db import (
    connect,
    export_jsonl,
    normalize_train_grid_json,
    normalize_train_mask_json,
    normalize_trains_json,
)


@pytest.fixture
def label_env(tmp_path: Path):
    shot = tmp_path / "shots"
    shot.mkdir()
    db = tmp_path / "labels.db"
    for name in ("a.png", "b.png"):
        cv2.imwrite(str(shot / name), np.zeros((8, 8, 3), dtype=np.uint8))
    (shot / "a.json").write_text(
        json.dumps({"present": True, "count": 1, "version": 1}),
        encoding="utf-8",
    )
    app = create_app(screenshot_dir=shot, label_db=db)
    app.config["TESTING"] = True
    return app, shot, db


def test_index_lists_first_unlabeled(label_env) -> None:
    app, shot, _db = label_env
    c = app.test_client()
    r = c.get("/")
    assert r.status_code == 200
    assert b"a.png" in r.data or b"b.png" in r.data


def test_submit_then_next(label_env) -> None:
    app, shot, db = label_env
    c = app.test_client()
    r = c.post(
        "/label/submit",
        data={
            "filename": "a.png",
            "present": "1",
            "provider": "amtrak",
            "engine_model": "ACS-64",
            "num_cars": "6",
            "notes": "test",
        },
        follow_redirects=False,
    )
    assert r.status_code == 302
    conn = connect(db)
    try:
        row = conn.execute(
            "SELECT * FROM screenshot_labels WHERE filename = ?",
            ("a.png",),
        ).fetchone()
        assert row is not None
        assert row["skipped"] == 0
        assert row["present"] == 1
        assert row["provider"] == "amtrak"
        assert row["engine_model"] == "ACS-64"
        assert row["num_cars"] == 6
        assert row["num_trains"] == 1
        assert row["train_grid_json"] is None
        assert row["train_mask_json"] is None
        assert row["trains_json"] is not None
        assert json.loads(row["trains_json"])[0]["provider"] == "amtrak"
    finally:
        conn.close()
    r2 = c.get("/")
    assert b"b.png" in r2.data


def test_skip(label_env) -> None:
    app, _shot, db = label_env
    c = app.test_client()
    r = c.post("/label/skip", data={"filename": "a.png"})
    assert r.status_code == 302
    conn = connect(db)
    try:
        row = conn.execute(
            "SELECT skipped FROM screenshot_labels WHERE filename = ?",
            ("a.png",),
        ).fetchone()
        assert row["skipped"] == 1
        body = export_jsonl(conn)
        assert "a.png" not in body
    finally:
        conn.close()


def test_submit_train_mask_json(label_env) -> None:
    app, _shot, db = label_env
    c = app.test_client()
    mask = json.dumps({"w": 2, "h": 2, "rle": [1, 1, 0, 3]})
    c.post(
        "/label/submit",
        data={
            "filename": "a.png",
            "present": "1",
            "train_mask_json": mask,
            "notes": "",
        },
    )
    conn = connect(db)
    try:
        row = conn.execute(
            "SELECT train_mask_json, num_trains FROM screenshot_labels WHERE filename = ?",
            ("a.png",),
        ).fetchone()
        assert row["num_trains"] == 1
        assert row["train_mask_json"] is not None
        assert json.loads(row["train_mask_json"])["w"] == 2
    finally:
        conn.close()


def test_submit_num_trains_and_grid(label_env) -> None:
    app, _shot, db = label_env
    c = app.test_client()
    cells = [None] * 16
    cells[0] = 1
    cells[1] = 1
    cells[2] = 2
    grid = json.dumps({"rows": 4, "cols": 4, "cells": cells})
    trains = [
        {"provider": "amtrak", "engine_model": "P42DC", "num_cars": 4},
        {"provider": "marc", "engine_model": "MP36PH-3C", "num_cars": 3},
    ]
    c.post(
        "/label/submit",
        data={
            "filename": "a.png",
            "present": "1",
            "train_grid_json": grid,
            "trains_json": json.dumps(trains),
            "notes": "",
        },
    )
    conn = connect(db)
    try:
        row = conn.execute(
            "SELECT num_trains, train_grid_json, train_mask_json, trains_json, provider FROM screenshot_labels WHERE filename = ?",
            ("a.png",),
        ).fetchone()
        assert row["num_trains"] == 2
        assert row["provider"] == "amtrak"
        g = json.loads(row["train_grid_json"])
        assert g["rows"] == 4 and g["cols"] == 4
        assert g["cells"][0] == 1 and g["cells"][2] == 2
        assert row["train_mask_json"] is None
        tj = json.loads(row["trains_json"])
        assert len(tj) == 2
        assert tj[1]["provider"] == "marc"
    finally:
        conn.close()


def test_export_jsonl(label_env) -> None:
    app, _shot, db = label_env
    c = app.test_client()
    c.post(
        "/label/submit",
        data={
            "filename": "a.png",
            "present": "1",
            "provider": "marc",
            "engine_model": "",
            "num_cars": "",
            "notes": "",
        },
    )
    r = c.get("/label/export.jsonl")
    assert r.status_code == 200
    line = r.data.decode("utf-8").strip().splitlines()[0]
    d = json.loads(line)
    assert d["filename"] == "a.png"
    assert d["provider"] == "marc"


def test_media_serves_png(label_env) -> None:
    app, _shot, _db = label_env
    c = app.test_client()
    r = c.get("/media/a.png")
    assert r.status_code == 200
    assert r.mimetype == "image/png"


def test_normalize_train_grid_json() -> None:
    assert normalize_train_grid_json("") is None
    assert normalize_train_grid_json("not json") is None
    assert normalize_train_grid_json('{"rows":4,"cols":4,"selected":[5,5]}') is None
    legacy = normalize_train_grid_json('{"rows":4,"cols":4,"selected":[10,0]}')
    d = json.loads(legacy)
    assert d["rows"] == 4 and d["cols"] == 4
    assert d["cells"][0] == 1 and d["cells"][10] == 1
    assert sum(1 for x in d["cells"] if x == 1) == 2


def test_normalize_trains_json() -> None:
    prov = frozenset({"unknown", "amtrak", "marc"})
    assert normalize_trains_json("", allowed_providers=prov, present=False) is None
    assert normalize_trains_json("[]", allowed_providers=prov, present=True) is None
    out = normalize_trains_json(
        '[{"provider":"amtrak","engine_model":"","num_cars":5}]',
        allowed_providers=prov,
        present=True,
    )
    assert json.loads(out)[0]["engine_model"] is None
    assert json.loads(out)[0]["num_cars"] == 5


def test_normalize_train_mask_json() -> None:
    assert normalize_train_mask_json("") is None
    assert normalize_train_mask_json("not json") is None
    ok = normalize_train_mask_json('{"w":3,"h":2,"rle":[0,3,1,2,0,1]}')
    assert json.loads(ok) == {"w": 3, "h": 2, "rle": [0, 3, 1, 2, 0, 1]}
    assert normalize_train_mask_json('{"w":3,"h":2,"rle":[0,5]}') is None  # wrong total


def test_media_path_traversal(label_env) -> None:
    app, _shot, _db = label_env
    c = app.test_client()
    assert c.get("/media/../labels.db").status_code in (400, 403, 404)


def test_assist_serves_sidecars(label_env) -> None:
    app, shot, _db = label_env
    from vision.label_assist import ASSIST_EDGE_SUFFIX, ASSIST_FG_SUFFIX

    cv2.imwrite(str(shot / f"a{ASSIST_FG_SUFFIX}"), np.full((8, 8), 200, dtype=np.uint8))
    cv2.imwrite(str(shot / f"a{ASSIST_EDGE_SUFFIX}"), np.full((8, 8), 50, dtype=np.uint8))
    c = app.test_client()
    r_fg = c.get("/assist/fg/a.png")
    r_edge = c.get("/assist/edge/a.png")
    assert r_fg.status_code == 200
    assert r_edge.status_code == 200
    assert r_fg.mimetype == "image/png"


def test_assist_missing_returns_404(label_env) -> None:
    app, _shot, _db = label_env
    c = app.test_client()
    assert c.get("/assist/fg/b.png").status_code == 404


def test_index_lists_assist_urls_when_present(label_env) -> None:
    app, shot, _db = label_env
    from vision.label_assist import ASSIST_FG_SUFFIX

    cv2.imwrite(str(shot / f"a{ASSIST_FG_SUFFIX}"), np.zeros((4, 4), dtype=np.uint8))
    c = app.test_client()
    r = c.get("/")
    assert r.status_code == 200
    assert b"/assist/fg/a.png" in r.data


def test_assist_path_traversal_rejected(label_env) -> None:
    app, _shot, _db = label_env
    c = app.test_client()
    assert c.get("/assist/fg/../labels.db").status_code in (400, 403, 404)
