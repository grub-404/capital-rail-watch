"""SQLite storage for crowd labels on vision screenshots (slice 5)."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS screenshot_labels (
    filename TEXT PRIMARY KEY,
    labeled_at TEXT NOT NULL,
    skipped INTEGER NOT NULL DEFAULT 0,
    present INTEGER,
    provider TEXT,
    engine_model TEXT,
    num_cars INTEGER,
    num_trains INTEGER,
    train_grid_json TEXT,
    train_mask_json TEXT,
    trains_json TEXT,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_screenshot_labels_skipped ON screenshot_labels (skipped);
"""


@dataclass(frozen=True)
class LabelRow:
    filename: str
    labeled_at: str
    skipped: bool
    present: bool | None
    provider: str | None
    engine_model: str | None
    num_cars: int | None
    num_trains: int | None
    train_grid_json: str | None
    train_mask_json: str | None
    trains_json: str | None
    notes: str | None


def _migrate_screenshot_labels(conn: sqlite3.Connection) -> None:
    """Add columns introduced after first deploy (SQLite has no IF NOT EXISTS for columns)."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(screenshot_labels)")}
    if "num_trains" not in cols:
        conn.execute("ALTER TABLE screenshot_labels ADD COLUMN num_trains INTEGER")
    if "train_grid_json" not in cols:
        conn.execute("ALTER TABLE screenshot_labels ADD COLUMN train_grid_json TEXT")
    if "trains_json" not in cols:
        conn.execute("ALTER TABLE screenshot_labels ADD COLUMN trains_json TEXT")
    if "train_mask_json" not in cols:
        conn.execute("ALTER TABLE screenshot_labels ADD COLUMN train_mask_json TEXT")


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    _migrate_screenshot_labels(conn)
    conn.commit()
    return conn


def list_screenshot_pngs(screenshot_dir: Path) -> list[str]:
    if not screenshot_dir.is_dir():
        return []
    out: list[str] = []
    for p in sorted(screenshot_dir.iterdir()):
        if p.is_file() and p.suffix.lower() == ".png" and "_assist_" not in p.name:
            out.append(p.name)
    return out


def _done_filenames(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT filename FROM screenshot_labels").fetchall()
    return {r[0] for r in rows}


def next_unlabeled_filename(screenshot_dir: Path, conn: sqlite3.Connection) -> str | None:
    done = _done_filenames(conn)
    for name in list_screenshot_pngs(screenshot_dir):
        if name not in done:
            return name
    return None


def read_detection_hint(screenshot_dir: Path, png_name: str) -> dict | None:
    """Load sibling ``.json`` sidecar if present (model output)."""
    stem = Path(png_name).stem
    side = screenshot_dir / f"{stem}.json"
    if not side.is_file():
        return None
    try:
        with open(side, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


_MAX_GRID_DIM = 24
_MAX_GRID_CELLS = 576  # 24*24
_MAX_TRAIN_INDEX_IN_GRID = 8
_MAX_MASK_W = 512
_MAX_MASK_H = 512
_MAX_MASK_PIXELS = 200_000
_MAX_MASK_TRAIN_ID = 8


def normalize_train_grid_json(raw: str | None) -> str | None:
    """Validate grid overlay: per-cell train id (1..K) or legacy ``selected`` indices (all train 1)."""
    if not raw or not str(raw).strip():
        return None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    rows = obj.get("rows")
    cols = obj.get("cols")
    if not isinstance(rows, int) or not isinstance(cols, int):
        return None
    if rows < 1 or rows > _MAX_GRID_DIM or cols < 1 or cols > _MAX_GRID_DIM:
        return None
    n = rows * cols
    if n > _MAX_GRID_CELLS:
        return None

    if "cells" in obj:
        cells_in = obj.get("cells")
        if not isinstance(cells_in, list) or len(cells_in) != n:
            return None
        out_cells: list[int | None] = []
        for v in cells_in:
            if v is None:
                out_cells.append(None)
            elif isinstance(v, int) and 1 <= v <= _MAX_TRAIN_INDEX_IN_GRID:
                out_cells.append(v)
            else:
                return None
        return json.dumps({"rows": rows, "cols": cols, "cells": out_cells}, separators=(",", ":"))

    selected = obj.get("selected")
    if not isinstance(selected, list):
        return None
    seen: set[int] = set()
    for x in selected:
        if not isinstance(x, int) or x < 0 or x >= n or x in seen:
            return None
        seen.add(x)
    cells: list[int | None] = [None] * n
    for i in sorted(seen):
        cells[i] = 1
    return json.dumps({"rows": rows, "cols": cols, "cells": cells}, separators=(",", ":"))


def normalize_trains_json(
    raw: str | None,
    *,
    allowed_providers: frozenset[str],
    present: bool,
) -> str | None:
    """Validate per-train labels from the form. When not ``present``, returns None."""
    if not present:
        return None
    if not raw or not str(raw).strip():
        return None
    try:
        arr = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(arr, list) or not arr:
        return None
    if len(arr) > 8:
        return None
    norm: list[dict[str, str | int | None]] = []
    for item in arr:
        if not isinstance(item, dict):
            return None
        p = str(item.get("provider", "unknown") or "unknown").strip().lower()
        if p not in allowed_providers:
            p = "unknown"
        em = item.get("engine_model")
        if em is not None and em != "":
            em = str(em).strip() or None
        else:
            em = None
        nc = item.get("num_cars")
        num_cars: int | None
        if nc is None or nc == "":
            num_cars = None
        else:
            try:
                num_cars = max(0, min(99, int(nc)))
            except (TypeError, ValueError):
                return None
        norm.append({"provider": p, "engine_model": em, "num_cars": num_cars})
    return json.dumps(norm, separators=(",", ":"), ensure_ascii=False)


def normalize_train_mask_json(raw: str | None) -> str | None:
    """Validate region-grow paint mask: compact RLE over a downsampled image grid.

    Expected shape::

        {"w":W,"h":H,"rle":[v0,n0,v1,n1,...]}

    Where v is 0..8 (0 = none, 1..K = train id) and sum(n) == W*H.
    """
    if not raw or not str(raw).strip():
        return None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    w = obj.get("w")
    h = obj.get("h")
    rle = obj.get("rle")
    if not isinstance(w, int) or not isinstance(h, int):
        return None
    if w < 1 or h < 1 or w > _MAX_MASK_W or h > _MAX_MASK_H:
        return None
    total = w * h
    if total > _MAX_MASK_PIXELS:
        return None
    if not isinstance(rle, list) or len(rle) < 2 or len(rle) % 2 != 0:
        return None
    out: list[int] = []
    acc = 0
    for i in range(0, len(rle), 2):
        v = rle[i]
        n = rle[i + 1]
        if not isinstance(v, int) or not 0 <= v <= _MAX_MASK_TRAIN_ID:
            return None
        if not isinstance(n, int) or n <= 0:
            return None
        acc += n
        if acc > total:
            return None
        out.extend([v, n])
    if acc != total:
        return None
    return json.dumps({"w": w, "h": h, "rle": out}, separators=(",", ":"))


def save_label(
    conn: sqlite3.Connection,
    *,
    filename: str,
    skipped: bool,
    present: bool | None,
    provider: str | None,
    engine_model: str | None,
    num_cars: int | None,
    num_trains: int | None,
    train_grid_json: str | None,
    train_mask_json: str | None,
    trains_json: str | None,
    notes: str | None,
) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO screenshot_labels
           (filename, labeled_at, skipped, present, provider, engine_model, num_cars,
            num_trains, train_grid_json, train_mask_json, trains_json, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            filename,
            utc_now_iso(),
            1 if skipped else 0,
            None if skipped else (1 if present else 0),
            None if skipped else (provider or "unknown"),
            None if skipped else (engine_model or "").strip() or None,
            num_cars if not skipped and num_cars is not None else None,
            None if skipped else num_trains,
            None if skipped else train_grid_json,
            None if skipped else train_mask_json,
            None if skipped else trains_json,
            (notes or "").strip() or None,
        ),
    )
    conn.commit()


def export_jsonl(conn: sqlite3.Connection) -> str:
    lines: list[str] = []
    for row in conn.execute(
        "SELECT * FROM screenshot_labels WHERE skipped = 0 ORDER BY labeled_at"
    ):
        d = dict(row)
        d["skipped"] = bool(d["skipped"])
        if d["present"] is not None:
            d["present"] = bool(d["present"])
        lines.append(json.dumps(d, ensure_ascii=False))
    return "\n".join(lines) + ("\n" if lines else "")


def count_pending(screenshot_dir: Path, conn: sqlite3.Connection) -> tuple[int, int]:
    """``(unlabeled_count, labeled_count)``."""
    all_png = list_screenshot_pngs(screenshot_dir)
    done = _done_filenames(conn)
    labeled = len(done)
    pending = sum(1 for n in all_png if n not in done)
    return pending, labeled
