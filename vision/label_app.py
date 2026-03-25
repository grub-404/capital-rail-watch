"""
Crowd labeling UI for vision screenshots (slice 5).

Run from repo root::

    python -m vision.label_app

Open http://127.0.0.1:8765 — set ``VISION_LABEL_PORT`` to change. Screenshots are read from
``VISION_SCREENSHOT_DIR``; labels go to ``db/vision_labels.db`` (or ``VISION_LABEL_DB``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from flask import (
    Flask,
    abort,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)

from vision.config import PROJECT_ROOT, load_vision_config
from vision.label_db import (
    connect,
    count_pending,
    export_jsonl,
    next_unlabeled_filename,
    normalize_train_grid_json,
    normalize_trains_json,
    read_detection_hint,
    save_label,
)

# Passenger / transit operators in the DC–Capital region (no freight class I roads).
PROVIDERS = [
    "unknown",
    "amtrak",
    "vre",
    "marc",
    "metro",
    "other",
]

# Common power / rolling stock for regional context (datalist hints; free text still allowed).
ENGINE_MODEL_SUGGESTIONS = sorted(
    {
        # Amtrak — NEC, Virginia, state-supported corridors
        "ACS-64",
        "ALC-42 (Siemens Charger)",
        "Acela (trainset)",
        "P42DC",
        "P40DC",
        "F59PHI",
        # MARC — Penn (diesel/electric mix historically), Camden, Brunswick
        "MP36PH-3C",
        "MP40PH-3C",
        "SC-44 Charger",
        "GP40WH-2",
        "HHP-8 (retired)",
        # VRE — Northern Virginia commuter
        "F59PH",
        "Gallery coach (Nippon Sharyo)",
        # WMATA Metrorail (rapid transit)
        "3000-series (Metro)",
        "5000-series (Metro)",
        "6000-series (Metro)",
        "7000-series (Metro)",
        "8000-series (Metro, future)",
    }
)

ROOT = Path(__file__).resolve().parent.parent
MAX_TRAINS_IN_FORM = 6


def create_app(
    *,
    screenshot_dir: Path | None = None,
    label_db: Path | None = None,
) -> Flask:
    cfg = load_vision_config()
    shot = (screenshot_dir or cfg.screenshot_dir).expanduser().resolve()
    db_path = (label_db or (PROJECT_ROOT / "db" / "vision_labels.db")).expanduser().resolve()

    app = Flask(__name__, template_folder=str(ROOT / "templates"))
    app.config["SCREENSHOT_DIR"] = str(shot)
    app.config["LABEL_DB"] = str(db_path)

    def _shot_dir() -> Path:
        return Path(app.config["SCREENSHOT_DIR"])

    def _db() -> Path:
        return Path(app.config["LABEL_DB"])

    @app.get("/")
    def index():
        conn = connect(_db())
        try:
            pending, labeled = count_pending(_shot_dir(), conn)
            filename = next_unlabeled_filename(_shot_dir(), conn)
            hint = read_detection_hint(_shot_dir(), filename) if filename else None
            return render_template(
                "vision_label/index.html",
                filename=filename,
                hint=hint,
                pending=pending,
                labeled=labeled,
                providers=PROVIDERS,
                engine_model_suggestions=ENGINE_MODEL_SUGGESTIONS,
                max_trains=MAX_TRAINS_IN_FORM,
                screenshot_dir=str(_shot_dir()),
            )
        finally:
            conn.close()

    @app.get("/media/<path:name>")
    def serve_screenshot(name: str):
        if ".." in name or name.startswith(("/\\", "\\")):
            abort(400)
        base = _shot_dir().resolve()
        path = (base / name).resolve()
        try:
            path.relative_to(base)
        except ValueError:
            abort(403)
        if not path.is_file() or path.suffix.lower() != ".png":
            abort(404)
        return send_file(path, mimetype="image/png")

    @app.post("/label/submit")
    def submit_label():
        filename = request.form.get("filename", "").strip()
        if not filename or Path(filename).name != filename:
            abort(400)
        present = request.form.get("present") == "1"
        notes = (request.form.get("notes") or "").strip() or None
        allowed_p = frozenset(PROVIDERS)
        trains_json = normalize_trains_json(
            request.form.get("trains_json"),
            allowed_providers=allowed_p,
            present=present,
        )
        provider = "unknown"
        engine_model: str | None = None
        num_cars: int | None = None
        num_trains: int | None = None
        if present:
            if trains_json is None:
                leg_p = (request.form.get("provider") or "unknown").strip().lower()
                if leg_p not in PROVIDERS:
                    leg_p = "unknown"
                leg_e = (request.form.get("engine_model") or "").strip() or None
                raw_cars = (request.form.get("num_cars") or "").strip()
                leg_c: int | None
                if raw_cars == "":
                    leg_c = None
                else:
                    try:
                        leg_c = int(raw_cars)
                    except ValueError:
                        leg_c = None
                trains_json = normalize_trains_json(
                    json.dumps(
                        [{"provider": leg_p, "engine_model": leg_e, "num_cars": leg_c}],
                        separators=(",", ":"),
                    ),
                    allowed_providers=allowed_p,
                    present=True,
                )
            if trains_json is not None:
                parsed = json.loads(trains_json)
                num_trains = len(parsed)
                first = parsed[0]
                provider = str(first["provider"])
                engine_model = first.get("engine_model")
                nc = first.get("num_cars")
                num_cars = int(nc) if nc is not None else None
        train_grid_json = normalize_train_grid_json(request.form.get("train_grid_json"))
        conn = connect(_db())
        try:
            save_label(
                conn,
                filename=filename,
                skipped=False,
                present=present,
                provider=provider,
                engine_model=engine_model,
                num_cars=num_cars,
                num_trains=num_trains,
                train_grid_json=train_grid_json,
                trains_json=trains_json,
                notes=notes,
            )
        finally:
            conn.close()
        return redirect(url_for("index"))

    @app.post("/label/skip")
    def skip_label():
        filename = request.form.get("filename", "").strip()
        if not filename or Path(filename).name != filename:
            abort(400)
        conn = connect(_db())
        try:
            save_label(
                conn,
                filename=filename,
                skipped=True,
                present=None,
                provider=None,
                engine_model=None,
                num_cars=None,
                num_trains=None,
                train_grid_json=None,
                trains_json=None,
                notes=None,
            )
        finally:
            conn.close()
        return redirect(url_for("index"))

    @app.get("/label/export.jsonl")
    def export_labels():
        conn = connect(_db())
        try:
            body = export_jsonl(conn)
        finally:
            conn.close()
        return (
            body,
            200,
            {
                "Content-Type": "application/x-ndjson; charset=utf-8",
                "Content-Disposition": 'attachment; filename="labels.jsonl"',
            },
        )

    return app


app = create_app()


def main() -> None:
    port = int(os.environ.get("VISION_LABEL_PORT", "8765"))
    host = os.environ.get("VISION_LABEL_HOST", "127.0.0.1")
    print(f"[vision.label_app] http://{host}:{port}/  (screenshots: {app.config['SCREENSHOT_DIR']})")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
