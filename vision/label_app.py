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
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)

from vision.config import PROJECT_ROOT, load_vision_config
from vision.label_assist import ASSIST_EDGE_SUFFIX, ASSIST_FG_SUFFIX
from vision.label_db import (
    connect,
    count_pending,
    export_jsonl,
    next_unlabeled_filename,
    normalize_train_grid_json,
    normalize_train_mask_json,
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


def _assist_json_for_filename(shot: Path, filename: str | None) -> dict:
    """URLs / flags for optional preprocess sidecars (see ``vision.label_assist``)."""
    if not filename:
        return {"fgUrl": None, "edgeUrl": None, "hasFg": False, "hasEdge": False}
    stem = Path(filename).stem
    base = shot.resolve()
    fg_path = base / f"{stem}{ASSIST_FG_SUFFIX}"
    edge_path = base / f"{stem}{ASSIST_EDGE_SUFFIX}"
    has_fg = fg_path.is_file()
    has_edge = edge_path.is_file()
    return {
        "fgUrl": url_for("serve_assist", kind="fg", name=filename) if has_fg else None,
        "edgeUrl": url_for("serve_assist", kind="edge", name=filename) if has_edge else None,
        "hasFg": has_fg,
        "hasEdge": has_edge,
    }


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
    app.config["SAM_MODEL_PATH"] = os.environ.get("VISION_SAM_MODEL", "sam_b.pt")

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
                assist=_assist_json_for_filename(_shot_dir(), filename),
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

    @app.get("/assist/<kind>/<path:name>")
    def serve_assist(kind: str, name: str):
        if ".." in name or name.startswith(("/\\", "\\")):
            abort(400)
        if not name.lower().endswith(".png"):
            abort(400)
        if kind not in ("fg", "edge"):
            abort(404)
        base = _shot_dir().resolve()
        stem = Path(name).stem
        suffix = ASSIST_FG_SUFFIX if kind == "fg" else ASSIST_EDGE_SUFFIX
        path = (base / f"{stem}{suffix}").resolve()
        try:
            path.relative_to(base)
        except ValueError:
            abort(403)
        if not path.is_file():
            abort(404)
        return send_file(path, mimetype="image/png")

    def _sam_model():
        """Lazy-load SAM on first use (kept in-process)."""
        m = getattr(app, "_sam_model", None)
        if m is not None:
            return m
        try:
            from ultralytics import SAM  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("SAM requires ultralytics extras; install requirements-dev.txt") from e
        model_path = str(app.config.get("SAM_MODEL_PATH") or "sam_b.pt")
        m = SAM(model_path)
        setattr(app, "_sam_model", m)
        return m

    def _encode_rle_u8(u8) -> list[int]:
        out: list[int] = []
        i = 0
        total = int(u8.size)
        while i < total:
            v = int(u8.flat[i])
            run = 1
            while i + run < total and int(u8.flat[i + run]) == v:
                run += 1
            out.extend([v, run])
            i += run
        return out

    @app.post("/assist/sam")
    def assist_sam():
        """
        Click-to-segment using SAM.

        Body JSON:
          { filename: "x.png", x: int, y: int, out_w: int, out_h: int, labels?: [0/1], points?: [[x,y]] }

        Returns:
          { w, h, rle } for a binary mask at output dims (0/1 values).
        """
        d = request.get_json(silent=True) or {}
        filename = str(d.get("filename") or "").strip()
        if not filename or Path(filename).name != filename or not filename.lower().endswith(".png"):
            abort(400)
        base = _shot_dir().resolve()
        img_path = (base / filename).resolve()
        try:
            img_path.relative_to(base)
        except ValueError:
            abort(403)
        if not img_path.is_file():
            abort(404)

        try:
            x = int(d.get("x"))
            y = int(d.get("y"))
            out_w = int(d.get("out_w"))
            out_h = int(d.get("out_h"))
        except Exception:
            abort(400)
        if out_w < 8 or out_h < 8 or out_w > 1024 or out_h > 1024:
            abort(400)

        points = d.get("points")
        labels = d.get("labels")
        if points is None:
            points = [[x, y]]
        if labels is None:
            labels = [1] * len(points)

        try:
            model = _sam_model()
            res = model.predict(
                source=str(img_path),
                points=points,
                labels=labels,
                device="mps",
                verbose=False,
            )
            r0 = res[0]
            masks = getattr(r0, "masks", None)
            if masks is None or getattr(masks, "data", None) is None:
                return jsonify({"w": out_w, "h": out_h, "rle": [0, out_w * out_h]})
            t = masks.data  # torch.Tensor [N,H,W] bool
            # pick the first mask for now
            t0 = t[0].to("cpu").numpy().astype("uint8")  # 0/1 at orig HxW
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        try:
            import cv2  # type: ignore
            import numpy as np

            if t0.shape[1] != out_w or t0.shape[0] != out_h:
                t0 = cv2.resize(t0, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            t0 = (t0 > 0).astype(np.uint8)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        return jsonify({"w": out_w, "h": out_h, "rle": _encode_rle_u8(t0)})

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
        train_mask_json = normalize_train_mask_json(request.form.get("train_mask_json"))
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
                train_mask_json=train_mask_json,
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
                train_mask_json=None,
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
