"""
CLI: pretrained YOLO → train-like ``DetectionResult`` JSON (and optional annotated image).

Usage::

    python -m vision.infer --input path/to/image.png
    python -m vision.infer --input path/to/video.mp4 --out-dir ./out
    python -m vision.infer --input path/to/folder_of_images/
    python -m vision.infer --input path/to/folder_of_mp4s/

Environment: optional ``VISION_MODEL_PATH``, ``VISION_CONF_THRESHOLD`` (see ``vision.config``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vision.config import load_vision_config
from vision.schema import detection_result_to_json_dict
from vision.yolo_infer import (
    DEFAULT_YOLO_WEIGHTS,
    infer_image,
    infer_video_frames,
    is_image_path,
    is_video_path,
    list_image_paths,
    list_video_paths,
    load_yolo_model,
)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _save_annotated(
    model: object, image_path: Path, dest: Path, *, conf_threshold: float
) -> None:
    import cv2

    results = model.predict(
        source=str(image_path), conf=conf_threshold, verbose=False
    )
    if not results:
        return
    bgr = results[0].plot()
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), bgr)


def main(argv: list[str] | None = None) -> int:
    cfg = load_vision_config()
    parser = argparse.ArgumentParser(description="YOLO train detection → DetectionResult JSON")
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=Path,
        help="Image file, video file, or directory of images and/or videos",
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        type=Path,
        default=None,
        help="Output directory (default: ./vision_infer_out)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help=f"YOLO weights path (default: env VISION_MODEL_PATH or {DEFAULT_YOLO_WEIGHTS})",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Confidence threshold 0–1 (default: VISION_CONF_THRESHOLD or 0.5)",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Save annotated image(s) next to JSON (images only)",
    )
    args = parser.parse_args(argv)

    inp = args.input.expanduser().resolve()
    out_dir = (args.out_dir or Path("vision_infer_out")).expanduser().resolve()
    weights = (args.model or cfg.model_path or DEFAULT_YOLO_WEIGHTS).strip()
    conf = float(args.conf) if args.conf is not None else float(cfg.conf_threshold)

    if not inp.exists():
        print(f"[vision.infer] not found: {inp}", file=sys.stderr)
        return 2

    model = load_yolo_model(weights)

    if inp.is_dir():
        vids = list_video_paths(inp)
        imgs = list_image_paths(inp)
        if not vids and not imgs:
            print(
                f"[vision.infer] no images or videos in {inp}",
                file=sys.stderr,
            )
            return 3
        out_dir.mkdir(parents=True, exist_ok=True)
        video_errors = 0
        for p in vids:
            jsonl_path = out_dir / f"{p.stem}_results.jsonl"
            try:
                n = 0
                with open(jsonl_path, "w", encoding="utf-8") as f:
                    for _i, res in infer_video_frames(p, model, conf_threshold=conf):
                        f.write(
                            json.dumps(
                                detection_result_to_json_dict(res),
                                separators=(",", ":"),
                            )
                        )
                        f.write("\n")
                        n += 1
                print(f"[vision.infer] {p.name} → {n} frame(s) → {jsonl_path.name}")
            except Exception as e:
                video_errors += 1
                if jsonl_path.exists():
                    try:
                        jsonl_path.unlink()
                    except OSError:
                        pass
                print(
                    f"[vision.infer] skip {p.name}: {e}",
                    file=sys.stderr,
                )
        if video_errors:
            print(
                f"[vision.infer] {video_errors} video(s) failed (e.g. incomplete file); others unchanged.",
                file=sys.stderr,
            )
        for p in imgs:
            res = infer_image(p, model, conf_threshold=conf)
            stem = p.stem
            _write_json(
                out_dir / f"{stem}_result.json", detection_result_to_json_dict(res)
            )
            if args.annotate:
                _save_annotated(
                    model, p, out_dir / f"{stem}_annotated.jpg", conf_threshold=conf
                )
        print(f"[vision.infer] wrote outputs under {out_dir}")
        return 1 if video_errors else 0

    if is_image_path(inp):
        res = infer_image(inp, model, conf_threshold=conf)
        _write_json(out_dir / "result.json", detection_result_to_json_dict(res))
        if args.annotate:
            _save_annotated(model, inp, out_dir / "annotated.jpg", conf_threshold=conf)
        print(f"[vision.infer] wrote {out_dir / 'result.json'}")
        return 0

    if is_video_path(inp):
        out_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = out_dir / "results.jsonl"
        n = 0
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for _i, res in infer_video_frames(inp, model, conf_threshold=conf):
                f.write(json.dumps(detection_result_to_json_dict(res), separators=(",", ":")))
                f.write("\n")
                n += 1
        print(f"[vision.infer] wrote {n} frame record(s) → {jsonl_path}")
        return 0

    print(f"[vision.infer] unsupported file type: {inp.suffix}", file=sys.stderr)
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
