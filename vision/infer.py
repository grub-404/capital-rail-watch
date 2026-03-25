"""
CLI: pretrained YOLO → train-like ``DetectionResult`` JSON (and optional annotated image).

Video inference uses **time-based sampling** by default (``VISION_VIDEO_SAMPLE_INTERVAL_SEC``):
Ultralytics ``vid_stride ≈ fps × interval`` so YOLO does not run on every frame.

Use ``--screenshots`` to write PNG + JSON sidecars when a train is present (slice 3).

Usage::

    python -m vision.infer --input path/to/image.png
    python -m vision.infer --input path/to/video.mp4 --out-dir ./out
    python -m vision.infer --input path/to/folder_of_mp4s/ --sample-interval-sec 2 --screenshots

Environment: ``VISION_*``, ``VISION_VIDEO_SAMPLE_INTERVAL_SEC``, ``VISION_SCREENSHOT_*`` (see ``vision.config``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vision.config import load_vision_config
from vision.schema import detection_result_to_json_dict
from vision.screenshot import ScreenshotWriter
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


def _infer_one_video(
    video_path: Path,
    model: object,
    *,
    conf: float,
    jsonl_path: Path,
    sample_interval_sec: float,
    shots: ScreenshotWriter | None,
) -> tuple[int, int]:
    """Run sampled inference; return (lines_written, screenshots_saved)."""
    source_slug = video_path.stem
    n = 0
    n_shot = 0
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for fi, res, bgr in infer_video_frames(
            video_path,
            model,
            conf_threshold=conf,
            sample_interval_sec=sample_interval_sec,
        ):
            f.write(
                json.dumps(
                    detection_result_to_json_dict(res),
                    separators=(",", ":"),
                )
            )
            f.write("\n")
            n += 1
            if shots is not None:
                pair = shots.maybe_save(
                    bgr, res, source_slug=source_slug, frame_index=fi
                )
                if pair:
                    n_shot += 1
    return n, n_shot


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
    parser.add_argument(
        "--sample-interval-sec",
        type=float,
        default=None,
        metavar="SEC",
        help=(
            "Approx. seconds between YOLO runs on video (default: "
            "VISION_VIDEO_SAMPLE_INTERVAL_SEC, often 1.0). Use 0 for every frame."
        ),
    )
    parser.add_argument(
        "--screenshots",
        action="store_true",
        help="Save PNG + JSON sidecars when train present (VISION_SCREENSHOT_DIR)",
    )
    parser.add_argument(
        "--screenshot-dir",
        type=Path,
        default=None,
        help="Override VISION_SCREENSHOT_DIR for --screenshots",
    )
    parser.add_argument(
        "--screenshot-min-interval-sec",
        type=float,
        default=None,
        metavar="SEC",
        help=(
            "Min real time between screenshots while train visible (default: "
            "VISION_SCREENSHOT_MIN_INTERVAL_SEC)"
        ),
    )
    args = parser.parse_args(argv)

    inp = args.input.expanduser().resolve()
    out_dir = (args.out_dir or Path("vision_infer_out")).expanduser().resolve()
    weights = (args.model or cfg.model_path or DEFAULT_YOLO_WEIGHTS).strip()
    conf = float(args.conf) if args.conf is not None else float(cfg.conf_threshold)
    sample_interval_sec = (
        float(args.sample_interval_sec)
        if args.sample_interval_sec is not None
        else float(cfg.video_sample_interval_sec)
    )
    shot_dir = (args.screenshot_dir or cfg.screenshot_dir).expanduser().resolve()
    shot_min = (
        float(args.screenshot_min_interval_sec)
        if args.screenshot_min_interval_sec is not None
        else float(cfg.screenshot_min_interval_sec)
    )
    shots: ScreenshotWriter | None = None
    if args.screenshots:
        shots = ScreenshotWriter(shot_dir, min_interval_sec=shot_min)

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
        total_shots = 0
        for p in vids:
            jsonl_path = out_dir / f"{p.stem}_results.jsonl"
            try:
                n, ns = _infer_one_video(
                    p,
                    model,
                    conf=conf,
                    jsonl_path=jsonl_path,
                    sample_interval_sec=sample_interval_sec,
                    shots=shots,
                )
                total_shots += ns
                print(
                    f"[vision.infer] {p.name} → {n} sample(s) → {jsonl_path.name}"
                )
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
            if shots is not None and res.present:
                import cv2

                bgr = cv2.imread(str(p))
                if bgr is not None:
                    if shots.maybe_save(bgr, res, source_slug=stem, frame_index=0):
                        total_shots += 1
        print(f"[vision.infer] wrote outputs under {out_dir}")
        if args.screenshots:
            print(
                f"[vision.infer] screenshots → {shot_dir} ({total_shots} saved this run)",
            )
        return 1 if video_errors else 0

    if is_image_path(inp):
        res = infer_image(inp, model, conf_threshold=conf)
        _write_json(out_dir / "result.json", detection_result_to_json_dict(res))
        if args.annotate:
            _save_annotated(model, inp, out_dir / "annotated.jpg", conf_threshold=conf)
        ns = 0
        if shots is not None and res.present:
            import cv2

            bgr = cv2.imread(str(inp))
            if bgr is not None and shots.maybe_save(
                bgr, res, source_slug=inp.stem, frame_index=0
            ):
                ns = 1
        print(f"[vision.infer] wrote {out_dir / 'result.json'}")
        if args.screenshots:
            print(f"[vision.infer] screenshots → {shot_dir} ({ns} saved this run)")
        return 0

    if is_video_path(inp):
        out_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = out_dir / "results.jsonl"
        try:
            n, ns = _infer_one_video(
                inp,
                model,
                conf=conf,
                jsonl_path=jsonl_path,
                sample_interval_sec=sample_interval_sec,
                shots=shots,
            )
        except Exception as e:
            print(f"[vision.infer] failed: {e}", file=sys.stderr)
            return 6
        print(f"[vision.infer] wrote {n} sample record(s) → {jsonl_path}")
        if args.screenshots:
            print(f"[vision.infer] screenshots → {shot_dir} ({ns} new this run)")
        return 0

    print(f"[vision.infer] unsupported file type: {inp.suffix}", file=sys.stderr)
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
