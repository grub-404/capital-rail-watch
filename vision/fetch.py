"""
Download short YouTube segments with yt-dlp into ``data/clips/{video_id}/{clip}.mp4``.

Reads committed ``data/clips/manifest.csv`` (or ``--manifest``). Requires ``yt-dlp`` on PATH
(``pip install yt-dlp`` via ``requirements-dev.txt``).

Examples::

    python -m vision.fetch --from-manifest data/clips/manifest.csv --dry-run
    python -m vision.fetch --from-manifest data/clips/manifest.csv
    python -m vision.fetch --url "https://www.youtube.com/watch?v=..." \\
        --clip-name my_clip --section "*0:00-1:30"

**Live streams:** a “live” watch URL may not yield a stable VOD clip until the stream ends.
Prefer the archived replay URL in ``manifest.csv`` and set ``section`` to a short window.

Environment: ``YTDLP_OUTPUT_DIR`` (default ``data/clips``) — output root; manifest path is separate.
"""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from vision.clip_fetch import (
    ClipSpec,
    ManifestError,
    build_ytdlp_command,
    ensure_clip_name_safe,
    iter_clip_specs,
    output_video_path,
    read_clip_manifest,
    youtube_video_id,
)
from vision.config import PROJECT_ROOT, load_vision_config


def _append_fetch_log(
    log_path: Path,
    row: dict[str, str],
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "utc_iso",
                "clip_name",
                "video_id",
                "url",
                "section",
                "dry_run",
                "exit_code",
                "output_path",
                "notes",
            ],
        )
        if new_file:
            w.writeheader()
        w.writerow(row)


def main(argv: list[str] | None = None) -> int:
    cfg = load_vision_config()
    base = cfg.ytdlp_output_dir
    default_manifest = PROJECT_ROOT / "data" / "clips" / "manifest.csv"

    p = argparse.ArgumentParser(description="Fetch YouTube clip segments via yt-dlp")
    p.add_argument(
        "--from-manifest",
        "--manifest",
        dest="manifest",
        nargs="?",
        const=default_manifest,
        default=None,
        type=Path,
        metavar="PATH",
        help=(
            "CSV manifest (clip_name,video_id,url,section,notes). "
            f"If the flag is given with no path, uses {default_manifest.name} under data/clips/"
        ),
    )
    p.add_argument("--url", type=str, default=None, help="Single video URL (requires --clip-name)")
    p.add_argument(
        "--clip-name",
        type=str,
        default=None,
        help="With --url: output base name; with --from-manifest: process only this row",
    )
    p.add_argument(
        "--section",
        type=str,
        default=None,
        help="Override yt-dlp section for all processed rows (e.g. 0:00-2:00 or *0:00-2:00)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"Output root (default: YTDLP_OUTPUT_DIR or {base})",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and append fetch log; do not run yt-dlp",
    )
    p.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Append CSV log path (default: <out-dir>/fetch_log.csv)",
    )
    args = p.parse_args(argv)

    if args.url and args.manifest is not None:
        p.error("use only one of --from-manifest or --url")
    if args.url is None and args.manifest is None:
        p.error("provide --from-manifest or --url")

    out_root = (args.out_dir or base).expanduser().resolve()
    log_path = (args.log or (out_root / "fetch_log.csv")).resolve()

    if args.url:
        if not args.clip_name:
            print("[vision.fetch] --url requires --clip-name", file=sys.stderr)
            return 2
        vid = youtube_video_id(args.url)
        specs = [
            ClipSpec(
                clip_name=args.clip_name,
                video_id=vid,
                url=args.url.strip(),
                section=(args.section or "").strip(),
                notes="cli",
            )
        ]
    else:
        mp = args.manifest.expanduser().resolve()
        try:
            specs = read_clip_manifest(mp)
        except FileNotFoundError:
            print(f"[vision.fetch] manifest not found: {mp}", file=sys.stderr)
            return 3
        except ManifestError as e:
            print(f"[vision.fetch] manifest error: {e}", file=sys.stderr)
            return 3
        try:
            specs = list(iter_clip_specs(specs, only_clip=args.clip_name))
        except ManifestError as e:
            print(f"[vision.fetch] {e}", file=sys.stderr)
            return 3
        if not specs:
            print("[vision.fetch] no rows to process", file=sys.stderr)
            return 4

    utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    for spec in specs:
        try:
            out_mp4 = output_video_path(out_root, spec)
            cmd = build_ytdlp_command(
                spec, output_mp4=out_mp4, section=args.section
            )
        except ManifestError as e:
            print(f"[vision.fetch] skip {spec.clip_name!r}: {e}", file=sys.stderr)
            _append_fetch_log(
                log_path,
                {
                    "utc_iso": utc,
                    "clip_name": spec.clip_name,
                    "video_id": spec.video_id,
                    "url": spec.url,
                    "section": spec.section,
                    "dry_run": "true" if args.dry_run else "false",
                    "exit_code": "skip",
                    "output_path": "",
                    "notes": str(e),
                },
            )
            continue

        print(shlex.join(cmd))
        if args.dry_run:
            _append_fetch_log(
                log_path,
                {
                    "utc_iso": utc,
                    "clip_name": spec.clip_name,
                    "video_id": spec.video_id,
                    "url": spec.url,
                    "section": args.section or spec.section,
                    "dry_run": "true",
                    "exit_code": "",
                    "output_path": str(out_mp4),
                    "notes": spec.notes,
                },
            )
            continue

        try:
            proc = subprocess.run(cmd, check=False)
        except FileNotFoundError:
            print(
                "[vision.fetch] yt-dlp not found; install: pip install yt-dlp",
                file=sys.stderr,
            )
            return 5
        _append_fetch_log(
            log_path,
            {
                "utc_iso": utc,
                "clip_name": spec.clip_name,
                "video_id": spec.video_id,
                "url": spec.url,
                "section": args.section or spec.section,
                "dry_run": "false",
                "exit_code": str(proc.returncode),
                "output_path": str(out_mp4),
                "notes": spec.notes,
            },
        )
        if proc.returncode != 0:
            return proc.returncode or 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
