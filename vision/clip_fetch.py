"""
YouTube clip specs → yt-dlp command lines. Used by ``python -m vision.fetch`` (slice 2).
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# watch?v=, youtu.be/, shorts/ (id still 11 chars in path)
_VIDEO_ID_RE = re.compile(r"(?:v=|youtu\.be/|shorts/)([a-zA-Z0-9_-]{11})")


@dataclass(frozen=True)
class ClipSpec:
    clip_name: str
    video_id: str
    url: str
    section: str  # yt-dlp --download-sections, e.g. *0:00-2:00
    notes: str = ""


class ManifestError(ValueError):
    pass


def youtube_video_id(url: str) -> str:
    m = _VIDEO_ID_RE.search(url.strip())
    if not m:
        raise ManifestError(f"could not parse YouTube video id from URL: {url!r}")
    return m.group(1)


def read_clip_manifest(path: Path) -> list[ClipSpec]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[ClipSpec] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"clip_name", "video_id", "url", "section"}
        if reader.fieldnames is None:
            raise ManifestError("manifest has no header row")
        headers = {h.strip() for h in reader.fieldnames if h}
        missing = required - headers
        if missing:
            raise ManifestError(f"manifest missing columns: {sorted(missing)}")
        for i, raw in enumerate(reader, start=2):
            clip_name = (raw.get("clip_name") or "").strip()
            video_id = (raw.get("video_id") or "").strip()
            url = (raw.get("url") or "").strip()
            section = (raw.get("section") or "").strip()
            notes = (raw.get("notes") or "").strip()
            if not clip_name or not url:
                continue
            parsed_id = youtube_video_id(url)
            if video_id and video_id != parsed_id:
                raise ManifestError(
                    f"row {i}: video_id {video_id!r} does not match URL id {parsed_id!r}"
                )
            vid = video_id or parsed_id
            rows.append(
                ClipSpec(
                    clip_name=clip_name,
                    video_id=vid,
                    url=url,
                    section=section,
                    notes=notes,
                )
            )
    return rows


def ensure_clip_name_safe(name: str) -> str:
    safe = re.sub(r"[^\w.\-]+", "_", name.strip(), flags=re.UNICODE).strip("._")
    if not safe:
        raise ManifestError(f"invalid clip_name: {name!r}")
    return safe


def output_video_path(base_dir: Path, spec: ClipSpec) -> Path:
    """Final merged file path (mp4)."""
    name = ensure_clip_name_safe(spec.clip_name)
    return (base_dir / spec.video_id / f"{name}.mp4").resolve()


def build_ytdlp_command(
    spec: ClipSpec,
    *,
    output_mp4: Path,
    section: str | None = None,
) -> list[str]:
    """
    yt-dlp argv to download a time range and merge to mp4.

    ``section`` overrides ``spec.section`` (must be non-empty for actual downloads).
    """
    sec = (section if section is not None else spec.section).strip()
    if not sec:
        raise ManifestError(
            "download_sections empty: set `section` in manifest (e.g. *0:00-2:00) "
            "or pass --section"
        )
    if not sec.startswith("*"):
        sec = f"*{sec}"

    output_mp4 = output_mp4.resolve()
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    outtmpl = str(output_mp4.with_suffix("")) + ".%(ext)s"

    return [
        "yt-dlp",
        "-f",
        "bv*[ext=mp4]+ba/bv*+ba/b",
        "--merge-output-format",
        "mp4",
        "--download-sections",
        sec,
        "--no-playlist",
        "--newline",
        "-o",
        outtmpl,
        spec.url,
    ]


def iter_clip_specs(
    specs: list[ClipSpec],
    *,
    only_clip: str | None = None,
) -> Iterator[ClipSpec]:
    if not only_clip:
        yield from specs
        return
    want = only_clip.strip()
    for s in specs:
        if s.clip_name == want or ensure_clip_name_safe(s.clip_name) == want:
            yield s
            return
    raise ManifestError(f"no manifest row with clip_name={only_clip!r}")
