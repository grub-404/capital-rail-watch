"""Slice 2: yt-dlp manifest parsing and command building (no network)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from vision.clip_fetch import (
    ClipSpec,
    ManifestError,
    build_ytdlp_command,
    output_video_path,
    read_clip_manifest,
    youtube_video_id,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTED_MANIFEST = REPO_ROOT / "data" / "clips" / "manifest.csv"


def test_youtube_video_id_watch() -> None:
    assert (
        youtube_video_id("https://www.youtube.com/watch?v=z7KgEnxJo_s")
        == "z7KgEnxJo_s"
    )


def test_youtube_video_id_short() -> None:
    assert youtube_video_id("https://youtu.be/ClennbOVxcI") == "ClennbOVxcI"


def test_youtube_video_id_bad() -> None:
    with pytest.raises(ManifestError):
        youtube_video_id("https://example.com/")


def test_read_committed_manifest() -> None:
    specs = read_clip_manifest(COMMITTED_MANIFEST)
    assert len(specs) >= 2
    ids = {s.video_id for s in specs}
    assert "z7KgEnxJo_s" in ids
    assert "ClennbOVxcI" in ids
    for s in specs:
        assert s.section.startswith("*")
        assert "youtube.com" in s.url or "youtu.be" in s.url


def test_video_id_mismatch_raises(tmp_path: Path) -> None:
    p = tmp_path / "m.csv"
    p.write_text(
        "clip_name,video_id,url,section,notes\n"
        "bad,aaaaaaaaaaa,https://www.youtube.com/watch?v=zzzzzzzzzzz,*0:00-1:00,\n",
        encoding="utf-8",
    )
    with pytest.raises(ManifestError, match="does not match"):
        read_clip_manifest(p)


def test_build_ytdlp_command(tmp_path: Path) -> None:
    spec = ClipSpec(
        clip_name="sample",
        video_id="z7KgEnxJo_s",
        url="https://www.youtube.com/watch?v=z7KgEnxJo_s",
        section="*0:00-0:30",
        notes="",
    )
    out = output_video_path(tmp_path, spec)
    cmd = build_ytdlp_command(spec, output_mp4=out)
    assert cmd[0] == "yt-dlp"
    assert "--download-sections" in cmd
    i = cmd.index("--download-sections")
    assert cmd[i + 1] == "*0:00-0:30"
    assert spec.url in cmd
    assert any("sample.%(ext)s" in a or "sample" in a for a in cmd)


def test_dry_run_writes_log(tmp_path: Path) -> None:
    from vision.fetch import main

    man = tmp_path / "manifest.csv"
    man.write_text(
        "clip_name,video_id,url,section,notes\n"
        "t,z7KgEnxJo_s,https://www.youtube.com/watch?v=z7KgEnxJo_s,*0:00-0:05,test\n",
        encoding="utf-8",
    )
    log = tmp_path / "fetch_log.csv"
    code = main(
        [
            "--from-manifest",
            str(man),
            "--out-dir",
            str(tmp_path / "out"),
            "--dry-run",
            "--log",
            str(log),
        ]
    )
    assert code == 0
    assert log.is_file()
    rows = list(csv.DictReader(log.open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["dry_run"] == "true"
    assert rows[0]["clip_name"] == "t"


@pytest.mark.integration
def test_ytdlp_e2e_smoke() -> None:
    """Set YTDLP_E2E=1 and network; runs yt-dlp (skipped by default)."""
    import os

    if os.environ.get("YTDLP_E2E", "").strip() != "1":
        pytest.skip("set YTDLP_E2E=1 to run live yt-dlp smoke")

    pytest.importorskip("yt_dlp", reason="yt-dlp not installed")
    import subprocess

    r = subprocess.run(
        ["yt-dlp", "--version"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert r.returncode == 0
