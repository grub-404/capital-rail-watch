#!/usr/bin/env bash
# Download many 2-minute segments from the archived stream (z7KgEnxJo_s).
# Prereq: repo-root venv with yt-dlp (pip install -r requirements-dev.txt).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
python -m vision.fetch --from-manifest "$REPO/data/clips/archived_stream_clips.csv" "$@"
