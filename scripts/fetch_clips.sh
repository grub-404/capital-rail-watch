#!/usr/bin/env bash
# From repo root: download rows from data/clips/manifest.csv via yt-dlp.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -m vision.fetch --from-manifest "$@"
