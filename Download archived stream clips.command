#!/bin/bash
# Double-click in Finder: downloads all rows from data/clips/archived_stream_clips.sh
cd "$(dirname "$0")" || exit 1
chmod +x scripts/download_archived_stream_clips.sh 2>/dev/null || true
./scripts/download_archived_stream_clips.sh "$@"
r=$?
echo ""
read -r -p "Press Enter to close…" || true
exit "$r"
