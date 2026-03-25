# Capital Rail Watch

Live Solari-style train board overlay for OBS, showing real-time Amtrak and MARC arrivals/departures at Washington Union Station.

## Quick Start

### 1. Install Python

- **Windows**: Download from [python.org](https://www.python.org/downloads/). During install, check **"Add Python to PATH"**.
- **Mac**: `brew install python` or download from [python.org](https://www.python.org/downloads/).

### 2. Create a virtual environment and install dependencies

Use a **venv** (virtual environment) so this project’s packages stay isolated from your system Python and from other projects. **`pip`** is the tool that downloads and installs those packages; **`venv`** is the folder + separate `python`/`pip` that you install *into*.

From the **repository root** (not `backend/`):

```bash
python3 -m venv .venv
source .venv/bin/activate    # Mac/Linux — Windows: .venv\Scripts\activate
python -m pip install -r backend/requirements.txt
```

To run **unit tests** (including `tests/vision/` and YOLO inference tests), also install dev dependencies (adds **PyTorch** + Ultralytics; first test run may download `yolov8n.pt`):

```bash
python -m pip install -r requirements-dev.txt
```

Always activate the venv before running the server. If you see **`pyenv: pip: command not found`**, your shell does not have a standalone `pip` on `PATH` for the active Python. Using **`python -m pip`** (as above) runs pip as a module and avoids that. Alternatively, pick the Python version pyenv lists as having pip: `pyenv shell 3.10.14` then retry.

### 3. Set up WMATA key (optional — for Metro data)

Get a free API key from [developer.wmata.com](https://developer.wmata.com/).

**Recommended:** put it in a **`.env` file at the repository root** (next to `README.md`, not inside `backend/`). The server loads this automatically on startup:

```bash
cp .env.example .env
# Edit .env and set: WMATA_KEY=your_key_here
```

`.env` is listed in `.gitignore` so it is not committed.

You can still use a normal environment variable instead (`export WMATA_KEY=...` on Mac/Linux, etc.); if both are set, the existing shell variable wins.

Set **`PORT=8080`** (or any free port 1–65535) in `.env` to change the server port; omit it to use **5555**.

The overlay works without a key — you just won't see Metro predictions.

Optional **`TICKER_*`** variables in the same `.env` file adjust the scrolling ticker (window length, scroll speed, dwell times, etc.). See `.env.example` for names and defaults. The ticker loads them via `GET /api/ticker-config` when the page opens.

### 4. Run the server

With the venv activated, from the repo root:

```bash
python backend/server.py
```

Use this entrypoint so the **background ingest thread** starts. (`flask run` alone will not start ingest unless you wire it up yourself.)

The server starts on `http://localhost:5555`. You should see:

```
[CRW] SQLite DB ready at ...
[CRW] Background ingest every 30s
[CRW] Starting on port 5555
```

The server runs a **background ingest loop** (every 30 seconds by default, configurable with `INGEST_INTERVAL_SEC` in `.env`) that fetches Amtrak, MARC, VRE, and Metro, then writes rows into **`db/trains.db`**. The Solari board and ticker **only read** that data through **`GET /api/overlay-trains`** and **`GET /api/ticker-cache`** — they no longer call Amtraker or feed URLs from the browser.

### 5. Add to OBS

1. In your OBS scene, click **+** under Sources → **Browser**
2. Set the URL to `http://localhost:5555` for the Solari board, or `http://localhost:5555/ticker.html` (alias: `/tracker.html`) for the scrolling Union Station departures/arrivals ticker (same server).
3. Set Width to **1920** and Height to **1080** (ticker can use a shorter height, e.g. **60**, if you only want the bar).
4. The overlay is transparent — layer it on top of your camera source

## Project Structure

```
backend/
  server.py          # Flask API — ingests feeds into SQLite, serves overlay JSON + static files
  requirements.txt   # Python dependencies
overlay/
  index.html         # Solari board overlay (HTML/CSS/JS)
  ticker.html        # Scrolling ticker (reads `/api/ticker-cache` → SQLite)
  ticker-facts.json  # “Did you know?” lines for the ticker (easy to edit; see TICKER-FACTS-README.txt)
  *.svg              # Logos for the ticker (Amtrak, MARC)
db/
  schema.sql         # SQLite schema for train logging
data/clips/
  manifest.csv       # YouTube URLs + time ranges for yt-dlp (slice 2); downloaded mp4s stay local-only
scripts/
  fetch_clips.sh     # Wrapper: python -m vision.fetch --from-manifest …
vision/
  PLAN.md            # Phased plan: YOLO, screenshots, yt-dlp corpus, labeling app
  infer.py           # CLI: python -m vision.infer (slice 1)
  fetch.py           # CLI: python -m vision.fetch (slice 2, yt-dlp)
  clip_fetch.py      # Manifest parsing + yt-dlp argv builder
  screenshot.py      # PNG + JSON sidecars when train present (slice 3)
  yolo_infer.py      # Ultralytics wrapper → DetectionResult
  test_stills/       # Regression images (with_train / without_train)
```

### Vision pipeline (experimental)

The **`vision/`** package will hold pretrained YOLO inference (train / locomotive / railcar), a **yes/no + box + count** payload, **screenshot capture** with JSON sidecars, **yt-dlp**-based test clips, and later a **crowd labeling** webapp. Roadmap and slices: **[vision/PLAN.md](vision/PLAN.md)**.

**Slice 0** — shared JSON contract and env-driven paths (no PyTorch required for schema-only tests):

- **`vision/schema.py`** — `DetectionResult` / `DetectionBox`; `present`, `count`, `boxes[]`, `frame_id`, `source`, `timestamp_utc`, `version`.
- **`vision/labels.py`** — which YOLO class names count as one “train” for aggregation.
- **`vision/config.py`** — reads optional **`VISION_*`** and **`YTDLP_OUTPUT_DIR`** from the repo-root `.env` (see `.env.example`).
- **`vision/fixtures/detection_example.json`** — example payload for tests and API docs.

**Slice 1** — pretrained **Ultralytics YOLO** (default **`yolov8n.pt`**, COCO includes a **train** class). Install dev deps (pulls PyTorch), then:

```bash
python -m vision.infer --input path/to/screenshot.png
python -m vision.infer --input path/to/clips/ --out-dir ./vision_infer_out
python -m vision.infer --input path/to/video.mp4 --out-dir ./out
python -m vision.infer --input data/clips/z7KgEnxJo_s/ --screenshots --sample-interval-sec 1
```

- **Image** (single file): writes **`result.json`** (`DetectionResult`) under `--out-dir`.
- **Directory** of images: writes **`{stem}_result.json`** per file.
- **Video** / **folder of `.mp4`**: writes **`*_results.jsonl`** — one JSON line per **sampled** frame (not every decoded frame by default).
- **Time sampling**: default **`VISION_VIDEO_SAMPLE_INTERVAL_SEC=1.0`** → Ultralytics **`vid_stride ≈ round(fps × interval)`**. Set **`0`** or **`--sample-interval-sec 0`** for every frame (slow).
- **Slice 3 — Screenshots**: **`--screenshots`** saves **`{timestamp}_{source}_{frame}.png`** + matching **`.json`** under **`VISION_SCREENSHOT_DIR`** when **`present`**, throttled by **`VISION_SCREENSHOT_MIN_INTERVAL_SEC`** (default 10s). Override with **`--screenshot-dir`** / **`--screenshot-min-interval-sec`**.
- **`--annotate`**: saves annotated JPEG next to JSON (images only).

**Slice 2** — **yt-dlp** clip harness. **`data/clips/manifest.csv`** lists `clip_name`, `video_id`, `url`, `section` (yt-dlp `--download-sections`, e.g. `*0:00-2:00`), and `notes`. Downloads go to **`data/clips/{video_id}/{clip_name}.mp4`** (or **`YTDLP_OUTPUT_DIR`**). Live watch URLs may fail until a VOD exists—update the CSV when the replay is up.

```bash
python -m pip install -r requirements-dev.txt   # includes yt-dlp
python -m vision.fetch --from-manifest --dry-run              # print commands + append fetch_log.csv
./scripts/fetch_clips.sh --dry-run
python -m vision.fetch --from-manifest                        # actually run yt-dlp
```

Regression stills live under **`vision/test_stills/`**. If COCO ever mis-fires on a true negative, you can add **`vision/test_stills/expectations.json`** with per-file `max_train_detections` under `without_train` (see `tests/vision/test_infer.py`).

Run vision tests from the repo root (first run may download `yolov8n.pt`):

```bash
python -m pytest tests/ -q
```

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Serves the overlay |
| `GET /ticker.html` | Union Station scrolling ticker (reads SQLite via `/api/ticker-cache`) |
| `GET /api/overlay-trains` | Solari board: trains from SQLite in a time window, split by `amtrak` / `marc` / `vre` / `metro` |
| `GET /api/marc` | Live MARC JSON (still fetches GTFS-RT; also logged to SQLite by background ingest) |
| `GET /api/vre` | Live VRE JSON (same pattern) |
| `GET /api/metro` | WMATA predictions JSON (proxied; requires `WMATA_KEY`) |
| `GET /api/stats` | Today's train count and delay stats |
| `GET /api/health` | Health check |
| `GET /api/ticker-config` | Ticker settings from `.env` (`TICKER_*`, optional) |
| `GET /api/ticker-cache` | Ticker: trains from SQLite in the configured time window |
| `POST /api/log` | Log external train rows (optional; ingest covers Amtrak server-side) |

## Troubleshooting

- **`pip` not found (pyenv / Mac)**: Use `python3 -m venv .venv`, then `source .venv/bin/activate`, then `python -m pip install -r backend/requirements.txt` from the repo root (see step 2).
- **`pip` not found (Windows)**: Use `python -m pip install -r backend/requirements.txt`
- **Port in use**: Add `PORT=8080` to `.env`, or run `PORT=8080 python backend/server.py` (Mac) / `set PORT=8080 && python backend/server.py` (Windows cmd)
- **GTFS download slow on first run**: The server downloads MARC schedule data (~2 MB) on the first API call. This is cached in memory for the session.

- **Board or ticker empty**: Both UIs read **`db/trains.db`** only. If ingest has not run yet, or every provider failed, or all stored event times fall outside the window, lists stay empty. Check server logs for ingest errors, confirm **`INGEST_INTERVAL_SEC`** is set, and that **`WMATA_KEY`** is set if you expect Metro. The first **`python backend/server.py`** run performs an immediate ingest, then repeats on the interval.
