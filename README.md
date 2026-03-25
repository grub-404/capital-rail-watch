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

To run **unit tests** (including `tests/vision/`), also install dev dependencies:

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
vision/
  PLAN.md            # Phased plan: YOLO, screenshots, yt-dlp corpus, labeling app
  schema + config    # Detection JSON contract and env-driven paths (slice 0)
```

### Vision pipeline (experimental)

The **`vision/`** package will hold pretrained YOLO inference (train / locomotive / railcar), a **yes/no + box + count** payload, **screenshot capture** with JSON sidecars, **yt-dlp**-based test clips, and later a **crowd labeling** webapp. Roadmap and slices: **[vision/PLAN.md](vision/PLAN.md)**.

**Slice 0 (current)** defines the shared contract only—no model weights required:

- **`vision/schema.py`** — `DetectionResult` / `DetectionBox`; `present`, `count`, `boxes[]`, `frame_id`, `source`, `timestamp_utc`, `version`.
- **`vision/labels.py`** — which YOLO class names count as one “train” for aggregation.
- **`vision/config.py`** — reads optional **`VISION_*`** and **`YTDLP_OUTPUT_DIR`** from the repo-root `.env` (see `.env.example`).
- **`vision/fixtures/detection_example.json`** — example payload for tests and API docs.

Run vision unit tests from the repo root:

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
