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

The overlay works without a key — you just won't see Metro predictions.

Optional **`TICKER_*`** variables in the same `.env` file adjust the scrolling ticker (window length, scroll speed, dwell times, etc.). See `.env.example` for names and defaults. The ticker loads them via `GET /api/ticker-config` when the page opens.

### 4. Run the server

With the venv activated, from the repo root:

```bash
python backend/server.py
```

The server starts on `http://localhost:5555`. You should see:

```
[CRW] SQLite DB ready at ...
[CRW] Starting on port 5555
```

### 5. Add to OBS

1. In your OBS scene, click **+** under Sources → **Browser**
2. Set the URL to `http://localhost:5555` for the Solari board, or `http://localhost:5555/ticker.html` (alias: `/tracker.html`) for the scrolling Union Station departures/arrivals ticker (same server).
3. Set Width to **1920** and Height to **1080** (ticker can use a shorter height, e.g. **60**, if you only want the bar).
4. The overlay is transparent — layer it on top of your camera source

## Project Structure

```
backend/
  server.py          # Flask API — proxies MARC GTFS-RT, serves overlay
  requirements.txt   # Python dependencies
overlay/
  index.html         # Solari board overlay (HTML/CSS/JS)
  ticker.html        # Scrolling ticker (same data sources as index: Amtraker + /api/marc + /api/metro)
  ticker-facts.json  # “Did you know?” lines for the ticker (easy to edit; see TICKER-FACTS-README.txt)
  *.svg              # Logos for the ticker (Amtrak, MARC)
db/
  schema.sql         # SQLite schema for train logging
```

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Serves the overlay |
| `GET /ticker.html` | Union Station scrolling ticker (loads Amtraker in-browser + `/api/marc` + `/api/metro`) |
| `GET /api/marc` | Live MARC train data (from GTFS-RT) |
| `GET /api/vre` | VRE trains at Union Station (GTFS static + GTFS-RT) |
| `GET /api/metro` | WMATA Metro predictions (proxied, requires `WMATA_KEY`) |
| `GET /api/stats` | Today's train count and delay stats |
| `GET /api/health` | Health check |
| `GET /api/ticker-config` | Ticker settings from `.env` (`TICKER_*`, optional) |
| `POST /api/log` | Log external train data (e.g. Amtrak) |

## Troubleshooting

- **`pip` not found (pyenv / Mac)**: Use `python3 -m venv .venv`, then `source .venv/bin/activate`, then `python -m pip install -r backend/requirements.txt` from the repo root (see step 2).
- **`pip` not found (Windows)**: Use `python -m pip install -r backend/requirements.txt`
- **Port in use**: Set a custom port with `PORT=8080 python backend/server.py` (Mac) or `set PORT=8080 && python backend/server.py` (Windows)
- **GTFS download slow on first run**: The server downloads MARC schedule data (~2 MB) on the first API call. This is cached in memory for the session.
