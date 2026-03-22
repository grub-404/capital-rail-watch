# Capital Rail Watch

Live Solari-style train board overlay for OBS, showing real-time Amtrak and MARC arrivals/departures at Washington Union Station.

## Quick Start

### 1. Install Python

- **Windows**: Download from [python.org](https://www.python.org/downloads/). During install, check **"Add Python to PATH"**.
- **Mac**: `brew install python` or download from [python.org](https://www.python.org/downloads/).

### 2. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 3. Run the server

```bash
python backend/server.py
```

The server starts on `http://localhost:5555`. You should see:

```
[CRW] SQLite DB ready at ...
[CRW] Starting on port 5555
```

### 4. Add to OBS

1. In your OBS scene, click **+** under Sources → **Browser**
2. Set the URL to `http://localhost:5555`
3. Set Width to **1920** and Height to **1080**
4. The overlay is transparent — layer it on top of your camera source

## Project Structure

```
backend/
  server.py          # Flask API — proxies MARC GTFS-RT, serves overlay
  requirements.txt   # Python dependencies
overlay/
  index.html         # Solari board overlay (HTML/CSS/JS)
db/
  schema.sql         # SQLite schema for train logging
```

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Serves the overlay |
| `GET /api/marc` | Live MARC train data (from GTFS-RT) |
| `GET /api/stats` | Today's train count and delay stats |
| `GET /api/health` | Health check |
| `POST /api/log` | Log external train data (e.g. Amtrak) |

## Troubleshooting

- **`pip` not found (Windows)**: Use `python -m pip install -r requirements.txt`
- **Port in use**: Set a custom port with `PORT=8080 python backend/server.py` (Mac) or `set PORT=8080 && python backend/server.py` (Windows)
- **GTFS download slow on first run**: The server downloads MARC schedule data (~2 MB) on the first API call. This is cached in memory for the session.
