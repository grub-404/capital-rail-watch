"""
Capital Rail Watch — Backend proxy
Fetches MARC GTFS-RT protobuf feed, decodes to JSON, serves to overlay.
Logs all train sightings to SQLite for stats and history.
"""
import csv, io, json, sqlite3, time, zipfile, urllib.request, os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from google.transit import gtfs_realtime_pb2

app = Flask(__name__)
CORS(app)

# ─── SQLite setup ─────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Optional secrets and local config (repo root — same folder as README.md)
load_dotenv(PROJECT_ROOT / ".env")

DB_PATH = PROJECT_ROOT / "db" / "trains.db"
SCHEMA_PATH = PROJECT_ROOT / "db" / "schema.sql"


def get_db():
    """Open a connection to the SQLite DB (one per request)."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create tables from schema.sql if they don't exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = get_db()
    with open(SCHEMA_PATH) as f:
        conn.executescript(f.read())
    conn.close()
    print(f"[CRW] SQLite DB ready at {DB_PATH}")


def upsert_train(conn, t):
    """Insert or update a train record for today.

    Match on train_num + direction + today's date (in first_seen_at).
    If the row exists, update delay and timestamps; otherwise insert.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    row = conn.execute(
        """SELECT id FROM trains
           WHERE train_num = ? AND direction = ?
             AND date(first_seen_at) = ?""",
        (t["num"], t["direction"], today),
    ).fetchone()

    if row:
        conn.execute(
            """UPDATE trains
               SET delay_minutes = ?, estimated_time = ?, actual_time = ?,
                   track = ?, last_updated_at = datetime('now')
               WHERE id = ?""",
            (t.get("delay", 0), t.get("estTime", ""), t.get("actTime", ""),
             t.get("platform", ""), row["id"]),
        )
    else:
        conn.execute(
            """INSERT INTO trains
               (train_num, route_name, system, direction, scheduled_time,
                estimated_time, actual_time, delay_minutes, track,
                origin, destination)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (t["num"], t.get("route", ""), t.get("system", ""),
             t["direction"], t.get("schTime", ""),
             t.get("estTime", ""), t.get("actTime", ""),
             t.get("delay", 0), t.get("platform", ""),
             t.get("origin", ""), t.get("destination", "")),
        )


def log_trains(train_list):
    """Upsert a list of train dicts into the DB."""
    conn = get_db()
    try:
        for t in train_list:
            upsert_train(conn, t)
        conn.commit()
    finally:
        conn.close()


# ─── Config ──────────────────────────────────
MARC_TU = "https://mdotmta-gtfs-rt.s3.amazonaws.com/MARC+RT/marc-tu.pb"
MARC_GTFS = "https://feeds.mta.maryland.gov/gtfs/marc"
WAS_STOP = "11958"
PENN_ROUTE = "11705"
ET = timezone(timedelta(hours=-4))
WMATA_KEY = os.environ.get("WMATA_KEY", "")
WMATA_PREDICTIONS = "https://api.wmata.com/StationPrediction.svc/json/GetPrediction/B35"

# ─── Static GTFS data (loaded once) ─────────
stops = {}       # stop_id → short name
schedules = {}   # trip_id → { stop_id: (arr_str, dep_str) }
gtfs_loaded = False

def clean_stop_name(raw):
    """PENN STATION MARC sb → BAL, UNION STATION MARC Washington → DC, etc."""
    n = raw.upper()
    if "UNION STATION" in n: return "DC"
    if "PENN STATION" in n: return "BAL"
    if "CAMDEN STATION" in n: return "BAL"
    if "PERRYVILLE" in n: return "PVL"
    if "ABERDEEN" in n: return "ABD"
    if "EDGEWOOD" in n: return "EDG"
    if "MARTIN AIRPORT" in n: return "BWI"
    if "BWI" in n: return "BWI"
    if "WEST BALTIMORE" in n: return "BAL"
    if "HALETHORPE" in n: return "HLT"
    if "ODENTON" in n: return "ODN"
    if "BOWIE" in n: return "BWE"
    if "SEABROOK" in n: return "SBK"
    if "NEW CARROLLTON" in n: return "NCR"
    if "SILVER SPRING" in n: return "SSP"
    if "ROCKVILLE" in n: return "RKV"
    if "GERMANTOWN" in n: return "GMT"
    if "BRUNSWICK" in n: return "BRN"
    if "FREDERICK" in n: return "FDK"
    # Fallback: first 3 chars
    return raw.split()[0][:3].upper()

def load_gtfs():
    global gtfs_loaded
    if gtfs_loaded:
        return
    try:
        print("[CRW] Downloading MARC GTFS static data...")
        data = urllib.request.urlopen(MARC_GTFS, timeout=15).read()
        zf = zipfile.ZipFile(io.BytesIO(data))

        # stops.txt → stop_id to short name
        with zf.open("stops.txt") as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
            for row in reader:
                stops[row["stop_id"]] = clean_stop_name(row["stop_name"])

        # stop_times.txt → scheduled times per trip at each stop
        with zf.open("stop_times.txt") as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
            for row in reader:
                tid = row["trip_id"]
                sid = row["stop_id"]
                if tid not in schedules:
                    schedules[tid] = {}
                schedules[tid][sid] = (row["arrival_time"], row["departure_time"])

        gtfs_loaded = True
        print(f"[CRW] Loaded {len(stops)} stops, {len(schedules)} trips")
    except Exception as e:
        print(f"[CRW] GTFS load error: {e}")

def get_origin_for_trip(trip_id):
    """Find the first stop in a trip's schedule → that's the origin."""
    sched = schedules.get(trip_id, {})
    if not sched:
        return "BAL"  # Penn Line default
    earliest_time = "99:99:99"
    earliest_stop = None
    for sid, (arr, dep) in sched.items():
        t = dep if dep else arr
        if t and t < earliest_time:
            earliest_time = t
            earliest_stop = sid
    return stops.get(earliest_stop, "BAL") if earliest_stop else "BAL"


def get_destination_for_trip(trip_id):
    """Find the last stop in a trip's schedule → that's the destination."""
    sched = schedules.get(trip_id, {})
    if not sched:
        return "BAL"
    latest_time = ""
    latest_stop = None
    for sid, (arr, dep) in sched.items():
        t = arr if arr else dep
        if t and t > latest_time:
            latest_time = t
            latest_stop = sid
    return stops.get(latest_stop, "BAL") if latest_stop else "BAL"

def hhmm_to_epoch(date_str, time_str):
    """Convert GTFS date (YYYYMMDD) + time (HH:MM:SS) to epoch seconds."""
    if not date_str or not time_str:
        return 0
    try:
        h, m, s = [int(x) for x in time_str.split(":")]
        dt = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=ET)
        dt = dt.replace(hour=h % 24, minute=m, second=s)
        if h >= 24:
            dt += timedelta(days=1)
        return int(dt.timestamp())
    except Exception:
        return 0

def epoch_to_iso(ts):
    """Epoch seconds → ISO string in ET."""
    if not ts:
        return ""
    return datetime.fromtimestamp(ts, tz=ET).isoformat()


# ─── VRE static GTFS + realtime (Union Station) ─────────────────
VRE_GTFS = "https://gtfs.vre.org/containercdngtfsupload/google_transit.zip"
VRE_TU = "https://gtfs.vre.org/containercdngtfsupload/TripUpdateFeed"

vre_gtfs_loaded = False
vre_stops_meta = {}  # stop_id → {stop_name, parent_station}
vre_schedules_vre = {}  # trip_id → {stop_id: (arr, dep)}
vre_trips_vre = {}  # trip_id → row dict
vre_routes_vre = {}  # route_id → row dict
vre_union_stop_ids = set()


def _vre_union_parent_row(row):
    name = (row.get("stop_name") or "").strip().lower()
    return name == "union station" and str(row.get("location_type") or "") == "1"


def load_vre_gtfs():
    global vre_gtfs_loaded, vre_stops_meta, vre_schedules_vre, vre_trips_vre
    global vre_routes_vre, vre_union_stop_ids
    if vre_gtfs_loaded:
        return
    try:
        print("[CRW] Downloading VRE GTFS static data...")
        data = urllib.request.urlopen(VRE_GTFS, timeout=30).read()
        zf = zipfile.ZipFile(io.BytesIO(data))

        with zf.open("stops.txt") as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
            stop_rows = list(reader)

        parents = {r["stop_id"] for r in stop_rows if _vre_union_parent_row(r)}
        vre_union_stop_ids = set(parents)
        for r in stop_rows:
            if r.get("parent_station") in parents:
                vre_union_stop_ids.add(r["stop_id"])
            vre_stops_meta[r["stop_id"]] = {
                "stop_name": r.get("stop_name") or "",
                "parent_station": r.get("parent_station") or "",
            }

        with zf.open("stop_times.txt") as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
            for row in reader:
                tid = row["trip_id"]
                sid = row["stop_id"]
                if tid not in vre_schedules_vre:
                    vre_schedules_vre[tid] = {}
                vre_schedules_vre[tid][sid] = (
                    row.get("arrival_time") or "",
                    row.get("departure_time") or "",
                )

        with zf.open("trips.txt") as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
            for row in reader:
                vre_trips_vre[row["trip_id"]] = row

        with zf.open("routes.txt") as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
            for row in reader:
                vre_routes_vre[row["route_id"]] = row

        vre_gtfs_loaded = True
        print(
            f"[CRW] VRE GTFS: {len(vre_stops_meta)} stops, "
            f"{len(vre_schedules_vre)} trips, union stop ids={len(vre_union_stop_ids)}"
        )
    except Exception as e:
        print(f"[CRW] VRE GTFS load error: {e}")


def _vre_short_stop(stop_id):
    name = (vre_stops_meta.get(stop_id) or {}).get("stop_name", "").upper()
    if "UNION" in name and "STATION" in name:
        return "DC"
    if not name:
        return "?"
    return name.split()[0][:3]


def vre_origin_dest_for_trip(trip_id):
    sched = vre_schedules_vre.get(trip_id, {})
    if not sched:
        return "?", "?"
    earliest_time = "99:99:99"
    earliest_stop = None
    latest_time = ""
    latest_stop = None
    for sid, (arr, dep) in sched.items():
        t_dep = dep if dep else arr
        if t_dep and t_dep < earliest_time:
            earliest_time = t_dep
            earliest_stop = sid
        t_arr = arr if arr else dep
        if t_arr and t_arr > latest_time:
            latest_time = t_arr
            latest_stop = sid
    o = _vre_short_stop(earliest_stop) if earliest_stop else "?"
    d = _vre_short_stop(latest_stop) if latest_stop else "?"
    return o, d


def _stu_time_or_delay(stu, field, sch_epoch):
    """Return predicted epoch from StopTimeUpdate arrival/departure .time or .delay."""
    if not stu.HasField(field):
        return 0
    block = getattr(stu, field)
    if block.HasField("time") and block.time:
        return int(block.time)
    if block.HasField("delay") and sch_epoch:
        return sch_epoch + int(block.delay)
    return 0


@app.route("/api/vre")
def vre_trains():
    """VRE trains at Union Station from GTFS-RT + static schedule (same shape as /api/marc)."""
    load_vre_gtfs()
    if not vre_union_stop_ids:
        return jsonify({"error": "VRE GTFS not loaded", "trains": []}), 503

    try:
        data = urllib.request.urlopen(VRE_TU, timeout=15).read()
    except Exception as e:
        return jsonify({"error": str(e), "trains": []}), 502

    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(data)

    today_yyyymmdd = datetime.now(ET).strftime("%Y%m%d")
    trains_out = []
    seen_trip_dir = set()

    for ent in feed.entity:
        if not ent.HasField("trip_update"):
            continue
        tu = ent.trip_update
        trip = tu.trip
        tid = str(trip.trip_id or "")
        if not tid:
            continue

        union_stu = None
        union_sid = None
        for stu in tu.stop_time_update:
            sid = str(stu.stop_id)
            if sid in vre_union_stop_ids:
                union_stu = stu
                union_sid = sid
                break
        if not union_stu or not union_sid:
            continue

        start_d = trip.start_date if trip.start_date else today_yyyymmdd
        trip_sched = vre_schedules_vre.get(tid, {})
        was_sched = trip_sched.get(union_sid, ("", ""))
        sch_arr_epoch = hhmm_to_epoch(start_d, was_sched[0])
        sch_dep_epoch = hhmm_to_epoch(start_d, was_sched[1])

        pred_arr = _stu_time_or_delay(union_stu, "arrival", sch_arr_epoch)
        pred_dep = _stu_time_or_delay(union_stu, "departure", sch_dep_epoch)

        if pred_dep and not pred_arr:
            direction = "dep"
            sch_epoch = sch_dep_epoch
            act_epoch = pred_dep
        else:
            direction = "arr"
            sch_epoch = sch_arr_epoch
            act_epoch = pred_arr if pred_arr else pred_dep

        if not act_epoch:
            continue

        dedupe_key = (str(tid), direction)
        if dedupe_key in seen_trip_dir:
            continue
        seen_trip_dir.add(dedupe_key)

        delay = round((act_epoch - sch_epoch) / 60) if (sch_epoch and act_epoch) else 0

        trip_row = vre_trips_vre.get(str(tid), {})
        train_num = (
            (trip_row.get("trip_short_name") or "").strip()
            or "".join(c for c in tid if c.isdigit())
            or tid
        )
        route_id = str(trip.route_id).strip() if trip.route_id else str(trip_row.get("route_id") or "")
        route = vre_routes_vre.get(route_id, {})
        rn = (route.get("route_short_name") or route.get("route_long_name") or "VRE").strip()
        route_label = f"VRE {rn}" if rn.upper() != "VRE" else "VRE"

        trip_o, trip_d = vre_origin_dest_for_trip(tid)
        if direction == "dep":
            origin, destination = "DC", trip_d
        else:
            origin, destination = trip_o, "DC"

        trains_out.append(
            {
                "num": str(train_num),
                "route": route_label,
                "direction": direction,
                "schTime": epoch_to_iso(sch_epoch),
                "actTime": epoch_to_iso(act_epoch),
                "delay": delay,
                "origin": origin,
                "destination": destination,
                "system": "vre",
                "platform": "",
                "status": "enroute",
            }
        )

    try:
        log_trains(trains_out)
    except Exception as e:
        print(f"[CRW] VRE DB log error: {e}")

    return jsonify({"trains": trains_out, "feed_time": feed.header.timestamp})


@app.route("/api/marc")
def marc_trains():
    load_gtfs()
    try:
        data = urllib.request.urlopen(MARC_TU, timeout=10).read()
    except Exception as e:
        return jsonify({"error": str(e), "trains": []}), 502

    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(data)

    trains = []
    for ent in feed.entity:
        if not ent.HasField("trip_update"):
            continue
        tu = ent.trip_update
        trip = tu.trip
        route_id = str(trip.route_id)

        # Find Union Station stop update
        was_update = None
        for stu in tu.stop_time_update:
            if str(stu.stop_id) == WAS_STOP:
                was_update = stu
                break
        if not was_update:
            continue

        # Predicted times (epoch)
        pred_arr = was_update.arrival.time if was_update.HasField("arrival") else 0
        pred_dep = was_update.departure.time if was_update.HasField("departure") else 0

        # Scheduled times from static GTFS
        trip_sched = schedules.get(trip.trip_id, {})
        was_sched = trip_sched.get(WAS_STOP, ("", ""))
        sch_arr_epoch = hhmm_to_epoch(trip.start_date, was_sched[0])
        sch_dep_epoch = hhmm_to_epoch(trip.start_date, was_sched[1])

        # Direction: if pred_dep exists → departing, else arriving
        if pred_dep and not pred_arr:
            direction = "dep"
            sch_epoch = sch_dep_epoch
            act_epoch = pred_dep
        else:
            direction = "arr"
            sch_epoch = sch_arr_epoch
            act_epoch = pred_arr

        delay = round((act_epoch - sch_epoch) / 60) if (sch_epoch and act_epoch) else 0

        # Train number from trip_id (e.g., "Train487Saturday" → "487")
        train_num = "".join(c for c in trip.trip_id if c.isdigit()) or trip.trip_id

        # Line name from route_id
        line_map = {PENN_ROUTE: "PENN", "11704": "BRUNSWICK", "11706": "CAMDEN"}
        line = line_map.get(route_id, "MARC")

        trip_origin = get_origin_for_trip(trip.trip_id)
        trip_dest = get_destination_for_trip(trip.trip_id)

        if direction == "dep":
            origin = "DC"
            destination = trip_dest
        else:
            origin = trip_origin
            destination = "DC"

        trains.append({
            "num": train_num,
            "route": f"MARC {line}",
            "direction": direction,
            "schTime": epoch_to_iso(sch_epoch),
            "actTime": epoch_to_iso(act_epoch),
            "delay": delay,
            "origin": origin,
            "destination": destination,
            "system": "marc",
            "platform": "",
            "status": "enroute"
        })

    # Log to SQLite
    try:
        log_trains(trains)
    except Exception as e:
        print(f"[CRW] DB log error: {e}")

    return jsonify({"trains": trains, "feed_time": feed.header.timestamp})


@app.route("/api/log", methods=["POST"])
def log_external():
    """Accept a JSON array of train objects (e.g. Amtrak from the overlay)."""
    payload = request.get_json(silent=True)
    if not payload or not isinstance(payload, list):
        return jsonify({"error": "Expected a JSON array of train objects"}), 400
    try:
        log_trains(payload)
        return jsonify({"logged": len(payload)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats")
def stats():
    """Today's train count, average delay, and most delayed train."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    conn = get_db()
    try:
        row = conn.execute(
            """SELECT COUNT(*) AS cnt,
                      COALESCE(ROUND(AVG(delay_minutes), 1), 0) AS avg_delay
               FROM trains WHERE date(first_seen_at) = ?""",
            (today,),
        ).fetchone()

        worst = conn.execute(
            """SELECT train_num, route_name, delay_minutes, direction
               FROM trains WHERE date(first_seen_at) = ?
               ORDER BY delay_minutes DESC LIMIT 1""",
            (today,),
        ).fetchone()

        most_delayed = None
        if worst and worst["delay_minutes"]:
            most_delayed = {
                "num": worst["train_num"],
                "route": worst["route_name"],
                "delay": worst["delay_minutes"],
                "direction": worst["direction"],
            }

        return jsonify({
            "date": today,
            "count": row["cnt"],
            "avg_delay": row["avg_delay"],
            "most_delayed": most_delayed,
        })
    finally:
        conn.close()


@app.route("/api/metro")
def metro_trains():
    """Proxy WMATA predictions so the API key stays server-side."""
    if not WMATA_KEY:
        return jsonify({"error": "WMATA_KEY not set", "Trains": []}), 503
    try:
        req = urllib.request.Request(
            WMATA_PREDICTIONS,
            headers={"api_key": WMATA_KEY},
        )
        data = urllib.request.urlopen(req, timeout=10).read()
        return jsonify(json.loads(data))
    except Exception as e:
        return jsonify({"error": str(e), "Trains": []}), 502


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "time": datetime.now(ET).isoformat()})


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@app.route("/api/ticker-config")
def ticker_config():
    """Ticker timing and data window — set TICKER_* in .env (optional)."""
    wm = max(5, min(1440, _env_int("TICKER_WINDOW_MINUTES", 60)))
    speed = max(10, min(500, _env_int("TICKER_SPEED_PX_S", 80)))
    start_d = max(0, min(120_000, _env_int("TICKER_START_DWELL_MS", 2200)))
    end_d = max(0, min(120_000, _env_int("TICKER_END_DWELL_MS", 900)))
    fact_d = max(0, min(300_000, _env_int("TICKER_FACT_DWELL_MS", 7000)))
    label_a = max(0, min(5000, _env_int("TICKER_LABEL_ANIM_MS", 360)))
    refresh = max(0, min(3_600_000, _env_int("TICKER_REFRESH_MS", 60_000)))
    return jsonify(
        {
            "windowMinutes": wm,
            "speedPxS": speed,
            "startDwellMs": start_d,
            "endDwellMs": end_d,
            "factDwellMs": fact_d,
            "labelAnimMs": label_a,
            "refreshMs": refresh,
        }
    )


# ─── Serve overlay static files ──────────────
OVERLAY_DIR = Path(__file__).resolve().parent.parent / "overlay"

@app.route("/")
def serve_overlay():
    return send_from_directory(OVERLAY_DIR, "index.html")


@app.route("/tracker.html")
def serve_tracker_alias():
    """Same page as /ticker.html (common URL mix-up)."""
    return send_from_directory(OVERLAY_DIR, "ticker.html")


@app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory(OVERLAY_DIR, filename)


def _listen_port() -> int:
    """HTTP port from env (`.env` → PORT). Defaults to 5555 if unset or invalid."""
    raw = os.environ.get("PORT", "").strip()
    if not raw:
        return 5555
    try:
        p = int(raw)
        return p if 1 <= p <= 65535 else 5555
    except ValueError:
        return 5555


if __name__ == "__main__":
    init_db()
    port = _listen_port()
    print(f"[CRW] Starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
