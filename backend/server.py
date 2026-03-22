"""
Capital Rail Watch — Backend proxy
Fetches MARC GTFS-RT protobuf feed, decodes to JSON, serves to overlay.
Logs all train sightings to SQLite for stats and history.
"""
import csv, io, sqlite3, time, zipfile, urllib.request, os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from google.transit import gtfs_realtime_pb2

app = Flask(__name__)
CORS(app)

# ─── SQLite setup ─────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
    # Find stop with lowest sequence (earliest departure)
    # Since we stored by stop_id, find the one with earliest time
    earliest_time = "99:99:99"
    earliest_stop = None
    for sid, (arr, dep) in sched.items():
        t = dep if dep else arr
        if t and t < earliest_time:
            earliest_time = t
            earliest_stop = sid
    return stops.get(earliest_stop, "BAL") if earliest_stop else "BAL"

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

        origin = get_origin_for_trip(trip.trip_id)

        trains.append({
            "num": train_num,
            "route": f"MARC {line}",
            "direction": direction,
            "schTime": epoch_to_iso(sch_epoch),
            "actTime": epoch_to_iso(act_epoch),
            "delay": delay,
            "origin": origin,
            "destination": "DC",
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


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "time": datetime.now(ET).isoformat()})


# ─── Serve overlay static files ──────────────
OVERLAY_DIR = Path(__file__).resolve().parent.parent / "overlay"

@app.route("/")
def serve_overlay():
    return send_from_directory(OVERLAY_DIR, "index.html")

@app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory(OVERLAY_DIR, filename)


if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 5555))
    print(f"[CRW] Starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
