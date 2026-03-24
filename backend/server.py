"""
Capital Rail Watch — Backend proxy
Fetches MARC GTFS-RT protobuf feed, decodes to JSON, serves to overlay.
Logs all train sightings to SQLite for stats and history.
"""
from __future__ import annotations

import csv, io, json, sqlite3, threading, time, zipfile, urllib.request, os
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
    _migrate_trains_columns()
    print(f"[CRW] SQLite DB ready at {DB_PATH}")


def _migrate_trains_columns():
    """Add columns introduced after first deploy (idempotent)."""
    conn = get_db()
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(trains)").fetchall()}
        if "status_flags" not in cols:
            conn.execute("ALTER TABLE trains ADD COLUMN status_flags TEXT")
            conn.commit()
            print("[CRW] Added column trains.status_flags")
    finally:
        conn.close()


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

    flags = t.get("status_flags") or ""
    if row:
        conn.execute(
            """UPDATE trains
               SET delay_minutes = ?, estimated_time = ?, actual_time = ?,
                   track = ?, status_flags = ?, last_updated_at = datetime('now')
               WHERE id = ?""",
            (t.get("delay", 0), t.get("estTime", ""), t.get("actTime", ""),
             t.get("platform", ""), flags, row["id"]),
        )
    else:
        conn.execute(
            """INSERT INTO trains
               (train_num, route_name, system, direction, scheduled_time,
                estimated_time, actual_time, delay_minutes, track, status_flags,
                origin, destination)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (t["num"], t.get("route", ""), t.get("system", ""),
             t["direction"], t.get("schTime", ""),
             t.get("estTime", ""), t.get("actTime", ""),
             t.get("delay", 0), t.get("platform", ""), flags,
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


def _parse_iso_to_utc(s: str | None) -> datetime | None:
    """Parse stored ISO timestamps to UTC for comparisons."""
    if not s or not str(s).strip():
        return None
    text = str(s).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ET)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


METRO_LINE_ABBR = {
    "RD": "RED",
    "BL": "BLUE",
    "OR": "ORANGE",
    "SV": "SILVER",
    "GR": "GREEN",
    "YL": "YELLOW",
}

METRO_ROUTE_TO_CLS = {
    "RED LINE": "metro-rd",
    "BLUE LINE": "metro-bl",
    "ORANGE LINE": "metro-or",
    "SILVER LINE": "metro-sv",
    "GREEN LINE": "metro-gr",
    "YELLOW LINE": "metro-yl",
}


def _metro_route_cls(route_name: str) -> str:
    if not route_name:
        return ""
    return METRO_ROUTE_TO_CLS.get(str(route_name).upper().strip(), "")


def _wmata_payload_to_train_rows(wmata_json: dict) -> list[dict]:
    """Same shape as overlay/ticker client for SQLite logging."""
    rows: list[dict] = []
    now = datetime.now(timezone.utc)
    for t in wmata_json.get("Trains") or []:
        line = t.get("Line")
        mn = t.get("Min")
        if not line or mn in (None, "", "---"):
            continue
        mn_s = str(mn).upper()
        if mn_s in ("BRD", "ARR"):
            mins = 0
        else:
            try:
                mins = int(str(mn))
            except ValueError:
                mins = 0
        eta = (now + timedelta(minutes=mins)).replace(microsecond=0).isoformat()
        code = str(line).upper().strip()
        abbr = METRO_LINE_ABBR.get(code, code)
        status_flags = "BRD" if mn_s == "BRD" else "ARR" if mn_s == "ARR" else ""
        rows.append(
            {
                "num": str(t.get("Car", "")) + "C",
                "route": f"{abbr} LINE",
                "direction": "arr",
                "schTime": eta,
                "estTime": eta,
                "actTime": eta,
                "delay": 0,
                "platform": str(t.get("Group", "")),
                "origin": "NoMa-Gallaudet U",
                "destination": str(t.get("DestinationName") or t.get("Destination") or ""),
                "system": "metro",
                "status_flags": status_flags,
            }
        )
    return rows


# ─── Config ──────────────────────────────────
MARC_TU = "https://mdotmta-gtfs-rt.s3.amazonaws.com/MARC+RT/marc-tu.pb"
MARC_GTFS = "https://feeds.mta.maryland.gov/gtfs/marc"
WAS_STOP = "11958"
PENN_ROUTE = "11705"
ET = timezone(timedelta(hours=-4))
WMATA_KEY = os.environ.get("WMATA_KEY", "")
WMATA_PREDICTIONS = "https://api.wmata.com/StationPrediction.svc/json/GetPrediction/B35"
AMTRAK_TRAINS_URL = "https://api-v3.amtraker.com/v3/trains"

AMTRAK_NORTH_SUBSTR = (
    "acela",
    "ne regional",
    "northeast regional",
    "vermonter",
    "capitol limited",
    "lake shore limited",
    "cardinal",
    "carolinian",
    "palmetto",
    "silver star",
    "silver meteor",
    "marc penn",
    "marc - penn",
)

_ingest_meta: dict = {"at": None, "sources": {}}
_ingest_lock = threading.Lock()

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


def _delay_minutes_from_sch_act(sch: str, act: str) -> int:
    if not sch or not act:
        return 0
    a = _parse_iso_to_utc(sch)
    b = _parse_iso_to_utc(act)
    if a is None or b is None:
        return 0
    return int(round((b - a).total_seconds() / 60))


def _parse_amtrak_api_payload(api: dict) -> list[dict]:
    """Amtraker v3 /v3/trains JSON → overlay-shaped train dicts (all WAS-related rows)."""
    if not api or not isinstance(api, dict):
        return []
    out: list[dict] = []
    seen: set[str] = set()
    for _k, ar in api.items():
        if not isinstance(ar, list):
            continue
        for t in ar:
            tn = t.get("trainNum")
            if not tn:
                continue
            ss = t.get("stations") or []
            w = None
            for s in ss:
                if str(s.get("code") or "").upper() == "WAS":
                    w = s
                    break
            if not w:
                continue
            oc = str(t.get("origCode") or "").upper()
            dc = str(t.get("destCode") or "").upper()
            st = str(w.get("status") or "").lower()
            if dc == "WAS":
                direction = "arr"
            elif oc == "WAS":
                direction = "dep"
            elif st in ("enroute", ""):
                direction = "arr"
            else:
                direction = "dep"
            dk = f"{tn}-{direction}"
            if dk in seen:
                continue
            seen.add(dk)
            sch = (w.get("schArr") if direction == "arr" else w.get("schDep")) or ""
            act = (w.get("arr") if direction == "arr" else w.get("dep")) or ""
            delay = _delay_minutes_from_sch_act(sch, act)
            out.append(
                {
                    "num": str(tn),
                    "route": t.get("routeName") or "",
                    "direction": direction,
                    "schTime": sch,
                    "actTime": act,
                    "delay": delay,
                    "platform": w.get("platform") or "",
                    "origin": t.get("origName") or "",
                    "destination": t.get("destName") or "",
                    "system": "amtrak",
                    "status": st,
                }
            )
    return out


def _filter_amtrak_north(trains: list[dict]) -> list[dict]:
    def ok(t: dict) -> bool:
        r = (t.get("route") or "").lower()
        return any(n in r for n in AMTRAK_NORTH_SUBSTR)

    return [t for t in trains if ok(t)]


def load_amtrak_trains_from_api() -> tuple[list[dict], str | None]:
    try:
        req = urllib.request.Request(
            AMTRAK_TRAINS_URL,
            headers={"User-Agent": "CapitalRailWatch/1.0"},
        )
        raw = urllib.request.urlopen(req, timeout=20).read()
        payload = json.loads(raw)
    except Exception as e:
        return [], str(e)
    return _filter_amtrak_north(_parse_amtrak_api_payload(payload)), None


def load_metro_trains_from_api() -> tuple[list[dict], str | None]:
    if not WMATA_KEY:
        return [], "WMATA_KEY not set"
    try:
        req = urllib.request.Request(
            WMATA_PREDICTIONS,
            headers={"api_key": WMATA_KEY},
        )
        data = urllib.request.urlopen(req, timeout=10).read()
        payload = json.loads(data)
    except Exception as e:
        return [], str(e)
    return _wmata_payload_to_train_rows(payload), None


def load_vre_trains_from_feed() -> tuple[list[dict], str | None, int | None]:
    load_vre_gtfs()
    if not vre_union_stop_ids:
        return [], "VRE GTFS not loaded", None
    try:
        data = urllib.request.urlopen(VRE_TU, timeout=15).read()
    except Exception as e:
        return [], str(e), None

    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(data)

    today_yyyymmdd = datetime.now(ET).strftime("%Y%m%d")
    trains_out: list[dict] = []
    seen_trip_dir: set[tuple[str, str]] = set()

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

    fts = int(feed.header.timestamp) if feed.header.timestamp else None
    return trains_out, None, fts


def load_marc_trains_from_feed() -> tuple[list[dict], str | None, int | None]:
    load_gtfs()
    try:
        data = urllib.request.urlopen(MARC_TU, timeout=10).read()
    except Exception as e:
        return [], str(e), None

    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(data)

    trains: list[dict] = []
    for ent in feed.entity:
        if not ent.HasField("trip_update"):
            continue
        tu = ent.trip_update
        trip = tu.trip
        route_id = str(trip.route_id)

        was_update = None
        for stu in tu.stop_time_update:
            if str(stu.stop_id) == WAS_STOP:
                was_update = stu
                break
        if not was_update:
            continue

        pred_arr = was_update.arrival.time if was_update.HasField("arrival") else 0
        pred_dep = was_update.departure.time if was_update.HasField("departure") else 0

        trip_sched = schedules.get(trip.trip_id, {})
        was_sched = trip_sched.get(WAS_STOP, ("", ""))
        sch_arr_epoch = hhmm_to_epoch(trip.start_date, was_sched[0])
        sch_dep_epoch = hhmm_to_epoch(trip.start_date, was_sched[1])

        if pred_dep and not pred_arr:
            direction = "dep"
            sch_epoch = sch_dep_epoch
            act_epoch = pred_dep
        else:
            direction = "arr"
            sch_epoch = sch_arr_epoch
            act_epoch = pred_arr

        delay = round((act_epoch - sch_epoch) / 60) if (sch_epoch and act_epoch) else 0

        train_num = "".join(c for c in trip.trip_id if c.isdigit()) or trip.trip_id

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

        trains.append(
            {
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
                "status": "enroute",
            }
        )

    fts = int(feed.header.timestamp) if feed.header.timestamp else None
    return trains, None, fts


def _ingest_log_trains(label: str, trains: list[dict]) -> None:
    if not trains:
        return
    try:
        log_trains(trains)
    except Exception as e:
        print(f"[CRW] ingest {label} DB log error: {e}")
        raise


def ingest_all_once() -> None:
    """Fetch all providers and upsert SQLite (background + startup)."""
    global _ingest_meta
    ts = datetime.now(timezone.utc).isoformat()
    sources: dict[str, dict] = {}

    amtrak, err_a = load_amtrak_trains_from_api()
    sources["amtrak"] = {"ok": err_a is None, "error": err_a, "count": len(amtrak)}
    if amtrak:
        try:
            _ingest_log_trains("amtrak", amtrak)
        except Exception as e:
            sources["amtrak"]["ok"] = False
            sources["amtrak"]["error"] = str(e)

    marc, err_m, _ftm = load_marc_trains_from_feed()
    sources["marc"] = {"ok": err_m is None, "error": err_m, "count": len(marc)}
    if marc:
        try:
            _ingest_log_trains("marc", marc)
        except Exception as e:
            sources["marc"]["ok"] = False
            sources["marc"]["error"] = str(e)

    vre, err_v, _ftv = load_vre_trains_from_feed()
    sources["vre"] = {"ok": err_v is None, "error": err_v, "count": len(vre)}
    if vre:
        try:
            _ingest_log_trains("vre", vre)
        except Exception as e:
            sources["vre"]["ok"] = False
            sources["vre"]["error"] = str(e)

    metro, err_w = load_metro_trains_from_api()
    sources["metro"] = {"ok": err_w is None, "error": err_w, "count": len(metro)}
    if metro:
        try:
            _ingest_log_trains("metro", metro)
        except Exception as e:
            sources["metro"]["ok"] = False
            sources["metro"]["error"] = str(e)

    with _ingest_lock:
        _ingest_meta = {"at": ts, "sources": sources}


def _ingest_loop() -> None:
    interval = max(5, _env_int("INGEST_INTERVAL_SEC", 30))
    while True:
        try:
            ingest_all_once()
        except Exception as e:
            print(f"[CRW] ingest error: {e}")
        time.sleep(interval)


def start_ingest_thread() -> None:
    interval = max(5, _env_int("INGEST_INTERVAL_SEC", 30))
    t = threading.Thread(target=_ingest_loop, name="crw-ingest", daemon=True)
    t.start()
    print(f"[CRW] Background ingest every {interval}s")


def _train_dict_from_db_row(row: sqlite3.Row) -> dict:
    sch = row["scheduled_time"] or ""
    est = row["estimated_time"] or ""
    act = row["actual_time"] or ""
    act_out = act or est or sch
    route_name = row["route_name"] or ""
    sys = row["system"] or "marc"
    direction = row["direction"] or "arr"
    num = row["train_num"] or ""
    d: dict = {
        "num": num,
        "route": route_name,
        "direction": direction,
        "schTime": sch,
        "actTime": act_out,
        "delay": int(row["delay_minutes"] or 0),
        "platform": row["track"] or "",
        "origin": row["origin"] or "",
        "destination": row["destination"] or "",
        "system": sys,
        "metroRouteCls": _metro_route_cls(route_name) if sys == "metro" else "",
    }
    try:
        sf = row["status_flags"] or ""
    except (KeyError, IndexError):
        sf = ""
    if sf == "BRD":
        d["_status"] = {"t": "BOARDNG", "c": "green"}
    elif sf == "ARR":
        d["_status"] = {"t": "ARRIVE", "c": "green"}
    return d


def _trains_from_db_event_window(
    win_start: datetime,
    win_end: datetime,
    *,
    limit: int = 2000,
    max_age_hours: int = 8,
) -> list[dict]:
    conn = get_db()
    try:
        db_rows = conn.execute(
            f"""SELECT train_num, route_name, system, direction, scheduled_time,
                      estimated_time, actual_time, delay_minutes, track, status_flags,
                      origin, destination
               FROM trains
               WHERE datetime(last_updated_at) >= datetime('now', '-{max_age_hours} hours')
               ORDER BY last_updated_at DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
    finally:
        conn.close()

    out: list[dict] = []
    seen: set[tuple[str, str, int]] = set()
    for row in db_rows:
        d = _train_dict_from_db_row(row)
        when = _parse_iso_to_utc(d["actTime"]) or _parse_iso_to_utc(d["schTime"])
        if when is None:
            continue
        if not (win_start <= when <= win_end):
            continue
        num = d["num"]
        direction = d["direction"]
        dedupe_key = (num, direction, int(when.timestamp() // 60))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        out.append(d)

    def _sort_key(tr: dict) -> float:
        w = _parse_iso_to_utc(tr.get("actTime") or tr.get("schTime") or "")
        return w.timestamp() if w else 0.0

    out.sort(key=_sort_key)
    return out


@app.route("/api/vre")
def vre_trains():
    """VRE trains at Union Station from GTFS-RT + static schedule (same shape as /api/marc)."""
    trains_out, err, fts = load_vre_trains_from_feed()
    if err:
        code = 503 if err == "VRE GTFS not loaded" else 502
        return jsonify({"error": err, "trains": []}), code
    try:
        log_trains(trains_out)
    except Exception as e:
        print(f"[CRW] VRE DB log error: {e}")
    return jsonify({"trains": trains_out, "feed_time": fts})


@app.route("/api/marc")
def marc_trains():
    trains, err, fts = load_marc_trains_from_feed()
    if err:
        return jsonify({"error": err, "trains": []}), 502
    try:
        log_trains(trains)
    except Exception as e:
        print(f"[CRW] DB log error: {e}")
    return jsonify({"trains": trains, "feed_time": fts})


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
        payload = json.loads(data)
        try:
            log_trains(_wmata_payload_to_train_rows(payload))
        except Exception as e:
            print(f"[CRW] Metro DB log error: {e}")
        return jsonify(payload)
    except Exception as e:
        return jsonify({"error": str(e), "Trains": []}), 502


@app.route("/api/overlay-trains")
def overlay_trains():
    """Board overlay: trains from SQLite in a time window, split by system (ingest must be running)."""
    raw_h = request.args.get("horizonMinutes", "720")
    try:
        hm = int(raw_h)
    except ValueError:
        hm = 720
    hm = max(30, min(24 * 60, hm))
    raw_p = request.args.get("pastSkewMinutes", "2")
    try:
        pm = int(raw_p)
    except ValueError:
        pm = 2
    pm = max(0, min(120, pm))
    now_utc = datetime.now(timezone.utc)
    win_start = now_utc - timedelta(minutes=pm)
    win_end = now_utc + timedelta(minutes=hm)
    bucket = _trains_from_db_event_window(
        win_start, win_end, limit=2500, max_age_hours=12
    )
    amtrak = [t for t in bucket if t.get("system") == "amtrak"]
    marc = [
        t
        for t in bucket
        if t.get("system") == "marc" and "PENN" in (t.get("route") or "").upper()
    ]
    vre = [t for t in bucket if t.get("system") == "vre"]
    metro = [t for t in bucket if t.get("system") == "metro"]
    meta = dict(_ingest_meta) if _ingest_meta else {}
    return jsonify(
        {
            "amtrak": amtrak,
            "marc": marc,
            "vre": vre,
            "metro": metro,
            "ingestedAt": meta.get("at"),
            "sources": meta.get("sources"),
            "source": "sqlite",
        }
    )


@app.route("/api/ticker-cache")
def ticker_cache():
    """Trains from SQLite within the ticker time window (same store as /api/overlay-trains)."""
    raw_wm = request.args.get("windowMinutes", "60")
    try:
        wm = int(raw_wm)
    except ValueError:
        wm = 60
    wm = max(5, min(1440, wm))
    raw_ps = request.args.get("pastSkewMinutes", "5")
    try:
        pm = int(raw_ps)
    except ValueError:
        pm = 5
    pm = max(0, min(120, pm))
    now_utc = datetime.now(timezone.utc)
    win_start = now_utc - timedelta(minutes=pm)
    win_end = now_utc + timedelta(minutes=wm)
    out = _trains_from_db_event_window(
        win_start, win_end, limit=2000, max_age_hours=12
    )
    return jsonify(
        {
            "trains": out,
            "source": "sqlite",
            "count": len(out),
            "ingestedAt": _ingest_meta.get("at"),
        }
    )


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
    try:
        ingest_all_once()
    except Exception as e:
        print(f"[CRW] Initial ingest error: {e}")
    start_ingest_thread()
    port = _listen_port()
    print(f"[CRW] Starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
