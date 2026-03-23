CREATE TABLE IF NOT EXISTS trains (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    train_num TEXT NOT NULL,
    route_name TEXT,
    system TEXT CHECK(system IN ('amtrak','marc','vre','metro')),
    direction TEXT CHECK(direction IN ('arr','dep')),
    scheduled_time TEXT,
    estimated_time TEXT,
    actual_time TEXT,
    delay_minutes INTEGER DEFAULT 0,
    track TEXT,
    status_flags TEXT,
    origin TEXT,
    destination TEXT,
    first_seen_at TEXT DEFAULT (datetime('now')),
    last_updated_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_trains_num ON trains(train_num);
CREATE INDEX IF NOT EXISTS idx_trains_date ON trains(first_seen_at);
