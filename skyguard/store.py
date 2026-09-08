"""SQLite audit-preserving store: raw data is immutable, corrections alongside.

Why append-only by API design: operational meteorology flags rather than
deletes, and every record must carry the flags and scores that produced its
verdict. This store exposes insert and reads only — there is deliberately no
update or delete method, so a correction can never silently become the truth.
The full record document is kept as JSON (lossless audit), with typed columns
for the queries operators actually run.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .data.loaders import to_utc
from .types import QCRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    station_id   TEXT NOT NULL,
    timestamp    TEXT NOT NULL,
    temp_c       REAL,
    pressure_hpa REAL,
    rh_pct       REAL,
    anomaly_score REAL NOT NULL,
    is_anomaly   INTEGER NOT NULL,
    fault_class  TEXT NOT NULL,
    doc          TEXT NOT NULL,
    PRIMARY KEY (station_id, timestamp)
);
CREATE INDEX IF NOT EXISTS idx_records_station_time
    ON records (station_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_records_anomalies
    ON records (is_anomaly, timestamp);
"""


class AuditStore:
    """Append-only audit store over a SQLite file (stdlib only)."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        # Shared across the API server's worker threads: sqlite handles
        # concurrent use when threads never share an operation, enforced here
        # with one lock around every statement batch.
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        """Release the database handle."""
        self._conn.close()

    def save_record(self, record: QCRecord) -> int:
        """Insert one audit record. Re-inserting the same (station, time)
        raises sqlite3.IntegrityError: first write wins, raw is never
        overwritten — that is the immutability guarantee, enforced by the
        primary key rather than by convention."""
        doc = json.dumps(record.to_dict(), default=str)
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO records "
                "(station_id, timestamp, temp_c, pressure_hpa, rh_pct, "
                " anomaly_score, is_anomaly, fault_class, doc) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.station_id,
                    record.timestamp.isoformat(),
                    record.observation.temp_c,
                    record.observation.pressure_hpa,
                    record.observation.rh_pct,
                    record.anomaly_score,
                    int(record.fusion.is_anomaly),
                    record.diagnosis.fault_class.value,
                    doc,
                ),
            )
            self._conn.commit()
            return int(cursor.lastrowid)

    def fetch_station(
        self, station_id: str, start: Optional[datetime] = None, end: Optional[datetime] = None
    ) -> List[Dict]:
        """Full audit documents for a station in time order (ISO bounds).

        Ordering and bounds are lexicographic on ISO text, which is only
        equivalent to chronological order when every stored timestamp shares one
        offset. The pipeline canonicalises to UTC on ingest; bounds are
        canonicalised here so a caller passing a local-offset datetime still
        compares against the same scale.
        """
        query = "SELECT doc FROM records WHERE station_id = ?"
        params: List[str] = [station_id]
        if start is not None:
            query += " AND timestamp >= ?"
            params.append(to_utc(start).isoformat())
        if end is not None:
            query += " AND timestamp <= ?"
            params.append(to_utc(end).isoformat())
        query += " ORDER BY timestamp"
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [json.loads(row[0]) for row in rows]

    def fetch_anomalies(self, station_id: Optional[str] = None) -> List[Dict]:
        """Anomaly documents only, newest first — the operator's worklist."""
        if station_id is None:
            query = "SELECT doc FROM records WHERE is_anomaly = 1 ORDER BY timestamp DESC"
            params: List[str] = []
        else:
            query = ("SELECT doc FROM records WHERE is_anomaly = 1 AND station_id = ? "
                     "ORDER BY timestamp DESC")
            params = [station_id]
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [json.loads(row[0]) for row in rows]

    def count(self, station_id: Optional[str] = None) -> int:
        """Stored record count, optionally per station."""
        with self._lock:
            if station_id is None:
                row = self._conn.execute("SELECT COUNT(*) FROM records").fetchone()
            else:
                row = self._conn.execute(
                    "SELECT COUNT(*) FROM records WHERE station_id = ?", (station_id,)
                ).fetchone()
            return int(row[0])
