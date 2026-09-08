"""Local dashboard API: loopback HTTP, bearer token, stdlib only.

Why stdlib: the API layer adds no dependencies — `http.server`, `json` and
`hmac` are enough for a local operator service. It binds loopback by default
and requires a bearer token (from `SKYGUARD_API_TOKEN`): this service exposes
station data and accepts ingest, so an unauthenticated LAN bind would put the
network's QC state on the local network in the clear.

Routes (JSON everywhere except `/`, which serves the dashboard):
  GET  /health                    liveness, no auth
  GET  /                          the single-file dashboard, no auth
  GET  /stations                  station registry
  GET  /stations/{id}/records?limit=N&anomalies=true
  GET  /anomalies?station=ID&limit=N   newest-first worklist
  POST /ingest                    {station_id, timestamp, temp_c,
                                   pressure_hpa, rh_pct, neighbours?}
"""

from __future__ import annotations

import hmac
import json
import math
import os
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from ..config import DEFAULT_CONFIG, ApiConfig
from ..data.loaders import to_utc
from ..pipeline import SkyGuardPipeline
from ..store import AuditStore
from ..types import Observation, Station

_VERSION = "1.0.0"


def _read_token(config: ApiConfig, override: Optional[str]) -> str:
    """Bearer token: explicit override wins, otherwise the env var."""
    if override is not None:
        return override
    return os.environ.get(config.token_env_var, "")


def _parse_observation(payload: Dict, default_station: str = "") -> Observation:
    """Strict ingest parsing: unknown shapes are 400s, never silent NaNs."""
    station_id = str(payload.get("station_id") or default_station)
    if not station_id:
        raise ValueError("ingest needs a station_id")
    raw_ts = payload.get("timestamp")
    if raw_ts is None:
        raise ValueError("ingest needs an ISO timestamp")
    # Canonicalise to UTC rather than preserving the sender's offset: the store
    # orders records by ISO text, so mixed offsets would not sort chronologically.
    stamp = to_utc(datetime.fromisoformat(str(raw_ts)))
    return Observation(
        station_id=station_id,
        timestamp=stamp,
        temp_c=_optional_float(payload.get("temp_c"), "temp_c"),
        pressure_hpa=_optional_float(payload.get("pressure_hpa"), "pressure_hpa"),
        rh_pct=_optional_float(payload.get("rh_pct"), "rh_pct"),
    )


def _optional_float(value: object, name: str) -> Optional[float]:
    """Parse a nullable number, rejecting non-finite values.

    Python's JSON parser accepts the non-standard `NaN`, `Infinity` and
    `-Infinity` tokens. Letting them through would break the ingest contract two
    ways: NaN is silently reinterpreted as "missing" downstream, and an infinity
    propagates through Layer 1 arithmetic into the audit store. The CSV loader
    already rejects NaN; this keeps the two entry points consistent.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number or null")
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number or null") from None
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return parsed


def _bounded_limit(raw: object, default: int, maximum: int = 5000) -> int:
    """Parse a query `limit` into a bounded positive int.

    Unvalidated `int(raw)` raises inside the request worker (a 500 for what is
    plainly a client error), and a negative value reaches Python's slicing rules
    where it silently means something else entirely.
    """
    if raw is None or raw == "":
        return default
    try:
        parsed = int(str(raw))
    except (TypeError, ValueError):
        raise ValueError("limit must be an integer") from None
    if parsed < 1:
        raise ValueError("limit must be 1 or greater")
    return min(parsed, maximum)


def build_handler(
    pipeline: SkyGuardPipeline,
    stations: Dict[str, Station],
    store: Optional[AuditStore],
    config: ApiConfig,
    token: str,
    dashboard_path: Optional[Path],
):
    """Handler factory closing over the live pipeline, store and registry."""

    class SkyGuardHandler(BaseHTTPRequestHandler):
        server_version = "SkyGuardAPI/1.0"

        # -- plumbing ---------------------------------------------------

        def log_message(self, fmt: str, *args: object) -> None:  # keep test output clean
            pass

        def _send(self, status: int, payload: object) -> None:
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorised(self) -> bool:
            if not config.require_token:
                return True
            presented = self.headers.get("Authorization", "")
            expected = f"Bearer {token}"
            return bool(token) and hmac.compare_digest(presented, expected)

        def _guard(self) -> bool:
            if self._authorised():
                return True
            self._send(401, {"error": "missing or invalid bearer token"})
            return False

        # -- routing ----------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802 (http.server convention)
            parsed = urlparse(self.path)
            if parsed.path in ("/health", "/api/health"):
                self._send(200, {"status": "ok", "version": _VERSION})
                return
            if parsed.path == "/":
                self._serve_dashboard()
                return
            if not self._guard():
                return
            if parsed.path == "/stations":
                self._send(200, {"stations": [s.to_dict() for s in stations.values()]})
            elif parsed.path == "/anomalies":
                self._serve_anomalies(parse_qs(parsed.query))
            elif parsed.path.startswith("/stations/") and parsed.path.endswith("/records"):
                station_id = parsed.path[len("/stations/"): -len("/records")]
                self._serve_records(station_id, parse_qs(parsed.query))
            else:
                self._send(404, {"error": "unknown route"})

        def do_POST(self) -> None:  # noqa: N802 (http.server convention)
            parsed = urlparse(self.path)
            if parsed.path != "/ingest":
                self._send(404, {"error": "unknown route"})
                return
            if not self._guard():
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                obs = _parse_observation(payload)
                neighbours = [_parse_observation(item, default_station="") 
                              for item in payload.get("neighbours", [])]
            except (ValueError, json.JSONDecodeError) as exc:
                self._send(400, {"error": f"bad ingest payload: {exc}"})
                return
            record = pipeline.process(obs, neighbours=neighbours)
            self._send(200, record.to_dict())

        # -- views ------------------------------------------------------

        def _serve_dashboard(self) -> None:
            if dashboard_path is None or not dashboard_path.exists():
                self._send(404, {"error": "dashboard not found"})
                return
            body = dashboard_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_records(self, station_id: str, query: Dict[str, List[str]]) -> None:
            if store is None:
                self._send(503, {"error": "no audit store attached"})
                return
            try:
                limit = _bounded_limit(query.get("limit", [None])[0], 200)
            except ValueError as exc:
                self._send(400, {"error": str(exc)})
                return
            only_anomalies = query.get("anomalies", ["false"])[0].lower() == "true"
            docs = (store.fetch_anomalies(station_id) if only_anomalies
                    else store.fetch_station(station_id))
            self._send(200, {"station_id": station_id, "records": docs[:limit]})

        def _serve_anomalies(self, query: Dict[str, List[str]]) -> None:
            if store is None:
                self._send(503, {"error": "no audit store attached"})
                return
            try:
                limit = _bounded_limit(query.get("limit", [None])[0], 200)
            except ValueError as exc:
                self._send(400, {"error": str(exc)})
                return
            station = query.get("station", [None])[0]
            self._send(200, {"anomalies": store.fetch_anomalies(station)[:limit]})

    return SkyGuardHandler


class LiveServer:
    """A running API instance: start/stop around tests or the real service."""

    def __init__(
        self, pipeline: SkyGuardPipeline, stations: Dict[str, Station],
        store: Optional[AuditStore] = None, config: ApiConfig | None = None,
        token: Optional[str] = None, host: Optional[str] = None, port: int = 0,
    ) -> None:
        self.config = config or DEFAULT_CONFIG.api
        self.token = _read_token(self.config, token)
        self.host = host or self.config.host
        dashboard = (Path(__file__).resolve().parents[2] / "dashboard" / "index.html")
        handler = build_handler(
            pipeline, stations, store, self.config, self.token,
            dashboard if dashboard.exists() else None,
        )
        self.server = ThreadingHTTPServer((self.host, port), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        """Base URL including the (possibly ephemeral) port."""
        host, port = self.server.server_address
        return f"http://{host}:{port}"

    def start(self) -> "LiveServer":
        """Serve in the background until `stop`."""
        self.thread.start()
        return self

    def stop(self) -> None:
        """Shut down cleanly (tests must call this — the thread is daemon)."""
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5.0)
