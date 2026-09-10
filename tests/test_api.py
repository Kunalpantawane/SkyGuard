"""API tests: live loopback server, real HTTP, real token gate.

Auth enforcement, ingest round-trip, record/anomaly reads, and bad-payload
handling — all through urllib against an ephemeral port. The server shuts
down in a finally so no test can leak a thread.
"""

import json
import os
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

from skyguard.api.server import LiveServer, _load_dotenv
from skyguard.pipeline import SkyGuardPipeline
from skyguard.store import AuditStore
from skyguard.types import Station

T0 = datetime(2025, 6, 1, tzinfo=timezone.utc)
TOKEN = "test-token-123"


def live_server(tmp):
    stations = {"S1": Station("S1", "One", 18.5, 73.9, 200.0, "valley")}
    store = AuditStore(os.path.join(tmp, "audit.db"))
    pipe = SkyGuardPipeline(stations=stations, store=store)
    server = LiveServer(pipe, stations, store=store, token=TOKEN).start()
    return server, store


def call(url, token=TOKEN, payload=None):
    """GET (payload None) or POST JSON; returns (status, body)."""
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, method="POST" if payload is not None else "GET")
    if token is not None:
        request.add_header("Authorization", f"Bearer {token}")
    if payload is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode() or "{}")


def test_health_is_open_and_stations_need_token():
    tmp = tempfile.TemporaryDirectory()
    server, store = live_server(tmp.name)
    try:
        status, body = call(server.url + "/health", token=None)
        assert status == 200 and body["status"] == "ok"
        assert call(server.url + "/stations", token=None)[0] == 401
        assert call(server.url + "/stations", token="wrong")[0] == 401
        status, body = call(server.url + "/stations")
        assert status == 200 and body["stations"][0]["station_id"] == "S1"
    finally:
        server.stop()
        store.close()
        tmp.cleanup()


def test_ingest_roundtrip_and_worklist():
    tmp = tempfile.TemporaryDirectory()
    server, store = live_server(tmp.name)
    try:
        good = {"station_id": "S1", "timestamp": T0.isoformat(),
                "temp_c": 28.0, "pressure_hpa": 1005.0, "rh_pct": 60.0}
        status, body = call(server.url + "/ingest", payload=good)
        assert status == 200
        assert body["qc_status"] == "PASS"
        evil = dict(good, timestamp=(T0 + timedelta(hours=1)).isoformat(), temp_c=-9999.0)
        status, body = call(server.url + "/ingest", payload=evil)
        assert status == 200
        assert body["fusion"]["is_anomaly"] is True
        status, body = call(server.url + "/stations/S1/records?limit=10")
        assert status == 200 and len(body["records"]) == 2
        status, body = call(server.url + "/anomalies?station=S1")
        assert status == 200 and len(body["anomalies"]) == 1
        status, body = call(server.url + "/stations/S1/records?anomalies=true")
        assert status == 200 and len(body["records"]) == 1
    finally:
        server.stop()
        store.close()
        tmp.cleanup()


def test_bad_payloads_are_400s():
    tmp = tempfile.TemporaryDirectory()
    server, store = live_server(tmp.name)
    try:
        assert call(server.url + "/ingest", payload={"station_id": "S1"})[0] == 400
        assert call(server.url + "/ingest",
                    payload={"station_id": "S1", "timestamp": T0.isoformat(),
                             "temp_c": "hot"})[0] == 400
        assert call(server.url + "/nope")[0] == 404
    finally:
        server.stop()
        store.close()
        tmp.cleanup()


def test_dotenv_fills_gap_but_env_wins():
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        dotenv = Path(tmp) / '.env'
        dotenv.write_text('SKYGUARD_API_TOKEN=file-token\n# comment\nEMPTY=\n', encoding='utf-8')
        os.environ.pop('SKYGUARD_API_TOKEN', None)
        try:
            _load_dotenv(dotenv)
            assert os.environ['SKYGUARD_API_TOKEN'] == 'file-token'
            os.environ['SKYGUARD_API_TOKEN'] = 'env-token'
            _load_dotenv(dotenv)
            assert os.environ['SKYGUARD_API_TOKEN'] == 'env-token'
        finally:
            os.environ.pop('SKYGUARD_API_TOKEN', None)


def test_dotenv_missing_file_is_silent():
    from pathlib import Path
    _load_dotenv(Path(tempfile.gettempdir()) / 'sg-no-such-env')

