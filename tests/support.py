"""Shared test fixtures. Not collected: the runner only globs `test_*.py`.

Tests that need a whole station network used to build one with the synthetic
weather simulator. That is gone, so they load a slice of the committed real
archive instead. Parsing a 1.5 MB CSV once per test would dominate the suite
runtime, so the parse is cached per slice length.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
OBS_CSV = ROOT / "examples" / "data" / "real_aws_hourly.csv"
STATIONS_CSV = ROOT / "examples" / "data" / "real_aws_stations.csv"

_CACHE: Dict[tuple, object] = {}


def archive_available() -> bool:
    return OBS_CSV.exists() and STATIONS_CSV.exists()


def real_network(days: int = 40, n_stations: Optional[int] = None):
    """A slice of the committed archive as a `StationNetwork`.

    `days` trims the time axis; `n_stations` trims the network. Both default to
    something small, because these are unit tests and the archive holds far more
    than any of them needs.
    """
    from skyguard.data.real import load_real_network

    key = (days, n_stations)
    if key not in _CACHE:
        ids = None
        if n_stations is not None:
            from skyguard.data.loaders import load_stations_csv

            ids = [s.station_id for s in load_stations_csv(str(STATIONS_CSV))][:n_stations]
        net = load_real_network(str(OBS_CSV), str(STATIONS_CSV), station_ids=ids)
        _CACHE[key] = net.slice_steps(days * 24)
    return _CACHE[key].copy()
