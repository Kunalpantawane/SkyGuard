"""Turn a real archive CSV into the array network the injector and harness want.

`loaders.py` gets real observations into the canonical `Observation` shape.
This module takes the next step and lays them on the regular station-by-time
grid that `injector.inject_faults` and `eval/harness.compare` address by index,
so an archive CSV becomes something the evaluation harness can plant labelled
faults into and score.

Two jobs live here:

`network_from_observations` grids the observations. Missing rows stay NaN
rather than being interpolated, because an interpolated gap is a fabricated
reading and Layer 1 exists to see gaps.

`detect_genuine_extremes` recovers the negative controls. A real archive does
not arrive labelled, and without those labels the flagship genuine-extreme
false-alarm number cannot be computed at all. So we find them: strip the daily
cycle, subtract a slow seasonal baseline, and take spans where the whole network
moves together past a threshold for long enough. Network-wide coherence is the
point. One station warming alone is what a fault looks like; 8 stations warming
together over 3 days is weather.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..types import Observation, Station
from .network import StationNetwork


def _grid_timestamps(stamps: Sequence[datetime], interval_minutes: int) -> List[datetime]:
    """Regular UTC grid spanning the observed range at the stated interval."""
    step = timedelta(minutes=interval_minutes)
    first, last = min(stamps), max(stamps)
    out: List[datetime] = []
    cursor = first
    while cursor <= last:
        out.append(cursor)
        cursor += step
    return out


def infer_interval_minutes(stamps: Sequence[datetime]) -> int:
    """Modal spacing between consecutive timestamps, in whole minutes.

    Modal and not mean: one long outage in an otherwise hourly file would drag
    a mean interval far off and silently rescale every rate-of-change limit.
    """
    ordered = sorted(set(stamps))
    if len(ordered) < 2:
        return 60
    deltas = [int(round((b - a).total_seconds() / 60.0))
              for a, b in zip(ordered[:-1], ordered[1:])]
    deltas = [d for d in deltas if d > 0]
    if not deltas:
        return 60
    counts: Dict[int, int] = {}
    for d in deltas:
        counts[d] = counts.get(d, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def network_from_observations(
    stations: Sequence[Station],
    observations: Sequence[Observation],
    interval_minutes: Optional[int] = None,
) -> StationNetwork:
    """Grid real observations onto the array container the harness indexes.

    Observations for stations not in `stations` are dropped. Grid cells with
    no observation stay NaN.
    """
    if not stations:
        raise ValueError("need at least one station")
    if not observations:
        raise ValueError("need at least one observation")

    index = {station.station_id: i for i, station in enumerate(stations)}
    known = [obs for obs in observations if obs.station_id in index]
    if not known:
        raise ValueError("no observation matched any supplied station")

    interval = interval_minutes or infer_interval_minutes([o.timestamp for o in known])
    grid = _grid_timestamps([o.timestamp for o in known], interval)
    slot = {stamp: t for t, stamp in enumerate(grid)}

    shape = (len(stations), len(grid))
    temp = np.full(shape, np.nan, dtype=np.float64)
    pressure = np.full(shape, np.nan, dtype=np.float64)
    rh = np.full(shape, np.nan, dtype=np.float64)

    for obs in known:
        t = slot.get(obs.timestamp)
        if t is None:      # off-grid stamp, e.g. a 10-minute record in an hourly file
            continue
        i = index[obs.station_id]
        if obs.temp_c is not None:
            temp[i, t] = obs.temp_c
        if obs.pressure_hpa is not None:
            pressure[i, t] = obs.pressure_hpa
        if obs.rh_pct is not None:
            rh[i, t] = obs.rh_pct

    return StationNetwork(
        stations=list(stations),
        timestamps=grid,
        temp=temp,
        pressure=pressure,
        rh=rh,
        genuine_events=[],
        interval_minutes=interval,
    )


def _centred_mean(series: np.ndarray, width: int, require_full: bool = False) -> np.ndarray:
    """NaN-tolerant centred moving average.

    `require_full` decides what happens at the ends. Removing the daily cycle
    needs a whole day in the window: half a day averages half a sine and leaves
    the diurnal swing sitting in the residual, which then reads as a synoptic
    anomaly at the start and end of every series. So that pass demands a full
    window and reports NaN where it cannot have one. The slow seasonal baseline
    is happy with a shortened window, because what it estimates barely moves
    across one.
    """
    n = series.size
    out = np.full(n, np.nan, dtype=np.float64)
    half = max(1, width // 2)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        if require_full and (hi - lo) < min(width, n):
            continue
        chunk = series[lo:hi]
        valid = chunk[~np.isnan(chunk)]
        if valid.size:
            out[i] = float(np.mean(valid))
    return out


def temperature_anomaly(net: StationNetwork, seasonal_days: int = 15) -> np.ndarray:
    """Network mean temperature anomaly: daily cycle out, slow season out.

    What is left is the synoptic signal. A heatwave shows here as a broad
    positive bump; the ordinary afternoon peak does not show at all, which is
    the whole reason for removing the daily cycle first.
    """
    steps_per_day = max(1, int(round(24 * 60 / net.interval_minutes)))
    per_station = []
    for i in range(net.n_stations):
        daily = _centred_mean(net.temp[i], steps_per_day, require_full=True)
        seasonal = _centred_mean(daily, steps_per_day * seasonal_days)
        per_station.append(daily - seasonal)
    stacked = np.vstack(per_station)
    return _column_mean(stacked)


def _column_mean(stacked: np.ndarray) -> np.ndarray:
    """Mean down each column, NaN where a column has nothing in it.

    `np.nanmean` does exactly this but warns on an all-NaN column, and the ends
    of the daily-smoothed series are legitimately all NaN, so the warning would
    fire on every well-formed run.
    """
    valid = ~np.isnan(stacked)
    counts = valid.sum(axis=0)
    totals = np.where(valid, stacked, 0.0).sum(axis=0)
    out = np.full(stacked.shape[1], np.nan, dtype=np.float64)
    np.divide(totals, counts, out=out, where=counts > 0)
    return out


def _spans_past(series: np.ndarray, threshold: float, min_steps: int,
                below: bool = False) -> List[Tuple[int, int]]:
    """Contiguous runs past `threshold`, as half-open [start, end) spans."""
    with np.errstate(invalid="ignore"):
        flags = (series <= threshold) if below else (series >= threshold)
    flags = np.where(np.isnan(series), False, flags)
    spans: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= min_steps:
                spans.append((start, i))
            start = None
    if start is not None and len(flags) - start >= min_steps:
        spans.append((start, len(flags)))
    return spans


def detect_genuine_extremes(
    net: StationNetwork,
    temp_anomaly_c: float = 1.8,
    pressure_drop_hpa: float = 4.0,
    min_hours: int = 24,
) -> List[Dict[str, object]]:
    """Label the network-coherent extreme-weather spans in a real series.

    Returned dicts carry `kind`, a half-open `start` and `end`, and a
    magnitude, which is what `inject_faults` reads to record them as
    GENUINE_EXTREME ground truth. Those labels are what makes
    `genuine_extreme_far` computable on an unlabelled archive.

    Thresholds are held low deliberately. A span wrongly called genuine weather
    only removes points from the fault-placement pool; a real heatwave left
    unlabelled gets scored as normal air and hides false alarms, which is the
    failure that actually matters.
    """
    steps_per_hour = max(1, int(round(60 / net.interval_minutes)))
    min_steps = min_hours * steps_per_hour
    events: List[Dict[str, object]] = []

    anomaly = temperature_anomaly(net)
    for start, end in _spans_past(anomaly, temp_anomaly_c, min_steps):
        peak = float(np.nanmax(anomaly[start:end]))
        events.append({"kind": "heatwave", "start": int(start), "end": int(end),
                       "peak_c": round(peak, 2)})

    # Pressure gets the same two-stage smoothing as temperature. Raw hourly
    # pressure carries the atmospheric tide, a semidiurnal wave of a couple of
    # hPa; subtracting only a slow baseline leaves that wave in and it chops
    # every depression into runs too short to clear `min_hours`.
    steps_per_day = 24 * steps_per_hour
    mean_pressure = _column_mean(net.pressure)
    daily_pressure = _centred_mean(mean_pressure, steps_per_day, require_full=True)
    baseline = _centred_mean(daily_pressure, steps_per_day * 10)
    depression = daily_pressure - baseline
    for start, end in _spans_past(depression, -abs(pressure_drop_hpa), min_steps, below=True):
        depth = float(np.nanmin(depression[start:end]))
        events.append({"kind": "pressure_depression", "start": int(start), "end": int(end),
                       "depth_hpa": round(abs(depth), 2)})

    events.sort(key=lambda event: int(event["start"]))
    return events


def load_real_network(
    observations_csv: str,
    stations_csv: str,
    station_ids: Optional[Sequence[str]] = None,
    label_extremes: bool = True,
) -> StationNetwork:
    """One call from two CSV paths to a gridded, extreme-labelled network."""
    from .loaders import load_csv, load_stations_csv

    stations = load_stations_csv(stations_csv)
    if station_ids is not None:
        wanted = set(station_ids)
        stations = [s for s in stations if s.station_id in wanted]
        if not stations:
            raise ValueError(f"none of {sorted(wanted)} are in {stations_csv}")

    observations = load_csv(observations_csv)
    net = network_from_observations(stations, observations)
    if label_extremes:
        net.genuine_events = detect_genuine_extremes(net)
    return net
