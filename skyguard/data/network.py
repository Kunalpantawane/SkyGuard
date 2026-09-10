"""The station-by-time array container every other data module works on.

Detection runs on `Observation` objects one at a time, which is how a real
network delivers data. Evaluation needs the same readings as arrays it can
address by index, so a fault can be planted at station 3, hour 412 and scored
against a label matrix of the same shape. `StationNetwork` is that second view.

It carries no opinion about where the readings came from. `real.py` fills it
from an archive CSV; `injector.py` copies one and plants labelled faults in the
copy. Nothing downstream can tell the difference, which is the point: real data
and injected faults reach the pipeline through one code path.

The Magnus dew-point pair lives here too. Both the injector and the multivariate
layer need it, and it is physics rather than data, so it belongs with the
container rather than in either caller.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

from ..types import Observation, Station


# --------------------------------------------------------------------------
# Humidity physics
# --------------------------------------------------------------------------

def saturation_vapour_pressure_hpa(temp_c: np.ndarray) -> np.ndarray:
    """Magnus-Tetens saturation vapour pressure."""
    return 6.112 * np.exp(17.67 * temp_c / (temp_c + 243.5))


def rh_from_dewpoint(temp_c: np.ndarray, dewpoint_c: np.ndarray) -> np.ndarray:
    """Relative humidity from air temperature and dew point.

    Dew point reflects absolute moisture and moves slowly; air temperature
    swings through the day. RH derived from the pair climbs overnight as
    temperature falls toward the dew point, which is the genuine relationship
    Layer 3 checks against.
    """
    e = saturation_vapour_pressure_hpa(dewpoint_c)
    es = saturation_vapour_pressure_hpa(temp_c)
    return np.clip(100.0 * e / es, 0.5, 100.0)


def dewpoint_from_rh(temp_c: float, rh_pct: float) -> float:
    """Inverse of the above; used by the multivariate layer's physical check."""
    rh = max(min(rh_pct, 100.0), 0.1)
    gamma = math.log(rh / 100.0) + (17.67 * temp_c) / (temp_c + 243.5)
    return 243.5 * gamma / (17.67 - gamma)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two stations, in kilometres."""
    radius = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (math.sin(d_phi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2)
    return 2 * radius * math.asin(math.sqrt(a))


# --------------------------------------------------------------------------
# The container
# --------------------------------------------------------------------------

class StationNetwork:
    """Readings for a station network as arrays, plus the shared time axis.

    `temp`, `pressure` and `rh` are all (n_stations, n_steps). NaN means the
    reading is absent, and it stays absent: filling a gap would fabricate the
    observation Layer 1 exists to catch.

    `genuine_events` holds the extreme-weather spans, half-open `[start, end)`,
    each with a `kind` and a magnitude. They are the negative controls. Without
    them the flagship "did we flag real weather as a fault" number cannot be
    computed at all.
    """

    def __init__(
        self,
        stations: List[Station],
        timestamps: List[datetime],
        temp: np.ndarray,
        pressure: np.ndarray,
        rh: np.ndarray,
        genuine_events: List[Dict],
        interval_minutes: int,
    ) -> None:
        self.stations = stations
        self.timestamps = timestamps
        self.temp = temp
        self.pressure = pressure
        self.rh = rh
        self.genuine_events = genuine_events
        self.interval_minutes = interval_minutes
        self.station_index = {s.station_id: i for i, s in enumerate(stations)}

    @property
    def n_stations(self) -> int:
        return len(self.stations)

    @property
    def n_steps(self) -> int:
        return len(self.timestamps)

    def observations(self) -> List[Observation]:
        """All observations in timestamp-major order.

        Timestamp-major, not station-major: this is how a real network delivers
        data, and the spatial layer needs contemporaneous readings available
        together or it has no neighbours to compare against.
        """
        out: List[Observation] = []
        for t_idx, ts in enumerate(self.timestamps):
            for s_idx, station in enumerate(self.stations):
                out.append(
                    Observation(
                        station_id=station.station_id,
                        timestamp=ts,
                        temp_c=_value(self.temp[s_idx, t_idx]),
                        pressure_hpa=_value(self.pressure[s_idx, t_idx]),
                        rh_pct=_value(self.rh[s_idx, t_idx]),
                    )
                )
        return out

    def station_series(self, station_id: str) -> Dict[str, np.ndarray]:
        i = self.station_index[station_id]
        return {
            "temp_c": self.temp[i].copy(),
            "pressure_hpa": self.pressure[i].copy(),
            "rh_pct": self.rh[i].copy(),
        }

    def copy(self) -> "StationNetwork":
        """Deep copy of the arrays, so injection cannot corrupt the clean data.

        The injector needs a mutable target and the harness needs the pristine
        original to compare against; sharing one array would silently destroy
        the ground truth.
        """
        return StationNetwork(
            stations=list(self.stations),
            timestamps=list(self.timestamps),
            temp=self.temp.copy(),
            pressure=self.pressure.copy(),
            rh=self.rh.copy(),
            genuine_events=[dict(e) for e in self.genuine_events],
            interval_minutes=self.interval_minutes,
        )

    def slice_steps(self, n_steps: int) -> "StationNetwork":
        """First `n_steps` columns, dropping events that fall outside them."""
        if n_steps >= self.n_steps:
            return self
        kept = [dict(e) for e in self.genuine_events if int(e["start"]) < n_steps]
        for event in kept:
            event["end"] = min(int(event["end"]), n_steps)
        return StationNetwork(
            stations=list(self.stations),
            timestamps=self.timestamps[:n_steps],
            temp=self.temp[:, :n_steps].copy(),
            pressure=self.pressure[:, :n_steps].copy(),
            rh=self.rh[:, :n_steps].copy(),
            genuine_events=kept,
            interval_minutes=self.interval_minutes,
        )

    def split_indices(
        self, train: float = 0.6, val: float = 0.2
    ) -> Tuple[slice, slice, slice]:
        """Chronological train/val/test split.

        Chronological, never random: splitting a time series at random leaks
        future information into training through adjacent windows and produces
        meaninglessly good results.
        """
        n = self.n_steps
        a = int(n * train)
        b = int(n * (train + val))
        return slice(0, a), slice(a, b), slice(b, n)


def _value(raw: float):
    """Array cell as a pipeline value: NaN is genuinely absent, never 0.0."""
    return None if np.isnan(raw) else float(raw)
