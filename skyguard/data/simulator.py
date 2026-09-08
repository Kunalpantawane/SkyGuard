"""Physically-grounded synthetic AWS network.

Why not random noise: every layer of the detector depends on structure that random
data does not have.

- Layer 2 (LSTM-AE) can only learn "normal" if there IS a learnable normal, i.e.
  diurnal and seasonal cycles.
- Layer 3 (multivariate) needs the real T-RH anticorrelation, which only appears
  if RH is derived from a slowly-varying dew point rather than drawn independently.
- Layer 4 (spatial) is untestable unless neighbouring stations genuinely move
  together. That requires spatially-correlated synoptic weather, not per-station
  noise.
- False-alarm measurement needs genuine extreme events that are extreme, real,
  and spatially coherent — labelled normal.

So the generator builds an atmospheric state, then samples sensors from it.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

import numpy as np

from ..config import SimulatorConfig
from ..types import Observation, Station

# Reference sea-level values for the barometric relation.
_P0_HPA = 1013.25
_T0_K = 288.15
_LAPSE_K_PER_M = 0.0065
_G = 9.80665
_R_DRY = 287.058


# --------------------------------------------------------------------------
# Station network
# --------------------------------------------------------------------------

# A plausible Indian AWS sub-network. Real coordinates and elevations, spanning
# coast, plain, plateau and hill terrain, because terrain diversity is exactly
# what makes naive "compare to nearest station" spatial QC fail.
_STATION_TEMPLATES: Tuple[Tuple[str, str, float, float, float, str], ...] = (
    ("AWS-1001", "Pune",        18.52, 73.86,  560.0, "plateau"),
    ("AWS-1002", "Lonavala",    18.75, 73.41,  622.0, "hill"),
    ("AWS-1003", "Mumbai",      19.08, 72.88,   14.0, "coastal"),
    ("AWS-1004", "Alibag",      18.64, 72.87,    7.0, "coastal"),
    ("AWS-1005", "Satara",      17.69, 74.00,  742.0, "plateau"),
    ("AWS-1006", "Mahabaleshwar", 17.92, 73.66, 1370.0, "hill"),
    ("AWS-1007", "Nashik",      19.99, 73.79,  565.0, "plateau"),
    ("AWS-1008", "Ahmednagar",  19.09, 74.74,  649.0, "inland"),
    ("AWS-1009", "Solapur",     17.66, 75.90,  457.0, "inland"),
    ("AWS-1010", "Ratnagiri",   16.99, 73.31,   67.0, "coastal"),
    ("AWS-1011", "Kolhapur",    16.70, 74.24,  569.0, "valley"),
    ("AWS-1012", "Aurangabad",  19.88, 75.34,  568.0, "inland"),
    ("AWS-1013", "Jalgaon",     21.01, 75.56,  209.0, "inland"),
    ("AWS-1014", "Nagpur",      21.15, 79.09,  310.0, "inland"),
    ("AWS-1015", "Panchgani",   17.92, 73.80, 1334.0, "hill"),
    ("AWS-1016", "Karad",       17.29, 74.18,  550.0, "valley"),
)


def build_network(n_stations: int) -> List[Station]:
    """Return the first `n_stations` template stations.

    Templates rather than random coordinates: real elevations and real spacing
    produce the terrain-driven differences that spatial QC must learn to
    tolerate. Random points would either be trivially similar or absurdly far.
    """
    if n_stations > len(_STATION_TEMPLATES):
        raise ValueError(
            f"only {len(_STATION_TEMPLATES)} template stations available, "
            f"asked for {n_stations}"
        )
    return [
        Station(sid, name, lat, lon, elev, terrain)
        for sid, name, lat, lon, elev, terrain in _STATION_TEMPLATES[:n_stations]
    ]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


# --------------------------------------------------------------------------
# Physics helpers
# --------------------------------------------------------------------------

def barometric_pressure_hpa(elevation_m: float) -> float:
    """Station pressure from elevation, standard atmosphere.

    Using the real barometric formula rather than a linear approximation matters
    because the network spans 7 m to 1370 m: a linear fudge would put hill
    stations at implausible pressures and make the multivariate layer learn a
    relationship that does not exist in real data.
    """
    exponent = _G / (_R_DRY * _LAPSE_K_PER_M)
    return _P0_HPA * (1.0 - _LAPSE_K_PER_M * elevation_m / _T0_K) ** exponent


def saturation_vapour_pressure_hpa(temp_c: np.ndarray) -> np.ndarray:
    """Magnus-Tetens saturation vapour pressure."""
    return 6.112 * np.exp(17.67 * temp_c / (temp_c + 243.5))


def rh_from_dewpoint(temp_c: np.ndarray, dewpoint_c: np.ndarray) -> np.ndarray:
    """Relative humidity from air temperature and dew point.

    This is the key coupling in the whole simulator. Dew point reflects absolute
    moisture and varies slowly; air temperature swings through the day. Deriving
    RH from the pair reproduces the real behaviour — RH climbing overnight as
    temperature falls toward the dew point — which is what gives Layer 3
    something genuine to check.
    """
    e = saturation_vapour_pressure_hpa(dewpoint_c)
    es = saturation_vapour_pressure_hpa(temp_c)
    return np.clip(100.0 * e / es, 0.5, 100.0)


def dewpoint_from_rh(temp_c: float, rh_pct: float) -> float:
    """Inverse of the above; used by the multivariate layer's physical check."""
    rh = max(min(rh_pct, 100.0), 0.1)
    gamma = math.log(rh / 100.0) + (17.67 * temp_c) / (temp_c + 243.5)
    return 243.5 * gamma / (17.67 - gamma)


# --------------------------------------------------------------------------
# Correlated synoptic fields
# --------------------------------------------------------------------------

def _smooth_random_walk(
    rng: np.random.Generator, n: int, timescale: float, amplitude: float
) -> np.ndarray:
    """An Ornstein-Uhlenbeck-ish series: random but temporally smooth.

    White noise would be trivially detectable as an anomaly and would teach the
    autoencoder nothing. Real weather wanders on a timescale of days, so the
    generator must too.
    """
    theta = 1.0 / max(timescale, 1e-6)
    x = np.zeros(n)
    sigma = amplitude * math.sqrt(2.0 * theta)
    for i in range(1, n):
        x[i] = x[i - 1] + (-theta * x[i - 1]) + sigma * rng.normal()
    # Normalise to the requested amplitude; the recursion's variance depends on
    # the timescale, so without this the amplitude config would be a lie.
    std = x.std()
    if std > 1e-9:
        x = x / std * amplitude
    return x


def _spatial_correlation_matrix(
    stations: List[Station], correlation_km: float
) -> np.ndarray:
    """Exponentially decaying spatial correlation between stations."""
    n = len(stations)
    c = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_km(
                stations[i].lat, stations[i].lon, stations[j].lat, stations[j].lon
            )
            rho = math.exp(-d / correlation_km)
            c[i, j] = c[j, i] = rho
    return c


def _correlated_synoptic_fields(
    rng: np.random.Generator,
    stations: List[Station],
    n_steps: int,
    cfg: SimulatorConfig,
    steps_per_hour: float,
) -> np.ndarray:
    """Generate spatially and temporally correlated synoptic anomalies.

    Returns (n_stations, n_steps). Built by drawing a small number of independent
    smooth "weather systems" and mixing them per station according to spatial
    correlation. This is what makes neighbouring stations move together — the
    single most important property for testing Layer 4.
    """
    n_st = len(stations)
    corr = _spatial_correlation_matrix(stations, cfg.synoptic_correlation_km)

    # Cholesky of the correlation matrix gives us the mixing weights. Jitter the
    # diagonal because a near-singular correlation matrix (very close stations)
    # otherwise breaks the decomposition.
    jitter = 1e-6
    for _ in range(8):
        try:
            chol = np.linalg.cholesky(corr + jitter * np.eye(n_st))
            break
        except np.linalg.LinAlgError:
            jitter *= 10
    else:
        chol = np.eye(n_st)

    timescale_steps = cfg.synoptic_timescale_hours * steps_per_hour
    base = np.stack(
        [
            _smooth_random_walk(rng, n_steps, timescale_steps, 1.0)
            for _ in range(n_st)
        ]
    )
    return chol @ base


# --------------------------------------------------------------------------
# Per-station climatology
# --------------------------------------------------------------------------

def _station_climatology(station: Station) -> Dict[str, float]:
    """Terrain-dependent climate parameters.

    Coastal stations have damped diurnal range (sea thermal inertia) and higher
    dew points; hill stations are cooler with larger diurnal range. Encoding this
    means one trained model has to cope with genuinely different regimes, which
    is the real deployment condition.
    """
    lapse_cooling = station.elevation_m * _LAPSE_K_PER_M

    if station.terrain == "coastal":
        mean_t, diurnal, seasonal, dew_base, dew_season = 27.5, 4.5, 4.0, 22.0, 4.0
    elif station.terrain == "hill":
        mean_t, diurnal, seasonal, dew_base, dew_season = 26.0, 8.0, 7.0, 15.0, 6.0
    elif station.terrain == "valley":
        mean_t, diurnal, seasonal, dew_base, dew_season = 27.0, 11.0, 8.0, 17.0, 6.5
    elif station.terrain == "plateau":
        mean_t, diurnal, seasonal, dew_base, dew_season = 26.5, 10.0, 7.5, 16.0, 6.0
    else:  # inland
        mean_t, diurnal, seasonal, dew_base, dew_season = 27.5, 11.5, 8.5, 16.0, 6.5

    return {
        "mean_temp_c": mean_t - lapse_cooling,
        "diurnal_amplitude": diurnal,
        "seasonal_amplitude": seasonal,
        "dewpoint_base_c": dew_base - lapse_cooling * 0.6,
        "dewpoint_seasonal": dew_season,
        "base_pressure_hpa": barometric_pressure_hpa(station.elevation_m),
    }


# --------------------------------------------------------------------------
# Genuine extreme events (negative controls)
# --------------------------------------------------------------------------

def _apply_genuine_extremes(
    rng: np.random.Generator,
    temp: np.ndarray,
    pressure: np.ndarray,
    dewpoint: np.ndarray,
    cfg: SimulatorConfig,
    steps_per_hour: float,
) -> List[Dict]:
    """Add real extreme weather, spatially coherent, labelled normal.

    These are the false-alarm test set. A detector that flags a heatwave has
    failed the most important operational requirement, and without coherent
    extremes in the clean data there is no way to measure that failure.

    Modifies arrays in place; returns descriptors for the evaluation harness.
    """
    n_st, n_steps = temp.shape
    events: List[Dict] = []

    def _window(duration_steps: int) -> Tuple[int, int]:
        # Short demo series cannot host a 25-day monsoon: truncate the event
        # to the series rather than broadcasting past its end.
        dur = max(1, min(int(duration_steps), n_steps - 1))
        start = int(rng.integers(0, max(1, n_steps - dur)))
        return start, min(start + dur, n_steps)

    # Heatwave: multi-day warm anomaly across the whole network, with the
    # physically correct signature — dew point does NOT rise with it, so RH
    # falls. A sensor fault would not reproduce that relationship.
    for _ in range(cfg.n_heatwaves):
        dur = int(rng.integers(3 * 24, 8 * 24) * steps_per_hour)
        s, e = _window(dur)
        peak = rng.uniform(4.5, 8.0)
        ramp = np.sin(np.linspace(0, math.pi, e - s)) ** 0.6
        for i in range(n_st):
            scale = rng.uniform(0.75, 1.0)   # not identical everywhere
            temp[i, s:e] += peak * scale * ramp
            dewpoint[i, s:e] -= 1.5 * scale * ramp
        events.append({"kind": "heatwave", "start": s, "end": e, "peak_c": peak})

    # Pressure surge / depression: coherent pressure excursion with the
    # temperature and moisture response you would actually see.
    for _ in range(cfg.n_pressure_surges):
        dur = int(rng.integers(18, 72) * steps_per_hour)
        s, e = _window(dur)
        depth = rng.uniform(-16.0, -7.0) if rng.random() < 0.6 else rng.uniform(7.0, 13.0)
        ramp = np.sin(np.linspace(0, math.pi, e - s))
        for i in range(n_st):
            scale = rng.uniform(0.8, 1.0)
            pressure[i, s:e] += depth * scale * ramp
            # Low pressure brings cloud and moisture: cooler, more humid.
            temp[i, s:e] -= 0.22 * depth * scale * ramp * (-1 if depth < 0 else 0.4)
            dewpoint[i, s:e] += (-0.16 * depth) * scale * ramp
        events.append({"kind": "pressure_surge", "start": s, "end": e, "depth_hpa": depth})

    # Monsoon onset: a fast step to a much wetter, cooler regime that persists.
    # This is the hardest legitimate event because it looks like a step fault —
    # the difference is that it happens everywhere at once and moves all three
    # variables coherently.
    for _ in range(cfg.n_monsoon_onsets):
        dur = int(rng.integers(10 * 24, 25 * 24) * steps_per_hour)
        s, e = _window(dur)
        ramp_len = max(1, int(18 * steps_per_hour))
        shape = np.ones(e - s)
        shape[:ramp_len] = np.linspace(0, 1, min(ramp_len, e - s))
        for i in range(n_st):
            scale = rng.uniform(0.85, 1.0)
            temp[i, s:e] -= 3.2 * scale * shape
            dewpoint[i, s:e] += 4.5 * scale * shape
            pressure[i, s:e] -= 3.0 * scale * shape
        events.append({"kind": "monsoon_onset", "start": s, "end": e})

    return events


# --------------------------------------------------------------------------
# Public result
# --------------------------------------------------------------------------

class SimulatedNetwork:
    """Clean synthetic observations for a station network.

    Holds the arrays as well as the Observation lists: the evaluation harness and
    the injector work on arrays (fast, index-addressable), while the pipeline
    consumes Observations (the real streaming interface).
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
        data, and the spatial layer needs contemporaneous readings to be
        available together.
        """
        out: List[Observation] = []
        for t_idx, ts in enumerate(self.timestamps):
            for s_idx, station in enumerate(self.stations):
                out.append(
                    Observation(
                        station_id=station.station_id,
                        timestamp=ts,
                        temp_c=float(self.temp[s_idx, t_idx]),
                        pressure_hpa=float(self.pressure[s_idx, t_idx]),
                        rh_pct=float(self.rh[s_idx, t_idx]),
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

    def copy(self) -> "SimulatedNetwork":
        """Deep copy of the arrays, so injection cannot corrupt the clean data.

        The injector needs a mutable target and the harness needs the pristine
        original to compare against; sharing one array would silently destroy the
        ground truth.
        """
        return SimulatedNetwork(
            stations=list(self.stations),
            timestamps=list(self.timestamps),
            temp=self.temp.copy(),
            pressure=self.pressure.copy(),
            rh=self.rh.copy(),
            genuine_events=[dict(e) for e in self.genuine_events],
            interval_minutes=self.interval_minutes,
        )

    def split_indices(
        self, train: float = 0.6, val: float = 0.2
    ) -> Tuple[slice, slice, slice]:
        """Chronological train/val/test split.

        Chronological, never random: randomly splitting a time series leaks
        future information into training through adjacent windows and produces
        meaninglessly good results.
        """
        n = self.n_steps
        a = int(n * train)
        b = int(n * (train + val))
        return slice(0, a), slice(a, b), slice(b, n)


# --------------------------------------------------------------------------
# Generator
# --------------------------------------------------------------------------

def simulate_network(cfg: SimulatorConfig | None = None) -> SimulatedNetwork:
    """Generate a clean, physically plausible AWS network time series."""
    cfg = cfg or SimulatorConfig()
    rng = np.random.default_rng(cfg.seed)

    stations = build_network(cfg.n_stations)
    steps_per_hour = 60.0 / cfg.interval_minutes
    n_steps = int(cfg.days * 24 * steps_per_hour)
    if n_steps < 2:
        raise ValueError("simulation too short: need at least 2 steps")

    start = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
    timestamps = [
        start + timedelta(minutes=cfg.interval_minutes * i) for i in range(n_steps)
    ]

    hours = np.arange(n_steps) / steps_per_hour
    day_of_year = (hours / 24.0) % 365.25

    # Diurnal phase: minimum near 05:00, maximum near 15:00. Getting this right
    # matters because the model's cyclic time features must align with reality
    # for "38 °C at 04:00 is wrong" to be learnable.
    diurnal = np.cos(2 * math.pi * (hours - 15.0) / 24.0)
    # Northern-hemisphere seasonal peak around mid-May for this region.
    seasonal = np.cos(2 * math.pi * (day_of_year - 135.0) / 365.25)

    synoptic = _correlated_synoptic_fields(rng, stations, n_steps, cfg, steps_per_hour)

    n_st = len(stations)
    temp = np.zeros((n_st, n_steps))
    pressure = np.zeros((n_st, n_steps))
    dewpoint = np.zeros((n_st, n_steps))

    for i, station in enumerate(stations):
        clim = _station_climatology(station)
        syn = synoptic[i]

        temp[i] = (
            clim["mean_temp_c"]
            + clim["diurnal_amplitude"] * diurnal
            + clim["seasonal_amplitude"] * seasonal
            + 1.7 * syn
        )

        # Pressure: station base, weak inverse-seasonal term, synoptic systems,
        # and the semidiurnal atmospheric tide — a real, textbook feature that
        # gives the autoencoder genuine fine structure to learn.
        tide = 0.9 * np.sin(2 * math.pi * (hours - 10.0) / 12.0)
        pressure[i] = (
            clim["base_pressure_hpa"]
            - 2.5 * seasonal
            + cfg.synoptic_pressure_amplitude * syn * 0.55
            + tide
        )

        # Dew point tracks the season strongly (monsoon moisture) and the synoptic
        # state weakly, with only a small diurnal component.
        dewpoint[i] = (
            clim["dewpoint_base_c"]
            + clim["dewpoint_seasonal"] * seasonal
            + 0.9 * syn
            + 0.5 * diurnal
        )

    genuine_events = _apply_genuine_extremes(
        rng, temp, pressure, dewpoint, cfg, steps_per_hour
    )

    # Dew point cannot exceed air temperature. Clamp before deriving RH so the
    # clean data never contains the very physical violation Layer 3 looks for —
    # otherwise the "clean" set would be full of impossible states.
    dewpoint = np.minimum(dewpoint, temp - 0.35)

    rh = rh_from_dewpoint(temp, dewpoint)

    # Independent measurement noise, added last so it is not smoothed by the
    # physics. This is the irreducible noise floor the detector must tolerate.
    temp += rng.normal(0.0, cfg.noise_temp_c, temp.shape)
    pressure += rng.normal(0.0, cfg.noise_pressure_hpa, pressure.shape)
    rh = np.clip(rh + rng.normal(0.0, cfg.noise_rh_pct, rh.shape), 0.5, 100.0)

    return SimulatedNetwork(
        stations=stations,
        timestamps=timestamps,
        temp=temp,
        pressure=pressure,
        rh=rh,
        genuine_events=genuine_events,
        interval_minutes=cfg.interval_minutes,
    )
