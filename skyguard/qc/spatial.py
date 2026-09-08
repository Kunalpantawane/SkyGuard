"""Layer 4 — spatial consensus against comparable neighbours.

Why neighbours, and why *comparable* ones: a station that disagrees with its
neighbourhood is the strongest single signal for "this station is broken"
versus "the weather changed". But a hilltop legitimately differs from a
valley, so comparison is weighted by true comparability (distance decay plus
elevation penalty) with physical corrections — lapse-rate for temperature,
barometric for pressure — not by raw proximity.

When too few comparable neighbours exist the layer abstains (`available=False`)
rather than guessing: a verdict from one distant station at a different
elevation is worse than no spatial opinion at all, and downstream fusion
renormalises over the signals that did speak.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from ..config import DEFAULT_CONFIG, VARIABLES, SpatialConfig
from ..types import Observation, SpatialResult, Station

# Dry-air gas constant and gravity for the barometric elevation correction.
_R_DRY = 287.058
_G = 9.80665
_EARTH_RADIUS_KM = 6371.0

# Residuals are already in spread units (sigma-like), so a unit logistic scale
# is principled here, matching Layer 3's Mahalanobis-scale reasoning.
_PROBABILITY_SCALE = 1.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two station coordinates."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    arc = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * _EARTH_RADIUS_KM * math.asin(math.sqrt(max(min(arc, 1.0), 0.0)))


def correct_temp_to_elevation(
    temp_neighbour: float, elev_neighbour_m: float, elev_target_m: float, lapse_rate: float
) -> float:
    """Bring a neighbour's temperature to the target's elevation.

    Air cools ~6.5 C per km of ascent, so a hill station reading 27 C can mean
    exactly the same airmass as 30 C in the valley below it. Comparing raw
    values would cry fault at terrain.
    """
    return temp_neighbour + lapse_rate * (elev_neighbour_m - elev_target_m)


def correct_pressure_to_elevation(
    pressure_neighbour: float, temp_neighbour_c: float,
    elev_neighbour_m: float, elev_target_m: float,
) -> float:
    """Bring a neighbour's station pressure to the target's elevation.

    Barometric relation with the neighbour's own temperature as the layer mean
    (falling back to standard 15 C when it is missing or unphysical). An
    approximation — the full atmosphere is not isothermal — but far closer
    than comparing station pressures across hundreds of metres directly.
    """
    temp_k = temp_neighbour_c + 273.15 if temp_neighbour_c is not None else 288.15
    if not math.isfinite(temp_k) or temp_k < 200.0:
        temp_k = 288.15
    climb = elev_target_m - elev_neighbour_m
    return pressure_neighbour * math.exp(-_G * climb / (_R_DRY * temp_k))


class SpatialQC:
    """Neighbour-consensus scoring with comparability weighting."""

    def __init__(
        self, stations: Dict[str, Station], config: SpatialConfig | None = None
    ) -> None:
        self.config: SpatialConfig = config or DEFAULT_CONFIG.spatial
        self.stations: Dict[str, Station] = dict(stations)

    # -- neighbour selection ------------------------------------------------

    def _comparable(
        self, target: Observation, neighbours: List[Observation]
    ) -> List[Tuple[Observation, float]]:
        """Filter to comparable neighbours with weights, best first."""
        cfg = self.config
        own = self.stations.get(target.station_id)
        if own is None:
            return []
        scored: List[Tuple[Observation, float]] = []
        for cand in neighbours:
            if cand.station_id == target.station_id:
                continue
            meta = self.stations.get(cand.station_id)
            if meta is None:
                continue  # unknown station: comparability cannot be assessed
            skew = abs((target.timestamp - cand.timestamp).total_seconds()) / 60.0
            if skew > cfg.max_time_skew_minutes:
                continue
            distance = haversine_km(own.lat, own.lon, meta.lat, meta.lon)
            if distance > cfg.max_distance_km:
                continue
            elev_gap = abs(meta.elevation_m - own.elevation_m)
            if elev_gap > cfg.max_elevation_diff_m:
                continue
            weight = math.exp(-distance / cfg.distance_scale_km) * math.exp(
                -elev_gap / cfg.elevation_scale_m
            )
            scored.append((cand, weight))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[: cfg.max_neighbours]

    # -- evaluation -----------------------------------------------------------

    def evaluate(
        self, target: Observation, neighbours: List[Observation]
    ) -> SpatialResult:
        """Score the target against its comparable neighbourhood."""
        cfg = self.config
        own = self.stations.get(target.station_id)
        if own is None:
            return SpatialResult(available=False)
        comparable = self._comparable(target, neighbours)
        if len(comparable) < cfg.min_neighbours:
            return SpatialResult(available=False)

        actual = target.values()
        expected: Dict[str, float] = {}
        residuals: Dict[str, float] = {}
        normalised: Dict[str, float] = {}
        for var in VARIABLES:
            value = actual[var]
            if value is None or not math.isfinite(value):
                continue
            corrected: List[float] = []
            weights: List[float] = []
            for cand, weight in comparable:
                reading = cand.value(var)
                if reading is None or not math.isfinite(reading):
                    continue
                meta = self.stations[cand.station_id]
                if var == "temp_c":
                    reading = correct_temp_to_elevation(
                        reading, meta.elevation_m, own.elevation_m, cfg.lapse_rate_c_per_m
                    )
                elif var == "pressure_hpa":
                    reading = correct_pressure_to_elevation(
                        reading, cand.temp_c, meta.elevation_m, own.elevation_m
                    )
                corrected.append(reading)
                weights.append(weight)
            if not corrected:
                continue
            arr = np.array(corrected, dtype=np.float64)
            wsum = float(sum(weights))
            mean = float(np.dot(arr, np.array(weights)) / wsum)
            variance = float(np.dot(weights, (arr - mean) ** 2) / wsum)
            spread = max(math.sqrt(max(variance, 0.0)), cfg.min_spread[var])
            residual = abs(float(value) - mean)
            expected[var] = mean
            residuals[var] = residual
            normalised[var] = residual / spread

        if not normalised:
            return SpatialResult(available=False)
        peak = max(normalised.values())
        probability = 1.0 / (1.0 + math.exp(-(peak - cfg.residual_sigma) / _PROBABILITY_SCALE))
        return SpatialResult(
            available=True,
            probability=float(probability),
            residuals=residuals,
            normalised=normalised,
            expected=expected,
            n_neighbours=len(comparable),
            neighbour_ids=[cand.station_id for cand, _ in comparable],
        )
