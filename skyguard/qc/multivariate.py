"""Layer 3 — multivariate joint-state consistency of T, P and RH.

Why this layer exists: the three variables are one atmospheric state, not three
independent streams. 55 C with 99 % RH can pass every individual range check
yet be jointly impossible. The autoencoder captures part of this, but an
explicit Mahalanobis check on levels plus step deltas says *which combination*
broke, in units a meteorologist can read.

Two mechanisms, applied where each is appropriate:

1. **Mahalanobis distance** on [T, P, RH, dT, dP, dRH] against the station's
   own clean history. Deltas matter as much as levels: a joint jump that lands
   on a plausible state is still suspicious arriving in one step.
2. **Dew-point guard**: dew point above air temperature is a hard physical
   impossibility (at RH <= 100 % it cannot happen), not a statistical opinion.
   It fires rarely — Layer 1 range checks already exclude most such states —
   but when it fires it is close to certain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from ..config import DEFAULT_CONFIG, VARIABLES, MultivariateConfig
from ..types import MultivariateResult

# Magnus-form coefficients (valid roughly -45..60 C). Shared with the
# simulator's dew-point coupling so "consistent" means the same thing in both.
_MAGNUS_B = 17.625
_MAGNUS_C = 243.04

# Probability model lives in Mahalanobis units, which are sigma units by
# construction — so a unit scale is principled here, not a magic number.
_PROBABILITY_SCALE = 1.0

# A hard physical violation sits above fusion's HIGH band but below a Layer 1
# hard-rule verdict: near-certain, but a sentinel value is more certain still.
_PHYSICAL_VIOLATION_PROBABILITY = 0.9

# Feature order: three levels, then three step deltas. Fixed because the
# precision matrix columns depend on it.
_N_DIMS = 6


def dewpoint_c(temp_c: float, rh_pct: float) -> float:
    """Dew point via the Magnus relation.

    RH is clamped into (0, 100]: exactly 0 % is unphysical sensor output and
    would send the logarithm to -inf, hiding the real problem behind a NaN.
    """
    rh = min(max(float(rh_pct), 0.1), 100.0)
    gamma = math.log(rh / 100.0) + _MAGNUS_B * temp_c / (_MAGNUS_C + temp_c)
    return _MAGNUS_C * gamma / (_MAGNUS_B - gamma)


@dataclass
class _StationModel:
    """A station's clean joint-state description."""

    median: np.ndarray       # (6,) robust location
    scale: np.ndarray        # (6,) MAD-based sigma per dimension
    precision: np.ndarray    # (6, 6) inverse of shrunk correlation
    n_samples: int


class MultivariateQC:
    """Joint-state consistency against per-station clean history.

    Fitted per station (never global): elevation and microclimate shift the
    joint distribution, and a global fit would blur exactly the structure this
    layer is meant to test. Streaming state is one previous observation per
    station — O(1), as the streaming convention requires.
    """

    def __init__(self, config: MultivariateConfig | None = None) -> None:
        self.config: MultivariateConfig = config or DEFAULT_CONFIG.multivariate
        self._models: Dict[str, _StationModel] = {}
        self._last: Dict[str, np.ndarray] = {}

    # -- fitting ----------------------------------------------------------

    def fit_station(self, station_id: str, history: np.ndarray) -> int:
        """Fit from clean level history, shape (N, 3) in VARIABLES order.

        Deltas are derived internally from consecutive rows, so the caller
        passes raw levels. Returns the feature rows used. Raises ValueError on
        non-finite input — fitting on NaNs would silently bless everything.
        """
        levels = np.asarray(history, dtype=np.float64)
        if levels.ndim != 2 or levels.shape[1] != 3:
            raise ValueError(f"history must be (N, 3), got {levels.shape}")
        if not np.all(np.isfinite(levels)):
            raise ValueError("multivariate fit requires finite clean history")
        if levels.shape[0] < self.config.min_samples + 1:
            return 0
        deltas = np.diff(levels, axis=0)
        features = np.concatenate([levels[1:], deltas], axis=1)
        median = np.median(features, axis=0)
        mad = np.median(np.abs(features - median), axis=0)
        scale = np.maximum(mad * 1.4826, 1e-9)
        standardised = (features - median) / scale
        corr = (standardised.T @ standardised) / features.shape[0]
        shrunk = (1.0 - self.config.shrinkage) * corr + self.config.shrinkage * np.eye(_N_DIMS)
        self._models[station_id] = _StationModel(
            median=median, scale=scale,
            precision=np.linalg.inv(shrunk), n_samples=features.shape[0],
        )
        self._last[station_id] = levels[-1].copy()
        return features.shape[0]

    def is_fitted(self, station_id: str) -> bool:
        """Whether this station has a usable joint-state model."""
        return station_id in self._models

    def reset(self, station_id: str) -> None:
        """Drop streaming delta state (keeps the fitted model)."""
        self._last.pop(station_id, None)

    # -- evaluation ---------------------------------------------------------

    def evaluate(
        self,
        station_id: str,
        temp_c: Optional[float],
        pressure_hpa: Optional[float],
        rh_pct: Optional[float],
        hard_invalid: bool = False,
    ) -> MultivariateResult:
        """Score one observation's joint state.

        Unavailable (not a pass) when the triplet is incomplete, the station
        is unfitted, or no previous observation exists for deltas — a joint
        verdict from a partial state would be a guess dressed as analysis.

        `hard_invalid` means Layer 1 already proved a value impossible (sentinel
        or out of physical range). Such a reading is abstained on AND withheld
        from the delta baseline: banking -9999 as "last" would hand the next
        perfectly healthy observation a delta of ten thousand and convict it.
        Layer 1 protects its own state the same way.
        """
        if hard_invalid:
            return MultivariateResult(available=False)
        if temp_c is None or pressure_hpa is None or rh_pct is None:
            return MultivariateResult(available=False)
        if station_id not in self._models:
            return MultivariateResult(available=False)
        model = self._models[station_id]
        current = np.array([temp_c, pressure_hpa, rh_pct], dtype=np.float64)
        if not np.all(np.isfinite(current)):
            return MultivariateResult(available=False)
        if station_id not in self._last:
            # Cold start (or post-reset): bank this observation as the delta
            # baseline and abstain for this one point. Without this the layer
            # could never recover after a reset.
            self._last[station_id] = current.copy()
            return MultivariateResult(available=False)
        delta = current - self._last[station_id]
        self._last[station_id] = current.copy()

        features = np.concatenate([current, delta])
        standardised = (features - model.median) / model.scale
        scored = standardised @ model.precision @ standardised
        distance = float(np.sqrt(max(scored, 0.0)))

        dew = dewpoint_c(temp_c, rh_pct)
        physical_violation = bool(dew > temp_c + self.config.dewpoint_excess_tolerance_c)

        probability = 1.0 / (1.0 + math.exp(-(distance - self.config.mahalanobis_threshold)
                                            / _PROBABILITY_SCALE))
        if physical_violation:
            probability = max(probability, _PHYSICAL_VIOLATION_PROBABILITY)

        per_variable = _attribute(model, standardised, scored)
        detail = (
            f"Mahalanobis d={distance:.2f} vs {self.config.mahalanobis_threshold:.2f}; "
            f"dewpoint {dew:.1f}C vs air {temp_c:.1f}C"
            + (" — PHYSICAL VIOLATION" if physical_violation else "")
        )
        return MultivariateResult(
            available=True,
            mahalanobis=distance,
            probability=float(min(max(probability, 0.0), 1.0)),
            per_variable=per_variable,
            dewpoint_c=float(dew),
            physical_violation=physical_violation,
            detail=detail,
        )


def _attribute(
    model: _StationModel, standardised: np.ndarray, scored: float
) -> Dict[str, float]:
    """Split d-squared across the three physical variables.

    The diagonal decomposition z_i * (P z)_i sums exactly to d-squared, so
    each variable's share is its honest contribution including correlation
    effects. Level and delta shares merge per variable; negative shares
    (suppression via correlation) clip to zero and the rest renormalise.
    """
    if scored <= 0.0:
        return {var: 0.0 for var in VARIABLES}
    weighted = standardised * (model.precision @ standardised) / scored
    shares = {}
    for k, var in enumerate(VARIABLES):
        share = float(weighted[k] + weighted[k + 3])
        shares[var] = max(share, 0.0)
    total = sum(shares.values())
    if total <= 0.0:
        return {var: 0.0 for var in VARIABLES}
    return {var: value / total for var, value in shares.items()}
