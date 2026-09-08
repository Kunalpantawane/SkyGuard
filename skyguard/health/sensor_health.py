"""Sensor health and degradation: "becoming untrustworthy" versus "wrong now".

Why this is not anomaly detection: a single spike says something unusual
happened; a rising anomaly rate with growing reconstruction error and bias
says the sensor is dying. Different questions, different time scales,
different actions — the first quarantines an observation, the second schedules
a maintenance visit *before* the hard fault (`context/evaluation.md`
health-lead time).

Each sensor starts at 100 and loses points across four capped deductions
(anomaly rate, error trend, bias, drift). The station inherits its sickest
sensor: a station is only as trustworthy as its worst channel.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..config import DEFAULT_CONFIG, HealthConfig
from ..types import HealthStatus, SensorHealth, StationHealth


def trend_indicator(errors: np.ndarray) -> float:
    """How strongly reconstruction error ramps over the window, in [0, 1].

    Least-squares slope scaled by window length over the mean: an error that
    doubles across the window scores ~1, flat noise scores ~0. Relative (not
    absolute) so one threshold serves quiet and noisy sensors alike.
    """
    series = np.asarray(errors, dtype=np.float64).ravel()
    if series.size < 2 or not np.all(np.isfinite(series)):
        return 0.0
    mean = float(series.mean())
    if mean <= 0.0:
        return 0.0
    x = np.arange(series.size, dtype=np.float64)
    slope = float(np.cov(x, series, bias=True)[0, 1] / max(np.var(x), 1e-12))
    return float(min(max(slope * series.size / mean, 0.0), 1.0))


def bias_indicator(residuals: np.ndarray) -> float:
    """Persistent offset versus expectation, in [0, 1].

    |mean| in units of spread: centred noise scores ~0, a steady one-sigma
    offset scores ~1. Random scatter must not read as bias.
    """
    series = np.asarray(residuals, dtype=np.float64).ravel()
    if series.size == 0 or not np.all(np.isfinite(series)):
        return 0.0
    spread = float(series.std())
    if spread <= 0.0:
        return 0.0 if float(series.mean()) == 0.0 else 1.0
    return float(min(abs(float(series.mean())) / spread, 1.0))


def score_sensor(
    variable: str,
    n_points: int,
    anomaly_rate: float,
    error_trend: float,
    bias: float,
    drift: float,
    config: HealthConfig | None = None,
) -> SensorHealth:
    """Deduct from 100 across the four capped penalties.

    All indicators are clipped to [0, 1] on entry so a wild input cannot
    deduct more than its configured weight — the weights sum to the maximum
    possible penalty, keeping the score an honest ledger.
    """
    cfg = config or DEFAULT_CONFIG.health
    if n_points < cfg.min_points:
        return SensorHealth(
            variable=variable, score=100.0, status=HealthStatus.UNKNOWN,
            reasons=[f"only {n_points} points (< {cfg.min_points}): not enough history"],
        )
    rate = float(min(max(anomaly_rate, 0.0), 1.0))
    trend = float(min(max(error_trend, 0.0), 1.0))
    bias_hit = float(min(max(bias, 0.0), 1.0))
    drift_hit = float(min(max(abs(drift), 0.0), 1.0))
    penalties = (
        ("anomaly_rate", rate, cfg.weight_anomaly_rate),
        ("error_trend", trend, cfg.weight_error_trend),
        ("bias", bias_hit, cfg.weight_bias),
        ("drift", drift_hit, cfg.weight_drift),
    )
    score = 100.0
    reasons: List[str] = []
    for name, level, weight in penalties:
        deduction = level * weight
        score -= deduction
        if deduction >= 1.0:
            reasons.append(f"{name}={level:.2f} (-{deduction:.1f})")
    score = float(max(score, 0.0))
    return SensorHealth(
        variable=variable, score=score, status=_status(score, cfg),
        anomaly_rate=rate, error_trend=trend, bias=bias_hit,
        drift_slope=float(drift), reasons=reasons or ["nominal"],
    )


def assess_station(
    station_id: str,
    sensors: Dict[str, SensorHealth],
    n_points: int,
    config: HealthConfig | None = None,
) -> StationHealth:
    """Roll sensors up: the sickest channel defines the station.

    Averaging would let two healthy sensors hide a dying one — exactly the
    failure this layer exists to catch early.
    """
    cfg = config or DEFAULT_CONFIG.health
    if not sensors or n_points < cfg.min_points:
        return StationHealth(station_id=station_id, status=HealthStatus.UNKNOWN,
                             sensors=dict(sensors), n_points=n_points)
    overall = min(sensor.score for sensor in sensors.values())
    status = _status(overall, cfg)
    risk = {
        HealthStatus.CRITICAL: "HIGH",
        HealthStatus.DEGRADING: "MEDIUM",
        HealthStatus.WARNING: "LOW",
        HealthStatus.HEALTHY: "LOW",
        HealthStatus.UNKNOWN: "UNKNOWN",
    }[status]
    return StationHealth(
        station_id=station_id, overall_score=float(overall), status=status,
        sensors=dict(sensors), maintenance_risk=risk, n_points=n_points,
    )


def _status(score: float, config: HealthConfig) -> HealthStatus:
    """First band whose upper edge exceeds the score."""
    for upper, name in config.bands:
        if score < upper:
            return HealthStatus[name]
    return HealthStatus.HEALTHY
