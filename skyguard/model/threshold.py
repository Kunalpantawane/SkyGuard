"""Adaptive per-station thresholding on LSTM reconstruction error.

Why per-station instead of global: a raw error of 0.42 means nothing in the
abstract — a Himalayan station and a coastal station have different normal
error structure, and error variance is itself seasonal. So each station's
threshold is fitted from that station's own *held-out clean* errors (never
training data, never injected faults; see `context/data.md`).

Robustness is the point: median and MAD, not mean/std, because outliers in the
calibration set would inflate the threshold and hide real faults. The
high-quantile fallback covers skewed error distributions where K-sigma alone
would sit too low and flood operators with false alarms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from ..config import DEFAULT_CONFIG, ThresholdConfig

# MAD-to-sigma consistency factor for a Gaussian. Lets MAD speak the same
# language as K-sigma while staying robust to calibration outliers.
_MAD_TO_SIGMA = 1.4826

# Floor under the robust scale so a freakishly constant calibration set cannot
# produce a division by zero or an absurdly sharp logistic cliff.
_MIN_SIGMA = 1e-9


@dataclass
class StationThreshold:
    """One station's calibrated anomaly boundary.

    `bucket_thresholds` holds optional per-bucket (hour-of-day, season)
    thresholds keyed by bucket id. Buckets with too little history are absent
    and fall back to `threshold` — an explicit abstention, not a guess.
    """

    station_id: str
    threshold: float
    median: float
    sigma_robust: float
    quantile_value: float
    n_samples: int
    from_network_fallback: bool = False
    bucket_thresholds: Dict[int, float] = field(default_factory=dict)

    def threshold_for(self, bucket: Optional[int] = None) -> float:
        """Active threshold for an observation, bucket-aware."""
        if bucket is not None and bucket in self.bucket_thresholds:
            return self.bucket_thresholds[bucket]
        return self.threshold

    def to_dict(self) -> Dict:
        return {
            "station_id": self.station_id,
            "threshold": self.threshold,
            "median": self.median,
            "sigma_robust": self.sigma_robust,
            "quantile_value": self.quantile_value,
            "n_samples": self.n_samples,
            "from_network_fallback": self.from_network_fallback,
            "bucket_thresholds": dict(self.bucket_thresholds),
        }


def _checked_errors(errors: np.ndarray) -> np.ndarray:
    """Validate calibration errors loudly: silent NaNs here become silent misses."""
    arr = np.asarray(errors, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("cannot fit a threshold on zero calibration errors")
    if not np.all(np.isfinite(arr)):
        raise ValueError("calibration errors must be finite (got NaN or inf)")
    return arr


def fit_level(
    errors: np.ndarray, config: ThresholdConfig | None = None
) -> Tuple[float, float, float, float]:
    """Fit one threshold level; returns (threshold, median, sigma, quantile).

    Threshold is `max(median + k·sigma, quantile_q)`: for near-Gaussian errors
    K-sigma rules, for skewed errors the high quantile takes over.
    """
    cfg = config or DEFAULT_CONFIG.threshold
    arr = _checked_errors(errors)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    sigma = mad * _MAD_TO_SIGMA
    quantile_value = float(np.quantile(arr, cfg.quantile))
    threshold = max(median + cfg.k_sigma * sigma, quantile_value)
    return threshold, median, sigma, quantile_value


def fit_bucket_thresholds(
    errors: np.ndarray,
    bucket_ids: np.ndarray,
    config: ThresholdConfig | None = None,
    min_bucket_samples: int = 50,
) -> Dict[int, float]:
    """Per-bucket thresholds where history suffices; thin buckets are skipped.

    `min_bucket_samples` defaults to 50: the floor for a stable MAD fit on a
    data slice. Skipped buckets are absent from the result so scoring falls
    back to the station threshold rather than trusting a noisy estimate.
    """
    cfg = config or DEFAULT_CONFIG.threshold
    arr = _checked_errors(errors)
    buckets = np.asarray(bucket_ids).ravel()
    if buckets.shape != arr.shape:
        raise ValueError(
            f"bucket_ids shape {buckets.shape} must match errors shape {arr.shape}"
        )
    out: Dict[int, float] = {}
    for bucket in sorted(set(buckets.tolist())):
        assert isinstance(bucket, (int, np.integer))
        key = int(bucket)
        sample = arr[buckets == bucket]
        if sample.size < min_bucket_samples:
            continue
        level, _, _, _ = fit_level(sample, cfg)
        out[key] = level
    # Skipped buckets stay absent: scoring falls back to the station threshold
    # rather than trusting a noisy thin-slice estimate.
    return out


def fit_station(
    station_id: str,
    errors: np.ndarray,
    network_threshold: float,
    network_median: float = 0.0,
    network_sigma: float = 0.0,
    config: ThresholdConfig | None = None,
    bucket_ids: np.ndarray | None = None,
    min_bucket_samples: int = 50,
) -> StationThreshold:
    """Calibrate one station, falling back to the network level when thin.

    Below `min_samples` the station's own distribution is statistically thin,
    so the network-wide threshold stands in and `from_network_fallback` says
    so — a guarded number beats a confident-but-noisy one.
    """
    cfg = config or DEFAULT_CONFIG.threshold
    arr = _checked_errors(errors)
    if arr.size < cfg.min_samples:
        buckets: Dict[int, float] = {}
        if bucket_ids is not None:
            buckets = fit_bucket_thresholds(
                arr, np.asarray(bucket_ids), cfg, min_bucket_samples
            )
        return StationThreshold(
            station_id=station_id,
            threshold=float(network_threshold),
            median=float(network_median),
            sigma_robust=float(network_sigma),
            quantile_value=float(network_threshold),
            n_samples=int(arr.size),
            from_network_fallback=True,
            bucket_thresholds=buckets,
        )
    threshold, median, sigma, quantile_value = fit_level(arr, cfg)
    bucket_map: Dict[int, float] = {}
    if bucket_ids is not None:
        bucket_map = fit_bucket_thresholds(
            arr, np.asarray(bucket_ids), cfg, min_bucket_samples
        )
    return StationThreshold(
        station_id=station_id,
        threshold=threshold,
        median=median,
        sigma_robust=sigma,
        quantile_value=quantile_value,
        n_samples=int(arr.size),
        from_network_fallback=False,
        bucket_thresholds=bucket_map,
    )


def fit_all(
    station_errors: Dict[str, np.ndarray],
    config: ThresholdConfig | None = None,
    station_buckets: Optional[Dict[str, np.ndarray]] = None,
    min_bucket_samples: int = 50,
) -> Dict[str, StationThreshold]:
    """Calibrate a whole network: pooled fallback first, then each station.

    The pooled network level comes from all stations' clean errors together,
    so a new station with little history still gets a sane boundary.
    """
    cfg = config or DEFAULT_CONFIG.threshold
    pools = [_checked_errors(e) for e in station_errors.values()]
    if not pools:
        raise ValueError("cannot fit network threshold on zero stations")
    pooled = np.concatenate(pools)
    network_threshold, network_median, network_sigma, _ = fit_level(pooled, cfg)
    buckets = station_buckets or {}
    return {
        station_id: fit_station(
            station_id,
            errors,
            network_threshold,
            network_median,
            network_sigma,
            cfg,
            buckets.get(station_id),
            min_bucket_samples,
        )
        for station_id, errors in station_errors.items()
    }


def error_to_probability(
    error: np.ndarray | float,
    threshold: float,
    sigma_robust: float,
    config: ThresholdConfig | None = None,
) -> np.ndarray | float:
    """Map reconstruction error to P(anomaly) via a logistic anchored at threshold.

    At the threshold the probability is exactly 0.5; distance is measured in
    robust sigmas so the steepness means the same thing on every station.
    Downstream layers receive this calibrated probability, never a bare
    distance.
    """
    cfg = config or DEFAULT_CONFIG.threshold
    scale = max(float(sigma_robust), _MIN_SIGMA)
    z = (np.asarray(error, dtype=np.float64) - float(threshold)) / scale
    with np.errstate(over="ignore"):  # far tails saturate to 0/1, correctly
        prob = 1.0 / (1.0 + np.exp(-cfg.logistic_scale * z))
    if np.ndim(prob) == 0:
        return float(prob)
    return prob


def score(
    calibrated: StationThreshold,
    errors: np.ndarray | float,
    bucket_ids: np.ndarray | int | None = None,
    config: ThresholdConfig | None = None,
) -> Tuple[np.ndarray | float, np.ndarray | float]:
    """Score errors against a calibration; returns (probability, threshold_used).

    Per-observation bucket ids route through bucket thresholds where they
    exist, otherwise the station threshold. Thresholds travel alongside so the
    audit record shows which boundary produced the verdict.
    """
    cfg = config or DEFAULT_CONFIG.threshold
    single = np.ndim(errors) == 0 and bucket_ids is None
    err = np.atleast_1d(np.asarray(errors, dtype=np.float64))
    if bucket_ids is None:
        used = np.full(err.shape, calibrated.threshold)
    else:
        bids = np.atleast_1d(np.asarray(bucket_ids)).ravel()
        if bids.shape != err.shape:
            raise ValueError("bucket_ids must match errors elementwise")
        used = np.array([calibrated.threshold_for(int(b)) for b in bids])
    prob: np.ndarray
    # Vectorised per-element logistic with each element's own threshold.
    scale = max(calibrated.sigma_robust, _MIN_SIGMA)
    z = (err - used) / scale
    with np.errstate(over="ignore"):  # far tails saturate to 0/1, correctly
        prob = 1.0 / (1.0 + np.exp(-cfg.logistic_scale * z))
    if single:
        return float(prob[0]), float(used[0])
    return prob, used
