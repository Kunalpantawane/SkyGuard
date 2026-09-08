"""Thresholding tests.

Each behaviour gets a hand-built fixture: Gaussian errors must follow K-sigma,
a heavy tail must trip the quantile fallback, thin stations must borrow the
network level, thin buckets must abstain. Plus the chain test that matters —
real LSTM errors through fit into probabilities.
"""

import numpy as np
import pytest

from skyguard.config import ModelConfig, ThresholdConfig
from skyguard.model.lstm import LstmAutoencoder, make_model_input
from skyguard.model.threshold import (
    StationThreshold,
    error_to_probability,
    fit_all,
    fit_bucket_thresholds,
    fit_level,
    fit_station,
    score,
)


def gauss_errors(seed: int, n: int = 2000, loc: float = 0.5, scale: float = 0.1):
    rng = np.random.default_rng(seed)
    return np.abs(rng.normal(loc, scale, size=n))


# --------------------------------------------------------------------------
# Core fit
# --------------------------------------------------------------------------

def test_gaussian_fit_follows_k_sigma():
    cfg = ThresholdConfig()
    err = gauss_errors(11)
    threshold, median, sigma, quantile_value = fit_level(err, cfg)
    # Near-Gaussian: K-sigma sits well above the 0.995 quantile, so it rules.
    assert threshold == pytest.approx(median + cfg.k_sigma * sigma)
    assert threshold > quantile_value
    assert abs(median - 0.5) < 0.02
    assert abs(sigma - 0.1) < 0.02


def test_heavy_tail_trips_quantile_fallback():
    """99 % zeros + 1 % huge: MAD sees nothing, the quantile must catch it."""
    err = np.array([0.0] * 990 + [10.0] * 10)
    threshold, _, _, quantile_value = fit_level(err)
    assert quantile_value == pytest.approx(10.0)
    assert threshold == pytest.approx(10.0)


def test_empty_and_nonfinite_raise():
    with pytest.raises(ValueError):
        fit_level(np.array([]))
    with pytest.raises(ValueError):
        fit_level(np.array([0.1, 0.2, np.nan]))
    with pytest.raises(ValueError):
        fit_all({})


# --------------------------------------------------------------------------
# Probability map
# --------------------------------------------------------------------------

def test_probability_anchored_and_monotone():
    cfg = ThresholdConfig()
    assert error_to_probability(1.0, 1.0, 0.1, cfg) == pytest.approx(0.5)
    assert error_to_probability(0.5, 1.0, 0.1, cfg) < 0.05
    assert error_to_probability(2.0, 1.0, 0.1, cfg) > 0.95
    grid = np.linspace(0.0, 3.0, 31)
    probs = error_to_probability(grid, 1.0, 0.1, cfg)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    # Non-decreasing everywhere; strictly rising through the transition band
    # (far tails saturate to exactly 0/1 in float64, which is correct).
    assert np.all(np.diff(probs) >= 0.0)
    mid = probs[(grid > 0.8) & (grid < 1.2)]
    assert np.all(np.diff(mid) > 0.0)


def test_zero_sigma_degrades_to_step():
    """A constant calibration set must not divide by zero or crash."""
    assert error_to_probability(1.5, 1.0, 0.0) == pytest.approx(1.0)
    assert error_to_probability(0.5, 1.0, 0.0) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Network fallback
# --------------------------------------------------------------------------

def test_thin_station_borrows_network_level():
    cfg = ThresholdConfig()
    err = {
        "RICH": gauss_errors(21, n=500),
        "NEW": gauss_errors(22, n=20),
    }
    fitted = fit_all(err, cfg)
    assert not fitted["RICH"].from_network_fallback
    assert fitted["NEW"].from_network_fallback
    pooled = np.concatenate([err["RICH"], err["NEW"]])
    network_threshold, _, _, _ = fit_level(pooled, cfg)
    assert fitted["NEW"].threshold == pytest.approx(network_threshold)
    assert fitted["NEW"].n_samples == 20


# --------------------------------------------------------------------------
# Buckets
# --------------------------------------------------------------------------

def test_buckets_split_day_from_night():
    rng = np.random.default_rng(31)
    night = np.abs(rng.normal(0.3, 0.05, size=300))
    day = np.abs(rng.normal(0.6, 0.20, size=300))
    err = np.concatenate([night, day])
    bids = np.array([0] * 300 + [1] * 300)
    buckets = fit_bucket_thresholds(err, bids, min_bucket_samples=50)
    assert set(buckets) == {0, 1}
    assert buckets[1] > buckets[0]


def test_thin_bucket_abstains_to_station_threshold():
    rng = np.random.default_rng(32)
    err = np.concatenate([np.abs(rng.normal(0.5, 0.1, size=300)),
                          np.abs(rng.normal(0.5, 0.1, size=10))])
    bids = np.array([0] * 300 + [1] * 10)
    cal = fit_station("S1", err, network_threshold=99.0, bucket_ids=bids)
    assert not cal.from_network_fallback
    assert 1 not in cal.bucket_thresholds
    assert cal.threshold_for(1) == pytest.approx(cal.threshold)
    assert cal.threshold_for(0) == pytest.approx(cal.bucket_thresholds[0])


def test_score_uses_bucket_thresholds():
    cal = StationThreshold(
        station_id="S1", threshold=1.0, median=0.5, sigma_robust=0.1,
        quantile_value=0.9, n_samples=500, bucket_thresholds={1: 2.0},
    )
    prob, used = score(cal, np.array([1.5, 1.5]), np.array([0, 1]))
    assert used[0] == pytest.approx(1.0)
    assert used[1] == pytest.approx(2.0)
    # Same error, stricter boundary at bucket 0 -> higher probability.
    assert prob[0] > prob[1]
    single_p, single_t = score(cal, 1.5)
    assert single_t == pytest.approx(1.0)
    assert 0.0 <= single_p <= 1.0


def test_bucket_shape_mismatch_raises():
    with pytest.raises(ValueError):
        fit_bucket_thresholds(np.array([0.1, 0.2]), np.array([0]))


# --------------------------------------------------------------------------
# Chain: real LSTM errors -> fit -> probabilities
# --------------------------------------------------------------------------

def test_lstm_errors_through_threshold_flag_spike():
    cfg = ModelConfig(window=6, hidden_size=8, learning_rate=0.01,
                      batch_size=8, max_epochs=30, patience=8, seed=7)
    model = LstmAutoencoder(config=cfg)
    rng = np.random.default_rng(41)
    hours = np.tile(np.arange(6, dtype=np.float64), (320, 1))
    doys = np.full((320, 6), 180.0)
    phys = np.stack([
        np.sin(2 * np.pi * hours / 24.0),
        0.5 * np.cos(2 * np.pi * hours / 24.0),
        -0.7 * np.sin(2 * np.pi * hours / 24.0),
    ], axis=-1) + 0.05 * rng.standard_normal((320, 6, 3))
    xb = make_model_input(phys, hours, doys)
    train_x, val_x, cal_x = xb[:48], xb[48:64], xb[64:]
    model.fit(train_x, train_x[:, :, :3], val_x, val_x[:, :, :3])
    _, cal_err, _ = model.reconstruct_batch(cal_x)  # 256 clean errors

    cal = fit_station("S1", cal_err, network_threshold=float(cal_err.max()))
    assert not cal.from_network_fallback  # 256 samples clears the 200 floor

    spiked = cal_x[:16].copy()
    spiked[:, 3, 0] += 6.0
    _, spike_err, _ = model.reconstruct_batch(spiked)
    spike_p, _ = score(cal, spike_err)
    clean_p, _ = score(cal, np.full_like(spike_err, np.median(cal_err)))
    assert float(np.median(spike_p)) > 0.6 > float(np.median(clean_p))
