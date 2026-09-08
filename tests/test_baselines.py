"""Baseline tests: every detector must rank a spike above clean air.

Bounds, determinism and input validation for each ladder row. Small fixed
data throughout — these are unit checks on the machinery, while the harness
test in `test_eval.py` runs the full injected-data comparison.
"""

import numpy as np
import pytest

from skyguard.baselines.iforest import IsolationForest
from skyguard.baselines.lof import LocalOutlierFactor
from skyguard.baselines.plain_ae import PlainAutoencoder
from skyguard.baselines.zscore import RobustZScore


def clean(n: int = 200, seed: int = 301):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    temp = 28.0 + 5.0 * np.sin(2 * np.pi * t / 24.0) + 0.2 * rng.standard_normal(n)
    rh = 70.0 - 2.0 * (temp - 28.0) + 1.0 * rng.standard_normal(n)
    pressure = 1005.0 + 0.15 * rng.standard_normal(n)
    return np.stack([temp, pressure, rh], axis=1)


def spike_point():
    return np.array([[45.0, 1005.0, 60.0]])


def ordered(detector, data):
    """Spike must outscore typical clean points, scores bounded in [0, 1]."""
    scores = detector.score_points(np.concatenate([data[:50], spike_point()]))
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)
    assert float(scores[-1]) > float(np.median(scores[:-1]))


def test_zscore_orders_and_validates():
    data = clean()
    detector = RobustZScore().fit(data)
    ordered(detector, data)
    with pytest.raises(ValueError):
        RobustZScore().fit(np.zeros((10, 2)))


def test_iforest_orders_and_is_deterministic():
    data = clean()
    first = IsolationForest(n_trees=20, seed=3).fit(data)
    second = IsolationForest(n_trees=20, seed=3).fit(data)
    ordered(first, data)
    assert np.array_equal(first.score_points(data[:10]), second.score_points(data[:10]))
    with pytest.raises(ValueError):
        IsolationForest().score_points(data[:5])


def test_lof_orders_and_validates():
    data = clean()
    detector = LocalOutlierFactor(k=10).fit(data)
    ordered(detector, data)
    with pytest.raises(ValueError):
        LocalOutlierFactor(k=500).fit(data[:50])
    with pytest.raises(ValueError):
        LocalOutlierFactor().score_points(data[:5])


def test_plain_ae_orders_and_validates():
    data = clean()
    detector = PlainAutoencoder(hidden=8, epochs=200, seed=4).fit(data)
    ordered(detector, data)
    with pytest.raises(ValueError):
        PlainAutoencoder().fit(np.zeros((4, 3)))
    with pytest.raises(ValueError):
        PlainAutoencoder().score_points(data[:5])
