"""Layer 3 tests.

The flagship case is the subtle one: a hot-but-plausible temperature paired
with a humid-but-plausible RH must score worse than a consistent observation —
that is the whole reason this layer exists. Plus dew-point wiring, abstention
on partial state, and the thin-history guard.
"""

import numpy as np
import pytest

from skyguard.config import VARIABLES, MultivariateConfig
from skyguard.qc.multivariate import MultivariateQC, dewpoint_c


def clean_history(seed: int, n: int = 300):
    """T/RH anticorrelated diurnal state, the structure Layer 3 must learn."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    temp = 28.0 + 5.0 * np.sin(2 * np.pi * t / 24.0) + 0.2 * rng.standard_normal(n)
    rh = 70.0 - 2.0 * (temp - 28.0) + 1.0 * rng.standard_normal(n)
    pressure = 1005.0 + 3.0 * np.sin(2 * np.pi * t / 120.0) + 0.15 * rng.standard_normal(n)
    return np.stack([temp, pressure, rh], axis=1)


def fitted(seed: int = 5) -> MultivariateQC:
    qc = MultivariateQC()
    assert qc.fit_station("S1", clean_history(seed)) > 0
    return qc


# --------------------------------------------------------------------------
# Physics helper
# --------------------------------------------------------------------------

def test_dewpoint_known_value():
    # 20 C at 50 % RH -> dew point ~9.3 C (Magnus relation).
    assert dewpoint_c(20.0, 50.0) == pytest.approx(9.3, abs=0.3)
    assert dewpoint_c(25.0, 100.0) == pytest.approx(25.0, abs=0.2)


# --------------------------------------------------------------------------
# Joint consistency
# --------------------------------------------------------------------------

def test_consistent_observation_scores_low():
    qc = fitted()
    result = qc.evaluate("S1", 29.0, 1005.5, 68.0)
    assert result.available
    assert result.probability < 0.5
    assert not result.physical_violation
    assert result.dewpoint_c < 29.0
    assert set(result.per_variable) == set(VARIABLES)
    assert "Mahalanobis" in result.detail


def test_subtle_joint_fault_scores_high():
    """Hot AND humid: each plausible alone, jointly wrong for this station."""
    qc = fitted()
    calm = qc.evaluate("S1", 29.0, 1005.5, 68.0)
    # Fresh state so the delta term cannot leak between the two verdicts.
    qc.reset("S1")
    qc.evaluate("S1", 29.0, 1005.5, 68.0)  # re-seed streaming delta
    suspect = qc.evaluate("S1", 32.5, 1004.0, 82.0)
    assert suspect.available
    assert suspect.probability > 0.5
    assert suspect.probability > calm.probability + 0.2
    # Attribution must blame the T/RH pair, not pressure.
    assert suspect.per_variable["temp_c"] + suspect.per_variable["rh_pct"] > 0.6
    assert abs(sum(suspect.per_variable.values()) - 1.0) < 1e-9


def test_physical_violation_wiring():
    """A negative tolerance forces the dew-point path to prove its wiring."""
    qc = MultivariateQC(config=MultivariateConfig(dewpoint_excess_tolerance_c=-2.0))
    assert qc.fit_station("S1", clean_history(6)) > 0
    result = qc.evaluate("S1", 25.0, 1005.0, 100.0)
    assert result.physical_violation
    assert result.probability >= 0.9
    assert "VIOLATION" in result.detail


# --------------------------------------------------------------------------
# Abstention, not guessing
# --------------------------------------------------------------------------

def test_partial_or_unknown_state_is_unavailable():
    qc = fitted()
    assert not qc.evaluate("S1", None, 1005.0, 60.0).available
    assert not qc.evaluate("UNKNOWN", 29.0, 1005.0, 60.0).available


def test_thin_history_refuses_to_fit():
    qc = MultivariateQC()
    assert qc.fit_station("S1", clean_history(7, n=50)) == 0
    assert not qc.is_fitted("S1")
    assert not qc.evaluate("S1", 29.0, 1005.0, 60.0).available


def test_nonfinite_history_raises():
    qc = MultivariateQC()
    bad = clean_history(8)
    bad[10, 0] = np.nan
    with pytest.raises(ValueError):
        qc.fit_station("S1", bad)
    with pytest.raises(ValueError):
        qc.fit_station("S1", np.zeros((200, 2)))
