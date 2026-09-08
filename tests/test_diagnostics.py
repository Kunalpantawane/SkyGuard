"""Layer 6 tests.

The forest must separate fault fingerprints, stay deterministic, and reject
bad input. Shapley must account completely (efficiency) and point at the
right evidence. The narrative must turn a verdict into severity + action.
"""

import numpy as np
import pytest

from skyguard.config import ForestConfig, ShapleyConfig
from skyguard.diagnostics.forest import (
    FAULT_ORDER,
    N_FEATURES,
    FaultClassifier,
    RandomForestClassifier,
)
from skyguard.diagnostics.narrative import RECOMMENDED_ACTIONS, diagnose, severity_for
from skyguard.diagnostics.shapley import rank_contributions, shapley_values
from skyguard.types import FaultClass, Severity


def small_forest(**kw) -> ForestConfig:
    base = {"n_trees": 10, "max_depth": 4, "min_samples_split": 4,
            "min_samples_leaf": 2, "max_features": "sqrt", "seed": 11,
            "class_balance": True}
    base.update(kw)
    return ForestConfig(**base)


def fingerprints(seed: int, per_class: int = 30):
    """Three well-separated fault fingerprints on the 14-column contract."""
    rng = np.random.default_rng(seed)
    none = np.abs(rng.normal(0.0, 0.2, size=(per_class, N_FEATURES)))
    spike = np.abs(rng.normal(0.0, 0.2, size=(per_class, N_FEATURES)))
    spike[:, 0] += 3.0   # recon_temp
    spike[:, 3] += 2.5   # rate_of_change
    stuck = np.abs(rng.normal(0.0, 0.2, size=(per_class, N_FEATURES)))
    stuck[:, 4] += 4.0   # repeat_run
    X = np.concatenate([none, spike, stuck])
    y = np.array([0] * per_class + [1] * per_class + [2] * per_class)
    return X, y


# --------------------------------------------------------------------------
# Forest engine
# --------------------------------------------------------------------------

def test_forest_separates_fault_fingerprints():
    forest = RandomForestClassifier(n_classes=3, config=small_forest())
    X, y = fingerprints(101)
    forest.fit(X, y)
    held_X, held_y = fingerprints(102)
    pred = forest.predict(held_X)
    assert float(np.mean(pred == held_y)) >= 0.9
    proba = forest.predict_proba(held_X)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_forest_is_deterministic_given_seed():
    X, y = fingerprints(103)
    first = RandomForestClassifier(n_classes=3, config=small_forest()).fit(X, y)
    second = RandomForestClassifier(n_classes=3, config=small_forest()).fit(X, y)
    assert np.array_equal(first.predict(X), second.predict(X))


def test_forest_rejects_bad_input():
    forest = RandomForestClassifier(n_classes=3, config=small_forest())
    X, y = fingerprints(104)
    bad = X.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        forest.fit(bad, y)
    with pytest.raises(ValueError):
        forest.predict_proba(X)
    forest.fit(X, y)
    with pytest.raises(ValueError):
        forest.predict_proba(X[:, :5])


# --------------------------------------------------------------------------
# Fault taxonomy wrapper
# --------------------------------------------------------------------------

def test_fault_classifier_maps_taxonomy():
    clf = FaultClassifier(config=small_forest())
    X, _ = fingerprints(105)
    labels = [FaultClass.NONE] * 30 + [FaultClass.SPIKE] * 30 + [FaultClass.STUCK] * 30
    clf.fit(X, labels)
    spike_print = np.zeros(N_FEATURES)
    spike_print[0] = 3.0
    spike_print[3] = 2.5
    result = clf.predict_one(spike_print)
    assert result.fault_class == FaultClass.SPIKE
    assert 0.0 <= result.confidence <= 1.0
    assert set(result.probabilities) == {fault.value for fault in FAULT_ORDER}
    with pytest.raises(ValueError):
        clf.fit(X[:, :5], labels)


# --------------------------------------------------------------------------
# Shapley
# --------------------------------------------------------------------------

def linear_model(weights):
    def score(X):
        return np.asarray(X, dtype=np.float64) @ np.asarray(weights, dtype=np.float64)
    return score


def test_exact_shapley_is_complete_and_points_right():
    rng = np.random.default_rng(201)
    background = rng.normal(0.0, 1.0, size=(60, 3))
    point = np.array([2.0, 0.5, -1.0])
    weights = np.array([2.0, -1.0, 0.5])
    phi = shapley_values(point, background, linear_model(weights))
    expected_gap = float(weights @ point - weights @ background.mean(axis=0))
    assert float(phi.sum()) == pytest.approx(expected_gap, rel=1e-9)
    ranked = rank_contributions(["a", "b", "c"], phi)
    assert ranked[0][0] == "a"  # weight 2.0 on the largest deviation dominates


def test_sampled_shapley_mode_runs():
    rng = np.random.default_rng(202)
    background = rng.normal(0.0, 1.0, size=(60, 5))
    point = np.array([2.0, 0.5, -1.0, 0.0, 1.5])
    weights = np.array([2.0, -1.0, 0.5, 0.1, 1.0])
    cfg = ShapleyConfig(exact_max_features=2, n_samples=500, seed=5)
    phi = shapley_values(point, background, linear_model(weights), cfg)
    expected_gap = float(weights @ point - weights @ background.mean(axis=0))
    assert float(phi.sum()) == pytest.approx(expected_gap, rel=0.3)
    assert rank_contributions(["a", "b", "c", "d", "e"], phi)[0][0] == "a"


def test_shapley_rejects_mismatched_input():
    background = np.zeros((10, 3))
    with pytest.raises(ValueError):
        shapley_values(np.zeros(4), background, linear_model([1, 1, 1]))
    with pytest.raises(ValueError):
        shapley_values(np.zeros(3), np.zeros((0, 3)), linear_model([1, 1, 1]))


# --------------------------------------------------------------------------
# Narrative
# --------------------------------------------------------------------------

def test_diagnose_spike_is_actionable():
    result = diagnose(
        FaultClass.SPIKE, 0.94,
        {"spike": 0.94, "none": 0.06},
        [("recon_temp", 0.55), ("rate_of_change", 0.30), ("spatial_residual", 0.10)],
        {"temp_c": 48.0, "pressure_hpa": 1005.0, "rh_pct": 60.0},
    )
    assert result.severity == Severity.CRITICAL
    assert "wiring" in result.recommended_action
    assert len(result.evidence) <= 5
    assert result.evidence[0].label == "recon_temp"
    assert abs(sum(e.contribution for e in result.evidence) - 1.0) < 1e-9
    assert "48.00" in result.evidence[0].detail


def test_none_verdict_carries_no_severity():
    assert severity_for(FaultClass.NONE, 0.99) == Severity.NONE
    assert severity_for(FaultClass.DRIFT, 0.4) == Severity.LOW
    assert severity_for(FaultClass.DRIFT, 0.6) == Severity.MODERATE
    assert severity_for(FaultClass.DRIFT, 0.75) == Severity.HIGH
    result = diagnose(FaultClass.NONE, 0.99, {"none": 0.99}, [], {"temp_c": 28.0})
    assert result.evidence == []
    assert result.recommended_action == RECOMMENDED_ACTIONS[FaultClass.NONE]
