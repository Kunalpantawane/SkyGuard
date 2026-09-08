"""Layer 5 tests.

Fusion's contract in four cases: silence stays NORMAL, a hard rule convicts
alone, no single soft signal convicts alone, and the genuine-weather guard
damps (never erases) when neighbours and covariates agree.
"""

import pytest

from skyguard.fusion.engine import FusionEngine
from skyguard.types import (
    ConfidenceBand,
    MultivariateResult,
    QCFlag,
    ReconResult,
    RuleResult,
    SpatialResult,
)


def quiet_rules(score: float = 0.0, hard: bool = False) -> RuleResult:
    flags = [QCFlag("rate_of_change", "temp_c", score, hard, "test")] if score > 0 else []
    return RuleResult(
        flags=flags, score=score,
        per_variable={"temp_c": score, "pressure_hpa": 0.0, "rh_pct": 0.0},
        has_hard_violation=hard, missing={},
    )


def recon(prob: float) -> ReconResult:
    return ReconResult(
        available=True, error_total=prob, per_variable={"temp_c": prob, "pressure_hpa": 0.0, "rh_pct": 0.0},
        probability=prob, threshold=0.5, reconstruction={},
    )


def multi(prob: float) -> MultivariateResult:
    return MultivariateResult(available=True, mahalanobis=1.0, probability=prob, per_variable={})


def spatial(prob: float) -> SpatialResult:
    return SpatialResult(available=True, probability=prob)


def off() -> tuple:
    """All soft layers abstaining."""
    return (ReconResult(), MultivariateResult(), SpatialResult())


def test_all_quiet_is_normal():
    engine = FusionEngine()
    result = engine.fuse(quiet_rules(), recon(0.1), multi(0.15), spatial(0.1))
    assert result.probability < 0.3
    assert result.band == ConfidenceBand.NORMAL
    assert not result.is_anomaly
    assert not result.genuine_weather_damped
    assert result.dominant_variable == "temp_c"  # argmax over tiny background errors
    assert set(result.evidence) == {"rule", "persistence", "reconstruction", "multivariate", "spatial"}


def test_no_evidence_means_no_dominant_variable():
    engine = FusionEngine()
    rules = RuleResult(flags=[], score=0.0, per_variable={}, has_hard_violation=False, missing={})
    result = engine.fuse(rules, *off())
    assert result.dominant_variable is None


def test_hard_rule_convicts_alone():
    engine = FusionEngine()
    rules = quiet_rules(score=1.0, hard=True)
    result = engine.fuse(rules, *off())
    assert result.probability >= 0.95
    assert result.band == ConfidenceBand.CRITICAL
    assert result.is_anomaly
    assert result.dominant_variable == "temp_c"


def test_single_soft_signal_cannot_convict_alone():
    """A screaming autoencoder with every other layer quiet is WATCH at most."""
    engine = FusionEngine()
    result = engine.fuse(quiet_rules(), recon(0.95), *off()[1:])
    assert result.probability < 0.5
    assert not result.is_anomaly


def test_unanimous_soft_layers_convict_without_rules():
    """All three soft layers certain + rules quiet must still flag.

    Locks the threshold semantics: no single signal decides alone, but rules
    are not necessary for every detection (the injected multivar class lives
    exactly here). Ceiling of this combination under current weights is 0.543.
    """
    engine = FusionEngine()
    result = engine.fuse(quiet_rules(), recon(1.0), multi(1.0), spatial(1.0))
    assert result.is_anomaly
    assert result.probability == pytest.approx(2.2 / 4.05)


def test_agreeing_layers_convict():
    """A realistic spike trips rules, AE, joint state and neighbours together."""
    engine = FusionEngine()
    rules = quiet_rules(score=0.8)
    result = engine.fuse(rules, recon(0.9), multi(0.7), spatial(0.85))
    assert result.is_anomaly
    assert result.probability >= 0.6
    assert result.dominant_variable == "temp_c"


def test_genuine_weather_guard_damps_but_never_erases():
    engine = FusionEngine()
    rules = quiet_rules(score=0.2)
    guarded = engine.fuse(rules, recon(0.9), multi(0.2), spatial(0.15))
    exposed = engine.fuse(rules, recon(0.9), multi(0.2), spatial(0.9))
    assert guarded.genuine_weather_damped
    assert not exposed.genuine_weather_damped
    assert 0.0 < guarded.probability < exposed.probability


def test_soft_rule_maps_to_watch():
    engine = FusionEngine()
    result = engine.fuse(quiet_rules(score=0.7), *off())
    assert result.band == ConfidenceBand.WATCH
    assert not result.is_anomaly


def test_abstention_renormalises_over_speakers():
    """With every soft layer abstaining, the verdict is the rule score alone."""
    engine = FusionEngine()
    result = engine.fuse(quiet_rules(score=0.4), *off())
    assert set(result.evidence) == {"rule", "persistence"}
    assert result.probability == pytest.approx(0.4 * 1.0 / (1.0 + 0.85))
