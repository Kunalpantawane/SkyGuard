"""Layer 5 — anomaly fusion: one calibrated verdict from independent evidence.

Why fusion instead of letting any layer decide: every signal has a blind spot.
Rules cannot see joint states, the autoencoder cannot tell a heatwave from a
stuck heater, spatial QC goes quiet without neighbours. Any one layer firing
is evidence, not a verdict — so the engine takes a trust-weighted mean over
whichever signals spoke, renormalising when a layer abstains.

Three safeguards ride on top of the mean:

1. **Hard-rule floor.** A physically impossible value (sentinel, RH = 150 %)
   short-circuits debate: no amount of model agreement makes it acceptable.
2. **Genuine-weather guard.** When neighbours move with the station *and* the
   covariates stay coherent, confidence is damped — weakened, not erased.
3. **Bands, not binaries.** Operators get a calibrated probability plus a band
   (NORMAL … CRITICAL); the binary flag is a documented threshold on top.
"""

from __future__ import annotations

from typing import Dict, Optional

from ..config import DEFAULT_CONFIG, VARIABLES, FusionConfig
from ..types import (
    ConfidenceBand,
    FusionResult,
    MultivariateResult,
    ReconResult,
    RuleResult,
    SpatialResult,
)


class FusionEngine:
    """Trust-weighted evidence fusion with guards."""

    def __init__(self, config: FusionConfig | None = None) -> None:
        self.config: FusionConfig = config or DEFAULT_CONFIG.fusion

    def fuse(
        self,
        rules: RuleResult,
        recon: ReconResult,
        multivariate: MultivariateResult,
        spatial: SpatialResult,
    ) -> FusionResult:
        """Combine one observation's layer outputs into a single verdict."""
        cfg = self.config
        evidence: Dict[str, float] = {
            "rule": _clamp(rules.score),
            "persistence": _clamp(_persistence_score(rules)),
        }
        if recon.available:
            evidence["reconstruction"] = _clamp(recon.probability)
        if multivariate.available:
            evidence["multivariate"] = _clamp(multivariate.probability)
        if spatial.available:
            evidence["spatial"] = _clamp(spatial.probability)

        total_weight = sum(cfg.weights[name] for name in evidence)
        probability = sum(evidence[name] * cfg.weights[name] for name in evidence) / total_weight

        if rules.has_hard_violation:
            probability = max(probability, cfg.hard_rule_floor)

        damped = False
        if (
            "reconstruction" in evidence
            and evidence["reconstruction"] >= cfg.anomaly_decision_threshold
            and spatial.available
            and multivariate.available
            and spatial.probability < cfg.spatial_agreement_threshold
            and multivariate.probability < cfg.covariate_coherence_threshold
        ):
            # An alarming temporal signal contradicted by context: neighbours
            # agree and covariates stay coherent, so the change smells like
            # weather the temporal model has simply not seen before. Damp the
            # confidence rather than erasing the evidence. Quiet records skip
            # this entirely — damping them would only noise the audit trail.
            probability *= 1.0 - cfg.genuine_weather_damping
            damped = True

        probability = _clamp(probability)
        band = self._band(probability)
        return FusionResult(
            probability=probability,
            band=band,
            is_anomaly=probability >= cfg.anomaly_decision_threshold,
            evidence=evidence,
            genuine_weather_damped=damped,
            dominant_variable=_dominant_variable(rules, recon),
        )

    def _band(self, probability: float) -> ConfidenceBand:
        """First band whose upper edge exceeds the probability."""
        for upper, name in self.config.bands:
            if probability < upper:
                return ConfidenceBand[name]
        return ConfidenceBand.CRITICAL


def _clamp(value: float) -> float:
    """Keep every signal inside [0, 1] no matter what a layer produced."""
    return min(max(float(value), 0.0), 1.0)


def _persistence_score(rules: RuleResult) -> float:
    """Strongest stuck-sensor evidence, or silence when there is none."""
    scores = [flag.score for flag in rules.flags if flag.name == "persistence"]
    return max(scores) if scores else 0.0


def _dominant_variable(rules: RuleResult, recon: ReconResult) -> Optional[str]:
    """Which sensor the evidence blames most.

    Rule per-variable scores are bounded and literal, so they lead; the
    autoencoder's per-variable errors (unbounded MSE) join normalised to their
    sum. No blame anywhere means no dominant variable, not a forced guess.
    """
    combined = {var: _clamp(rules.per_variable.get(var, 0.0)) for var in VARIABLES}
    if recon.available and recon.per_variable:
        total = sum(max(recon.per_variable.get(var, 0.0), 0.0) for var in VARIABLES)
        if total > 0.0:
            for var in VARIABLES:
                combined[var] += max(recon.per_variable.get(var, 0.0), 0.0) / total
    if all(score <= 0.0 for score in combined.values()):
        return None
    return max(VARIABLES, key=lambda var: combined[var])
