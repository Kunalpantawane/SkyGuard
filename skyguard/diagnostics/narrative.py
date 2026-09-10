"""Layer 6c: operator-facing narrative: verdict, severity, action.

Why a separate module: the forest says "spike, 94 %", Shapley says which
fingerprint features voted: but neither tells an operator what to *do*.
This module turns (fault class, confidence, ranked contributions, observed
values) into a DiagnosisResult with a severity and a recommended action.
Explainability a meteorologist cannot read is not explainability.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from ..config import DEFAULT_CONFIG, ShapleyConfig
from ..types import DiagnosisResult, EvidenceLine, FaultClass, Severity

# One recommended action per fault class. Concrete first ("inspect wiring"),
# never a shrug: an alert without an action is just anxiety.
RECOMMENDED_ACTIONS: Dict[FaultClass, str] = {
    FaultClass.NONE: "No action required. Observation accepted.",
    FaultClass.SPIKE: "Inspect temperature sensor and wiring for intermittent contact. Mark observation suspect.",
    FaultClass.DROP: "Inspect sensor for dropout or power dip. Mark observation suspect.",
    FaultClass.STUCK: "Inspect sensor for stuck/frozen output; check ventilation and power. Schedule maintenance visit.",
    FaultClass.DRIFT: "Sensor shows growing bias: schedule recalibration before it becomes a hard fault.",
    FaultClass.STEP: "Permanent level shift detected: check for sensor replacement or repositioning event, then recalibrate.",
    FaultClass.NOISE: "Abnormal signal variance: check shielding, grounding and power stability.",
    FaultClass.MISSING: "Communication outage: check telemetry link, power and logger status.",
    FaultClass.CORRUPT: "Corrupt encoding received: check telemetry chain for truncation or byte errors.",
    FaultClass.MULTIVARIATE: "Variables jointly inconsistent: cross-check all three sensors; likely one has drifted.",
    FaultClass.SPATIAL: "Station disagrees with neighbours: verify on-site conditions, then inspect sensors.",
    FaultClass.GENUINE_EXTREME: "Real extreme weather confirmed by context. Trust the observation; no maintenance action.",
}


def severity_for(fault_class: FaultClass, confidence: float) -> Severity:
    """Confidence-gated severity. NONE is never "a little severe": a clean
    verdict carries no severity at all, so it cannot be averaged or escalated
    by accident."""
    if fault_class == FaultClass.NONE:
        return Severity.NONE
    if confidence >= 0.9:
        return Severity.CRITICAL
    if confidence >= 0.7:
        return Severity.HIGH
    if confidence >= 0.5:
        return Severity.MODERATE
    return Severity.LOW


def diagnose(
    fault_class: FaultClass,
    confidence: float,
    probabilities: Dict[str, float],
    contributions: Sequence[Tuple[str, float]],
    observed: Dict[str, float],
    config: ShapleyConfig | None = None,
) -> DiagnosisResult:
    """Build the operator-facing diagnosis from classifier outputs.

    `contributions` are (label, value) pairs such as Shapley-ranked features;
    the top entries become evidence lines with the observed values attached.
    """
    cfg = config or DEFAULT_CONFIG.shapley
    ranked = sorted(contributions, key=lambda pair: -abs(pair[1]))[: cfg.top_k]
    total = sum(abs(value) for _, value in ranked)
    evidence: List[EvidenceLine] = []
    for label, value in ranked:
        share = abs(value) / total if total > 0 else 0.0
        reading = ", ".join(f"{var}={observed[var]:.2f}" for var in observed)
        evidence.append(EvidenceLine(
            label=label,
            contribution=float(share),
            detail=f"push {value:+.3f} toward {fault_class.value} | observed: {reading}",
        ))
    return DiagnosisResult(
        fault_class=fault_class,
        confidence=float(min(max(confidence, 0.0), 1.0)),
        severity=severity_for(fault_class, confidence),
        probabilities=dict(probabilities),
        evidence=evidence,
        recommended_action=RECOMMENDED_ACTIONS.get(
            fault_class, "Review observation manually."
        ),
    )
