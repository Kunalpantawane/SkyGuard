"""Self-healing estimates: quarantine first, estimate second, never overwrite.

Why this module is deliberately narrow: it blends three independent estimates
(temporal persistence, spatial neighbours, autoencoder reconstruction) into one
value with a confidence — it does *not* produce those estimates (the pipeline
does) and it never touches raw data. Corrections travel alongside the original
in the audit record; a refused estimate (long outage, low confidence) is an
honest gap, and an honest gap beats confident fiction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

from ..config import DEFAULT_CONFIG, CorrectionConfig

# The three estimators this blender knows. Anything else is a caller bug, and
# failing loud beats silently dropping a source's weight on the floor.
KNOWN_SOURCES = ("temporal", "spatial", "reconstruction")


@dataclass(frozen=True)
class SourceEstimate:
    """One estimator's opinion: a value, how sure it is, and how it got it."""

    value: Optional[float]
    confidence: float = 0.0
    method: str = ""


@dataclass(frozen=True)
class ImputedValue:
    """The blended estimate for one variable."""

    available: bool
    value: Optional[float] = None
    confidence: float = 0.0
    method: str = "none"


def impute(
    estimates: Dict[str, SourceEstimate],
    n_consecutive_bad: int,
    config: CorrectionConfig | None = None,
) -> ImputedValue:
    """Blend source estimates with the configured trust weights.

    Weights renormalise over the sources that actually spoke; confidence is
    the weighted mean of source confidences and must clear `min_confidence`.
    Beyond `max_consecutive_points` the answer is refusal: extrapolating a
    long outage produces fiction, and fiction with a confidence score is worse
    than a gap.
    """
    cfg = config or DEFAULT_CONFIG.correction
    if n_consecutive_bad > cfg.max_consecutive_points:
        return ImputedValue(
            available=False,
            method=f"refused: {n_consecutive_bad} consecutive bad points "
                   f"exceeds {cfg.max_consecutive_points}",
        )
    unknown = set(estimates) - set(KNOWN_SOURCES)
    if unknown:
        raise ValueError(f"unknown estimate sources: {sorted(unknown)}")
    weights = {
        "temporal": cfg.weight_temporal,
        "spatial": cfg.weight_spatial,
        "reconstruction": cfg.weight_reconstruction,
    }
    usable: Dict[str, float] = {}
    usable_conf: Dict[str, float] = {}
    for name in KNOWN_SOURCES:
        est = estimates.get(name)
        if (
            est is None or est.value is None or not math.isfinite(est.value)
            or est.confidence <= 0.0
        ):
            continue
        usable[name] = float(est.value)
        usable_conf[name] = float(min(max(est.confidence, 0.0), 1.0))
    if not usable:
        return ImputedValue(available=False, method="none: no usable source")
    total = sum(weights[name] for name in usable)
    value = sum(usable[name] * weights[name] for name in usable) / total
    confidence = sum(usable_conf[name] * weights[name] for name in usable) / total
    if confidence < cfg.min_confidence:
        return ImputedValue(
            available=False, confidence=float(confidence),
            method=f"refused: confidence {confidence:.2f} below {cfg.min_confidence}",
        )
    return ImputedValue(
        available=True, value=float(value), confidence=float(confidence),
        method="+".join(sorted(usable)),
    )
