"""Event aggregation: one operator ticket per fault, not per point.

Why grouping matters: a six-hour drift is one maintenance ticket, not 72
alarms. Point-level output alone makes a dashboard unusable and trains
operators to ignore it. Consecutive anomalous points merge into an event,
tolerating a short flicker of normal points inside (a sputtering sensor is
still one fault), and the event carries the peak evidence, the majority fault
class and the union of affected sensors.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Sequence

from ..config import DEFAULT_CONFIG, EventConfig
from ..diagnostics.narrative import severity_for
from ..types import AnomalyEvent, FaultClass


@dataclass(frozen=True)
class EventPoint:
    """One fused verdict, the aggregator's input grain."""

    timestamp: datetime
    is_anomaly: bool
    fault_class: FaultClass = FaultClass.NONE
    confidence: float = 0.0
    probability: float = 0.0
    affected_variables: List[str] = field(default_factory=list)


def aggregate(
    station_id: str, points: Sequence[EventPoint], config: EventConfig | None = None
) -> List[AnomalyEvent]:
    """Group a station's point verdicts into anomaly events.

    Points must be in time order; out-of-order input raises rather than
    silently producing scrambled events.
    """
    cfg = config or DEFAULT_CONFIG.events
    ordered = list(points)
    for prev, curr in zip(ordered, ordered[1:]):
        if curr.timestamp < prev.timestamp:
            raise ValueError("event points must be in time order")

    events: List[AnomalyEvent] = []
    run: List[EventPoint] = []
    gap = 0
    counter = 0

    def flush() -> None:
        nonlocal run, gap, counter
        anomalous = [p for p in run if p.is_anomaly]
        if len(anomalous) >= cfg.min_event_points:
            counter += 1
            # Trailing tolerated normals pad the run but are not the event:
            # the span ends at the last anomalous point.
            span = run if gap == 0 else run[:-gap]
            events.append(_build_event(station_id, counter, span, anomalous))
        run = []
        gap = 0

    for point in ordered:
        if point.is_anomaly:
            run.append(point)
            gap = 0
        elif run:
            gap += 1
            run.append(point)
            if gap > cfg.max_gap_points:
                run = run[: -gap]  # trailing normals belong to no event
                gap = 0
                flush()
    flush()
    return events


def _build_event(
    station_id: str, number: int, span: List[EventPoint], anomalous: List[EventPoint]
) -> AnomalyEvent:
    """Summarise one span: majority class, peak evidence, sensor union."""
    classes = [p.fault_class for p in anomalous if p.fault_class != FaultClass.NONE]
    fault = Counter(classes).most_common(1)[0][0] if classes else FaultClass.NONE
    confidence = max(p.confidence for p in anomalous)
    peak = max(p.probability for p in anomalous)
    affected = sorted({var for p in anomalous for var in p.affected_variables})
    return AnomalyEvent(
        event_id=f"{station_id}-E{number:04d}",
        station_id=station_id,
        start=span[0].timestamp,
        end=span[-1].timestamp,
        n_points=len(anomalous),
        fault_class=fault,
        severity=severity_for(fault, confidence),
        confidence=float(confidence),
        peak_probability=float(peak),
        affected_variables=affected,
        explanation=f"{fault.value} over {len(anomalous)} anomalous points "
                    f"(peak p={peak:.2f}) affecting {', '.join(affected) or 'no sensor'}",
    )
