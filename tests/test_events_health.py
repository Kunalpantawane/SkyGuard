"""Events and sensor-health tests.

Aggregation must group bursts, tolerate flicker, split distant bursts, and
never invent events from clean stretches. Health must fall as trouble rises,
stay UNKNOWN on thin history, and let the sickest sensor define the station.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from skyguard.events.aggregator import EventPoint, aggregate
from skyguard.health.sensor_health import (
    assess_station,
    bias_indicator,
    score_sensor,
    trend_indicator,
)
from skyguard.types import FaultClass, HealthStatus

T0 = datetime(2025, 6, 1, tzinfo=timezone.utc)


def point(i: int, anomaly: bool, fault: FaultClass = FaultClass.SPIKE,
          var: str = "temp_c") -> EventPoint:
    return EventPoint(
        timestamp=T0 + timedelta(hours=i), is_anomaly=anomaly, fault_class=fault,
        confidence=0.9 if anomaly else 0.1, probability=0.85 if anomaly else 0.1,
        affected_variables=[var] if anomaly else [],
    )


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def test_burst_with_flicker_is_one_event():
    series = [point(0, True), point(1, True), point(2, False),
              point(3, True), point(4, False), point(5, False)]
    events = aggregate("S1", series)
    assert len(events) == 1
    event = events[0]
    assert event.event_id == "S1-E0001"
    assert event.n_points == 3
    assert event.start == T0
    assert event.end == T0 + timedelta(hours=3)
    assert event.fault_class == FaultClass.SPIKE
    assert event.peak_probability == pytest.approx(0.85)
    assert event.affected_variables == ["temp_c"]
    assert event.explanation != ""
    assert "spike" in event.explanation


def test_distant_bursts_split_and_clean_stays_silent():
    series = ([point(i, True) for i in range(2)]
              + [point(i, False) for i in range(2, 8)]
              + [point(8, True, FaultClass.STUCK)])
    events = aggregate("S1", series)
    assert len(events) == 2
    assert [e.event_id for e in events] == ["S1-E0001", "S1-E0002"]
    assert events[1].fault_class == FaultClass.STUCK
    assert aggregate("S1", [point(i, False) for i in range(6)]) == []


def test_majority_class_and_sensor_union():
    series = [point(0, True, FaultClass.SPIKE, "temp_c"),
              point(1, True, FaultClass.SPIKE, "temp_c"),
              point(2, True, FaultClass.DRIFT, "pressure_hpa")]
    events = aggregate("S1", series)
    assert events[0].fault_class == FaultClass.SPIKE
    assert events[0].affected_variables == ["pressure_hpa", "temp_c"]


def test_unordered_points_raise():
    with pytest.raises(ValueError):
        aggregate("S1", [point(1, True), point(0, True)])


# --------------------------------------------------------------------------
# Health indicators
# --------------------------------------------------------------------------

def test_trend_and_bias_indicators():
    flat = np.full(100, 0.5) + np.random.default_rng(7).normal(0, 0.01, 100)
    assert trend_indicator(flat) < 0.2
    ramp = np.linspace(0.2, 1.0, 100)
    assert trend_indicator(ramp) > 0.8
    assert bias_indicator(np.random.default_rng(8).normal(0, 1, 200)) < 0.3
    assert bias_indicator(np.full(200, 2.0)) == pytest.approx(1.0)


def test_health_falls_as_trouble_rises():
    clean = score_sensor("temp_c", 500, 0.0, 0.0, 0.0, 0.0)
    assert clean.status == HealthStatus.HEALTHY
    assert clean.score == pytest.approx(100.0)
    sick = score_sensor("temp_c", 500, 0.5, 0.6, 0.4, 0.3)
    assert sick.score < 60.0
    assert sick.status in (HealthStatus.DEGRADING, HealthStatus.CRITICAL)
    assert len(sick.reasons) >= 2


def test_thin_history_is_unknown_not_healthy():
    result = score_sensor("temp_c", 10, 0.9, 0.9, 0.9, 0.9)
    assert result.status == HealthStatus.UNKNOWN


def test_sickest_sensor_defines_station():
    good = score_sensor("temp_c", 500, 0.0, 0.0, 0.0, 0.0)
    bad = score_sensor("rh_pct", 500, 0.6, 0.5, 0.3, 0.2)
    station = assess_station("S1", {"temp_c": good, "rh_pct": bad}, 500)
    assert station.overall_score == pytest.approx(bad.score)
    assert station.status == bad.status
    assert station.maintenance_risk in ("MEDIUM", "HIGH")
    assert len(bad.reasons) >= 2
    assert any("anomaly_rate" in reason for reason in bad.reasons)
    unknown = assess_station("S1", {}, 500)
    assert unknown.status == HealthStatus.UNKNOWN
