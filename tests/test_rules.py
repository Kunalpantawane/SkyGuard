"""Layer 1 tests.

Each check gets a hand-built fixture with the expected flag asserted, plus the
negative cases that matter most: real extreme weather and genuine RH saturation
must NOT be flagged. A QC layer that passes only its positive tests is the one
that floods operators with false alarms.
"""

from datetime import datetime, timedelta, timezone

import pytest

from skyguard.config import Config
from skyguard.qc.rules import RuleEngine
from skyguard.types import Observation

T0 = datetime(2025, 6, 1, tzinfo=timezone.utc)


def obs(i, station="S1", **kw):
    """A benign observation with a mild trend, so nothing trips by accident."""
    base = {
        "temp_c": 28.0 + 0.1 * i,
        "pressure_hpa": 1005.0 + 0.1 * i,
        "rh_pct": 60.0 - 0.1 * i,
    }
    base.update(kw)
    return Observation(station, T0 + timedelta(hours=i), **base)


def run(observations, interval_minutes=60, config=None):
    engine = RuleEngine(config=config, interval_minutes=interval_minutes)
    return [engine.evaluate(o) for o in observations]


def names(result):
    return {f.name for f in result.flags}


def flagged_vars(result, name):
    return {f.variable for f in result.flags if f.name == name}


# --------------------------------------------------------------------------
# Negative controls — the most important tests here
# --------------------------------------------------------------------------

def test_normal_series_raises_nothing():
    results = run([obs(i) for i in range(24)])
    assert all(not r.flags for r in results)
    assert all(r.score == 0.0 for r in results)


def test_record_extreme_weather_passes_range_check():
    """Range bounds must admit record-breaking but real weather.

    51 C at 8 % RH in a pre-monsoon heatwave is a real Indian observation. A
    tighter bound would 'catch' it and destroy the system's credibility.
    """
    results = run([obs(0), obs(1, temp_c=51.0, rh_pct=8.0), obs(2)])
    assert "range" not in names(results[1])


def test_genuine_rh_saturation_is_not_flagged_as_stuck():
    """Fog and rain hold RH at ~99 % for hours; that is weather, not a fault."""
    series = [
        Observation("S1", T0 + timedelta(hours=i), 28.0 + 0.1 * i, 1005.0, 99.0)
        for i in range(30)
    ]
    results = run(series)
    assert all("rh_pct" not in flagged_vars(r, "persistence") for r in results)


def test_slow_legitimate_warming_is_not_a_rate_violation():
    """A real 8 C rise over 8 hours must pass; only implausible speed is a fault."""
    series = [obs(i, temp_c=24.0 + i) for i in range(9)]
    results = run(series)
    assert all("rate_of_change" not in names(r) for r in results)


# --------------------------------------------------------------------------
# Corrupt encoding
# --------------------------------------------------------------------------

@pytest.mark.parametrize("sentinel", [999.0, -999.0, 9999.0, -9999.0])
def test_sentinel_values_flagged_hard(sentinel):
    results = run([obs(0), obs(1, temp_c=sentinel), obs(2)])
    assert "corrupt_encoding" in names(results[1])
    assert results[1].has_hard_violation
    assert results[1].score == pytest.approx(1.0)


def test_corrupt_value_does_not_poison_next_observation():
    """A sentinel must not become the rate-check baseline for the next reading."""
    results = run([obs(0), obs(1, temp_c=-9999.0), obs(2)])
    assert not results[2].flags


def test_sentinel_suppresses_redundant_range_flag():
    """One fault, one explanation. -9999 is corrupt, not merely out of range."""
    results = run([obs(0), obs(1, temp_c=-9999.0)])
    assert "range" not in names(results[1])


# --------------------------------------------------------------------------
# Range
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "field,value",
    [("rh_pct", 150.0), ("rh_pct", -5.0), ("temp_c", 95.0), ("pressure_hpa", 300.0)],
)
def test_impossible_values_flagged_hard(field, value):
    results = run([obs(0), obs(1, **{field: value})])
    assert "range" in names(results[1])
    assert results[1].has_hard_violation


def test_hard_range_violation_does_not_poison_next_observation():
    results = run([obs(0), obs(1, rh_pct=150.0), obs(2)])
    assert not results[2].flags


# --------------------------------------------------------------------------
# Rate of change
# --------------------------------------------------------------------------

def test_spike_trips_rate_check():
    results = run([obs(0), obs(1, temp_c=48.0)])
    assert "rate_of_change" in names(results[1])
    assert "temp_c" in flagged_vars(results[1], "rate_of_change")


def test_rate_check_skipped_after_long_gap():
    """After a long outage the weather has legitimately moved on."""
    series = [
        obs(0),
        Observation("S1", T0 + timedelta(hours=12), 40.0, 1005.0, 60.0),
    ]
    results = run(series)
    assert "rate_of_change" not in names(results[1])


def test_rate_limit_scales_with_sampling_interval():
    """The same per-hour config must work at 5-minute sampling.

    A 3 C jump in 5 minutes is implausible; the same 3 C over an hour is routine.
    """
    fast = [
        Observation("S1", T0, 28.0, 1005.0, 60.0),
        Observation("S1", T0 + timedelta(minutes=5), 31.5, 1005.0, 60.0),
    ]
    slow = [
        Observation("S1", T0, 28.0, 1005.0, 60.0),
        Observation("S1", T0 + timedelta(hours=1), 31.5, 1005.0, 60.0),
    ]
    assert "rate_of_change" in names(run(fast, interval_minutes=5)[1])
    assert "rate_of_change" not in names(run(slow)[1])


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def test_frozen_temperature_flagged():
    series = [
        Observation("S1", T0 + timedelta(hours=i), 27.4, 1005.0 + 0.1 * i, 60.0)
        for i in range(16)
    ]
    results = run(series)
    assert any("temp_c" in flagged_vars(r, "persistence") for r in results)


def test_persistence_respects_sensor_resolution_tolerance():
    """A 0.1 C sensor genuinely repeats values; tolerance must absorb that."""
    cfg = Config()
    series = [
        Observation("S1", T0 + timedelta(hours=i), 27.4 + (0.02 if i % 2 else 0.0),
                    1005.0 + 0.1 * i, 60.0)
        for i in range(16)
    ]
    results = run(series, config=cfg)
    # 0.02 C jitter is below the 0.05 C tolerance, so this still reads as frozen.
    assert any("temp_c" in flagged_vars(r, "persistence") for r in results)


def test_gap_breaks_the_persistence_run():
    """Values either side of an outage are not evidence of a stuck sensor."""
    series = [Observation("S1", T0 + timedelta(hours=i), 27.4, 1005.0, 60.0)
              for i in range(8)]
    series.append(Observation("S1", T0 + timedelta(hours=8), None, 1005.0, 60.0))
    series += [Observation("S1", T0 + timedelta(hours=9 + i), 27.4, 1005.0, 60.0)
               for i in range(8)]
    results = run(series)
    assert "temp_c" not in flagged_vars(results[-1], "persistence")


# --------------------------------------------------------------------------
# Missing and timestamps
# --------------------------------------------------------------------------

def test_missing_score_grows_with_run_length():
    series = [obs(0)] + [obs(i, temp_c=None) for i in range(1, 6)] + [obs(6)]
    results = run(series)
    scores = [
        max((f.score for f in r.flags if f.name == "missing"), default=0.0)
        for r in results[1:6]
    ]
    assert scores == sorted(scores)
    assert scores[0] < scores[-1]
    assert all(results[i].missing["temp_c"] for i in range(1, 6))


def test_missing_run_resets_after_recovery():
    series = [obs(0), obs(1, temp_c=None), obs(2), obs(3, temp_c=None)]
    results = run(series)
    first = max(f.score for f in results[1].flags if f.name == "missing")
    later = max(f.score for f in results[3].flags if f.name == "missing")
    assert first == pytest.approx(later)


def test_duplicate_timestamp_flagged():
    results = run([obs(0), obs(0)])
    assert "duplicate_timestamp" in names(results[1])


def test_timestamp_reversal_flagged():
    series = [
        obs(5),
        Observation("S1", T0, 28.0, 1005.0, 60.0),
    ]
    results = run(series)
    assert "timestamp_reversal" in names(results[1])


def test_reversal_does_not_corrupt_later_gap_checks():
    """A bad clock must not overwrite the last-good timestamp."""
    series = [obs(5), Observation("S1", T0, 28.0, 1005.0, 60.0), obs(6)]
    results = run(series)
    assert "timestamp_gap" not in names(results[2])


def test_timestamp_gap_flagged_and_scaled():
    series = [obs(0), Observation("S1", T0 + timedelta(hours=6), 28.0, 1005.0, 60.0)]
    results = run(series)
    assert "timestamp_gap" in names(results[1])


def test_minor_jitter_is_not_a_gap():
    """Transmission jitter is normal and must not raise an alarm."""
    series = [
        obs(0),
        Observation("S1", T0 + timedelta(minutes=68), 28.1, 1005.0, 60.0),
    ]
    assert "timestamp_gap" not in names(run(series)[1])


# --------------------------------------------------------------------------
# Aggregation and state isolation
# --------------------------------------------------------------------------

def test_score_is_max_not_sum():
    """One broken reading failing several checks must not exceed 1.0."""
    results = run([obs(0), obs(1, temp_c=60.5)])
    assert 0.0 <= results[1].score <= 1.0


def test_per_variable_scores_isolate_the_faulty_sensor():
    results = run([obs(0), obs(1, temp_c=48.0)])
    assert results[1].per_variable["temp_c"] > 0.0
    assert results[1].per_variable["pressure_hpa"] == 0.0
    assert results[1].per_variable["rh_pct"] == 0.0


def test_station_state_is_independent():
    """One station's fault must never affect another's checks."""
    engine = RuleEngine(interval_minutes=60)
    engine.evaluate(obs(0, station="A"))
    engine.evaluate(obs(1, station="A", temp_c=-9999.0))
    result = engine.evaluate(obs(1, station="B"))
    assert not result.flags


def test_reset_clears_state():
    engine = RuleEngine(interval_minutes=60)
    for i in range(16):
        engine.evaluate(Observation("S1", T0 + timedelta(hours=i), 27.4, 1005.0, 60.0))
    engine.reset("S1")
    result = engine.evaluate(Observation("S1", T0 + timedelta(hours=99), 27.4, 1005.0, 60.0))
    assert "persistence" not in names(result)


def test_all_flags_carry_readable_detail():
    """Explainability is structural: every flag must be presentable to a human."""
    results = run([obs(0), obs(1, temp_c=-9999.0, rh_pct=150.0), obs(2, temp_c=None)])
    for r in results:
        for f in r.flags:
            assert f.detail, f"flag {f.name} has no operator-facing detail"
            assert 0.0 <= f.score <= 1.0
