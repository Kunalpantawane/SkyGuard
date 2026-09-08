"""Layer 4 tests.

The two cases that matter most pull in opposite directions: a lone spike must
score high, while a heatwave every station shares must score low — that split
is the genuine-weather discriminator. Plus terrain correction, abstention, and
staleness filtering.
"""

from datetime import datetime, timedelta, timezone

import pytest

from skyguard.qc.spatial import (
    SpatialQC,
    correct_pressure_to_elevation,
    correct_temp_to_elevation,
    haversine_km,
)
from skyguard.types import Observation, Station

T0 = datetime(2025, 6, 1, tzinfo=timezone.utc)


def registry() -> dict:
    return {
        "TGT": Station("TGT", "Valley", 18.50, 73.90, 200.0, "valley"),
        "N1": Station("N1", "Near", 18.55, 73.95, 250.0, "valley"),
        "N2": Station("N2", "Mid", 18.70, 74.10, 300.0, "inland"),
        "HILL": Station("HILL", "High", 18.52, 73.92, 1300.0, "hill"),
        "FAR": Station("FAR", "Far", 22.00, 78.00, 250.0, "inland"),
    }


def obs(station: str, temp_c, pressure_hpa=1005.0, rh_pct=60.0, at: datetime = T0):
    return Observation(station, at, temp_c, pressure_hpa, rh_pct)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def test_haversine_sanity():
    assert haversine_km(18.5, 73.9, 18.5, 73.9) == pytest.approx(0.0)
    assert haversine_km(18.0, 73.0, 19.0, 73.0) == pytest.approx(111.2, rel=0.01)


def test_lapse_and_pressure_corrections():
    # 400 m of climb at 6.5 C/km costs 2.6 C.
    assert correct_temp_to_elevation(27.4, 600.0, 200.0, 0.0065) == pytest.approx(30.0)
    # Pressure rises going downhill; 100 m near sea level is ~12 hPa.
    assert correct_pressure_to_elevation(1000.0, 25.0, 300.0, 200.0) > 1000.0
    assert correct_pressure_to_elevation(1000.0, 25.0, 300.0, 200.0) == pytest.approx(
        1012.0, abs=2.0
    )


# --------------------------------------------------------------------------
# The discriminator pair
# --------------------------------------------------------------------------

def test_lone_spike_scores_high():
    qc = SpatialQC(registry())
    target = obs("TGT", 45.0)
    neighbours = [obs("N1", 30.0), obs("N2", 30.2), obs("HILL", 24.0), obs("FAR", 30.0)]
    result = qc.evaluate(target, neighbours)
    assert result.available
    assert result.probability > 0.7
    assert result.n_neighbours == 2  # HILL too high, FAR too far
    assert result.neighbour_ids == ["N1", "N2"]  # nearer first
    assert result.expected["temp_c"] == pytest.approx(30.1, abs=0.5)
    assert result.normalised["temp_c"] > 3.5


def test_shared_heatwave_scores_low():
    """Every station at 38 C is weather, not a sensor fault."""
    qc = SpatialQC(registry())
    target = obs("TGT", 38.0, 1002.0, 45.0)
    neighbours = [obs("N1", 37.7, 1001.8, 46.0), obs("N2", 37.4, 1001.5, 47.0)]
    result = qc.evaluate(target, neighbours)
    assert result.available
    assert result.probability < 0.4


def test_terrain_difference_is_not_a_fault():
    """A hill station reading colder is the lapse rate, not a broken sensor."""
    stations = {
        "TGT": Station("TGT", "Valley", 18.50, 73.90, 200.0, "valley"),
        "UP": Station("UP", "Slope", 18.53, 73.93, 600.0, "hill"),
        "UP2": Station("UP2", "Slope2", 18.56, 73.96, 500.0, "hill"),
    }
    qc = SpatialQC(stations)
    target = obs("TGT", 30.0)
    # 27.4 = 30 - 0.0065*400; 28.05 = 30 - 0.0065*300. Pressures are the
    # matching station pressures at altitude (~960 at 600 m, ~971 at 500 m
    # against 1005 in the valley) — comparing raw values would be the bug.
    neighbours = [obs("UP", 27.4, pressure_hpa=960.0), obs("UP2", 28.05, pressure_hpa=971.0)]
    result = qc.evaluate(target, neighbours)
    assert result.available
    assert result.normalised["temp_c"] < 1.0
    assert result.probability < 0.4


# --------------------------------------------------------------------------
# Abstention and filtering
# --------------------------------------------------------------------------

def test_incomparable_neighbourhood_abstains():
    qc = SpatialQC(registry())
    result = qc.evaluate(obs("TGT", 45.0), [obs("HILL", 24.0), obs("FAR", 30.0)])
    assert not result.available


def test_stale_and_unknown_neighbours_excluded():
    qc = SpatialQC(registry())
    stale = obs("N1", 30.0, at=T0 + timedelta(hours=2))
    ghost = Observation("GHOST", T0, 30.0, 1005.0, 60.0)
    result = qc.evaluate(obs("TGT", 45.0), [stale, ghost, obs("N2", 30.2)])
    assert not result.available  # only N2 survives: fewer than two


def test_missing_values_skip_variable_not_verdict():
    qc = SpatialQC(registry())
    neighbours = [obs("N1", 30.0), obs("N2", 30.2)]
    partial = Observation("TGT", T0, None, 1005.0, 60.0)
    result = qc.evaluate(partial, neighbours)
    assert result.available
    assert "temp_c" not in result.normalised
    assert "pressure_hpa" in result.normalised
    empty = Observation("TGT", T0, None, None, None)
    assert not qc.evaluate(empty, neighbours).available


def test_unknown_target_abstains():
    qc = SpatialQC(registry())
    assert not qc.evaluate(obs("NOWHERE", 30.0), [obs("N1", 30.0)]).available
