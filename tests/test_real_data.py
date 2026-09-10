"""Real-archive adapter tests: gridding, gap handling, extreme labelling.

The negative controls lead. A network-coherent warm spell is weather and must
be labelled; one station warming alone is what a fault looks like and must not
be. Getting that wrong silently breaks the genuine-extreme false-alarm number,
which is the flagship metric on real data.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from skyguard.data.real import (
    detect_genuine_extremes,
    infer_interval_minutes,
    network_from_observations,
    temperature_anomaly,
)
from skyguard.types import Observation, Station

UTC = timezone.utc


def _stations(n=3):
    return [
        Station(f"S{i}", f"Station {i}", 18.0 + 0.3 * i, 73.0 + 0.3 * i, 100.0 * i, "inland")
        for i in range(n)
    ]


def _hourly(start, hours):
    return [start + timedelta(hours=h) for h in range(hours)]


def _diurnal(stamps, offset=0.0):
    """Plain daily cycle so a synthetic series has structure to strip."""
    return [26.0 + offset + 6.0 * np.sin(2 * np.pi * (t.hour - 4) / 24.0) for t in stamps]


# --------------------------------------------------------------------------
# Gridding
# --------------------------------------------------------------------------

def test_observations_land_on_the_right_grid_cells():
    stations = _stations(2)
    stamps = _hourly(datetime(2024, 1, 1, tzinfo=UTC), 5)
    obs = [Observation(st.station_id, t, 20.0 + k, 1000.0 + k, 50.0 + k)
           for st in stations for k, t in enumerate(stamps)]
    net = network_from_observations(stations, obs)

    assert net.n_stations == 2
    assert net.n_steps == 5
    assert net.interval_minutes == 60
    assert net.temp[net.station_index["S1"], 3] == pytest.approx(23.0)
    assert net.rh[0, 0] == pytest.approx(50.0)


def test_absent_rows_stay_nan_and_are_never_interpolated():
    stations = _stations(1)
    stamps = _hourly(datetime(2024, 1, 1, tzinfo=UTC), 6)
    obs = [Observation("S0", t, 20.0, 1000.0, 50.0) for t in stamps if t.hour != 3]
    net = network_from_observations(stations, obs)

    assert net.n_steps == 6
    assert np.isnan(net.temp[0, 3])
    # The neighbours either side are untouched: a gap must not smear.
    assert net.temp[0, 2] == pytest.approx(20.0)
    assert net.temp[0, 4] == pytest.approx(20.0)


def test_missing_single_variable_is_nan_while_the_others_land():
    stations = _stations(1)
    stamps = _hourly(datetime(2024, 1, 1, tzinfo=UTC), 3)
    obs = [Observation("S0", stamps[0], 20.0, None, 50.0),
           Observation("S0", stamps[1], 21.0, 1001.0, 51.0),
           Observation("S0", stamps[2], 22.0, 1002.0, None)]
    net = network_from_observations(stations, obs)

    assert np.isnan(net.pressure[0, 0])
    assert net.temp[0, 0] == pytest.approx(20.0)
    assert np.isnan(net.rh[0, 2])


def test_unknown_stations_are_dropped_not_invented():
    stations = _stations(1)
    stamps = _hourly(datetime(2024, 1, 1, tzinfo=UTC), 3)
    obs = ([Observation("S0", t, 20.0, 1000.0, 50.0) for t in stamps]
           + [Observation("GHOST", t, 99.0, 900.0, 10.0) for t in stamps])
    net = network_from_observations(stations, obs)

    assert net.n_stations == 1
    assert "GHOST" not in net.station_index
    assert float(np.nanmax(net.temp)) == pytest.approx(20.0)


def test_naive_input_needs_no_special_casing():
    stations = _stations(1)
    stamps = [datetime(2024, 1, 1, h, tzinfo=UTC) for h in range(4)]
    obs = [Observation("S0", t, 20.0, 1000.0, 50.0) for t in stamps]
    net = network_from_observations(stations, obs)
    assert net.timestamps[0] == stamps[0]


def test_empty_inputs_are_refused_loudly():
    with pytest.raises(ValueError):
        network_from_observations([], [Observation("S0", datetime(2024, 1, 1, tzinfo=UTC), 1, 2, 3)])
    with pytest.raises(ValueError):
        network_from_observations(_stations(1), [])
    with pytest.raises(ValueError):
        network_from_observations(_stations(1),
                                  [Observation("OTHER", datetime(2024, 1, 1, tzinfo=UTC), 1, 2, 3)])


# --------------------------------------------------------------------------
# Interval inference
# --------------------------------------------------------------------------

def test_interval_is_modal_so_one_outage_cannot_shift_it():
    base = datetime(2024, 1, 1, tzinfo=UTC)
    stamps = [base + timedelta(minutes=10 * i) for i in range(20)]
    stamps.append(stamps[-1] + timedelta(hours=30))     # one long gap
    assert infer_interval_minutes(stamps) == 10


def test_interval_falls_back_to_hourly_on_a_single_timestamp():
    assert infer_interval_minutes([datetime(2024, 1, 1, tzinfo=UTC)]) == 60


# --------------------------------------------------------------------------
# Genuine extreme labelling: the negative controls
# --------------------------------------------------------------------------

def test_network_wide_warm_spell_is_labelled_as_weather():
    stations = _stations(4)
    stamps = _hourly(datetime(2024, 1, 1, tzinfo=UTC), 24 * 20)
    obs = []
    for st in stations:
        for k, t in enumerate(stamps):
            bump = 5.0 if 24 * 8 <= k < 24 * 11 else 0.0     # every station together
            obs.append(Observation(st.station_id, t, _diurnal([t])[0] + bump, 1000.0, 50.0))
    net = network_from_observations(stations, obs)

    events = detect_genuine_extremes(net, temp_anomaly_c=1.8, min_hours=24)
    heat = [e for e in events if e["kind"] == "heatwave"]
    assert heat, "a 3-day network-wide warm spell must be labelled genuine weather"
    assert heat[0]["start"] < 24 * 11 and heat[0]["end"] > 24 * 8


def test_one_station_warming_alone_is_not_weather():
    stations = _stations(4)
    stamps = _hourly(datetime(2024, 1, 1, tzinfo=UTC), 24 * 20)
    obs = []
    for st in stations:
        for k, t in enumerate(stamps):
            bump = 5.0 if (st.station_id == "S0" and 24 * 8 <= k < 24 * 11) else 0.0
            obs.append(Observation(st.station_id, t, _diurnal([t])[0] + bump, 1000.0, 50.0))
    net = network_from_observations(stations, obs)

    events = detect_genuine_extremes(net, temp_anomaly_c=1.8, min_hours=24)
    assert not [e for e in events if e["kind"] == "heatwave"], (
        "a lone station warming is the signature of a fault, not of weather")


def test_short_warm_spell_does_not_clear_the_duration_floor():
    stations = _stations(3)
    stamps = _hourly(datetime(2024, 1, 1, tzinfo=UTC), 24 * 20)
    obs = []
    for st in stations:
        for k, t in enumerate(stamps):
            bump = 6.0 if 24 * 8 <= k < 24 * 8 + 6 else 0.0    # 6 hours only
            obs.append(Observation(st.station_id, t, _diurnal([t])[0] + bump, 1000.0, 50.0))
    net = network_from_observations(stations, obs)
    assert not [e for e in detect_genuine_extremes(net, min_hours=24) if e["kind"] == "heatwave"]


def test_events_are_half_open_and_sorted():
    stations = _stations(3)
    stamps = _hourly(datetime(2024, 1, 1, tzinfo=UTC), 24 * 30)
    obs = []
    for st in stations:
        for k, t in enumerate(stamps):
            bump = 5.0 if (24 * 5 <= k < 24 * 8 or 24 * 18 <= k < 24 * 21) else 0.0
            obs.append(Observation(st.station_id, t, _diurnal([t])[0] + bump, 1000.0, 50.0))
    net = network_from_observations(stations, obs)

    events = detect_genuine_extremes(net, min_hours=24)
    assert events == sorted(events, key=lambda e: e["start"])
    for event in events:
        assert 0 <= event["start"] < event["end"] <= net.n_steps


def test_anomaly_series_removes_the_daily_cycle():
    """A pure diurnal swing carries no synoptic anomaly to speak of."""
    stations = _stations(2)
    stamps = _hourly(datetime(2024, 1, 1, tzinfo=UTC), 24 * 12)
    obs = [Observation(st.station_id, t, _diurnal([t])[0], 1000.0, 50.0)
           for st in stations for t in stamps]
    net = network_from_observations(stations, obs)

    anomaly = temperature_anomaly(net)
    assert float(np.nanmax(np.abs(anomaly))) < 1.0


# --------------------------------------------------------------------------
# The shipped archive file, when it is present
# --------------------------------------------------------------------------

def test_bundled_real_csv_loads_into_a_usable_network():
    from pathlib import Path

    from skyguard.data.real import load_real_network

    root = Path(__file__).resolve().parents[1]
    obs_csv = root / "examples" / "data" / "real_aws_hourly.csv"
    stations_csv = root / "examples" / "data" / "real_aws_stations.csv"
    if not obs_csv.exists():
        pytest.skip("run examples/fetch_real_data.py to download the archive CSV")

    net = load_real_network(str(obs_csv), str(stations_csv))
    assert net.n_stations >= 3
    assert net.interval_minutes == 60
    assert net.n_steps > 24 * 30
    # A real archive of this size should be essentially complete, and the
    # values have to be physically plausible or the loader mangled a column.
    assert np.isnan(net.temp).mean() < 0.02
    assert -20.0 < float(np.nanmin(net.temp)) < float(np.nanmax(net.temp)) < 60.0
    assert 500.0 < float(np.nanmin(net.pressure)) < float(np.nanmax(net.pressure)) < 1085.0
    assert 0.0 <= float(np.nanmin(net.rh)) and float(np.nanmax(net.rh)) <= 100.0
    assert net.genuine_events, "real weather in this window should yield extreme spans"


def test_bundled_real_network_accepts_injection():
    """The real network has to reach the injector through the same path as the
    simulator, or the whole point of the adapter is lost."""
    from pathlib import Path

    from skyguard.config import InjectorConfig
    from skyguard.data.injector import inject_faults
    from skyguard.data.real import load_real_network

    root = Path(__file__).resolve().parents[1]
    obs_csv = root / "examples" / "data" / "real_aws_hourly.csv"
    if not obs_csv.exists():
        pytest.skip("run examples/fetch_real_data.py to download the archive CSV")

    net = load_real_network(str(obs_csv), str(root / "examples" / "data" / "real_aws_stations.csv"))
    result = inject_faults(net, InjectorConfig(seed=11), index_range=(0, 600))
    labels = result.label_matrix()
    classes = result.class_matrix()
    assert result.faults
    assert labels[:, :601].any()
    assert len({str(c) for c in classes.ravel().tolist()} - {"none"}) >= 4
