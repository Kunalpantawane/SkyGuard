"""Eval tests: metric arithmetic plus the harness on injected data.

Hand-built fixtures pin the metric math; the harness test runs rules and
z-scores over a small seeded simulator slice with injected faults and checks
the scoreboard is structurally sound — and that deterministic Layer 1 catches
(corrupt encodings) are fully recalled.
"""

import numpy as np
import pytest

from skyguard.baselines.zscore import RobustZScore
from skyguard.config import InjectorConfig, SimulatorConfig
from skyguard.data.injector import inject_faults
from skyguard.data.simulator import simulate_network
from skyguard.eval.harness import RulesBaseline, compare, format_table
from skyguard.eval.metrics import (
    confusion_matrix,
    event_recall,
    expected_calibration_error,
    genuine_extreme_far,
    per_class_recall,
    point_metrics,
    spans_from_labels,
)


# --------------------------------------------------------------------------
# Metric arithmetic
# --------------------------------------------------------------------------

def test_point_metrics_hand_checked():
    m = point_metrics(
        np.array([True, True, False, False]),
        np.array([True, False, True, False]),
    )
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)
    assert m["false_alarm_rate"] == pytest.approx(0.5)
    assert m["n_true"] == 2.0
    with pytest.raises(ValueError):
        point_metrics(np.array([True]), np.array([True, False]))


def test_event_recall_and_spans():
    assert spans_from_labels(np.array([False, True, True, False, True])) == [(1, 2), (4, 4)]
    out = event_recall([(2, 4), (10, 12)],
                       np.array([False] * 3 + [True] + [False] * 7 + [True] + [False]))
    assert out["event_recall"] == pytest.approx(1.0)
    assert out["mean_detection_delay"] == pytest.approx(1.0)
    missed = event_recall([(0, 1)], np.zeros(4, dtype=bool))
    assert missed["event_recall"] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        event_recall([(0, 99)], np.zeros(4, dtype=bool))


def test_per_class_and_genuine_far():
    classes = np.array(["spike", "spike", "none", "genuine_extreme", "genuine_extreme"],
                       dtype=object)
    pred = np.array([True, False, False, True, False])
    recall = per_class_recall(classes, pred)
    assert set(recall) == {"spike"}
    assert recall["spike"] == pytest.approx(0.5)
    assert genuine_extreme_far(classes, pred) == pytest.approx(0.5)
    assert genuine_extreme_far(np.array(["none"]), np.array([True])) == 0.0


def test_confusion_and_calibration():
    matrix = confusion_matrix(["a", "b", "a"], ["a", "a", "b"], ["a", "b"])
    assert matrix.tolist() == [[1, 1], [1, 0]]
    perfect = expected_calibration_error(
        np.array([0.95, 0.9, 0.1, 0.05]), np.array([True, True, False, False]))
    assert perfect["ece"] < 0.1
    vague = expected_calibration_error(
        np.array([0.5, 0.5, 0.5, 0.5]), np.array([True, True, False, False]))
    assert vague["ece"] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        expected_calibration_error(np.array([0.5]), np.array([True, False]))


# --------------------------------------------------------------------------
# Harness on injected data
# --------------------------------------------------------------------------

def injected_slice():
    """Small seeded network, faults confined to the test split."""
    net = simulate_network(SimulatorConfig(n_stations=2, days=40, seed=7))
    train, _, test = (slice(0, 576), slice(576, 768), slice(768, 960))
    result = inject_faults(net, InjectorConfig(seed=11), index_range=(test.start, test.stop - 1))
    faulted = result.network
    levels = np.stack([faulted.temp[0, test], faulted.pressure[0, test],
                       faulted.rh[0, test]], axis=1)
    train_levels = np.stack([net.temp[0, train], net.pressure[0, train],
                             net.rh[0, train]], axis=1)
    return (levels, train_levels, result.label_matrix()[0, test],
            result.class_matrix()[0, test], faulted.timestamps[test.start:test.stop])


def test_harness_compares_detectors_on_injected_data():
    levels, train_levels, truth, classes, stamps = injected_slice()
    assert truth.sum() > 0  # the slice must contain real faults to be a test
    detectors = {
        "rules": RulesBaseline(station_id="S0"),
        "zscore": RobustZScore().fit(train_levels),
    }
    reports = compare(detectors, levels, truth, classes, timestamps=stamps)
    assert [r.name for r in reports] == ["rules", "zscore"]
    for report in reports:
        assert 0.0 <= report.point["f1"] <= 1.0
        assert 0.0 <= report.genuine_extreme_far <= 1.0
    # Deterministic Layer 1 catches: every corrupt point is a hard violation.
    corrupt = classes == "corrupt"
    if corrupt.sum():
        assert reports[0].per_class["corrupt"] == pytest.approx(1.0)
    table = format_table(reports)
    assert "rules" in table and "zscore" in table


def test_rules_baseline_needs_timestamps():
    with pytest.raises(NotImplementedError):
        RulesBaseline().score_points(np.zeros((4, 3)))
