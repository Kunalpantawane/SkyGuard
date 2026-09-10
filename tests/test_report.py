"""Tests for the benchmark report: curve metrics, layer muting, end to end.

The end-to-end test runs the whole generator on a 30-day slice of the archive.
It is slow by the standards of this suite and worth it: `report.py` produces
every number anyone will quote, so a structural break in it has to fail here
rather than in a demo.
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from skyguard.eval.metrics import pr_curve, roc_curve, score_histogram
from skyguard.eval.report import LADDER, refuse
from skyguard.fusion.engine import FusionEngine
from skyguard.types import (
    MultivariateResult,
    Observation,
    QCFlag,
    QCRecord,
    ReconResult,
    RuleResult,
    SpatialResult,
)
from support import archive_available


# --------------------------------------------------------------------------
# Curve metrics
# --------------------------------------------------------------------------

def test_perfect_ranking_scores_auc_one():
    labels = np.array([False] * 40 + [True] * 10)
    scores = np.array([0.1] * 40 + [0.9] * 10)
    assert roc_curve(scores, labels)["auc"] == pytest.approx(1.0)
    assert pr_curve(scores, labels)["average_precision"] == pytest.approx(1.0)


def test_inverted_ranking_scores_auc_zero():
    labels = np.array([False] * 40 + [True] * 10)
    scores = np.array([0.9] * 40 + [0.1] * 10)
    assert roc_curve(scores, labels)["auc"] == pytest.approx(0.0)


def test_constant_scores_land_on_the_diagonal():
    labels = np.array([False] * 30 + [True] * 30)
    scores = np.full(60, 0.5)
    assert roc_curve(scores, labels)["auc"] == pytest.approx(0.5, abs=1e-9)


def test_pr_baseline_is_the_positive_rate():
    labels = np.array([False] * 90 + [True] * 10)
    assert pr_curve(np.linspace(0, 1, 100), labels)["baseline"] == pytest.approx(0.1)


def test_curves_stay_within_the_point_budget():
    rng = np.random.default_rng(3)
    labels = rng.random(5000) < 0.2
    scores = rng.random(5000)
    roc = roc_curve(scores, labels, max_points=50)
    # The sweep is capped at max_points, plus the two anchor corners.
    assert len(roc["fpr"]) <= 52
    assert len(pr_curve(scores, labels, max_points=50)["recall"]) <= 50


def test_single_class_truth_returns_nan_rather_than_a_made_up_number():
    labels = np.zeros(20, dtype=bool)
    assert np.isnan(roc_curve(np.linspace(0, 1, 20), labels)["auc"])


def test_mismatched_shapes_are_refused():
    with pytest.raises(ValueError):
        roc_curve(np.zeros(5), np.zeros(6, dtype=bool))
    with pytest.raises(ValueError):
        pr_curve(np.zeros(5), np.zeros(6, dtype=bool))


def test_histogram_counts_every_observation_once():
    rng = np.random.default_rng(5)
    labels = rng.random(400) < 0.3
    scores = rng.random(400)
    hist = score_histogram(scores, labels, n_bins=10)
    assert sum(hist["normal"]) + sum(hist["anomalous"]) == 400
    assert len(hist["edges"]) == 11


# --------------------------------------------------------------------------
# Ablation by re-fusion
# --------------------------------------------------------------------------

def _record(rule_score=0.0, recon_prob=0.0, multi_prob=0.0, spatial_prob=0.0):
    obs = Observation("S0", datetime(2024, 1, 1, tzinfo=timezone.utc), 25.0, 1000.0, 60.0)
    rules = RuleResult(
        flags=[QCFlag(name="rate", variable="temp_c", score=rule_score)],
        score=rule_score,
    ) if rule_score else RuleResult()
    return QCRecord(
        observation=obs,
        rules=rules,
        recon=ReconResult(available=True, error_total=1.0, probability=recon_prob),
        multivariate=MultivariateResult(available=True, mahalanobis=3.0, probability=multi_prob),
        spatial=SpatialResult(available=True, probability=spatial_prob),
    )


def test_muting_a_layer_removes_its_vote():
    """A loud spatial layer must not move the verdict once it is muted."""
    fusion = FusionEngine()
    record = _record(recon_prob=0.4, multi_prob=0.4, spatial_prob=1.0)
    with_spatial = refuse(fusion, record, ("rules", "recon", "multi", "spatial"))
    without_spatial = refuse(fusion, record, ("rules", "recon", "multi"))
    assert with_spatial > without_spatial


def test_refusing_everything_reproduces_the_records_own_verdict():
    fusion = FusionEngine()
    record = _record(rule_score=0.3, recon_prob=0.6, multi_prob=0.5, spatial_prob=0.4)
    record.fusion = fusion.fuse(record.rules, record.recon, record.multivariate, record.spatial)
    assert refuse(fusion, record, ("rules", "recon", "multi", "spatial")) == pytest.approx(
        record.fusion.probability)


def test_ladder_is_cumulative():
    """Each rung may only add layers, never drop one the rung below had."""
    seen = set()
    for _, allowed in LADDER[1:]:
        assert seen <= set(allowed), "a ladder rung dropped a layer the rung below used"
        seen = set(allowed)


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------

class _Args:
    """Smallest run that still exercises every stage."""
    days = 30
    window = 6
    hidden = 6
    epochs = 2
    batch_size = 16
    learning_rate = 0.02
    stride = 4
    seed = 5
    lof_sample = 400
    verbose = False


_PAYLOAD = {}


def _payload():
    """Run the generator once and reuse it; it is the slow part of this file."""
    if not archive_available():
        pytest.skip("run examples/fetch_real_data.py to download the archive CSV")
    if "value" not in _PAYLOAD:
        from skyguard.eval.report import run

        _PAYLOAD["value"] = run(_Args())
    return _PAYLOAD["value"]


def test_report_run_produces_a_complete_payload():
    payload = _payload()

    for key in ("meta", "stations", "training", "thresholds", "injection", "ablation",
                "hybrid", "operational", "performance", "learned", "timelines",
                "cases", "health"):
        assert key in payload, f"payload is missing {key}"

    assert "Open-Meteo" in payload["meta"]["source_label"]
    assert payload["training"]["gradcheck"]["passed"], "the BPTT gate must pass in every run"
    assert len(payload["ablation"]) >= 8, "the ablation ladder lost rows"
    assert len(payload["timelines"]) == payload["meta"]["n_stations"]

    # Every rate has to be a rate, or a chart axis somewhere is lying.
    for row in payload["ablation"]:
        for key in ("precision", "recall", "f1", "false_alarm_rate",
                    "event_recall", "genuine_extreme_far", "roc_auc"):
            assert 0.0 <= row[key] <= 1.0, f"{row['name']}.{key} is out of range"

    # The two decision tiers must nest: anything quarantined was also held back.
    strict = payload["hybrid"]["point"]["recall"]
    review = payload["operational"]["point"]["recall"]
    assert review >= strict - 1e-9, "quarantining more than review does is incoherent"

    timeline = payload["timelines"][0]
    lengths = {len(timeline[k]) for k in ("timestamps", "temp_c", "probability",
                                          "truth", "predicted", "true_class")}
    assert len(lengths) == 1, "timeline arrays must line up point for point"


def test_written_payload_is_valid_json_and_loadable_js():
    from skyguard.eval.report import write_payload

    with tempfile.TemporaryDirectory() as tmp:
        js_path = write_payload(_payload(), Path(tmp) / "results.js")
        assert not js_path.with_suffix(".json").exists(), (
            "the duplicate JSON payload should not be written any more")

        text = js_path.read_text(encoding="utf-8")
        assert text.startswith("// Generated by")
        assert "window.SKYGUARD_RESULTS = {" in text

        prefix, _, body = text.partition("\n")
        loaded = json.loads(body.strip().removeprefix("window.SKYGUARD_RESULTS = ").rstrip(";\n"))
        assert loaded["meta"]["seed"] == _Args.seed
        # NaN and Infinity are not JSON, and a browser parsing the bundle would
        # choke on them; `allow_nan=False` is what keeps them out.
        assert "NaN" not in text and "Infinity" not in text
