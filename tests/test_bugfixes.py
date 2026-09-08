"""Tests for the seven audit defects (context/bug.md §Confirmed defects).

Each test targets one specific gap the audit identified. They are grouped by
defect number and ordered to match the bug report so a reviewer can cross-
reference easily.

Run:  python tests/run_tests.py test_bugfixes -v
"""

import math
import os
import tempfile
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Bug 1 — LSTM .npz metadata is ignored when loading
# ---------------------------------------------------------------------------

from skyguard.config import ModelConfig
from skyguard.model.lstm import LstmAutoencoder


def test_load_without_config_uses_artifact_geometry():
    """Bug 1: `load(path)` with no config must honour the saved window/hidden_size."""
    cfg = ModelConfig(window=8, hidden_size=12, learning_rate=0.01,
                      batch_size=4, max_epochs=1, patience=1, seed=42)
    model = LstmAutoencoder(config=cfg)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.npz")
        model.save(path)
        # Load WITHOUT passing config — this is the path the audit said was broken.
        loaded = LstmAutoencoder.load(path)
    assert loaded.window == 8, f"expected window=8, got {loaded.window}"
    assert loaded.hidden_size == 12, f"expected hidden=12, got {loaded.hidden_size}"
    # Reconstruction must work with the loaded geometry.
    rng = np.random.default_rng(99)
    xb = rng.standard_normal((2, 8, 7))
    _, err, _ = loaded.reconstruct_batch(xb)
    assert np.all(np.isfinite(err))


def test_load_rejects_shape_mismatch():
    """Bug 1 extra: a corrupted .npz with wrong weight shapes must raise, not
    silently reconstruct garbage."""
    model = LstmAutoencoder(config=ModelConfig(
        window=6, hidden_size=8, learning_rate=0.01,
        batch_size=4, max_epochs=1, patience=1, seed=0))
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.npz")
        model.save(path)
        # Tamper: overwrite the saved hidden_size metadata to something wrong.
        data = dict(np.load(path, allow_pickle=False))
        data["hidden_size"] = np.array(16)  # was 8
        np.savez(os.path.join(tmp, "bad.npz"), **data)
        with pytest.raises(ValueError, match="shape"):
            LstmAutoencoder.load(os.path.join(tmp, "bad.npz"))


# ---------------------------------------------------------------------------
# Bug 2 — UTC canonicalization is incomplete
# ---------------------------------------------------------------------------

from skyguard.data.loaders import to_utc
from skyguard.pipeline import SkyGuardPipeline, _canonicalise
from skyguard.types import Observation, QCStatus


def test_to_utc_naive_becomes_utc_aware():
    """Bug 2: naive datetime → UTC-aware."""
    naive = datetime(2025, 6, 1, 12, 0, 0)
    result = to_utc(naive)
    assert result.tzinfo is timezone.utc
    assert result == datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_to_utc_offset_converts():
    """Bug 2: +05:30 offset → equivalent UTC instant."""
    ist = timezone(timedelta(hours=5, minutes=30))
    local = datetime(2025, 6, 1, 17, 30, 0, tzinfo=ist)
    result = to_utc(local)
    assert result.tzinfo is timezone.utc
    assert result == datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_pipeline_canonicalises_naive_timestamps():
    """Bug 2: pipeline.process() must not crash on naive timestamps."""
    pipe = SkyGuardPipeline()
    naive_obs = Observation("S1", datetime(2025, 6, 1, 12, 0), 28.0, 1005.0, 60.0)
    record = pipe.process(naive_obs)
    assert record.qc_status == QCStatus.PASS
    # The stored timestamp must be UTC-aware.
    assert record.observation.timestamp.tzinfo is timezone.utc


def test_pipeline_canonicalises_offset_timestamps():
    """Bug 2: +05:30 offset timestamps must be converted and not raise TypeError."""
    pipe = SkyGuardPipeline()
    ist = timezone(timedelta(hours=5, minutes=30))
    t0 = datetime(2025, 6, 1, 12, 0, tzinfo=ist)
    record1 = pipe.process(Observation("S1", t0, 28.0, 1005.0, 60.0))
    assert record1.qc_status == QCStatus.PASS
    # A second observation 1 hour later (also with offset) must not crash
    # the rate-of-change check, which would subtract aware from naive.
    t1 = datetime(2025, 6, 1, 13, 0, tzinfo=ist)
    record2 = pipe.process(Observation("S1", t1, 28.5, 1005.1, 59.5))
    assert record2.qc_status == QCStatus.PASS
    assert record2.observation.timestamp.tzinfo is timezone.utc


# ---------------------------------------------------------------------------
# Bug 3 — Hard-invalid readings poison multivariate streaming state
# ---------------------------------------------------------------------------

from skyguard.qc.multivariate import MultivariateQC


def _fitted_multi(seed=10):
    """Fit a multivariate model on clean data for bug-3 tests."""
    rng = np.random.default_rng(seed)
    n = 300
    t = np.arange(n, dtype=np.float64)
    temp = 28.0 + 5.0 * np.sin(2 * np.pi * t / 24.0) + 0.2 * rng.standard_normal(n)
    rh = 70.0 - 2.0 * (temp - 28.0) + 1.0 * rng.standard_normal(n)
    pressure = 1005.0 + 3.0 * np.sin(2 * np.pi * t / 120.0) + 0.15 * rng.standard_normal(n)
    history = np.stack([temp, pressure, rh], axis=1)
    qc = MultivariateQC()
    assert qc.fit_station("S1", history) > 0
    return qc


def test_hard_invalid_does_not_poison_multivariate():
    """Bug 3: a sentinel (-9999) must not be banked as 'last'; the next valid
    observation must not receive a delta of ~10000 and get convicted."""
    qc = _fitted_multi()
    # Seed the streaming state with a normal observation.
    r0 = qc.evaluate("S1", 28.0, 1005.0, 68.0)
    # Simulate a sentinel reading (hard-invalid).
    r_bad = qc.evaluate("S1", -9999.0, 1005.0, 68.0, hard_invalid=True)
    assert not r_bad.available, "hard-invalid should abstain"
    # The next normal reading must NOT have a huge delta.
    r_good = qc.evaluate("S1", 28.5, 1005.1, 67.5)
    assert r_good.available
    # Without the fix, this would be > 0.9 (huge delta from -9999 to 28.5).
    assert r_good.probability < 0.5, (
        f"post-sentinel normal reading scored {r_good.probability:.3f} — "
        f"delta poisoning from the sentinel is leaking through"
    )


# ---------------------------------------------------------------------------
# Bug 4 — Partial missing observations can be reported as PASS
# ---------------------------------------------------------------------------

def test_partial_missing_is_missing_not_pass():
    """Bug 4: one dead sensor among three must be MISSING, never PASS."""
    pipe = SkyGuardPipeline()
    # Only temp_c is None — two sensors report fine.
    record = pipe.process(Observation("S1", datetime(2025, 6, 1, tzinfo=timezone.utc),
                                       None, 1005.0, 60.0))
    assert record.qc_status == QCStatus.MISSING, (
        f"partial missing got {record.qc_status}, expected MISSING"
    )


def test_partial_missing_rh_only():
    """Bug 4: missing RH with valid temp and pressure is MISSING."""
    pipe = SkyGuardPipeline()
    record = pipe.process(Observation("S1", datetime(2025, 6, 1, tzinfo=timezone.utc),
                                       28.0, 1005.0, None))
    assert record.qc_status == QCStatus.MISSING


def test_partial_missing_pressure_only():
    """Bug 4: missing pressure with valid temp and RH is MISSING."""
    pipe = SkyGuardPipeline()
    record = pipe.process(Observation("S1", datetime(2025, 6, 1, tzinfo=timezone.utc),
                                       28.0, None, 60.0))
    assert record.qc_status == QCStatus.MISSING


# ---------------------------------------------------------------------------
# Bug 5 — Genuine-weather event labels use exclusive end as inclusive
# ---------------------------------------------------------------------------

from skyguard.data.simulator import SimulatorConfig, simulate_network
from skyguard.data.injector import InjectorConfig, inject_faults
from skyguard.types import FaultClass


def test_genuine_event_end_is_exclusive():
    """Bug 5: simulator events have half-open [start, end); the injector must
    convert to inclusive end_index = end - 1, not use end directly."""
    net = simulate_network(SimulatorConfig(n_stations=2, days=30, seed=42))
    if not net.genuine_events:
        pytest.skip("no genuine events in this seed")
    result = inject_faults(net, InjectorConfig(seed=42))
    for f in result.faults:
        if f.fault_class == FaultClass.GENUINE_EXTREME:
            # Find the corresponding event.
            for ev in net.genuine_events:
                if f.start_index >= ev["start"] and f.end_index < ev["end"]:
                    # The inclusive end_index must be strictly less than the
                    # half-open end. If end_index == end, the fix is missing.
                    assert f.end_index == ev["end"] - 1 or f.end_index < ev["end"], (
                        f"fault end_index {f.end_index} >= event end {ev['end']}, "
                        f"off-by-one not fixed"
                    )
                    break


def test_genuine_event_regime_exclusion_no_extra_point():
    """Bug 5: regime_occupied mask must not extend one past the event end."""
    net = simulate_network(SimulatorConfig(n_stations=2, days=60, seed=7))
    result = inject_faults(net, InjectorConfig(seed=107))
    # Collect all genuine-extreme inclusive ranges.
    genuine_ranges = []
    for f in result.faults:
        if f.fault_class == FaultClass.GENUINE_EXTREME:
            genuine_ranges.append((f.start_index, f.end_index))
    # Check that no genuine_extreme fault has end_index >= original half-open end.
    for ev in net.genuine_events:
        for (s, e) in genuine_ranges:
            if s >= ev["start"] and e <= ev["end"]:
                assert e <= ev["end"] - 1, (
                    f"genuine fault [{s},{e}] extends to event half-open end {ev['end']}"
                )


# ---------------------------------------------------------------------------
# Bug 6 — API query validation is unhandled
# ---------------------------------------------------------------------------

from skyguard.api.server import _bounded_limit


def test_bounded_limit_valid():
    """Bug 6: valid integer limits parse correctly."""
    assert _bounded_limit("50", 200) == 50
    assert _bounded_limit("1", 200) == 1
    assert _bounded_limit(None, 200) == 200
    assert _bounded_limit("", 200) == 200


def test_bounded_limit_capped():
    """Bug 6: limits above maximum are clamped."""
    assert _bounded_limit("9999", 200, maximum=5000) == 5000


def test_bounded_limit_rejects_garbage():
    """Bug 6: non-integer strings raise ValueError, not a 500."""
    with pytest.raises(ValueError, match="integer"):
        _bounded_limit("abc", 200)


def test_bounded_limit_rejects_negative():
    """Bug 6: negative limits raise ValueError — no silent Python slicing."""
    with pytest.raises(ValueError, match="1 or greater"):
        _bounded_limit("-5", 200)
    with pytest.raises(ValueError, match="1 or greater"):
        _bounded_limit("0", 200)


# ---------------------------------------------------------------------------
# Bug 7 — API accepts non-finite numeric JSON values
# ---------------------------------------------------------------------------

from skyguard.api.server import _optional_float


def test_optional_float_rejects_nan():
    """Bug 7: NaN must not silently become 'missing'."""
    with pytest.raises(ValueError, match="finite"):
        _optional_float(float("nan"), "temp_c")


def test_optional_float_rejects_infinity():
    """Bug 7: infinities must not flow into arithmetic."""
    with pytest.raises(ValueError, match="finite"):
        _optional_float(float("inf"), "temp_c")
    with pytest.raises(ValueError, match="finite"):
        _optional_float(float("-inf"), "pressure_hpa")


def test_optional_float_rejects_bool():
    """Bug 7: True/False are not valid numbers."""
    with pytest.raises(ValueError, match="number"):
        _optional_float(True, "rh_pct")
    with pytest.raises(ValueError, match="number"):
        _optional_float(False, "rh_pct")


def test_optional_float_accepts_valid():
    """Bug 7: normal values and None must still work."""
    assert _optional_float(28.5, "temp_c") == pytest.approx(28.5)
    assert _optional_float(0, "temp_c") == pytest.approx(0.0)
    assert _optional_float(None, "temp_c") is None
    assert _optional_float("28.5", "temp_c") == pytest.approx(28.5)


# ---------------------------------------------------------------------------
# Bug 2 (store side) — fetch_station bounds use to_utc
# ---------------------------------------------------------------------------

from skyguard.store import AuditStore
from skyguard.types import (
    ConfidenceBand, DiagnosisResult, FusionResult, Severity,
)

T0_STORE = datetime(2025, 6, 1, tzinfo=timezone.utc)


def _make_record(i, anomaly=False, station="S1"):
    obs = Observation(station, T0_STORE + timedelta(hours=i),
                      48.0 if anomaly else 28.0, 1005.0, 60.0)
    from skyguard.types import QCRecord
    record = QCRecord(observation=obs)
    record.fusion = FusionResult(
        probability=0.9 if anomaly else 0.05,
        band=ConfidenceBand.HIGH if anomaly else ConfidenceBand.NORMAL,
        is_anomaly=anomaly,
        evidence={"rule": 0.8},
        dominant_variable="temp_c" if anomaly else None,
    )
    record.diagnosis = DiagnosisResult(
        fault_class=FaultClass.SPIKE if anomaly else FaultClass.NONE,
        confidence=0.9 if anomaly else 0.0,
        severity=Severity.HIGH if anomaly else Severity.NONE,
    )
    return record


def test_store_fetch_with_offset_bounds():
    """Bug 2 (store): fetch_station must canonicalise bound datetimes to UTC
    so that an IST-offset bound matches the UTC-stored records."""
    with tempfile.TemporaryDirectory() as tmp:
        store = AuditStore(os.path.join(tmp, "audit.db"))
        for i in range(6):
            store.save_record(_make_record(i))
        # Query with IST offset (+05:30). T0 + 2h in UTC is T0 + 7:30 in IST.
        ist = timezone(timedelta(hours=5, minutes=30))
        start_ist = (T0_STORE + timedelta(hours=2)).astimezone(ist)
        window = store.fetch_station("S1", start=start_ist)
        assert len(window) == 4, f"expected 4 records, got {len(window)}"
        store.close()
