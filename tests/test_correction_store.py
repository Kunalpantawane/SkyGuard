"""Correction and audit-store tests.

The blender must weight honestly, renormalise over speaking sources, refuse
long outages and weak confidence — and reject unknown sources loudly. The
store must round-trip records losslessly and refuse to overwrite raw data.
"""

import os
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from skyguard.correction.imputer import ImputedValue, SourceEstimate, impute
from skyguard.store import AuditStore
from skyguard.types import (
    ConfidenceBand,
    DiagnosisResult,
    FaultClass,
    FusionResult,
    Observation,
    QCRecord,
    Severity,
)

T0 = datetime(2025, 6, 1, tzinfo=timezone.utc)


def estimate(value, confidence, method="test"):
    return SourceEstimate(value=value, confidence=confidence, method=method)


# --------------------------------------------------------------------------
# Blender
# --------------------------------------------------------------------------

def test_blend_weights_all_sources():
    result = impute({
        "temporal": estimate(30.0, 0.8),
        "spatial": estimate(31.0, 0.6),
        "reconstruction": estimate(29.0, 0.4),
    }, n_consecutive_bad=1)
    assert result.available
    # 0.4*30 + 0.4*31 + 0.2*29 = 30.2 ; conf 0.4*.8 + 0.4*.6 + 0.2*.4 = 0.64.
    assert result.value == pytest.approx(30.2)
    assert result.confidence == pytest.approx(0.64)
    assert result.method == "reconstruction+spatial+temporal"


def test_weights_renormalise_over_speakers():
    result = impute({"spatial": estimate(31.0, 0.9)}, n_consecutive_bad=2)
    assert result.available
    assert result.value == pytest.approx(31.0)
    assert result.confidence == pytest.approx(0.9)
    assert result.method == "spatial"


def test_long_outage_and_weak_confidence_refuse():
    refused = impute({"spatial": estimate(31.0, 0.9)}, n_consecutive_bad=13)
    assert not refused.available
    assert "refused" in refused.method
    weak = impute({"spatial": estimate(31.0, 0.1)}, n_consecutive_bad=1)
    assert not weak.available
    assert isinstance(refused, ImputedValue)


def test_no_source_and_unknown_source():
    empty = impute({"spatial": estimate(None, 0.0)}, n_consecutive_bad=1)
    assert not empty.available
    with pytest.raises(ValueError):
        impute({"telepathy": estimate(30.0, 0.9)}, n_consecutive_bad=1)


# --------------------------------------------------------------------------
# Audit store
# --------------------------------------------------------------------------

def make_record(i: int, anomaly: bool, station: str = "S1") -> QCRecord:
    obs = Observation(station, T0 + timedelta(hours=i),
                      48.0 if anomaly else 28.0, 1005.0, 60.0)
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


def test_store_roundtrip_is_lossless():
    with tempfile.TemporaryDirectory() as tmp:
        store = AuditStore(os.path.join(tmp, "audit.db"))
        record = make_record(0, True)
        store.save_record(record)
        docs = store.fetch_station("S1")
        assert len(docs) == 1
        assert docs[0] == record.to_dict()
        assert docs[0]["observation"]["temp_c"] == pytest.approx(48.0)
        assert store.count("S1") == 1
        store.close()


def test_raw_is_immutable_first_write_wins():
    with tempfile.TemporaryDirectory() as tmp:
        store = AuditStore(os.path.join(tmp, "audit.db"))
        store.save_record(make_record(0, True))
        with pytest.raises(sqlite3.IntegrityError):
            store.save_record(make_record(0, False))  # same station+time
        docs = store.fetch_station("S1")
        assert len(docs) == 1
        assert docs[0]["observation"]["temp_c"] == pytest.approx(48.0)
        store.close()


def test_anomaly_worklist_and_time_bounds():
    with tempfile.TemporaryDirectory() as tmp:
        store = AuditStore(os.path.join(tmp, "audit.db"))
        for i in range(6):
            store.save_record(make_record(i, anomaly=(i in (1, 4))))
        anomalies = store.fetch_anomalies("S1")
        assert len(anomalies) == 2
        assert anomalies[0]["observation"]["timestamp"] > anomalies[1]["observation"]["timestamp"]
        window = store.fetch_station("S1", start=T0 + timedelta(hours=2))
        assert len(window) == 4
        assert store.count() == 6
        store.close()


def test_store_exposes_no_mutation_api():
    assert not hasattr(AuditStore, "update_record")
    assert not hasattr(AuditStore, "delete_record")
