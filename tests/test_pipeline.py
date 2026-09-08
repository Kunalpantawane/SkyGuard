"""Pipeline tests: cold start, gaps, missing data, and the full-stack proof.

The integration test fits every layer on clean synthetic structure, then
streams it: normal points must pass, a lone spike must be caught with the
blame on temperature, and the audit store must record it all.
"""

import os
import tempfile
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from skyguard.config import ModelConfig, ThresholdConfig
from skyguard.model.lstm import LstmAutoencoder
from skyguard.model.scaler import build_window, fit_all_scalers
from skyguard.model.threshold import fit_all as fit_all_thresholds
from skyguard.pipeline import SkyGuardPipeline
from skyguard.qc.multivariate import MultivariateQC
from skyguard.qc.spatial import SpatialQC
from skyguard.store import AuditStore
from skyguard.types import Observation, QCStatus, Station

T0 = datetime(2025, 6, 1, tzinfo=timezone.utc)


def diurnal(n: int, seed: int, phase: float = 0.0):
    """Clean coupled T/P/RH histories for two stations."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    temp = 28.0 + 5.0 * np.sin(2 * np.pi * t / 24.0 + phase) + 0.2 * rng.standard_normal(n)
    rh = 70.0 - 2.0 * (temp - 28.0) + 1.0 * rng.standard_normal(n)
    pressure = 1005.0 + 3.0 * np.sin(2 * np.pi * t / 120.0) + 0.15 * rng.standard_normal(n)
    stamps = [T0 + timedelta(hours=int(i)) for i in range(n)]
    return np.stack([temp, pressure, rh], axis=1), stamps


def make_windows(levels, stamps, scaler, width=6):
    """Sliding (W, 7) windows over clean levels + stamps."""
    return np.stack([build_window(scaler, levels[s:s + width], stamps[s:s + width])
                     for s in range(len(stamps) - width + 1)])


def fitted_stack(tmp_path: str):
    """A fully fitted three-station pipeline (no classifier: diagnosis stays NONE).

    Three stations because spatial QC needs two comparable neighbours to speak.
    """
    stations = {
        "A": Station("A", "Alpha", 18.50, 73.90, 200.0, "valley"),
        "B": Station("B", "Beta", 18.60, 74.00, 250.0, "valley"),
        "C": Station("C", "Gamma", 18.55, 74.05, 280.0, "valley"),
    }
    ids = ("A", "B", "C")
    clean = {sid: diurnal(220, seed)[0]
             for sid, seed in zip(ids, (501, 502, 503))}
    stamps = diurnal(220, 501)[1]
    scalers = fit_all_scalers(clean)

    cfg = ModelConfig(window=6, hidden_size=8, learning_rate=0.01,
                      batch_size=8, max_epochs=30, patience=8, seed=7)
    lstm = LstmAutoencoder(config=cfg)
    train_x = np.concatenate([make_windows(clean[s][:140], stamps[:140], scalers[s])
                              for s in ids])
    val_x = np.concatenate([make_windows(clean[s][140:200], stamps[140:200], scalers[s])
                            for s in ids])
    lstm.fit(train_x, train_x[:, :, :3], val_x, val_x[:, :, :3])

    _, val_err, _ = lstm.reconstruct_batch(val_x)
    thirds = np.array_split(val_err, 3)
    per_station = dict(zip(ids, thirds))
    thresholds = fit_all_thresholds(per_station, config=ThresholdConfig(min_samples=30))

    multi = MultivariateQC()
    for sid in ids:
        assert multi.fit_station(sid, clean[sid][:140]) > 0

    store = AuditStore(os.path.join(tmp_path, "audit.db"))
    pipe = SkyGuardPipeline(
        stations=stations, lstm=lstm, scalers=scalers, thresholds=thresholds,
        multivariate=multi, spatial=SpatialQC(stations), store=store,
    )
    return pipe, clean, stamps, store


def obs(sid: str, ts: datetime, triple):
    temp, pressure, rh = triple
    return Observation(sid, ts, temp, pressure, rh)


# --------------------------------------------------------------------------
# Streaming mechanics (no fitting needed)
# --------------------------------------------------------------------------

def test_cold_start_and_gap_reset_temporal_context():
    pipe = SkyGuardPipeline()

    def gentle(i):
        return Observation("S1", T0 + timedelta(hours=i),
                           28.0 + 0.1 * i, 1005.0 + 0.1 * i, 60.0 - 0.1 * i)

    first = pipe.process(gentle(0))
    assert not first.recon.available
    assert first.qc_status == QCStatus.PASS
    for i in range(1, 25):
        assert pipe.process(gentle(i)).qc_status == QCStatus.PASS
    # A gap resets temporal context; the verdict stays well-formed throughout.
    assert pipe.process(Observation("S1", T0 + timedelta(hours=25), None, None, None)).qc_status == QCStatus.MISSING
    assert pipe.process(gentle(26)).qc_status == QCStatus.PASS


def test_missing_observation_is_missing_not_anomaly():
    pipe = SkyGuardPipeline()
    record = pipe.process(Observation("S1", T0, None, None, None))
    assert record.qc_status == QCStatus.MISSING
    assert not record.fusion.is_anomaly


def test_sentinel_quarantines():
    pipe = SkyGuardPipeline()
    record = pipe.process(Observation("S1", T0, -9999.0, 1005.0, 60.0))
    assert record.fusion.is_anomaly
    assert record.qc_status == QCStatus.QUARANTINED
    assert record.fusion.probability >= 0.95


# --------------------------------------------------------------------------
# Full-stack integration
# --------------------------------------------------------------------------

def test_end_to_end_spike_caught_quiet_passes():
    tmp = tempfile.TemporaryDirectory()
    store = None
    try:
        pipe, clean, stamps, store = fitted_stack(tmp.name)
        series = {sid: clean[sid][190:] for sid in ("A", "B", "C")}
        stamps_tail = stamps[190:]

        def neighbours_of(sid, i):
            return [obs(other, stamps_tail[i], series[other][i])
                    for other in ("A", "B", "C") if other != sid]

        # 20 quiet points stream first to fill every layer's context.
        for i in range(20):
            for sid in ("A", "B", "C"):
                record = pipe.process(obs(sid, stamps_tail[i], series[sid][i]),
                                      neighbours=neighbours_of(sid, i))
                assert not record.fusion.is_anomaly
        # Lone spike on A.
        spiked = series["A"][20].copy()
        spiked[0] += 8.0
        caught = pipe.process(obs("A", stamps_tail[20], spiked),
                              neighbours=neighbours_of("A", 20))
        assert caught.recon.available
        assert caught.spatial.available
        assert caught.fusion.is_anomaly
        assert caught.fusion.dominant_variable == "temp_c"
        assert caught.qc_status in (QCStatus.QUARANTINED, QCStatus.ESTIMATED)
        assert caught.diagnosis.fault_class.value == "none"  # no classifier wired
        assert caught.latency_ms >= 0.0
        assert "p=" in caught.explanation
        assert store.count("A") == 21
    finally:
        if store is not None:
            store.close()
        tmp.cleanup()


def test_reset_station_clears_stream():
    pipe = SkyGuardPipeline()
    pipe.process(Observation("S1", T0, 28.0, 1005.0, 60.0))
    pipe.reset_station("S1")
    record = pipe.process(Observation("S1", T0 + timedelta(hours=1), 28.0, 1005.0, 60.0))
    assert record.qc_status == QCStatus.PASS
