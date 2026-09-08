"""SkyGuard AI quickstart: simulate -> train -> stream -> score.

Run from the repo root:

    python examples/quickstart.py

What it does, end to end, on a small synthetic network:
  1. Simulates 3 clean AWS stations (16 days, hourly).
  2. Trains the numpy LSTM autoencoder on the train split only.
  3. Calibrates per-station thresholds + multivariate models on clean splits.
  4. Injects labelled faults into the held-out test split.
  5. Streams the test slice through the full pipeline and scores it.
  6. Replays the SIH headline example (55 C spike, neighbours normal).

`main()` returns the summary dict so tests and notebooks can reuse it.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from skyguard.config import (  # noqa: E402
    InjectorConfig,
    ModelConfig,
    SimulatorConfig,
    ThresholdConfig,
)
from skyguard.data.injector import inject_faults  # noqa: E402
from skyguard.data.simulator import simulate_network  # noqa: E402
from skyguard.eval.harness import compare, format_table  # noqa: E402
from skyguard.eval.metrics import (  # noqa: E402
    event_recall,
    genuine_extreme_far,
    per_class_recall,
    point_metrics,
    spans_from_labels,
)
from skyguard.baselines.zscore import RobustZScore  # noqa: E402
from skyguard.eval.harness import RulesBaseline  # noqa: E402
from skyguard.model.lstm import LstmAutoencoder  # noqa: E402
from skyguard.model.scaler import fit_all_scalers  # noqa: E402
from skyguard.model.threshold import fit_all as fit_all_thresholds  # noqa: E402
from skyguard.pipeline import SkyGuardPipeline  # noqa: E402
from skyguard.qc.multivariate import MultivariateQC  # noqa: E402
from skyguard.qc.spatial import SpatialQC  # noqa: E402
from skyguard.types import Observation  # noqa: E402

WINDOW = 6


def _windows(levels: np.ndarray, stamps: list, scaler, width: int = WINDOW):
    from skyguard.model.scaler import build_window

    return np.stack([build_window(scaler, levels[s:s + width], stamps[s:s + width])
                     for s in range(len(stamps) - width + 1)])


def _clean_triplet(faulted, station_id: str, t: int):
    """Faulted-array row as pipeline-grade values (NaN outage -> None)."""
    i = faulted.station_index[station_id]
    out = []
    for series in (faulted.temp[i, t], faulted.pressure[i, t], faulted.rh[i, t]):
        out.append(None if np.isnan(series) else float(series))
    return tuple(out)


class _PooledScores:
    """Detector-protocol wrapper around precomputed station-pooled scores."""

    def __init__(self, scores: np.ndarray) -> None:
        self.scores = np.asarray(scores, dtype=np.float64)

    def score_points(self, levels: np.ndarray) -> np.ndarray:
        """Ignore the levels; the streaming scores are already pooled."""
        assert levels.shape[0] == self.scores.shape[0]
        return self.scores


def main(days: int = 40, epochs: int = 12, seed: int = 7) -> Dict[str, object]:
    """Run the demo; returns point/event/class metrics plus the SIH verdict."""
    net = simulate_network(SimulatorConfig(n_stations=3, days=days, seed=seed))
    ids = [s.station_id for s in net.stations]
    n = net.n_steps
    train, val, test = slice(0, int(n * 0.6)), slice(int(n * 0.6), int(n * 0.8)), slice(int(n * 0.8), n)

    histories = {sid: np.stack([net.temp[net.station_index[sid], train],
                                net.pressure[net.station_index[sid], train],
                                net.rh[net.station_index[sid], train]], axis=1)
                 for sid in ids}
    scalers = fit_all_scalers(histories)

    model_cfg = ModelConfig(window=WINDOW, hidden_size=8, learning_rate=0.01,
                            batch_size=8, max_epochs=epochs, patience=max(3, epochs // 3),
                            seed=seed)
    lstm = LstmAutoencoder(config=model_cfg)
    train_x = np.concatenate([_windows(histories[sid], net.timestamps[train], scalers[sid])
                              for sid in ids])
    val_block = {sid: np.stack([net.temp[net.station_index[sid], val],
                                net.pressure[net.station_index[sid], val],
                                net.rh[net.station_index[sid], val]], axis=1)
                 for sid in ids}
    val_x = np.concatenate([_windows(val_block[sid], net.timestamps[val], scalers[sid])
                            for sid in ids])
    fit_info = lstm.fit(train_x, train_x[:, :, :3], val_x, val_x[:, :, :3])

    _, val_err, _ = lstm.reconstruct_batch(val_x)
    parts = np.array_split(val_err, len(ids))
    thresholds = fit_all_thresholds(dict(zip(ids, parts)),
                                    config=ThresholdConfig(min_samples=30))
    multi = MultivariateQC()
    for sid in ids:
        multi.fit_station(sid, histories[sid])

    stations = {s.station_id: s for s in net.stations}
    pipe = SkyGuardPipeline(stations=stations, lstm=lstm, scalers=scalers,
                            thresholds=thresholds, multivariate=multi,
                            spatial=SpatialQC(stations))

    injected = inject_faults(net, InjectorConfig(seed=seed + 100),
                             index_range=(test.start, test.stop - 1))
    faulted, truth = injected.network, injected.label_matrix()
    classes = injected.class_matrix()
    stamps = faulted.timestamps[test]

    # Stream every station's test slice (timestamp-major within a station,
    # neighbours alongside) and pool the verdicts in label-matrix order.
    pooled_pred, pooled_true, pooled_classes = [], [], []
    pooled_levels, pooled_train, pooled_stamps = [], [], []
    for sid in ids:
        i = faulted.station_index[sid]
        for t in range(test.start, test.stop):
            raw = (faulted.temp[i, t], faulted.pressure[i, t], faulted.rh[i, t])
            # Injected outages arrive as NaN; the pipeline contract wants None
            # (a sentinel-looking float must never become a silent zero).
            triple = tuple(None if np.isnan(v) else float(v) for v in raw)
            others = [Observation(o, stamps[t - test.start],
                                  *_clean_triplet(faulted, o, t))
                      for o in ids if o != sid]
            record = pipe.process(Observation(sid, stamps[t - test.start], *triple),
                                  neighbours=others)
            pooled_pred.append(record.fusion.is_anomaly)
        pooled_true.extend(truth[i, test].tolist())
        pooled_classes.extend(classes[i, test].tolist())
        pooled_levels.append(np.stack([faulted.temp[i, test], faulted.pressure[i, test],
                                       faulted.rh[i, test]], axis=1))
        pooled_train.append(histories[sid])
        pooled_stamps.extend(list(stamps))
    y_pred = np.array(pooled_pred, dtype=bool)
    y_true = np.array(pooled_true, dtype=bool)
    class_all = np.array(pooled_classes, dtype=object)
    test_levels = np.concatenate(pooled_levels)
    train_levels = np.concatenate(pooled_train)

    calm = class_all != "genuine_extreme"
    point = point_metrics(y_true & calm, y_pred)
    recalls, delays = [], []
    width = test.stop - test.start
    for s in range(len(ids)):
        seg_pred, seg_true = y_pred[s * width:(s + 1) * width], y_true[s * width:(s + 1) * width]
        out = event_recall(spans_from_labels(seg_true), seg_pred)
        recalls.append(float(out["event_recall"]))
        if not math.isnan(float(out["mean_detection_delay"])):
            delays.append(float(out["mean_detection_delay"]))

    rules_scores = np.concatenate([
        RulesBaseline(station_id=sid).score_series(
            list(stamps), np.stack([faulted.temp[faulted.station_index[sid], test],
                                    faulted.pressure[faulted.station_index[sid], test],
                                    faulted.rh[faulted.station_index[sid], test]], axis=1))
        for sid in ids
    ])
    _ = rules_scores  # per-station rule stream, pooled below via compare
    ladder = compare(
        {"rules": _PooledScores(rules_scores),
         "zscore": RobustZScore().fit(train_levels)},
        test_levels, y_true, class_all,
    )

    # The SIH headline example: 55 C + soaked + odd pressure, neighbours fine.
    station = ids[0]
    neighbours = [Observation(o, stamps[0], 31.0, 1004.0, 55.0) for o in ids[1:]]
    headline = pipe.process(
        Observation(station, stamps[0], 55.0, 985.0, 96.0), neighbours=neighbours)

    mean_recall = float(np.mean(recalls)) if recalls else 0.0
    mean_delay = float(np.mean(delays)) if delays else float("nan")
    summary: Dict[str, object] = {
        "train_epochs": fit_info["epochs"],
        "best_val_loss": round(float(fit_info["best_val_loss"]), 5),
        "point": {k: round(float(v), 4) for k, v in point.items()},
        "event_recall": round(mean_recall, 4),
        "mean_detection_delay": round(mean_delay, 2) if delays else None,
        "per_class_recall": {k: round(float(v), 4)
                             for k, v in per_class_recall(class_all, y_pred).items()},
        "genuine_extreme_far": round(float(genuine_extreme_far(class_all, y_pred)), 4),
        "ladder": [row.headline() for row in ladder],
        "headline_verdict": headline.explanation,
        "headline_status": headline.qc_status.value,
    }
    return summary


if __name__ == "__main__":
    for key, value in main().items():
        print(f"{key}: {value}")
