"""Live demo: a trained network, real weather, real faults, on a real console.

    python serve.py            # train, replay, then serve on :8765
    python serve.py 9000       # a different port
    python serve.py --fast     # smaller model, quicker to get on screen

What it does before the server starts listening:

  1. Loads the committed real archive (8 west-India stations, hourly).
  2. Trains the LSTM autoencoder on the clean early part of it, calibrates a
     threshold per station, fits the multivariate model, trains the fault
     classifier on injected validation data.
  3. Injects labelled faults into the recent part and streams it through the
     full six-layer pipeline into `./audit.db`, so the dashboard opens with a
     populated worklist rather than an empty screen.
  4. Replays the headline case from the problem statement: one station reporting
     55 C with soaked humidity while its neighbours read normal.

Then it serves the API and `dashboard/index.html` at the printed URL. The bearer
token comes from `.env` (`SKYGUARD_API_TOKEN`); paste it into the dashboard's
token box and press Connect. Ctrl+C stops.

Earlier this script built `SkyGuardPipeline(stations=..., store=...)` and nothing
else, which left Layers 2, 4 and 6 permanently unavailable: the console came up
empty and the 55 C headline case scored PASS at p=0.00. A demo has to run the
same fitted stack the benchmark scores, or it is demonstrating something else.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from skyguard.api.server import LiveServer          # noqa: E402
from skyguard.config import InjectorConfig          # noqa: E402
from skyguard.data.injector import inject_faults    # noqa: E402
from skyguard.eval.report import (                  # noqa: E402
    REAL_OBS_CSV,
    build_dataset,
    stream_split,
    train_classifier,
    train_model,
)
from skyguard.pipeline import SkyGuardPipeline      # noqa: E402
from skyguard.qc.spatial import SpatialQC           # noqa: E402
from skyguard.store import AuditStore               # noqa: E402
from skyguard.types import Observation              # noqa: E402


class _Setup:
    """Training knobs. Small on purpose: this is a demo, not the benchmark."""

    days = 90
    window = 12
    hidden = 16
    epochs = 20
    batch_size = 32
    learning_rate = 0.01
    stride = 3
    seed = 7
    verbose = False


def _fast(setup: _Setup) -> _Setup:
    setup.days, setup.epochs, setup.hidden, setup.stride = 45, 8, 12, 4
    return setup


def build_live_pipeline(setup: _Setup):
    """Train everything, then return the fitted pipeline and its stations."""
    if not REAL_OBS_CSV.exists():
        raise SystemExit(
            f"missing {REAL_OBS_CSV.relative_to(ROOT)}\n"
            "run: python examples/fetch_real_data.py"
        )

    net, label = build_dataset(setup.days)
    train, val, test = net.split_indices()
    print(f"data    {label}")
    print(f"        {net.n_stations} stations, {net.n_steps} hourly steps")

    print("train   fitting the autoencoder on clean weather ...")
    fitted = train_model(net, train, val, setup)
    info = fitted["fit_info"]
    print(f"        {info['epochs']} epochs, best val loss {info['best_val_loss']:.5f}, "
          f"{fitted['train_seconds']:.1f}s")

    print("train   fitting the fault classifier on injected validation data ...")
    classifier, _ = train_classifier(net, val, fitted, setup)

    stations = {s.station_id: s for s in net.stations}
    store = AuditStore(_fresh_demo_db())
    pipe = SkyGuardPipeline(
        stations=stations,
        lstm=fitted["lstm"], scalers=fitted["scalers"], thresholds=fitted["thresholds"],
        multivariate=fitted["multivariate"], spatial=SpatialQC(stations),
        classifier=classifier, store=store,
    )
    return pipe, stations, store, net, test


def _fresh_demo_db() -> Path:
    """Start each demo from an empty database.

    The audit store is append-only on purpose: re-inserting a station and
    timestamp raises rather than overwriting, which is what keeps a raw reading
    recoverable. That guarantee also means a second demo run against a database
    still holding the first run's replay fails on the first duplicate. The
    database is a local demo artifact and gitignored, so it is cleared here
    rather than worked around.
    """
    path = ROOT / "audit.db"
    for suffix in ("", "-wal", "-shm"):
        stale = path.with_name(path.name + suffix)
        if stale.exists():
            stale.unlink()
    return path


def replay_recent(pipe, net, test, seed: int) -> None:
    """Stream the injected test split so the console opens with real content."""
    print("replay  injecting labelled faults into the recent split ...")
    injected = inject_faults(net, InjectorConfig(seed=seed + 100),
                             index_range=(test.start, test.stop - 1))
    records, _ = stream_split(pipe, injected.network, test, explain_anomalies=True)
    flagged = sum(1 for r in records if r.fusion.is_anomaly)
    held = sum(1 for r in records if r.qc_status.value != "PASS")
    print(f"        {len(records)} observations stored, {held} held back, "
          f"{flagged} quarantined")


def replay_headline(pipe, net) -> None:
    """The problem statement's own example, one interval after the replay ends.

    The station has a full window of real history behind it by then, which is
    what lets Layer 2 have an opinion at all. Fired at a cold station this would
    only prove that the rules layer works.
    """
    ids = [s.station_id for s in net.stations]
    target, neighbours = ids[0], ids[1:4]
    stamp = net.timestamps[-1] + timedelta(minutes=net.interval_minutes)

    others = [Observation(n, stamp, 31.0, 1004.0, 55.0) for n in neighbours]
    record = pipe.process(Observation(target, stamp, 55.0, 985.0, 96.0),
                          neighbours=others, explain=True)
    print(f"headline {target} reports 55.0 C / 985 hPa / 96% RH, neighbours read 31 C")
    print(f"         -> {record.qc_status.value}, {record.explanation}")
    if record.diagnosis.fault_class.value != "none":
        print(f"         -> {record.diagnosis.fault_class.value} "
              f"({record.diagnosis.severity.value}): {record.diagnosis.recommended_action}")
    if record.correction.available:
        fixed = ", ".join(f"{k.split('_')[0]} {v:.1f}"
                          for k, v in record.correction.values.items())
        print(f"         -> suggested replacement: {fixed}")


def main(argv) -> None:
    # Rule details carry the degree sign and station names can too. The Windows
    # console defaults to cp1252 and renders those as replacement characters,
    # which looks like a data bug during a demo. Ask for UTF-8 instead.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):    # already UTF-8, or not a real tty
            pass

    port = 8765
    setup = _Setup()
    for arg in argv:
        if arg == "--fast":
            setup = _fast(setup)
        elif arg.isdigit():
            port = int(arg)

    pipe, stations, store, net, test = build_live_pipeline(setup)
    replay_recent(pipe, net, test, setup.seed)
    replay_headline(pipe, net)

    server = LiveServer(pipe, stations, store=store, port=port)
    print()
    print(f"SkyGuard live at {server.url}   (dashboard at {server.url}/)")
    print("token: " + ("from .env SKYGUARD_API_TOKEN" if server.token
                       else "NONE SET, export SKYGUARD_API_TOKEN first"))
    print("Ctrl+C to stop.")
    try:
        server.server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        store.close()


if __name__ == "__main__":
    main(sys.argv[1:])
