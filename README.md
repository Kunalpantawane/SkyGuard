# SkyGuard AI

Explainable spatio-temporal quality assurance for Automatic Weather Station networks — real-time anomaly detection, root-cause diagnosis, sensor-health tracking, and self-healing data correction for temperature, pressure, and humidity streams. Built for the SIH problem statement on intelligent AWS anomaly detection.

## Run it (numpy is the only dependency)

```bash
pip install numpy
python tests/run_tests.py            # full suite: 120 tests, must all pass
python examples/quickstart.py        # end-to-end demo: simulate → train → stream → score
```

No torch, no npm, no build step. Works offline; runs on any CPU.

## How it works: six layers, not one model

| # | Layer | Module | Answers |
|---|---|---|---|
| 1 | Deterministic QC | `skyguard/qc/rules.py` | Physically possible? Stuck? Impossible jump? |
| 2 | Temporal AI | `skyguard/model/lstm.py` | Does this sequence match learned normal? |
| 3 | Multivariate | `skyguard/qc/multivariate.py` | Do T, P, RH make sense *together*? |
| 4 | Spatial | `skyguard/qc/spatial.py` | Do comparable neighbours agree? |
| 5 | Fusion | `skyguard/fusion/engine.py` | Combined calibrated anomaly probability |
| 6 | Diagnosis | `skyguard/diagnostics/` | Which fault class, how sure, what to do? |

Around the core: per-station adaptive thresholds (`model/threshold.py`), event aggregation (`events/`), sensor health (`health/`), correction blending (`correction/`), audit store (`store.py`), streaming orchestration (`pipeline.py`), baselines + harness (`baselines/`, `eval/`), API + dashboard (`api/`, `dashboard/`).

Key commitments: raw data is immutable (corrections alongside, never in place); no global thresholds (per-station, from clean data); event-level verdicts, not point spam; every layer streams in O(1) memory.

## What the demo actually measured

`python examples/quickstart.py` (3 stations, 40 days hourly, injected faults, seeds fixed — rerun it, the numbers below must match):

- Hybrid pipeline: precision **0.91**, recall **0.44**, event recall **0.87** @ 3.2-step delay, false-alarm rate **0.029**, genuine-weather FAR **0.024**
- Rules only: F1 0.77, event recall 0.82, genuine FAR 0.017
- z-score: F1 0.65, false-alarm rate **0.45**, genuine FAR **0.42** — the naive detector cries wolf on real heatwaves
- Per-class recall: corrupt 1.0, spike 0.73, noise 0.64, drop 0.58, stuck 0.42, missing 0.24
- SIH headline example (55 °C, soaked, odd pressure, neighbours fine): `SUSPICIOUS p=0.74`, quarantined with an estimated correction

Read honestly: rules win F1 on small slices; the hybrid wins precision and weather discrimination; short outages and short stuck spans are the known-weak classes (see `context/bug.md`).

## Use it on your data

```python
from skyguard.data.loaders import load_csv            # GHCNh / Meteostat / generic CSV
from skyguard.pipeline import SkyGuardPipeline
from skyguard.types import Station

obs = load_csv("my_station.csv", station_id="S1")     # -> List[Observation]
pipe = SkyGuardPipeline(stations={"S1": Station("S1", "Site", 18.5, 73.9, 200.0, "valley")})
for o in obs:
    record = pipe.process(o)                          # -> QCRecord (verdict + diagnosis + correction)
    print(record.qc_status.value, record.explanation)
```

New station checklist: 1) collect clean history, 2) `fit_all_scalers` + train LSTM on it, 3) `fit_all_thresholds` on held-out clean errors, 4) `MultivariateQC.fit_station`, 5) register neighbours for spatial. Full runbook with fault playbook and a 5-minute judges' demo script: [`docs/use-case.md`](docs/use-case.md).

## Serve it

```python
from skyguard.api.server import LiveServer
server = LiveServer(pipe, stations, store=store, token="secret").start()  # loopback + bearer token
# POST /ingest  GET /stations  GET /stations/{id}/records  GET /anomalies  GET / (dashboard)
```

Set `SKYGUARD_API_TOKEN` instead of passing `token`. Open `dashboard/index.html` for the single-file UI (charts, verdict, evidence, worklist).

## Repo map

```
skyguard/  pipeline.py  config.py  types.py  store.py
  data/         simulator.py  injector.py  loaders.py
  qc/           rules.py  multivariate.py  spatial.py
  model/        lstm.py  gradcheck.py  scaler.py  threshold.py
  fusion/       engine.py
  diagnostics/  forest.py  shapley.py  narrative.py
  events/  health/  correction/  baselines/  eval/  api/
dashboard/index.html        examples/quickstart.py        tests/ (120)
context/  goal, architecture, data, ml, evaluation, decisions, conventions, progress, bug
```

## Limits (not hidden)

- Trained on simulated normality; real AWS data needs recalibration (thresholds are per-station, so a fit — not a rewrite).
- Slow drift is a health-layer signal over days, not a per-observation alarm — by design.
- Spatial QC abstains without comparable neighbours; sparse networks lose the strongest fault/weather discriminator.
- Short outages (<~9 pts) and short stuck spans reach WATCH, not anomaly. See `context/bug.md` for everything known-wrong.
