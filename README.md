# SkyGuard AI

Explainable spatio-temporal quality assurance for Automatic Weather Station networks: real-time anomaly detection, root-cause diagnosis, sensor-health tracking, and self-healing data correction for temperature, pressure, and humidity streams. Built for the SIH problem statement on intelligent AWS anomaly detection.

## Run it (numpy is the only dependency)

```bash
pip install numpy
python tests/run_tests.py            # full suite: 176 tests, must all pass
python -m skyguard.eval.report       # benchmark on real weather -> dashboard/results.js
python serve.py                      # train, replay real faults, serve the live console
```

Three commands, three jobs. The second writes the evidence report: open
`dashboard/results.html` in a browser to see what the autoencoder learned, how each layer
contributes, where the decision boundary sits, and what the run does not prove. No server needed,
double-click works.

The third trains the stack, replays real observations with injected faults, and serves the live
operations console at http://127.0.0.1:8765/ with the worklist already populated. Paste the token
from `.env` and press Connect.

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

Around the core: per-station adaptive thresholds (`model/threshold.py`), event aggregation (`events/`), sensor health (`health/`), correction blending (`correction/`), audit store (`store.py`), streaming orchestration (`pipeline.py`), baselines + harness (`baselines/`, `eval/`), API + dashboards (`api/`, `dashboard/`).

Key commitments: raw data is immutable (corrections alongside, never in place); no global thresholds (per-station, from clean data); event-level verdicts, not point spam; every layer streams in O(1) memory.

## Measured on real weather

`examples/data/real_aws_hourly.csv` holds real hourly observations for 8 west-India stations from
the Open-Meteo ERA5 archive, January to June 2024. Real diurnal cycles, real synoptic systems, the
real April heatwaves. Faults are still injected, because no public AWS archive is labelled per
reading and the evaluation criterion is anomaly-injected data.

`python -m skyguard.eval.report` scores 4,608 held-out observations across 8 stations in about
2 minutes. The ablation ladder, all on the same data with the same seed:

| Detector | F1 | Precision | False alarms | Real weather flagged | ROC AUC |
|---|---|---|---|---|---|
| Layer 1 rules only | 0.453 | 0.974 | 0.2% | 0.2% | 0.652 |
| Robust z-score | 0.367 | 0.243 | 66.7% | 70.5% | 0.594 |
| Isolation Forest | 0.364 | 0.236 | 74.6% | 73.4% | 0.588 |
| Local Outlier Factor | 0.357 | 0.221 | 93.8% | 91.6% | 0.605 |
| Plain autoencoder | 0.351 | 0.232 | 68.0% | 66.2% | 0.612 |
| Layer 2 LSTM-AE only | 0.447 | 0.327 | 41.4% | 45.0% | 0.687 |
| Rules + LSTM-AE | 0.392 | 0.969 | 0.2% | 0.2% | 0.815 |
| + Layer 3 multivariate | 0.375 | 0.968 | 0.2% | 0.2% | 0.819 |
| + Layer 4 spatial | 0.387 | 0.969 | 0.2% | 0.2% | **0.848** |

The headline is that fourth-from-last row. The autoencoder alone flags 45% of real heatwave
observations as sensor faults, which is precisely the failure the problem statement warns about,
and every classical baseline is worse. Layering deterministic rules under it takes that to 0.2%
while ranking quality climbs from 0.687 to 0.848. That contrast only appears on data with real
weather in it.

The pipeline gives two answers and the report prints both, because one number cannot cover both:

| Question | Precision | Recall | Event recall | Real weather flagged |
|---|---|---|---|---|
| Hold this record back for review | 0.607 | 0.707 | 0.929 | 19.6% |
| Quarantine or replace this reading | 0.969 | 0.241 | 0.536 | 0.2% |

A missing-data run is the clean example: one dead sensor cannot outvote three quiet layers, so it
never scores as an anomaly, yet it is marked MISSING and never handed on as usable. Root cause is
named correctly on 77% of detections. Throughput is 0.93 ms median per observation for detection on
one core, 49 obs/s with Shapley explanations switched on for everything flagged.

Also honest: rules alone still beat the full stack on F1 (0.453 against 0.387). Every layer that
abstains pulls the fused mean down, and that is where the recall goes. Drift recall is 0.127 by
design, since a bias growing slowly enough is meant to look normal to a windowed model and gets
caught by the health layer over days instead.

### The headline example, live

`python serve.py` replays the problem statement's own case at the end of its run:

```
headline PUNE reports 55.0 C / 985 hPa / 96% RH, neighbours read 31 C
         -> ESTIMATED, SUSPICIOUS p=0.76 | multivar (20%) via recon_rh @ pressure_hpa
         -> multivar (LOW): Variables jointly inconsistent: cross-check all three sensors
         -> suggested replacement: temp 24.6, pressure 977.3, rh 68.7
```

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

For a whole network on a time grid, `skyguard.data.real.load_real_network` turns two CSVs (readings
plus station metadata) into the `StationNetwork` array form the injector and evaluation harness index
directly.

New station checklist: 1) collect clean history, 2) `fit_all_scalers` + train LSTM on it, 3) `fit_all_thresholds` on held-out clean errors, 4) `MultivariateQC.fit_station`, 5) register neighbours for spatial. Full runbook with fault playbook and a 5-minute judges' demo script: [`docs/use-case.md`](docs/use-case.md).

## Serve it

```python
from skyguard.api.server import LiveServer
server = LiveServer(pipe, stations, store=store, token="secret").start()  # loopback + bearer token
# POST /ingest  GET /stations  GET /stations/{id}/records  GET /anomalies  GET / (dashboard)
```

Set `SKYGUARD_API_TOKEN` instead of passing `token`, or just run `python serve.py`, which trains the
stack and brings the console up populated. `dashboard/index.html` is the live operations console
(map, charts, verdicts, worklist); `dashboard/results.html` is the static evidence report from the
last benchmark run.

## Repo map

```
skyguard/  pipeline.py  config.py  types.py  store.py
  data/         network.py  real.py  injector.py  loaders.py
  qc/           rules.py  multivariate.py  spatial.py
  model/        lstm.py  gradcheck.py  scaler.py  threshold.py
  fusion/       engine.py
  diagnostics/  forest.py  shapley.py  narrative.py
  eval/         metrics.py  harness.py  report.py
  events/  health/  correction/  baselines/  api/
dashboard/  index.html (live)  results.html (evidence)
examples/   fetch_real_data.py  data/*.csv
serve.py    tests/ (176)
docs/       use-case.md  sih-readiness.md
context/  goal, architecture, data, ml, evaluation, benchmark, decisions, conventions, progress, bug
```

## Limits (not hidden)

- The weather is real; the faults are not. ERA5 reanalysis carries real meteorology but no sensor pathology, which is why we inject faults to get ground truth. Its pressure sits at model terrain height, a few hPa from a real barometer, absorbed by per-station calibration.
- Slow drift is a health-layer signal over days, not a per-observation alarm. By design.
- Spatial QC abstains without comparable neighbours; sparse networks lose the strongest fault-versus-weather discriminator. On real data its residual spread is wider than the simulator's, and spatial recall (0.365) shows it.
- Short outages (under about 9 points) and short stuck spans reach WATCH, not anomaly. They are still held back from use.
- One seed, one split, one network per run. Vary `--seed` to see the spread. See `context/bug.md` for everything known-wrong and `docs/sih-readiness.md` for the full audit.
