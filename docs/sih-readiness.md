# SkyGuard AI: SIH readiness audit

Problem statement 26073, AI/ML-Based Intelligent Anomaly Detection for Automatic Weather Stations.
Ministry of Earth Sciences / India Meteorological Department. Software, Disaster Management.

Audited 2026-09-10 by inspecting and running the code, not by reading the docs. Every claim below
was checked against a live run. Where something did not work, the failing output is quoted.

---

## 1. Verdict

The engine is strong and the evidence behind it is unusually solid for a prototype. Two things were
broken in ways that would have shown up in front of judges, and both are now fixed:

- **The live demo was hollow.** `serve.py` built a pipeline with no model, no thresholds, no spatial
  layer and no classifier. Opening the dashboard gave you an empty screen, and the problem
  statement's own 55 °C example scored `PASS, p=0.00`. Fixed: `serve.py` now trains the full stack,
  replays real data with injected faults, and comes up populated.
- **Synthetic weather was still the default story.** The repo now runs on real observations
  everywhere, and the synthetic weather generator is gone.

Against the eight grading criteria, weighted:

| Criterion | Weight | State | Note |
|---|---|---|---|
| Innovation and novelty | 25% | Strong | Six layers, not one model; the ablation proves each one earns its place |
| Detection accuracy | 20% | Good, honestly reported | Two tiers reported, not one flattering number |
| Real-time capability | 15% | Strong | 0.94 ms median per observation, one core, measured |
| Explainability | 10% | Strong | Shapley over the classifier, per-sensor errors, operator narrative |
| Scalability | 10% | Strong | O(1) state per station, measured throughput, per-station calibration |
| Practical deployability | 10% | Strong | One dependency, no build step, offline, SQLite audit |
| Visualization and UI | 5% | Strong | Two dashboards, both zero-build, both verified rendering |
| Energy efficiency | 5% | Partial | CPU-only argument is real; no ESP32 code exists yet |

---

## 2. What exists and works

Checked by running it.

### The engine

Six layers, orchestrated by `SkyGuardPipeline.process()` into one auditable record per observation.

| Layer | File | Verified |
|---|---|---|
| 1. Deterministic QC | `skyguard/qc/rules.py` | 33 tests, including negative controls: a 51 °C heatwave, a fog plateau and slow warming must all pass unflagged |
| 2. LSTM autoencoder | `skyguard/model/lstm.py` | Hand-written BPTT, gradient-checked against finite differences at 8.6e-12 relative error every run |
| 3. Multivariate | `skyguard/qc/multivariate.py` | Mahalanobis over levels and step deltas, Magnus dew-point guard, abstains on partial state |
| 4. Spatial | `skyguard/qc/spatial.py` | Comparability-weighted, lapse and barometric corrected, abstains below 2 neighbours |
| 5. Fusion | `skyguard/fusion/engine.py` | Trust-weighted, renormalised over speaking layers, genuine-weather damping |
| 6. Diagnosis | `skyguard/diagnostics/` | numpy random forest, exact or sampled Shapley, operator narrative |

Plus per-station adaptive thresholds, event aggregation, sensor health scoring, correction blending,
and an append-only SQLite audit store that refuses to overwrite a reading.

### Data

Real observations. `examples/data/real_aws_hourly.csv` holds 34,944 hourly readings for 8 west-India
stations from the Open-Meteo ERA5 archive, January to June 2024. Committed, so everything reproduces
offline. The cluster is deliberately dense, because spatial QC abstains below two comparable
neighbours and a scattered network would silence Layer 4.

Faults are injected on top, which is not a shortcut: the grading line says *"to be evaluated in
anomaly injected data"*, and no public AWS archive carries per-reading fault labels. Ten fault
classes, plus genuine extreme weather detected from the real series and labelled as the negative
control. This slice contains 3 real heatwaves, 545 labelled points that must not be flagged.

### Model results

From `python -m skyguard.eval.report`, 4,608 held-out observations, 8 stations, seed 7.

The ablation is the centrepiece. Same data, same seed, only the detector changes:

| Detector | F1 | Precision | False alarms | Real weather flagged | ROC AUC |
|---|---|---|---|---|---|
| Layer 1 rules only | 0.453 | 0.974 | 0.2% | 0.2% | 0.652 |
| Robust z-score | 0.367 | 0.243 | 66.7% | 70.5% | 0.594 |
| Isolation Forest | 0.364 | 0.236 | 74.6% | 73.4% | 0.588 |
| Local Outlier Factor | 0.357 | 0.221 | 93.8% | 91.6% | 0.605 |
| Plain autoencoder | 0.351 | 0.232 | 68.0% | 66.2% | 0.612 |
| Layer 2 LSTM-AE alone | 0.447 | 0.327 | 41.4% | 45.0% | 0.687 |
| Rules + LSTM-AE | 0.392 | 0.969 | 0.2% | 0.2% | 0.815 |
| + Layer 3 multivariate | 0.375 | 0.968 | 0.2% | 0.2% | 0.819 |
| + Layer 4 spatial | 0.387 | 0.969 | 0.2% | 0.2% | **0.848** |

The autoencoder alone calls 45% of real heatwave readings sensor faults. Every classical baseline is
worse. Rules underneath it take that to 0.2% while ranking quality climbs 0.687 to 0.848. That
argument only exists because the data has real weather in it.

Detection is reported at two tiers, because the pipeline genuinely has two:

| Question | Precision | Recall | Event recall | Real weather flagged |
|---|---|---|---|---|
| Hold this record back for review | 0.607 | 0.707 | 0.929 | 19.6% |
| Quarantine or replace this reading | 0.969 | 0.241 | 0.536 | 0.2% |

Also measured: ROC AUC 0.848, average precision 0.709 against a 0.223 baseline, calibration error
0.071, root cause named correctly on 77.4% of 248 detections, 0.94 ms median detection latency and
1.19 ms at the 95th percentile on one core.

### Dashboards

Both are single-file, zero-build, and both were rendered headless with no console errors.

- `dashboard/results.html` reads `results.js` and is the evidence report: reconstruction against
  real weather with fault spans shaded, loss curve, gradient gate, learned daily cycle, two-tier
  scorecard, ablation ladder, ROC and PR and reliability curves, score separation, root-cause
  confusion matrix, worked Shapley explanations, sensor health ledger, throughput, and a closing
  section on what the run does not prove.
- `dashboard/index.html` is the live console: station list, spatial map with neighbour links, four
  synchronised charts (T, P, RH, fused probability) with anomaly markers and dew-point trace,
  Shapley panel, health gauges, anomaly worklist.

### Tests

176 tests, one command, no extra tooling. `python tests/run_tests.py` passes clean.

---

## 3. What was broken, and what changed

### The live demo did not demonstrate the system

`serve.py` constructed `SkyGuardPipeline(stations=..., store=...)` and nothing else. Every learned
layer was therefore unavailable for the entire demo. Ingesting the problem statement's own example
returned:

```
status: PASS
fusion: {'probability': 0.0, 'band': 'NORMAL', 'is_anomaly': False}
recon available: False
spatial available: False
```

A judge following the README would have started the server, opened the dashboard, seen two demo
station pins and no data, and watched the flagship example pass as normal.

`serve.py` now loads the archive, trains the autoencoder, calibrates thresholds, fits the
multivariate model, trains the fault classifier, replays an injected split into the audit store, and
appends the headline case. The same input now returns:

```
headline PUNE reports 55.0 C / 985 hPa / 96% RH, neighbours read 31 C
         -> ESTIMATED, SUSPICIOUS p=0.76 | multivar (20%) via recon_rh @ pressure_hpa
         -> multivar (LOW): Variables jointly inconsistent: cross-check all three sensors
         -> suggested replacement: temp 24.6, pressure 977.3, rh 68.7
```

The console opens with 1,728 observations stored, 388 held back and 91 quarantined.

### Synthetic weather was still in the repo

`skyguard/data/simulator.py` generated synthetic clean weather that the demo and several tests ran
on. With a real archive committed there is no reason to judge the system on invented weather, and
the sim-to-real gap was the top risk in the submission draft. Deleted. Its array container moved to
`skyguard/data/network.py` as `StationNetwork`, which is what the injector and harness work on.

The fault injector stays, and that is deliberate. The grading criteria say the evaluation runs on
anomaly-injected data, and the expected-inputs line explicitly allows simulated anomalies. Injected
faults on real weather is exactly the combination the brief describes.

### Two execution paths did the same job

`examples/quickstart.py` simulated, trained, injected and scored. `python -m skyguard.eval.report`
does the same on real data, better, and writes the dashboard. Removed the first.

### Operator text broke on a Windows console

Rule details and fault narratives carried em dashes and degree signs. On the cp1252 console a judge
would run, they rendered as replacement characters, which reads like a data bug. Narrative strings
are ASCII now and `serve.py` forces UTF-8 output.

### Housekeeping

Nine `.pyc` files were tracked in git from before `.gitignore` existed. `dashboard/results.json` was
a 1.2 MB byte-for-byte duplicate of `results.js`, which is the file the page actually loads. Both
removed.

---

## 4. What is still missing

Ordered by how much a judge would notice.

1. **No ESP32 or edge code.** The problem statement suggests Edge AI for low-power deployment and
   energy efficiency carries 5%. Layer 1 is arithmetic that would port to a microcontroller, and the
   numpy-only choice was made partly for this, but no firmware exists. Present it as a designed
   Phase 2 with a named port target, not as done.
2. **Spatial recall is 0.365 on real data.** Real residual spread is wider than a simulator's, so
   `residual_sigma = 3.5` is tuned for the wrong distribution. Refitting it per network from clean
   data is the fix, and it is a config change rather than research.
3. **One seed, one split.** Everything reproduces exactly, which is not the same as a confidence
   interval. A short seed sweep would let the numbers carry a range.
4. **No streaming ingest demo.** The API accepts POSTs and the console polls, but nothing
   demonstrates a station pushing live data. A 30-line feeder script would make the "real-time"
   claim visible rather than asserted.
5. **Drift recall is 0.127.** This is by design and documented, since a slow enough bias is meant to
   look normal to a windowed model and gets caught by the health layer over days. The health layer
   has no lead-time measurement yet, so the compensating claim is unproven.
6. **Fault-class confusion on the subtle classes.** `multivar` is detected 94% of the time but named
   correctly only 4.7%, usually called `spike` or `noise`. Detection is what matters operationally,
   but the confusion matrix is on screen and a judge may ask.

---

## 5. How to demonstrate this to judges

Three commands, three distinct jobs. Rehearse in this order.

### Before the room

```bash
pip install numpy
python -m skyguard.eval.report      # ~2 min, regenerates dashboard/results.js
python serve.py                     # ~2 min to train, then serves on :8765
```

Have `dashboard/results.html` open in one browser tab and `http://127.0.0.1:8765/` in another, token
already pasted and connected. Never train in front of judges.

### The 5-minute run

**0:00 The hook.** One sentence: a weather station reports 55 °C while its neighbours read 31 °C.
A fixed-limit system either rejects every hot reading or accepts this one. Show the live console
with the worklist already populated.

**0:40 The failure it avoids.** Switch to the evidence report, section 1. The blue trace is what the
station reported, orange is what the model expected from the preceding hours alone. Point at a red
band where they separate, then at an amber band where they do not: that is a real April heatwave the
system left alone.

**1:40 The one number that matters.** Section 4, the ablation table. The autoencoder alone flags 45%
of real heatwave readings as faults. Every classical baseline is worse. Add deterministic rules and
it drops to 0.2% while ranking quality goes from 0.687 to 0.848. This is the argument for a layered
system and it is the whole pitch.

**2:40 Honesty as a feature.** Same table, point at rules-only beating the full stack on F1. Say it
before they find it: raw F1 is not the operational number, false alarms on real weather are, and the
system reports both tiers rather than picking the flattering one.

**3:20 Explainability.** Section 7, a worked case card. Fault class, severity, ranked Shapley
evidence, the operator action, the suggested replacement value. A meteorologist can read it.

**4:00 It is real.** Back to the live console. Click a worklist item. Show the station map, the
health gauges, the four synchronised charts. Mention 0.94 ms per observation on one core, one
dependency, no build step, works offline.

**4:40 Limits.** Section 10 of the report writes its own limitations from the run. Drift recall is
low by design, spatial needs neighbours, the weather is real but the faults are injected because no
public archive is labelled. Judges trust a team that says this first.

### Questions to expect

- *"Why not just an autoencoder?"* The ablation row is the answer: 45% false alarms on real weather.
- *"Is your data real?"* Real weather from a reanalysis archive at real station coordinates; the
  faults are injected because the grading runs on injected data and no archive is labelled per
  reading. Both facts are printed on the report page.
- *"Why is recall only 24%?"* That is the quarantine tier. The review tier catches 70.7% of faulty
  observations and 92.9% of fault events. Two questions, two answers, both on screen.
- *"Does it run on hardware?"* Not yet. One dependency and O(1) state per station were chosen for
  that path; Layer 1 is the port target.

### Do not

Do not run the benchmark live, do not show code, do not quote a number that is not on the dashboard,
and do not claim the ESP32 tier exists.

---

## 6. Reproducing this audit

```bash
python tests/run_tests.py                 # 176 tests
python -m skyguard.eval.report            # every number in section 2
python serve.py --fast                    # the live path, quicker to train
```

Numbers here came from a run on 2026-09-10 with the default arguments and seed 7. They go stale the
moment anything changes, so regenerate rather than trusting this file.
