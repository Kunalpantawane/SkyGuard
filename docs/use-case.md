# SkyGuard AI — use cases and operator runbook

How to perform every job on this system: run it, feed it, read it, act on it, and demo it to judges. Numbers below are the verified `examples/quickstart.py` output — rerun it if you doubt any of them.

## 1. Perform: the runbook

**A. Prove it works (2 min).** `pip install numpy`, then `python tests/run_tests.py` (120 green) and `python examples/quickstart.py` (expect: hybrid precision 0.91, event recall 0.87, genuine-weather FAR 0.02; zscore FAR 0.45 as the contrast).

**B. Feed your own CSV.** `load_csv(path, station_id=...)` auto-detects timestamp/station/T/P/RH columns (Fahrenheit, sea-level pressure and dew-point archives are converted; pass `column_map` for awkward headers). Stream each `Observation` through `SkyGuardPipeline.process()` — one call returns the full `QCRecord`.

**C. Commission a new station (do this once per site).** Collect clean history, then: `fit_all_scalers` → train the LSTM on train-split windows → `fit_all_thresholds` on held-out clean errors (never training data) → `MultivariateQC.fit_station` (needs 100+ points) → register 2+ comparable neighbours (similar elevation) for spatial. Thin stations borrow the network threshold and are flagged as doing so.

**D. Serve and watch.** `LiveServer(pipe, stations, store=store, token=...)` (or `SKYGUARD_API_TOKEN`); POST `/ingest`, read `/stations/{id}/records`, work `/anomalies`. Open `dashboard/index.html`, paste the token, pick a station: T/P/RH charts with red anomaly markers, verdict panel, evidence list.

**E. Read a verdict.** Band tells urgency (`NORMAL < WATCH < SUSPICIOUS < HIGH < CRITICAL`, flag at mid-WATCH); `fault_class` + `confidence` tell what and how sure; `evidence` lines tell why (ranked, with observed values); `recommended_action` tells what to do; `correction` gives an estimated replacement with its own confidence — the raw value is never overwritten. `genuine_weather_damped` means context contradicted the alarm: trust but verify.

## 2. Fault playbook (what you see → what you do)

| Signal | Meaning | Action |
|---|---|---|
| `SPIKE`/`DROP`, 90 %+ | Intermittent contact, power dip | Inspect wiring; quarantine point; use estimate |
| `STUCK` | Frozen output past 12–24 repeats | Check ventilation/power; schedule visit (fog can fake RH flatlines — the layer knows) |
| `DRIFT` + falling health | Growing bias, pre-failure | Recalibrate now; health lead time is the early warning |
| `STEP` | Permanent shift (swap/reposition event) | Confirm on site, then recalibrate baseline |
| `NOISE` | Shielding/grounding/power issue | Check grounding and supply stability |
| `MISSING` | Telemetry/logger outage | Check link, power, logger; short gaps stay WATCH, 10+ points flag |
| `CORRUPT` (999/-999) | Telemetry corruption, always hard | Fix the chain; never average these into products |
| `MULTIVARIATE` | One sensor disagreeing with the other two | Cross-check all three; likely one drifted subtly |
| `SPATIAL` | Station vs neighbourhood | Verify site conditions, then inspect |
| `GENUINE_EXTREME` | Real weather, confirmed by context | Trust it. No maintenance. This verdict is the product's headline feature. |

## 3. Use cases

1. **The SIH example.** 55 °C + 96 % RH + odd pressure while neighbours read normal → `SUSPICIOUS p=0.74`, quarantined, correction estimated, wiring inspection recommended. (Reproduced live by the quickstart's final step.)
2. **Heatwave vs broken heater.** All stations rise together → spatial agrees, guard damps, no alarm. One station spikes → convicted. Demo measured genuine-weather FAR 0.02 vs 0.42 for threshold logic.
3. **Monsoon onset.** Step-like cooling + soaking across the network with coherent covariates → trusted as weather; the injector is explicitly barred from scoring regime faults inside such spans.
4. **Stuck humidity probe.** RH flatlines for days → persistence + health trend fire; fog plateaus are exempted by the saturation rule, so this alarm means something.
5. **Comms outage.** NaN stream → MISSING verdicts, gap-aware (no fake rate alarms across the hole), refusal to "correct" long outages — an honest gap beats confident fiction.
6. **Slow thermometer death.** Weeks of creeping bias → health score slides HEALTHY → WARNING → DEGRADING with reasons attached, before any hard fault. Maintenance scheduled, not scrambled.
7. **New station, week one.** Rules work from observation one; temporal/spatial join as windows fill and neighbours register; thresholds borrow network levels until history suffices. No cold-start silence, no cold-start spam.
8. **Forecaster's clean feed.** Downstream models consume `ESTIMATED` values with confidence attached and full audit (`raw + flags + scores + reason` per record, append-only SQLite) — reproducible, defensible, never silently edited.

## 4. The 5-minute judges' demo (maps to the weights)

1. **(0:00) Problem + headline replay** — run the quickstart's SIH example live; show quarantine + explanation (Innovation, Deployability).
2. **(1:00) Heatwave honesty** — shared heatwave stays quiet while a lone spike fires; quote genuine FAR 0.02 vs 0.42 (Accuracy, Explainability).
3. **(2:00) Streaming proof** — `process()` one observation, show `latency_ms`; bounded state, O(1)/point, numpy-only CPU (Real-time, Scalability, Energy).
4. **(3:00) Diagnosis + action** — open a verdict: fault class, confidence, evidence lines, recommended action, correction with confidence (Explainability).
5. **(4:00) Health + dashboard** — declining sensor score with reasons, anomaly worklist, charts with markers (Visualization, Practical use).
6. **(4:30) Honesty slide** — weak classes (short stuck/missing), rules-beat-hybrid-F1 on small slices, all in `context/bug.md`. Judges trust teams that know their failure modes.

## 5. Performance and limits, plainly

Throughput is single-core observations/sec with per-record latency printed on every `QCRecord`; memory per station is bounded (window + capped buffers). What it will not do: catch sub-hour faults below rate limits without corroboration, diagnose without a fitted classifier (verdicts still work), or replace recalibration — it tells you *when*, with evidence. Full limits list: `context/bug.md`.
