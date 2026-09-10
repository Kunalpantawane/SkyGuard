"""Controlled fault injection with per-point ground truth.

The evaluation criterion is explicit: the system is judged on anomaly-injected
data. So injection is not a testing convenience, it is a core research component
and needs to be as principled as the detector.

Two rules govern every injector below:

1. Faults are injected on top of clean data, and the clean original is preserved,
   so the correction stage can be scored against the truth it should have
   recovered.
2. Each fault type has a distinct *signature*, not just a distinct magnitude.
   Injecting "a big number" for every class would make fault classification
   trivial and the reported accuracy meaningless.

The subtlest injector is `multivar`: it moves one variable to a value that is
individually plausible and leaves the others untouched, breaking only the
*relationship*. A range check cannot see it; only a joint-state model can.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import VARIABLES, InjectorConfig
from ..types import FaultClass, InjectedFault
from .network import StationNetwork, dewpoint_from_rh


class InjectionResult:
    """A faulted copy of a network plus the ground truth needed to score it."""

    def __init__(
        self,
        network: StationNetwork,
        clean: StationNetwork,
        faults: List[InjectedFault],
    ) -> None:
        self.network = network      # faulted data — what the detector sees
        self.clean = clean          # pristine original — the correction target
        self.faults = faults

    # -- point-level truth -------------------------------------------------

    def label_matrix(self) -> np.ndarray:
        """(n_stations, n_steps) boolean: is this point a genuine sensor fault?

        GENUINE_EXTREME spans are excluded: they are real weather and must be
        counted as normal. Treating them as positives would reward exactly the
        false-alarm behaviour we are trying to eliminate.
        """
        labels = np.zeros((self.network.n_stations, self.network.n_steps), dtype=bool)
        for f in self.faults:
            if f.fault_class == FaultClass.GENUINE_EXTREME:
                continue
            i = self.network.station_index[f.station_id]
            labels[i, f.start_index : f.end_index + 1] = True
        return labels

    def class_matrix(self) -> np.ndarray:
        """(n_stations, n_steps) object array of fault-class strings."""
        classes = np.full(
            (self.network.n_stations, self.network.n_steps),
            FaultClass.NONE.value,
            dtype=object,
        )
        for f in self.faults:
            i = self.network.station_index[f.station_id]
            classes[i, f.start_index : f.end_index + 1] = f.fault_class.value
        return classes

    def faults_for_station(self, station_id: str) -> List[InjectedFault]:
        return [f for f in self.faults if f.station_id == station_id]

    def summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for f in self.faults:
            counts[f.fault_class.value] = counts.get(f.fault_class.value, 0) + 1
        return dict(sorted(counts.items()))


# --------------------------------------------------------------------------
# Individual fault injectors
# --------------------------------------------------------------------------
# Each takes the mutable series and returns the affected index span. They work on
# a single station/variable slice so the caller controls placement policy.


def _inject_spike(
    rng: np.random.Generator, series: np.ndarray, idx: int, sigma: float, cfg: InjectorConfig
) -> Tuple[int, int, float]:
    """A brief extreme excursion, 1-3 points wide.

    Scaled by the series' own local standard deviation rather than an absolute
    value: a 15 °C spike is dramatic for temperature and invisible for pressure,
    so a fixed offset would produce wildly uneven difficulty across variables.
    """
    width = int(rng.integers(1, 4))
    end = min(idx + width - 1, len(series) - 1)
    mag = rng.uniform(*cfg.spike_magnitude_sigma) * sigma
    series[idx : end + 1] += mag
    return idx, end, mag


def _inject_drop(
    rng: np.random.Generator, series: np.ndarray, idx: int, sigma: float, cfg: InjectorConfig
) -> Tuple[int, int, float]:
    """A downward spike. Kept as a separate class because sensor physics differ:
    a short-circuit reads low, an open circuit reads high, and knowing which
    guides the maintenance action."""
    width = int(rng.integers(1, 4))
    end = min(idx + width - 1, len(series) - 1)
    mag = rng.uniform(*cfg.spike_magnitude_sigma) * sigma
    series[idx : end + 1] -= mag
    return idx, end, -mag


def _inject_stuck(
    rng: np.random.Generator, series: np.ndarray, idx: int, sigma: float, cfg: InjectorConfig
) -> Tuple[int, int, float]:
    """Frozen output: the last good value repeats.

    Note the value is held *exactly*, with no noise. That is the real signature —
    a stuck ADC or frozen firmware buffer returns bit-identical values, which is
    why run-length detection works so well on it.
    """
    dur = int(rng.integers(*cfg.stuck_duration_points))
    end = min(idx + dur - 1, len(series) - 1)
    series[idx : end + 1] = series[idx]
    return idx, end, 0.0


def _inject_drift(
    rng: np.random.Generator, series: np.ndarray, idx: int, sigma: float, cfg: InjectorConfig
) -> Tuple[int, int, float]:
    """Calibration drift: a slowly growing bias.

    Deliberately hard. A windowed autoencoder should *not* catch a slow drift
    within a single window, because within any 24-point window the data looks
    normal. Catching it is the health/degradation layer's job, and this injector
    is what proves that layer earns its place.
    """
    dur = int(rng.integers(*cfg.drift_duration_points))
    end = min(idx + dur - 1, len(series) - 1)
    total = rng.uniform(*cfg.drift_magnitude) * sigma
    sign = 1.0 if rng.random() < 0.7 else -1.0   # drift is usually upward-biased
    n = end - idx + 1
    series[idx : end + 1] += sign * total * np.linspace(0, 1, n)
    return idx, end, sign * total


def _inject_step(
    rng: np.random.Generator, series: np.ndarray, idx: int, sigma: float, cfg: InjectorConfig
) -> Tuple[int, int, float]:
    """A permanent level shift, e.g. after a botched recalibration.

    The defining feature is that it does not recover, so the span is long — but
    bounded rather than running to the end of the series. An unbounded step would
    occupy an entire station for the rest of the record, making it impossible to
    place other faults there or to inject more than one step per station.

    The shift is instantaneous at onset (unlike drift) and constant thereafter,
    which is the signature that separates the two classes.
    """
    dur = int(rng.integers(*cfg.step_duration_points))
    end = min(idx + dur - 1, len(series) - 1)
    mag = rng.uniform(*cfg.step_magnitude) * sigma
    sign = 1.0 if rng.random() < 0.5 else -1.0
    series[idx : end + 1] += sign * mag
    return idx, end, sign * mag


def _inject_noise(
    rng: np.random.Generator, series: np.ndarray, idx: int, sigma: float, cfg: InjectorConfig
) -> Tuple[int, int, float]:
    """Degraded signal integrity: variance explodes, mean stays put.

    A mean-preserving fault. Any detector that only watches levels will miss it
    entirely, which makes it a good test of whether the AE learned dynamics or
    just ranges.
    """
    dur = int(rng.integers(12, 96))
    end = min(idx + dur - 1, len(series) - 1)
    mult = rng.uniform(*cfg.noise_multiplier)
    n = end - idx + 1
    series[idx : end + 1] += rng.normal(0.0, sigma * mult, n)
    return idx, end, mult


def _inject_missing(
    rng: np.random.Generator, series: np.ndarray, idx: int, sigma: float, cfg: InjectorConfig
) -> Tuple[int, int, float]:
    """Communication outage: NaN, meaning genuinely absent."""
    dur = int(rng.integers(*cfg.missing_duration_points))
    end = min(idx + dur - 1, len(series) - 1)
    series[idx : end + 1] = np.nan
    return idx, end, 0.0


def _inject_corrupt(
    rng: np.random.Generator, series: np.ndarray, idx: int, sigma: float, cfg: InjectorConfig
) -> Tuple[int, int, float]:
    """Telemetry corruption: a sentinel or impossible encoded value.

    Distinct from missing on purpose. A sentinel that reaches a forecast model as
    a real number is far more dangerous than a gap, and the two demand different
    fixes (bad parser versus bad link).
    """
    dur = int(rng.integers(1, 6))
    end = min(idx + dur - 1, len(series) - 1)
    sentinel = float(rng.choice([999.0, -999.0, 9999.0, -9999.0]))
    series[idx : end + 1] = sentinel
    return idx, end, sentinel


# --------------------------------------------------------------------------
# Multivariate and spatial injectors (need cross-variable / cross-station view)
# --------------------------------------------------------------------------

def _inject_multivariate(
    rng: np.random.Generator,
    net: StationNetwork,
    s_idx: int,
    idx: int,
) -> Tuple[int, int, str, float]:
    """Break the physical relationship while keeping each value plausible.

    The hardest and most important case. Example: hold temperature and pressure,
    push RH to 97 % on a hot dry afternoon. Every individual range check passes.
    Only a model of the joint state — or the dew-point identity — can see that
    this combination cannot physically occur at this station.

    Implemented by driving RH toward saturation while temperature stays high, or
    by shifting temperature so far from the dew point that the reported RH
    becomes impossible for that moisture content.
    """
    dur = int(rng.integers(3, 24))
    end = min(idx + dur - 1, net.n_steps - 1)
    n = end - idx + 1

    if rng.random() < 0.5:
        # RH inconsistent with T: force near-saturation while T stays warm.
        target = rng.uniform(93.0, 99.5)
        net.rh[s_idx, idx : end + 1] = target + rng.normal(0.0, 0.6, n)
        return idx, end, "rh_pct", target

    # T inconsistent with the moisture state: warm the air sharply without any
    # change in RH, which implies a dew point that never actually moved.
    offset = rng.uniform(6.0, 12.0)
    net.temp[s_idx, idx : end + 1] += offset
    return idx, end, "temp_c", offset


def _inject_spatial(
    rng: np.random.Generator,
    net: StationNetwork,
    s_idx: int,
    idx: int,
    cfg: InjectorConfig,
    sigma: float,
) -> Tuple[int, int, str, float]:
    """One station diverges while its neighbours stay normal.

    This is the problem statement's own example: a station reporting 55 °C while
    the neighbourhood reports 31 °C. The offset is sustained and moderate — small
    enough that temporal QC alone might accept it, so only the spatial layer
    reliably resolves it.
    """
    dur = int(rng.integers(*cfg.spatial_duration_points))
    end = min(idx + dur - 1, net.n_steps - 1)
    n = end - idx + 1
    variable = str(rng.choice(["temp_c", "pressure_hpa", "rh_pct"], p=[0.6, 0.2, 0.2]))
    mag = rng.uniform(*cfg.spatial_offset_sigma) * sigma
    sign = 1.0 if rng.random() < 0.6 else -1.0
    # Smooth ramp in and out: an instantaneous jump would be caught by the rate
    # check, which would mask whether the spatial layer works at all.
    shape = np.ones(n)
    ramp = max(1, n // 6)
    shape[:ramp] = np.linspace(0, 1, ramp)
    shape[-ramp:] = np.linspace(1, 0, ramp)

    arr = {"temp_c": net.temp, "pressure_hpa": net.pressure, "rh_pct": net.rh}[variable]
    arr[s_idx, idx : end + 1] += sign * mag * shape
    return idx, end, variable, sign * mag


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

_POINT_INJECTORS = {
    FaultClass.SPIKE: _inject_spike,
    FaultClass.DROP: _inject_drop,
    FaultClass.STUCK: _inject_stuck,
    FaultClass.DRIFT: _inject_drift,
    FaultClass.STEP: _inject_step,
    FaultClass.NOISE: _inject_noise,
    FaultClass.MISSING: _inject_missing,
    FaultClass.CORRUPT: _inject_corrupt,
}

# Relative frequencies. Spikes and dropouts dominate real AWS fault logs; steps
# are rare because they need a maintenance error. Weighting by realism rather
# than uniformly keeps the measured performance honest about what actually
# happens in the field.
_FAULT_WEIGHTS: Dict[FaultClass, float] = {
    FaultClass.SPIKE: 0.18,
    FaultClass.DROP: 0.10,
    FaultClass.STUCK: 0.14,
    FaultClass.DRIFT: 0.10,
    FaultClass.STEP: 0.05,
    FaultClass.NOISE: 0.09,
    FaultClass.MISSING: 0.13,
    FaultClass.CORRUPT: 0.07,
    FaultClass.MULTIVARIATE: 0.08,
    FaultClass.SPATIAL: 0.06,
}


def _robust_sigma(series: np.ndarray) -> float:
    """Scale estimate from the first difference, via MAD.

    Differencing first removes the diurnal and seasonal cycle, so this measures
    genuine short-term variability rather than the amplitude of the daily swing.
    Using the raw standard deviation would overstate sigma several-fold and make
    every injected fault enormous.
    """
    d = np.diff(series[~np.isnan(series)])
    if d.size < 8:
        return 1.0
    mad = np.median(np.abs(d - np.median(d)))
    sigma = 1.4826 * mad
    return float(max(sigma, 1e-3))


def inject_faults(
    clean: StationNetwork,
    cfg: InjectorConfig | None = None,
    index_range: Optional[Tuple[int, int]] = None,
) -> InjectionResult:
    """Inject a realistic mixture of faults into a copy of `clean`.

    `index_range` restricts injection to a time slice, so faults can be confined
    to the test split and kept out of the data the autoencoder trains on. Without
    that separation the model would learn the faults as normal.
    """
    cfg = cfg or InjectorConfig()
    rng = np.random.default_rng(cfg.seed)

    net = clean.copy()
    lo, hi = index_range or (0, net.n_steps - 1)
    span = hi - lo + 1
    if span < 64:
        raise ValueError("injection range too short to place faults meaningfully")

    faults: List[InjectedFault] = []

    # Record the network's genuine extreme-weather events as ground truth so
    # the harness can measure false alarms on real weather.
    #
    # Events carry half-open windows [start, end); InjectedFault's
    # end_index is inclusive. Converting requires end - 1. Skipping that step
    # labels one extra point after every event as genuine weather, which both
    # inflates the false-alarm denominator and blocks a fault from being placed
    # immediately after a heatwave.
    for ev in net.genuine_events:
        last = int(ev["end"]) - 1
        if last < lo or ev["start"] > hi:
            continue
        for station in net.stations:
            faults.append(
                InjectedFault(
                    station_id=station.station_id,
                    start_index=max(int(ev["start"]), lo),
                    end_index=min(last, hi),
                    fault_class=FaultClass.GENUINE_EXTREME,
                    variable=None,
                    magnitude=float(ev.get("peak_c", ev.get("depth_hpa", 0.0)) or 0.0),
                )
            )

    # Budget: how many anomalous points to place in total.
    total_points = net.n_stations * span
    budget = int(total_points * cfg.target_anomaly_rate)

    classes = list(_FAULT_WEIGHTS.keys())
    probs = np.array([_FAULT_WEIGHTS[c] for c in classes], dtype=float)
    probs /= probs.sum()

    # Per-station occupancy, so two faults do not overlap and produce an
    # ambiguous label. Overlapping injections would make per-class recall
    # uninterpretable.
    occupied: Dict[int, np.ndarray] = {
        i: np.zeros(net.n_steps, dtype=bool) for i in range(net.n_stations)
    }

    # Genuine extreme-weather spans occupy a second mask. Point-signature
    # faults (spike, stuck, noise, corrupt, missing) stay placeable anywhere:
    # real weather cannot flatline a sensor or emit 9999, so those labels stay
    # crisp even inside a heatwave. Regime faults (step, drift, multivar,
    # spatial) are barred from genuine spans — a step during monsoon onset is
    # genuinely ambiguous ground truth, and scoring it would punish correct
    # detectors for the injector's sins.
    regime_occupied: Dict[int, np.ndarray] = {
        i: np.zeros(net.n_steps, dtype=bool) for i in range(net.n_stations)
    }
    for ev in net.genuine_events:
        # Half-open [start, end) on the event, so the inclusive last index
        # is end - 1; see the GENUINE_EXTREME truth block above.
        s, e = max(int(ev["start"]), lo), min(int(ev["end"]) - 1, hi)
        if e >= s:
            for i in range(net.n_stations):
                regime_occupied[i][s : e + 1] = True

    arrays = {"temp_c": net.temp, "pressure_hpa": net.pressure, "rh_pct": net.rh}

    # Two-phase placement.
    #
    # Phase 1 guarantees a minimum number of spans per class, because per-class
    # recall computed from one or two spans is noise. Purely budget-driven
    # placement starves the rare classes: a single `step` fault runs to the end of
    # the series and can consume the entire point budget by itself.
    #
    # Phase 2 then fills any remaining budget by realistic frequency weights.
    min_spans_per_class = max(3, int(cfg.min_spans_per_class))
    plan: List[FaultClass] = []
    for cls in classes:
        # Steps and drifts are intrinsically long; too many would swamp the
        # anomaly rate, so they get the floor and nothing more in phase 1.
        n = min_spans_per_class
        if cls in (FaultClass.STEP,):
            n = max(2, min_spans_per_class // 2)
        plan.extend([cls] * n)
    rng.shuffle(plan)

    placed = 0
    attempts = 0
    max_attempts = budget * 40 + 4000
    plan_idx = 0

    while attempts < max_attempts:
        attempts += 1
        in_phase_one = plan_idx < len(plan)
        if not in_phase_one and placed >= budget:
            break

        s_idx = int(rng.integers(0, net.n_stations))
        station_id = net.stations[s_idx].station_id
        # Leave a margin so a fault cannot start in the last few points.
        start = int(rng.integers(lo, max(lo + 1, hi - 8)))

        if in_phase_one:
            fault_class = plan[plan_idx]
        else:
            fault_class = classes[int(rng.choice(len(classes), p=probs))]

        def _advance(success: bool) -> None:
            """Consume a plan slot on success; rotate it to the back on failure.

            Rotation prevents head-of-line blocking: without it, one
            unplaceable plan head (a STEP needing calm-to-end inside a stormy
            slice) would burn every attempt while placeable classes starve.
            """
            nonlocal plan_idx
            if not in_phase_one:
                return
            if success:
                plan_idx += 1
            else:
                plan.append(plan.pop(plan_idx))

        def _stall() -> None:
            """Shorthand for the failure paths below."""
            _advance(False)

        if fault_class == FaultClass.MULTIVARIATE:
            probe_end = min(start + 24, hi)
            if occupied[s_idx][start:probe_end].any():
                _stall()
                continue
            if regime_occupied[s_idx][start:probe_end].any():
                _stall()
                continue
            s, e, var, mag = _inject_multivariate(rng, net, s_idx, start)
            e = min(e, hi)
            occupied[s_idx][s : e + 1] = True
            faults.append(
                InjectedFault(station_id, s, e, FaultClass.MULTIVARIATE, var, float(mag))
            )
            placed += e - s + 1
            _advance(True)
            continue

        if fault_class == FaultClass.SPATIAL:
            probe_end = min(start + cfg.spatial_duration_points[1], hi)
            if occupied[s_idx][start:probe_end].any():
                _stall()
                continue
            if regime_occupied[s_idx][start:probe_end].any():
                _stall()
                continue
            sigma = _robust_sigma(net.temp[s_idx])
            s, e, var, mag = _inject_spatial(rng, net, s_idx, start, cfg, sigma)
            e = min(e, hi)
            occupied[s_idx][s : e + 1] = True
            faults.append(
                InjectedFault(station_id, s, e, FaultClass.SPATIAL, var, float(mag))
            )
            placed += e - s + 1
            _advance(True)
            continue

        # Single-variable point/span faults.
        variable = str(rng.choice(VARIABLES, p=[0.5, 0.2, 0.3]))
        series = arrays[variable][s_idx]
        sigma = _robust_sigma(series)

        # Probe only as far as this fault class can actually reach. A blanket
        # 400-point probe made short spikes fail to place on any station that
        # already had a long fault anywhere nearby.
        if fault_class == FaultClass.STEP:
            probe_end = hi           # a step runs to the end of the series
        elif fault_class == FaultClass.DRIFT:
            probe_end = min(start + cfg.drift_duration_points[1], hi)
        elif fault_class == FaultClass.STUCK:
            probe_end = min(start + cfg.stuck_duration_points[1], hi)
        elif fault_class == FaultClass.NOISE:
            probe_end = min(start + 96, hi)
        elif fault_class == FaultClass.MISSING:
            probe_end = min(start + cfg.missing_duration_points[1], hi)
        else:
            probe_end = min(start + 8, hi)
        if occupied[s_idx][start : probe_end + 1].any():
            _stall()
            continue
        if fault_class in (FaultClass.STEP, FaultClass.DRIFT):
            if regime_occupied[s_idx][start : probe_end + 1].any():
                _stall()
                continue

        injector = _POINT_INJECTORS[fault_class]
        s, e, mag = injector(rng, series, start, sigma, cfg)
        e = min(e, hi)
        if e < s:
            _stall()
            continue
        occupied[s_idx][s : e + 1] = True
        faults.append(
            InjectedFault(station_id, s, e, fault_class, variable, float(mag))
        )
        placed += e - s + 1
        _advance(True)

    # Clip physical variables back into representable ranges — except the
    # deliberate sentinel and missing values, which must survive untouched for
    # Layer 1 to detect them.
    _clip_preserving_faults(net, faults)

    return InjectionResult(network=net, clean=clean, faults=faults)


def _clip_preserving_faults(net: StationNetwork, faults: List[InjectedFault]) -> None:
    """Keep RH within 0-100 without erasing injected corruption.

    A blanket clip would silently convert every sentinel 999 into 100 and destroy
    the corrupt-encoding test set, so protected spans are masked out first.
    """
    protected = np.zeros((net.n_stations, net.n_steps), dtype=bool)
    for f in faults:
        if f.fault_class in (FaultClass.CORRUPT, FaultClass.MISSING):
            i = net.station_index[f.station_id]
            protected[i, f.start_index : f.end_index + 1] = True

    rh = net.rh
    clipped = np.clip(rh, 0.0, 100.0)
    net.rh = np.where(protected, rh, clipped)


def summarise_injection(result: InjectionResult) -> str:
    """Human-readable injection report for the harness log."""
    labels = result.label_matrix()
    total = labels.size
    n_anom = int(labels.sum())
    lines = [
        f"stations={result.network.n_stations} steps={result.network.n_steps}",
        f"anomalous points={n_anom} ({100.0 * n_anom / total:.2f}%)",
        "fault spans by class:",
    ]
    for name, count in result.summary().items():
        lines.append(f"  {name:<16} {count}")
    return "\n".join(lines)
