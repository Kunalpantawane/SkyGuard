"""Layer 1 — deterministic meteorological quality control.

This layer is not legacy baggage kept for appearances. It is the most reliable
part of the system for the faults it covers, and it earns its place for four
reasons:

1. **Certainty.** A relative humidity of 150 % is not "unusual", it is impossible.
   No learned model should be allowed to argue otherwise, which is why these
   checks can emit `hard=True` and short-circuit fusion.
2. **Speed.** Microseconds per observation, so it can run on an ESP32 where the
   autoencoder cannot.
3. **Explainability.** "Sentinel value -9999 received" needs no SHAP plot.
4. **Cold start.** It works on observation one, before any window has filled or
   any model has been trained.

Design decision that matters most here: the range limits are deliberately WIDE —
wider than any climatology. Layer 1 answers "is this physically possible?", never
"is this typical?". Tightening these bounds to catch more faults would start
rejecting real extreme weather, which is the exact failure this project exists to
avoid. Deciding whether a possible-but-unusual value is genuine is the job of
Layers 2 through 5.

Every check returns a named flag with a bounded score and a human-readable
detail, so the audit trail and the operator explanation come for free.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from ..config import VARIABLES, VARIABLE_LABELS, VARIABLE_UNITS, Config, DEFAULT_CONFIG
from ..types import Observation, QCFlag, RuleResult


class StationRuleState:
    """Rolling state needed by the stateful checks, per station.

    Bounded memory by construction: only the previous observation and per-variable
    run counters are retained. Nothing here grows with the length of the stream,
    which is what lets the system scale to a large network at constant cost per
    observation.
    """

    __slots__ = (
        "last_timestamp",
        "last_values",
        "run_length",
        "run_value",
        "missing_run",
        "seen",
    )

    def __init__(self) -> None:
        self.last_timestamp: Optional[datetime] = None
        self.last_values: Dict[str, Optional[float]] = {v: None for v in VARIABLES}
        # Run length counts how many times the CURRENT value has repeated,
        # starting at 1 for the first occurrence.
        self.run_length: Dict[str, int] = {v: 0 for v in VARIABLES}
        self.run_value: Dict[str, Optional[float]] = {v: None for v in VARIABLES}
        self.missing_run: Dict[str, int] = {v: 0 for v in VARIABLES}
        self.seen: int = 0


# --------------------------------------------------------------------------
# Individual checks
# --------------------------------------------------------------------------

def _check_sentinel(
    obs: Observation, cfg: Config
) -> List[QCFlag]:
    """Detect telemetry corruption: values that encode "no data" as a number.

    Checked before anything else and marked hard. A sentinel that slips through
    into a forecast model is more dangerous than a gap, because downstream code
    treats -9999 °C as a real measurement and it silently poisons averages.
    """
    flags: List[QCFlag] = []
    for var in VARIABLES:
        value = obs.value(var)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        for sentinel in cfg.sentinels.values:
            if abs(value - sentinel) < cfg.sentinels.tolerance:
                flags.append(
                    QCFlag(
                        name="corrupt_encoding",
                        variable=var,
                        score=1.0,
                        hard=True,
                        detail=(
                            f"{VARIABLE_LABELS[var]} reported sentinel value "
                            f"{value:g} — telemetry or parser fault, not a "
                            f"measurement"
                        ),
                    )
                )
                break
    return flags


def _check_missing(obs: Observation, state: StationRuleState) -> List[QCFlag]:
    """Detect absent values and track how long they have been absent.

    Missing is scored, not merely noted, and the score grows with the run length:
    one dropped packet is routine, twenty consecutive is a communication failure
    that needs an engineer.
    """
    flags: List[QCFlag] = []
    for var in VARIABLES:
        value = obs.value(var)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            state.missing_run[var] += 1
            run = state.missing_run[var]
            # Saturates at 1.0 by ~10 consecutive missing points.
            score = min(1.0, 0.35 + 0.065 * run)
            flags.append(
                QCFlag(
                    name="missing",
                    variable=var,
                    score=score,
                    hard=False,
                    detail=(
                        f"{VARIABLE_LABELS[var]} absent for {run} consecutive "
                        f"observation(s)"
                    ),
                )
            )
        else:
            state.missing_run[var] = 0
    return flags


def _check_range(obs: Observation, cfg: Config) -> List[QCFlag]:
    """Physical plausibility bounds.

    Hard flags: exceeding these is not improbable, it is impossible for a working
    sensor. Bounds are set to record-book extremes plus margin, so real weather —
    including record-breaking weather — passes.
    """
    flags: List[QCFlag] = []
    limits = {
        "temp_c": cfg.ranges.temp_c,
        "pressure_hpa": cfg.ranges.pressure_hpa,
        "rh_pct": cfg.ranges.rh_pct,
    }
    for var in VARIABLES:
        value = obs.value(var)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        # Sentinels are reported by their own check; flagging them here too would
        # double-count the same fault in the aggregate score.
        if any(abs(value - s) < cfg.sentinels.tolerance for s in cfg.sentinels.values):
            continue
        lo, hi = limits[var]
        if value < lo or value > hi:
            flags.append(
                QCFlag(
                    name="range",
                    variable=var,
                    score=1.0,
                    hard=True,
                    detail=(
                        f"{VARIABLE_LABELS[var]} {value:.2f}{VARIABLE_UNITS[var]} "
                        f"outside physically possible range "
                        f"[{lo:g}, {hi:g}]{VARIABLE_UNITS[var]}"
                    ),
                )
            )
    return flags


def _check_rate_of_change(
    obs: Observation,
    state: StationRuleState,
    cfg: Config,
    skip: Optional[set] = None,
) -> List[QCFlag]:
    """Could the atmosphere plausibly have moved this far, this fast?

    Far more informative than an absolute range test. 42 °C is possible; moving
    from 29 °C to 42 °C in ten minutes is not, and that transition is the
    signature of a spike.

    Two guards keep this from producing false alarms:
      - limits are per-hour and scaled by the ACTUAL elapsed time, so the same
        config works at 1-minute and 1-hour sampling;
      - after a long gap the check is skipped entirely, because real weather has
        had time to move and a large legitimate change would look like a fault.
    """
    flags: List[QCFlag] = []
    if state.last_timestamp is None:
        return flags

    gap_hours = (obs.timestamp - state.last_timestamp).total_seconds() / 3600.0
    if gap_hours <= 0 or gap_hours > cfg.rates.max_gap_hours_for_rate:
        return flags

    limits = {
        "temp_c": cfg.rates.temp_c_per_hour,
        "pressure_hpa": cfg.rates.pressure_hpa_per_hour,
        "rh_pct": cfg.rates.rh_pct_per_hour,
    }

    skip = skip or set()

    for var in VARIABLES:
        if var in skip:
            continue
        value = obs.value(var)
        previous = state.last_values[var]
        if value is None or previous is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue

        delta = abs(value - previous)
        # Sub-hourly sampling gets a floor on the allowance: at 1-minute spacing a
        # strict per-hour scaling would allow only 0.2 °C, and ordinary sensor
        # noise plus genuine gusty conditions would trip it constantly.
        allowed = limits[var] * max(gap_hours, 0.25)
        if delta > allowed:
            excess = delta / allowed
            # Ramps from 0.55 at the limit toward 1.0 for gross violations, so a
            # marginal exceedance is evidence rather than a conviction.
            score = min(1.0, 0.55 + 0.15 * (excess - 1.0))
            flags.append(
                QCFlag(
                    name="rate_of_change",
                    variable=var,
                    score=score,
                    hard=False,
                    detail=(
                        f"{VARIABLE_LABELS[var]} changed {delta:.2f}"
                        f"{VARIABLE_UNITS[var]} in {gap_hours * 60:.0f} min "
                        f"(plausible limit {allowed:.2f}{VARIABLE_UNITS[var]})"
                    ),
                )
            )
    return flags


def _check_persistence(
    obs: Observation,
    state: StationRuleState,
    cfg: Config,
    skip: Optional[set] = None,
) -> List[QCFlag]:
    """Detect a frozen sensor by counting exactly-repeated values.

    The signature is real: a stuck ADC or frozen firmware buffer returns
    bit-identical values, whereas a working sensor in still conditions still
    jitters at the resolution limit.

    Two refinements that stop this from firing on real weather:
      - the tolerance comes from sensor resolution, since a 0.1 °C sensor
        genuinely repeats readings;
      - RH near saturation legitimately flatlines during fog and rain, so the
        threshold is multiplied in that regime rather than flagging every foggy
        night as a fault.
    """
    flags: List[QCFlag] = []
    skip = skip or set()
    thresholds = {
        "temp_c": cfg.persistence.temp_c,
        "pressure_hpa": cfg.persistence.pressure_hpa,
        "rh_pct": cfg.persistence.rh_pct,
    }

    for var in VARIABLES:
        if var in skip:
            continue
        value = obs.value(var)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            # A gap breaks the run: values either side of an outage are not
            # evidence of a stuck sensor.
            state.run_length[var] = 0
            state.run_value[var] = None
            continue

        tolerance = cfg.persistence.tolerance[var]
        previous = state.run_value[var]

        if previous is not None and abs(value - previous) <= tolerance:
            state.run_length[var] += 1
        else:
            state.run_length[var] = 1
            state.run_value[var] = value

        limit = thresholds[var]
        if var == "rh_pct" and value >= cfg.persistence.rh_saturation_threshold:
            limit = int(limit * cfg.persistence.rh_saturation_multiplier)

        run = state.run_length[var]
        if run >= limit:
            over = run / limit
            score = min(1.0, 0.6 + 0.2 * (over - 1.0))
            flags.append(
                QCFlag(
                    name="persistence",
                    variable=var,
                    score=score,
                    hard=False,
                    detail=(
                        f"{VARIABLE_LABELS[var]} frozen at {value:.2f}"
                        f"{VARIABLE_UNITS[var]} for {run} consecutive "
                        f"observations (limit {limit})"
                    ),
                )
            )
    return flags


def _check_timestamp(
    obs: Observation, state: StationRuleState, interval_minutes: Optional[float]
) -> List[QCFlag]:
    """Timestamp integrity: duplicates, reversals and unexpected gaps.

    A duplicated or out-of-order timestamp usually means a retransmission or a
    clock fault. It matters beyond bookkeeping: the spatial layer compares
    contemporaneous observations, so a wrong clock silently corrupts a
    neighbourhood comparison and produces residuals that look like sensor faults.
    """
    flags: List[QCFlag] = []
    if state.last_timestamp is None:
        return flags

    delta = (obs.timestamp - state.last_timestamp).total_seconds()

    if delta == 0:
        flags.append(
            QCFlag(
                name="duplicate_timestamp",
                variable=None,
                score=0.7,
                hard=False,
                detail=f"repeated timestamp {obs.timestamp.isoformat()}",
            )
        )
    elif delta < 0:
        flags.append(
            QCFlag(
                name="timestamp_reversal",
                variable=None,
                score=0.8,
                hard=False,
                detail=(
                    f"timestamp {obs.timestamp.isoformat()} precedes previous "
                    f"{state.last_timestamp.isoformat()} — clock or ordering fault"
                ),
            )
        )
    elif interval_minutes:
        expected = interval_minutes * 60.0
        # Tolerate up to 1.5x the nominal interval before calling it a gap: minor
        # jitter in transmission time is normal and not worth an alarm.
        if delta > expected * 1.5:
            n_missed = int(round(delta / expected)) - 1
            score = min(1.0, 0.3 + 0.07 * n_missed)
            flags.append(
                QCFlag(
                    name="timestamp_gap",
                    variable=None,
                    score=score,
                    hard=False,
                    detail=(
                        f"gap of {delta / 60.0:.0f} min "
                        f"(~{n_missed} missed observation(s))"
                    ),
                )
            )
    return flags


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------

class RuleEngine:
    """Stateful Layer 1 engine, one instance per network.

    Per-station state is created lazily so a network of any size costs only what
    it actually uses.
    """

    def __init__(
        self,
        config: Config | None = None,
        interval_minutes: Optional[float] = None,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        self.interval_minutes = interval_minutes
        self._states: Dict[str, StationRuleState] = {}

    def state_for(self, station_id: str) -> StationRuleState:
        state = self._states.get(station_id)
        if state is None:
            state = StationRuleState()
            self._states[station_id] = state
        return state

    def reset(self, station_id: Optional[str] = None) -> None:
        if station_id is None:
            self._states.clear()
        else:
            self._states.pop(station_id, None)

    def evaluate(self, obs: Observation) -> RuleResult:
        """Run every Layer 1 check against one observation.

        Order matters: timestamp and rate checks read the previous observation, so
        state is updated only after all checks have run.
        """
        cfg = self.config
        state = self.state_for(obs.station_id)

        flags: List[QCFlag] = []

        # Hard checks first. A variable that is physically impossible or corrupt
        # is fully diagnosed already, so the statistical checks are skipped for
        # it — reporting "RH 150 % is impossible" AND "RH changed too fast" is one
        # fault described twice, and it clutters the operator's explanation.
        hard_flags = _check_sentinel(obs, cfg) + _check_range(obs, cfg)
        flags.extend(hard_flags)
        hard_vars = {f.variable for f in hard_flags if f.variable is not None}

        flags.extend(_check_missing(obs, state))
        flags.extend(_check_timestamp(obs, state, self.interval_minutes))
        flags.extend(_check_rate_of_change(obs, state, cfg, skip=hard_vars))
        flags.extend(_check_persistence(obs, state, cfg, skip=hard_vars))

        # Aggregate. Per variable we take the MAXIMUM rather than the sum: a
        # single observation failing three checks is one broken reading, and
        # summing would push the score past 1.0 and overstate the evidence.
        per_variable: Dict[str, float] = {v: 0.0 for v in VARIABLES}
        for flag in flags:
            if flag.variable is not None:
                per_variable[flag.variable] = max(
                    per_variable[flag.variable], flag.score
                )

        record_level = [f.score for f in flags if f.variable is None]
        score = max(list(per_variable.values()) + record_level + [0.0])

        missing = {
            var: any(
                f.name == "missing" and f.variable == var for f in flags
            )
            for var in VARIABLES
        }

        result = RuleResult(
            flags=flags,
            score=score,
            per_variable=per_variable,
            has_hard_violation=any(f.hard for f in flags),
            missing=missing,
        )

        self._commit(obs, state, hard_vars)
        return result

    def _commit(
        self,
        obs: Observation,
        state: StationRuleState,
        hard_vars: set,
    ) -> None:
        """Advance rolling state.

        Only monotonically-advancing timestamps update `last_timestamp`: letting a
        reversed clock overwrite it would corrupt every subsequent rate and gap
        check on that station.
        """
        if state.last_timestamp is None or obs.timestamp > state.last_timestamp:
            state.last_timestamp = obs.timestamp

        for var in VARIABLES:
            value = obs.value(var)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            # Never let a value we just declared impossible become the reference
            # point for the next observation — the fault would propagate forward
            # and flag a perfectly healthy reading.
            if var in hard_vars:
                continue
            state.last_values[var] = value

        state.seen += 1


def evaluate_batch(
    observations: List[Observation],
    config: Config | None = None,
    interval_minutes: Optional[float] = None,
) -> List[RuleResult]:
    """Convenience wrapper for offline evaluation of one station's series.

    Observations must be in chronological order for the stateful checks to be
    meaningful.
    """
    engine = RuleEngine(config=config, interval_minutes=interval_minutes)
    return [engine.evaluate(obs) for obs in observations]
