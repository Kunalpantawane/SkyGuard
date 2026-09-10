"""Streaming pipeline: `process(obs) -> QCRecord`, the whole engine in one call.

Why an orchestrator instead of ad-hoc layer calls: layer order matters (rules
feed fusion, the temporal window feeds the AE, fusion feeds diagnosis), every
layer keeps bounded per-station streaming state, and the audit record must
carry all of it. One method owns the order, the state and the record — callers
(API, harness, dashboard) never reimplement the wiring.

Per-observation cost is O(1) with bounded memory: the temporal buffer holds W
points, recent-flag and background buffers are capped deques, and no history
grows with stream length. Fitted components (LSTM, scalers, thresholds,
multivariate models, forest) arrive trained — fitting on the live stream
would let faults teach the model that faults are normal.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import timezone
from typing import Callable, Dict, List, Optional

import numpy as np

from .config import DEFAULT_CONFIG, VARIABLES, Config
from .correction.imputer import SourceEstimate, impute
from .data.loaders import to_utc
from .diagnostics.forest import FEATURE_NAMES, FaultClassifier
from .diagnostics.narrative import diagnose
from .diagnostics.shapley import rank_contributions, shapley_values
from .fusion.engine import FusionEngine
from .model.lstm import LstmAutoencoder
from .model.scaler import StationScaler, build_window
from .model.threshold import StationThreshold, score as threshold_score
from .qc.multivariate import MultivariateQC
from .qc.rules import RuleEngine
from .qc.spatial import SpatialQC
from .store import AuditStore
from .types import (
    ConfidenceBand,
    CorrectionResult,
    DiagnosisResult,
    FaultClass,
    FusionResult,
    MultivariateResult,
    Observation,
    QCRecord,
    QCStatus,
    ReconResult,
    RuleResult,
    SpatialResult,
)

# Recent-verdict memory for the anomaly-rate fingerprint. Long enough to be
# stable, short enough to forget: 48 hourly points cover two diurnal cycles.
_RECENT_FLAG_WINDOW = 48

# Normal-fingerprint memory for the Shapley background. Matches the
# explainability background budget without growing over time.
_BACKGROUND_WINDOW = 60


def _canonicalise(obs: Observation) -> Observation:
    """Return `obs` with a UTC-aware timestamp, rebuilding only when needed."""
    stamp = to_utc(obs.timestamp)
    if stamp == obs.timestamp and obs.timestamp.tzinfo is timezone.utc:
        return obs
    return replace(obs, timestamp=stamp)


@dataclass
class _StationStream:
    """Bounded per-station streaming state. Nothing here grows with time."""

    window: deque = field(default_factory=deque)          # (timestamp, levels) up to W
    recent_flags: deque = field(default_factory=lambda: deque(maxlen=_RECENT_FLAG_WINDOW))
    background: deque = field(default_factory=lambda: deque(maxlen=_BACKGROUND_WINDOW))
    missing_run: int = 0
    bad_run: int = 0
    last_good: Optional[np.ndarray] = None                # last trusted triplet


class SkyGuardPipeline:
    """One-call orchestration over injected, pre-fitted layer components."""

    def __init__(
        self,
        stations: Dict[str, object] | None = None,
        config: Config | None = None,
        lstm: LstmAutoencoder | None = None,
        scalers: Dict[str, StationScaler] | None = None,
        thresholds: Dict[str, StationThreshold] | None = None,
        multivariate: MultivariateQC | None = None,
        spatial: SpatialQC | None = None,
        fusion: FusionEngine | None = None,
        rules: RuleEngine | None = None,
        classifier: FaultClassifier | None = None,
        store: AuditStore | None = None,
        fingerprint_sink: Callable[[Observation, List[float]], None] | None = None,
    ) -> None:
        self.config: Config = config or DEFAULT_CONFIG
        # The window geometry belongs to the model artifact, not the pipeline:
        # taking it from config while injecting a differently-shaped LSTM
        # would silently starve Layer 2 of full windows forever.
        self.window_size = lstm.window if lstm is not None else self.config.model.window
        self.rules = rules or RuleEngine()
        self.lstm = lstm
        self.scalers = scalers or {}
        self.thresholds = thresholds or {}
        self.multivariate = multivariate or MultivariateQC(config=self.config.multivariate)
        self.spatial = spatial
        self.fusion = fusion or FusionEngine(config=self.config.fusion)
        self.classifier = classifier
        self.store = store
        # Training the Layer 6 forest needs the same 14-column fingerprints the
        # pipeline builds at inference, and rebuilding them outside would be a
        # second implementation of the feature contract waiting to drift. The
        # sink hands each one out as it is built; None means nobody is watching.
        self.fingerprint_sink = fingerprint_sink
        self._streams: Dict[str, _StationStream] = {}

    # -- state ------------------------------------------------------------

    def _stream(self, station_id: str) -> _StationStream:
        stream = self._streams.get(station_id)
        if stream is None:
            stream = _StationStream()
            # Bounded: only the latest window matters, and an unbounded buffer
            # would both leak memory and break the fixed-size reshape below.
            stream.window = deque(maxlen=self.window_size)
            self._streams[station_id] = stream
        return stream

    def reset_station(self, station_id: str) -> None:
        """Drop streaming state after an outage or a sensor swap."""
        self._streams.pop(station_id, None)
        self.rules.reset(station_id)
        self.multivariate.reset(station_id)

    # -- the one call -------------------------------------------------------

    def process(
        self, obs: Observation, neighbours: List[Observation] | None = None,
        explain: bool = False,
    ) -> QCRecord:
        """Run one observation through every layer and return the audit record."""
        started = time.perf_counter()
        # Canonicalise here, at the single entry point, rather than trusting every
        # caller. A naive timestamp reaching Layer 1's rate check raises TypeError
        # against the stored aware one, and mixed offsets break the store's
        # text-ordered queries and the spatial layer's skew comparison.
        obs = _canonicalise(obs)
        nearby = [_canonicalise(n) for n in (neighbours or [])]
        stream = self._stream(obs.station_id)

        rule_result = self.rules.evaluate(obs)
        levels = np.array([obs.temp_c, obs.pressure_hpa, obs.rh_pct], dtype=np.float64)
        complete = bool(np.all(np.isfinite(levels)))

        stream.missing_run = stream.missing_run + 1 if not complete else 0
        recon_result, reconstruction = self._temporal(obs, stream, levels, complete)
        multi_result = self.multivariate.evaluate(
            obs.station_id, obs.temp_c, obs.pressure_hpa, obs.rh_pct,
            hard_invalid=rule_result.has_hard_violation)
        if self.spatial is not None:
            spatial_result = self.spatial.evaluate(obs, nearby)
        else:
            spatial_result = SpatialResult(available=False)
        fused = self.fusion.fuse(rule_result, recon_result, multi_result, spatial_result)

        fingerprint = self._fingerprint(obs, stream, levels, complete,
                                        recon_result, multi_result, spatial_result, rule_result)
        if self.fingerprint_sink is not None and fingerprint is not None:
            self.fingerprint_sink(obs, fingerprint)
        diagnosis = self._diagnose(obs, stream, levels, complete, fused, fingerprint, explain)
        correction = self._correct(obs, stream, levels, complete, fused,
                                   recon_result, reconstruction, spatial_result)

        if all(v is None for v in (obs.temp_c, obs.pressure_hpa, obs.rh_pct)):
            status = QCStatus.MISSING
        elif fused.is_anomaly:
            status = QCStatus.ESTIMATED if correction.available else QCStatus.QUARANTINED
        elif not complete:
            # A partially missing reading is MISSING, never PASS. One dead sensor
            # among three scores below the fusion threshold, but calling it PASS
            # would tell an operator the record is usable and hide exactly the
            # communication/sensor fault the brief asks us to surface.
            status = QCStatus.MISSING
        elif fused.band == ConfidenceBand.WATCH:
            status = QCStatus.SUSPECT
        else:
            status = QCStatus.PASS

        record = QCRecord(
            observation=obs, qc_status=status, rules=rule_result, recon=recon_result,
            multivariate=multi_result, spatial=spatial_result, fusion=fused,
            diagnosis=diagnosis, correction=correction,
            explanation=self._narrate(fused, diagnosis),
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

        # Learn only from trustworthy points: anomalies must never become the
        # baseline, the background, or the "last good" estimate.
        stream.recent_flags.append(fused.is_anomaly)
        if fused.is_anomaly:
            stream.bad_run += 1
        else:
            stream.bad_run = 0
            if complete and not rule_result.has_hard_violation:
                stream.last_good = levels.copy()
                if fingerprint is not None:
                    stream.background.append(np.asarray(fingerprint, dtype=np.float64))
        if self.store is not None:
            self.store.save_record(record)
        return record

    # -- temporal (Layer 2 + threshold) ---------------------------------------

    def _temporal(
        self, obs: Observation, stream: _StationStream,
        levels: np.ndarray, complete: bool,
    ) -> tuple:
        """Window the stream, reconstruct, and calibrate. A gap resets the
        temporal context: stitching across an outage would invent a trajectory
        the atmosphere never took."""
        if not complete:
            stream.window.clear()
            return ReconResult(available=False), {}
        stream.window.append((obs.timestamp, levels.copy()))
        if len(stream.window) < self.window_size:
            return ReconResult(available=False), {}
        scaler = self.scalers.get(obs.station_id)
        calibration = self.thresholds.get(obs.station_id)
        if self.lstm is None or scaler is None or calibration is None:
            return ReconResult(available=False), {}
        stamps = [ts for ts, _ in stream.window]
        stacked = np.stack([lv for _, lv in stream.window])
        window = build_window(scaler, stacked, stamps).reshape(1, self.window_size, -1)
        y_hat, error_total, error_per_var = self.lstm.reconstruct_batch(window)
        probability, used = threshold_score(calibration, float(error_total[0]))
        per_variable = {var: float(error_per_var[0, k]) for k, var in enumerate(VARIABLES)}
        reconstruction = {
            var: float(value)
            for var, value in zip(VARIABLES, scaler.inverse(y_hat[0, -1, :]))
        }
        return ReconResult(
            available=True, error_total=float(error_total[0]),
            per_variable=per_variable, probability=float(probability),
            threshold=float(used), reconstruction=reconstruction,
        ), reconstruction

    # -- fingerprint (Layer 6 input) ------------------------------------------

    def _fingerprint(
        self, obs: Observation, stream: _StationStream, levels: np.ndarray, complete: bool,
        recon: ReconResult, multi_result: MultivariateResult,
        spatial_result: SpatialResult, rule_result: RuleResult,
    ) -> Optional[List[float]]:
        """Assemble the 14-column diagnostic fingerprint in FEATURE_NAMES order.

        Unavailable signals read as zero (no evidence), never as imputed
        guesses — the forest was trained with the same convention.
        """
        if not complete:
            return None
        recon_errors = [recon.per_variable.get(var, 0.0) if recon.available else 0.0
                        for var in VARIABLES]
        rate = self._rate_feature(obs, stream, levels)
        persistence = max(
            (f.score for f in rule_result.flags if f.name == "persistence"), default=0.0)
        repeat_run = (persistence * self.config.persistence.temp_c
                      if persistence > 0 else 0.0)
        variance_ratio, drift_slope = self._volatility_features(stream)
        mahalanobis = multi_result.mahalanobis if multi_result.available else 0.0
        spatial_residual = (max(spatial_result.normalised.values())
                            if spatial_result.available and spatial_result.normalised else 0.0)
        sentinel = 1.0 if any(f.name == "corrupt_encoding" for f in rule_result.flags) else 0.0
        recent_rate = (float(sum(stream.recent_flags)) / len(stream.recent_flags)
                       if stream.recent_flags else 0.0)
        hour = obs.timestamp.hour + obs.timestamp.minute / 60.0
        return [
            *recon_errors, rate, repeat_run, variance_ratio, drift_slope,
            mahalanobis, spatial_residual, float(stream.missing_run), sentinel,
            recent_rate, math.sin(2 * math.pi * hour / 24.0),
            math.cos(2 * math.pi * hour / 24.0),
        ]

    def _rate_feature(self, obs: Observation, stream: _StationStream, levels: np.ndarray) -> float:
        """Largest standardized step versus the rate limits, dimensionless."""
        if len(stream.window) < 2:
            return 0.0
        prev_ts, prev = stream.window[-2]
        dt_hours = max((obs.timestamp - prev_ts).total_seconds() / 3600.0, 1e-6)
        limits = (self.config.rates.temp_c_per_hour, self.config.rates.pressure_hpa_per_hour,
                  self.config.rates.rh_pct_per_hour)
        return float(max(abs(levels[k] - prev[k]) / (limits[k] * dt_hours) for k in range(3)))

    def _volatility_features(self, stream: _StationStream) -> tuple:
        """Rolling variance ratio and drift slope of temperature."""
        if len(stream.window) < 4:
            return 1.0, 0.0
        temps = np.array([lv[0] for _, lv in stream.window])
        baseline = temps[-min(len(temps), 48):]
        recent = baseline[-min(len(baseline), 12):]
        ratio = float(np.var(recent) / max(np.var(baseline), 1e-12))
        x = np.arange(len(baseline), dtype=np.float64)
        slope = float(np.cov(x, baseline, bias=True)[0, 1] / max(np.var(x), 1e-12))
        drift = float(slope * len(baseline) / max(abs(float(baseline.mean())), 1e-9))
        return ratio, drift

    # -- diagnosis --------------------------------------------------------------

    def _diagnose(
        self, obs: Observation, stream: _StationStream, levels: np.ndarray, complete: bool,
        fused: FusionResult, fingerprint: Optional[List[float]], explain: bool,
    ) -> DiagnosisResult:
        """Classify the fault when the verdict is anomalous and a fitted
        classifier exists. Clean points carry no diagnosis — NONE with no
        severity, never a forced guess."""
        if (not fused.is_anomaly or not complete or fingerprint is None
                or self.classifier is None or not self.classifier.is_fitted):
            return DiagnosisResult()
        prediction = self.classifier.predict_one(fingerprint)
        contributions: List[tuple] = []
        if explain and self.classifier is not None and len(stream.background) >= 5:
            background = np.stack(list(stream.background))
            point = np.asarray(fingerprint, dtype=np.float64)
            target = prediction.fault_class.value
            classifier = self.classifier
            phi = shapley_values(
                point, background,
                lambda mat: _class_proba(classifier, mat, target),
            )
            contributions = rank_contributions(list(FEATURE_NAMES), phi)
        observed = {var: float(levels[k]) for k, var in enumerate(VARIABLES)}
        return diagnose(prediction.fault_class, prediction.confidence,
                        prediction.probabilities, contributions, observed)

    # -- correction ---------------------------------------------------------------

    def _correct(
        self, obs: Observation, stream: _StationStream, levels: np.ndarray, complete: bool,
        fused: FusionResult, recon: ReconResult, reconstruction: dict,
        spatial_result: SpatialResult,
    ) -> CorrectionResult:
        """Estimate replacements for anomalous points from three sources."""
        if not fused.is_anomaly or not complete:
            return CorrectionResult()
        values: Dict[str, float] = {}
        confidences: Dict[str, float] = {}
        methods: Dict[str, str] = {}
        for k, var in enumerate(VARIABLES):
            sources: Dict[str, SourceEstimate] = {}
            if stream.last_good is not None:
                sources["temporal"] = SourceEstimate(
                    value=float(stream.last_good[k]), confidence=0.8,
                    method="last-good-carry")
            if spatial_result.available and var in spatial_result.expected:
                sources["spatial"] = SourceEstimate(
                    value=float(spatial_result.expected[var]),
                    confidence=float(min(0.9, 0.4 + 0.1 * spatial_result.n_neighbours)),
                    method="neighbour-consensus")
            if recon.available and var in reconstruction:
                sources["reconstruction"] = SourceEstimate(
                    value=float(reconstruction[var]),
                    confidence=float(max(0.0, 1.0 - recon.probability)),
                    method="autoencoder")
            blended = impute(sources, stream.bad_run + 1, self.config.correction)
            if blended.available and blended.value is not None:
                values[var] = blended.value
                confidences[var] = blended.confidence
                methods[var] = blended.method
        return CorrectionResult(
            available=bool(values), values=values,
            confidence=confidences, method=methods,
        )

    # -- narrative ------------------------------------------------------------------

    @staticmethod
    def _narrate(fused: FusionResult, diagnosis: DiagnosisResult) -> str:
        """One-line operator summary for the record and the dashboard."""
        text = f"{fused.band.value} p={fused.probability:.2f}"
        if diagnosis.fault_class != FaultClass.NONE:
            text += f" | {diagnosis.fault_class.value} ({diagnosis.confidence:.0%})"
            if diagnosis.evidence:
                text += f" via {diagnosis.evidence[0].label}"
        if fused.genuine_weather_damped:
            text += " [weather-damped]"
        if fused.dominant_variable:
            text += f" @ {fused.dominant_variable}"
        return text


def _class_proba(classifier: FaultClassifier, mat: np.ndarray, target: str) -> np.ndarray:
    """Probability column of the predicted class, for Shapley scoring."""
    from .diagnostics.forest import FAULT_ORDER
    proba = classifier.predict_proba(mat)
    index = next(k for k, fault in enumerate(FAULT_ORDER) if fault.value == target)
    return proba[:, index]
