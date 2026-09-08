"""Core record types shared across every layer.

These dataclasses are the contract between layers. Passing dicts across module
boundaries would make it impossible to know what a layer actually produced, and
the audit trail depends on every field being explicit.

Design rule enforced here: raw observations are immutable. No layer ever rewrites
a measured value. Corrections travel in separate fields alongside the original.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------

class Severity(str, Enum):
    """Operational severity. An enum, not a number, so it cannot be averaged —
    averaging severities is meaningless and invites accidental nonsense."""

    NONE = "NONE"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ConfidenceBand(str, Enum):
    """Calibrated confidence bands presented to operators."""

    NORMAL = "NORMAL"
    WATCH = "WATCH"
    SUSPICIOUS = "SUSPICIOUS"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class FaultClass(str, Enum):
    """Root-cause taxonomy.

    GENUINE_EXTREME is a first-class outcome, not an absence of fault: correctly
    concluding "this is real weather, trust it" is one of the system's most
    valuable answers and must be reportable.
    """

    NONE = "none"
    SPIKE = "spike"
    DROP = "drop"
    STUCK = "stuck"
    DRIFT = "drift"
    STEP = "step"
    NOISE = "noise"
    MISSING = "missing"
    CORRUPT = "corrupt"
    MULTIVARIATE = "multivar"
    SPATIAL = "spatial"
    GENUINE_EXTREME = "genuine_extreme"


class HealthStatus(str, Enum):
    """Sensor lifecycle state. UNKNOWN is distinct from HEALTHY — insufficient
    history is not evidence of good health."""

    UNKNOWN = "UNKNOWN"
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    DEGRADING = "DEGRADING"
    CRITICAL = "CRITICAL"


class QCStatus(str, Enum):
    """MADIS-style disposition of an observation.

    QUARANTINED means "do not use downstream but do not delete" — the
    auditability requirement. ESTIMATED means a corrected value is available.
    """

    PASS = "PASS"
    SUSPECT = "SUSPECT"
    QUARANTINED = "QUARANTINED"
    ESTIMATED = "ESTIMATED"
    MISSING = "MISSING"


# --------------------------------------------------------------------------
# Station metadata
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Station:
    """Static station description.

    Elevation and terrain are not decoration: spatial QC must compare a hilltop
    station against other hilltop stations, otherwise legitimate terrain-driven
    differences read as sensor faults.
    """

    station_id: str
    name: str
    lat: float
    lon: float
    elevation_m: float
    terrain: str = "inland"   # coastal | inland | hill | valley | plateau

    def to_dict(self) -> Dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Observation
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Observation:
    """One raw AWS reading. Frozen: this is the source of truth.

    None means genuinely absent. Sentinel values (999, -999) are deliberately
    preserved as-is so Layer 1 can distinguish telemetry corruption from a
    missing packet — collapsing both to None would destroy that distinction
    before anything got a chance to see it.
    """

    station_id: str
    timestamp: datetime
    temp_c: Optional[float] = None
    pressure_hpa: Optional[float] = None
    rh_pct: Optional[float] = None

    def value(self, variable: str) -> Optional[float]:
        return getattr(self, variable)

    def values(self) -> Dict[str, Optional[float]]:
        return {
            "temp_c": self.temp_c,
            "pressure_hpa": self.pressure_hpa,
            "rh_pct": self.rh_pct,
        }

    def to_dict(self) -> Dict:
        return {
            "station_id": self.station_id,
            "timestamp": self.timestamp.isoformat(),
            "temp_c": self.temp_c,
            "pressure_hpa": self.pressure_hpa,
            "rh_pct": self.rh_pct,
        }


# --------------------------------------------------------------------------
# Layer outputs
# --------------------------------------------------------------------------

@dataclass
class QCFlag:
    """A single named check result.

    Carrying the check name, the variable, a bounded score and a human-readable
    detail means explainability falls out of the QC layer for free rather than
    being reconstructed later.
    """

    name: str
    variable: Optional[str]
    score: float             # [0, 1] contribution
    hard: bool = False       # physically impossible -> cannot be argued away
    detail: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class RuleResult:
    """Layer 1 output."""

    flags: List[QCFlag] = field(default_factory=list)
    score: float = 0.0                                  # aggregate rule score
    per_variable: Dict[str, float] = field(default_factory=dict)
    has_hard_violation: bool = False
    missing: Dict[str, bool] = field(default_factory=dict)

    def flag_names(self) -> List[str]:
        return [f.name for f in self.flags]

    def to_dict(self) -> Dict:
        return {
            "flags": [f.to_dict() for f in self.flags],
            "score": self.score,
            "per_variable": self.per_variable,
            "has_hard_violation": self.has_hard_violation,
            "missing": self.missing,
        }


@dataclass
class ReconResult:
    """Layer 2 output — the autoencoder's verdict.

    Per-variable errors are kept separate because they are the AE's native
    explanation: they say *which sensor* looks wrong, with no post-hoc method.
    """

    available: bool = False       # False when the window is not yet full
    error_total: float = 0.0
    per_variable: Dict[str, float] = field(default_factory=dict)
    probability: float = 0.0      # calibrated via station threshold
    threshold: float = 0.0
    reconstruction: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class MultivariateResult:
    """Layer 3 output — joint physical consistency of T, P, RH."""

    available: bool = False
    mahalanobis: float = 0.0
    probability: float = 0.0
    per_variable: Dict[str, float] = field(default_factory=dict)
    dewpoint_c: Optional[float] = None
    physical_violation: bool = False   # e.g. dew point above air temperature
    detail: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SpatialResult:
    """Layer 4 output — disagreement with comparable neighbours.

    `available` False means the station had too few comparable neighbours and the
    layer abstained. Abstaining is correct: a guess from one distant station of
    different elevation is worse than no spatial opinion at all.
    """

    available: bool = False
    probability: float = 0.0
    residuals: Dict[str, float] = field(default_factory=dict)
    normalised: Dict[str, float] = field(default_factory=dict)
    expected: Dict[str, float] = field(default_factory=dict)
    n_neighbours: int = 0
    neighbour_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class FusionResult:
    """Layer 5 output — the combined verdict.

    `evidence` retains every contributing signal so the final probability is
    always traceable back to what produced it.
    """

    probability: float = 0.0
    band: ConfidenceBand = ConfidenceBand.NORMAL
    is_anomaly: bool = False
    evidence: Dict[str, float] = field(default_factory=dict)
    genuine_weather_damped: bool = False
    dominant_variable: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "probability": self.probability,
            "band": self.band.value,
            "is_anomaly": self.is_anomaly,
            "evidence": self.evidence,
            "genuine_weather_damped": self.genuine_weather_damped,
            "dominant_variable": self.dominant_variable,
        }


@dataclass
class EvidenceLine:
    """One ranked explanation line, operator-facing."""

    label: str
    contribution: float
    detail: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DiagnosisResult:
    """Layer 6 output — root cause, severity and the reasoning behind them."""

    fault_class: FaultClass = FaultClass.NONE
    confidence: float = 0.0
    severity: Severity = Severity.NONE
    probabilities: Dict[str, float] = field(default_factory=dict)
    evidence: List[EvidenceLine] = field(default_factory=list)
    recommended_action: str = ""

    def to_dict(self) -> Dict:
        return {
            "fault_class": self.fault_class.value,
            "confidence": self.confidence,
            "severity": self.severity.value,
            "probabilities": self.probabilities,
            "evidence": [e.to_dict() for e in self.evidence],
            "recommended_action": self.recommended_action,
        }


@dataclass
class CorrectionResult:
    """Optional self-healing output. Estimates only — never a replacement."""

    available: bool = False
    values: Dict[str, float] = field(default_factory=dict)
    confidence: Dict[str, float] = field(default_factory=dict)
    method: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


# --------------------------------------------------------------------------
# The audit record
# --------------------------------------------------------------------------

@dataclass
class QCRecord:
    """Everything the system concluded about one observation.

    This is the auditable unit: raw values, every layer's score, the verdict, the
    diagnosis, any correction, and the explanation. Operational meteorological
    systems flag rather than delete, and this record is how that principle is
    honoured — the original reading is always recoverable from `observation`.
    """

    observation: Observation
    qc_status: QCStatus = QCStatus.PASS
    rules: RuleResult = field(default_factory=RuleResult)
    recon: ReconResult = field(default_factory=ReconResult)
    multivariate: MultivariateResult = field(default_factory=MultivariateResult)
    spatial: SpatialResult = field(default_factory=SpatialResult)
    fusion: FusionResult = field(default_factory=FusionResult)
    diagnosis: DiagnosisResult = field(default_factory=DiagnosisResult)
    correction: CorrectionResult = field(default_factory=CorrectionResult)
    explanation: str = ""
    latency_ms: float = 0.0

    @property
    def station_id(self) -> str:
        return self.observation.station_id

    @property
    def timestamp(self) -> datetime:
        return self.observation.timestamp

    @property
    def anomaly_score(self) -> float:
        return self.fusion.probability

    def to_dict(self) -> Dict:
        return {
            "observation": self.observation.to_dict(),
            "qc_status": self.qc_status.value,
            "rules": self.rules.to_dict(),
            "recon": self.recon.to_dict(),
            "multivariate": self.multivariate.to_dict(),
            "spatial": self.spatial.to_dict(),
            "fusion": self.fusion.to_dict(),
            "diagnosis": self.diagnosis.to_dict(),
            "correction": self.correction.to_dict(),
            "explanation": self.explanation,
            "latency_ms": self.latency_ms,
        }


# --------------------------------------------------------------------------
# Events and health
# --------------------------------------------------------------------------

@dataclass
class AnomalyEvent:
    """A contiguous anomaly span.

    Operators think in events, not points: a six-hour drift is one maintenance
    ticket, not 72 alarms. Point-level output alone makes a dashboard unusable.
    """

    event_id: str
    station_id: str
    start: datetime
    end: datetime
    n_points: int
    fault_class: FaultClass
    severity: Severity
    confidence: float
    peak_probability: float
    affected_variables: List[str] = field(default_factory=list)
    explanation: str = ""

    @property
    def duration_minutes(self) -> float:
        return (self.end - self.start).total_seconds() / 60.0

    def to_dict(self) -> Dict:
        return {
            "event_id": self.event_id,
            "station_id": self.station_id,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "duration_minutes": self.duration_minutes,
            "n_points": self.n_points,
            "fault_class": self.fault_class.value,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "peak_probability": self.peak_probability,
            "affected_variables": self.affected_variables,
            "explanation": self.explanation,
        }


@dataclass
class SensorHealth:
    """Lifecycle state of one sensor.

    Distinct from anomaly detection: "something is wrong now" and "this sensor is
    becoming untrustworthy" are different questions with different time scales
    and different actions.
    """

    variable: str
    score: float = 100.0
    status: HealthStatus = HealthStatus.UNKNOWN
    anomaly_rate: float = 0.0
    error_trend: float = 0.0        # slope of reconstruction error over window
    bias: float = 0.0               # persistent offset vs neighbours/expectation
    drift_slope: float = 0.0
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "variable": self.variable,
            "score": self.score,
            "status": self.status.value,
            "anomaly_rate": self.anomaly_rate,
            "error_trend": self.error_trend,
            "bias": self.bias,
            "drift_slope": self.drift_slope,
            "reasons": self.reasons,
        }


@dataclass
class StationHealth:
    """Aggregate station health plus a forward-looking maintenance risk.

    The risk is an estimate of deterioration, not a predicted failure date. The
    system has no basis for claiming a date and should not pretend otherwise.
    """

    station_id: str
    overall_score: float = 100.0
    status: HealthStatus = HealthStatus.UNKNOWN
    sensors: Dict[str, SensorHealth] = field(default_factory=dict)
    maintenance_risk: str = "LOW"
    n_points: int = 0

    def to_dict(self) -> Dict:
        return {
            "station_id": self.station_id,
            "overall_score": self.overall_score,
            "status": self.status.value,
            "sensors": {k: v.to_dict() for k, v in self.sensors.items()},
            "maintenance_risk": self.maintenance_risk,
            "n_points": self.n_points,
        }


# --------------------------------------------------------------------------
# Ground truth (evaluation only)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class InjectedFault:
    """A known injected fault span, for scoring.

    Kept strictly outside the detection path — nothing in `skyguard/qc`,
    `model`, `fusion` or `diagnostics` may import this. It exists only so the
    evaluation harness knows the truth.
    """

    station_id: str
    start_index: int
    end_index: int
    fault_class: FaultClass
    variable: Optional[str]
    magnitude: float = 0.0

    def covers(self, index: int) -> bool:
        return self.start_index <= index <= self.end_index

    def to_dict(self) -> Dict:
        return {
            "station_id": self.station_id,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "fault_class": self.fault_class.value,
            "variable": self.variable,
            "magnitude": self.magnitude,
        }


# Fault classes that represent real weather rather than sensor problems. The
# system must NOT alarm on these; they are the false-alarm test set.
BENIGN_CLASSES: Tuple[FaultClass, ...] = (FaultClass.NONE, FaultClass.GENUINE_EXTREME)
