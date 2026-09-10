"""Central configuration for SkyGuard AI.

Every tunable in the system lives here. Logic modules import these dataclasses
rather than embedding literals, so a deployment can be re-tuned for a different
network (different climate, different sampling interval) without touching the
detection code.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Tuple
import json


# --------------------------------------------------------------------------
# Physical limits (Layer 1)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RangeLimits:
    """Physically plausible bounds per variable.

    These are deliberately *generous* — wider than any climatology. Layer 1 must
    only catch the physically impossible; deciding whether a possible-but-unusual
    value is genuine is the job of Layers 2-5. A tight range check here would
    reject real extreme weather, which is precisely the failure mode we are
    trying to avoid.
    """

    temp_c: Tuple[float, float] = (-90.0, 60.0)      # Vostok record to hottest reliable
    pressure_hpa: Tuple[float, float] = (500.0, 1085.0)  # high-altitude station to record high
    rh_pct: Tuple[float, float] = (0.0, 100.0)       # definitional


@dataclass(frozen=True)
class RateLimits:
    """Maximum plausible change per hour.

    Expressed per hour and scaled by the actual gap between observations, so the
    same config works at 1-minute or 1-hour sampling. Asking "could the sensor
    physically move this far this fast?" is far more informative than asking
    whether the absolute value is possible.
    """

    temp_c_per_hour: float = 12.0
    pressure_hpa_per_hour: float = 12.0
    rh_pct_per_hour: float = 60.0

    # A gap longer than this makes rate-of-change meaningless: real weather has
    # had time to move, so we stop testing rather than raise a false flag.
    max_gap_hours_for_rate: float = 3.0


@dataclass(frozen=True)
class PersistenceLimits:
    """How many identical consecutive readings before a sensor looks stuck.

    Pressure legitimately sits still far longer than temperature, and RH plateaus
    at saturation, so the thresholds differ per variable. Values are counts of
    consecutive *exactly equal* readings (within a tolerance).
    """

    temp_c: int = 12
    pressure_hpa: int = 24
    rh_pct: int = 16

    # Two readings closer than this count as identical. Set from sensor
    # resolution: a 0.1 °C sensor genuinely repeats values.
    tolerance: Dict[str, float] = field(
        default_factory=lambda: {"temp_c": 0.05, "pressure_hpa": 0.05, "rh_pct": 0.5}
    )

    # RH at/near saturation legitimately flatlines during fog or rain, so we
    # require a longer run before flagging it.
    rh_saturation_threshold: float = 97.0
    rh_saturation_multiplier: float = 2.5


@dataclass(frozen=True)
class SentinelConfig:
    """Values that indicate telemetry corruption rather than a real measurement."""

    values: Tuple[float, ...] = (999.0, -999.0, 9999.0, -9999.0, 99.9, -99.9, 32767.0)
    tolerance: float = 1e-6


# --------------------------------------------------------------------------
# Model (Layer 2)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """LSTM autoencoder geometry and training schedule."""

    window: int = 24            # observations per window
    hidden_size: int = 24       # LSTM hidden units (also latent width)
    learning_rate: float = 0.005
    batch_size: int = 32
    max_epochs: int = 60
    patience: int = 8           # early-stopping patience on clean val loss
    grad_clip: float = 5.0      # LSTMs need this; exploding gradients are routine
    seed: int = 7

    # Adam
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

    @property
    def n_physical(self) -> int:
        """Variables actually reconstructed: T, P, RH."""
        return 3

    @property
    def n_features(self) -> int:
        """Input width: 3 physical + 4 cyclic time features."""
        return 7


@dataclass(frozen=True)
class ThresholdConfig:
    """Adaptive, station-specific thresholding on reconstruction error."""

    k_sigma: float = 4.0        # robust MAD-sigma multiplier
    quantile: float = 0.995     # fallback for skewed error distributions
    min_samples: int = 200      # below this, fall back to network-wide threshold
    # Logistic steepness for error -> probability. Higher = sharper transition
    # around the threshold.
    logistic_scale: float = 2.5


# --------------------------------------------------------------------------
# Multivariate (Layer 3) and spatial (Layer 4)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MultivariateConfig:
    """Joint-state consistency between T, P and RH."""

    # Chi-square-ish cut on Mahalanobis distance for 6 dims (3 levels + 3 deltas).
    mahalanobis_threshold: float = 4.5
    min_samples: int = 100
    shrinkage: float = 0.05     # ridge on the covariance so it stays invertible

    # Dew point cannot exceed air temperature by more than measurement slack.
    # Violation is a hard physical inconsistency, not a statistical one.
    dewpoint_excess_tolerance_c: float = 1.0


@dataclass(frozen=True)
class SpatialConfig:
    """Neighbour comparison, weighted by true comparability not just distance."""

    max_neighbours: int = 6
    max_distance_km: float = 250.0
    max_elevation_diff_m: float = 900.0
    min_neighbours: int = 2     # below this, spatial QC abstains rather than guessing

    lapse_rate_c_per_m: float = 0.0065   # standard atmosphere temperature lapse
    # Weighting: distance decay length and elevation penalty length.
    distance_scale_km: float = 80.0
    elevation_scale_m: float = 300.0

    # Residual is normalised by robust neighbour spread; these floors stop a
    # freakishly tight neighbourhood from making every residual look enormous.
    min_spread: Dict[str, float] = field(
        default_factory=lambda: {"temp_c": 1.0, "pressure_hpa": 0.8, "rh_pct": 4.0}
    )
    residual_sigma: float = 3.5          # residual/spread ratio treated as anomalous
    max_time_skew_minutes: float = 20.0  # neighbours must be roughly contemporaneous


# --------------------------------------------------------------------------
# Fusion (Layer 5)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FusionConfig:
    """How independent evidence streams combine into one probability.

    Weights reflect trust, not equality. Deterministic rules are near-certain
    when they fire (a sentinel value IS corrupt), so they carry the most weight.
    The AE is powerful but the least literal, so it cannot convict alone.
    """

    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "rule": 1.00,
            "reconstruction": 0.75,
            "multivariate": 0.65,
            "spatial": 0.80,
            "persistence": 0.85,
        }
    )

    # A hard rule violation (impossible value, sentinel) short-circuits fusion:
    # no amount of contrary evidence makes RH = 150 % acceptable.
    hard_rule_floor: float = 0.95

    # Genuine-weather guard. When neighbours move with the station and the
    # covariates stay physically coherent, we damp the score rather than
    # suppressing it entirely — the evidence is weakened, not erased.
    genuine_weather_damping: float = 0.35
    spatial_agreement_threshold: float = 0.35   # low spatial score = neighbours agree
    covariate_coherence_threshold: float = 0.40

    # Confidence bands. Calibrated on validation data; these are the defaults.
    bands: Tuple[Tuple[float, str], ...] = (
        (0.30, "NORMAL"),
        (0.60, "WATCH"),
        (0.80, "SUSPICIOUS"),
        (0.95, "HIGH"),
        (1.01, "CRITICAL"),
    )

    anomaly_decision_threshold: float = 0.50    # mid-WATCH and above counts as flagged


# --------------------------------------------------------------------------
# Events, health, correction
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class EventConfig:
    """Grouping consecutive anomalous points into operator-facing events."""

    # Tolerate this many normal points inside an event before closing it. A
    # flickering sensor is one fault, not fifty alarms.
    max_gap_points: int = 3
    min_event_points: int = 1


@dataclass(frozen=True)
class HealthConfig:
    """Per-sensor health scoring and degradation tracking."""

    window_points: int = 720            # ~30 days at hourly sampling
    # Health penalty weights; they sum to the maximum deduction from 100.
    weight_anomaly_rate: float = 40.0
    weight_error_trend: float = 25.0
    weight_bias: float = 20.0
    weight_drift: float = 15.0

    bands: Tuple[Tuple[float, str], ...] = (
        (40.0, "CRITICAL"),
        (60.0, "DEGRADING"),
        (80.0, "WARNING"),
        (100.1, "HEALTHY"),
    )

    # Minimum history before a health score is meaningful; below this we report
    # UNKNOWN rather than a falsely confident number.
    min_points: int = 48


@dataclass(frozen=True)
class CorrectionConfig:
    """Estimating replacement values for quarantined observations."""

    # Blend weights across the three independent estimators.
    weight_temporal: float = 0.4
    weight_spatial: float = 0.4
    weight_reconstruction: float = 0.2

    # Refuse to correct beyond this many consecutive bad points: extrapolating
    # a long outage produces fiction, and fiction with a confidence score is
    # worse than a gap.
    max_consecutive_points: int = 12
    min_confidence: float = 0.35


# --------------------------------------------------------------------------
# Fault injection
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class InjectorConfig:
    """Fault injection for evaluation."""

    seed: int = 1337
    # Target fraction of anomalous points. Real networks are far cleaner, but a
    # very low rate makes the test set statistically thin; per-class recall is
    # reported separately so the aggregate rate does not distort conclusions.
    target_anomaly_rate: float = 0.05

    # Guaranteed minimum injected spans per fault class. Without a floor, rare
    # long-duration classes (step, drift) consume the point budget and end up
    # with one or two spans, making their per-class recall statistically
    # meaningless.
    min_spans_per_class: int = 6

    spike_magnitude_sigma: Tuple[float, float] = (5.0, 15.0)
    drift_magnitude: Tuple[float, float] = (2.0, 8.0)
    step_magnitude: Tuple[float, float] = (1.5, 6.0)
    noise_multiplier: Tuple[float, float] = (4.0, 12.0)
    stuck_duration_points: Tuple[int, int] = (8, 48)
    drift_duration_points: Tuple[int, int] = (72, 400)
    step_duration_points: Tuple[int, int] = (48, 200)
    missing_duration_points: Tuple[int, int] = (2, 24)
    spatial_offset_sigma: Tuple[float, float] = (4.0, 10.0)
    spatial_duration_points: Tuple[int, int] = (6, 48)


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ForestConfig:
    """Random forest fault classifier."""

    n_trees: int = 40
    max_depth: int = 9
    min_samples_split: int = 12
    min_samples_leaf: int = 4
    max_features: str = "sqrt"
    seed: int = 11
    class_balance: bool = True   # rare fault classes must not be drowned out


@dataclass(frozen=True)
class ShapleyConfig:
    """Explainability for the fault classifier."""

    exact_max_features: int = 12   # above this, exact enumeration is intractable
    n_samples: int = 220           # KernelSHAP sampling budget
    background_size: int = 60
    seed: int = 5
    top_k: int = 5                 # evidence lines shown to an operator


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ApiConfig:
    """Local dashboard API.

    Binds to loopback and requires a bearer token by default. This service
    exposes station data and accepts ingest, so an unauthenticated bind to
    0.0.0.0 would put the network's QC state on the local network in the clear.
    """

    host: str = "127.0.0.1"
    port: int = 8765
    require_token: bool = True
    token_env_var: str = "SKYGUARD_API_TOKEN"


# --------------------------------------------------------------------------
# Root
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    """The whole system's configuration."""

    ranges: RangeLimits = field(default_factory=RangeLimits)
    rates: RateLimits = field(default_factory=RateLimits)
    persistence: PersistenceLimits = field(default_factory=PersistenceLimits)
    sentinels: SentinelConfig = field(default_factory=SentinelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    multivariate: MultivariateConfig = field(default_factory=MultivariateConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    events: EventConfig = field(default_factory=EventConfig)
    health: HealthConfig = field(default_factory=HealthConfig)
    correction: CorrectionConfig = field(default_factory=CorrectionConfig)
    injector: InjectorConfig = field(default_factory=InjectorConfig)
    forest: ForestConfig = field(default_factory=ForestConfig)
    shapley: ShapleyConfig = field(default_factory=ShapleyConfig)
    api: ApiConfig = field(default_factory=ApiConfig)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, default=str)


DEFAULT_CONFIG = Config()

# Canonical variable names, used as dict keys throughout the system. Order is
# fixed because the model's feature matrix columns depend on it.
VARIABLES = ("temp_c", "pressure_hpa", "rh_pct")

VARIABLE_LABELS = {
    "temp_c": "Temperature",
    "pressure_hpa": "Pressure",
    "rh_pct": "Humidity",
}

VARIABLE_UNITS = {"temp_c": "°C", "pressure_hpa": "hPa", "rh_pct": "%"}
