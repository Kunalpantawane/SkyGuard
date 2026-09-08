"""SkyGuard AI — explainable spatio-temporal quality assurance for Automatic
Weather Station networks.

A six-layer hybrid engine: deterministic meteorological QC, an LSTM autoencoder
for temporal normality, multivariate consistency across temperature/pressure/
humidity, spatial consensus against comparable neighbours, evidence fusion, and
fault classification with Shapley explanations. Plus sensor health tracking,
event aggregation and auditable self-healing correction.

Runtime dependency: numpy only.
"""

__version__ = "1.0.0"

from .config import DEFAULT_CONFIG, VARIABLES, Config
from .types import (
    AnomalyEvent,
    ConfidenceBand,
    FaultClass,
    HealthStatus,
    Observation,
    QCRecord,
    QCStatus,
    SensorHealth,
    Severity,
    Station,
    StationHealth,
)

__all__ = [
    "__version__",
    "Config",
    "DEFAULT_CONFIG",
    "VARIABLES",
    "Observation",
    "Station",
    "QCRecord",
    "QCStatus",
    "FaultClass",
    "Severity",
    "ConfidenceBand",
    "HealthStatus",
    "SensorHealth",
    "StationHealth",
    "AnomalyEvent",
]
