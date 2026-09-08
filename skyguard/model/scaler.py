"""Per-station robust scaling and LSTM window assembly.

Why per-station: a Himalayan and a coastal station share one LSTM without one
dominating the loss only if each speaks in its own z-scores. Median and MAD
(not mean/std) so a few dirty points in the fitting history cannot shift the
scale and hide real faults. Time features ride alongside so the model knows
38 C at 14:00 differs from 38 C at 04:00.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Sequence

import numpy as np

from ..config import VARIABLES
from .lstm import make_model_input


@dataclass
class StationScaler:
    """Robust median/MAD standardiser for one station's three variables."""

    station_id: str
    median: np.ndarray
    scale: np.ndarray
    n_samples: int = 0

    def transform(self, levels: np.ndarray) -> np.ndarray:
        """Physical (..., 3) -> z-scores in VARIABLES order."""
        return (np.asarray(levels, dtype=np.float64) - self.median) / self.scale

    def inverse(self, z_scores: np.ndarray) -> np.ndarray:
        """Z-scores -> physical units."""
        return np.asarray(z_scores, dtype=np.float64) * self.scale + self.median

    def to_dict(self) -> Dict:
        return {
            "station_id": self.station_id,
            "median": self.median.tolist(),
            "scale": self.scale.tolist(),
            "n_samples": self.n_samples,
        }


def fit_scaler(station_id: str, history: np.ndarray) -> StationScaler:
    """Fit from clean level history, shape (N, 3) in VARIABLES order."""
    levels = np.asarray(history, dtype=np.float64)
    if levels.ndim != 2 or levels.shape[1] != 3:
        raise ValueError(f"history must be (N, 3), got {levels.shape}")
    if not np.all(np.isfinite(levels)):
        raise ValueError("scaler fit requires finite clean history")
    median = np.median(levels, axis=0)
    mad = np.median(np.abs(levels - median), axis=0)
    scale = np.maximum(mad * 1.4826, 1e-6)
    return StationScaler(station_id=station_id, median=median, scale=scale,
                         n_samples=levels.shape[0])


def fit_all_scalers(histories: Dict[str, np.ndarray]) -> Dict[str, StationScaler]:
    """Fit one scaler per station from its clean history."""
    return {station_id: fit_scaler(station_id, history) for station_id, history in histories.items()}


def build_window(
    scaler: StationScaler, levels: np.ndarray, timestamps: Sequence[datetime]
) -> np.ndarray:
    """Assemble one (W, 7) model window: standardised physics + time context.

    `levels` (W, 3) physical in VARIABLES order, `timestamps` the matching
    UTC stamps. Hour and day-of-year come from the stamps themselves — no
    external calendar needed at inference time.
    """
    phys = np.asarray(levels, dtype=np.float64)
    stamps: List[datetime] = list(timestamps)
    if phys.shape[0] != len(stamps):
        raise ValueError("levels and timestamps must share their length")
    hours = np.array([s.hour + s.minute / 60.0 + s.second / 3600.0 for s in stamps])
    doys = np.array([float(s.timetuple().tm_yday) for s in stamps])
    return make_model_input(scaler.transform(phys), hours, doys)
