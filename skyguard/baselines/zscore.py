"""Robust z-score baseline: the honest classical floor.

Per-variable median/MAD standardisation, score = worst variable's |z|,
calibrated by ECDF on clean data. Catches gross outliers and nothing subtle —
which is precisely why the ladder needs it: every fancier row must earn its
place above this line.
"""

from __future__ import annotations

import numpy as np

from . import ecdf_scores, fit_ecdf


class RobustZScore:
    """Univariate robust z-score detector over the three variables."""

    def __init__(self) -> None:
        self.median: np.ndarray = np.zeros(3)
        self.scale: np.ndarray = np.ones(3)
        self.calibration: np.ndarray = np.array([0.0])

    def fit(self, clean_levels: np.ndarray) -> "RobustZScore":
        """Fit location/scale/ECDF on clean (N, 3) levels."""
        mat = np.asarray(clean_levels, dtype=np.float64)
        if mat.ndim != 2 or mat.shape[1] != 3 or mat.shape[0] == 0:
            raise ValueError("clean levels must be non-empty (N, 3)")
        self.median = np.median(mat, axis=0)
        mad = np.median(np.abs(mat - self.median), axis=0)
        self.scale = np.maximum(mad * 1.4826, 1e-9)
        self.calibration = fit_ecdf(self.raw_scores(mat))
        return self

    def raw_scores(self, levels: np.ndarray) -> np.ndarray:
        """Worst-variable |z| per row, uncalibrated."""
        mat = np.asarray(levels, dtype=np.float64)
        return np.max(np.abs((mat - self.median) / self.scale), axis=1)

    def score_points(self, levels: np.ndarray) -> np.ndarray:
        """ECDF-calibrated [0, 1] anomaly scores."""
        return ecdf_scores(self.calibration, self.raw_scores(levels))
