"""Local Outlier Factor baseline, numpy-only.

Density comparison: a point whose neighbours are much denser than it is scores
high. O(N^2) on the fit set, so this stays a small-scale baseline — it earns
its row on the ladder (density perspective) without pretending to scale.
"""

from __future__ import annotations

import numpy as np

from . import ecdf_scores, fit_ecdf


class LocalOutlierFactor:
    """LOF over the three raw variables with k neighbours."""

    def __init__(self, k: int = 10) -> None:
        self.k = max(1, int(k))
        self.reference: np.ndarray = np.zeros((0, 3))
        self.reach_densities: np.ndarray = np.zeros(0)
        self.k_distances: np.ndarray = np.zeros(0)
        self.calibration: np.ndarray = np.array([0.0])

    def fit(self, clean_levels: np.ndarray) -> "LocalOutlierFactor":
        """Memorise clean (N, 3) levels and their reachability densities."""
        mat = np.asarray(clean_levels, dtype=np.float64)
        if mat.ndim != 2 or mat.shape[1] != 3 or mat.shape[0] <= self.k:
            raise ValueError(f"need more than k={self.k} clean rows")
        self.reference = mat
        dist = self._distances(mat, mat)
        np.fill_diagonal(dist, np.inf)
        order = np.argsort(dist, axis=1, kind="stable")[:, : self.k]
        self.k_distances = np.take_along_axis(dist, order[:, -1:], axis=1).ravel()
        self.reach_densities = np.array([
            1.0 / max(np.mean(np.maximum(dist[nbr, r], self.k_distances[nbr])), 1e-12)
            for r, nbr in enumerate(order)
        ])
        self.calibration = fit_ecdf(self.raw_scores(mat))
        return self

    @staticmethod
    def _distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        diff = a[:, None, :] - b[None, :, :]
        return np.sqrt(np.maximum((diff * diff).sum(axis=2), 0.0))

    def raw_scores(self, levels: np.ndarray) -> np.ndarray:
        """LOF ratio per row, uncalibrated (1.0 = as dense as neighbours)."""
        mat = np.asarray(levels, dtype=np.float64)
        dist = self._distances(mat, self.reference)
        order = np.argsort(dist, axis=1, kind="stable")[:, : self.k]
        out = np.empty(mat.shape[0], dtype=np.float64)
        for r, nbr in enumerate(order):
            # Reachability of the query point from each neighbour's density.
            reach = np.maximum(dist[r, nbr], self.k_distances[nbr])
            density = 1.0 / max(float(np.mean(reach)), 1e-12)
            out[r] = float(np.mean(self.reach_densities[nbr]) / max(density, 1e-12))
        return out

    def score_points(self, levels: np.ndarray) -> np.ndarray:
        """ECDF-calibrated [0, 1] anomaly scores."""
        if self.reference.shape[0] == 0:
            raise ValueError("LOF is not fitted")
        return ecdf_scores(self.calibration, self.raw_scores(levels))
