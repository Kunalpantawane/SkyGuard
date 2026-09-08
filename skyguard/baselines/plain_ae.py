"""Plain non-temporal autoencoder baseline: 3 -> H -> 3 MLP, numpy backprop.

Same reconstruction idea as Layer 2 minus time: it learns the shape of normal
*points* but not normal *trajectories*. The LSTM-AE row must beat this one —
that gap is the evidence that sequence behaviour (not just values) carries
the signal.
"""

from __future__ import annotations

import numpy as np

from . import ecdf_scores, fit_ecdf


class PlainAutoencoder:
    """Single-hidden-layer tanh autoencoder over the three variables."""

    def __init__(self, hidden: int = 8, learning_rate: float = 0.05,
                 epochs: int = 200, seed: int = 0) -> None:
        self.hidden = hidden
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed
        self.mean: np.ndarray = np.zeros(3)
        self.scale: np.ndarray = np.ones(3)
        self.weights: dict = {}
        self.calibration: np.ndarray = np.array([0.0])

    def fit(self, clean_levels: np.ndarray) -> "PlainAutoencoder":
        """Train on clean (N, 3) levels with full-batch gradient descent."""
        mat = np.asarray(clean_levels, dtype=np.float64)
        if mat.ndim != 2 or mat.shape[1] != 3 or mat.shape[0] < 8:
            raise ValueError("need at least 8 clean rows of 3 variables")
        self.mean = mat.mean(axis=0)
        self.scale = np.maximum(mat.std(axis=0), 1e-9)
        target = (mat - self.mean) / self.scale
        rng = np.random.default_rng(self.seed)
        self.weights = {
            "W1": rng.normal(0, np.sqrt(1.0 / 3), size=(3, self.hidden)),
            "b1": np.zeros(self.hidden),
            "W2": rng.normal(0, np.sqrt(1.0 / self.hidden), size=(self.hidden, 3)),
            "b2": np.zeros(3),
        }
        n = target.shape[0]
        for _ in range(self.epochs):
            hidden = np.tanh(target @ self.weights["W1"] + self.weights["b1"])
            out = hidden @ self.weights["W2"] + self.weights["b2"]
            error = (out - target) / n
            grad_W2 = hidden.T @ error
            grad_b2 = error.sum(axis=0)
            back = (error @ self.weights["W2"].T) * (1.0 - hidden * hidden)
            grad_W1 = target.T @ back
            grad_b1 = back.sum(axis=0)
            for key, grad in (("W1", grad_W1), ("b1", grad_b1),
                              ("W2", grad_W2), ("b2", grad_b2)):
                self.weights[key] -= self.learning_rate * grad
        self.calibration = fit_ecdf(self.raw_scores(mat))
        return self

    def reconstruct(self, levels: np.ndarray) -> np.ndarray:
        """Standardised-space reconstruction of each row."""
        mat = (np.asarray(levels, dtype=np.float64) - self.mean) / self.scale
        hidden = np.tanh(mat @ self.weights["W1"] + self.weights["b1"])
        return hidden @ self.weights["W2"] + self.weights["b2"]

    def raw_scores(self, levels: np.ndarray) -> np.ndarray:
        """Mean-squared reconstruction error per row, uncalibrated."""
        mat = (np.asarray(levels, dtype=np.float64) - self.mean) / self.scale
        return np.mean((self.reconstruct(levels) - mat) ** 2, axis=1)

    def score_points(self, levels: np.ndarray) -> np.ndarray:
        """ECDF-calibrated [0, 1] anomaly scores."""
        if not self.weights:
            raise ValueError("autoencoder is not fitted")
        return ecdf_scores(self.calibration, self.raw_scores(levels))
