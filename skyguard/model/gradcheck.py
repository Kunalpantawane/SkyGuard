"""Finite-difference gradient gate for the numpy LSTM.

Why this module exists: the deployed system has no framework to cross-check
against, so BPTT correctness is proven by comparing analytic gradients with
central differences. Per `context/conventions.md`, anything with gradients
gets this check before it is trusted — the test suite fails if BPTT drifts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .lstm import PARAM_KEYS, LstmAutoencoder


@dataclass
class GradCheckReport:
    """Outcome of one gradient check run."""

    passed: bool = False
    max_rel_error: float = float("inf")
    per_param: Dict[str, float] = field(default_factory=dict)
    tolerance: float = 1e-5
    detail: str = ""


def _relative_error(analytic: float, numeric: float) -> float:
    """Scale-invariant error; falls back to absolute when both are ~zero."""
    denom = max(1.0, abs(analytic), abs(numeric))
    return abs(analytic - numeric) / denom


def check_gradients(
    model: LstmAutoencoder,
    inputs: np.ndarray,
    targets: np.ndarray,
    epsilon: float = 1e-5,
    tolerance: float = 1e-5,
    checks_per_param: int = 4,
    seed: int = 0,
) -> GradCheckReport:
    """Compare BPTT gradients against central differences on a few entries.

    Only `checks_per_param` entries per parameter are probed (full numeric
    gradients would cost a forward pass per weight). Entries are chosen by a
    seeded RNG so the gate is deterministic and reproducible.
    """
    xb = np.asarray(inputs, dtype=np.float64)
    yb = np.asarray(targets, dtype=np.float64)
    rng = np.random.default_rng(seed)

    _, _, cache = model.forward(xb, yb)
    analytic = model.backward(xb, yb, cache)

    per_param: Dict[str, float] = {}
    worst = 0.0
    worst_where = ""
    for key in PARAM_KEYS:
        param = model.params[key]
        grad = analytic[key]
        flat_param = param.ravel()
        flat_grad = grad.ravel()
        n_check = min(checks_per_param, flat_param.size)
        indices = rng.choice(flat_param.size, size=n_check, replace=False)
        worst_param = 0.0
        for idx in indices:
            original = flat_param[int(idx)]
            flat_param[int(idx)] = original + epsilon
            loss_plus = model.loss(xb, yb)
            flat_param[int(idx)] = original - epsilon
            loss_minus = model.loss(xb, yb)
            flat_param[int(idx)] = original
            numeric = (loss_plus - loss_minus) / (2.0 * epsilon)
            err = _relative_error(float(flat_grad[int(idx)]), float(numeric))
            worst_param = max(worst_param, err)
            if err > worst:
                worst = err
                worst_where = f"{key}[{int(idx)}] analytic={float(flat_grad[int(idx)]):.6e} numeric={numeric:.6e}"
        per_param[key] = worst_param

    passed = worst <= tolerance
    detail = f"max_rel_error={worst:.3e} at {worst_where}" if worst_where else "no entries checked"
    return GradCheckReport(
        passed=passed,
        max_rel_error=worst,
        per_param=per_param,
        tolerance=tolerance,
        detail=detail,
    )


def checked_window_data(
    window: int = 4, n_features: int = 7, batch: int = 2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Small deterministic batch for the gradient gate.

    Smooth sinusoids, not white noise: the check should exercise realistic
    gate regimes rather than saturated corners.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(window, dtype=np.float64)
    xb = np.empty((batch, window, n_features), dtype=np.float64)
    for b in range(batch):
        phase = rng.uniform(0, 2 * np.pi)
        xb[b, :, 0] = np.sin(0.5 * t + phase)
        xb[b, :, 1] = np.cos(0.3 * t + phase)
        xb[b, :, 2] = np.sin(0.2 * t - phase)
        xb[b, :, 3] = np.sin(2 * np.pi * (t % 24) / 24.0)
        xb[b, :, 4] = np.cos(2 * np.pi * (t % 24) / 24.0)
        xb[b, :, 5] = np.sin(2 * np.pi * (t % 365) / 365.0)
        xb[b, :, 6] = np.cos(2 * np.pi * (t % 365) / 365.0)
    yb = xb[:, :, :3] + 0.05 * rng.standard_normal((batch, window, 3))
    return xb, yb


def gate_summary(report: GradCheckReport) -> str:
    """One-line human summary for logs and CI output."""
    status = "PASS" if report.passed else "FAIL"
    lines: List[str] = [f"gradcheck {status}: {report.detail} (tol={report.tolerance:.0e})"]
    for key in PARAM_KEYS:
        lines.append(f"  {key}: rel_err={report.per_param.get(key, float('nan')):.3e}")
    return "\n".join(lines)
