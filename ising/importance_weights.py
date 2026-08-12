"""Pareto-smoothed importance weights and weighted posterior summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ._deps import az

UncertaintyMetric = Literal["trace_cov", "max_marginal_var"]


@dataclass(frozen=True)
class ImportanceWeightResult:
    """Smoothed importance weights and diagnostics."""

    log_weights: np.ndarray
    weights: np.ndarray
    ess: float
    pareto_k: float


def effective_sample_size(weights: np.ndarray) -> float:
    """Normalized-weight effective sample size."""
    w = np.asarray(weights, dtype=np.float64).ravel()
    total = w.sum()
    if total <= 0.0 or not np.isfinite(total):
        return 0.0
    w = w / total
    denom = np.sum(w**2)
    if denom <= 0.0:
        return 0.0
    return float(1.0 / denom)


def pareto_smooth_log_weights(
    log_weights: np.ndarray,
    *,
    reff: float = 1.0,
) -> ImportanceWeightResult:
    """Apply PSIS to unnormalized log importance weights (1D sample axis)."""
    log_w = np.asarray(log_weights, dtype=np.float64).ravel()
    if log_w.size == 0:
        return ImportanceWeightResult(
            log_weights=log_w,
            weights=np.array([], dtype=np.float64),
            ess=0.0,
            pareto_k=np.nan,
        )

    lw_smoothed, k = az.psislw(log_w[None, :], reff=reff, normalize=True)
    log_smoothed = np.asarray(lw_smoothed[0], dtype=np.float64)
    max_log = np.max(log_smoothed)
    weights = np.exp(log_smoothed - max_log)
    weights /= weights.sum()
    k_val = float(np.asarray(k).ravel()[0])
    return ImportanceWeightResult(
        log_weights=log_smoothed,
        weights=weights,
        ess=effective_sample_size(weights),
        pareto_k=k_val,
    )


def weighted_mean_cov(
    samples: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted mean and covariance for sample matrix (n_samples, n_dim)."""
    x = np.asarray(samples, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).ravel()
    if x.ndim != 2:
        raise ValueError("samples must have shape (n_samples, n_dim)")
    if x.shape[0] != w.size:
        raise ValueError("weights length must match number of samples")
    w = w / np.sum(w)
    mean = w @ x
    centered = x - mean
    cov = centered.T @ (centered * w[:, None])
    return mean, cov


def uncertainty_scalar(
    cov: np.ndarray,
    *,
    metric: UncertaintyMetric = "trace_cov",
) -> float:
    cov = np.asarray(cov, dtype=np.float64)
    if metric == "trace_cov":
        return float(np.trace(cov))
    if metric == "max_marginal_var":
        return float(np.max(np.diag(cov)))
    raise ValueError(f"Unknown uncertainty metric: {metric!r}")


def weighted_uncertainty(
    samples: np.ndarray,
    weights: np.ndarray,
    *,
    metric: UncertaintyMetric = "trace_cov",
) -> float:
    _, cov = weighted_mean_cov(samples, weights)
    return uncertainty_scalar(cov, metric=metric)
