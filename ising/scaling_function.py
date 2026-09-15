"""Pluggable priors/likelihoods for the universal scaling function."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pytensor.tensor as pt

from ._deps import pm as _pm
from .constants import NU_EXACT, TC_EXACT
from .gp_kernels import DEFAULT_GP_KERNEL, GpKernelKind
from .gp_pt import gp_log_marginal_likelihood

ScalingBackend = Literal["gp", "spline"]
GP_JITTER = 1e-5
DEFAULT_SPLINE_DEGREE = 3
DEFAULT_N_SPLINE_COEFFS = 20


@dataclass(frozen=True)
class ScalingFunctionConfig:
    backend: ScalingBackend = "gp"
    n_spline_coeffs: int = DEFAULT_N_SPLINE_COEFFS
    spline_degree: int = DEFAULT_SPLINE_DEGREE
    spline_coeff_sigma: float = 2.0
    spline_rw_sigma: float = 1.0
    knot_pad_fraction: float = 0.6
    gp_ell_prior_mu: float | None = None
    gp_ell_prior_sigma: float = 1.0
    gp_ell_fixed: float | None = None
    gp_eta_fixed: float | None = None
    gp_kernel: GpKernelKind = DEFAULT_GP_KERNEL


def median_nearest_neighbor_spacing(z: np.ndarray) -> float:
    """Characteristic z spacing for calibrating GP length-scale priors."""
    z = np.sort(np.unique(np.asarray(z, dtype=np.float64).ravel()))
    if z.size < 2:
        return 1.0
    left = np.concatenate([[np.inf], np.diff(z)])
    right = np.concatenate([np.diff(z), [np.inf]])
    nn = np.minimum(left, right)
    nn = nn[np.isfinite(nn) & (nn > 0)]
    return float(np.median(nn)) if nn.size else max(float(z.max() - z.min()), 1.0)


def provisional_z(
    L: np.ndarray,
    T: np.ndarray,
    *,
    T_c: float = TC_EXACT,
    nu: float = NU_EXACT,
) -> np.ndarray:
    L = np.asarray(L, dtype=np.float64).ravel()
    T = np.asarray(T, dtype=np.float64).ravel()
    t = (T - T_c) / T_c
    return t * L ** (1.0 / nu)


def clamped_knot_vector(
    z_min: float,
    z_max: float,
    *,
    n_coeffs: int,
    degree: int,
) -> np.ndarray:
    if z_max <= z_min:
        z_max = z_min + 1.0
    n_interior = max(n_coeffs - degree + 1, 2)
    interior = np.linspace(z_min, z_max, n_interior)
    return np.concatenate(
        [
            np.full(degree + 1, z_min),
            interior[1:-1],
            np.full(degree + 1, z_max),
        ]
    ).astype(np.float64)


def default_z_knots(
    L: np.ndarray,
    T: np.ndarray,
    *,
    n_coeffs: int,
    degree: int,
    pad_fraction: float,
) -> np.ndarray:
    z = provisional_z(L, T)
    span = float(z.max() - z.min())
    pad = max(pad_fraction * span, 1.0)
    return clamped_knot_vector(
        float(z.min()) - pad,
        float(z.max()) + pad,
        n_coeffs=n_coeffs,
        degree=degree,
    )


def _bspline_basis_scalar(
    x: pt.TensorVariable,
    knots: pt.TensorVariable,
    index: int,
    degree: int,
) -> pt.TensorVariable:
    if degree == 0:
        left = knots[index]
        right = knots[index + 1]
        on_right = pt.switch(
            pt.eq(index, knots.shape[0] - 2),
            pt.and_(x >= left, x <= right),
            pt.and_(x >= left, x < right),
        )
        return pt.switch(on_right, 1.0, 0.0)

    left_den = knots[index + degree] - knots[index]
    right_den = knots[index + degree + 1] - knots[index + 1]
    left_term = pt.switch(
        pt.abs(left_den) < 1e-12,
        0.0,
        (x - knots[index]) / left_den
        * _bspline_basis_scalar(x, knots, index, degree - 1),
    )
    right_term = pt.switch(
        pt.abs(right_den) < 1e-12,
        0.0,
        (knots[index + degree + 1] - x) / right_den
        * _bspline_basis_scalar(x, knots, index + 1, degree - 1),
    )
    return left_term + right_term


def bspline_basis_matrix_pt(
    z: pt.TensorVariable,
    knots: np.ndarray,
    *,
    degree: int,
) -> pt.TensorVariable:
    z = z.flatten()
    knots_pt = pt.as_tensor_variable(knots.astype(np.float64))
    n_coeffs = knots.size - degree - 1
    cols = [
        _bspline_basis_scalar(z, knots_pt, i, degree) for i in range(n_coeffs)
    ]
    return pt.stack(cols, axis=1)


def bspline_eval_pt(
    z: pt.TensorVariable,
    knots: np.ndarray,
    coeffs: pt.TensorVariable,
    *,
    degree: int,
) -> pt.TensorVariable:
    knots_arr = np.asarray(knots, dtype=np.float64)
    z_min = float(knots_arr[degree])
    z_max = float(knots_arr[-degree - 1])
    z_eval = pt.clip(z.flatten(), z_min, z_max)
    basis = bspline_basis_matrix_pt(z_eval, knots_arr, degree=degree)
    return pt.dot(basis, coeffs.flatten())


def _bspline_basis_scalar_np(
    x: float,
    knots: np.ndarray,
    index: int,
    degree: int,
) -> float:
    if degree == 0:
        left = knots[index]
        right = knots[index + 1]
        if index == knots.size - 2:
            return 1.0 if left <= x <= right else 0.0
        return 1.0 if left <= x < right else 0.0

    left_den = knots[index + degree] - knots[index]
    right_den = knots[index + degree + 1] - knots[index + 1]
    left_term = 0.0
    if abs(left_den) > 1e-12:
        left_term = (x - knots[index]) / left_den * _bspline_basis_scalar_np(
            x, knots, index, degree - 1
        )
    right_term = 0.0
    if abs(right_den) > 1e-12:
        right_term = (knots[index + degree + 1] - x) / right_den * _bspline_basis_scalar_np(
            x, knots, index + 1, degree - 1
        )
    return left_term + right_term


def bspline_basis_matrix_np(
    z: np.ndarray,
    knots: np.ndarray,
    *,
    degree: int,
) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64).ravel()
    n_coeffs = knots.size - degree - 1
    out = np.empty((z.size, n_coeffs), dtype=np.float64)
    for row, x in enumerate(z):
        for col in range(n_coeffs):
            out[row, col] = _bspline_basis_scalar_np(float(x), knots, col, degree)
    return out


def bspline_eval_np(
    z: np.ndarray,
    knots: np.ndarray,
    coeffs: np.ndarray,
    *,
    degree: int,
) -> np.ndarray:
    knots = np.asarray(knots, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64).ravel()
    z_eval = np.clip(z, knots[degree], knots[-degree - 1])
    return bspline_basis_matrix_np(z_eval, knots, degree=degree) @ np.asarray(
        coeffs, dtype=np.float64
    ).ravel()


def add_gp_scaling_likelihood(
    z: pt.TensorVariable,
    log_target: pt.TensorVariable,
    sigma_log: pt.TensorVariable,
    *,
    gp_ell,
    gp_eta,
    kernel: GpKernelKind | str = DEFAULT_GP_KERNEL,
    jitter: float = GP_JITTER,
    name: str = "gp_like",
) -> None:
    log_ml = gp_log_marginal_likelihood(
        z,
        log_target,
        sigma_log,
        gp_ell,
        gp_eta,
        kernel=kernel,
        jitter=jitter,
    )
    _pm.Potential(name, log_ml)


def add_spline_scaling_likelihood(
    z: pt.TensorVariable,
    log_target: pt.TensorVariable,
    sigma_log: pt.TensorVariable,
    *,
    knots: np.ndarray,
    degree: int,
    coeff_sigma: float,
    rw_sigma: float,
) -> pt.TensorVariable:
    n_coeffs = knots.size - degree - 1
    coeffs = _pm.Normal(
        "spline_coeff",
        mu=0.0,
        sigma=coeff_sigma,
        shape=n_coeffs,
    )
    mu = bspline_eval_pt(z, knots, coeffs, degree=degree)
    _dist = _pm.Normal.dist(mu=mu, sigma=sigma_log)
    _pm.Potential(
        "spline_like",
        _pm.logp(_dist, log_target).sum(),
    )
    if n_coeffs > 1:
        _pm.Potential(
            "spline_smooth",
            -0.5 * pt.sum(pt.diff(coeffs) ** 2) / (rw_sigma**2),
        )
    return coeffs


def add_scaling_function(
    z: pt.TensorVariable,
    log_target: pt.TensorVariable,
    sigma_log: pt.TensorVariable,
    *,
    config: ScalingFunctionConfig,
    knots: np.ndarray | None = None,
) -> dict[str, object]:
    """Attach a GP or spline distribution for the smooth part of log Phi / log psi."""
    if config.backend == "gp":
        out: dict[str, object] = {"knots": None}
        if config.gp_ell_fixed is not None:
            gp_ell = pt.as_tensor_variable(config.gp_ell_fixed)
        elif config.gp_ell_prior_mu is not None:
            gp_ell = _pm.Lognormal(
                "gp_ell",
                mu=config.gp_ell_prior_mu,
                sigma=config.gp_ell_prior_sigma,
            )
            out["gp_ell"] = gp_ell
        else:
            gp_ell = _pm.HalfNormal("gp_ell", sigma=2.0)
            out["gp_ell"] = gp_ell

        if config.gp_eta_fixed is not None:
            gp_eta = pt.as_tensor_variable(config.gp_eta_fixed)
        else:
            gp_eta = _pm.HalfNormal("gp_eta", sigma=2.0)
            out["gp_eta"] = gp_eta

        add_gp_scaling_likelihood(
            z,
            log_target,
            sigma_log,
            gp_ell=gp_ell,
            gp_eta=gp_eta,
            kernel=config.gp_kernel,
        )
        return out

    if config.backend == "spline":
        if knots is None:
            raise ValueError("knots are required for spline scaling backend")
        add_spline_scaling_likelihood(
            z,
            log_target,
            sigma_log,
            knots=knots,
            degree=config.spline_degree,
            coeff_sigma=config.spline_coeff_sigma,
            rw_sigma=config.spline_rw_sigma,
        )
        return {"knots": knots}

    raise ValueError(f"scaling backend must be 'gp' or 'spline', got {config.backend!r}")


def eval_scaling_curve_np(
    z_train: np.ndarray,
    log_target_train: np.ndarray,
    z_new: np.ndarray,
    *,
    backend: ScalingBackend,
    draw: dict[str, float | np.ndarray],
    knots: np.ndarray | None = None,
    spline_degree: int = DEFAULT_SPLINE_DEGREE,
) -> np.ndarray:
    """Evaluate the fitted smooth scaling function on a grid (post-processing)."""
    z_new = np.asarray(z_new, dtype=np.float64).ravel()
    if backend == "gp":
        from .gp_utils import gp_latent_conditional

        mean, _ = gp_latent_conditional(
            np.asarray(z_train, dtype=np.float64),
            np.asarray(log_target_train, dtype=np.float64),
            z_new,
            length_scale=float(draw["gp_ell"]),
            amplitude=float(draw["gp_eta"]),
        )
        return mean

    if backend == "spline":
        if knots is None:
            raise ValueError("knots are required for spline evaluation")
        return bspline_eval_np(
            z_new,
            knots,
            np.asarray(draw["spline_coeff"], dtype=np.float64),
            degree=spline_degree,
        )

    raise ValueError(f"unknown scaling backend {backend!r}")
