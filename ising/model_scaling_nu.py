"""Collapsed GP marginal likelihoods for the FSS model (f0 and f0 + L^-omega f1)."""

from __future__ import annotations

import pytensor.tensor as pt
from pytensor.tensor import slinalg

from ._deps import pm as _pm
from .gp_kernels import DEFAULT_GP_KERNEL, GpKernelKind
from .gp_pt import stationary_kernel_1d
from .scaling_function import GP_JITTER

GP_ELL_STIFF_FACTOR = 2.0
CORRECTION_BASE_GP_ETA_FIXED = 2.0
CORRECTION_GP_ELL_FACTOR = 1.0
CORRECTION_GP_ETA_FIXED = 1.0

# T_c uniform prior bounds (used by model_fss when infer_Tc=True).
TC_PRIOR_LOWER = 2.0
TC_PRIOR_UPPER = 2.5


def _add_correction_gp_likelihood(
    z: pt.TensorVariable,
    target: pt.TensorVariable,
    sigma: pt.TensorVariable,
    L: pt.TensorVariable,
    *,
    omega: pt.TensorVariable,
    gp_ell,
    gp_eta,
    correction_gp_ell,
    correction_gp_eta,
    kernel: GpKernelKind | str = DEFAULT_GP_KERNEL,
    jitter: float = GP_JITTER,
    name: str = "correction_gp_like",
) -> None:
    """GP marginal likelihood for f0(z) + L^-omega f1(z)."""
    z = z.flatten()
    target = target.flatten()
    sigma = sigma.flatten()
    L = L.flatten()
    n = z.shape[0]
    gp_ell_pt = pt.as_tensor_variable(gp_ell)
    gp_eta_pt = pt.as_tensor_variable(gp_eta)
    correction_gp_ell_pt = pt.as_tensor_variable(correction_gp_ell)
    correction_gp_eta_pt = pt.as_tensor_variable(correction_gp_eta)

    k0 = stationary_kernel_1d(
        kernel,
        z,
        z,
        length_scale=gp_ell_pt,
        amplitude=gp_eta_pt,
    )
    k1 = stationary_kernel_1d(
        kernel,
        z,
        z,
        length_scale=correction_gp_ell_pt,
        amplitude=correction_gp_eta_pt,
    )
    correction_scale = L ** (-omega)
    k = k0 + correction_scale.dimshuffle(0, "x") * k1 * correction_scale.dimshuffle("x", 0)
    k = k + pt.diag(sigma**2 + jitter)

    chol = slinalg.cholesky(k)
    alpha = slinalg.solve_triangular(chol, target, lower=True)
    sol = slinalg.solve_triangular(chol.T, alpha, lower=False)
    log_det = 2.0 * pt.sum(pt.log(pt.diag(chol)))
    log_ml = -0.5 * pt.dot(target, sol) - 0.5 * log_det - 0.5 * n * pt.log(
        2.0 * pt.pi
    )
    _pm.Potential(name, log_ml)


def _add_plain_gp_likelihood(
    z: pt.TensorVariable,
    target: pt.TensorVariable,
    sigma: pt.TensorVariable,
    *,
    gp_ell,
    gp_eta,
    kernel: GpKernelKind | str = DEFAULT_GP_KERNEL,
    jitter: float = GP_JITTER,
    name: str = "plain_gp_like",
) -> None:
    """GP marginal likelihood for f0(z) on linear targets."""
    z = z.flatten()
    target = target.flatten()
    sigma = sigma.flatten()
    n = z.shape[0]
    gp_ell_pt = pt.as_tensor_variable(gp_ell)
    gp_eta_pt = pt.as_tensor_variable(gp_eta)

    k = stationary_kernel_1d(
        kernel,
        z,
        z,
        length_scale=gp_ell_pt,
        amplitude=gp_eta_pt,
    )
    k = k + pt.diag(sigma**2 + jitter)

    chol = slinalg.cholesky(k)
    alpha = slinalg.solve_triangular(chol, target, lower=True)
    sol = slinalg.solve_triangular(chol.T, alpha, lower=False)
    log_det = 2.0 * pt.sum(pt.log(pt.diag(chol)))
    log_ml = -0.5 * pt.dot(target, sol) - 0.5 * log_det - 0.5 * n * pt.log(
        2.0 * pt.pi
    )
    _pm.Potential(name, log_ml)
