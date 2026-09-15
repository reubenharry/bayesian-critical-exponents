"""PyTensor GP marginal likelihood for Harada-style FSS inference."""

from __future__ import annotations

import pytensor.tensor as pt
from pytensor.tensor import slinalg

from .gp_kernels import (
    DEFAULT_GP_KERNEL,
    GpKernelKind,
    normalize_gp_kernel,
    polynomial_kernel_degree,
)

_LOG2PI = pt.log(2.0 * pt.pi)


def matern52_kernel_1d(
    x1: pt.TensorVariable,
    x2: pt.TensorVariable,
    length_scale: pt.TensorVariable,
    amplitude: pt.TensorVariable,
) -> pt.TensorVariable:
    x1 = x1.flatten()
    x2 = x2.flatten()
    r = pt.abs(x1.dimshuffle(0, "x") - x2.dimshuffle("x", 0))
    ls = pt.maximum(length_scale, 1e-6)
    scaled = pt.sqrt(5.0) * r / ls
    return (amplitude**2) * (1.0 + scaled + (scaled**2) / 3.0) * pt.exp(-scaled)


def gaussian_kernel_1d(
    x1: pt.TensorVariable,
    x2: pt.TensorVariable,
    length_scale: pt.TensorVariable,
    amplitude: pt.TensorVariable,
) -> pt.TensorVariable:
    """Squared-exponential kernel used in Harada BSA (PRE 84, 056704)."""
    x1 = x1.flatten()
    x2 = x2.flatten()
    r = x1.dimshuffle(0, "x") - x2.dimshuffle("x", 0)
    ls = pt.maximum(length_scale, 1e-6)
    return (amplitude**2) * pt.exp(-0.5 * (r / ls) ** 2)


def polynomial_kernel_1d(
    x1: pt.TensorVariable,
    x2: pt.TensorVariable,
    *,
    degree: int,
    length_scale: pt.TensorVariable,
    amplitude: pt.TensorVariable,
) -> pt.TensorVariable:
    """Monomial kernel k(x,x') = eta^2 sum_{k=0}^degree x^k x'^k."""
    del length_scale
    x1 = x1.flatten()
    x2 = x2.flatten()
    k = pt.zeros((x1.shape[0], x2.shape[0]))
    for power in range(degree + 1):
        k = k + (x1**power).dimshuffle(0, "x") * (x2**power).dimshuffle("x", 0)
    return (amplitude**2) * k


def poly2_kernel_1d(
    x1: pt.TensorVariable,
    x2: pt.TensorVariable,
    length_scale: pt.TensorVariable,
    amplitude: pt.TensorVariable,
) -> pt.TensorVariable:
    """Quadratic monomial kernel k(x,x') = eta^2 (1 + x x' + x^2 x'^2)."""
    return polynomial_kernel_1d(
        x1, x2, degree=2, length_scale=length_scale, amplitude=amplitude
    )


def poly4_kernel_1d(
    x1: pt.TensorVariable,
    x2: pt.TensorVariable,
    length_scale: pt.TensorVariable,
    amplitude: pt.TensorVariable,
) -> pt.TensorVariable:
    """Quartic monomial kernel through z^4."""
    return polynomial_kernel_1d(
        x1, x2, degree=4, length_scale=length_scale, amplitude=amplitude
    )


def stationary_kernel_1d(
    kernel: GpKernelKind | str,
    x1: pt.TensorVariable,
    x2: pt.TensorVariable,
    length_scale: pt.TensorVariable,
    amplitude: pt.TensorVariable,
) -> pt.TensorVariable:
    kind = normalize_gp_kernel(kernel) if isinstance(kernel, str) else kernel
    degree = polynomial_kernel_degree(kind)
    if degree is not None:
        return polynomial_kernel_1d(
            x1, x2, degree=degree, length_scale=length_scale, amplitude=amplitude
        )
    if kind == "gaussian":
        return gaussian_kernel_1d(x1, x2, length_scale, amplitude)
    return matern52_kernel_1d(x1, x2, length_scale, amplitude)


def gp_log_marginal_likelihood(
    x: pt.TensorVariable,
    y: pt.TensorVariable,
    sigma: pt.TensorVariable,
    length_scale: pt.TensorVariable,
    amplitude: pt.TensorVariable,
    *,
    kernel: GpKernelKind | str = DEFAULT_GP_KERNEL,
    jitter: float = 1e-5,
) -> pt.TensorVariable:
    """Log marginal likelihood for GP regression with observation noise."""
    x = x.flatten()
    y = y.flatten()
    sigma = sigma.flatten()
    n = x.shape[0]

    k = stationary_kernel_1d(kernel, x, x, length_scale, amplitude)
    k = k + pt.diag(sigma**2 + jitter)

    chol = slinalg.cholesky(k)
    alpha = slinalg.solve_triangular(chol, y, lower=True)
    sol = slinalg.solve_triangular(chol.T, alpha, lower=False)
    log_det = 2.0 * pt.sum(pt.log(pt.diag(chol)))
    return (
        -0.5 * pt.dot(y, sol)
        - 0.5 * log_det
        - 0.5 * n * _LOG2PI
    )


def product_kernel_2d(
    kernel: GpKernelKind | str,
    L1: pt.TensorVariable,
    T1: pt.TensorVariable,
    L2: pt.TensorVariable,
    T2: pt.TensorVariable,
    *,
    length_scale_L: pt.TensorVariable,
    length_scale_T: pt.TensorVariable,
    amplitude: pt.TensorVariable,
) -> pt.TensorVariable:
    """Separable 1D kernel on (L, T): k = eta^2 * k_L(L) * k_T(T)."""
    unit = pt.ones_like(length_scale_L)
    k_l = stationary_kernel_1d(kernel, L1, L2, length_scale_L, unit)
    k_t = stationary_kernel_1d(kernel, T1, T2, length_scale_T, unit)
    return (amplitude**2) * k_l * k_t


def matern52_product_kernel_2d(
    L1: pt.TensorVariable,
    T1: pt.TensorVariable,
    L2: pt.TensorVariable,
    T2: pt.TensorVariable,
    *,
    length_scale_L: pt.TensorVariable,
    length_scale_T: pt.TensorVariable,
    amplitude: pt.TensorVariable,
) -> pt.TensorVariable:
    """Separable Matern-5/2 kernel: k = eta^2 * k_L(L) * k_T(T)."""
    return product_kernel_2d(
        "matern52",
        L1,
        T1,
        L2,
        T2,
        length_scale_L=length_scale_L,
        length_scale_T=length_scale_T,
        amplitude=amplitude,
    )


def gp_log_marginal_likelihood_2d(
    L: pt.TensorVariable,
    T: pt.TensorVariable,
    y: pt.TensorVariable,
    sigma: pt.TensorVariable,
    *,
    length_scale_L: pt.TensorVariable,
    length_scale_T: pt.TensorVariable,
    amplitude: pt.TensorVariable,
    kernel: GpKernelKind | str = DEFAULT_GP_KERNEL,
    jitter: float = 1e-5,
) -> pt.TensorVariable:
    """GP marginal likelihood with a product kernel on (L, T)."""
    L = L.flatten()
    T = T.flatten()
    y = y.flatten()
    sigma = sigma.flatten()
    n = L.shape[0]

    k = product_kernel_2d(
        kernel,
        L,
        T,
        L,
        T,
        length_scale_L=length_scale_L,
        length_scale_T=length_scale_T,
        amplitude=amplitude,
    )
    k = k + pt.diag(sigma**2 + jitter)

    chol = slinalg.cholesky(k)
    alpha = slinalg.solve_triangular(chol, y, lower=True)
    sol = slinalg.solve_triangular(chol.T, alpha, lower=False)
    log_det = 2.0 * pt.sum(pt.log(pt.diag(chol)))
    return (
        -0.5 * pt.dot(y, sol)
        - 0.5 * log_det
        - 0.5 * n * _LOG2PI
    )


def gp_conditional_mean(
    x_train: pt.TensorVariable,
    y_train: pt.TensorVariable,
    x_new: pt.TensorVariable,
    length_scale: pt.TensorVariable,
    amplitude: pt.TensorVariable,
    *,
    kernel: GpKernelKind | str = DEFAULT_GP_KERNEL,
    jitter: float = 1e-6,
) -> pt.TensorVariable:
    """GP regression posterior mean at x_new given noise-free latents at x_train."""
    x_train = x_train.flatten()
    y_train = y_train.flatten()
    x_new = x_new.flatten()

    k_tt = stationary_kernel_1d(kernel, x_train, x_train, length_scale, amplitude)
    k_tt = k_tt + pt.eye(x_train.shape[0]) * jitter
    k_st = stationary_kernel_1d(kernel, x_new, x_train, length_scale, amplitude)

    chol = slinalg.cholesky(k_tt)
    alpha = slinalg.solve_triangular(chol, y_train, lower=True)
    sol = slinalg.solve_triangular(chol.T, alpha, lower=False)
    return pt.dot(k_st, sol)
