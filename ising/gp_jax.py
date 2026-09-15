"""JAX GP marginal likelihood kernels (mirrors :mod:`gp_utils`)."""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp

from .gp_kernels import (
    DEFAULT_GP_KERNEL,
    GpKernelKind,
    normalize_gp_kernel,
    polynomial_kernel_degree,
)

_LOG2PI = jnp.log(2.0 * jnp.pi)

DiscCoupling = Literal["additive", "mixture", "weighted"]


def _discrepancy_weights(
    a: jax.Array,
    coupling: DiscCoupling | str,
) -> tuple[jax.Array, jax.Array]:
    """Return ``(w_f, w_g)`` so ``y = w_f f + w_g g + ε``.

    ``a`` is the collapsed g-coefficient (additive) or the unit gate ``π``
    (mixture / weighted).
    """
    a = jnp.ravel(a)
    ones = jnp.ones_like(a)
    if coupling == "weighted":
        return ones - a, jnp.zeros_like(a)
    if coupling == "mixture":
        return ones - a, a
    return ones, a


def matern52_kernel(
    x1: jax.Array,
    x2: jax.Array,
    *,
    length_scale: jax.Array | float,
    amplitude: jax.Array | float,
) -> jax.Array:
    x1 = jnp.ravel(x1)
    x2 = jnp.ravel(x2)
    r = jnp.abs(x1[:, None] - x2[None, :])
    ls = jnp.maximum(jnp.asarray(length_scale), 1e-6)
    amp = jnp.asarray(amplitude)
    sqrt5 = jnp.sqrt(5.0)
    scaled = sqrt5 * r / ls
    return (amp**2) * (1.0 + scaled + (scaled**2) / 3.0) * jnp.exp(-scaled)


def gaussian_kernel(
    x1: jax.Array,
    x2: jax.Array,
    *,
    length_scale: jax.Array | float,
    amplitude: jax.Array | float,
) -> jax.Array:
    """Squared-exponential kernel used in Harada BSA (PRE 84, 056704)."""
    x1 = jnp.ravel(x1)
    x2 = jnp.ravel(x2)
    r = x1[:, None] - x2[None, :]
    ls = jnp.maximum(jnp.asarray(length_scale), 1e-6)
    amp = jnp.asarray(amplitude)
    return (amp**2) * jnp.exp(-0.5 * (r / ls) ** 2)


def polynomial_kernel(
    x1: jax.Array,
    x2: jax.Array,
    *,
    degree: int,
    length_scale: jax.Array | float,
    amplitude: jax.Array | float,
) -> jax.Array:
    """Monomial kernel k(x,x') = eta^2 sum_{k=0}^degree x^k x'^k."""
    del length_scale
    x1 = jnp.ravel(x1)
    x2 = jnp.ravel(x2)
    amp = jnp.asarray(amplitude)
    k = jnp.zeros((x1.shape[0], x2.shape[0]), dtype=x1.dtype)
    for power in range(degree + 1):
        k = k + (x1[:, None] ** power) * (x2[None, :] ** power)
    return (amp**2) * k


def poly2_kernel(
    x1: jax.Array,
    x2: jax.Array,
    *,
    length_scale: jax.Array | float,
    amplitude: jax.Array | float,
) -> jax.Array:
    """Quadratic monomial kernel k(x,x') = eta^2 (1 + x x' + x^2 x'^2)."""
    return polynomial_kernel(
        x1, x2, degree=2, length_scale=length_scale, amplitude=amplitude
    )


def poly4_kernel(
    x1: jax.Array,
    x2: jax.Array,
    *,
    length_scale: jax.Array | float,
    amplitude: jax.Array | float,
) -> jax.Array:
    """Quartic monomial kernel through z^4."""
    return polynomial_kernel(
        x1, x2, degree=4, length_scale=length_scale, amplitude=amplitude
    )


def stationary_kernel(
    kernel: GpKernelKind | str,
    x1: jax.Array,
    x2: jax.Array,
    *,
    length_scale: jax.Array | float,
    amplitude: jax.Array | float,
) -> jax.Array:
    kind = normalize_gp_kernel(kernel) if isinstance(kernel, str) else kernel
    degree = polynomial_kernel_degree(kind)
    if degree is not None:
        return polynomial_kernel(
            x1, x2, degree=degree, length_scale=length_scale, amplitude=amplitude
        )
    fn = gaussian_kernel if kind == "gaussian" else matern52_kernel
    return fn(x1, x2, length_scale=length_scale, amplitude=amplitude)


def gp_log_marginal_likelihood(
    x: jax.Array,
    y: jax.Array,
    sigma: jax.Array,
    *,
    length_scale: jax.Array | float,
    amplitude: jax.Array | float,
    kernel: GpKernelKind | str = DEFAULT_GP_KERNEL,
    jitter: float = 1e-6,
) -> jax.Array:
    """Gaussian-process regression log marginal likelihood."""
    x = jnp.ravel(x)
    y = jnp.ravel(y)
    sigma = jnp.ravel(sigma)
    n = x.shape[0]

    k = stationary_kernel(
        kernel, x, x, length_scale=length_scale, amplitude=amplitude
    )
    k = k.at[jnp.diag_indices(n)].add(sigma**2 + jitter)

    chol = jax.scipy.linalg.cholesky(k, lower=True)
    alpha = jax.scipy.linalg.cho_solve((chol, True), y)
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
    return (
        -0.5 * jnp.dot(y, alpha)
        - 0.5 * log_det
        - 0.5 * n * _LOG2PI
    )


def correction_gp_log_marginal_likelihood(
    z: jax.Array,
    y: jax.Array,
    sigma: jax.Array,
    L: jax.Array,
    *,
    omega: jax.Array | float,
    gp_ell: jax.Array | float,
    gp_eta: jax.Array | float,
    correction_gp_ell: jax.Array | float,
    correction_gp_eta: jax.Array | float,
    kernel: GpKernelKind | str = DEFAULT_GP_KERNEL,
    jitter: float = 1e-5,
) -> jax.Array:
    """Log marginal likelihood for phi = f0(z) + L^-omega f1(z)."""
    z = jnp.ravel(z)
    y = jnp.ravel(y)
    sigma = jnp.ravel(sigma)
    L = jnp.ravel(L)
    n = z.shape[0]

    d = L ** (-jnp.asarray(omega))
    k0 = stationary_kernel(
        kernel, z, z, length_scale=gp_ell, amplitude=gp_eta
    )
    k1 = stationary_kernel(
        kernel,
        z,
        z,
        length_scale=correction_gp_ell,
        amplitude=correction_gp_eta,
    )
    k = k0 + d[:, None] * k1 * d[None, :]
    k = k.at[jnp.diag_indices(n)].add(sigma**2 + jitter)

    chol = jax.scipy.linalg.cholesky(k, lower=True)
    alpha = jax.scipy.linalg.cho_solve((chol, True), y)
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
    return (
        -0.5 * jnp.dot(y, alpha)
        - 0.5 * log_det
        - 0.5 * n * _LOG2PI
    )


def discrepancy_gp_log_marginal_likelihood(
    z: jax.Array,
    y: jax.Array,
    sigma: jax.Array,
    a: jax.Array,
    *,
    gp_ell: jax.Array | float,
    gp_eta: jax.Array | float,
    disc_gp_ell: jax.Array | float,
    disc_gp_eta: jax.Array | float,
    kernel: GpKernelKind | str = DEFAULT_GP_KERNEL,
    universal_kernel: GpKernelKind | str | None = None,
    jitter: float = 1e-5,
    coupling: DiscCoupling | str = "additive",
) -> jax.Array:
    """Log marginal likelihood for a gated two-GP observation of ``y``.

    Independent zero-mean GPs ``f`` and ``g`` on the collapsed coordinate
    ``z``.  ``coupling`` chooses the observation model (``a`` is already
    the collapsed g-coefficient or the unit gate ``π``)::

        additive:  y = f + a g + ε
        mixture:   y = (1-a) f + a g + ε
        weighted:  y = (1-a) f + ε

    Marginal covariance::

        K_ij = w_f_i w_f_j η² k_ℓ(z_i, z_j)
             + w_g_i w_g_j σ_g² k_ℓg(z_i, z_j)
             + δ_ij (σ_i² + ε)
    """
    z = jnp.ravel(z)
    y = jnp.ravel(y)
    sigma = jnp.ravel(sigma)
    a = jnp.ravel(a)
    n = z.shape[0]
    w_f, w_g = _discrepancy_weights(a, coupling)

    uk = kernel if universal_kernel is None else universal_kernel
    k0 = stationary_kernel(uk, z, z, length_scale=gp_ell, amplitude=gp_eta)
    k = w_f[:, None] * k0 * w_f[None, :]
    if coupling != "weighted":
        kg = stationary_kernel(
            kernel, z, z, length_scale=disc_gp_ell, amplitude=disc_gp_eta
        )
        k = k + w_g[:, None] * kg * w_g[None, :]
    k = k.at[jnp.diag_indices(n)].add(sigma**2 + jitter)

    chol = jax.scipy.linalg.cholesky(k, lower=True)
    alpha = jax.scipy.linalg.cho_solve((chol, True), y)
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
    return (
        -0.5 * jnp.dot(y, alpha)
        - 0.5 * log_det
        - 0.5 * n * _LOG2PI
    )


def discrepancy_f_posterior_predictive(
    z_train: jax.Array,
    y_train: jax.Array,
    sigma_train: jax.Array,
    a_train: jax.Array,
    z_new: jax.Array,
    *,
    gp_ell: jax.Array | float,
    gp_eta: jax.Array | float,
    disc_gp_ell: jax.Array | float,
    disc_gp_eta: jax.Array | float,
    kernel: GpKernelKind | str = DEFAULT_GP_KERNEL,
    jitter: float = 1e-5,
    coupling: DiscCoupling | str = "additive",
) -> tuple[jax.Array, jax.Array]:
    """Posterior mean and std of the universal GP ``f``.

    Uses ``E[f(z_*)|y] = k_{f,y}(z_*, z) K^{-1} y`` with the same marginal
    ``K`` as ``discrepancy_gp_log_marginal_likelihood``.  For mixture /
    weighted, ``cov(f_*, y_j) = k_f(z_*, z_j) w_{f,j}``.
    """
    z_train = jnp.ravel(z_train)
    y_train = jnp.ravel(y_train)
    sigma_train = jnp.ravel(sigma_train)
    a_train = jnp.ravel(a_train)
    z_new = jnp.ravel(z_new)
    n = z_train.shape[0]
    w_f, w_g = _discrepancy_weights(a_train, coupling)

    k0_tt = stationary_kernel(
        kernel, z_train, z_train, length_scale=gp_ell, amplitude=gp_eta
    )
    k_yy = w_f[:, None] * k0_tt * w_f[None, :]
    if coupling != "weighted":
        kg_tt = stationary_kernel(
            kernel, z_train, z_train, length_scale=disc_gp_ell, amplitude=disc_gp_eta
        )
        k_yy = k_yy + w_g[:, None] * kg_tt * w_g[None, :]
    k_yy = k_yy.at[jnp.diag_indices(n)].add(sigma_train**2 + jitter)
    chol = jax.scipy.linalg.cholesky(k_yy, lower=True)
    alpha = jax.scipy.linalg.cho_solve((chol, True), y_train)
    k0_st = stationary_kernel(
        kernel, z_new, z_train, length_scale=gp_ell, amplitude=gp_eta
    )
    k_fy = k0_st * w_f[None, :]
    mean = k_fy @ alpha
    v = jax.scipy.linalg.solve_triangular(chol, k_fy.T, lower=True)
    k0_ss = stationary_kernel(
        kernel, z_new, z_new, length_scale=gp_ell, amplitude=gp_eta
    )
    var = jnp.diag(k0_ss) - jnp.sum(v * v, axis=0)
    return mean, jnp.sqrt(jnp.maximum(var, 0.0))


def discrepancy_f_posterior_mean(
    z_train: jax.Array,
    y_train: jax.Array,
    sigma_train: jax.Array,
    a_train: jax.Array,
    z_new: jax.Array,
    *,
    gp_ell: jax.Array | float,
    gp_eta: jax.Array | float,
    disc_gp_ell: jax.Array | float,
    disc_gp_eta: jax.Array | float,
    kernel: GpKernelKind | str = DEFAULT_GP_KERNEL,
    jitter: float = 1e-5,
    coupling: DiscCoupling | str = "additive",
) -> jax.Array:
    """Posterior mean of the universal GP ``f`` under y = f(z) + a g(z) + noise."""
    mean, _ = discrepancy_f_posterior_predictive(
        z_train,
        y_train,
        sigma_train,
        a_train,
        z_new,
        gp_ell=gp_ell,
        gp_eta=gp_eta,
        disc_gp_ell=disc_gp_ell,
        disc_gp_eta=disc_gp_eta,
        kernel=kernel,
        jitter=jitter,
        coupling=coupling,
    )
    return mean


def _gp_posterior_kernel_system(
    x_train: jax.Array,
    sigma_train: jax.Array,
    *,
    length_scale: jax.Array | float,
    amplitude: jax.Array | float,
    kernel: GpKernelKind | str = DEFAULT_GP_KERNEL,
    jitter: float,
) -> jax.Array:
    x_train = jnp.ravel(x_train)
    sigma_train = jnp.ravel(sigma_train)
    k_tt = stationary_kernel(
        kernel,
        x_train,
        x_train,
        length_scale=length_scale,
        amplitude=amplitude,
    )
    n = x_train.shape[0]
    k_tt = k_tt.at[jnp.diag_indices(n)].add(sigma_train**2 + jitter)
    return jax.scipy.linalg.cholesky(k_tt, lower=True)


def gp_posterior_predictive(
    x_train: jax.Array,
    y_train: jax.Array,
    sigma_train: jax.Array,
    x_new: jax.Array,
    *,
    length_scale: jax.Array | float,
    amplitude: jax.Array | float,
    kernel: GpKernelKind | str = DEFAULT_GP_KERNEL,
    jitter: float = 1e-6,
) -> tuple[jax.Array, jax.Array]:
    """GP regression posterior mean and std at ``x_new``."""
    x_train = jnp.ravel(x_train)
    y_train = jnp.ravel(y_train)
    sigma_train = jnp.ravel(sigma_train)
    x_new = jnp.ravel(x_new)

    chol = _gp_posterior_kernel_system(
        x_train,
        sigma_train,
        length_scale=length_scale,
        amplitude=amplitude,
        kernel=kernel,
        jitter=jitter,
    )
    k_st = stationary_kernel(
        kernel,
        x_new,
        x_train,
        length_scale=length_scale,
        amplitude=amplitude,
    )
    alpha = jax.scipy.linalg.cho_solve((chol, True), y_train)
    mean = k_st @ alpha

    # Triangular solve L v = k_st^T (cho_solve would solve K v = k_st^T instead).
    v = jax.scipy.linalg.solve_triangular(chol, k_st.T, lower=True)
    k_ss = stationary_kernel(
        kernel,
        x_new,
        x_new,
        length_scale=length_scale,
        amplitude=amplitude,
    )
    var = jnp.diag(k_ss) - jnp.sum(v * v, axis=0)
    return mean, jnp.sqrt(jnp.maximum(var, 0.0))


def _gaussian_log_density_jax(
    y: jax.Array,
    mean: jax.Array,
    variance: jax.Array,
) -> jax.Array:
    y = jnp.ravel(y)
    mean = jnp.ravel(mean)
    variance = jnp.maximum(jnp.ravel(variance), 1e-24)
    return (
        -0.5 * jnp.log(2.0 * jnp.pi * variance)
        - 0.5 * (y - mean) ** 2 / variance
    )


def gp_log_predictive_density(
    x_train: jax.Array,
    y_train: jax.Array,
    sigma_train: jax.Array,
    x_new: jax.Array,
    y_new: jax.Array,
    *,
    sigma_new: jax.Array | float,
    length_scale: jax.Array | float,
    amplitude: jax.Array | float,
    kernel: GpKernelKind | str = DEFAULT_GP_KERNEL,
    jitter: float = 1e-6,
) -> jax.Array:
    """Log predictive density of noisy observations ``y_new`` at ``x_new``."""
    x_new = jnp.ravel(x_new)
    y_new = jnp.ravel(y_new)
    sigma_new_arr = jnp.broadcast_to(jnp.asarray(sigma_new), (x_new.size,))

    mean, std_latent = gp_posterior_predictive(
        x_train,
        y_train,
        sigma_train,
        x_new,
        length_scale=length_scale,
        amplitude=amplitude,
        kernel=kernel,
        jitter=jitter,
    )
    variance = std_latent**2 + sigma_new_arr**2 + jitter
    return _gaussian_log_density_jax(y_new, mean, variance)


def correction_gp_posterior_predictive(
    z_train: jax.Array,
    L_train: jax.Array,
    y_train: jax.Array,
    sigma_train: jax.Array,
    z_test: jax.Array,
    L_test: jax.Array,
    *,
    omega: jax.Array | float,
    gp_ell: jax.Array | float,
    gp_eta: jax.Array | float,
    correction_gp_ell: jax.Array | float,
    correction_gp_eta: jax.Array | float,
    kernel: GpKernelKind | str = DEFAULT_GP_KERNEL,
    jitter: float = 1e-5,
) -> tuple[jax.Array, jax.Array]:
    """Posterior predictive mean and std for phi = f0(z) + L^-omega f1(z)."""
    z_train = jnp.ravel(z_train)
    L_train = jnp.ravel(L_train)
    y_train = jnp.ravel(y_train)
    sigma_train = jnp.ravel(sigma_train)
    z_test = jnp.ravel(z_test)
    L_test = jnp.ravel(L_test)

    d_train = L_train ** (-jnp.asarray(omega))
    d_test = L_test ** (-jnp.asarray(omega))

    k0_tt = stationary_kernel(
        kernel, z_train, z_train, length_scale=gp_ell, amplitude=gp_eta
    )
    k1_tt = stationary_kernel(
        kernel,
        z_train,
        z_train,
        length_scale=correction_gp_ell,
        amplitude=correction_gp_eta,
    )
    k_yy = k0_tt + d_train[:, None] * k1_tt * d_train[None, :]
    n = z_train.shape[0]
    k_yy = k_yy.at[jnp.diag_indices(n)].add(sigma_train**2 + jitter)

    k0_st = stationary_kernel(
        kernel, z_test, z_train, length_scale=gp_ell, amplitude=gp_eta
    )
    k1_st = stationary_kernel(
        kernel,
        z_test,
        z_train,
        length_scale=correction_gp_ell,
        amplitude=correction_gp_eta,
    )
    k_sy = k0_st + d_test[:, None] * k1_st * d_train[None, :]

    k0_ss = stationary_kernel(
        kernel, z_test, z_test, length_scale=gp_ell, amplitude=gp_eta
    )
    k1_ss = stationary_kernel(
        kernel,
        z_test,
        z_test,
        length_scale=correction_gp_ell,
        amplitude=correction_gp_eta,
    )
    k_ss = k0_ss + d_test[:, None] * k1_ss * d_test[None, :]

    chol = jax.scipy.linalg.cholesky(k_yy, lower=True)
    alpha = jax.scipy.linalg.cho_solve((chol, True), y_train)
    mean = k_sy @ alpha

    v = jax.scipy.linalg.solve_triangular(chol, k_sy.T, lower=True)
    var = jnp.diag(k_ss) - jnp.sum(v * v, axis=0)
    return mean, jnp.sqrt(jnp.maximum(var, 0.0))


def correction_gp_log_predictive_density(
    z_train: jax.Array,
    L_train: jax.Array,
    y_train: jax.Array,
    sigma_train: jax.Array,
    z_test: jax.Array,
    L_test: jax.Array,
    y_test: jax.Array,
    *,
    sigma_test: jax.Array | float,
    omega: jax.Array | float,
    gp_ell: jax.Array | float,
    gp_eta: jax.Array | float,
    correction_gp_ell: jax.Array | float,
    correction_gp_eta: jax.Array | float,
    kernel: GpKernelKind | str = DEFAULT_GP_KERNEL,
    jitter: float = 1e-5,
) -> jax.Array:
    """Log predictive density for correction-GP observations at test points."""
    z_test = jnp.ravel(z_test)
    L_test = jnp.ravel(L_test)
    y_test = jnp.ravel(y_test)
    sigma_test_arr = jnp.broadcast_to(jnp.asarray(sigma_test), (z_test.size,))

    mean, std_latent = correction_gp_posterior_predictive(
        z_train,
        L_train,
        y_train,
        sigma_train,
        z_test,
        L_test,
        omega=omega,
        gp_ell=gp_ell,
        gp_eta=gp_eta,
        correction_gp_ell=correction_gp_ell,
        correction_gp_eta=correction_gp_eta,
        kernel=kernel,
        jitter=jitter,
    )
    variance = std_latent**2 + sigma_test_arr**2 + jitter
    return _gaussian_log_density_jax(y_test, mean, variance)
