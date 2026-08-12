"""JAX GP marginal likelihood kernels (mirrors :mod:`gp_utils`)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .gp_kernels import DEFAULT_GP_KERNEL, GpKernelKind, normalize_gp_kernel

_LOG2PI = jnp.log(2.0 * jnp.pi)


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


def stationary_kernel(
    kernel: GpKernelKind | str,
    x1: jax.Array,
    x2: jax.Array,
    *,
    length_scale: jax.Array | float,
    amplitude: jax.Array | float,
) -> jax.Array:
    kind = normalize_gp_kernel(kernel) if isinstance(kernel, str) else kernel
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
    jitter: float = 1e-5,
) -> jax.Array:
    """Log marginal likelihood for y = f(z) + a * g(z) + noise.

    Independent zero-mean GPs ``f`` and ``g`` on the collapsed coordinate
    ``z``.  Here ``a`` is the *collapsed-space* amplitude multiplying ``g``
    (for magnetization slide form ``m = L^{-β/ν} f + a_raw g``, pass
    ``a = a_raw * L^{β/ν}`` so that ``y = Φ = m L^{β/ν}``).

    Marginal covariance::

        K_ij = η² k_ℓ(z_i, z_j) + σ_g² a_i a_j k_ℓg(z_i, z_j) + δ_ij (σ_i² + ε)
    """
    z = jnp.ravel(z)
    y = jnp.ravel(y)
    sigma = jnp.ravel(sigma)
    a = jnp.ravel(a)
    n = z.shape[0]

    k0 = stationary_kernel(
        kernel, z, z, length_scale=gp_ell, amplitude=gp_eta
    )
    kg = stationary_kernel(
        kernel, z, z, length_scale=disc_gp_ell, amplitude=disc_gp_eta
    )
    k = k0 + a[:, None] * kg * a[None, :]
    k = k.at[jnp.diag_indices(n)].add(sigma**2 + jitter)

    chol = jax.scipy.linalg.cholesky(k, lower=True)
    alpha = jax.scipy.linalg.cho_solve((chol, True), y)
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
    return (
        -0.5 * jnp.dot(y, alpha)
        - 0.5 * log_det
        - 0.5 * n * _LOG2PI
    )


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
) -> jax.Array:
    """Posterior mean of the universal GP ``f`` under y = f(z) + a g(z) + noise.

    Uses ``E[f(z_*)|y] = k_f(z_*, z) K^{-1} y`` with the additive-GP marginal
    covariance ``K`` (same as ``discrepancy_gp_log_marginal_likelihood``).
    """
    z_train = jnp.ravel(z_train)
    y_train = jnp.ravel(y_train)
    sigma_train = jnp.ravel(sigma_train)
    a_train = jnp.ravel(a_train)
    z_new = jnp.ravel(z_new)
    n = z_train.shape[0]

    k0_tt = stationary_kernel(
        kernel, z_train, z_train, length_scale=gp_ell, amplitude=gp_eta
    )
    kg_tt = stationary_kernel(
        kernel, z_train, z_train, length_scale=disc_gp_ell, amplitude=disc_gp_eta
    )
    k_yy = k0_tt + a_train[:, None] * kg_tt * a_train[None, :]
    k_yy = k_yy.at[jnp.diag_indices(n)].add(sigma_train**2 + jitter)
    chol = jax.scipy.linalg.cholesky(k_yy, lower=True)
    alpha = jax.scipy.linalg.cho_solve((chol, True), y_train)
    k0_st = stationary_kernel(
        kernel, z_new, z_train, length_scale=gp_ell, amplitude=gp_eta
    )
    return k0_st @ alpha


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
