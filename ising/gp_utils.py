"""NumPy GP helpers for Harada-style marginal likelihood."""

from __future__ import annotations

from typing import Literal

import numpy as np

from .gp_kernels import (
    DEFAULT_GP_KERNEL,
    GpKernelKind,
    normalize_gp_kernel,
    polynomial_kernel_degree,
)

DiscCoupling = Literal["additive", "mixture", "weighted"]


def _discrepancy_weights(
    a: np.ndarray,
    coupling: DiscCoupling | str,
) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=np.float64).ravel()
    ones = np.ones_like(a)
    if coupling == "weighted":
        return ones - a, np.zeros_like(a)
    if coupling == "mixture":
        return ones - a, a
    return ones, a


def matern52_kernel(
    x1: np.ndarray,
    x2: np.ndarray,
    *,
    length_scale: float,
    amplitude: float,
) -> np.ndarray:
    x1 = np.asarray(x1, dtype=np.float64).reshape(-1, 1)
    x2 = np.asarray(x2, dtype=np.float64).reshape(-1, 1)
    r = np.abs(x1 - x2.T)
    ls = max(float(length_scale), 1e-6)
    amp = float(amplitude)
    sqrt5 = np.sqrt(5.0)
    scaled = sqrt5 * r / ls
    return (amp**2) * (1.0 + scaled + (scaled**2) / 3.0) * np.exp(-scaled)


def gaussian_kernel(
    x1: np.ndarray,
    x2: np.ndarray,
    *,
    length_scale: float,
    amplitude: float,
) -> np.ndarray:
    """Squared-exponential kernel used in Harada BSA (PRE 84, 056704)."""
    x1 = np.asarray(x1, dtype=np.float64).reshape(-1, 1)
    x2 = np.asarray(x2, dtype=np.float64).reshape(-1, 1)
    r = x1 - x2.T
    ls = max(float(length_scale), 1e-6)
    amp = float(amplitude)
    return (amp**2) * np.exp(-0.5 * (r / ls) ** 2)


def polynomial_kernel(
    x1: np.ndarray,
    x2: np.ndarray,
    *,
    degree: int,
    length_scale: float,
    amplitude: float,
) -> np.ndarray:
    """Monomial kernel k(x,x') = eta^2 sum_{k=0}^degree x^k x'^k."""
    del length_scale
    x1 = np.asarray(x1, dtype=np.float64).ravel()
    x2 = np.asarray(x2, dtype=np.float64).ravel()
    amp = float(amplitude)
    k = np.zeros((x1.size, x2.size), dtype=np.float64)
    for power in range(degree + 1):
        k += (x1[:, None] ** power) * (x2[None, :] ** power)
    return (amp**2) * k


def poly2_kernel(
    x1: np.ndarray,
    x2: np.ndarray,
    *,
    length_scale: float,
    amplitude: float,
) -> np.ndarray:
    """Quadratic monomial kernel k(x,x') = eta^2 (1 + x x' + x^2 x'^2)."""
    return polynomial_kernel(
        x1, x2, degree=2, length_scale=length_scale, amplitude=amplitude
    )


def poly4_kernel(
    x1: np.ndarray,
    x2: np.ndarray,
    *,
    length_scale: float,
    amplitude: float,
) -> np.ndarray:
    """Quartic monomial kernel through z^4."""
    return polynomial_kernel(
        x1, x2, degree=4, length_scale=length_scale, amplitude=amplitude
    )


def stationary_kernel(
    kernel: GpKernelKind | str,
    x1: np.ndarray,
    x2: np.ndarray,
    *,
    length_scale: float,
    amplitude: float,
) -> np.ndarray:
    kind = normalize_gp_kernel(kernel) if isinstance(kernel, str) else kernel
    degree = polynomial_kernel_degree(kind)
    if degree is not None:
        return polynomial_kernel(
            x1, x2, degree=degree, length_scale=length_scale, amplitude=amplitude
        )
    fn = gaussian_kernel if kind == "gaussian" else matern52_kernel
    return fn(x1, x2, length_scale=length_scale, amplitude=amplitude)


def gp_log_marginal_likelihood(
    x: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    *,
    length_scale: float,
    amplitude: float,
    kernel: GpKernelKind | str = DEFAULT_GP_KERNEL,
    jitter: float = 1e-6,
) -> float:
    """Gaussian-process regression log marginal likelihood."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    n = x.size
    if n == 0:
        return -np.inf

    k = stationary_kernel(
        kernel, x, x, length_scale=length_scale, amplitude=amplitude
    )
    k[np.diag_indices(n)] += sigma**2 + jitter

    chol = np.linalg.cholesky(k)
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y))
    log_det = 2.0 * np.sum(np.log(np.diag(chol)))
    return float(
        -0.5 * (y @ alpha)
        -0.5 * log_det
        -0.5 * n * np.log(2.0 * np.pi)
    )


def correction_gp_log_marginal_likelihood(
    z: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    L: np.ndarray,
    *,
    omega: float,
    gp_ell: float,
    gp_eta: float,
    correction_gp_ell: float,
    correction_gp_eta: float,
    kernel: GpKernelKind | str = DEFAULT_GP_KERNEL,
    jitter: float = 1e-5,
) -> float:
    """Log marginal likelihood for phi = f0(z) + L^-omega f1(z)."""
    z = np.asarray(z, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    L = np.asarray(L, dtype=np.float64).ravel()
    n = z.size
    if n == 0:
        return -np.inf

    d = L ** (-float(omega))
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
    k[np.diag_indices(n)] += sigma**2 + jitter

    chol = np.linalg.cholesky(k)
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y))
    log_det = 2.0 * np.sum(np.log(np.diag(chol)))
    return float(
        -0.5 * (y @ alpha)
        -0.5 * log_det
        -0.5 * n * np.log(2.0 * np.pi)
    )


def discrepancy_gp_log_marginal_likelihood(
    z: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    a: np.ndarray,
    *,
    gp_ell: float,
    gp_eta: float,
    disc_gp_ell: float,
    disc_gp_eta: float,
    kernel: GpKernelKind | str = DEFAULT_GP_KERNEL,
    universal_kernel: GpKernelKind | str | None = None,
    jitter: float = 1e-5,
    coupling: DiscCoupling | str = "additive",
) -> float:
    """Log marginal likelihood for y = w_f f + w_g g + noise.

    ``a`` is the collapsed-space amplitude (additive) or unit gate ``π``
    (mixture / weighted).  See ``gp_jax`` docstring.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    a = np.asarray(a, dtype=np.float64).ravel()
    n = z.size
    if n == 0:
        return -np.inf

    w_f, w_g = _discrepancy_weights(a, coupling)
    uk = kernel if universal_kernel is None else universal_kernel
    k0 = stationary_kernel(uk, z, z, length_scale=gp_ell, amplitude=gp_eta)
    k = w_f[:, None] * k0 * w_f[None, :]
    if coupling != "weighted":
        kg = stationary_kernel(
            kernel, z, z, length_scale=disc_gp_ell, amplitude=disc_gp_eta
        )
        k = k + w_g[:, None] * kg * w_g[None, :]
    k[np.diag_indices(n)] += sigma**2 + jitter

    chol = np.linalg.cholesky(k)
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y))
    log_det = 2.0 * np.sum(np.log(np.diag(chol)))
    return float(
        -0.5 * (y @ alpha)
        -0.5 * log_det
        -0.5 * n * np.log(2.0 * np.pi)
    )


def _gp_posterior_kernel_system(
    x_train: np.ndarray,
    sigma_train: np.ndarray,
    *,
    length_scale: float,
    amplitude: float,
    kernel: GpKernelKind | str = DEFAULT_GP_KERNEL,
    jitter: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_train = np.asarray(x_train, dtype=np.float64).ravel()
    sigma_train = np.asarray(sigma_train, dtype=np.float64).ravel()
    k_tt = stationary_kernel(
        kernel,
        x_train,
        x_train,
        length_scale=length_scale,
        amplitude=amplitude,
    )
    k_tt[np.diag_indices(x_train.size)] += sigma_train**2 + jitter
    chol = np.linalg.cholesky(k_tt)
    return k_tt, chol


def gp_posterior_mean(
    x_train: np.ndarray,
    y_train: np.ndarray,
    sigma_train: np.ndarray,
    x_new: np.ndarray,
    *,
    length_scale: float,
    amplitude: float,
    kernel: GpKernelKind | str = DEFAULT_GP_KERNEL,
    jitter: float = 1e-6,
) -> np.ndarray:
    mean, _ = gp_posterior_predictive(
        x_train,
        y_train,
        sigma_train,
        x_new,
        length_scale=length_scale,
        amplitude=amplitude,
        kernel=kernel,
        jitter=jitter,
    )
    return mean


def gp_posterior_predictive(
    x_train: np.ndarray,
    y_train: np.ndarray,
    sigma_train: np.ndarray,
    x_new: np.ndarray,
    *,
    length_scale: float,
    amplitude: float,
    kernel: GpKernelKind | str = DEFAULT_GP_KERNEL,
    jitter: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """GP regression posterior mean and standard deviation at x_new."""
    x_train = np.asarray(x_train, dtype=np.float64).ravel()
    y_train = np.asarray(y_train, dtype=np.float64).ravel()
    sigma_train = np.asarray(sigma_train, dtype=np.float64).ravel()
    x_new = np.asarray(x_new, dtype=np.float64).ravel()
    if x_train.size == 0:
        # Zero-mean GP prior: mean 0, latent std = amplitude.
        mean = np.zeros(x_new.size, dtype=np.float64)
        std = np.full(x_new.size, float(amplitude), dtype=np.float64)
        return mean, std

    k_tt, chol = _gp_posterior_kernel_system(
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
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y_train))
    mean = k_st @ alpha

    v = np.linalg.solve(chol, k_st.T)
    k_ss = stationary_kernel(
        kernel,
        x_new,
        x_new,
        length_scale=length_scale,
        amplitude=amplitude,
    )
    var = np.diag(k_ss) - np.sum(v * v, axis=0)
    return mean, np.sqrt(np.maximum(var, 0.0))


def _gaussian_log_density(
    y: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
) -> np.ndarray:
    """Log N(y | mean, variance) for scalar or vector y."""
    y = np.asarray(y, dtype=np.float64).ravel()
    mean = np.asarray(mean, dtype=np.float64).ravel()
    variance = np.asarray(variance, dtype=np.float64).ravel()
    variance = np.maximum(variance, 1e-24)
    return (
        -0.5 * np.log(2.0 * np.pi * variance)
        - 0.5 * (y - mean) ** 2 / variance
    )


def gp_log_predictive_density(
    x_train: np.ndarray,
    y_train: np.ndarray,
    sigma_train: np.ndarray,
    x_new: np.ndarray,
    y_new: np.ndarray,
    *,
    sigma_new: np.ndarray | float,
    length_scale: float,
    amplitude: float,
    kernel: GpKernelKind | str = DEFAULT_GP_KERNEL,
    jitter: float = 1e-6,
) -> np.ndarray:
    """Log predictive density of noisy observations ``y_new`` at ``x_new``."""
    x_new = np.asarray(x_new, dtype=np.float64).ravel()
    y_new = np.asarray(y_new, dtype=np.float64).ravel()
    sigma_new_arr = np.asarray(sigma_new, dtype=np.float64)
    if sigma_new_arr.ndim == 0:
        sigma_new_arr = np.full(x_new.size, float(sigma_new_arr))
    else:
        sigma_new_arr = sigma_new_arr.ravel()
    if not (x_new.size == y_new.size == sigma_new_arr.size):
        raise ValueError("x_new, y_new, and sigma_new must have the same length")

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
    return _gaussian_log_density(y_new, mean, variance)


def correction_gp_log_predictive_density(
    z_train: np.ndarray,
    L_train: np.ndarray,
    y_train: np.ndarray,
    sigma_train: np.ndarray,
    z_test: np.ndarray,
    L_test: np.ndarray,
    y_test: np.ndarray,
    *,
    sigma_test: np.ndarray | float,
    omega: float,
    gp_ell: float,
    gp_eta: float,
    correction_gp_ell: float,
    correction_gp_eta: float,
    kernel: GpKernelKind | str = DEFAULT_GP_KERNEL,
    jitter: float = 1e-5,
) -> np.ndarray:
    """Log predictive density for correction-GP observations at test points."""
    z_test = np.asarray(z_test, dtype=np.float64).ravel()
    L_test = np.asarray(L_test, dtype=np.float64).ravel()
    y_test = np.asarray(y_test, dtype=np.float64).ravel()
    sigma_test_arr = np.asarray(sigma_test, dtype=np.float64)
    if sigma_test_arr.ndim == 0:
        sigma_test_arr = np.full(z_test.size, float(sigma_test_arr))
    else:
        sigma_test_arr = sigma_test_arr.ravel()
    if not (z_test.size == L_test.size == y_test.size == sigma_test_arr.size):
        raise ValueError("z_test, L_test, y_test, and sigma_test must match in length")

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
    return _gaussian_log_density(y_test, mean, variance)


def correction_gp_posterior_predictive(
    z_train: np.ndarray,
    L_train: np.ndarray,
    y_train: np.ndarray,
    sigma_train: np.ndarray,
    z_test: np.ndarray,
    L_test: np.ndarray,
    *,
    omega: float,
    gp_ell: float,
    gp_eta: float,
    correction_gp_ell: float,
    correction_gp_eta: float,
    kernel: GpKernelKind | str = DEFAULT_GP_KERNEL,
    jitter: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior predictive mean and std for phi = f0(z) + L^-omega f1(z)."""
    z_train = np.asarray(z_train, dtype=np.float64).ravel()
    L_train = np.asarray(L_train, dtype=np.float64).ravel()
    y_train = np.asarray(y_train, dtype=np.float64).ravel()
    sigma_train = np.asarray(sigma_train, dtype=np.float64).ravel()
    z_test = np.asarray(z_test, dtype=np.float64).ravel()
    L_test = np.asarray(L_test, dtype=np.float64).ravel()

    d_train = L_train ** (-omega)
    d_test = L_test ** (-omega)

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
    k_yy[np.diag_indices(k_yy.shape[0])] += sigma_train**2 + jitter

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

    chol = np.linalg.cholesky(k_yy)
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y_train))
    mean = k_sy @ alpha

    v = np.linalg.solve(chol, k_sy.T)
    var = np.diag(k_ss) - np.sum(v * v, axis=0)
    return mean, np.sqrt(np.maximum(var, 0.0))


def correction_gp_latent_conditional(
    z_train: np.ndarray,
    L_train: np.ndarray,
    y_train: np.ndarray,
    sigma_train: np.ndarray,
    z_test: np.ndarray,
    L_test: np.ndarray,
    *,
    omega: float,
    gp_ell: float,
    gp_eta: float,
    correction_gp_ell: float,
    correction_gp_eta: float,
    kernel: GpKernelKind | str = DEFAULT_GP_KERNEL,
    jitter: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Conditional mean and std given noisy observations at training points."""
    return correction_gp_posterior_predictive(
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


def gp_latent_conditional(
    z_train: np.ndarray,
    log_phi_train: np.ndarray,
    z_new: np.ndarray,
    *,
    sigma_log_train: np.ndarray | None = None,
    length_scale: float,
    amplitude: float,
    kernel: GpKernelKind | str = DEFAULT_GP_KERNEL,
    jitter: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Conditional mean and std of log Phi given log observations at z_train."""
    z_train = np.asarray(z_train, dtype=np.float64).ravel()
    log_phi_train = np.asarray(log_phi_train, dtype=np.float64).ravel()
    if sigma_log_train is None:
        sigma_log_train = np.zeros_like(log_phi_train)
    else:
        sigma_log_train = np.asarray(sigma_log_train, dtype=np.float64).ravel()
    return gp_posterior_predictive(
        z_train,
        log_phi_train,
        sigma_log_train,
        z_new,
        length_scale=length_scale,
        amplitude=amplitude,
        kernel=kernel,
        jitter=jitter,
    )


def product_kernel_2d(
    kernel: GpKernelKind | str,
    L1: np.ndarray,
    T1: np.ndarray,
    L2: np.ndarray,
    T2: np.ndarray,
    *,
    length_scale_L: float,
    length_scale_T: float,
    amplitude: float,
) -> np.ndarray:
    k_l = stationary_kernel(
        kernel,
        np.asarray(L1, dtype=np.float64).reshape(-1, 1),
        np.asarray(L2, dtype=np.float64).reshape(-1, 1),
        length_scale=length_scale_L,
        amplitude=1.0,
    )
    k_t = stationary_kernel(
        kernel,
        np.asarray(T1, dtype=np.float64).reshape(-1, 1),
        np.asarray(T2, dtype=np.float64).reshape(-1, 1),
        length_scale=length_scale_T,
        amplitude=1.0,
    )
    return (float(amplitude) ** 2) * k_l * k_t


def matern52_product_kernel_2d(
    L1: np.ndarray,
    T1: np.ndarray,
    L2: np.ndarray,
    T2: np.ndarray,
    *,
    length_scale_L: float,
    length_scale_T: float,
    amplitude: float,
) -> np.ndarray:
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


def gp_posterior_mean_2d(
    L_train: np.ndarray,
    T_train: np.ndarray,
    y_train: np.ndarray,
    L_new: np.ndarray,
    T_new: np.ndarray,
    *,
    length_scale_L: float,
    length_scale_T: float,
    amplitude: float,
    kernel: GpKernelKind | str = DEFAULT_GP_KERNEL,
    jitter: float = 1e-6,
) -> np.ndarray:
    L_train = np.asarray(L_train, dtype=np.float64).ravel()
    T_train = np.asarray(T_train, dtype=np.float64).ravel()
    y_train = np.asarray(y_train, dtype=np.float64).ravel()
    L_new = np.asarray(L_new, dtype=np.float64).ravel()
    T_new = np.asarray(T_new, dtype=np.float64).ravel()

    k_tt = product_kernel_2d(
        kernel,
        L_train,
        T_train,
        L_train,
        T_train,
        length_scale_L=length_scale_L,
        length_scale_T=length_scale_T,
        amplitude=amplitude,
    )
    k_tt[np.diag_indices(L_train.size)] += jitter
    chol = np.linalg.cholesky(k_tt)
    k_st = product_kernel_2d(
        kernel,
        L_new,
        T_new,
        L_train,
        T_train,
        length_scale_L=length_scale_L,
        length_scale_T=length_scale_T,
        amplitude=amplitude,
    )
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y_train))
    return k_st @ alpha
