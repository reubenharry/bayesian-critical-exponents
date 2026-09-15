"""Tests for FSS GP kernel dispatch."""

from __future__ import annotations

import numpy as np
import pytest

from ising.gp_kernels import normalize_gp_kernel
from ising.gp_utils import (
    discrepancy_gp_log_marginal_likelihood,
    gaussian_kernel,
    gp_log_marginal_likelihood,
    matern52_kernel,
    poly2_kernel,
    stationary_kernel,
)


def test_normalize_gp_kernel_aliases() -> None:
    assert normalize_gp_kernel("harada") == "gaussian"
    assert normalize_gp_kernel("rbf") == "gaussian"
    assert normalize_gp_kernel("matern52") == "matern52"
    assert normalize_gp_kernel("poly") == "poly2"
    assert normalize_gp_kernel("polynomial") == "poly2"
    assert normalize_gp_kernel("poly4") == "poly4"
    assert normalize_gp_kernel("quartic") == "poly4"
    with pytest.raises(ValueError):
        normalize_gp_kernel("unknown")


def test_gaussian_kernel_matches_harada_formula() -> None:
    x = np.array([0.0, 1.0, 2.0])
    ell, eta = 1.5, 0.8
    k = gaussian_kernel(x, x, length_scale=ell, amplitude=eta)
    expected = (eta**2) * np.exp(-0.5 * ((x[:, None] - x[None, :]) / ell) ** 2)
    np.testing.assert_allclose(k, expected)


def test_kernels_differ_at_same_hyperparams() -> None:
    x = np.linspace(-2.0, 2.0, 5)
    ell, eta = 1.0, 1.0
    k_mat = matern52_kernel(x, x, length_scale=ell, amplitude=eta)
    k_gauss = gaussian_kernel(x, x, length_scale=ell, amplitude=eta)
    assert not np.allclose(k_mat, k_gauss)


def test_gp_log_marginal_likelihood_kernel_switch() -> None:
    rng = np.random.default_rng(0)
    x = np.linspace(-1.0, 1.0, 8)
    y = np.sin(x) + 0.05 * rng.normal(size=x.size)
    sigma = np.full(x.size, 0.05)
    ll_mat = gp_log_marginal_likelihood(
        x, y, sigma, length_scale=1.0, amplitude=1.0, kernel="matern52"
    )
    ll_gauss = gp_log_marginal_likelihood(
        x, y, sigma, length_scale=1.0, amplitude=1.0, kernel="gaussian"
    )
    assert ll_mat != ll_gauss
    assert np.isfinite(ll_mat)
    assert np.isfinite(ll_gauss)


def test_stationary_kernel_dispatch() -> None:
    x = np.array([0.0, 1.0])
    assert np.allclose(
        stationary_kernel("harada", x, x, length_scale=1.0, amplitude=1.0),
        gaussian_kernel(x, x, length_scale=1.0, amplitude=1.0),
    )


def test_poly2_kernel_matches_monomial_inner_product() -> None:
    x = np.array([-1.0, 0.5, 2.0])
    eta = 0.7
    k = poly2_kernel(x, x, length_scale=1.0, amplitude=eta)
    expected = (eta**2) * (
        1.0 + x[:, None] * x[None, :] + (x[:, None] ** 2) * (x[None, :] ** 2)
    )
    np.testing.assert_allclose(k, expected)


def test_poly4_kernel_matches_monomial_inner_product() -> None:
    from ising.gp_utils import poly4_kernel

    x = np.array([-1.0, 0.5, 2.0])
    eta = 0.7
    k = poly4_kernel(x, x, length_scale=1.0, amplitude=eta)
    expected = (eta**2) * sum(
        (x[:, None] ** power) * (x[None, :] ** power) for power in range(5)
    )
    np.testing.assert_allclose(k, expected)


def test_poly2_gp_log_marginal_likelihood_finite() -> None:
    rng = np.random.default_rng(0)
    z = np.linspace(-2.0, 2.0, 10)
    y = 0.5 + 0.2 * z - 0.05 * z**2 + 0.03 * rng.normal(size=z.size)
    sigma = np.full(z.size, 0.05)
    ll = gp_log_marginal_likelihood(
        z, y, sigma, length_scale=1.0, amplitude=1.0, kernel="poly2"
    )
    assert np.isfinite(ll)


def test_discrepancy_coupling_mixture_and_weighted_finite() -> None:
    rng = np.random.default_rng(1)
    z = np.linspace(-1.5, 1.5, 8)
    y = 0.4 - 0.15 * z + 0.02 * rng.normal(size=z.size)
    sigma = np.full(z.size, 0.03)
    pi = np.linspace(0.0, 0.9, z.size)
    kwargs = dict(
        gp_ell=0.4,
        gp_eta=1.0,
        disc_gp_ell=0.4,
        disc_gp_eta=0.7,
        jitter=1e-6,
    )
    ll_add = discrepancy_gp_log_marginal_likelihood(
        z, y, sigma, pi, coupling="additive", **kwargs
    )
    ll_mix = discrepancy_gp_log_marginal_likelihood(
        z, y, sigma, pi, coupling="mixture", **kwargs
    )
    ll_w = discrepancy_gp_log_marginal_likelihood(
        z, y, sigma, pi, coupling="weighted", **kwargs
    )
    assert np.isfinite(ll_add) and np.isfinite(ll_mix) and np.isfinite(ll_w)
    assert ll_mix != pytest.approx(ll_add, abs=1e-6)
    assert ll_w != pytest.approx(ll_mix, abs=1e-6)

    # π=0 mixture/weighted reduce to a plain GP (g is gated off; f is not).
    zeros = np.zeros_like(pi)
    ll_plain = gp_log_marginal_likelihood(
        z, y, sigma, length_scale=0.4, amplitude=1.0, jitter=1e-6
    )
    ll_mix0 = discrepancy_gp_log_marginal_likelihood(
        z, y, sigma, zeros, coupling="mixture", **kwargs
    )
    ll_w0 = discrepancy_gp_log_marginal_likelihood(
        z, y, sigma, zeros, coupling="weighted", **kwargs
    )
    assert ll_mix0 == pytest.approx(ll_plain, rel=1e-6, abs=1e-6)
    assert ll_w0 == pytest.approx(ll_plain, rel=1e-6, abs=1e-6)
