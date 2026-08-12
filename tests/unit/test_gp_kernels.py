"""Tests for FSS GP kernel dispatch."""

from __future__ import annotations

import numpy as np
import pytest

from ising.gp_kernels import normalize_gp_kernel
from ising.gp_utils import (
    gaussian_kernel,
    gp_log_marginal_likelihood,
    matern52_kernel,
    stationary_kernel,
)


def test_normalize_gp_kernel_aliases() -> None:
    assert normalize_gp_kernel("harada") == "gaussian"
    assert normalize_gp_kernel("rbf") == "gaussian"
    assert normalize_gp_kernel("matern52") == "matern52"
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
