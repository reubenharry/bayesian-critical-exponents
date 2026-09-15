"""Unit tests for discrepancy amplitude and noise inflation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from ising.constants import TC_EXACT
from ising.discrepancy import (
    discrepancy_amplitude,
    discrepancy_amplitude_fss,
    discrepancy_mix_weight,
    discrepancy_q_prior_bounds,
    effective_obs_sigma,
    point_weight,
)
from ising.model_fss import build_fss_model


def test_discrepancy_amplitude_vanishes_in_scaling_limit():
    a = discrepancy_amplitude(t=0.0, L=1e6, t0=0.1, L0=32.0, p=2.0, q=2.0)
    assert float(a) == pytest.approx(0.0, abs=1e-8)


def test_discrepancy_amplitude_grows_at_small_L_large_t():
    a_near = discrepancy_amplitude(t=0.01, L=256.0, t0=0.1, L0=32.0, p=2.0, q=2.0)
    a_far = discrepancy_amplitude(t=1.0, L=8.0, t0=0.1, L0=32.0, p=2.0, q=2.0)
    assert float(a_far) > float(a_near)


def test_discrepancy_amplitude_fss_matches_z_rewrite():
    t = np.array([0.0, 0.05, -0.2])
    L = np.array([8.0, 32.0, 128.0])
    omega, kappa, nu = 2.0, 1.5, 1.0
    z = t * L ** (1.0 / nu)
    got = discrepancy_amplitude_fss(t, L, omega=omega, kappa=kappa, nu=nu)
    expected = L ** (-omega) * (1.0 + kappa * np.abs(z) ** (omega * nu))
    np.testing.assert_allclose(got, expected)


def test_discrepancy_amplitude_fss_vanishes_in_scaling_limit():
    a = discrepancy_amplitude_fss(t=0.0, L=1e6, omega=2.0, kappa=1.0, nu=1.0)
    assert float(a) == pytest.approx(0.0, abs=1e-8)


def test_discrepancy_q_prior_bounds_fss_is_uniform_1_to_10():
    assert discrepancy_q_prior_bounds("additive_gp_fss") == (1.0, 10.0)
    assert discrepancy_q_prior_bounds("additive_gp") == (0.5, 4.0)


def test_discrepancy_mix_weight_squash_and_clip():
    a = np.array([0.0, 1.0, 3.0])
    pi = discrepancy_mix_weight(a, squash=True)
    np.testing.assert_allclose(pi, np.array([0.0, 0.5, 0.75]))
    np.testing.assert_allclose(
        discrepancy_mix_weight(np.array([-1.0, 0.5, 1.5]), squash=False),
        np.array([0.0, 0.5, 1.0]),
    )
    assert 0.0 <= float(discrepancy_mix_weight(1e6)) < 1.0


def test_effective_sigma_matches_mc_when_a_zero():
    sigma = np.array([0.01, 0.02])
    t = np.zeros(2)
    L = np.array([64.0, 128.0])
    out = effective_obs_sigma(
        sigma, t, L, t0=0.1, L0=32.0, p=2.0, q=2.0, sigma_model=0.5
    )
    # a = (L0/L)^q only when t=0
    a = (32.0 / L) ** 2
    expected = np.sqrt(sigma**2 + (0.5 * a) ** 2)
    np.testing.assert_allclose(out, expected)


def test_point_weight_in_unit_interval():
    w = point_weight(0.01, t=0.5, L=8.0, t0=0.1, L0=32.0, p=2.0, q=2.0, sigma_model=0.2)
    assert 0.0 < float(w) <= 1.0


def _tiny_binder_df() -> pd.DataFrame:
    rows = []
    for L, T, u4, std in (
        (64, TC_EXACT, 0.61, 0.01),
        (64, 2.30, 0.55, 0.01),
        (128, TC_EXACT, 0.61, 0.008),
        (128, 2.28, 0.58, 0.008),
    ):
        rows.append(
            {
                "L": L,
                "T": T,
                "binder_cumulant": u4,
                "binder_cumulant_std": std,
                "n_eff_binder": 100.0,
                # Pad required columns unused for binder-only.
                "magnetization": 0.1,
                "magnetization_std": 0.01,
                "n_eff": 100.0,
                "log_magnetization": -2.0,
                "log_magnetization_std": 0.1,
                "n_eff_log": 100.0,
                "susceptibility": 1.0,
                "susceptibility_std": 0.1,
                "n_eff_susceptibility": 100.0,
                "log_susceptibility": 0.0,
                "log_susceptibility_std": 0.1,
                "n_eff_log_susceptibility": 100.0,
                "m2": 0.01,
                "m2_std": 0.001,
                "n_eff_m2": 100.0,
                "m4": 0.0001,
                "m4_std": 0.00001,
                "n_eff_m4": 100.0,
            }
        )
    return pd.DataFrame(rows)


def test_build_fss_model_discrepancy_off_has_no_disc_rvs():
    df = _tiny_binder_df()
    model = build_fss_model(
        df,
        use_m=False,
        use_binder=True,
        correction_binder=False,
        discrepancy_binder=False,
        infer_Tc=True,
        infer_nu=True,
        infer_beta=False,
    )
    names = {rv.name for rv in model.free_RVs}
    assert "disc_t0" not in names
    assert "disc_sigma_model" not in names


def test_build_fss_model_discrepancy_on_adds_disc_rvs():
    df = _tiny_binder_df()
    model = build_fss_model(
        df,
        use_m=False,
        use_binder=True,
        correction_binder=False,
        discrepancy_binder=True,
        infer_Tc=True,
        infer_nu=True,
        infer_beta=False,
    )
    names = {rv.name for rv in model.free_RVs}
    assert {"disc_t0", "disc_L0", "disc_p", "disc_q", "disc_sigma_model"} <= names
