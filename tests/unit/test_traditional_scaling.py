"""Tests for classical Binder + log-log FSS analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ising.constants import (
    BETA_EXACT,
    GAMMA_EXACT,
    NU_EXACT,
    TC_EXACT,
)
from ising.traditional_scaling import (
    all_binder_crossings,
    estimate_beta_over_nu_at_tc,
    estimate_nu_from_collapse_chi2,
    estimate_tc_from_binder_crossings,
    fit_single_scaling_function_spline,
    load_observables_table,
    magnetization_collapse_arrays,
    run_traditional_scaling_analysis,
)


def _synthetic_binder_df() -> pd.DataFrame:
    """Minimal grid with Binder crossings and power-law |m|, χ at T_c."""
    rows = []
    seed = 0
    T_c = TC_EXACT
    for L in (32, 64, 128):
        T_ref = T_c - 0.001 * np.log(L / 32.0)
        slope = 3.5 + 0.5 * np.log(L / 32.0)
        for T in np.linspace(2.24, 2.30, 11):
            t = (T - T_c) / T_c
            u = 0.60 - slope * (T - T_ref)
            m = 0.85 * (L ** (-BETA_EXACT / NU_EXACT)) * max(1.0 - 2.0 * t, 0.05)
            chi = 350.0 * (L ** (GAMMA_EXACT / NU_EXACT)) * (1.0 + 0.1 * abs(t))
            rows.append(
                dict(
                    L=L,
                    T=float(T),
                    magnetization=float(m),
                    magnetization_std=0.02,
                    n_eff=1000.0,
                    binder_cumulant=float(u),
                    susceptibility=float(chi),
                    susceptibility_std=20.0,
                    n_eff_susceptibility=1000.0,
                    seed=seed,
                )
            )
            seed += 1
    return pd.DataFrame(rows)


def test_binder_crossing_interpolation_finds_temperature() -> None:
    df = _synthetic_binder_df()
    crossings = all_binder_crossings(df)
    assert len(crossings) >= 2
    assert crossings[0].L_small == 32
    assert 2.24 < crossings[0].T_cross < 2.30


def test_loglog_beta_over_nu_recovers_power_law() -> None:
    df = _synthetic_binder_df()
    T_c, _ = estimate_tc_from_binder_crossings(all_binder_crossings(df))
    fit = estimate_beta_over_nu_at_tc(df, T_c)
    assert fit.slope == pytest.approx(BETA_EXACT / NU_EXACT, rel=0.15, abs=0.03)


def test_fit_single_scaling_function_spline_on_synthetic() -> None:
    df = _synthetic_binder_df()
    T_c, _ = estimate_tc_from_binder_crossings(all_binder_crossings(df))
    z, phi, sigma_phi, _ = magnetization_collapse_arrays(df, T_c, NU_EXACT, BETA_EXACT)
    z_line, phi_line = fit_single_scaling_function_spline(z, phi, sigma_phi)
    assert z_line.size > 10
    assert phi_line.shape == z_line.shape
    assert np.all(np.isfinite(phi_line))


@pytest.mark.requires_analysis
def test_collapse_chi2_nu_on_critical_fss() -> None:
    path = Path(
        "ising/data/observables_critical_fss.csv"
    )
    if not path.exists():
        pytest.skip("critical_fss observables not present")
    from ising.datasets import get_dataset

    df = load_observables_table(path)
    spec = get_dataset("critical_fss")
    result = estimate_nu_from_collapse_chi2(df, spec, channels=("m",))
    assert 0.9 < result.nu_hat < 1.1
    assert result.chi2_min < result.chi2_exact
    assert result.n_dof > 0


@pytest.mark.requires_analysis
def test_run_traditional_scaling_on_critical_large_z() -> None:
    path = Path(
        "ising/data/observables_critical_large_z.csv"
    )
    if not path.exists():
        pytest.skip("critical_large_z observables not present")
    result = run_traditional_scaling_analysis("critical_large_z", data=path)
    assert len(result.crossings) >= 2
    assert 2.2 < result.T_c_binder < 2.32
    assert result.beta_over_nu > 0.0
    assert result.gamma_over_nu > 0.0
