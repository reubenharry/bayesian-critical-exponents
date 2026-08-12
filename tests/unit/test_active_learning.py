"""Tests for active learning utilities."""

from __future__ import annotations

import numpy as np
import pytest

from ising.active_learning import (
    CandidateSpec,
    PosteriorDraws,
    counterfactual_include_point,
    enumerate_extra_sweep_candidates,
    load_posterior_draws,
    rank_candidates,
    score_candidate,
)
from ising.constants import BETA_EXACT, NU_EXACT, TC_EXACT
from ising.explorer_inference import (
    weighted_posterior_histogram,
)
from ising.fss_likelihood import (
    extract_likelihood_arrays,
    fss_log_marginal_likelihood_from_arrays,
)
from ising.gp_utils import (
    gp_log_marginal_likelihood,
    gp_log_predictive_density,
)
from ising.importance_weights import (
    effective_sample_size,
    pareto_smooth_log_weights,
    weighted_uncertainty,
)
from ising.incremental_likelihood import (
    CandidateObservation,
    fss_delta_log_likelihood,
)
from ising.observables import (
    make_observables_df,
    merge_observation_row,
    precision_updated_sigma,
)
from ising.profile_likelihood import FssProfileConfig

az = pytest.importorskip("arviz")


def _binder_observables(n: int = 6) -> "pd.DataFrame":
    import pandas as pd

    L = np.full(n, 48.0)
    T = np.linspace(2.265, 2.275, n)
    u4 = np.linspace(0.62, 0.58, n)
    sigma_u = np.linspace(0.02, 0.005, n)
    return make_observables_df(
        L=L,
        T=T,
        magnetization=np.linspace(0.4, 0.15, n),
        binder_cumulant=u4,
        binder_cumulant_std=sigma_u,
        n_eff_binder=np.full(n, 200.0),
    )


def _binder_config() -> FssProfileConfig:
    return FssProfileConfig(
        use_m=False,
        use_m2=False,
        use_m4=False,
        use_binder=True,
        use_chi=False,
        correction_binder=False,
        infer_Tc=True,
        infer_nu=True,
        infer_beta=False,
    )


def _synthetic_posterior(n: int = 80) -> PosteriorDraws:
    rng = np.random.default_rng(0)
    return PosteriorDraws(
        T_c=TC_EXACT + rng.normal(0.0, 2e-4, n),
        nu=NU_EXACT + rng.normal(0.0, 0.02, n),
        beta=np.full(n, BETA_EXACT),
    )


def test_gp_log_predictive_matches_incremental_marginal_likelihood():
    x = np.linspace(-2.0, 2.0, 8)
    y = np.sin(x)
    sigma = np.full(x.size, 0.05)
    x_new = np.array([0.3])
    y_new = np.array([0.28])
    sigma_new = 0.04
    ell, eta = 0.8, 1.0

    log_ml_base = gp_log_marginal_likelihood(
        x, y, sigma, length_scale=ell, amplitude=eta
    )
    x_aug = np.concatenate([x, x_new])
    y_aug = np.concatenate([y, y_new])
    sigma_aug = np.concatenate([sigma, [sigma_new]])
    log_ml_aug = gp_log_marginal_likelihood(
        x_aug, y_aug, sigma_aug, length_scale=ell, amplitude=eta
    )
    incremental = log_ml_aug - log_ml_base
    predictive = gp_log_predictive_density(
        x,
        y,
        sigma,
        x_new,
        y_new,
        sigma_new=sigma_new,
        length_scale=ell,
        amplitude=eta,
    )[0]
    assert incremental == pytest.approx(predictive, rel=1e-5, abs=1e-5)


def test_precision_updated_sigma_decreases_with_extra_sweeps():
    sigma = np.array([0.01])
    n_eff = np.array([100.0])
    sigma_more = precision_updated_sigma(sigma, n_eff, 100)
    assert sigma_more[0] < sigma[0]
    assert sigma_more[0] == pytest.approx(0.01 * np.sqrt(100.0 / 200.0))


def test_merge_observation_row_updates_matching_lt():
    df = _binder_observables(4)
    L = int(df.iloc[1]["L"])
    T = float(df.iloc[1]["T"])
    old_u = float(df.iloc[1]["binder_cumulant"])
    new_row = {
        "magnetization": float(df.iloc[1]["magnetization"]),
        "magnetization_std": 0.02,
        "n_eff": 100.0,
        "log_magnetization": float(df.iloc[1]["log_magnetization"]),
        "log_magnetization_std": 0.02,
        "n_eff_log": 100.0,
        "binder_cumulant": old_u + 0.001,
        "binder_cumulant_std": 0.004,
        "n_eff_binder": 150.0,
        "m2": float(df.iloc[1]["m2"]),
        "m2_std": 0.01,
        "n_eff_m2": 100.0,
        "m4": float(df.iloc[1]["m4"]),
        "m4_std": 0.005,
        "n_eff_m4": 100.0,
        "susceptibility": float(df.iloc[1]["susceptibility"]),
        "susceptibility_std": 10.0,
        "n_eff_susceptibility": 100.0,
        "log_susceptibility": float(df.iloc[1]["log_susceptibility"]),
        "log_susceptibility_std": 0.1,
        "n_eff_log_susceptibility": 100.0,
        "n_sweeps": 500,
    }
    merged = merge_observation_row(df, L=L, T=T, new_row=new_row)
    row = merged[(merged["L"] == L) & np.isclose(merged["T"], T)].iloc[0]
    assert float(row["n_eff_binder"]) == pytest.approx(350.0)
    assert float(row["binder_cumulant"]) != old_u


def test_fss_delta_log_likelihood_binder_predictive_mean_mode():
    observables = _binder_observables(6)
    config = _binder_config()
    arrays = extract_likelihood_arrays(observables, config)
    row = observables.iloc[2]
    candidate = CandidateObservation(
        L=float(row["L"]),
        T=float(row["T"]),
        y_by_channel={"binder": 999.0},
        sigma_by_channel={"binder": 0.001},
    )
    delta_given = fss_delta_log_likelihood(
        arrays,
        config,
        candidate,
        T_c=TC_EXACT,
        nu=NU_EXACT,
        beta=BETA_EXACT,
        y_mode="given",
    )
    delta_mean = fss_delta_log_likelihood(
        arrays,
        config,
        CandidateObservation(
            L=float(row["L"]),
            T=float(row["T"]),
            y_by_channel={},
            sigma_by_channel={"binder": 0.001},
        ),
        T_c=TC_EXACT,
        nu=NU_EXACT,
        beta=BETA_EXACT,
        y_mode="predictive_mean",
    )
    assert np.isfinite(delta_given)
    assert np.isfinite(delta_mean)
    assert delta_given != pytest.approx(delta_mean)


def test_fss_delta_log_likelihood_binder_channel():
    observables = _binder_observables(6)
    config = _binder_config()
    arrays = extract_likelihood_arrays(observables, config)
    row = observables.iloc[2]
    candidate = CandidateObservation(
        L=float(row["L"]),
        T=float(row["T"]),
        y_by_channel={"binder": float(row["binder_cumulant"])},
        sigma_by_channel={"binder": 0.001},
    )
    delta = fss_delta_log_likelihood(
        arrays,
        config,
        candidate,
        T_c=TC_EXACT,
        nu=NU_EXACT,
        beta=BETA_EXACT,
    )
    assert np.isfinite(delta)


def test_importance_weights_normalize_and_ess():
    log_w = np.array([0.0, -0.5, -1.0, -3.0])
    result = pareto_smooth_log_weights(log_w)
    assert np.isfinite(result.ess)
    assert result.weights.sum() == pytest.approx(1.0)
    samples = np.column_stack([np.arange(4.0), np.arange(4.0) ** 2])
    u = weighted_uncertainty(samples, result.weights)
    assert u > 0.0
    assert effective_sample_size(result.weights) == pytest.approx(result.ess)


def test_rank_candidates_returns_finite_sorted_scores():
    observables = _binder_observables(6)
    config = _binder_config()
    draws = _synthetic_posterior(60)
    candidates = enumerate_extra_sweep_candidates(observables, (5_000,))
    ranked = rank_candidates(
        observables,
        config,
        draws,
        candidates,
        infer_Tc=True,
        infer_nu=True,
        infer_beta=False,
        seed=0,
    )
    assert not ranked.empty
    assert ranked["delta_uncertainty"].notna().all()
    assert ranked["delta_uncertainty"].is_monotonic_decreasing


def test_load_posterior_draws_from_idata():
    idata = az.from_dict(
        posterior={
            "T_c": np.random.default_rng(0).normal(TC_EXACT, 1e-4, (2, 10)),
            "nu": np.random.default_rng(1).normal(NU_EXACT, 0.01, (2, 10)),
        }
    )
    draws = load_posterior_draws(idata, infer_Tc=True, infer_nu=True, infer_beta=False)
    assert draws.n_samples == 20
    assert draws.beta.shape == (20,)


def test_score_candidate_returns_finite_metrics():
    observables = _binder_observables(6)
    config = _binder_config()
    draws = _synthetic_posterior(40)
    row = observables.iloc[0]
    candidate = CandidateSpec(
        L=int(row["L"]),
        T=float(row["T"]),
        extra_sweeps=10_000,
    )
    score = score_candidate(
        observables,
        config,
        draws,
        candidate,
        infer_beta=False,
        n_y_samples=2,
        seed=1,
    )
    assert np.isfinite(score.delta_uncertainty)
    assert score.ess > 0.0


def test_weighted_posterior_histogram_preserves_support():
    rng = np.random.default_rng(0)
    tc = rng.normal(2.269, 0.01, 200)
    nu = rng.normal(0.99, 0.05, 200)
    weights = np.exp(-0.5 * ((tc - 2.27) / 0.005) ** 2)
    weights /= weights.sum()
    tc_edges = np.linspace(2.25, 2.29, 51)
    nu_edges = np.linspace(0.9, 1.1, 51)
    z = weighted_posterior_histogram(tc, nu, weights, tc_edges, nu_edges)
    assert z.shape == (50, 50)
    assert np.any(z > 0)
    # Mass should span multiple bins, not collapse to one column.
    assert int(np.count_nonzero(z > 0.01 * z.max())) > 5


def test_counterfactual_include_point_reduces_or_changes_tr_cov():
    observables = _binder_observables(6)
    config = _binder_config()
    draws = _synthetic_posterior(40)
    row = observables.iloc[3]
    result = counterfactual_include_point(
        observables.iloc[:3].reset_index(drop=True),
        config,
        draws,
        row,
        row_index=3,
        infer_Tc=True,
        infer_nu=True,
        infer_beta=False,
    )
    assert result.row_index == 3
    assert np.isfinite(result.delta_tr_cov)
    assert result.tr_cov_before > 0.0
    assert result.ess > 0.0
    assert result.weights.sum() == pytest.approx(1.0)


def test_counterfactual_ignores_row_binder_uses_predictive_mean():
    observables = _binder_observables(6)
    config = _binder_config()
    draws = _synthetic_posterior(40)
    included = observables.iloc[:3].reset_index(drop=True)
    row = observables.iloc[3].copy()
    row_alt = row.copy()
    row_alt["binder_cumulant"] = 999.0
    result = counterfactual_include_point(
        included,
        config,
        draws,
        row,
        row_index=3,
        infer_Tc=True,
        infer_nu=True,
        infer_beta=False,
    )
    result_alt = counterfactual_include_point(
        included,
        config,
        draws,
        row_alt,
        row_index=3,
        infer_Tc=True,
        infer_nu=True,
        infer_beta=False,
    )
    assert result.delta_tr_cov == pytest.approx(result_alt.delta_tr_cov)
    assert result.tr_cov_after == pytest.approx(result_alt.tr_cov_after)


def test_is_uncertainty_below_refit_baseline_smoke():
    """IS updated trace-cov should differ from baseline when data are informative."""
    observables = _binder_observables(6)
    config = _binder_config()
    arrays = extract_likelihood_arrays(observables, config)
    draws = _synthetic_posterior(30)
    exponent_samples = draws.as_matrix(infer_Tc=True, infer_nu=True, infer_beta=False)
    u0 = np.trace(np.cov(exponent_samples, rowvar=False))

    row = observables.iloc[np.argmax(observables["binder_cumulant_std"])]
    candidate = CandidateObservation(
        L=float(row["L"]),
        T=float(row["T"]),
        y_by_channel={"binder": float(row["binder_cumulant"])},
        sigma_by_channel={"binder": 0.0005},
    )
    log_w = np.array(
        [
            fss_delta_log_likelihood(
                arrays,
                config,
                candidate,
                T_c=float(draws.T_c[i]),
                nu=float(draws.nu[i]),
                beta=float(draws.beta[i]),
            )
            for i in range(draws.n_samples)
        ]
    )
    iw = pareto_smooth_log_weights(log_w)
    u1 = weighted_uncertainty(exponent_samples, iw.weights)
    assert u1 != pytest.approx(u0, rel=0.0, abs=0.0)

    log_ml_ref = fss_log_marginal_likelihood_from_arrays(
        arrays,
        config,
        T_c=float(np.mean(draws.T_c)),
        nu=float(np.mean(draws.nu)),
        beta=BETA_EXACT,
    )
    assert np.isfinite(log_ml_ref)
