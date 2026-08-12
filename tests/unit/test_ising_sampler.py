"""Smoke tests for the 2D Ising Wolff sampler."""

from __future__ import annotations

import numpy as np
import pytest
from ising.constants import TC_EXACT
from ising.sampler import (
    _blackjax_effective_sample_size,
    _sample_mean_std,
    run_ising,
)


def test_run_ising_returns_reasonable_magnetization():
    result = run_ising(16, TC_EXACT * 0.95, n_thermalize=100, n_sweeps=500, seed=0)
    assert result.L == 16
    assert 0.0 < result.magnetization < 1.0
    assert result.magnetization_std > 0.0
    assert result.n_eff >= 1.0
    assert result.m2 > 0.0
    assert result.m2_std > 0.0
    assert result.n_eff_m2 >= 1.0
    assert result.m4 > 0.0
    assert result.m4_std > 0.0
    assert result.n_eff_m4 >= 1.0
    assert result.m2 >= result.m4
    assert result.log_magnetization < np.log(result.magnetization)
    assert result.log_magnetization_std > 0.0
    assert result.n_eff_log >= 1.0
    assert 0.0 < result.binder_cumulant < 1.0
    assert result.binder_cumulant_std > 0.0
    assert result.n_eff_binder >= 1.0


def test_run_ising_binder_near_tc():
    result = run_ising(16, TC_EXACT, n_thermalize=200, n_sweeps=2000, seed=3)
    assert 0.55 < result.binder_cumulant < 0.68


def test_run_ising_binder_high_temperature():
    result = run_ising(64, 2.5, n_thermalize=500, n_sweeps=5000, seed=42)
    assert result.binder_cumulant < 0.15
    assert abs(result.binder_cumulant - (1.0 - result.m4 / (3.0 * result.m2**2))) < 1e-12


def test_binder_uncertainty_uses_jackknife_not_delta_method():
    """Binder SE comes from block jackknife; delta-method helper must not exist."""
    import ising.sampler as sampler_mod
    from ising.sampler import (
        _binder_from_moments,
        _jackknife_binder_se,
        summarize_signed_m_samples,
    )

    assert not hasattr(sampler_mod, "_binder_std_from_moments")

    rng = np.random.default_rng(0)
    # Near-Gaussian magnetization draws → small U4, nontrivial jackknife SE.
    m = rng.normal(0.0, 0.1, size=20_000)
    m2 = m * m
    m4 = m**4
    se = _jackknife_binder_se(m2, m4, n_blocks=50)
    u = _binder_from_moments(float(m2.mean()), float(m4.mean()))
    assert se > 0.0
    assert se < 0.05
    assert u < 0.1

    result = summarize_signed_m_samples(
        m,
        L=64,
        T=2.5,
        n_thermalize=0,
        n_sweeps=m.size,
        seed=0,
    )
    sigma = result.binder_cumulant_std / np.sqrt(result.n_eff_binder)
    assert sigma == pytest.approx(se, rel=0.05, abs=1e-4)


def _observables_df(
    n: int = 8,
    *,
    L: np.ndarray | None = None,
    T: np.ndarray | None = None,
    magnetization: np.ndarray | None = None,
    **kwargs: object,
):
    from ising.observables import make_observables_df

    if L is None:
        L = np.full(n, 16.0)
    if T is None:
        T = np.linspace(2.25, 2.28, n)
    if magnetization is None:
        magnetization = np.linspace(0.5, 0.2, n)
    return make_observables_df(L=L, T=T, magnetization=magnetization, **kwargs)


def test_fss_model_builds_with_binder():
    from ising.model_fss import build_fss_model

    n = 8
    u4 = np.linspace(0.62, 0.58, n)
    sigma_u = np.full(n, 0.01)
    df = _observables_df(
        n,
        binder_cumulant=u4,
        sigma_binder=sigma_u,
    )
    model = build_fss_model(
        df,
        use_log_m=False,
        infer_nu=True,
        infer_beta=False,
        correction_m=True,
        correction_binder=True,
    )
    assert model._ising_model == "fss"  # type: ignore[attr-defined]
    assert model._ising_include_binder is True  # type: ignore[attr-defined]
    assert model._ising_fss_correction is True  # type: ignore[attr-defined]
    assert [v.name for v in model.value_vars] == ["nu_interval__"]

    legacy = build_fss_model(
        df,
        use_log_m=False,
        correction_m=False,
        correction_binder=False,
    )
    assert legacy._ising_fss_correction is False  # type: ignore[attr-defined]


def test_fss_model_builds_with_chi_correction():
    from ising.model_fss import build_fss_model

    n = 8
    chi = np.linspace(80.0, 120.0, n)
    sigma_chi = np.full(n, 10.0)
    df = _observables_df(n, susceptibility=chi, sigma_chi=sigma_chi)
    model = build_fss_model(
        df,
        use_m=False,
        use_chi=True,
        infer_nu=True,
        correction_chi=True,
    )
    assert model._ising_include_chi is True  # type: ignore[attr-defined]
    assert model._ising_fss_correction is True  # type: ignore[attr-defined]
    assert [v.name for v in model.value_vars] == ["nu_interval__"]


def test_fss_model_builds_magnetization_only():
    from ising.model_fss import build_fss_model

    df = _observables_df()
    model = build_fss_model(
        df,
        use_log_m=False,
        infer_Tc=True,
        infer_nu=True,
        infer_beta=False,
        correction_m=True,
    )
    assert model._ising_model == "fss"  # type: ignore[attr-defined]
    assert model._ising_include_binder is False  # type: ignore[attr-defined]
    assert model._ising_fss_correction is True  # type: ignore[attr-defined]
    assert {v.name for v in model.value_vars} == {"T_c_interval__", "nu_interval__"}


def test_fss_model_builds_with_m2_m4():
    from ising.model_fss import build_fss_model

    n = 8
    m2 = np.linspace(0.3, 0.05, n)
    sigma_m2 = np.full(n, 0.01)
    m4 = np.linspace(0.15, 0.01, n)
    sigma_m4 = np.full(n, 0.005)
    df = _observables_df(n, m2=m2, sigma_m2=sigma_m2, m4=m4, sigma_m4=sigma_m4)
    model = build_fss_model(
        df,
        use_m=False,
        use_m2=True,
        use_m4=True,
        infer_nu=True,
        infer_beta=False,
        correction_m2=True,
        correction_m4=False,
    )
    assert model._ising_include_m2 is True  # type: ignore[attr-defined]
    assert model._ising_include_m4 is True  # type: ignore[attr-defined]
    assert model._ising_fss_correction_m2 is True  # type: ignore[attr-defined]
    assert model._ising_fss_correction_m4 is False  # type: ignore[attr-defined]
    assert [v.name for v in model.value_vars] == ["nu_interval__"]


def test_fss_model_builds_with_inferred_omega_and_gp_hyperparams():
    from ising.model_fss import build_fss_model

    n = 8
    m2 = np.linspace(0.3, 0.05, n)
    sigma_m2 = np.full(n, 0.01)
    df = _observables_df(n, m2=m2, sigma_m2=sigma_m2)
    model = build_fss_model(
        df,
        use_m=False,
        use_m2=True,
        infer_nu=True,
        infer_beta=False,
        correction_m2=True,
        infer_omega=True,
        infer_gp_hyperparams=True,
    )
    assert model._ising_infer_omega is True  # type: ignore[attr-defined]
    assert model._ising_infer_gp_hyperparams is True  # type: ignore[attr-defined]
    assert {v.name for v in model.value_vars} == {
        "nu_interval__",
        "gp_ell_log__",
        "gp_eta_log__",
        "correction_gp_ell_log__",
        "correction_gp_eta_log__",
        "omega_interval__",
    }


def test_profile_tc_likelihood_matches_pymc_logp():
    from ising.constants import (
        BETA_EXACT,
        NU_EXACT,
        TC_EXACT,
    )
    from ising.model_fss import build_fss_model
    from ising.profile_likelihood import (
        FssProfileConfig,
        fss_log_marginal_likelihood,
    )

    n = 8
    m2 = np.linspace(0.3, 0.05, n)
    sigma_m2 = np.full(n, 0.01)
    df = _observables_df(n, m2=m2, sigma_m2=sigma_m2)

    cfg = FssProfileConfig(
        use_m=False,
        use_m2=True,
        use_m4=False,
        correction_m2=True,
        use_log_m=False,
        infer_Tc=False,
        infer_nu=False,
        infer_beta=False,
    )
    ll_np = fss_log_marginal_likelihood(
        observables=df,
        T_c=TC_EXACT,
        nu=NU_EXACT,
        beta=BETA_EXACT,
        config=cfg,
    )

    model = build_fss_model(
        df,
        use_m=False,
        use_m2=True,
        infer_Tc=False,
        infer_nu=False,
        infer_beta=False,
        correction_m2=True,
        use_log_m=False,
    )
    ll_pm = float(model.compile_logp()({}))
    assert ll_np == pytest.approx(ll_pm, rel=1e-4, abs=1e-2)


def _synthetic_profile_observables(n: int = 12):
    from ising.observables import make_observables_df

    L = np.full(n, 48.0)
    T = np.linspace(2.265, 2.275, n)
    m = np.linspace(0.4, 0.15, n)
    sigma_m = np.full(n, 0.02)
    m2 = np.linspace(0.2, 0.04, n)
    sigma_m2 = np.full(n, 0.01)
    m4 = np.linspace(0.08, 0.01, n)
    sigma_m4 = np.full(n, 0.005)
    return make_observables_df(
        L=L,
        T=T,
        magnetization=m,
        sigma_m=sigma_m,
        m2=m2,
        sigma_m2=sigma_m2,
        m4=m4,
        sigma_m4=sigma_m4,
    )


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(
            {
                "use_m": True,
                "use_m2": True,
                "use_m4": True,
                "correction_m": True,
                "correction_m2": False,
                "correction_m4": False,
                "use_log_m": False,
                "infer_Tc": True,
                "infer_nu": True,
                "infer_beta": True,
                "reparametrize_Tc": True,
            },
            id="run-like-reparam",
        ),
        pytest.param(
            {
                "use_m": False,
                "use_m2": True,
                "use_m4": False,
                "correction_m2": True,
                "use_log_m": False,
                "infer_Tc": True,
                "infer_nu": True,
                "infer_beta": True,
                "reparametrize_Tc": False,
            },
            id="m2-only-direct-tc",
        ),
        pytest.param(
            {
                "use_m": False,
                "use_m2": True,
                "use_m4": False,
                "correction_m2": True,
                "use_log_m": False,
                "infer_Tc": False,
                "infer_nu": False,
                "infer_beta": False,
            },
            id="fixed-exponents",
        ),
    ],
)
def test_profile_numba_matches_slow_loglikelihood(config: dict[str, object]) -> None:
    """Numba and point_logps profile evaluators must agree on random exponent triples."""
    from ising.constants import (
        BETA_EXACT,
        NU_EXACT,
        TC_EXACT,
    )
    from ising.model_fss import (
        BETA_PRIOR_LOWER,
        BETA_PRIOR_UPPER,
        NU_PRIOR_LOWER,
        NU_PRIOR_UPPER,
        TC_PRIOR_LOWER,
        TC_PRIOR_UPPER,
        compile_fss_exponent_log_likelihood,
    )
    from ising.profile_likelihood import (
        FssProfileConfig,
        _profile_build_kwargs,
    )

    obs = _synthetic_profile_observables()
    cfg = FssProfileConfig(**config)
    _, fast = compile_fss_exponent_log_likelihood(
        obs,
        use_numba=True,
        **_profile_build_kwargs(cfg),
    )
    _, slow = compile_fss_exponent_log_likelihood(
        obs,
        use_numba=False,
        **_profile_build_kwargs(cfg),
    )

    rng = np.random.default_rng(0)
    triples = [(TC_EXACT, NU_EXACT, BETA_EXACT)]
    for _ in range(15):
        triples.append(
            (
                float(rng.uniform(TC_PRIOR_LOWER, TC_PRIOR_UPPER)),
                float(rng.uniform(NU_PRIOR_LOWER, NU_PRIOR_UPPER)),
                float(rng.uniform(BETA_PRIOR_LOWER, BETA_PRIOR_UPPER)),
            )
        )

    for tc, nu, beta in triples:
        fast_ll = fast(tc, nu, beta)
        slow_ll = slow(tc, nu, beta)
        assert fast_ll == pytest.approx(
            slow_ll,
            rel=1e-5,
            abs=0.02,
        ), (
            f"numba/slow mismatch at (T_c, nu, beta)=({tc}, {nu}, {beta}): "
            f"fast={fast_ll}, slow={slow_ll}, diff={fast_ll - slow_ll}"
        )


def test_binder_only_profile_likelihood_compiles() -> None:
    """Binder-only likelihood does not depend on beta; profile compile must not fail."""
    from ising.constants import (
        BETA_EXACT,
        NU_EXACT,
        TC_EXACT,
    )
    from ising.model_fss import (
        compile_fss_exponent_log_likelihood,
    )
    from ising.profile_likelihood import (
        FssProfileConfig,
        _profile_build_kwargs,
    )

    obs = _synthetic_profile_observables()
    cfg = FssProfileConfig(
        use_m=False,
        use_m2=False,
        use_m4=False,
        use_binder=True,
        use_chi=False,
    )
    assert _profile_build_kwargs(cfg)["infer_beta"] is False
    _, evaluate = compile_fss_exponent_log_likelihood(
        obs,
        use_numba=True,
        **_profile_build_kwargs(cfg),
    )
    ll = evaluate(TC_EXACT, NU_EXACT, BETA_EXACT)
    assert np.isfinite(ll)


def test_tc_profile_loglikelihood_matches_numpy_reference() -> None:
    """T_c profile must evaluate the same GP likelihood as the NumPy reference."""
    from ising.constants import (
        BETA_EXACT,
        NU_EXACT,
        OMEGA_EXACT,
        TC_EXACT,
    )
    from ising.gp_utils import (
        correction_gp_log_marginal_likelihood,
        gp_log_marginal_likelihood,
    )
    from ising.model_fss import (
        TC_PRIOR_LOWER,
        TC_PRIOR_UPPER,
        compile_fss_exponent_log_likelihood,
    )
    from ising.profile_likelihood import (
        FssProfileConfig,
        _default_grid,
        _profile_build_kwargs,
    )
    from ising.scaling_function import (
        GP_JITTER,
        provisional_z,
    )

    obs = _synthetic_profile_observables()
    cfg = FssProfileConfig(
        use_m=True,
        use_m2=True,
        use_m4=True,
        correction_m=True,
        correction_m2=False,
        correction_m4=False,
        use_log_m=False,
        infer_Tc=True,
        infer_nu=False,
        infer_beta=False,
        reparametrize_Tc=True,
        gp_ell_factor=2.0,
        gp_eta=2.0,
    )
    _, pymc_eval = compile_fss_exponent_log_likelihood(
        obs,
        use_numba=False,
        **_profile_build_kwargs(cfg),
    )

    L = obs["L"].to_numpy(dtype=np.float64)
    T = obs["T"].to_numpy(dtype=np.float64)
    magnetization = obs["magnetization"].to_numpy(dtype=np.float64)
    sigma_m = obs["magnetization_std"].to_numpy(dtype=np.float64) / np.sqrt(
        np.maximum(obs["n_eff"].to_numpy(dtype=np.float64), 1.0)
    )
    m2 = obs["m2"].to_numpy(dtype=np.float64)
    sigma_m2 = obs["m2_std"].to_numpy(dtype=np.float64) / np.sqrt(
        np.maximum(obs["n_eff_m2"].to_numpy(dtype=np.float64), 1.0)
    )
    m4 = obs["m4"].to_numpy(dtype=np.float64)
    sigma_m4 = obs["m4_std"].to_numpy(dtype=np.float64) / np.sqrt(
        np.maximum(obs["n_eff_m4"].to_numpy(dtype=np.float64), 1.0)
    )

    z_ref = provisional_z(L, T, T_c=TC_EXACT, nu=NU_EXACT)
    z_span = float(z_ref.max() - z_ref.min())
    gp_ell = cfg.gp_ell_factor * z_span

    def numpy_log_likelihood(T_c: float) -> float:
        t = (T - T_c) / T_c
        z = t * L ** (1.0 / NU_EXACT)
        ll_m = correction_gp_log_marginal_likelihood(
            z,
            magnetization * L ** (BETA_EXACT / NU_EXACT),
            sigma_m * L ** (BETA_EXACT / NU_EXACT),
            L,
            omega=OMEGA_EXACT,
            gp_ell=gp_ell,
            gp_eta=cfg.gp_eta,
            correction_gp_ell=1.0 * z_span,
            correction_gp_eta=1.0,
            jitter=GP_JITTER,
        )
        ll_m2 = gp_log_marginal_likelihood(
            z,
            m2 * L ** (2 * BETA_EXACT / NU_EXACT),
            sigma_m2 * L ** (2 * BETA_EXACT / NU_EXACT),
            length_scale=gp_ell,
            amplitude=cfg.gp_eta,
            jitter=GP_JITTER,
        )
        ll_m4 = gp_log_marginal_likelihood(
            z,
            m4 * L ** (4 * BETA_EXACT / NU_EXACT),
            sigma_m4 * L ** (4 * BETA_EXACT / NU_EXACT),
            length_scale=gp_ell,
            amplitude=cfg.gp_eta,
            jitter=GP_JITTER,
        )
        return ll_m + ll_m2 + ll_m4

    tc_values = [TC_EXACT, *list(_default_grid("T_c", 7))]
    for tc in tc_values:
        tc = float(tc)
        pymc_ll = pymc_eval(tc, NU_EXACT, BETA_EXACT)
        numpy_ll = numpy_log_likelihood(tc)
        assert pymc_ll == pytest.approx(
            numpy_ll,
            rel=1e-5,
            abs=0.02,
        ), (
            f"T_c profile mismatch at T_c={tc}: pymc={pymc_ll}, numpy={numpy_ll}, "
            f"diff={pymc_ll - numpy_ll}"
        )

    # Interval map must use PyMC's logit transform, not a linear fraction of the prior.
    from ising.model_fss import _pymc_interval_forward

    u_exact = _pymc_interval_forward(TC_EXACT, TC_PRIOR_LOWER, TC_PRIOR_UPPER)
    u_linear = (TC_EXACT - TC_PRIOR_LOWER) / (TC_PRIOR_UPPER - TC_PRIOR_LOWER)
    assert u_exact != pytest.approx(u_linear, rel=1e-3)
    s = 1.0 / (1.0 + np.exp(-u_exact))
    tc_roundtrip = s * TC_PRIOR_UPPER + (1.0 - s) * TC_PRIOR_LOWER
    assert tc_roundtrip == pytest.approx(TC_EXACT, rel=1e-6, abs=1e-6)


def test_run_ising_is_reproducible_with_seed():
    kwargs = dict(L=12, T=2.1, n_thermalize=50, n_sweeps=200)
    a = run_ising(**kwargs, seed=7)
    b = run_ising(**kwargs, seed=7)
    assert a.magnetization == b.magnetization


def test_magnetization_noise_uses_sample_std_and_blackjax_ess(monkeypatch):
    rng = np.random.default_rng(0)
    samples = rng.normal(loc=0.7, scale=0.05, size=500)
    mean, std = _sample_mean_std(samples)
    assert mean == pytest.approx(float(samples.mean()))
    assert std == pytest.approx(float(samples.std(ddof=1)))

    ess_calls: list[np.ndarray] = []

    def _record_ess(chain_samples: np.ndarray) -> np.ndarray:
        ess_calls.append(np.asarray(chain_samples))
        return np.asarray(123.0)

    monkeypatch.setattr(
        "ising.sampler.effective_sample_size",
        _record_ess,
    )
    ess = _blackjax_effective_sample_size(samples)
    assert ess == 123.0
    assert ess_calls[0].shape == (1, samples.size)
