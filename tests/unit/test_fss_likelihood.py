"""Parity tests for shared FSS log marginal likelihood (NumPy / JAX / PyMC)."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from ising.constants import (  # noqa: E402
    BETA_EXACT,
    NU_EXACT,
    TC_EXACT,
)
from ising.fss_likelihood import (  # noqa: E402
    compile_fss_log_marginal_likelihood_jax,
    fss_log_marginal_likelihood,
)
from ising.gp_jax import (  # noqa: E402
    correction_gp_log_marginal_likelihood as corr_gp_jax,
    correction_gp_log_predictive_density as corr_pred_jax,
    gp_log_marginal_likelihood as gp_jax,
    gp_log_predictive_density as gp_pred_jax,
)
from ising.gp_utils import (  # noqa: E402
    correction_gp_log_marginal_likelihood as corr_gp_np,
    correction_gp_log_predictive_density as corr_pred_np,
    gp_log_marginal_likelihood as gp_np,
    gp_log_predictive_density as gp_pred_np,
)
from ising.model_fss import (  # noqa: E402
    BETA_PRIOR_LOWER,
    BETA_PRIOR_UPPER,
    NU_PRIOR_LOWER,
    NU_PRIOR_UPPER,
    TC_PRIOR_LOWER,
    TC_PRIOR_UPPER,
    compile_fss_exponent_log_likelihood,
)
from ising.observables import (  # noqa: E402
    make_observables_df,
    read_harada_binder_observables,
)
from ising.profile_likelihood import (  # noqa: E402
    FssProfileConfig,
    _profile_build_kwargs,
    default_profile_config,
)


def _synthetic_observables(n: int = 12):
    L = np.full(n, 48.0)
    T = np.linspace(2.265, 2.275, n)
    m = np.linspace(0.4, 0.15, n)
    sigma_m = np.full(n, 0.02)
    m2 = np.linspace(0.2, 0.04, n)
    sigma_m2 = np.full(n, 0.01)
    m4 = np.linspace(0.08, 0.01, n)
    sigma_m4 = np.full(n, 0.005)
    u4 = np.linspace(0.62, 0.58, n)
    sigma_u = np.full(n, 0.01)
    chi = np.linspace(80.0, 120.0, n)
    sigma_chi = np.full(n, 10.0)
    return make_observables_df(
        L=L,
        T=T,
        magnetization=m,
        sigma_m=sigma_m,
        m2=m2,
        sigma_m2=sigma_m2,
        m4=m4,
        sigma_m4=sigma_m4,
        binder_cumulant=u4,
        sigma_binder=sigma_u,
        susceptibility=chi,
        sigma_chi=sigma_chi,
    )


def _exponent_triples(n_random: int = 10):
    rng = np.random.default_rng(0)
    triples = [(TC_EXACT, NU_EXACT, BETA_EXACT)]
    for _ in range(n_random):
        triples.append(
            (
                float(rng.uniform(TC_PRIOR_LOWER, TC_PRIOR_UPPER)),
                float(rng.uniform(NU_PRIOR_LOWER, NU_PRIOR_UPPER)),
                float(rng.uniform(BETA_PRIOR_LOWER, BETA_PRIOR_UPPER)),
            )
        )
    return triples


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(
            FssProfileConfig(
                use_m=False,
                use_m2=False,
                use_m4=False,
                use_binder=True,
                use_chi=False,
                correction_binder=True,
                infer_Tc=True,
                infer_nu=True,
                infer_beta=False,
            ),
            id="binder-correction",
        ),
        pytest.param(
            FssProfileConfig(
                use_m=True,
                use_m2=True,
                use_m4=True,
                correction_m=True,
                correction_m2=False,
                correction_m4=False,
                use_log_m=False,
                infer_Tc=True,
                infer_nu=True,
                infer_beta=True,
                reparametrize_Tc=True,
                gp_ell_factor=2.0,
                gp_eta=2.0,
            ),
            id="m-m2-m4-mixed",
        ),
        pytest.param(
            FssProfileConfig(
                use_m=True,
                use_m2=False,
                use_m4=False,
                use_binder=False,
                use_chi=False,
                correction_m=False,
                use_log_m=True,
                infer_Tc=True,
                infer_nu=True,
                infer_beta=True,
            ),
            id="log-m-plain",
        ),
        pytest.param(
            FssProfileConfig(
                use_m=False,
                use_m2=False,
                use_m4=False,
                use_binder=False,
                use_chi=True,
                correction_chi=True,
                infer_Tc=True,
                infer_nu=True,
                infer_beta=True,
            ),
            id="chi-correction-infer-nu",
        ),
        pytest.param(
            FssProfileConfig(
                use_m=False,
                use_m2=False,
                use_m4=False,
                use_binder=False,
                use_chi=True,
                correction_chi=False,
                infer_Tc=True,
                infer_nu=False,
                infer_beta=True,
            ),
            id="chi-plain-fixed-nu",
        ),
    ],
)
def test_fss_likelihood_jax_matches_numpy(config: FssProfileConfig) -> None:
    obs = _synthetic_observables()
    jax_eval = compile_fss_log_marginal_likelihood_jax(obs, config)
    for tc, nu, beta in _exponent_triples():
        np_ll = fss_log_marginal_likelihood(
            obs, config, T_c=tc, nu=nu, beta=beta, backend="numpy"
        )
        jax_ll = jax_eval(tc, nu, beta)
        assert jax_ll == pytest.approx(np_ll, rel=1e-5, abs=1e-4), (
            f"jax/numpy mismatch at ({tc}, {nu}, {beta}): "
            f"jax={jax_ll}, numpy={np_ll}"
        )


def test_fss_likelihood_jax_batched_profiles_match_scalar(
    config: FssProfileConfig | None = None,
) -> None:
    config = config or FssProfileConfig(
        use_m=False,
        use_m2=False,
        use_m4=False,
        use_binder=True,
        use_chi=False,
        correction_binder=True,
        infer_Tc=True,
        infer_nu=True,
        infer_beta=False,
    )
    obs = _synthetic_observables()
    compiled = compile_fss_log_marginal_likelihood_jax(obs, config)
    rng = np.random.default_rng(2)
    tc, nu, beta = TC_EXACT, NU_EXACT, BETA_EXACT
    tc_grid = np.linspace(2.2, 2.3, 12)
    nu_grid = np.linspace(0.8, 1.2, 12)
    beta_grid = np.linspace(0.05, 0.25, 12)
    np.testing.assert_allclose(
        compiled.profile_tc(tc_grid, nu, beta),
        [compiled(tc, nu, beta) for tc in tc_grid],
        rtol=1e-6,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        compiled.profile_nu(nu_grid, tc, beta),
        [compiled(tc, n, beta) for n in nu_grid],
        rtol=1e-6,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        compiled.profile_beta(beta_grid, tc, nu),
        [compiled(tc, nu, b) for b in beta_grid],
        rtol=1e-6,
        atol=1e-5,
    )
    joint = compiled.profile_joint(tc_grid, nu_grid, "T_c_nu", tc, nu, beta)
    for j, y in enumerate(nu_grid):
        for i, x in enumerate(tc_grid):
            assert joint[j, i] == pytest.approx(compiled(x, y, beta), rel=1e-6, abs=1e-5)
    joint_nb = compiled.profile_joint(nu_grid, beta_grid, "nu_beta", tc, nu, beta)
    for j, b in enumerate(beta_grid):
        for i, n in enumerate(nu_grid):
            assert joint_nb[j, i] == pytest.approx(compiled(tc, n, b), rel=1e-6, abs=1e-5)
    cube = compiled.profile_triple_grid(tc_grid[:5], nu_grid[:5], beta_grid[:5])
    for i, t in enumerate(tc_grid[:5]):
        for j, n in enumerate(nu_grid[:5]):
            for k, b in enumerate(beta_grid[:5]):
                assert cube[i, j, k] == pytest.approx(compiled(t, n, b), rel=1e-6, abs=1e-5)


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(
            FssProfileConfig(
                use_m=False,
                use_m2=False,
                use_m4=False,
                use_binder=True,
                use_chi=False,
                correction_binder=False,
                infer_Tc=True,
                infer_nu=True,
                infer_beta=False,
            ),
            id="binder-only",
        ),
        pytest.param(
            FssProfileConfig(
                use_m=True,
                use_m2=True,
                use_m4=True,
                correction_m=True,
                correction_m2=False,
                correction_m4=False,
                use_log_m=False,
                infer_Tc=True,
                infer_nu=True,
                infer_beta=True,
            ),
            id="multi-channel",
        ),
    ],
)
def test_fss_likelihood_matches_pymc_compile(config: FssProfileConfig) -> None:
    obs = _synthetic_observables()
    _, pymc_eval = compile_fss_exponent_log_likelihood(
        obs,
        use_numba=False,
        **_profile_build_kwargs(config),
    )
    jax_eval = compile_fss_log_marginal_likelihood_jax(obs, config)
    for tc, nu, beta in _exponent_triples(n_random=8):
        shared = fss_log_marginal_likelihood(
            obs, config, T_c=tc, nu=nu, beta=beta, backend="numpy"
        )
        pymc_ll = pymc_eval(tc, nu, beta)
        jax_ll = jax_eval(tc, nu, beta)
        assert shared == pytest.approx(pymc_ll, rel=1e-4, abs=0.02)
        assert jax_ll == pytest.approx(pymc_ll, rel=1e-4, abs=0.02)


def test_gp_jax_kernels_match_numpy() -> None:
    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(1)
    n = 10
    x = rng.uniform(-1.0, 1.0, size=n)
    y = rng.uniform(0.1, 1.0, size=n)
    sigma = rng.uniform(0.01, 0.05, size=n)
    L = rng.integers(16, 64, size=n).astype(np.float64)
    ls, amp = 0.5, 1.2

    np_ll = gp_np(x, y, sigma, length_scale=ls, amplitude=amp)
    jax_ll = float(
        gp_jax(
            jax.numpy.asarray(x),
            jax.numpy.asarray(y),
            jax.numpy.asarray(sigma),
            length_scale=ls,
            amplitude=amp,
        )
    )
    assert jax_ll == pytest.approx(np_ll, rel=1e-8, abs=1e-6)

    np_corr = corr_gp_np(
        x,
        y,
        sigma,
        L,
        omega=2.0,
        gp_ell=ls,
        gp_eta=amp,
        correction_gp_ell=0.3,
        correction_gp_eta=0.8,
    )
    jax_corr = float(
        corr_gp_jax(
            jax.numpy.asarray(x),
            jax.numpy.asarray(y),
            jax.numpy.asarray(sigma),
            jax.numpy.asarray(L),
            omega=2.0,
            gp_ell=ls,
            gp_eta=amp,
            correction_gp_ell=0.3,
            correction_gp_eta=0.8,
        )
    )
    assert jax_corr == pytest.approx(np_corr, rel=1e-8, abs=1e-6)


def test_gp_predictive_density_jax_matches_numpy() -> None:
    """Incremental likelihood imports these from gp_utils."""
    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(3)
    n = 8
    x_train = rng.uniform(-1.0, 1.0, size=n)
    y_train = rng.uniform(0.1, 1.0, size=n)
    sigma_train = rng.uniform(0.01, 0.05, size=n)
    L_train = rng.integers(16, 64, size=n).astype(np.float64)
    x_new = rng.uniform(-0.5, 0.5, size=3)
    y_new = rng.uniform(0.2, 0.8, size=3)
    sigma_new = rng.uniform(0.01, 0.04, size=3)
    L_new = rng.integers(16, 64, size=3).astype(np.float64)
    ls, amp = 0.5, 1.2

    np_pred = gp_pred_np(
        x_train,
        y_train,
        sigma_train,
        x_new,
        y_new,
        sigma_new=sigma_new,
        length_scale=ls,
        amplitude=amp,
    )
    jax_pred = np.asarray(
        gp_pred_jax(
            jax.numpy.asarray(x_train),
            jax.numpy.asarray(y_train),
            jax.numpy.asarray(sigma_train),
            jax.numpy.asarray(x_new),
            jax.numpy.asarray(y_new),
            sigma_new=jax.numpy.asarray(sigma_new),
            length_scale=ls,
            amplitude=amp,
        )
    )
    np.testing.assert_allclose(jax_pred, np_pred, rtol=1e-8, atol=1e-6)

    np_corr_pred = corr_pred_np(
        x_train,
        L_train,
        y_train,
        sigma_train,
        x_new,
        L_new,
        y_new,
        sigma_test=sigma_new,
        omega=2.0,
        gp_ell=ls,
        gp_eta=amp,
        correction_gp_ell=0.3,
        correction_gp_eta=0.8,
    )
    jax_corr_pred = np.asarray(
        corr_pred_jax(
            jax.numpy.asarray(x_train),
            jax.numpy.asarray(L_train),
            jax.numpy.asarray(y_train),
            jax.numpy.asarray(sigma_train),
            jax.numpy.asarray(x_new),
            jax.numpy.asarray(L_new),
            jax.numpy.asarray(y_new),
            sigma_test=jax.numpy.asarray(sigma_new),
            omega=2.0,
            gp_ell=ls,
            gp_eta=amp,
            correction_gp_ell=0.3,
            correction_gp_eta=0.8,
        )
    )
    np.testing.assert_allclose(jax_corr_pred, np_corr_pred, rtol=1e-8, atol=1e-6)


def test_incremental_likelihood_imports_gp_predictive_helpers() -> None:
    from ising import incremental_likelihood

    assert hasattr(incremental_likelihood, "fss_delta_log_likelihood")


def test_harada_bsa_jax_matches_pymc() -> None:
    """Harada BSA binder profile config: JAX vs PyMC on real published data."""
    config = default_profile_config()
    obs = read_harada_binder_observables()
    _, pymc_eval = compile_fss_exponent_log_likelihood(
        obs,
        use_numba=False,
        **_profile_build_kwargs(config),
    )
    jax_eval = compile_fss_log_marginal_likelihood_jax(obs, config)
    for tc, nu, beta in _exponent_triples(n_random=12):
        pymc_ll = pymc_eval(tc, nu, beta)
        jax_ll = jax_eval(tc, nu, beta)
        assert jax_ll == pytest.approx(pymc_ll, rel=1e-4, abs=0.02), (
            f"harada_bsa mismatch at ({tc}, {nu}, {beta}): "
            f"jax={jax_ll}, pymc={pymc_ll}"
        )


def test_additive_gp_uses_collapsed_amplitude_a_times_L_power() -> None:
    """Slide form m = L^{-β/ν} f + a g ⇒ Φ-likelihood uses a_collapsed = a L^{β/ν}."""
    from ising.discrepancy import (
        DISCREPANCY_L0_INIT,
        DISCREPANCY_P_INIT,
        DISCREPANCY_Q_INIT,
        DISCREPANCY_SIGMA_MODEL_INIT,
        DISCREPANCY_T0_INIT,
        discrepancy_amplitude,
    )
    from ising.gp_jax import (
        discrepancy_gp_log_marginal_likelihood as disc_gp_jax,
    )

    n = 8
    L = np.array([8.0, 8.0, 16.0, 16.0, 32.0, 32.0, 64.0, 64.0])
    T = np.linspace(2.20, 2.40, n)
    m = np.linspace(0.55, 0.12, n)
    sigma_m = np.full(n, 0.02)
    obs = make_observables_df(L=L, T=T, magnetization=m, sigma_m=sigma_m)
    config = FssProfileConfig(
        use_m=True,
        use_m2=False,
        use_m4=False,
        use_binder=False,
        use_chi=False,
        use_log_m=False,
        correction_m=False,
        discrepancy_m=True,
        discrepancy_form="additive_gp",
    )
    compiled = compile_fss_log_marginal_likelihood_jax(obs, config)
    scales = compiled.gp_scales
    tc, nu, beta = TC_EXACT, NU_EXACT, BETA_EXACT
    t0 = DISCREPANCY_T0_INIT
    L0 = DISCREPANCY_L0_INIT
    p = DISCREPANCY_P_INIT
    q = DISCREPANCY_Q_INIT
    sigma_model = DISCREPANCY_SIGMA_MODEL_INIT

    got = float(
        compiled.evaluate_with_gp_jax(
            tc,
            nu,
            beta,
            scales.base_gp_ell,
            scales.base_gp_eta,
            scales.correction_gp_ell,
            scales.correction_gp_eta,
            t0,
            L0,
            p,
            q,
            sigma_model,
            1.0,
            False,
        )
    )

    t = (T - tc) / tc
    z = t * L ** (1.0 / nu)
    phi = m * L ** (beta / nu)
    sigma_phi = sigma_m * L ** (beta / nu)
    a_raw = discrepancy_amplitude(t, L, t0=t0, L0=L0, p=p, q=q)
    a_collapsed = a_raw * L ** (beta / nu)
    jnp = jax.numpy
    expected = float(
        disc_gp_jax(
            jnp.asarray(z),
            jnp.asarray(phi),
            jnp.asarray(sigma_phi),
            jnp.asarray(a_collapsed),
            gp_ell=scales.base_gp_ell,
            gp_eta=scales.base_gp_eta,
            disc_gp_ell=scales.base_gp_ell,
            disc_gp_eta=sigma_model,
        )
    )
    unscaled = float(
        disc_gp_jax(
            jnp.asarray(z),
            jnp.asarray(phi),
            jnp.asarray(sigma_phi),
            jnp.asarray(a_raw),
            gp_ell=scales.base_gp_ell,
            gp_eta=scales.base_gp_eta,
            disc_gp_ell=scales.base_gp_ell,
            disc_gp_eta=sigma_model,
        )
    )
    assert got == pytest.approx(expected, rel=1e-6, abs=1e-5)
    assert abs(got - unscaled) > 0.1


def test_additive_gp_disc_gp_ell_changes_lml() -> None:
    """ℓ_g is independent of ℓ_f; omitting it keeps the legacy shared scale."""
    from ising.discrepancy import (
        DISCREPANCY_L0_INIT,
        DISCREPANCY_P_INIT,
        DISCREPANCY_Q_INIT,
        DISCREPANCY_SIGMA_MODEL_INIT,
        DISCREPANCY_T0_INIT,
    )

    n = 8
    L = np.array([8.0, 8.0, 16.0, 16.0, 32.0, 32.0, 64.0, 64.0])
    T = np.linspace(2.20, 2.40, n)
    m = np.linspace(0.55, 0.12, n)
    sigma_m = np.full(n, 0.02)
    obs = make_observables_df(L=L, T=T, magnetization=m, sigma_m=sigma_m)
    config = FssProfileConfig(
        use_m=True,
        use_m2=False,
        use_m4=False,
        use_binder=False,
        use_chi=False,
        use_log_m=False,
        correction_m=False,
        discrepancy_m=True,
        discrepancy_form="additive_gp",
    )
    compiled = compile_fss_log_marginal_likelihood_jax(obs, config)
    scales = compiled.gp_scales
    args = (
        TC_EXACT,
        NU_EXACT,
        BETA_EXACT,
        scales.base_gp_ell,
        scales.base_gp_eta,
        scales.correction_gp_ell,
        scales.correction_gp_eta,
        DISCREPANCY_T0_INIT,
        DISCREPANCY_L0_INIT,
        DISCREPANCY_P_INIT,
        DISCREPANCY_Q_INIT,
        DISCREPANCY_SIGMA_MODEL_INIT,
        1.0,
        False,
    )
    ll_default = float(compiled.evaluate_with_gp_jax(*args))
    ll_shared = float(compiled.evaluate_with_gp_jax(*args, scales.base_gp_ell))
    ll_other = float(compiled.evaluate_with_gp_jax(*args, 3.0 * scales.base_gp_ell))
    assert ll_default == pytest.approx(ll_shared, rel=1e-10, abs=1e-8)
    assert abs(ll_other - ll_shared) > 0.01


def test_additive_gp_z_threshold_uses_indicator_amplitude() -> None:
    """Collapsed a = 1_{|z|>threshold}; matches hand-built disc GP LML."""
    from ising.gp_jax import (
        discrepancy_gp_log_marginal_likelihood as disc_gp_jax,
    )
    from ising.profile_likelihood import Z_DISC_THRESHOLD

    n = 10
    L = np.array([8.0, 8.0, 16.0, 16.0, 32.0, 32.0, 64.0, 64.0, 128.0, 128.0])
    # Spread T so |z| straddles the threshold at exact exponents.
    T = np.array([2.20, 2.24, 2.26, 2.28, 2.30, 2.40, 2.60, 3.00, 4.00, 6.00])
    m = np.linspace(0.55, 0.05, n)
    sigma_m = np.full(n, 0.02)
    obs = make_observables_df(L=L, T=T, magnetization=m, sigma_m=sigma_m)
    config = FssProfileConfig(
        use_m=True,
        use_m2=False,
        use_m4=False,
        use_binder=False,
        use_chi=False,
        use_log_m=False,
        correction_m=False,
        discrepancy_m=True,
        discrepancy_form="additive_gp_z_threshold",
        z_disc_threshold=float(Z_DISC_THRESHOLD),
    )
    compiled = compile_fss_log_marginal_likelihood_jax(obs, config)
    scales = compiled.gp_scales
    tc, nu, beta = TC_EXACT, NU_EXACT, BETA_EXACT
    sigma_model = 1.5

    got = float(
        compiled.evaluate_with_gp_jax(
            tc,
            nu,
            beta,
            scales.base_gp_ell,
            scales.base_gp_eta,
            scales.correction_gp_ell,
            scales.correction_gp_eta,
            0.1,
            32.0,
            2.0,
            2.0,
            sigma_model,
            1.0,
            False,
        )
    )

    t = (T - tc) / tc
    z = t * L ** (1.0 / nu)
    phi = m * L ** (beta / nu)
    sigma_phi = sigma_m * L ** (beta / nu)
    a = (np.abs(z) > Z_DISC_THRESHOLD).astype(np.float64)
    assert a.sum() >= 1 and a.sum() < n
    jnp = jax.numpy
    expected = float(
        disc_gp_jax(
            jnp.asarray(z),
            jnp.asarray(phi),
            jnp.asarray(sigma_phi),
            jnp.asarray(a),
            gp_ell=scales.base_gp_ell,
            gp_eta=scales.base_gp_eta,
            disc_gp_ell=scales.base_gp_ell,
            disc_gp_eta=sigma_model,
        )
    )
    assert got == pytest.approx(expected, rel=1e-6, abs=1e-5)


def test_additive_gp_fss_uses_collapsed_omega_kappa_amplitude() -> None:
    """Collapsed a = L^{-ω} + κ |t|^{ω ν}; disc_q=ω, disc_t0=κ; no L^{β/ν}."""
    from ising.discrepancy import discrepancy_amplitude_fss
    from ising.gp_jax import (
        discrepancy_gp_log_marginal_likelihood as disc_gp_jax,
    )

    n = 8
    L = np.array([8.0, 8.0, 16.0, 16.0, 32.0, 32.0, 64.0, 64.0])
    T = np.linspace(2.20, 2.40, n)
    m = np.linspace(0.55, 0.12, n)
    sigma_m = np.full(n, 0.02)
    obs = make_observables_df(L=L, T=T, magnetization=m, sigma_m=sigma_m)
    config = FssProfileConfig(
        use_m=True,
        use_m2=False,
        use_m4=False,
        use_binder=False,
        use_chi=False,
        use_log_m=False,
        correction_m=False,
        discrepancy_m=True,
        discrepancy_form="additive_gp_fss",
    )
    compiled = compile_fss_log_marginal_likelihood_jax(obs, config)
    scales = compiled.gp_scales
    tc, nu, beta = TC_EXACT, NU_EXACT, BETA_EXACT
    omega = 2.0
    kappa = 1.5
    sigma_model = 1.5

    got = float(
        compiled.evaluate_with_gp_jax(
            tc,
            nu,
            beta,
            scales.base_gp_ell,
            scales.base_gp_eta,
            scales.correction_gp_ell,
            scales.correction_gp_eta,
            kappa,
            1.0,
            2.0,
            omega,
            sigma_model,
            1.0,
            False,
        )
    )

    t = (T - tc) / tc
    z = t * L ** (1.0 / nu)
    phi = m * L ** (beta / nu)
    sigma_phi = sigma_m * L ** (beta / nu)
    a = discrepancy_amplitude_fss(t, L, omega=omega, kappa=kappa, nu=nu)
    jnp = jax.numpy
    expected = float(
        disc_gp_jax(
            jnp.asarray(z),
            jnp.asarray(phi),
            jnp.asarray(sigma_phi),
            jnp.asarray(a),
            gp_ell=scales.base_gp_ell,
            gp_eta=scales.base_gp_eta,
            disc_gp_ell=scales.base_gp_ell,
            disc_gp_eta=sigma_model,
        )
    )
    wrongly_scaled = float(
        disc_gp_jax(
            jnp.asarray(z),
            jnp.asarray(phi),
            jnp.asarray(sigma_phi),
            jnp.asarray(a * L ** (beta / nu)),
            gp_ell=scales.base_gp_ell,
            gp_eta=scales.base_gp_eta,
            disc_gp_ell=scales.base_gp_ell,
            disc_gp_eta=sigma_model,
        )
    )
    assert got == pytest.approx(expected, rel=1e-6, abs=1e-5)
    assert got != pytest.approx(wrongly_scaled, rel=1e-3, abs=1e-3)


def test_discrepancy_f_posterior_predictive_matches_mean_and_nonneg_std():
    from ising.gp_jax import (
        discrepancy_f_posterior_mean,
        discrepancy_f_posterior_predictive,
    )

    z = np.linspace(-1.0, 1.0, 6)
    y = 0.5 - 0.1 * z
    sigma = np.full(6, 0.02)
    a = np.array([0.0, 0.0, 0.1, 0.2, 0.8, 1.0])
    z_new = np.linspace(-1.2, 1.2, 21)
    kwargs = dict(
        gp_ell=0.3,
        gp_eta=1.0,
        disc_gp_ell=0.3,
        disc_gp_eta=0.5,
    )
    mean, std = discrepancy_f_posterior_predictive(
        z,
        y,
        sigma,
        a,
        z_new,
        **kwargs,
    )
    mean_only = discrepancy_f_posterior_mean(
        z,
        y,
        sigma,
        a,
        z_new,
        **kwargs,
    )
    np.testing.assert_allclose(np.asarray(mean), np.asarray(mean_only), atol=1e-10)
    assert np.all(np.asarray(std) >= 0.0)


def test_discrepancy_coupling_posterior_pi_zero_matches_plain_gp():
    from ising.gp_jax import (
        discrepancy_f_posterior_predictive,
        gp_posterior_predictive,
    )

    z = np.linspace(-1.0, 1.0, 6)
    y = 0.5 - 0.1 * z
    sigma = np.full(6, 0.02)
    z_new = np.linspace(-1.2, 1.2, 11)
    zeros = np.zeros(6)
    kwargs = dict(
        gp_ell=0.3,
        gp_eta=1.0,
        disc_gp_ell=0.3,
        disc_gp_eta=0.8,
        jitter=1e-6,
    )
    plain_mean, plain_std = gp_posterior_predictive(
        z, y, sigma, z_new, length_scale=0.3, amplitude=1.0, jitter=1e-6
    )
    for coupling in ("mixture", "weighted"):
        mean, std = discrepancy_f_posterior_predictive(
            z, y, sigma, zeros, z_new, coupling=coupling, **kwargs
        )
        np.testing.assert_allclose(np.asarray(mean), np.asarray(plain_mean), atol=1e-8)
        np.testing.assert_allclose(np.asarray(std), np.asarray(plain_std), atol=1e-8)


def test_additive_gp_fss_mixture_uses_squashed_window_amplitude() -> None:
    """FSS mixture: π = a/(1+a) in collapsed space; no L^{β/ν}."""
    from ising.discrepancy import discrepancy_amplitude_fss, discrepancy_mix_weight
    from ising.gp_jax import (
        discrepancy_gp_log_marginal_likelihood as disc_gp_jax,
    )

    n = 8
    L = np.array([8.0, 8.0, 16.0, 16.0, 32.0, 32.0, 64.0, 64.0])
    T = np.linspace(2.20, 2.40, n)
    m = np.linspace(0.55, 0.12, n)
    sigma_m = np.full(n, 0.02)
    obs = make_observables_df(L=L, T=T, magnetization=m, sigma_m=sigma_m)
    config = FssProfileConfig(
        use_m=True,
        use_m2=False,
        use_m4=False,
        use_binder=False,
        use_chi=False,
        use_log_m=False,
        correction_m=False,
        discrepancy_m=True,
        discrepancy_form="additive_gp_fss",
        discrepancy_coupling="mixture",
    )
    compiled = compile_fss_log_marginal_likelihood_jax(obs, config)
    scales = compiled.gp_scales
    tc, nu, beta = TC_EXACT, NU_EXACT, BETA_EXACT
    omega = 2.0
    kappa = 1.5
    sigma_model = 1.5

    got = float(
        compiled.evaluate_with_gp_jax(
            tc,
            nu,
            beta,
            scales.base_gp_ell,
            scales.base_gp_eta,
            scales.correction_gp_ell,
            scales.correction_gp_eta,
            kappa,
            1.0,
            2.0,
            omega,
            sigma_model,
            1.0,
            False,
        )
    )

    t = (T - tc) / tc
    z = t * L ** (1.0 / nu)
    phi = m * L ** (beta / nu)
    sigma_phi = sigma_m * L ** (beta / nu)
    a = discrepancy_amplitude_fss(t, L, omega=omega, kappa=kappa, nu=nu)
    pi = discrepancy_mix_weight(a, squash=True)
    jnp = jax.numpy
    expected = float(
        disc_gp_jax(
            jnp.asarray(z),
            jnp.asarray(phi),
            jnp.asarray(sigma_phi),
            jnp.asarray(pi),
            gp_ell=scales.base_gp_ell,
            gp_eta=scales.base_gp_eta,
            disc_gp_ell=scales.base_gp_ell,
            disc_gp_eta=sigma_model,
            coupling="mixture",
        )
    )
    additive_expected = float(
        disc_gp_jax(
            jnp.asarray(z),
            jnp.asarray(phi),
            jnp.asarray(sigma_phi),
            jnp.asarray(a),
            gp_ell=scales.base_gp_ell,
            gp_eta=scales.base_gp_eta,
            disc_gp_ell=scales.base_gp_ell,
            disc_gp_eta=sigma_model,
            coupling="additive",
        )
    )
    assert got == pytest.approx(expected, rel=1e-6, abs=1e-5)
    assert abs(got - additive_expected) > 0.01


def test_additive_gp_fss_weighted_drops_second_gp() -> None:
    from ising.discrepancy import discrepancy_amplitude_fss, discrepancy_mix_weight
    from ising.gp_jax import (
        discrepancy_gp_log_marginal_likelihood as disc_gp_jax,
    )

    n = 8
    L = np.array([8.0, 8.0, 16.0, 16.0, 32.0, 32.0, 64.0, 64.0])
    T = np.linspace(2.20, 2.40, n)
    m = np.linspace(0.55, 0.12, n)
    sigma_m = np.full(n, 0.02)
    obs = make_observables_df(L=L, T=T, magnetization=m, sigma_m=sigma_m)
    config = FssProfileConfig(
        use_m=True,
        use_m2=False,
        use_m4=False,
        use_binder=False,
        use_chi=False,
        use_log_m=False,
        correction_m=False,
        discrepancy_m=True,
        discrepancy_form="additive_gp_fss",
        discrepancy_coupling="weighted",
    )
    compiled = compile_fss_log_marginal_likelihood_jax(obs, config)
    scales = compiled.gp_scales
    tc, nu, beta = TC_EXACT, NU_EXACT, BETA_EXACT
    omega = 2.0
    kappa = 1.5

    got = float(
        compiled.evaluate_with_gp_jax(
            tc,
            nu,
            beta,
            scales.base_gp_ell,
            scales.base_gp_eta,
            scales.correction_gp_ell,
            scales.correction_gp_eta,
            kappa,
            1.0,
            2.0,
            omega,
            9.0,
            1.0,
            False,
        )
    )

    t = (T - tc) / tc
    z = t * L ** (1.0 / nu)
    phi = m * L ** (beta / nu)
    sigma_phi = sigma_m * L ** (beta / nu)
    a = discrepancy_amplitude_fss(t, L, omega=omega, kappa=kappa, nu=nu)
    pi = discrepancy_mix_weight(a, squash=True)
    jnp = jax.numpy
    expected = float(
        disc_gp_jax(
            jnp.asarray(z),
            jnp.asarray(phi),
            jnp.asarray(sigma_phi),
            jnp.asarray(pi),
            gp_ell=scales.base_gp_ell,
            gp_eta=scales.base_gp_eta,
            disc_gp_ell=scales.base_gp_ell,
            disc_gp_eta=0.0,
            coupling="weighted",
        )
    )
    # σ_g is unused for weighted: changing it must not change LML.
    got_other_sigma = float(
        compiled.evaluate_with_gp_jax(
            tc,
            nu,
            beta,
            scales.base_gp_ell,
            scales.base_gp_eta,
            scales.correction_gp_ell,
            scales.correction_gp_eta,
            kappa,
            1.0,
            2.0,
            omega,
            0.01,
            1.0,
            False,
        )
    )
    assert got == pytest.approx(expected, rel=1e-6, abs=1e-5)
    assert got == pytest.approx(got_other_sigma, rel=1e-6, abs=1e-5)


def test_profile_joint_with_disc_respects_gp_hyperparams():
    from ising.fss_likelihood import profile_joint_with_disc

    obs = _synthetic_observables(n=10)
    config = FssProfileConfig(
        use_m=True,
        use_m2=True,
        use_m4=False,
        use_binder=False,
        use_chi=False,
        use_log_m=False,
        correction_m=False,
        correction_m2=False,
        discrepancy_m=True,
        discrepancy_m2=True,
        discrepancy_form="noise",
        gp_ell=0.15,
        gp_eta=1.0,
    )
    compiled = compile_fss_log_marginal_likelihood_jax(obs, config)
    tc_grid = np.array([TC_EXACT])
    nu_grid = np.array([NU_EXACT, NU_EXACT + 0.05])
    kwargs = dict(
        pair="T_c_nu",
        T_c=0.0,
        nu=0.0,
        beta=BETA_EXACT,
        t0=0.1,
        L0=32.0,
        p=2.0,
        q=2.0,
        sigma_model=0.0,
    )
    z_default = profile_joint_with_disc(compiled, tc_grid, nu_grid, **kwargs)
    z_wide = profile_joint_with_disc(
        compiled, tc_grid, nu_grid, gp_ell=1.5, gp_eta=1.0, **kwargs
    )
    z_eta = profile_joint_with_disc(
        compiled, tc_grid, nu_grid, gp_ell=0.15, gp_eta=0.2, **kwargs
    )
    assert z_default.shape == (2, 1)
    assert not np.allclose(z_default, z_wide)
    assert not np.allclose(z_default, z_eta)


def test_profile_joint_with_disc_gp_ell_grid_is_max_over_scalars():
    from ising.fss_likelihood import profile_joint_with_disc

    obs = _synthetic_observables(n=10)
    config = FssProfileConfig(
        use_m=True,
        use_m2=True,
        use_m4=False,
        use_binder=False,
        use_chi=False,
        use_log_m=False,
        correction_m=False,
        correction_m2=False,
        discrepancy_m=True,
        discrepancy_m2=True,
        discrepancy_form="additive_gp",
        gp_ell=0.15,
        gp_eta=1.0,
    )
    compiled = compile_fss_log_marginal_likelihood_jax(obs, config)
    tc_grid = np.array([TC_EXACT, TC_EXACT + 0.01])
    nu_grid = np.array([NU_EXACT, NU_EXACT + 0.05])
    kwargs = dict(
        pair="T_c_nu",
        T_c=0.0,
        nu=0.0,
        beta=BETA_EXACT,
        t0=0.1,
        L0=32.0,
        p=2.0,
        q=2.0,
        sigma_model=0.5,
        gp_eta=1.0,
    )
    ell_a, ell_b = 0.15, 1.5
    z_a = profile_joint_with_disc(compiled, tc_grid, nu_grid, gp_ell=ell_a, **kwargs)
    z_b = profile_joint_with_disc(compiled, tc_grid, nu_grid, gp_ell=ell_b, **kwargs)
    z_grid = profile_joint_with_disc(
        compiled, tc_grid, nu_grid, gp_ell_grid=np.array([ell_a, ell_b]), **kwargs
    )
    np.testing.assert_allclose(z_grid, np.maximum(z_a, z_b), rtol=1e-6, atol=1e-5)
    assert not np.allclose(z_a, z_b)


def test_profile_gp_ell_matches_scalar_joint_and_peaks_on_grid():
    from ising.fss_likelihood import profile_gp_ell, profile_joint_with_disc

    obs = _synthetic_observables(n=10)
    config = FssProfileConfig(
        use_m=True,
        use_m2=False,
        use_m4=False,
        use_binder=False,
        use_chi=False,
        use_log_m=False,
        correction_m=False,
        discrepancy_m=True,
        discrepancy_form="noise",
        gp_ell=0.3,
        gp_eta=1.0,
    )
    compiled = compile_fss_log_marginal_likelihood_jax(obs, config)
    ell_grid = np.array([0.08, 0.3, 1.2])
    kwargs = dict(
        t0=0.1,
        L0=32.0,
        p=2.0,
        q=2.0,
        sigma_model=0.0,
        gp_eta=1.0,
    )
    ll = profile_gp_ell(
        compiled,
        ell_grid,
        T_c=TC_EXACT,
        nu=NU_EXACT,
        beta=BETA_EXACT,
        **kwargs,
    )
    assert ll.shape == (3,)
    assert np.all(np.isfinite(ll))
    z_mid = profile_joint_with_disc(
        compiled,
        np.array([TC_EXACT]),
        np.array([NU_EXACT]),
        pair="T_c_nu",
        T_c=0.0,
        nu=0.0,
        beta=BETA_EXACT,
        gp_ell=float(ell_grid[1]),
        **kwargs,
    )
    assert float(ll[1]) == pytest.approx(float(z_mid[0, 0]), rel=1e-6, abs=1e-5)


def test_obs_sigma_scale_runtime_matches_config_scale() -> None:
    """Runtime s_σ × unscaled arrays matches compiling with obs_sigma_scale."""
    from ising.discrepancy import (
        DISCREPANCY_L0_INIT,
        DISCREPANCY_P_INIT,
        DISCREPANCY_Q_INIT,
        DISCREPANCY_SIGMA_MODEL_INIT,
        DISCREPANCY_T0_INIT,
    )

    obs = _synthetic_observables(n=10)
    common = {
        "use_m": True,
        "use_m2": True,
        "use_m4": False,
        "use_binder": False,
        "use_chi": False,
        "use_log_m": False,
        "correction_m": False,
        "correction_m2": False,
        "discrepancy_m": True,
        "discrepancy_m2": True,
        "discrepancy_form": "additive_gp",
        "gp_ell": 0.15,
        "gp_eta": 1.0,
    }
    compiled_unit = compile_fss_log_marginal_likelihood_jax(
        obs, FssProfileConfig(**common, obs_sigma_scale=1.0)
    )
    compiled_scaled = compile_fss_log_marginal_likelihood_jax(
        obs, FssProfileConfig(**common, obs_sigma_scale=3.0)
    )
    scales = compiled_unit.gp_scales
    args_common = (
        TC_EXACT,
        NU_EXACT,
        BETA_EXACT,
        scales.base_gp_ell,
        scales.base_gp_eta,
        scales.correction_gp_ell,
        scales.correction_gp_eta,
        DISCREPANCY_T0_INIT,
        DISCREPANCY_L0_INIT,
        DISCREPANCY_P_INIT,
        DISCREPANCY_Q_INIT,
        DISCREPANCY_SIGMA_MODEL_INIT,
    )
    ll_runtime = float(
        compiled_unit.evaluate_with_gp_jax(*args_common, 3.0, False)
    )
    ll_config = float(
        compiled_scaled.evaluate_with_gp_jax(*args_common, 1.0, False)
    )
    ll_unit = float(compiled_unit.evaluate_with_gp_jax(*args_common, 1.0, False))
    assert ll_runtime == pytest.approx(ll_config, rel=1e-6, abs=1e-5)
    assert abs(ll_runtime - ll_unit) > 0.01


def test_poly2_universal_kernel_jax_matches_numpy() -> None:
    from ising.profile_likelihood import FssProfileConfig

    obs = _synthetic_observables()
    config = FssProfileConfig(
        use_m=False,
        use_m2=False,
        use_m4=False,
        use_binder=True,
        use_chi=False,
        correction_binder=False,
        universal_kernel="poly2",
        gp_kernel="gaussian",
    )
    jax_eval = compile_fss_log_marginal_likelihood_jax(obs, config)
    for tc, nu, beta in _exponent_triples():
        np_ll = fss_log_marginal_likelihood(
            obs, config, T_c=tc, nu=nu, beta=beta, backend="numpy"
        )
        jax_ll = jax_eval(tc, nu, beta)
        assert jax_ll == pytest.approx(np_ll, rel=1e-5, abs=1e-4)


def test_poly2_universal_kernel_rejects_correction() -> None:
    from ising.profile_likelihood import FssProfileConfig

    obs = _synthetic_observables()
    config = FssProfileConfig(
        use_m=False,
        use_m2=False,
        use_m4=False,
        use_binder=True,
        use_chi=False,
        correction_binder=True,
        universal_kernel="poly2",
    )
    with pytest.raises(ValueError, match="polynomial universal"):
        compile_fss_log_marginal_likelihood_jax(obs, config)


def test_max_abs_z_window_reduces_harada_points() -> None:
    from ising.datasets import observables_path
    from ising.fss_likelihood import apply_max_abs_z_window, extract_likelihood_arrays
    from ising.observables import ensure_harada_bsa_observables_csv, read_observables_for_fss
    from ising.profile_likelihood import FssProfileConfig

    path = ensure_harada_bsa_observables_csv(observables_path("harada_bsa"))
    obs = read_observables_for_fss(path, use_binder=True)
    full = extract_likelihood_arrays(obs, FssProfileConfig(use_binder=True))
    trimmed = apply_max_abs_z_window(
        full,
        FssProfileConfig(use_binder=True, max_abs_z=2.0),
    )
    assert len(full.L) == 86
    assert len(trimmed.L) == 53
