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
