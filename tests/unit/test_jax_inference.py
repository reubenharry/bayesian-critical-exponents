"""Smoke tests for JAX posterior sampling."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from ising.constants import (  # noqa: E402
    BETA_EXACT,
    NU_EXACT,
    TC_EXACT,
)
from ising.jax_inference import (  # noqa: E402
    compile_fss_log_posterior,
    sample_fss_posterior_jax,
    sample_fss_posterior_laps,
)
from ising.observables import (
    make_observables_df,  # noqa: E402
)
from ising.profile_likelihood import (
    FssProfileConfig,  # noqa: E402
)

from tests.unit.test_fss_likelihood import _synthetic_observables  # noqa: E402


def test_jax_log_posterior_finite_at_exact() -> None:
    obs = _synthetic_observables(n=8)
    config = FssProfileConfig(
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
    posterior = compile_fss_log_posterior(
        obs,
        config,
        infer_Tc=True,
        infer_nu=True,
        infer_beta=False,
    )
    u0 = posterior.initial_position(T_c=TC_EXACT, nu=NU_EXACT, beta=BETA_EXACT)
    logp = float(posterior.logdensity(u0))
    assert np.isfinite(logp)


def test_jax_log_posterior_gp_hyperparams_finite() -> None:
    obs = _synthetic_observables(n=8)
    config = FssProfileConfig(
        use_m=False,
        use_m2=False,
        use_m4=False,
        use_binder=True,
        use_chi=False,
        correction_binder=True,
        infer_Tc=True,
        infer_nu=True,
        infer_beta=False,
        gp_kernel="gaussian",
    )
    posterior = compile_fss_log_posterior(
        obs,
        config,
        infer_Tc=True,
        infer_nu=True,
        infer_beta=False,
        infer_gp_hyperparams=True,
    )
    assert posterior.layout.infer_gp_hyperparams is True
    assert "gp_ell" in {slot.name for slot in posterior.layout.slots}
    u0 = posterior.initial_position(T_c=TC_EXACT, nu=NU_EXACT, beta=BETA_EXACT)
    logp = float(posterior.logdensity(u0))
    assert np.isfinite(logp)
    grad = jax.grad(lambda u: posterior.logdensity(u))(u0)
    assert np.all(np.isfinite(np.asarray(grad)))


def test_jax_gp_hyperparams_match_fixed_likelihood_at_reference() -> None:
    obs = _synthetic_observables(n=8)
    config = FssProfileConfig(
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
    fixed_posterior = compile_fss_log_posterior(
        obs,
        config,
        infer_Tc=True,
        infer_nu=True,
        infer_beta=False,
        infer_gp_hyperparams=False,
    )
    gp_posterior = compile_fss_log_posterior(
        obs,
        config,
        infer_Tc=True,
        infer_nu=True,
        infer_beta=False,
        infer_gp_hyperparams=True,
    )
    layout = gp_posterior.layout
    log_like_fixed = float(
        fixed_posterior.likelihood.evaluate(TC_EXACT, NU_EXACT, BETA_EXACT)
    )
    log_like_gp = float(
        gp_posterior.likelihood.evaluate_with_gp_jax(
            TC_EXACT,
            NU_EXACT,
            BETA_EXACT,
            layout.fixed_gp_ell,
            layout.fixed_gp_eta,
            layout.fixed_correction_gp_ell,
            layout.fixed_correction_gp_eta,
            layout.fixed_disc_t0,
            layout.fixed_disc_L0,
            layout.fixed_disc_p,
            layout.fixed_disc_q,
            layout.fixed_disc_sigma_model,
            True,
        )
    )
    assert log_like_gp == pytest.approx(log_like_fixed, rel=1e-10, abs=1e-8)


@pytest.mark.slow
def test_jax_sampler_smoke() -> None:
    obs = make_observables_df(
        L=np.full(8, 32.0),
        T=np.linspace(2.26, 2.28, 8),
        magnetization=np.linspace(0.4, 0.2, 8),
        binder_cumulant=np.linspace(0.62, 0.58, 8),
        sigma_binder=np.full(8, 0.01),
    )
    config = FssProfileConfig(
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
    idata = sample_fss_posterior_jax(
        obs,
        config,
        infer_Tc=True,
        infer_nu=True,
        infer_beta=False,
        draws=20,
        tune=30,
        chains=2,
        seed=0,
        progress_bar=False,
    )
    assert "nu" in idata.posterior
    assert idata.posterior["nu"].shape == (2, 20)


@pytest.mark.slow
def test_laps_sampler_smoke() -> None:
    obs = make_observables_df(
        L=np.full(8, 32.0),
        T=np.linspace(2.26, 2.28, 8),
        magnetization=np.linspace(0.4, 0.2, 8),
        binder_cumulant=np.linspace(0.62, 0.58, 8),
        sigma_binder=np.full(8, 0.01),
    )
    config = FssProfileConfig(
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
    idata = sample_fss_posterior_laps(
        obs,
        config,
        infer_Tc=True,
        infer_nu=True,
        infer_beta=False,
        tune=30,
        draws=30,
        chains=4,
        seed=0,
        progress_bar=False,
    )
    assert idata.posterior.attrs["sampler_backend"] == "laps"
    assert "nu" in idata.posterior
    assert idata.posterior["nu"].shape == (4, 1)
