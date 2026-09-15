"""FSS models: magnetization and optional Binder; infer T_c, nu, and/or beta."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytensor.tensor as pt
from pytensor import function
from pytensor.graph import graph_inputs

from ._deps import pm as _pm
from .constants import (
    BETA_EXACT,
    NU_EXACT,
    OMEGA_EXACT,
    OMEGA_PRIOR_LOWER,
    OMEGA_PRIOR_UPPER,
    SPATIAL_DIMENSION,
    TC_EXACT,
)
from .observables import (
    df_L_T,
    df_binder_sigmas,
    df_chi_sigmas,
    df_magnetization_sigmas,
    df_moment_sigmas,
    validate_observables_df,
)
from .gp_kernels import DEFAULT_GP_KERNEL, GpKernelKind, normalize_gp_kernel
from .discrepancy import (
    DISCREPANCY_L0_INIT,
    DISCREPANCY_L0_PRIOR_LOWER,
    DISCREPANCY_L0_PRIOR_UPPER,
    DISCREPANCY_P_INIT,
    DISCREPANCY_P_PRIOR_LOWER,
    DISCREPANCY_P_PRIOR_UPPER,
    DISCREPANCY_Q_INIT,
    DISCREPANCY_Q_PRIOR_LOWER,
    DISCREPANCY_Q_PRIOR_UPPER,
    DISCREPANCY_SIGMA_MODEL_INIT,
    DISCREPANCY_SIGMA_MODEL_PRIOR_SIGMA,
    DISCREPANCY_T0_INIT,
    DISCREPANCY_T0_PRIOR_LOWER,
    DISCREPANCY_T0_PRIOR_UPPER,
)
from .model_scaling_nu import (
    CORRECTION_BASE_GP_ETA_FIXED,
    CORRECTION_GP_ELL_FACTOR,
    CORRECTION_GP_ETA_FIXED,
    GP_ELL_STIFF_FACTOR,
    TC_PRIOR_LOWER,
    TC_PRIOR_UPPER,
    _add_correction_gp_likelihood,
    _add_plain_gp_likelihood,
)
from .scaling_function import (
    ScalingFunctionConfig,
    add_gp_scaling_likelihood,
    median_nearest_neighbor_spacing,
    provisional_z,
)

NU_PRIOR_MU = NU_EXACT
NU_PRIOR_SIGMA = 0.15
NU_PRIOR_LOWER = 0.5
NU_PRIOR_UPPER = 1.5

BETA_PRIOR_MU = BETA_EXACT
BETA_PRIOR_SIGMA = 0.04
BETA_PRIOR_LOWER = 0.05
BETA_PRIOR_UPPER = 0.25

# T_c = tc_reparam_ref + T_c_delta / tc_delta_scale (delta in "milli-Kelvin" units).
TC_DELTA_SCALE = 1000.0


def _effective_sigma_pt(
    sigma: pt.TensorVariable,
    t: pt.TensorVariable,
    L: pt.TensorVariable,
    *,
    t0,
    L0,
    p,
    q,
    sigma_model,
) -> pt.TensorVariable:
    """sigma_eff = sqrt(sigma_MC^2 + (sigma_model * a(t,L))^2)."""
    a = (pt.abs(t) / t0) ** p + (L0 / L) ** q
    return pt.sqrt(sigma**2 + (sigma_model * a) ** 2)


def _maybe_inflate_sigma(
    sigma: pt.TensorVariable,
    *,
    inflate: bool,
    t: pt.TensorVariable,
    L: pt.TensorVariable,
    t0,
    L0,
    p,
    q,
    sigma_model,
) -> pt.TensorVariable:
    if not inflate:
        return sigma
    return _effective_sigma_pt(
        sigma, t, L, t0=t0, L0=L0, p=p, q=q, sigma_model=sigma_model
    )


def _add_linear_moment_likelihood(
    *,
    z: pt.TensorVariable,
    L_obs: pt.TensorVariable,
    phi: pt.TensorVariable,
    sigma_phi: pt.TensorVariable,
    correction: bool,
    base_gp_ell,
    base_gp_eta,
    correction_gp_ell,
    correction_gp_eta,
    omega,
    name_prefix: str,
    kernel: GpKernelKind | str = DEFAULT_GP_KERNEL,
) -> None:
    if correction:
        _add_correction_gp_likelihood(
            z,
            phi,
            sigma_phi,
            L_obs,
            omega=omega,
            gp_ell=base_gp_ell,
            gp_eta=base_gp_eta,
            correction_gp_ell=correction_gp_ell,
            correction_gp_eta=correction_gp_eta,
            kernel=kernel,
            name=f"correction_gp_like_{name_prefix}",
        )
    else:
        _add_plain_gp_likelihood(
            z,
            phi,
            sigma_phi,
            gp_ell=base_gp_ell,
            gp_eta=base_gp_eta,
            kernel=kernel,
            name=f"plain_gp_like_{name_prefix}",
        )


def build_fss_model(
    observables: pd.DataFrame,
    *,
    use_m: bool = True,
    use_m2: bool = False,
    use_m4: bool = False,
    use_binder: bool = False,
    use_chi: bool = False,
    scaling_config: ScalingFunctionConfig | None = None,
    correction_m: bool = True,
    correction_m2: bool = True,
    correction_m4: bool = True,
    correction_binder: bool = True,
    correction_chi: bool = True,
    discrepancy_m: bool = False,
    discrepancy_m2: bool = False,
    discrepancy_m4: bool = False,
    discrepancy_binder: bool = False,
    discrepancy_chi: bool = False,
    omega_fixed: float = OMEGA_EXACT,
    infer_omega: bool = False,
    infer_gp_hyperparams: bool = False,
    omega_prior_lower: float = OMEGA_PRIOR_LOWER,
    omega_prior_upper: float = OMEGA_PRIOR_UPPER,
    gp_ell_prior_sigma: float = 1.0,
    infer_Tc: bool = False,
    infer_nu: bool = True,
    infer_beta: bool = False,
    tc_prior_lower: float = TC_PRIOR_LOWER,
    tc_prior_upper: float = TC_PRIOR_UPPER,
    nu_prior_mu: float = NU_PRIOR_MU,
    nu_prior_sigma: float = NU_PRIOR_SIGMA,
    nu_prior_lower: float = NU_PRIOR_LOWER,
    nu_prior_upper: float = NU_PRIOR_UPPER,
    beta_prior_mu: float = BETA_PRIOR_MU,
    beta_prior_sigma: float = BETA_PRIOR_SIGMA,
    beta_prior_lower: float = BETA_PRIOR_LOWER,
    beta_prior_upper: float = BETA_PRIOR_UPPER,
    gp_ell_factor: float = GP_ELL_STIFF_FACTOR,
    gp_eta: float = CORRECTION_BASE_GP_ETA_FIXED,
    correction_gp_ell_factor: float = CORRECTION_GP_ELL_FACTOR,
    correction_gp_eta: float = CORRECTION_GP_ETA_FIXED,
    gp_kernel: GpKernelKind | str = DEFAULT_GP_KERNEL,
    obs_sigma_scale: float = 1.0,
    use_log_m: bool = True,
    reparametrize_Tc: bool = False,
    tc_reparam_ref: float | None = None,
    tc_delta_scale: float = TC_DELTA_SCALE,
    use_exponent_data: bool = False,
) -> _pm.Model:
    """Marginal GP FSS model with optional inference of T_c, nu, and beta.

    Magnetization (when ``use_m=True``):
        Phi_m = m L^(beta / nu), or log Phi_m = log m + (beta / nu) log L when
        ``use_log_m=True`` (default).

    Second moment (when ``use_m2=True``):
        Phi_m2 = <m^2> L^(2 beta / nu)

    Fourth moment (when ``use_m4=True``):
        Phi_m4 = <m^4> L^(4 beta / nu)

    Binder cumulant (when ``use_binder=True``):
        U_4(L, T) = Phi_U(z)

    Susceptibility (when ``use_chi=True``):
        Phi_chi(z) = chi L^(-2 + 2 beta / nu) = chi L^(-gamma / nu)
        with chi = (beta / N) Var(M) = beta L^d Var(m) per site.
        Uses the per-site mean susceptibility, not log-batch means, which are
        biased by Jensen's inequality.

    with z = t L^(1/nu), t = (T - T_c) / T_c.

    When ``correction_*`` is True for a channel, it uses
    Phi(z, L) = f0(z) + L^{-omega} f1(z) with fixed GP hyperparameters.
    When False, that channel uses a plain f0(z) GP only.

    When ``discrepancy_*`` is True for a channel, observation noise is inflated by
    sigma_model^2 * a(t, L)^2 with a = (|t|/t0)^p + (L0/L)^q and (t0, L0, p, q,
    sigma_model) inferred (shared across discrepancy channels).
    """
    if not (use_m or use_m2 or use_m4 or use_binder or use_chi):
        raise ValueError(
            "At least one of use_m, use_m2, use_m4, use_binder, or use_chi must be True"
        )
    if obs_sigma_scale <= 0.0:
        raise ValueError(f"obs_sigma_scale must be positive, got {obs_sigma_scale}")
    gp_kernel = normalize_gp_kernel(gp_kernel)
    if use_exponent_data and (infer_Tc or infer_nu or infer_beta):
        raise ValueError(
            "use_exponent_data=True requires infer_Tc=infer_nu=infer_beta=False "
            "(profile likelihood with pm.Data exponents)."
        )

    validate_observables_df(
        observables,
        use_m=use_m,
        use_m2=use_m2,
        use_m4=use_m4,
        use_binder=use_binder,
        use_chi=use_chi,
    )
    L, T = df_L_T(observables)
    n_obs = L.size

    log_m: np.ndarray | None = None
    sigma_log: np.ndarray | None = None
    m_obs: np.ndarray | None = None
    sigma_m_obs: np.ndarray | None = None
    m2_obs: np.ndarray | None = None
    sigma_m2_obs: np.ndarray | None = None
    m4_obs: np.ndarray | None = None
    sigma_m4_obs: np.ndarray | None = None
    if use_m:
        magnetization, sigma_m, log_m, sigma_log = df_magnetization_sigmas(observables)
        if use_log_m:
            pass
        else:
            m_obs = magnetization
            sigma_m_obs = sigma_m
    if use_m2:
        m2_obs, sigma_m2_obs = df_moment_sigmas(
            observables, value_col="m2", std_col="m2_std", n_eff_col="n_eff_m2"
        )
        if not (m2_obs.size == n_obs == sigma_m2_obs.size):
            raise ValueError("L, T, m2, and sigma_m2 must have the same length")
    if use_m4:
        m4_obs, sigma_m4_obs = df_moment_sigmas(
            observables, value_col="m4", std_col="m4_std", n_eff_col="n_eff_m4"
        )
        if not (m4_obs.size == n_obs == sigma_m4_obs.size):
            raise ValueError("L, T, m4, and sigma_m4 must have the same length")
    if use_binder:
        binder, sigma_u = df_binder_sigmas(observables)
        if not (binder.size == n_obs == sigma_u.size):
            raise ValueError(
                "L, T, binder_cumulant, and sigma_binder must have the same length"
            )
    if use_chi:
        chi, sigma_chi_obs = df_chi_sigmas(observables)
        if not (chi.size == n_obs == sigma_chi_obs.size):
            raise ValueError(
                "L, T, susceptibility, and sigma_chi must have the same length"
            )

    if obs_sigma_scale != 1.0:
        if sigma_log is not None:
            sigma_log = sigma_log * obs_sigma_scale
        if sigma_m_obs is not None:
            sigma_m_obs = sigma_m_obs * obs_sigma_scale
        if sigma_m2_obs is not None:
            sigma_m2_obs = sigma_m2_obs * obs_sigma_scale
        if sigma_m4_obs is not None:
            sigma_m4_obs = sigma_m4_obs * obs_sigma_scale
        if use_binder:
            sigma_u = sigma_u * obs_sigma_scale
        if use_chi:
            sigma_chi_obs = sigma_chi_obs * obs_sigma_scale

    z_ref = provisional_z(L, T, T_c=TC_EXACT, nu=NU_EXACT)
    z_span = float(z_ref.max() - z_ref.min())
    z_scale = max(median_nearest_neighbor_spacing(z_ref), 0.2 * z_span)
    base_gp_ell = gp_ell_factor * z_span
    base_gp_eta = gp_eta
    correction_gp_ell = correction_gp_ell_factor * z_span
    correction_gp_eta = correction_gp_eta

    uses_correction = (
        (use_m and correction_m)
        or (use_m2 and correction_m2)
        or (use_m4 and correction_m4)
        or (use_binder and correction_binder)
        or (use_chi and correction_chi)
    )
    uses_discrepancy = (
        (use_m and discrepancy_m)
        or (use_m2 and discrepancy_m2)
        or (use_m4 and discrepancy_m4)
        or (use_binder and discrepancy_binder)
        or (use_chi and discrepancy_chi)
    )

    base_config = scaling_config or ScalingFunctionConfig()
    if base_config.backend != "gp":
        raise ValueError("fss models currently support scaling='gp' only")

    coords = {"obs": np.arange(n_obs)}
    with _pm.Model(coords=coords) as model:
        L_obs = _pm.Data("L", L, dims="obs")
        T_obs = _pm.Data("T", T, dims="obs")
        if use_m:
            if use_log_m:
                log_m_obs = _pm.Data("log_m", log_m, dims="obs")
                sigma_log_obs = _pm.Data("sigma_log", sigma_log, dims="obs")
            else:
                m_obs_pt = _pm.Data("m", m_obs, dims="obs")
                sigma_m_obs_pt = _pm.Data("sigma_m", sigma_m_obs, dims="obs")
        if use_m2:
            m2_obs_pt = _pm.Data("m2", m2_obs, dims="obs")
            sigma_m2_obs_pt = _pm.Data("sigma_m2", sigma_m2_obs, dims="obs")
        if use_m4:
            m4_obs_pt = _pm.Data("m4", m4_obs, dims="obs")
            sigma_m4_obs_pt = _pm.Data("sigma_m4", sigma_m4_obs, dims="obs")
        if use_binder:
            binder_obs = _pm.Data("binder_cumulant", binder, dims="obs")
            sigma_binder_obs = _pm.Data("sigma_binder", sigma_u, dims="obs")
        if use_chi:
            chi_obs = _pm.Data("chi", chi, dims="obs")
            sigma_chi_obs = _pm.Data("sigma_chi", sigma_chi_obs, dims="obs")

        if use_exponent_data:
            T_c = _pm.Data("T_c_exponent", np.float64(TC_EXACT))
            nu = _pm.Data("nu_exponent", np.float64(NU_EXACT))
            beta = _pm.Data("beta_exponent", np.float64(BETA_EXACT))
        else:
            if infer_nu:
                nu = _pm.Uniform(
                    "nu",
                    lower=nu_prior_lower,
                    upper=nu_prior_upper,
                )
            else:
                nu = NU_EXACT

            if infer_beta:
                beta = _pm.Uniform(
                    "beta",
                    lower=beta_prior_lower,
                    upper=beta_prior_upper,
                )
            else:
                beta = BETA_EXACT

            tc_reparam_ref_val = TC_EXACT if tc_reparam_ref is None else float(tc_reparam_ref)
            if infer_Tc:
                if reparametrize_Tc:
                    if tc_delta_scale <= 0.0:
                        raise ValueError(
                            f"tc_delta_scale must be positive, got {tc_delta_scale}"
                        )
                    delta_lo = (tc_prior_lower - tc_reparam_ref_val) * tc_delta_scale
                    delta_hi = (tc_prior_upper - tc_reparam_ref_val) * tc_delta_scale
                    T_c_delta = _pm.Uniform(
                        "T_c_delta",
                        lower=delta_lo,
                        upper=delta_hi,
                    )
                    T_c = _pm.Deterministic(
                        "T_c",
                        tc_reparam_ref_val + T_c_delta / tc_delta_scale,
                    )
                else:
                    T_c = _pm.Uniform(
                        "T_c",
                        lower=tc_prior_lower,
                        upper=tc_prior_upper,
                    )
            else:
                T_c = TC_EXACT

        nu_pt = pt.as_tensor_variable(nu)
        beta_pt = pt.as_tensor_variable(beta)
        gamma = _pm.Deterministic(
            "gamma",
            SPATIAL_DIMENSION * nu_pt - 2.0 * beta_pt,
        )

        if infer_gp_hyperparams:
            base_gp_ell_pt = _pm.Lognormal(
                "gp_ell",
                mu=math.log(base_gp_ell),
                sigma=gp_ell_prior_sigma,
            )
            base_gp_eta_pt = _pm.HalfNormal(
                "gp_eta",
                sigma=max(base_gp_eta, 1.0),
            )
            if uses_correction:
                correction_gp_ell_pt = _pm.Lognormal(
                    "correction_gp_ell",
                    mu=math.log(correction_gp_ell),
                    sigma=gp_ell_prior_sigma,
                )
                correction_gp_eta_pt = _pm.HalfNormal(
                    "correction_gp_eta",
                    sigma=max(correction_gp_eta, 1.0),
                )
            else:
                correction_gp_ell_pt = correction_gp_ell
                correction_gp_eta_pt = correction_gp_eta
        else:
            base_gp_ell_pt = base_gp_ell
            base_gp_eta_pt = base_gp_eta
            correction_gp_ell_pt = correction_gp_ell
            correction_gp_eta_pt = correction_gp_eta

        if uses_correction and infer_omega:
            omega_pt = _pm.Uniform(
                "omega",
                lower=omega_prior_lower,
                upper=omega_prior_upper,
            )
        else:
            omega_pt = omega_fixed

        if uses_discrepancy:
            disc_t0 = _pm.Uniform(
                "disc_t0",
                lower=DISCREPANCY_T0_PRIOR_LOWER,
                upper=DISCREPANCY_T0_PRIOR_UPPER,
                initval=DISCREPANCY_T0_INIT,
            )
            disc_L0 = _pm.Uniform(
                "disc_L0",
                lower=DISCREPANCY_L0_PRIOR_LOWER,
                upper=DISCREPANCY_L0_PRIOR_UPPER,
                initval=DISCREPANCY_L0_INIT,
            )
            disc_p = _pm.Uniform(
                "disc_p",
                lower=DISCREPANCY_P_PRIOR_LOWER,
                upper=DISCREPANCY_P_PRIOR_UPPER,
                initval=DISCREPANCY_P_INIT,
            )
            disc_q = _pm.Uniform(
                "disc_q",
                lower=DISCREPANCY_Q_PRIOR_LOWER,
                upper=DISCREPANCY_Q_PRIOR_UPPER,
                initval=DISCREPANCY_Q_INIT,
            )
            disc_sigma_model = _pm.HalfNormal(
                "disc_sigma_model",
                sigma=DISCREPANCY_SIGMA_MODEL_PRIOR_SIGMA,
                initval=DISCREPANCY_SIGMA_MODEL_INIT,
            )
        else:
            disc_t0 = disc_L0 = disc_p = disc_q = disc_sigma_model = None

        t = (T_obs - T_c) / T_c
        z = _pm.Deterministic("z", t * L_obs ** (1.0 / nu_pt), dims="obs")
        if infer_nu and use_chi and not (use_m or use_m2 or use_m4 or use_binder):
            z_span_pt = pt.max(z) - pt.min(z)
        else:
            z_span_pt = z_span
        if use_m:
            if use_log_m:
                log_phi_m = _pm.Deterministic(
                    "log_phi_m",
                    log_m_obs + (beta_pt / nu_pt) * pt.log(L_obs),
                    dims="obs",
                )
                phi_m = _pm.Deterministic(
                    "phi_m",
                    pt.exp(log_phi_m),
                    dims="obs",
                )
                sigma_phi_m = _pm.Deterministic(
                    "sigma_phi_m",
                    phi_m * sigma_log_obs,
                    dims="obs",
                )
            else:
                phi_m = _pm.Deterministic(
                    "phi_m",
                    m_obs_pt * L_obs ** (beta_pt / nu_pt),
                    dims="obs",
                )
                log_phi_m = _pm.Deterministic(
                    "log_phi_m",
                    pt.log(pt.maximum(phi_m, 1e-12)),
                    dims="obs",
                )
                sigma_phi_m = _pm.Deterministic(
                    "sigma_phi_m",
                    sigma_m_obs_pt * L_obs ** (beta_pt / nu_pt),
                    dims="obs",
                )
        if use_m2:
            phi_m2 = _pm.Deterministic(
                "phi_m2",
                m2_obs_pt * L_obs ** (2.0 * beta_pt / nu_pt),
                dims="obs",
            )
            log_phi_m2 = _pm.Deterministic(
                "log_phi_m2",
                pt.log(pt.maximum(phi_m2, 1e-12)),
                dims="obs",
            )
            sigma_phi_m2 = _pm.Deterministic(
                "sigma_phi_m2",
                sigma_m2_obs_pt * L_obs ** (2.0 * beta_pt / nu_pt),
                dims="obs",
            )
        if use_m4:
            phi_m4 = _pm.Deterministic(
                "phi_m4",
                m4_obs_pt * L_obs ** (4.0 * beta_pt / nu_pt),
                dims="obs",
            )
            log_phi_m4 = _pm.Deterministic(
                "log_phi_m4",
                pt.log(pt.maximum(phi_m4, 1e-12)),
                dims="obs",
            )
            sigma_phi_m4 = _pm.Deterministic(
                "sigma_phi_m4",
                sigma_m4_obs_pt * L_obs ** (4.0 * beta_pt / nu_pt),
                dims="obs",
            )
        if use_chi:
            chi_scale = pt.exp(
                (-gamma / nu_pt) * pt.log(L_obs),
            )
            phi_chi = _pm.Deterministic(
                "phi_chi",
                chi_obs * chi_scale,
                dims="obs",
            )
            sigma_phi_chi = _pm.Deterministic(
                "sigma_phi_chi",
                sigma_chi_obs * chi_scale,
                dims="obs",
            )
            log_phi_chi = _pm.Deterministic(
                "log_phi_chi",
                pt.log(pt.maximum(phi_chi, 1e-12)),
                dims="obs",
            )

        inflate_kw = dict(
            t=t,
            L=L_obs,
            t0=disc_t0,
            L0=disc_L0,
            p=disc_p,
            q=disc_q,
            sigma_model=disc_sigma_model,
        )

        if use_m:
            if use_log_m and not correction_m:
                sigma_m_like = _maybe_inflate_sigma(
                    sigma_log_obs, inflate=discrepancy_m, **inflate_kw
                )
                add_gp_scaling_likelihood(
                    z,
                    log_phi_m,
                    sigma_m_like,
                    gp_ell=base_gp_ell_pt,
                    gp_eta=base_gp_eta_pt,
                    kernel=gp_kernel,
                    name="gp_like_m",
                )
            elif correction_m:
                sigma_m_like = _maybe_inflate_sigma(
                    sigma_phi_m, inflate=discrepancy_m, **inflate_kw
                )
                _add_correction_gp_likelihood(
                    z,
                    phi_m,
                    sigma_m_like,
                    L_obs,
                    omega=omega_pt,
                    gp_ell=base_gp_ell_pt,
                    gp_eta=base_gp_eta_pt,
                    correction_gp_ell=correction_gp_ell_pt,
                    correction_gp_eta=correction_gp_eta_pt,
                    kernel=gp_kernel,
                    name="correction_gp_like_m",
                )
            else:
                sigma_m_like = _maybe_inflate_sigma(
                    sigma_phi_m, inflate=discrepancy_m, **inflate_kw
                )
                _add_plain_gp_likelihood(
                    z,
                    phi_m,
                    sigma_m_like,
                    gp_ell=base_gp_ell_pt,
                    gp_eta=base_gp_eta_pt,
                    kernel=gp_kernel,
                    name="plain_gp_like_m",
                )
        if use_m2:
            sigma_m2_like = _maybe_inflate_sigma(
                sigma_phi_m2, inflate=discrepancy_m2, **inflate_kw
            )
            _add_linear_moment_likelihood(
                z=z,
                L_obs=L_obs,
                phi=phi_m2,
                sigma_phi=sigma_m2_like,
                correction=correction_m2,
                base_gp_ell=base_gp_ell_pt,
                base_gp_eta=base_gp_eta_pt,
                correction_gp_ell=correction_gp_ell_pt,
                correction_gp_eta=correction_gp_eta_pt,
                omega=omega_pt,
                name_prefix="m2",
                kernel=gp_kernel,
            )
        if use_m4:
            sigma_m4_like = _maybe_inflate_sigma(
                sigma_phi_m4, inflate=discrepancy_m4, **inflate_kw
            )
            _add_linear_moment_likelihood(
                z=z,
                L_obs=L_obs,
                phi=phi_m4,
                sigma_phi=sigma_m4_like,
                correction=correction_m4,
                base_gp_ell=base_gp_ell_pt,
                base_gp_eta=base_gp_eta_pt,
                correction_gp_ell=correction_gp_ell_pt,
                correction_gp_eta=correction_gp_eta_pt,
                omega=omega_pt,
                name_prefix="m4",
                kernel=gp_kernel,
            )
        if use_binder:
            sigma_u_like = _maybe_inflate_sigma(
                sigma_binder_obs, inflate=discrepancy_binder, **inflate_kw
            )
            if correction_binder:
                _add_correction_gp_likelihood(
                    z,
                    binder_obs,
                    sigma_u_like,
                    L_obs,
                    omega=omega_pt,
                    gp_ell=base_gp_ell_pt,
                    gp_eta=base_gp_eta_pt,
                    correction_gp_ell=correction_gp_ell_pt,
                    correction_gp_eta=correction_gp_eta_pt,
                    kernel=gp_kernel,
                    name="correction_gp_like_u",
                )
            else:
                add_gp_scaling_likelihood(
                    z,
                    binder_obs,
                    sigma_u_like,
                    gp_ell=base_gp_ell_pt,
                    gp_eta=base_gp_eta_pt,
                    kernel=gp_kernel,
                    name="gp_like_u",
                )
        if use_chi:
            if infer_gp_hyperparams:
                chi_gp_ell = base_gp_ell_pt
                chi_correction_gp_ell = correction_gp_ell_pt
            else:
                chi_gp_ell = gp_ell_factor * z_span_pt
                chi_correction_gp_ell = correction_gp_ell_factor * z_span_pt
            if correction_chi:
                sigma_chi_like = _maybe_inflate_sigma(
                    sigma_phi_chi, inflate=discrepancy_chi, **inflate_kw
                )
                _add_correction_gp_likelihood(
                    z,
                    phi_chi,
                    sigma_chi_like,
                    L_obs,
                    omega=omega_pt,
                    gp_ell=chi_gp_ell,
                    gp_eta=base_gp_eta_pt,
                    correction_gp_ell=chi_correction_gp_ell,
                    correction_gp_eta=correction_gp_eta_pt,
                    kernel=gp_kernel,
                    name="correction_gp_like_chi",
                )
            else:
                sigma_chi_log = sigma_phi_chi / pt.maximum(phi_chi, 1e-12)
                sigma_chi_like = _maybe_inflate_sigma(
                    sigma_chi_log, inflate=discrepancy_chi, **inflate_kw
                )
                add_gp_scaling_likelihood(
                    z,
                    log_phi_chi,
                    sigma_chi_like,
                    gp_ell=chi_gp_ell,
                    gp_eta=base_gp_eta_pt,
                    kernel=gp_kernel,
                    name="gp_like_chi",
                )

        model._ising_model = "fss"  # type: ignore[attr-defined]
        model._ising_include_m = use_m  # type: ignore[attr-defined]
        model._ising_include_m2 = use_m2  # type: ignore[attr-defined]
        model._ising_include_m4 = use_m4  # type: ignore[attr-defined]
        model._ising_include_binder = use_binder  # type: ignore[attr-defined]
        model._ising_include_chi = use_chi  # type: ignore[attr-defined]
        model._ising_z_scale = z_scale  # type: ignore[attr-defined]
        model._ising_fss_correction = uses_correction  # type: ignore[attr-defined]
        model._ising_fss_discrepancy = uses_discrepancy  # type: ignore[attr-defined]
        if use_m:
            model._ising_fss_correction_m = correction_m  # type: ignore[attr-defined]
            model._ising_fss_discrepancy_m = discrepancy_m  # type: ignore[attr-defined]
        if use_m2:
            model._ising_fss_correction_m2 = correction_m2  # type: ignore[attr-defined]
            model._ising_fss_discrepancy_m2 = discrepancy_m2  # type: ignore[attr-defined]
        if use_m4:
            model._ising_fss_correction_m4 = correction_m4  # type: ignore[attr-defined]
            model._ising_fss_discrepancy_m4 = discrepancy_m4  # type: ignore[attr-defined]
        if use_binder:
            model._ising_fss_correction_binder = correction_binder  # type: ignore[attr-defined]
            model._ising_fss_discrepancy_binder = discrepancy_binder  # type: ignore[attr-defined]
        if use_chi:
            model._ising_fss_correction_chi = correction_chi  # type: ignore[attr-defined]
            model._ising_fss_discrepancy_chi = discrepancy_chi  # type: ignore[attr-defined]
        model._ising_infer_Tc = infer_Tc  # type: ignore[attr-defined]
        model._ising_infer_nu = infer_nu  # type: ignore[attr-defined]
        model._ising_infer_beta = infer_beta  # type: ignore[attr-defined]
        model._ising_infer_omega = infer_omega and uses_correction  # type: ignore[attr-defined]
        model._ising_infer_gp_hyperparams = infer_gp_hyperparams  # type: ignore[attr-defined]
        if use_m:
            model._ising_use_log_m = use_log_m  # type: ignore[attr-defined]
        if infer_Tc:
            model._ising_tc_prior_lower = tc_prior_lower  # type: ignore[attr-defined]
            model._ising_tc_prior_upper = tc_prior_upper  # type: ignore[attr-defined]
            model._ising_reparametrize_Tc = reparametrize_Tc  # type: ignore[attr-defined]
            if reparametrize_Tc:
                model._ising_tc_reparam_ref = tc_reparam_ref_val  # type: ignore[attr-defined]
                model._ising_tc_delta_scale = tc_delta_scale  # type: ignore[attr-defined]
        model._ising_gp_ell_factor = gp_ell_factor  # type: ignore[attr-defined]
        model._ising_gp_ell_fixed = base_gp_ell  # type: ignore[attr-defined]
        model._ising_gp_eta_fixed = base_gp_eta  # type: ignore[attr-defined]
        model._ising_gp_kernel = gp_kernel  # type: ignore[attr-defined]
        model._ising_correction_gp_ell_factor = correction_gp_ell_factor  # type: ignore[attr-defined]
        model._ising_obs_sigma_scale = obs_sigma_scale  # type: ignore[attr-defined]
        model._ising_nu_prior_mu = nu_prior_mu  # type: ignore[attr-defined]
        model._ising_nu_prior_sigma = nu_prior_sigma  # type: ignore[attr-defined]
        model._ising_nu_prior_lower = nu_prior_lower  # type: ignore[attr-defined]
        model._ising_nu_prior_upper = nu_prior_upper  # type: ignore[attr-defined]
        model._ising_beta_prior_mu = beta_prior_mu  # type: ignore[attr-defined]
        model._ising_beta_prior_sigma = beta_prior_sigma  # type: ignore[attr-defined]
        model._ising_beta_prior_lower = beta_prior_lower  # type: ignore[attr-defined]
        model._ising_beta_prior_upper = beta_prior_upper  # type: ignore[attr-defined]
        if infer_gp_hyperparams:
            model._ising_gp_ell_prior_mu = math.log(base_gp_ell)  # type: ignore[attr-defined]
            model._ising_gp_ell_prior_sigma = gp_ell_prior_sigma  # type: ignore[attr-defined]
            if uses_correction:
                model._ising_correction_gp_ell_prior_mu = math.log(  # type: ignore[attr-defined]
                    correction_gp_ell
                )
        if uses_correction:
            if infer_omega:
                model._ising_omega_prior_lower = omega_prior_lower  # type: ignore[attr-defined]
                model._ising_omega_prior_upper = omega_prior_upper  # type: ignore[attr-defined]
            else:
                model._ising_omega_fixed = omega_fixed  # type: ignore[attr-defined]
            model._ising_correction_gp_ell_fixed = correction_gp_ell  # type: ignore[attr-defined]
            model._ising_correction_gp_eta_fixed = correction_gp_eta  # type: ignore[attr-defined]
    return model


def _pymc_interval_forward(value: float, lo: float, hi: float) -> float:
    """Map a bounded Uniform RV to PyMC 5's ``*_interval__`` coordinates.

    PyMC uses ``IntervalTransform`` (logit on (a, b)), not the linear
    ``(x - a) / (b - a)`` parameterization.
    """
    v = max(lo + 1e-12, min(hi - 1e-12, float(value)))
    return float(np.log(v - lo) - np.log(hi - v))


def _exponent_interval_point(
    model: _pm.Model,
    T_c: float,
    nu: float,
    beta: float,
    *,
    tc_reparam_ref: float,
    tc_delta_scale: float,
    reparametrize_Tc: bool,
) -> dict[str, float]:
    """Map physical exponents to transformed value-var coordinates for ``potentiallogp``."""
    point: dict[str, float] = {}
    for rv in model.free_RVs:
        key = f"{rv.name}_interval__"
        if rv.name == "T_c_delta":
            lo = (TC_PRIOR_LOWER - tc_reparam_ref) * tc_delta_scale
            hi = (TC_PRIOR_UPPER - tc_reparam_ref) * tc_delta_scale
            val = (float(T_c) - tc_reparam_ref) * tc_delta_scale
        elif rv.name == "T_c":
            lo, hi, val = TC_PRIOR_LOWER, TC_PRIOR_UPPER, float(T_c)
        elif rv.name == "nu":
            lo, hi, val = NU_PRIOR_LOWER, NU_PRIOR_UPPER, float(nu)
        elif rv.name == "beta":
            lo, hi, val = BETA_PRIOR_LOWER, BETA_PRIOR_UPPER, float(beta)
        else:
            raise ValueError(
                f"Unexpected free RV {rv.name!r} in exponent profile likelihood"
            )
        point[key] = _pymc_interval_forward(val, lo, hi)
    return point


def beta_affects_fss_likelihood(
    *,
    use_m: bool,
    use_m2: bool,
    use_m4: bool,
    use_chi: bool,
) -> bool:
    """Return whether beta enters the FSS GP likelihood (false for binder-only)."""
    return use_m or use_m2 or use_m4 or use_chi


def _active_value_vars(value_vars, output):
    """Value vars that the potential logp actually depends on."""
    inputs = set(graph_inputs([output]))
    return [v for v in value_vars if v in inputs]


def compile_fss_exponent_log_likelihood(
    observables: pd.DataFrame,
    *,
    infer_Tc: bool = True,
    infer_nu: bool = True,
    infer_beta: bool = True,
    reparametrize_Tc: bool = False,
    tc_reparam_ref: float | None = None,
    tc_delta_scale: float = TC_DELTA_SCALE,
    use_numba: bool = True,
    **build_kwargs,
) -> tuple[_pm.Model, Callable[[float, float, float], float]]:
    """Compile log p(data | T_c, nu, beta) using the same PyMC graph as ``fit``.

    Sums GP ``Potential`` terms only (no priors on exponents), evaluated on the
    inference graph (``infer_Tc`` / ``infer_nu`` / ``infer_beta``) so profile
    slices match what NUTS optimizes. By default the evaluator is compiled with
    PyTensor's Numba linker (~1000x faster than ``point_logps``).
    """
    model = build_fss_model(
        observables,
        infer_Tc=infer_Tc,
        infer_nu=infer_nu,
        infer_beta=infer_beta,
        reparametrize_Tc=reparametrize_Tc,
        tc_reparam_ref=tc_reparam_ref,
        tc_delta_scale=tc_delta_scale,
        use_exponent_data=False,
        **build_kwargs,
    )
    tc_ref = TC_EXACT if tc_reparam_ref is None else float(tc_reparam_ref)
    value_vars = list(model.value_vars)
    pot_logp = model.potentiallogp
    active_value_vars = _active_value_vars(value_vars, pot_logp)

    if use_numba:
        from .pytensor_numba import numba_compile_mode

        compiled = function(
            active_value_vars,
            pot_logp,
            mode=numba_compile_mode(),
        )
    else:
        pot_names = tuple(p.name for p in model.potentials)

        def evaluate(T_c: float, nu: float, beta: float) -> float:
            point = _exponent_interval_point(
                model,
                T_c,
                nu,
                beta,
                tc_reparam_ref=tc_ref,
                tc_delta_scale=tc_delta_scale,
                reparametrize_Tc=reparametrize_Tc,
            )
            logps = model.point_logps(point)
            return float(sum(logps[name] for name in pot_names))

        return model, evaluate

    def evaluate(T_c: float, nu: float, beta: float) -> float:
        if active_value_vars:
            point = _exponent_interval_point(
                model,
                T_c,
                nu,
                beta,
                tc_reparam_ref=tc_ref,
                tc_delta_scale=tc_delta_scale,
                reparametrize_Tc=reparametrize_Tc,
            )
            args = [point[v.name] for v in active_value_vars]
            return float(compiled(*args))
        return float(compiled())

    return model, evaluate
