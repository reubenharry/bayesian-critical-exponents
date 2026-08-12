"""FSS log marginal likelihood (NumPy + JAX), shared by profile and sampling."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from .constants import SPATIAL_DIMENSION, TC_EXACT, NU_EXACT
from .gp_kernels import DEFAULT_GP_KERNEL
from .gp_utils import (
    correction_gp_log_marginal_likelihood as _correction_gp_log_ml_numpy,
    gp_log_marginal_likelihood as _gp_log_ml_numpy,
)
from .observables import (
    df_L_T,
    df_binder_sigmas,
    df_chi_sigmas,
    df_magnetization_sigmas,
    df_moment_sigmas,
    validate_observables_df,
)
from .scaling_function import GP_JITTER, provisional_z

LikelihoodBackend = Literal["numpy", "jax"]


def uses_fss_correction(config) -> bool:
    """True when any enabled observable channel uses the L^{-omega} correction GP."""
    return (
        (config.use_m and config.correction_m)
        or (config.use_m2 and config.correction_m2)
        or (config.use_m4 and config.correction_m4)
        or (config.use_binder and config.correction_binder)
        or (config.use_chi and config.correction_chi)
    )


def uses_fss_discrepancy(config) -> bool:
    """True when any enabled channel uses a(t,L) observation-noise inflation."""
    return (
        (config.use_m and getattr(config, "discrepancy_m", False))
        or (config.use_m2 and getattr(config, "discrepancy_m2", False))
        or (config.use_m4 and getattr(config, "discrepancy_m4", False))
        or (config.use_binder and getattr(config, "discrepancy_binder", False))
        or (config.use_chi and getattr(config, "discrepancy_chi", False))
    )


@dataclass(frozen=True)
class FssGpScales:
    z_span: float
    base_gp_ell: float
    base_gp_eta: float
    correction_gp_ell: float
    correction_gp_eta: float


@dataclass(frozen=True)
class CompiledFssLikelihoodJax:
    """JIT-compiled FSS marginal likelihood with vmap'd profile helpers."""

    evaluate: Callable[[float, float, float], float]
    evaluate_jax: Callable[..., Any]
    evaluate_with_gp_jax: Callable[..., Any]
    profile_tc: Callable[[np.ndarray, float, float], np.ndarray]
    profile_nu: Callable[[np.ndarray, float, float], np.ndarray]
    profile_beta: Callable[[np.ndarray, float, float], np.ndarray]
    profile_joint: Callable[[np.ndarray, np.ndarray, str, float, float, float], np.ndarray]
    profile_joint_with_disc: Callable[
        [
            np.ndarray,
            np.ndarray,
            str,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
        ],
        np.ndarray,
    ]
    profile_joint_tc_nu_with_disc: Callable[
        [np.ndarray, np.ndarray, float, float, float, float, float, float], np.ndarray
    ]
    profile_triple_grid: Callable[
        [np.ndarray, np.ndarray, np.ndarray], np.ndarray
    ]
    gp_scales: FssGpScales

    def __call__(self, T_c: float, nu: float, beta: float) -> float:
        return self.evaluate(T_c, nu, beta)


def _config_gp_kernel(config) -> str:
    return getattr(config, "gp_kernel", DEFAULT_GP_KERNEL)


@dataclass(frozen=True)
class FssLikelihoodArrays:
    L: np.ndarray
    T: np.ndarray
    magnetization: np.ndarray | None = None
    sigma_m: np.ndarray | None = None
    log_m: np.ndarray | None = None
    sigma_log: np.ndarray | None = None
    m2: np.ndarray | None = None
    sigma_m2: np.ndarray | None = None
    m4: np.ndarray | None = None
    sigma_m4: np.ndarray | None = None
    binder: np.ndarray | None = None
    sigma_binder: np.ndarray | None = None
    chi: np.ndarray | None = None
    sigma_chi: np.ndarray | None = None


def fss_gp_scales(
    L: np.ndarray,
    T: np.ndarray,
    *,
    gp_ell_factor: float,
    gp_eta: float,
    correction_gp_ell_factor: float,
    correction_gp_eta: float,
    gp_ell: float | None = None,
    correction_gp_ell: float | None = None,
) -> FssGpScales:
    """GP amplitude/length scales.

    By default ``base_gp_ell = gp_ell_factor * z_span`` with provisional
    ``z`` at exact ``(T_c, ν)``.  Pass absolute ``gp_ell`` /
    ``correction_gp_ell`` to fix length scales independent of the data.
    """
    z_ref = provisional_z(L, T, T_c=TC_EXACT, nu=NU_EXACT)
    if z_ref.size < 2:
        z_span = 1.0
    else:
        z_span = float(z_ref.max() - z_ref.min())
        if not np.isfinite(z_span) or z_span <= 0.0:
            z_span = 1.0
    base_ell = float(gp_ell) if gp_ell is not None else gp_ell_factor * z_span
    corr_ell = (
        float(correction_gp_ell)
        if correction_gp_ell is not None
        else correction_gp_ell_factor * z_span
    )
    return FssGpScales(
        z_span=z_span,
        base_gp_ell=base_ell,
        base_gp_eta=gp_eta,
        correction_gp_ell=corr_ell,
        correction_gp_eta=correction_gp_eta,
    )


def collapse_z(
    L: np.ndarray,
    T: np.ndarray,
    *,
    T_c: float,
    nu: float,
) -> np.ndarray:
    L = np.asarray(L, dtype=np.float64).ravel()
    T = np.asarray(T, dtype=np.float64).ravel()
    t = (T - T_c) / T_c
    return t * L ** (1.0 / nu)


def extract_likelihood_arrays(
    observables: pd.DataFrame,
    config,
) -> FssLikelihoodArrays:
    validate_observables_df(
        observables,
        use_m=config.use_m,
        use_m2=config.use_m2,
        use_m4=config.use_m4,
        use_binder=config.use_binder,
        use_chi=config.use_chi,
    )
    L, T = df_L_T(observables)
    scale = float(config.obs_sigma_scale)
    arrays = FssLikelihoodArrays(L=L, T=T)

    if config.use_m:
        magnetization, sigma_m, log_m, sigma_log = df_magnetization_sigmas(observables)
        if scale != 1.0:
            sigma_m = sigma_m * scale
            sigma_log = sigma_log * scale
        arrays = FssLikelihoodArrays(
            L=L,
            T=T,
            magnetization=magnetization,
            sigma_m=sigma_m,
            log_m=log_m,
            sigma_log=sigma_log,
        )

    if config.use_m2:
        m2, sigma_m2 = df_moment_sigmas(
            observables, value_col="m2", std_col="m2_std", n_eff_col="n_eff_m2"
        )
        if scale != 1.0:
            sigma_m2 = sigma_m2 * scale
        arrays = FssLikelihoodArrays(
            **{
                **arrays.__dict__,
                "m2": m2,
                "sigma_m2": sigma_m2,
            }
        )

    if config.use_m4:
        m4, sigma_m4 = df_moment_sigmas(
            observables, value_col="m4", std_col="m4_std", n_eff_col="n_eff_m4"
        )
        if scale != 1.0:
            sigma_m4 = sigma_m4 * scale
        arrays = FssLikelihoodArrays(
            **{
                **arrays.__dict__,
                "m4": m4,
                "sigma_m4": sigma_m4,
            }
        )

    if config.use_binder:
        binder, sigma_binder = df_binder_sigmas(observables)
        if scale != 1.0:
            sigma_binder = sigma_binder * scale
        arrays = FssLikelihoodArrays(
            **{
                **arrays.__dict__,
                "binder": binder,
                "sigma_binder": sigma_binder,
            }
        )

    if config.use_chi:
        chi, sigma_chi = df_chi_sigmas(observables)
        if scale != 1.0:
            sigma_chi = sigma_chi * scale
        arrays = FssLikelihoodArrays(
            **{
                **arrays.__dict__,
                "chi": chi,
                "sigma_chi": sigma_chi,
            }
        )

    return arrays


def _gp_ell_kwargs(config) -> dict[str, float | None]:
    return {
        "gp_ell_factor": config.gp_ell_factor,
        "gp_eta": config.gp_eta,
        "correction_gp_ell_factor": config.correction_gp_ell_factor,
        "correction_gp_eta": config.correction_gp_eta,
        "gp_ell": getattr(config, "gp_ell", None),
        "correction_gp_ell": getattr(config, "correction_gp_ell", None),
    }


def _chi_gp_ell_scales(
    *,
    z: np.ndarray,
    config,
    scales: FssGpScales,
) -> tuple[float, float]:
    chi_only = config.use_chi and not (
        config.use_m or config.use_m2 or config.use_m4 or config.use_binder
    )
    if chi_only and config.infer_nu and getattr(config, "gp_ell", None) is None:
        chi_z_span = float(z.max() - z.min())
        return (
            config.gp_ell_factor * chi_z_span,
            config.correction_gp_ell_factor * chi_z_span,
        )
    return scales.base_gp_ell, scales.correction_gp_ell


def _gp_log_ml(
    x: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    *,
    length_scale: float,
    amplitude: float,
    jitter: float,
    backend: LikelihoodBackend,
    kernel: str = DEFAULT_GP_KERNEL,
) -> float:
    if backend == "numpy":
        return _gp_log_ml_numpy(
            x,
            y,
            sigma,
            length_scale=length_scale,
            amplitude=amplitude,
            jitter=jitter,
            kernel=kernel,
        )
    import jax.numpy as jnp

    from .gp_jax import gp_log_marginal_likelihood as _gp_log_ml_jax

    return float(
        _gp_log_ml_jax(
            jnp.asarray(x),
            jnp.asarray(y),
            jnp.asarray(sigma),
            length_scale=length_scale,
            amplitude=amplitude,
            jitter=jitter,
            kernel=kernel,
        )
    )


def _correction_gp_log_ml(
    z: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    L: np.ndarray,
    *,
    omega: float,
    gp_ell: float,
    gp_eta: float,
    correction_gp_ell: float,
    correction_gp_eta: float,
    jitter: float,
    backend: LikelihoodBackend,
    kernel: str = DEFAULT_GP_KERNEL,
) -> float:
    if backend == "numpy":
        return _correction_gp_log_ml_numpy(
            z,
            y,
            sigma,
            L,
            omega=omega,
            gp_ell=gp_ell,
            gp_eta=gp_eta,
            correction_gp_ell=correction_gp_ell,
            correction_gp_eta=correction_gp_eta,
            jitter=jitter,
            kernel=kernel,
        )
    import jax.numpy as jnp

    from .gp_jax import correction_gp_log_marginal_likelihood as _corr_jax

    return float(
        _corr_jax(
            jnp.asarray(z),
            jnp.asarray(y),
            jnp.asarray(sigma),
            jnp.asarray(L),
            omega=omega,
            gp_ell=gp_ell,
            gp_eta=gp_eta,
            correction_gp_ell=correction_gp_ell,
            correction_gp_eta=correction_gp_eta,
            jitter=jitter,
            kernel=kernel,
        )
    )


def _linear_moment_channel_log_ml(
    *,
    z: np.ndarray,
    L: np.ndarray,
    values: np.ndarray,
    sigma: np.ndarray,
    beta: float,
    nu: float,
    moment: int,
    correction: bool,
    scales: FssGpScales,
    omega: float,
    backend: LikelihoodBackend,
    kernel: str = DEFAULT_GP_KERNEL,
) -> float:
    power = float(moment) * beta / nu
    phi = values * L**power
    sigma_phi = sigma * L**power
    if correction:
        return _correction_gp_log_ml(
            z,
            phi,
            sigma_phi,
            L,
            omega=omega,
            gp_ell=scales.base_gp_ell,
            gp_eta=scales.base_gp_eta,
            correction_gp_ell=scales.correction_gp_ell,
            correction_gp_eta=scales.correction_gp_eta,
            jitter=GP_JITTER,
            backend=backend,
            kernel=kernel,
        )
    return _gp_log_ml(
        z,
        phi,
        sigma_phi,
        length_scale=scales.base_gp_ell,
        amplitude=scales.base_gp_eta,
        jitter=GP_JITTER,
        backend=backend,
        kernel=kernel,
    )


def fss_log_marginal_likelihood_from_arrays(
    arrays: FssLikelihoodArrays,
    config,
    *,
    T_c: float,
    nu: float,
    beta: float,
    backend: LikelihoodBackend = "numpy",
) -> float:
    scales = fss_gp_scales(
        arrays.L,
        arrays.T,
        **_gp_ell_kwargs(config),
    )
    z = collapse_z(arrays.L, arrays.T, T_c=T_c, nu=nu)
    omega = float(config.omega_fixed)
    gamma = SPATIAL_DIMENSION * nu - 2.0 * beta
    chi_gp_ell, chi_correction_gp_ell = _chi_gp_ell_scales(
        z=z, config=config, scales=scales
    )
    gp_kernel = _config_gp_kernel(config)

    log_ml = 0.0

    if config.use_m:
        assert arrays.magnetization is not None
        assert arrays.sigma_m is not None
        assert arrays.log_m is not None
        assert arrays.sigma_log is not None
        if config.use_log_m and not config.correction_m:
            log_phi_m = arrays.log_m + (beta / nu) * np.log(arrays.L)
            log_ml += _gp_log_ml(
                z,
                log_phi_m,
                arrays.sigma_log,
                length_scale=scales.base_gp_ell,
                amplitude=scales.base_gp_eta,
                jitter=GP_JITTER,
                backend=backend,
                kernel=gp_kernel,
            )
        elif config.correction_m:
            phi_m = arrays.magnetization * arrays.L ** (beta / nu)
            sigma_phi_m = arrays.sigma_m * arrays.L ** (beta / nu)
            log_ml += _correction_gp_log_ml(
                z,
                phi_m,
                sigma_phi_m,
                arrays.L,
                omega=omega,
                gp_ell=scales.base_gp_ell,
                gp_eta=scales.base_gp_eta,
                correction_gp_ell=scales.correction_gp_ell,
                correction_gp_eta=scales.correction_gp_eta,
                jitter=GP_JITTER,
                backend=backend,
                kernel=gp_kernel,
            )
        else:
            phi_m = arrays.magnetization * arrays.L ** (beta / nu)
            sigma_phi_m = arrays.sigma_m * arrays.L ** (beta / nu)
            log_ml += _gp_log_ml(
                z,
                phi_m,
                sigma_phi_m,
                length_scale=scales.base_gp_ell,
                amplitude=scales.base_gp_eta,
                jitter=GP_JITTER,
                backend=backend,
                kernel=gp_kernel,
            )

    if config.use_m2:
        assert arrays.m2 is not None and arrays.sigma_m2 is not None
        log_ml += _linear_moment_channel_log_ml(
            z=z,
            L=arrays.L,
            values=arrays.m2,
            sigma=arrays.sigma_m2,
            beta=beta,
            nu=nu,
            moment=2,
            correction=config.correction_m2,
            scales=scales,
            omega=omega,
            backend=backend,
            kernel=gp_kernel,
        )

    if config.use_m4:
        assert arrays.m4 is not None and arrays.sigma_m4 is not None
        log_ml += _linear_moment_channel_log_ml(
            z=z,
            L=arrays.L,
            values=arrays.m4,
            sigma=arrays.sigma_m4,
            beta=beta,
            nu=nu,
            moment=4,
            correction=config.correction_m4,
            scales=scales,
            omega=omega,
            backend=backend,
            kernel=gp_kernel,
        )

    if config.use_binder:
        assert arrays.binder is not None and arrays.sigma_binder is not None
        if config.correction_binder:
            log_ml += _correction_gp_log_ml(
                z,
                arrays.binder,
                arrays.sigma_binder,
                arrays.L,
                omega=omega,
                gp_ell=scales.base_gp_ell,
                gp_eta=scales.base_gp_eta,
                correction_gp_ell=scales.correction_gp_ell,
                correction_gp_eta=scales.correction_gp_eta,
                jitter=GP_JITTER,
                backend=backend,
                kernel=gp_kernel,
            )
        else:
            log_ml += _gp_log_ml(
                z,
                arrays.binder,
                arrays.sigma_binder,
                length_scale=scales.base_gp_ell,
                amplitude=scales.base_gp_eta,
                jitter=GP_JITTER,
                backend=backend,
                kernel=gp_kernel,
            )

    if config.use_chi:
        assert arrays.chi is not None and arrays.sigma_chi is not None
        chi_scale = np.exp((-gamma / nu) * np.log(arrays.L))
        phi_chi = arrays.chi * chi_scale
        sigma_phi_chi = arrays.sigma_chi * chi_scale
        if config.correction_chi:
            log_ml += _correction_gp_log_ml(
                z,
                phi_chi,
                sigma_phi_chi,
                arrays.L,
                omega=omega,
                gp_ell=chi_gp_ell,
                gp_eta=scales.base_gp_eta,
                correction_gp_ell=chi_correction_gp_ell,
                correction_gp_eta=scales.correction_gp_eta,
                jitter=GP_JITTER,
                backend=backend,
                kernel=gp_kernel,
            )
        else:
            log_phi_chi = np.log(np.maximum(phi_chi, 1e-12))
            sigma_log_chi = sigma_phi_chi / np.maximum(phi_chi, 1e-12)
            log_ml += _gp_log_ml(
                z,
                log_phi_chi,
                sigma_log_chi,
                length_scale=chi_gp_ell,
                amplitude=scales.base_gp_eta,
                jitter=GP_JITTER,
                backend=backend,
                kernel=gp_kernel,
            )

    return float(log_ml)


def fss_log_marginal_likelihood(
    observables: pd.DataFrame,
    config,
    *,
    T_c: float,
    nu: float,
    beta: float,
    backend: LikelihoodBackend = "numpy",
) -> float:
    """Sum of marginalized GP channel log-likelihoods at fixed exponents."""
    arrays = extract_likelihood_arrays(observables, config)
    return fss_log_marginal_likelihood_from_arrays(
        arrays,
        config,
        T_c=T_c,
        nu=nu,
        beta=beta,
        backend=backend,
    )


def compile_fss_log_marginal_likelihood_jax(
    observables: pd.DataFrame,
    config,
) -> CompiledFssLikelihoodJax:
    """Return JIT + vmap helpers for ``(T_c, nu, beta) -> log p(data | exponents)``."""
    import jax
    import jax.numpy as jnp

    from .jax_config import configure_jax

    configure_jax()

    arrays = extract_likelihood_arrays(observables, config)
    scales = fss_gp_scales(
        arrays.L,
        arrays.T,
        **_gp_ell_kwargs(config),
    )

    L = jnp.asarray(arrays.L)
    T = jnp.asarray(arrays.T)
    omega = float(config.omega_fixed)
    chi_only = config.use_chi and not (
        config.use_m or config.use_m2 or config.use_m4 or config.use_binder
    )
    # Dynamic chi span only when ell is data-dependent (no absolute gp_ell).
    chi_dynamic_span = (
        chi_only and config.infer_nu and getattr(config, "gp_ell", None) is None
    )
    static = {
        "use_m": config.use_m,
        "use_m2": config.use_m2,
        "use_m4": config.use_m4,
        "use_binder": config.use_binder,
        "use_chi": config.use_chi,
        "correction_m": config.correction_m,
        "correction_m2": config.correction_m2,
        "correction_m4": config.correction_m4,
        "correction_binder": config.correction_binder,
        "correction_chi": config.correction_chi,
        "discrepancy_m": bool(getattr(config, "discrepancy_m", False)),
        "discrepancy_m2": bool(getattr(config, "discrepancy_m2", False)),
        "discrepancy_m4": bool(getattr(config, "discrepancy_m4", False)),
        "discrepancy_binder": bool(getattr(config, "discrepancy_binder", False)),
        "discrepancy_chi": bool(getattr(config, "discrepancy_chi", False)),
        "discrepancy_form": str(getattr(config, "discrepancy_form", "noise")),
        "z_disc_threshold": float(
            getattr(config, "z_disc_threshold", 10.0)
        ),
        "use_log_m": config.use_log_m,
        "chi_dynamic_span": chi_dynamic_span,
        "gp_ell_factor": config.gp_ell_factor,
        "correction_gp_ell_factor": config.correction_gp_ell_factor,
        "base_gp_ell": scales.base_gp_ell,
        "base_gp_eta": scales.base_gp_eta,
        "correction_gp_ell": scales.correction_gp_ell,
        "correction_gp_eta": scales.correction_gp_eta,
        "fixed_z_span": scales.z_span,
        "gp_kernel": _config_gp_kernel(config),
    }

    channel_arrays: dict[str, jnp.ndarray | None] = {
        "magnetization": None,
        "sigma_m": None,
        "log_m": None,
        "sigma_log": None,
        "m2": None,
        "sigma_m2": None,
        "m4": None,
        "sigma_m4": None,
        "binder": None,
        "sigma_binder": None,
        "chi": None,
        "sigma_chi": None,
    }
    for key in channel_arrays:
        value = getattr(arrays, key)
        if value is not None:
            channel_arrays[key] = jnp.asarray(value)

    from .gp_jax import (
        correction_gp_log_marginal_likelihood as corr_gp_jax,
        discrepancy_gp_log_marginal_likelihood as disc_gp_jax,
        gp_log_marginal_likelihood as gp_jax,
    )

    def _collapse_z_jax(T_c: float, nu: float) -> jnp.ndarray:
        t = (T - T_c) / T_c
        return t * L ** (1.0 / nu)

    def _chi_ells(
        z: jnp.ndarray,
        gp_ell: jnp.ndarray,
        correction_gp_ell: jnp.ndarray,
        *,
        infer_gp_hyperparams: bool,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if infer_gp_hyperparams:
            return gp_ell, correction_gp_ell
        if static["chi_dynamic_span"]:
            chi_z_span = jnp.max(z) - jnp.min(z)
            return (
                static["gp_ell_factor"] * chi_z_span,
                static["correction_gp_ell_factor"] * chi_z_span,
            )
        return gp_ell, correction_gp_ell

    def evaluate_with_gp(
        T_c: float,
        nu: float,
        beta: float,
        gp_ell: float,
        gp_eta: float,
        correction_gp_ell: float,
        correction_gp_eta: float,
        disc_t0: float,
        disc_L0: float,
        disc_p: float,
        disc_q: float,
        disc_sigma_model: float,
        infer_gp_hyperparams: bool,
    ) -> jnp.ndarray:
        z = _collapse_z_jax(T_c, nu)
        t = (T - T_c) / T_c
        log_ml = jnp.array(0.0, dtype=jnp.float64)
        gamma = SPATIAL_DIMENSION * nu - 2.0 * beta

        def _amplitude() -> jnp.ndarray:
            return (jnp.abs(t) / disc_t0) ** disc_p + (disc_L0 / L) ** disc_q

        def _inflate(sigma: jnp.ndarray, *, enabled: bool) -> jnp.ndarray:
            if not enabled or static["discrepancy_form"] != "noise":
                return sigma
            return jnp.sqrt(sigma**2 + (disc_sigma_model * _amplitude()) ** 2)

        def _plain_or_disc_gp(
            y: jnp.ndarray,
            sigma: jnp.ndarray,
            *,
            disc_enabled: bool,
            length_scale: jnp.ndarray | float,
            amplitude: jnp.ndarray | float,
            phi_scale: jnp.ndarray | float | None = None,
        ) -> jnp.ndarray:
            """Plain GP, noise-inflated GP, or additive a·g GP.

            Additive form is defined on the *raw* observable
            ``raw = phi_scale^{-1} f(z) + a(t,L) g(z) + ε`` (slide; Φ₁
            dropped).  With collapsed ``y = raw * phi_scale`` this is
            ``y = f + (a * phi_scale) g + ε_y``.  Pass ``phi_scale = L^{β/ν}``
            for magnetization, ``L^{2β/ν}`` for ``m²``, ``1`` for Binder, etc.
            """
            form = static["discrepancy_form"]
            if disc_enabled and form in (
                "additive_gp",
                "additive_gp_z_threshold",
            ):
                if form == "additive_gp_z_threshold":
                    # Collapsed-space gate: a=1 for |z|>threshold, else 0.
                    a_collapsed = jnp.where(
                        jnp.abs(z) > static["z_disc_threshold"],
                        1.0,
                        0.0,
                    )
                else:
                    a_raw = _amplitude()
                    if phi_scale is None:
                        a_collapsed = a_raw
                    else:
                        a_collapsed = a_raw * phi_scale
                return disc_gp_jax(
                    z,
                    y,
                    sigma,
                    a_collapsed,
                    gp_ell=length_scale,
                    gp_eta=amplitude,
                    disc_gp_ell=length_scale,
                    disc_gp_eta=disc_sigma_model,
                    jitter=GP_JITTER,
                    kernel=static["gp_kernel"],
                )
            return gp_jax(
                z,
                y,
                _inflate(sigma, enabled=disc_enabled),
                length_scale=length_scale,
                amplitude=amplitude,
                jitter=GP_JITTER,
                kernel=static["gp_kernel"],
            )

        if static["use_m"]:
            assert channel_arrays["magnetization"] is not None
            assert channel_arrays["sigma_m"] is not None
            assert channel_arrays["log_m"] is not None
            assert channel_arrays["sigma_log"] is not None
            m_phi_scale = L ** (beta / nu)
            if static["use_log_m"] and not static["correction_m"]:
                log_phi_m = channel_arrays["log_m"] + (beta / nu) * jnp.log(L)
                # log-Φ is not the slide's additive-on-m model; keep a unscaled.
                log_ml = log_ml + _plain_or_disc_gp(
                    log_phi_m,
                    channel_arrays["sigma_log"],
                    disc_enabled=static["discrepancy_m"],
                    length_scale=gp_ell,
                    amplitude=gp_eta,
                )
            elif static["correction_m"]:
                phi_m = channel_arrays["magnetization"] * m_phi_scale
                sigma_phi_m = _inflate(
                    channel_arrays["sigma_m"] * m_phi_scale,
                    enabled=static["discrepancy_m"],
                )
                log_ml = log_ml + corr_gp_jax(
                    z,
                    phi_m,
                    sigma_phi_m,
                    L,
                    omega=omega,
                    gp_ell=gp_ell,
                    gp_eta=gp_eta,
                    correction_gp_ell=correction_gp_ell,
                    correction_gp_eta=correction_gp_eta,
                    jitter=GP_JITTER,
                    kernel=static["gp_kernel"],
                )
            else:
                phi_m = channel_arrays["magnetization"] * m_phi_scale
                sigma_phi_m = channel_arrays["sigma_m"] * m_phi_scale
                log_ml = log_ml + _plain_or_disc_gp(
                    phi_m,
                    sigma_phi_m,
                    disc_enabled=static["discrepancy_m"],
                    length_scale=gp_ell,
                    amplitude=gp_eta,
                    phi_scale=m_phi_scale,
                )

        if static["use_m2"]:
            assert channel_arrays["m2"] is not None
            assert channel_arrays["sigma_m2"] is not None
            power = 2.0 * beta / nu
            m2_phi_scale = L**power
            phi = channel_arrays["m2"] * m2_phi_scale
            sigma_phi = channel_arrays["sigma_m2"] * m2_phi_scale
            if static["correction_m2"]:
                log_ml = log_ml + corr_gp_jax(
                    z,
                    phi,
                    _inflate(sigma_phi, enabled=static["discrepancy_m2"]),
                    L,
                    omega=omega,
                    gp_ell=gp_ell,
                    gp_eta=gp_eta,
                    correction_gp_ell=correction_gp_ell,
                    correction_gp_eta=correction_gp_eta,
                    jitter=GP_JITTER,
                    kernel=static["gp_kernel"],
                )
            else:
                log_ml = log_ml + _plain_or_disc_gp(
                    phi,
                    sigma_phi,
                    disc_enabled=static["discrepancy_m2"],
                    length_scale=gp_ell,
                    amplitude=gp_eta,
                    phi_scale=m2_phi_scale,
                )

        if static["use_m4"]:
            assert channel_arrays["m4"] is not None
            assert channel_arrays["sigma_m4"] is not None
            power = 4.0 * beta / nu
            m4_phi_scale = L**power
            phi = channel_arrays["m4"] * m4_phi_scale
            sigma_phi = channel_arrays["sigma_m4"] * m4_phi_scale
            if static["correction_m4"]:
                log_ml = log_ml + corr_gp_jax(
                    z,
                    phi,
                    _inflate(sigma_phi, enabled=static["discrepancy_m4"]),
                    L,
                    omega=omega,
                    gp_ell=gp_ell,
                    gp_eta=gp_eta,
                    correction_gp_ell=correction_gp_ell,
                    correction_gp_eta=correction_gp_eta,
                    jitter=GP_JITTER,
                    kernel=static["gp_kernel"],
                )
            else:
                log_ml = log_ml + _plain_or_disc_gp(
                    phi,
                    sigma_phi,
                    disc_enabled=static["discrepancy_m4"],
                    length_scale=gp_ell,
                    amplitude=gp_eta,
                    phi_scale=m4_phi_scale,
                )

        if static["use_binder"]:
            assert channel_arrays["binder"] is not None
            assert channel_arrays["sigma_binder"] is not None
            if static["correction_binder"]:
                log_ml = log_ml + corr_gp_jax(
                    z,
                    channel_arrays["binder"],
                    _inflate(
                        channel_arrays["sigma_binder"],
                        enabled=static["discrepancy_binder"],
                    ),
                    L,
                    omega=omega,
                    gp_ell=gp_ell,
                    gp_eta=gp_eta,
                    correction_gp_ell=correction_gp_ell,
                    correction_gp_eta=correction_gp_eta,
                    jitter=GP_JITTER,
                    kernel=static["gp_kernel"],
                )
            else:
                log_ml = log_ml + _plain_or_disc_gp(
                    channel_arrays["binder"],
                    channel_arrays["sigma_binder"],
                    disc_enabled=static["discrepancy_binder"],
                    length_scale=gp_ell,
                    amplitude=gp_eta,
                    phi_scale=1.0,
                )

        if static["use_chi"]:
            assert channel_arrays["chi"] is not None
            assert channel_arrays["sigma_chi"] is not None
            # Φ_χ = χ L^{-γ/ν}  ⇔  χ = L^{γ/ν} Φ_χ + …; additive on χ ⇒
            # Φ_χ = f + a L^{-γ/ν} g.
            chi_phi_scale = jnp.exp((-gamma / nu) * jnp.log(L))
            phi_chi = channel_arrays["chi"] * chi_phi_scale
            sigma_phi_chi = channel_arrays["sigma_chi"] * chi_phi_scale
            chi_gp_ell, chi_correction_gp_ell = _chi_ells(
                z,
                gp_ell,
                correction_gp_ell,
                infer_gp_hyperparams=infer_gp_hyperparams,
            )
            if static["correction_chi"]:
                log_ml = log_ml + corr_gp_jax(
                    z,
                    phi_chi,
                    _inflate(sigma_phi_chi, enabled=static["discrepancy_chi"]),
                    L,
                    omega=omega,
                    gp_ell=chi_gp_ell,
                    gp_eta=gp_eta,
                    correction_gp_ell=chi_correction_gp_ell,
                    correction_gp_eta=correction_gp_eta,
                    jitter=GP_JITTER,
                    kernel=static["gp_kernel"],
                )
            elif static["discrepancy_chi"] and static["discrepancy_form"] in (
                "additive_gp",
                "additive_gp_z_threshold",
            ):
                # Slide model is additive on χ; fit linear Φ_χ so
                # Φ = f + a L^{-γ/ν} g.
                log_ml = log_ml + _plain_or_disc_gp(
                    phi_chi,
                    sigma_phi_chi,
                    disc_enabled=True,
                    length_scale=chi_gp_ell,
                    amplitude=gp_eta,
                    phi_scale=chi_phi_scale,
                )
            else:
                log_phi_chi = jnp.log(jnp.maximum(phi_chi, 1e-12))
                sigma_log_chi = sigma_phi_chi / jnp.maximum(phi_chi, 1e-12)
                log_ml = log_ml + _plain_or_disc_gp(
                    log_phi_chi,
                    sigma_log_chi,
                    disc_enabled=static["discrepancy_chi"],
                    length_scale=chi_gp_ell,
                    amplitude=gp_eta,
                )

        return log_ml

    def evaluate(T_c: float, nu: float, beta: float) -> jnp.ndarray:
        from .discrepancy import (
            DISCREPANCY_L0_INIT,
            DISCREPANCY_P_INIT,
            DISCREPANCY_Q_INIT,
            DISCREPANCY_SIGMA_MODEL_INIT,
            DISCREPANCY_T0_INIT,
        )

        return evaluate_with_gp(
            T_c,
            nu,
            beta,
            static["base_gp_ell"],
            static["base_gp_eta"],
            static["correction_gp_ell"],
            static["correction_gp_eta"],
            DISCREPANCY_T0_INIT,
            DISCREPANCY_L0_INIT,
            DISCREPANCY_P_INIT,
            DISCREPANCY_Q_INIT,
            DISCREPANCY_SIGMA_MODEL_INIT,
            False,
        )

    evaluate_with_gp_jit = jax.jit(evaluate_with_gp, static_argnames=("infer_gp_hyperparams",))
    evaluate_jit = jax.jit(evaluate)
    profile_tc_jit = jax.jit(
        lambda tc_grid, nu, beta: jax.vmap(
            lambda tc: evaluate(tc, nu, beta), in_axes=(0,)
        )(jnp.asarray(tc_grid, dtype=jnp.float64))
    )
    profile_nu_jit = jax.jit(
        lambda nu_grid, tc, beta: jax.vmap(
            lambda nu: evaluate(tc, nu, beta), in_axes=(0,)
        )(jnp.asarray(nu_grid, dtype=jnp.float64))
    )
    profile_beta_jit = jax.jit(
        lambda beta_grid, tc, nu: jax.vmap(
            lambda beta: evaluate(tc, nu, beta), in_axes=(0,)
        )(jnp.asarray(beta_grid, dtype=jnp.float64))
    )

    def _profile_joint_jit(
        x_grid: jnp.ndarray,
        y_grid: jnp.ndarray,
        pair: str,
        tc: float,
        nu: float,
        beta: float,
    ) -> jnp.ndarray:
        x_grid = jnp.asarray(x_grid, dtype=jnp.float64)
        y_grid = jnp.asarray(y_grid, dtype=jnp.float64)

        def eval_xy(x: float, y: float) -> jnp.ndarray:
            if pair == "T_c_nu":
                return evaluate(x, y, beta)
            if pair == "T_c_beta":
                return evaluate(x, nu, y)
            if pair == "nu_beta":
                return evaluate(tc, x, y)
            raise ValueError(f"Unsupported joint pair {pair!r}")

        return jax.vmap(jax.vmap(eval_xy, in_axes=(None, 0)), in_axes=(0, None))(
            x_grid, y_grid
        )

    profile_joint_jit = jax.jit(_profile_joint_jit, static_argnames=("pair",))

    def _profile_joint_with_disc_jit(
        x_grid: jnp.ndarray,
        y_grid: jnp.ndarray,
        pair: str,
        tc: float,
        nu: float,
        beta: float,
        disc_t0: float,
        disc_L0: float,
        disc_p: float,
        disc_q: float,
        disc_sigma_model: float,
    ) -> jnp.ndarray:
        """log ML on a joint pair with slider-controlled discrepancy params."""
        x_grid = jnp.asarray(x_grid, dtype=jnp.float64)
        y_grid = jnp.asarray(y_grid, dtype=jnp.float64)

        def _eval_exponents(T_c: float, nu_v: float, beta_v: float) -> jnp.ndarray:
            return evaluate_with_gp(
                T_c,
                nu_v,
                beta_v,
                static["base_gp_ell"],
                static["base_gp_eta"],
                static["correction_gp_ell"],
                static["correction_gp_eta"],
                disc_t0,
                disc_L0,
                disc_p,
                disc_q,
                disc_sigma_model,
                False,
            )

        def eval_xy(x: float, y: float) -> jnp.ndarray:
            if pair == "T_c_nu":
                return _eval_exponents(x, y, beta)
            if pair == "T_c_beta":
                return _eval_exponents(x, nu, y)
            if pair == "nu_beta":
                return _eval_exponents(tc, x, y)
            raise ValueError(f"Unsupported joint pair {pair!r}")

        return jax.vmap(jax.vmap(eval_xy, in_axes=(None, 0)), in_axes=(0, None))(
            x_grid, y_grid
        )

    profile_joint_with_disc_jit = jax.jit(
        _profile_joint_with_disc_jit, static_argnames=("pair",)
    )

    def _profile_joint_tc_nu_with_disc_jit(
        x_grid: jnp.ndarray,
        y_grid: jnp.ndarray,
        beta: float,
        disc_t0: float,
        disc_L0: float,
        disc_p: float,
        disc_q: float,
        disc_sigma_model: float,
    ) -> jnp.ndarray:
        """Backward-compatible (T_c, nu) disc profile."""
        return _profile_joint_with_disc_jit(
            x_grid,
            y_grid,
            "T_c_nu",
            0.0,
            0.0,
            beta,
            disc_t0,
            disc_L0,
            disc_p,
            disc_q,
            disc_sigma_model,
        )

    profile_joint_tc_nu_with_disc_jit = jax.jit(_profile_joint_tc_nu_with_disc_jit)

    @jax.jit
    def triple_grid_jit(
        tc_grid: jnp.ndarray,
        nu_grid: jnp.ndarray,
        beta_grid: jnp.ndarray,
    ) -> jnp.ndarray:
        tc_grid = jnp.asarray(tc_grid, dtype=jnp.float64)
        nu_grid = jnp.asarray(nu_grid, dtype=jnp.float64)
        beta_grid = jnp.asarray(beta_grid, dtype=jnp.float64)
        tc_mesh, nu_mesh, beta_mesh = jnp.meshgrid(
            tc_grid, nu_grid, beta_grid, indexing="ij"
        )
        flat = jax.vmap(
            lambda tc, nu, beta: evaluate(tc, nu, beta), in_axes=(0, 0, 0)
        )(tc_mesh.ravel(), nu_mesh.ravel(), beta_mesh.ravel())
        return flat.reshape(tc_mesh.shape)

    def _to_numpy(arr) -> np.ndarray:
        return np.asarray(arr, dtype=np.float64)

    def scalar(T_c: float, nu: float, beta: float) -> float:
        return float(evaluate_jit(T_c, nu, beta))

    return CompiledFssLikelihoodJax(
        evaluate=scalar,
        evaluate_jax=evaluate_jit,
        evaluate_with_gp_jax=evaluate_with_gp_jit,
        profile_tc=lambda grid, nu, beta: _to_numpy(profile_tc_jit(grid, nu, beta)),
        profile_nu=lambda grid, tc, beta: _to_numpy(profile_nu_jit(grid, tc, beta)),
        profile_beta=lambda grid, tc, nu: _to_numpy(profile_beta_jit(grid, tc, nu)),
        profile_joint=lambda xg, yg, pair, tc, nu, beta: _to_numpy(
            profile_joint_jit(xg, yg, pair, tc, nu, beta).T
        ),
        profile_joint_with_disc=lambda xg, yg, pair, tc, nu, beta, t0, L0, p, q, sigma_model: (
            _to_numpy(
                profile_joint_with_disc_jit(
                    xg, yg, pair, tc, nu, beta, t0, L0, p, q, sigma_model
                ).T
            )
        ),
        profile_joint_tc_nu_with_disc=lambda xg, yg, beta, t0, L0, p, q, sigma_model: (
            _to_numpy(
                profile_joint_tc_nu_with_disc_jit(
                    xg, yg, beta, t0, L0, p, q, sigma_model
                ).T
            )
        ),
        profile_triple_grid=lambda tcg, nug, betag: _to_numpy(
            triple_grid_jit(tcg, nug, betag)
        ),
        gp_scales=scales,
    )


def profile_joint_with_disc(
    compiled: CompiledFssLikelihoodJax,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    *,
    pair: str,
    T_c: float,
    nu: float,
    beta: float,
    t0: float,
    L0: float,
    p: float,
    q: float,
    sigma_model: float,
) -> np.ndarray:
    """Evaluate log ML on a joint exponent pair with fixed discrepancy params.

    ``pair`` is one of ``\"T_c_nu\"``, ``\"T_c_beta\"``, ``\"nu_beta\"``. The unused
    fixed exponent among ``(T_c, nu, beta)`` is ignored. Returns shape
    ``(len(y_grid), len(x_grid))``.
    """
    return compiled.profile_joint_with_disc(
        np.asarray(x_grid, dtype=np.float64),
        np.asarray(y_grid, dtype=np.float64),
        str(pair),
        float(T_c),
        float(nu),
        float(beta),
        float(t0),
        float(L0),
        float(p),
        float(q),
        float(sigma_model),
    )


def profile_joint_tc_nu_with_disc(
    compiled: CompiledFssLikelihoodJax,
    tc_grid: np.ndarray,
    nu_grid: np.ndarray,
    *,
    beta: float,
    t0: float,
    L0: float,
    p: float,
    q: float,
    sigma_model: float,
) -> np.ndarray:
    """Evaluate log marginal likelihood on a (T_c, nu) grid with fixed discrepancy params.

    Returns an array of shape ``(len(nu_grid), len(tc_grid))``. Unlike
    ``compiled.profile_joint``, discrepancy parameters are caller-controlled (for
    interactive sliders) rather than hardcoded to ``DISCREPANCY_*_INIT``.
    """
    return profile_joint_with_disc(
        compiled,
        tc_grid,
        nu_grid,
        pair="T_c_nu",
        T_c=0.0,
        nu=0.0,
        beta=beta,
        t0=t0,
        L0=L0,
        p=p,
        q=q,
        sigma_model=sigma_model,
    )
