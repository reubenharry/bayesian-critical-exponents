#!/usr/bin/env python3
"""Sample the Bayesian FSS + GP model and save posterior draws."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ._deps import az, pm
from .constants import BETA_EXACT, NU_EXACT, OMEGA_EXACT, TC_EXACT
from .datasets import get_dataset, list_datasets, resolve_paths
from .jax_config import configure_jax, jax_device_summary
from .model_fss import TC_DELTA_SCALE
from .model_registry import build_pipeline_model
from .model_scaling_nu import TC_PRIOR_LOWER, TC_PRIOR_UPPER
from .observables import (
    ensure_harada_bsa_observables_csv,
    read_observables_for_fss,
    validate_observables_df,
)
from .pytensor_numba import pymc_sample_compile_kwargs
from .sampling_guard import SamplingTimeGuard

SAMPLER_BACKEND = "jax"
LAPS_DEFAULT_CHAINS = 200
DEFAULT_CHAINS = 4


def _resolve_chains(chains: int | None, sampler_backend: str) -> int:
    if chains is not None:
        return chains
    return LAPS_DEFAULT_CHAINS if sampler_backend == "laps" else DEFAULT_CHAINS


def _trace_var_names(idata: az.InferenceData) -> list[str]:
    names = [
        "T_c",
        "nu",
        "beta",
        "gamma",
        "disc_t0",
        "disc_L0",
        "disc_p",
        "disc_q",
        "disc_sigma_model",
    ]
    return [name for name in names if name in idata.posterior]


def _profile_config_from_fit_kwargs(
    *,
    use_m: bool,
    use_m2: bool,
    use_m4: bool,
    use_binder: bool,
    use_chi: bool,
    correction_m: bool,
    correction_m2: bool,
    correction_m4: bool,
    correction_binder: bool,
    correction_chi: bool,
    use_log_m: bool,
    gp_ell_factor: float | None,
    gp_eta: float | None,
    correction_gp_ell_factor: float | None,
    correction_gp_eta: float | None,
    gp_kernel: str | None,
    universal_kernel: str | None = None,
    max_abs_z: float | None = None,
    obs_sigma_scale: float | None,
    infer_Tc: bool = True,
    infer_nu: bool = True,
    infer_beta: bool = True,
    reparametrize_Tc: bool = False,
    tc_delta_scale: float | None = None,
):
    """Build profile-likelihood config matching the PyMC model (lazy import)."""
    from .model_fss import TC_DELTA_SCALE
    from .profile_likelihood import FssProfileConfig

    kwargs: dict[str, object] = {
        "use_m": use_m,
        "use_m2": use_m2,
        "use_m4": use_m4,
        "use_binder": use_binder,
        "use_chi": use_chi,
        "correction_m": correction_m,
        "correction_m2": correction_m2,
        "correction_m4": correction_m4,
        "correction_binder": correction_binder,
        "correction_chi": correction_chi,
        "use_log_m": use_log_m,
        "infer_Tc": infer_Tc,
        "infer_nu": infer_nu,
        "infer_beta": infer_beta,
        "reparametrize_Tc": reparametrize_Tc,
        "tc_delta_scale": TC_DELTA_SCALE if tc_delta_scale is None else tc_delta_scale,
    }
    if gp_ell_factor is not None:
        kwargs["gp_ell_factor"] = gp_ell_factor
    if gp_eta is not None:
        kwargs["gp_eta"] = gp_eta
    if correction_gp_ell_factor is not None:
        kwargs["correction_gp_ell_factor"] = correction_gp_ell_factor
    if correction_gp_eta is not None:
        kwargs["correction_gp_eta"] = correction_gp_eta
    if gp_kernel is not None:
        from .gp_kernels import normalize_gp_kernel

        kwargs["gp_kernel"] = normalize_gp_kernel(gp_kernel)
    if universal_kernel is not None:
        from .gp_kernels import normalize_gp_kernel

        kwargs["universal_kernel"] = normalize_gp_kernel(universal_kernel)
    if max_abs_z is not None:
        kwargs["max_abs_z"] = float(max_abs_z)
    if obs_sigma_scale is not None:
        kwargs["obs_sigma_scale"] = obs_sigma_scale
    return FssProfileConfig(**kwargs)


def _profile_mle_initvals(
    *,
    data_path: Path,
    dataset: str,
    infer_Tc: bool,
    infer_nu: bool,
    infer_beta: bool,
    profile_config,
    chains: int,
    seed: int,
    reparametrize_Tc: bool = False,
    tc_reparam_ref: float | None = None,
    tc_delta_scale: float = TC_DELTA_SCALE,
) -> tuple[list[dict[str, float]], tuple[float, float, float, float]]:
    """Joint profile MLE starting points with small per-chain jitter (uses data)."""
    from .profile_likelihood import find_joint_mle_log_marginal_likelihood

    tc, nu, beta, log_ml = find_joint_mle_log_marginal_likelihood(
        data=data_path,
        dataset=dataset,
        config=profile_config,
    )
    rng = np.random.default_rng(seed)
    jitter = {
        "T_c": 2.0e-4,
        "nu": 0.02,
        "beta": 0.003,
    }
    tc_ref = TC_EXACT if tc_reparam_ref is None else float(tc_reparam_ref)
    per_chain: list[dict[str, float]] = []
    for _ in range(chains):
        point: dict[str, float] = {}
        if infer_Tc:
            tc_start = float(tc + rng.normal(0.0, jitter["T_c"]))
            if reparametrize_Tc:
                point["T_c_delta"] = (tc_start - tc_ref) * tc_delta_scale
            else:
                point["T_c"] = tc_start
        if infer_nu:
            point["nu"] = float(nu + rng.normal(0.0, jitter["nu"]))
        if infer_beta:
            point["beta"] = float(beta + rng.normal(0.0, jitter["beta"]))
        per_chain.append(point)
    return per_chain, (tc, nu, beta, log_ml)


def _profile_mle_jax_init_positions(
    *,
    per_chain_pymc: list[dict[str, float]],
    infer_Tc: bool,
    infer_nu: bool,
    infer_beta: bool,
    infer_gp_hyperparams: bool = False,
    uses_correction: bool = False,
    base_gp_ell: float = 1.0,
    base_gp_eta: float = 1.0,
    correction_gp_ell: float = 1.0,
    correction_gp_eta: float = 1.0,
    reparametrize_Tc: bool,
    tc_reparam_ref: float | None,
    tc_delta_scale: float,
) -> list:
    from .jax_inference import build_fss_inference_layout

    layout = build_fss_inference_layout(
        infer_Tc=infer_Tc,
        infer_nu=infer_nu,
        infer_beta=infer_beta,
        infer_gp_hyperparams=infer_gp_hyperparams,
        uses_correction=uses_correction,
        base_gp_ell=base_gp_ell,
        base_gp_eta=base_gp_eta,
        correction_gp_ell=correction_gp_ell,
        correction_gp_eta=correction_gp_eta,
        reparametrize_Tc=reparametrize_Tc,
        tc_reparam_ref=tc_reparam_ref,
        tc_delta_scale=tc_delta_scale,
    )
    tc_ref = TC_EXACT if tc_reparam_ref is None else float(tc_reparam_ref)
    positions = []
    for point in per_chain_pymc:
        tc = TC_EXACT
        nu = NU_EXACT
        beta = BETA_EXACT
        if infer_Tc:
            if reparametrize_Tc:
                tc = tc_ref + float(point["T_c_delta"]) / tc_delta_scale
            else:
                tc = float(point["T_c"])
        if infer_nu:
            nu = float(point["nu"])
        if infer_beta:
            beta = float(point["beta"])
        positions.append(
            layout.initial_unconstrained(
                T_c=tc,
                nu=nu,
                beta=beta,
                gp_ell=base_gp_ell,
                gp_eta=base_gp_eta,
                correction_gp_ell=correction_gp_ell,
                correction_gp_eta=correction_gp_eta,
            )
        )
    return positions


def _attach_common_posterior_attrs(
    idata: az.InferenceData,
    *,
    dataset: str,
    data_path: Path,
    method: str,
    sampler_backend: str,
    profile_mle_init: bool,
    profile_mle: tuple[float, float, float, float] | None,
    reparametrize_Tc: bool,
    tc_reparam_ref: float | None,
    tc_delta_scale: float,
    nuts_init: str | None,
) -> None:
    idata.posterior.attrs["dataset"] = dataset
    idata.posterior.attrs["model"] = "fss"
    idata.posterior.attrs["method"] = method
    idata.posterior.attrs["variant"] = "plain"
    idata.posterior.attrs["scaling"] = "gp"
    idata.posterior.attrs["observables_path"] = str(data_path)
    idata.posterior.attrs["sampler_backend"] = sampler_backend
    idata.posterior.attrs["profile_mle_init"] = str(profile_mle_init)
    if nuts_init is not None:
        idata.posterior.attrs["nuts_init"] = nuts_init
    idata.posterior.attrs["reparametrize_Tc"] = str(reparametrize_Tc)
    if reparametrize_Tc:
        tc_ref = TC_EXACT if tc_reparam_ref is None else tc_reparam_ref
        idata.posterior.attrs["tc_reparam_ref"] = str(tc_ref)
        idata.posterior.attrs["tc_delta_scale"] = str(tc_delta_scale)
    if profile_mle is not None:
        tc, nu, beta, log_ml = profile_mle
        idata.posterior.attrs["profile_mle_T_c"] = str(tc)
        idata.posterior.attrs["profile_mle_nu"] = str(nu)
        idata.posterior.attrs["profile_mle_beta"] = str(beta)
        idata.posterior.attrs["profile_mle_log_ml"] = str(log_ml)


def _attach_fss_fit_attrs(
    idata: az.InferenceData,
    *,
    observables: pd.DataFrame,
    profile_config,
    infer_Tc: bool,
    infer_nu: bool,
    infer_beta: bool,
    infer_gp_hyperparams: bool = False,
) -> None:
    """Mirror PyMC posterior attrs so plotting works for JAX/LAPS traces."""
    from .fss_likelihood import extract_likelihood_arrays, fss_gp_scales

    cfg = profile_config
    attrs = idata.posterior.attrs
    attrs["infer_Tc"] = str(infer_Tc)
    attrs["infer_nu"] = str(infer_nu)
    attrs["infer_beta"] = str(infer_beta)
    attrs["infer_gp_hyperparams"] = str(infer_gp_hyperparams)
    attrs["fss_m"] = str(cfg.use_m)
    attrs["fss_m2"] = str(cfg.use_m2)
    attrs["fss_m4"] = str(cfg.use_m4)
    attrs["fss_binder"] = str(cfg.use_binder)
    attrs["fss_chi"] = str(cfg.use_chi)
    attrs["fss_correction_m"] = str(cfg.correction_m)
    attrs["fss_correction_m2"] = str(cfg.correction_m2)
    attrs["fss_correction_m4"] = str(cfg.correction_m4)
    attrs["fss_correction_binder"] = str(cfg.correction_binder)
    attrs["fss_correction_chi"] = str(cfg.correction_chi)
    attrs["fss_discrepancy_m"] = str(cfg.discrepancy_m)
    attrs["fss_discrepancy_m2"] = str(cfg.discrepancy_m2)
    attrs["fss_discrepancy_m4"] = str(cfg.discrepancy_m4)
    attrs["fss_discrepancy_binder"] = str(cfg.discrepancy_binder)
    attrs["fss_discrepancy_chi"] = str(cfg.discrepancy_chi)
    attrs["discrepancy_form"] = str(getattr(cfg, "discrepancy_form", "noise"))
    attrs["use_log_m"] = str(cfg.use_log_m)
    attrs["obs_sigma_scale"] = str(cfg.obs_sigma_scale)
    attrs["gp_ell_factor"] = str(cfg.gp_ell_factor)
    attrs["gp_eta"] = str(cfg.gp_eta)
    attrs["correction_gp_ell_factor"] = str(cfg.correction_gp_ell_factor)
    attrs["correction_gp_eta"] = str(cfg.correction_gp_eta)
    attrs["gp_kernel"] = str(cfg.gp_kernel)
    attrs["universal_kernel"] = str(
        cfg.universal_kernel if cfg.universal_kernel is not None else cfg.gp_kernel
    )
    if getattr(cfg, "max_abs_z", None) is not None:
        attrs["max_abs_z"] = str(cfg.max_abs_z)
    attrs["omega_fixed"] = str(cfg.omega_fixed)
    any_correction = (
        (cfg.use_m and cfg.correction_m)
        or (cfg.use_m2 and cfg.correction_m2)
        or (cfg.use_m4 and cfg.correction_m4)
        or (cfg.use_binder and cfg.correction_binder)
        or (cfg.use_chi and cfg.correction_chi)
    )
    if any_correction:
        attrs["fss_correction"] = "True"
    arrays = extract_likelihood_arrays(observables, cfg)
    scales = fss_gp_scales(
        arrays.L,
        arrays.T,
        gp_ell_factor=cfg.gp_ell_factor,
        gp_eta=cfg.gp_eta,
        correction_gp_ell_factor=cfg.correction_gp_ell_factor,
        correction_gp_eta=cfg.correction_gp_eta,
        gp_ell=getattr(cfg, "gp_ell", None),
        correction_gp_ell=getattr(cfg, "correction_gp_ell", None),
    )
    attrs["gp_ell_fixed"] = str(scales.base_gp_ell)
    attrs["gp_eta_fixed"] = str(scales.base_gp_eta)
    if infer_gp_hyperparams:
        import math

        attrs["gp_ell_prior_mu"] = str(math.log(scales.base_gp_ell))
        if any_correction:
            attrs["correction_gp_ell_prior_mu"] = str(math.log(scales.correction_gp_ell))
    if any_correction:
        attrs["correction_gp_ell_fixed"] = str(scales.correction_gp_ell)
        attrs["correction_gp_eta_fixed"] = str(scales.correction_gp_eta)


def summarize_posterior(idata: az.InferenceData) -> pd.DataFrame:
    var_names = _trace_var_names(idata)
    if not var_names:
        print("No sampled variables to summarize.")
        return pd.DataFrame()
    summary = az.summary(idata, var_names=var_names, kind="stats", hdi_prob=0.94)
    print(summary)
    fixed = []
    if "T_c" not in idata.posterior:
        fixed.append(f"T_c={TC_EXACT:.6f}")
    if "nu" not in idata.posterior:
        fixed.append(f"nu={NU_EXACT:.6f}")
    if "beta" not in idata.posterior:
        fixed.append(f"beta={BETA_EXACT:.6f}")
    if fixed:
        print("\nFixed exponents:", " ".join(fixed))
    return summary


def fit_dataset(
    dataset: str,
    *,
    infer_Tc: bool = False,
    infer_nu: bool = True,
    infer_beta: bool = False,
    infer_omega: bool = False,
    infer_gp_hyperparams: bool = False,
    use_m: bool = True,
    use_m2: bool = False,
    use_m4: bool = False,
    use_binder: bool = False,
    use_chi: bool = False,
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
    discrepancy_form: str | None = None,
    infer_discrepancy: bool | None = None,
    infer_disc_t0: bool | None = None,
    infer_disc_L0: bool | None = None,
    infer_disc_p: bool | None = None,
    infer_disc_q: bool | None = None,
    infer_disc_sigma_model: bool | None = None,
    fixed_disc_t0: float | None = None,
    fixed_disc_L0: float | None = None,
    fixed_disc_p: float | None = None,
    fixed_disc_q: float | None = None,
    fixed_disc_sigma_model: float | None = None,
    draws: int = 500,
    tune: int = 500,
    chains: int | None = None,
    cores: int | None = None,
    target_accept: float = 0.9,
    seed: int = 1,
    max_eta_hours: float | None = 1.0,
    data: Path | None = None,
    out: Path | None = None,
    gp_ell_factor: float | None = None,
    gp_ell: float | None = None,
    gp_eta: float | None = None,
    correction_gp_ell_factor: float | None = None,
    correction_gp_ell: float | None = None,
    correction_gp_eta: float | None = None,
    gp_kernel: str | None = None,
    universal_kernel: str | None = None,
    max_abs_z: float | None = None,
    obs_sigma_scale: float | None = None,
    use_log_m: bool = True,
    profile_mle_init: bool = False,
    nuts_init: str | None = None,
    reparametrize_Tc: bool = False,
    tc_reparam_ref: float | None = None,
    tc_delta_scale: float = TC_DELTA_SCALE,
    tc_prior_lower: float | None = None,
    tc_prior_upper: float | None = None,
    tc_init: float | None = None,
    sampler_backend: str = SAMPLER_BACKEND,
    jax_platform: str | None = None,
    # Accepted for backward compatibility with older call sites; ignored.
    model: str = "fss",
    method: str = "marginal",
    variant: str = "plain",
    scaling: str = "gp",
    **_ignored: object,
) -> Path:
    if model != "fss":
        raise ValueError(f"Only model='fss' is supported, got {model!r}")

    spec = get_dataset(dataset)
    data_path, posterior_out, _ = resolve_paths(
        dataset,
        model="fss",
        method=method,
        variant="plain",
        scaling="gp",
        data=data,
        posterior=out,
    )
    if dataset == "harada_bsa":
        data_path = ensure_harada_bsa_observables_csv(data_path)
    elif not data_path.exists():
        raise FileNotFoundError(
            f"Observables not found at {data_path}. "
            f"Run: python -m ising.simulate --dataset {dataset}"
        )

    chains = _resolve_chains(chains, sampler_backend)
    cores = cores if cores is not None else min(chains, 4)
    tc_prior_lo = TC_PRIOR_LOWER if tc_prior_lower is None else float(tc_prior_lower)
    tc_prior_hi = TC_PRIOR_UPPER if tc_prior_upper is None else float(tc_prior_upper)
    if tc_prior_lo >= tc_prior_hi:
        raise ValueError(
            f"tc_prior_lower ({tc_prior_lo}) must be < tc_prior_upper ({tc_prior_hi})"
        )
    tc_init_eff = float(TC_EXACT if tc_init is None else tc_init)
    if infer_Tc and not (tc_prior_lo < tc_init_eff < tc_prior_hi):
        raise ValueError(
            f"tc_init={tc_init_eff} must lie strictly inside "
            f"({tc_prior_lo}, {tc_prior_hi})"
        )
    observables = read_observables_for_fss(
        data_path,
        use_m=use_m,
        use_m2=use_m2,
        use_m4=use_m4,
        use_binder=use_binder,
        use_chi=use_chi,
    )
    if not (use_m or use_m2 or use_m4 or use_binder or use_chi):
        raise ValueError(
            "At least one of use_m, use_m2, use_m4, use_binder, or use_chi must be True"
        )
    if sampler_backend not in ("jax", "laps", "pymc"):
        raise ValueError(
            f"sampler_backend must be 'jax', 'laps', or 'pymc', got {sampler_backend!r}"
        )
    if sampler_backend in ("jax", "laps") and infer_omega:
        raise ValueError(
            "JAX/LAPS samplers do not support infer_omega; use --sampler-backend pymc "
            "for infer_omega"
        )
    uses_discrepancy = (
        (use_m and discrepancy_m)
        or (use_m2 and discrepancy_m2)
        or (use_m4 and discrepancy_m4)
        or (use_binder and discrepancy_binder)
        or (use_chi and discrepancy_chi)
    )
    _ = uses_discrepancy  # allowed for JAX/LAPS

    configure_jax(platform=jax_platform)  # type: ignore[arg-type]

    if sampler_backend in ("jax", "laps"):
        from .jax_inference import profile_config_from_fit_kwargs

        if sampler_backend == "laps":
            from .jax_inference import sample_fss_posterior_laps
        else:
            from .jax_inference import sample_fss_posterior_jax

        profile_config = profile_config_from_fit_kwargs(
            use_m=use_m,
            use_m2=use_m2,
            use_m4=use_m4,
            use_binder=use_binder,
            use_chi=use_chi,
            correction_m=correction_m,
            correction_m2=correction_m2,
            correction_m4=correction_m4,
            correction_binder=correction_binder,
            correction_chi=correction_chi,
            discrepancy_m=discrepancy_m,
            discrepancy_m2=discrepancy_m2,
            discrepancy_m4=discrepancy_m4,
            discrepancy_binder=discrepancy_binder,
            discrepancy_chi=discrepancy_chi,
            discrepancy_form=discrepancy_form,
            use_log_m=use_log_m,
            gp_ell_factor=gp_ell_factor,
            gp_ell=gp_ell,
            gp_eta=gp_eta,
            correction_gp_ell_factor=correction_gp_ell_factor,
            correction_gp_ell=correction_gp_ell,
            correction_gp_eta=correction_gp_eta,
            gp_kernel=gp_kernel,
            universal_kernel=universal_kernel,
            max_abs_z=max_abs_z,
            obs_sigma_scale=obs_sigma_scale,
            infer_Tc=infer_Tc,
            infer_nu=infer_nu,
            infer_beta=infer_beta,
            reparametrize_Tc=reparametrize_Tc,
            tc_delta_scale=tc_delta_scale,
        )
        print(
            f"Dataset {dataset!r} ({spec.n_points} points, {len(observables)} rows), "
            f"model='fss', sampler={sampler_backend!r}, {chains} chains, "
            f"tune={tune}, draws={draws}"
            + (f", target_accept={target_accept}" if sampler_backend == "jax" else "")
        )
        print(f"  JAX: {jax_device_summary()}")
        print(
            f"  infer_Tc={infer_Tc}, infer_nu={infer_nu}, infer_beta={infer_beta}, "
            f"infer_gp_hyperparams={infer_gp_hyperparams}, "
            f"use_m={use_m}, use_m2={use_m2}, use_m4={use_m4}, "
            f"use_binder={use_binder}, use_chi={use_chi}, reparametrize_Tc={reparametrize_Tc}"
        )
        if infer_Tc:
            print(
                f"  T_c prior U({tc_prior_lo:g}, {tc_prior_hi:g}), "
                f"tc_init={tc_init_eff:g}"
            )

        from .fss_likelihood import extract_likelihood_arrays, fss_gp_scales, uses_fss_correction

        likelihood_arrays = extract_likelihood_arrays(observables, profile_config)
        gp_scales = fss_gp_scales(
            likelihood_arrays.L,
            likelihood_arrays.T,
            gp_ell_factor=profile_config.gp_ell_factor,
            gp_eta=profile_config.gp_eta,
            correction_gp_ell_factor=profile_config.correction_gp_ell_factor,
            correction_gp_eta=profile_config.correction_gp_eta,
            gp_ell=getattr(profile_config, "gp_ell", None),
            correction_gp_ell=getattr(profile_config, "correction_gp_ell", None),
        )
        uses_correction = uses_fss_correction(profile_config)

        jax_init_positions = None
        laps_mle_init: tuple[float, float, float] | None = None
        profile_mle: tuple[float, float, float, float] | None = None
        nuts_init_label = nuts_init
        if profile_mle_init:
            sample_initvals, profile_mle = _profile_mle_initvals(
                data_path=data_path,
                dataset=dataset,
                infer_Tc=infer_Tc,
                infer_nu=infer_nu,
                infer_beta=infer_beta,
                profile_config=profile_config,
                chains=chains,
                seed=seed,
                reparametrize_Tc=reparametrize_Tc,
                tc_reparam_ref=tc_reparam_ref,
                tc_delta_scale=tc_delta_scale,
            )
            tc, nu, beta, log_ml = profile_mle
            laps_mle_init = (tc, nu, beta)
            if sampler_backend == "jax":
                jax_init_positions = _profile_mle_jax_init_positions(
                    per_chain_pymc=sample_initvals,
                    infer_Tc=infer_Tc,
                    infer_nu=infer_nu,
                    infer_beta=infer_beta,
                    infer_gp_hyperparams=infer_gp_hyperparams,
                    uses_correction=uses_correction,
                    base_gp_ell=gp_scales.base_gp_ell,
                    base_gp_eta=gp_scales.base_gp_eta,
                    correction_gp_ell=gp_scales.correction_gp_ell,
                    correction_gp_eta=gp_scales.correction_gp_eta,
                    reparametrize_Tc=reparametrize_Tc,
                    tc_reparam_ref=tc_reparam_ref,
                    tc_delta_scale=tc_delta_scale,
                )
            print(
                "  WARNING: profile_mle_init uses data-derived starting points "
                "(remove before production runs)"
            )
            print(
                f"  {sampler_backend} init from profile MLE: "
                f"T_c={tc:.8g}, nu={nu:.6g}, beta={beta:.6g}, log p={log_ml:.4g}"
            )
            nuts_init_label = nuts_init if nuts_init is not None else "profile_mle"

        from .discrepancy import (
            DISCREPANCY_L0_INIT,
            DISCREPANCY_P_INIT,
            DISCREPANCY_Q_INIT,
            DISCREPANCY_SIGMA_MODEL_INIT,
            DISCREPANCY_T0_INIT,
        )

        disc_fixed = dict(
            infer_discrepancy=infer_discrepancy,
            infer_disc_t0=infer_disc_t0,
            infer_disc_L0=infer_disc_L0,
            infer_disc_p=infer_disc_p,
            infer_disc_q=infer_disc_q,
            infer_disc_sigma_model=infer_disc_sigma_model,
            fixed_disc_t0=(
                DISCREPANCY_T0_INIT if fixed_disc_t0 is None else float(fixed_disc_t0)
            ),
            fixed_disc_L0=(
                DISCREPANCY_L0_INIT if fixed_disc_L0 is None else float(fixed_disc_L0)
            ),
            fixed_disc_p=(
                DISCREPANCY_P_INIT if fixed_disc_p is None else float(fixed_disc_p)
            ),
            fixed_disc_q=(
                DISCREPANCY_Q_INIT if fixed_disc_q is None else float(fixed_disc_q)
            ),
            fixed_disc_sigma_model=(
                DISCREPANCY_SIGMA_MODEL_INIT
                if fixed_disc_sigma_model is None
                else float(fixed_disc_sigma_model)
            ),
        )

        if sampler_backend == "laps":
            idata = sample_fss_posterior_laps(
                observables,
                profile_config,
                infer_Tc=infer_Tc,
                infer_nu=infer_nu,
                infer_beta=infer_beta,
                infer_gp_hyperparams=infer_gp_hyperparams,
                tune=tune,
                draws=draws,
                chains=chains,
                seed=seed,
                reparametrize_Tc=reparametrize_Tc,
                tc_reparam_ref=tc_reparam_ref,
                tc_delta_scale=tc_delta_scale,
                tc_prior_lower=tc_prior_lo,
                tc_prior_upper=tc_prior_hi,
                tc_init=tc_init_eff,
                fixed_Tc=tc_init_eff,
                mle_init=laps_mle_init,
                progress_bar=True,
                **disc_fixed,
            )
        else:
            idata = sample_fss_posterior_jax(
                observables,
                profile_config,
                infer_Tc=infer_Tc,
                infer_nu=infer_nu,
                infer_beta=infer_beta,
                infer_gp_hyperparams=infer_gp_hyperparams,
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                seed=seed,
                reparametrize_Tc=reparametrize_Tc,
                tc_reparam_ref=tc_reparam_ref,
                tc_delta_scale=tc_delta_scale,
                tc_prior_lower=tc_prior_lo,
                tc_prior_upper=tc_prior_hi,
                tc_init=tc_init_eff,
                fixed_Tc=tc_init_eff,
                init_positions=jax_init_positions,
                progress_bar=False,
                **disc_fixed,
            )
        _attach_common_posterior_attrs(
            idata,
            dataset=dataset,
            data_path=data_path,
            method=method,
            sampler_backend=sampler_backend,
            profile_mle_init=profile_mle_init,
            profile_mle=profile_mle,
            reparametrize_Tc=reparametrize_Tc,
            tc_reparam_ref=tc_reparam_ref,
            tc_delta_scale=tc_delta_scale,
            nuts_init=nuts_init_label,
        )
        _attach_fss_fit_attrs(
            idata,
            observables=observables,
            profile_config=profile_config,
            infer_Tc=infer_Tc,
            infer_nu=infer_nu,
            infer_beta=infer_beta,
            infer_gp_hyperparams=infer_gp_hyperparams,
        )
        summarize_posterior(idata)
        posterior_out.parent.mkdir(parents=True, exist_ok=True)
        idata.to_netcdf(posterior_out)
        print(f"\nSaved posterior to {posterior_out}")
        return posterior_out

    build_kwargs = dict(
        infer_Tc=infer_Tc,
        infer_nu=infer_nu,
        infer_beta=infer_beta,
        infer_omega=infer_omega,
        infer_gp_hyperparams=infer_gp_hyperparams,
        use_m=use_m,
        use_m2=use_m2,
        use_m4=use_m4,
        use_binder=use_binder,
        use_chi=use_chi,
        correction_m=correction_m,
        correction_m2=correction_m2,
        correction_m4=correction_m4,
        correction_binder=correction_binder,
        correction_chi=correction_chi,
        discrepancy_m=discrepancy_m,
        discrepancy_m2=discrepancy_m2,
        discrepancy_m4=discrepancy_m4,
        discrepancy_binder=discrepancy_binder,
        discrepancy_chi=discrepancy_chi,
        use_log_m=use_log_m,
        reparametrize_Tc=reparametrize_Tc,
    )
    if tc_reparam_ref is not None:
        build_kwargs["tc_reparam_ref"] = tc_reparam_ref
    if tc_delta_scale != TC_DELTA_SCALE:
        build_kwargs["tc_delta_scale"] = tc_delta_scale
    if tc_prior_lower is not None:
        build_kwargs["tc_prior_lower"] = tc_prior_lo
    if tc_prior_upper is not None:
        build_kwargs["tc_prior_upper"] = tc_prior_hi
    if gp_ell_factor is not None:
        build_kwargs["gp_ell_factor"] = gp_ell_factor
    if gp_eta is not None:
        build_kwargs["gp_eta"] = gp_eta
    if correction_gp_ell_factor is not None:
        build_kwargs["correction_gp_ell_factor"] = correction_gp_ell_factor
    if correction_gp_eta is not None:
        build_kwargs["correction_gp_eta"] = correction_gp_eta
    if gp_kernel is not None:
        from .gp_kernels import normalize_gp_kernel

        build_kwargs["gp_kernel"] = normalize_gp_kernel(gp_kernel)
    if universal_kernel is not None:
        from .gp_kernels import normalize_gp_kernel

        build_kwargs["universal_kernel"] = normalize_gp_kernel(universal_kernel)
    if obs_sigma_scale is not None:
        build_kwargs["obs_sigma_scale"] = obs_sigma_scale
    pymc_model = build_pipeline_model(observables, **build_kwargs)

    print(
        f"Dataset {dataset!r} ({spec.n_points} points, {len(observables)} rows), "
        f"model='fss', sampler='pymc', "
        f"{chains} chains on {cores} core(s), "
        f"tune={tune}, draws={draws}, "
        f"target_accept={target_accept}"
    )
    print(
        f"  infer_Tc={infer_Tc}, infer_nu={infer_nu}, infer_beta={infer_beta}, "
        f"infer_omega={infer_omega}, infer_gp_hyperparams={infer_gp_hyperparams}, "
        f"use_m={use_m}, use_m2={use_m2}, use_m4={use_m4}, "
        f"use_binder={use_binder}, use_chi={use_chi}, "
        f"correction_m={correction_m}, correction_m2={correction_m2}, "
        f"correction_m4={correction_m4}, correction_binder={correction_binder}, "
        f"correction_chi={correction_chi}, use_log_m={use_log_m}, "
        f"reparametrize_Tc={reparametrize_Tc}"
    )
    if reparametrize_Tc:
        tc_ref = TC_EXACT if tc_reparam_ref is None else tc_reparam_ref
        print(
            f"  T_c = {tc_ref:.8g} + T_c_delta / {tc_delta_scale:g} "
            f"(delta prior matches uniform T_c prior)"
        )
    gp_bits = []
    if gp_ell_factor is not None:
        gp_bits.append(f"gp_ell_factor={gp_ell_factor:g}")
    if gp_eta is not None:
        gp_bits.append(f"gp_eta={gp_eta:g}")
    if correction_gp_ell_factor is not None:
        gp_bits.append(f"correction_gp_ell_factor={correction_gp_ell_factor:g}")
    if correction_gp_eta is not None:
        gp_bits.append(f"correction_gp_eta={correction_gp_eta:g}")
    if gp_kernel is not None:
        gp_bits.append(f"gp_kernel={gp_kernel}")
    if obs_sigma_scale is not None:
        gp_bits.append(f"obs_sigma_scale={obs_sigma_scale:g}")
    if gp_bits:
        print(f"  GP hyperparameters: {', '.join(gp_bits)}")
    if max_eta_hours is not None:
        print(f"  Abort if estimated remaining sampling time exceeds {max_eta_hours:.2f} h")

    sample_init = nuts_init if nuts_init is not None else "auto"
    sample_initvals: list[dict[str, float]] | None = None
    profile_mle: tuple[float, float, float, float] | None = None
    if profile_mle_init:
        # adapt_full (dense mass matrix); per-chain jitter is added below, not PyMC's ±1.
        sample_init = nuts_init if nuts_init is not None else "adapt_full"
        profile_config = _profile_config_from_fit_kwargs(
            use_m=use_m,
            use_m2=use_m2,
            use_m4=use_m4,
            use_binder=use_binder,
            use_chi=use_chi,
            correction_m=correction_m,
            correction_m2=correction_m2,
            correction_m4=correction_m4,
            correction_binder=correction_binder,
            correction_chi=correction_chi,
            use_log_m=use_log_m,
            gp_ell_factor=gp_ell_factor,
            gp_eta=gp_eta,
            correction_gp_ell_factor=correction_gp_ell_factor,
            correction_gp_eta=correction_gp_eta,
            gp_kernel=gp_kernel,
            universal_kernel=universal_kernel,
            max_abs_z=max_abs_z,
            obs_sigma_scale=obs_sigma_scale,
            infer_Tc=infer_Tc,
            infer_nu=infer_nu,
            infer_beta=infer_beta,
            reparametrize_Tc=reparametrize_Tc,
            tc_delta_scale=tc_delta_scale,
        )
        sample_initvals, profile_mle = _profile_mle_initvals(
            data_path=data_path,
            dataset=dataset,
            infer_Tc=infer_Tc,
            infer_nu=infer_nu,
            infer_beta=infer_beta,
            profile_config=profile_config,
            chains=chains,
            seed=seed,
            reparametrize_Tc=reparametrize_Tc,
            tc_reparam_ref=tc_reparam_ref,
            tc_delta_scale=tc_delta_scale,
        )
        if not sample_initvals[0]:
            raise ValueError(
                "profile_mle_init requires at least one of infer_Tc, infer_nu, infer_beta"
            )
        tc, nu, beta, log_ml = profile_mle
        print(
            "  WARNING: profile_mle_init uses data-derived starting points "
            "(remove before production runs)"
        )
        print(
            f"  NUTS init={sample_init!r}, profile MLE center: "
            f"T_c={tc:.8g}, nu={nu:.6g}, beta={beta:.6g}, log p={log_ml:.4g}; "
            f"per-chain jitter (T_c, nu, beta) = (2e-4, 0.02, 0.003)"
            + (
                f" on T_c_delta (scale {tc_delta_scale:g})"
                if reparametrize_Tc
                else ""
            )
        )
    elif nuts_init is not None:
        print(f"  NUTS init={sample_init!r}")

    callback = None
    if max_eta_hours is not None:
        callback = SamplingTimeGuard(
            chains=chains,
            tune=tune,
            draws=draws,
            max_eta_seconds=max_eta_hours * 3600.0,
        )

    with pymc_model:
        try:
            sample_kwargs: dict[str, object] = {
                "draws": draws,
                "tune": tune,
                "chains": chains,
                "cores": cores,
                "target_accept": target_accept,
                "random_seed": seed,
                "progressbar": True,
                "callback": callback,
                "init": sample_init,
                "compile_kwargs": pymc_sample_compile_kwargs(),
            }
            if sample_initvals is not None:
                sample_kwargs["initvals"] = sample_initvals
            idata = pm.sample(**sample_kwargs)
        except KeyboardInterrupt as exc:
            if callback is not None and callback.completed >= callback._eta_check_start:
                raise RuntimeError(str(exc)) from exc
            raise
        except ValueError as exc:
            if "Not enough samples to build a trace" in str(exc):
                raise RuntimeError(
                    "Sampling stopped before collecting draws (often caused by "
                    "the 1 h time guard firing during early tuning). Re-run with "
                    "max_eta_hours=0 or a larger limit."
                ) from exc
            raise
        idata.posterior.attrs["dataset"] = dataset
        idata.posterior.attrs["model"] = "fss"
        idata.posterior.attrs["method"] = method
        idata.posterior.attrs["variant"] = "plain"
        idata.posterior.attrs["scaling"] = "gp"
        idata.posterior.attrs["observables_path"] = str(data_path)
        idata.posterior.attrs["sampler_backend"] = "pymc"
        idata.posterior.attrs["nuts_init"] = sample_init
        idata.posterior.attrs["profile_mle_init"] = str(profile_mle_init)
        reparam_tc = getattr(pymc_model, "_ising_reparametrize_Tc", None)
        if reparam_tc is not None:
            idata.posterior.attrs["reparametrize_Tc"] = str(reparam_tc)
            if reparam_tc:
                tc_ref_attr = getattr(pymc_model, "_ising_tc_reparam_ref", None)
                tc_scale_attr = getattr(pymc_model, "_ising_tc_delta_scale", None)
                if tc_ref_attr is not None:
                    idata.posterior.attrs["tc_reparam_ref"] = str(tc_ref_attr)
                if tc_scale_attr is not None:
                    idata.posterior.attrs["tc_delta_scale"] = str(tc_scale_attr)
        if profile_mle is not None:
            tc, nu, beta, log_ml = profile_mle
            idata.posterior.attrs["profile_mle_T_c"] = str(tc)
            idata.posterior.attrs["profile_mle_nu"] = str(nu)
            idata.posterior.attrs["profile_mle_beta"] = str(beta)
            idata.posterior.attrs["profile_mle_log_ml"] = str(log_ml)
        z_scale = getattr(pymc_model, "_ising_z_scale", None)
        if z_scale is not None:
            idata.posterior.attrs["z_scale"] = str(z_scale)
        tc_sigma = getattr(pymc_model, "_ising_tc_prior_sigma", None)
        if tc_sigma is not None:
            idata.posterior.attrs["tc_prior_sigma"] = str(tc_sigma)
        tc_lo = getattr(pymc_model, "_ising_tc_prior_lower", None)
        tc_hi = getattr(pymc_model, "_ising_tc_prior_upper", None)
        if tc_lo is not None and tc_hi is not None:
            idata.posterior.attrs["tc_prior_lower"] = str(tc_lo)
            idata.posterior.attrs["tc_prior_upper"] = str(tc_hi)
        nu_lo = getattr(pymc_model, "_ising_nu_prior_lower", None)
        nu_hi = getattr(pymc_model, "_ising_nu_prior_upper", None)
        if nu_lo is not None and nu_hi is not None:
            idata.posterior.attrs["nu_prior_lower"] = str(nu_lo)
            idata.posterior.attrs["nu_prior_upper"] = str(nu_hi)
        beta_lo = getattr(pymc_model, "_ising_beta_prior_lower", None)
        beta_hi = getattr(pymc_model, "_ising_beta_prior_upper", None)
        if beta_lo is not None and beta_hi is not None:
            idata.posterior.attrs["beta_prior_lower"] = str(beta_lo)
            idata.posterior.attrs["beta_prior_upper"] = str(beta_hi)
        beta_mu = getattr(pymc_model, "_ising_beta_prior_mu", None)
        beta_sigma = getattr(pymc_model, "_ising_beta_prior_sigma", None)
        if beta_mu is not None:
            idata.posterior.attrs["beta_prior_mu"] = str(beta_mu)
        if beta_sigma is not None:
            idata.posterior.attrs["beta_prior_sigma"] = str(beta_sigma)
        nu_mu = getattr(pymc_model, "_ising_nu_prior_mu", None)
        nu_sigma = getattr(pymc_model, "_ising_nu_prior_sigma", None)
        if nu_mu is not None:
            idata.posterior.attrs["nu_prior_mu"] = str(nu_mu)
        if nu_sigma is not None:
            idata.posterior.attrs["nu_prior_sigma"] = str(nu_sigma)
        fss_correction = getattr(pymc_model, "_ising_fss_correction", None)
        if fss_correction is not None:
            idata.posterior.attrs["fss_correction"] = str(fss_correction)
            idata.posterior.attrs["binder_correction"] = str(fss_correction)
        for attr, key in (
            ("_ising_fss_correction_m", "fss_correction_m"),
            ("_ising_fss_correction_m2", "fss_correction_m2"),
            ("_ising_fss_correction_m4", "fss_correction_m4"),
            ("_ising_fss_correction_binder", "fss_correction_binder"),
            ("_ising_fss_correction_chi", "fss_correction_chi"),
        ):
            val = getattr(pymc_model, attr, None)
            if val is not None:
                idata.posterior.attrs[key] = str(val)
        include_binder = getattr(pymc_model, "_ising_include_binder", None)
        if include_binder is not None:
            idata.posterior.attrs["fss_binder"] = str(include_binder)
        include_m = getattr(pymc_model, "_ising_include_m", None)
        if include_m is not None:
            idata.posterior.attrs["fss_m"] = str(include_m)
        include_m2 = getattr(pymc_model, "_ising_include_m2", None)
        if include_m2 is not None:
            idata.posterior.attrs["fss_m2"] = str(include_m2)
        include_m4 = getattr(pymc_model, "_ising_include_m4", None)
        if include_m4 is not None:
            idata.posterior.attrs["fss_m4"] = str(include_m4)
        include_chi = getattr(pymc_model, "_ising_include_chi", None)
        if include_chi is not None:
            idata.posterior.attrs["fss_chi"] = str(include_chi)
        for attr_name, key in (
            ("_ising_infer_Tc", "infer_Tc"),
            ("_ising_infer_nu", "infer_nu"),
            ("_ising_infer_beta", "infer_beta"),
            ("_ising_infer_omega", "infer_omega"),
            ("_ising_infer_gp_hyperparams", "infer_gp_hyperparams"),
        ):
            val = getattr(pymc_model, attr_name, None)
            if val is not None:
                idata.posterior.attrs[key] = str(val)
        omega_lo = getattr(pymc_model, "_ising_omega_prior_lower", None)
        omega_hi = getattr(pymc_model, "_ising_omega_prior_upper", None)
        if omega_lo is not None and omega_hi is not None:
            idata.posterior.attrs["omega_prior_lower"] = str(omega_lo)
            idata.posterior.attrs["omega_prior_upper"] = str(omega_hi)
        omega_fixed = getattr(pymc_model, "_ising_omega_fixed", None)
        if omega_fixed is not None:
            idata.posterior.attrs["omega_fixed"] = str(omega_fixed)
        gp_ell_fixed = getattr(pymc_model, "_ising_gp_ell_fixed", None)
        if gp_ell_fixed is not None:
            idata.posterior.attrs["gp_ell_fixed"] = str(gp_ell_fixed)
        gp_eta_fixed = getattr(pymc_model, "_ising_gp_eta_fixed", None)
        if gp_eta_fixed is not None:
            idata.posterior.attrs["gp_eta_fixed"] = str(gp_eta_fixed)
        correction_gp_ell_fixed = getattr(
            pymc_model, "_ising_correction_gp_ell_fixed", None
        )
        if correction_gp_ell_fixed is not None:
            idata.posterior.attrs["correction_gp_ell_fixed"] = str(
                correction_gp_ell_fixed
            )
        correction_gp_eta_fixed = getattr(
            pymc_model, "_ising_correction_gp_eta_fixed", None
        )
        if correction_gp_eta_fixed is not None:
            idata.posterior.attrs["correction_gp_eta_fixed"] = str(
                correction_gp_eta_fixed
            )
        for attr, key in (
            ("_ising_gp_ell_prior_mu", "gp_ell_prior_mu"),
            ("_ising_gp_ell_prior_sigma", "gp_ell_prior_sigma"),
            ("_ising_correction_gp_ell_prior_mu", "correction_gp_ell_prior_mu"),
            ("_ising_gp_ell_factor", "gp_ell_factor"),
            ("_ising_correction_gp_ell_factor", "correction_gp_ell_factor"),
            ("_ising_gp_kernel", "gp_kernel"),
            ("_ising_obs_sigma_scale", "obs_sigma_scale"),
            ("_ising_use_log_m", "use_log_m"),
        ):
            val = getattr(pymc_model, attr, None)
            if val is not None:
                idata.posterior.attrs[key] = str(val)

    summarize_posterior(idata)
    posterior_out.parent.mkdir(parents=True, exist_ok=True)
    idata.to_netcdf(posterior_out)
    print(f"\nSaved posterior to {posterior_out}")
    return posterior_out


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=list_datasets(),
        default="medium",
        help="Named observables preset",
    )
    parser.add_argument(
        "--harada-bsa",
        action="store_true",
        help="Use published Harada BSA Binder data (sets --dataset harada_bsa)",
    )
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--infer-Tc",
        action="store_true",
        help="Infer T_c (uniform prior)",
    )
    parser.add_argument(
        "--infer-nu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Infer nu (default: infer)",
    )
    parser.add_argument(
        "--infer-beta",
        action="store_true",
        help="Infer beta",
    )
    parser.add_argument(
        "--infer-omega",
        action="store_true",
        help="Infer correction exponent omega (requires a correction channel)",
    )
    parser.add_argument(
        "--infer-gp-hyperparams",
        action="store_true",
        help=(
            "Infer GP length scales and amplitudes (gp_ell, gp_eta, ...) "
            "instead of pinning ℓ_f by hand. Recommended: a fixed ℓ_f "
            "partly identifies ν."
        ),
    )
    parser.add_argument(
        "--use-m",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include magnetization |m| in the FSS likelihood",
    )
    parser.add_argument(
        "--use-m2",
        action="store_true",
        help="Include mean m^2 in the FSS likelihood (Phi_m2 = m^2 L^(2 beta/nu))",
    )
    parser.add_argument(
        "--use-m4",
        action="store_true",
        help="Include mean m^4 in the FSS likelihood (Phi_m4 = m^4 L^(4 beta/nu))",
    )
    parser.add_argument(
        "--use-binder",
        action="store_true",
        help="Include Binder cumulant U_4 in the FSS likelihood",
    )
    parser.add_argument(
        "--use-chi",
        action="store_true",
        help="Include susceptibility chi in the FSS likelihood",
    )
    parser.add_argument(
        "--correction-m",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use L^-omega f1(z) correction for magnetization",
    )
    parser.add_argument(
        "--correction-m2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use L^-omega f1(z) correction for m^2",
    )
    parser.add_argument(
        "--correction-m4",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use L^-omega f1(z) correction for m^4",
    )
    parser.add_argument(
        "--correction-binder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use L^-omega f1(z) correction for Binder cumulant",
    )
    parser.add_argument(
        "--correction-chi",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use L^-omega f1(z) correction for susceptibility",
    )
    parser.add_argument(
        "--discrepancy-m",
        action="store_true",
        help="Enable discrepancy on magnetization channel",
    )
    parser.add_argument(
        "--discrepancy-m2",
        action="store_true",
        help="Enable discrepancy on m^2 channel",
    )
    parser.add_argument(
        "--discrepancy-m4",
        action="store_true",
        help="Enable discrepancy on m^4 channel",
    )
    parser.add_argument(
        "--discrepancy-binder",
        action="store_true",
        help="Enable discrepancy on Binder channel",
    )
    parser.add_argument(
        "--discrepancy-chi",
        action="store_true",
        help="Enable discrepancy on susceptibility channel",
    )
    parser.add_argument(
        "--discrepancy-form",
        choices=(
            "noise",
            "additive_gp",
            "additive_gp_z_threshold",
            "additive_gp_fss",
        ),
        default=None,
        help="Discrepancy model (default: noise)",
    )
    parser.add_argument(
        "--infer-discrepancy",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Sample discrepancy hyperparameters (default: yes if any discrepancy channel is on)",
    )
    parser.add_argument(
        "--infer-disc-t0",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override whether disc_t0 is sampled (default: follow --infer-discrepancy)",
    )
    parser.add_argument(
        "--infer-disc-L0",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override whether disc_L0 is sampled (default: follow --infer-discrepancy)",
    )
    parser.add_argument(
        "--infer-disc-p",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override whether disc_p is sampled (default: follow --infer-discrepancy)",
    )
    parser.add_argument(
        "--infer-disc-q",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override whether disc_q is sampled (default: follow --infer-discrepancy)",
    )
    parser.add_argument(
        "--infer-disc-sigma-model",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override whether disc_sigma_model is sampled (default: follow --infer-discrepancy)",
    )
    parser.add_argument("--fixed-disc-t0", type=float, default=None)
    parser.add_argument("--fixed-disc-L0", type=float, default=None)
    parser.add_argument("--fixed-disc-p", type=float, default=None)
    parser.add_argument("--fixed-disc-q", type=float, default=None)
    parser.add_argument(
        "--fixed-disc-sigma-model",
        type=float,
        default=None,
        help="Fixed sigma_model / sigma_g for discrepancy",
    )
    parser.add_argument(
        "--gp-ell",
        type=float,
        default=None,
        help="Absolute f0(z) GP length scale (overrides --gp-ell-factor)",
    )
    parser.add_argument(
        "--correction-gp-ell",
        type=float,
        default=None,
        help="Absolute f1(z) GP length scale (overrides --correction-gp-ell-factor)",
    )
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--tune", type=int, default=500)
    parser.add_argument(
        "--chains",
        type=int,
        default=None,
        help="Number of chains (default: 4 for jax/pymc, 200 for laps)",
    )
    parser.add_argument("--cores", type=int, default=None)
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--max-eta-hours",
        type=float,
        default=1.0,
        help="Abort sampling if projected remaining time exceeds this (0 disables)",
    )
    parser.add_argument(
        "--gp-ell-factor",
        type=float,
        default=None,
        help="f0(z) GP length scale as factor * z_span (larger => stiffer)",
    )
    parser.add_argument(
        "--gp-eta",
        type=float,
        default=None,
        help="f0(z) GP amplitude eta (smaller => stiffer)",
    )
    parser.add_argument(
        "--correction-gp-ell-factor",
        type=float,
        default=None,
        help="f1(z) GP length scale as factor * z_span (larger => stiffer)",
    )
    parser.add_argument(
        "--correction-gp-eta",
        type=float,
        default=None,
        help="f1(z) GP amplitude eta (smaller => stiffer)",
    )
    parser.add_argument(
        "--gp-kernel",
        choices=("matern52", "gaussian", "harada"),
        default=None,
        help=(
            "GP kernel on z: matern52 (default) or gaussian/harada "
            "(squared-exponential, Harada PRE 84)"
        ),
    )
    parser.add_argument(
        "--universal-kernel",
        choices=("matern52", "gaussian", "harada", "poly2", "poly", "poly4", "quartic"),
        default=None,
        help=(
            "Kernel for universal f_0(z); default follows --gp-kernel. "
            "Use poly2/poly4 for marginalized monomial universal functions."
        ),
    )
    parser.add_argument(
        "--max-abs-z",
        type=float,
        default=None,
        help=(
            "Keep only points with |z| <= this (provisional z at exact T_c, nu); "
            "Harada-style local window for polynomial universal fits"
        ),
    )
    parser.add_argument(
        "--obs-sigma-scale",
        type=float,
        default=None,
        help="Multiplier on MC observation std devs in GP likelihood (>1 => noisier)",
    )
    parser.add_argument(
        "--use-log-m",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use batch mean(log|m|) observables (default). "
        "Pass --no-use-log-m for linear |m| with GP on Phi_m = m L^(beta/nu).",
    )
    parser.add_argument(
        "--profile-mle-init",
        action="store_true",
        help="Start NUTS at joint profile MLE (data-derived; debugging only)",
    )
    parser.add_argument(
        "--nuts-init",
        default=None,
        help="NUTS mass-matrix init (default auto; profile-mle-init uses adapt_full)",
    )
    parser.add_argument(
        "--reparametrize-Tc",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Sample T_c_delta with T_c = ref + delta/scale (ref=exact T_c by default)",
    )
    parser.add_argument(
        "--tc-delta-scale",
        type=float,
        default=TC_DELTA_SCALE,
        help="T_c = tc_reparam_ref + T_c_delta / this (default 1000)",
    )
    parser.add_argument(
        "--tc-prior-lower",
        type=float,
        default=None,
        help=f"Lower bound of uniform T_c prior (default: {TC_PRIOR_LOWER})",
    )
    parser.add_argument(
        "--tc-prior-upper",
        type=float,
        default=None,
        help=f"Upper bound of uniform T_c prior (default: {TC_PRIOR_UPPER})",
    )
    parser.add_argument(
        "--tc-init",
        type=float,
        default=None,
        help=f"Initial / fixed-T_c reference value (default: {TC_EXACT})",
    )
    parser.add_argument(
        "--sampler-backend",
        choices=("jax", "laps", "pymc"),
        default=SAMPLER_BACKEND,
        help="Posterior sampler backend (default: jax). laps: Blackjax LAPS, 200 chains",
    )
    parser.add_argument(
        "--jax-platform",
        choices=("auto", "cpu", "gpu", "tpu"),
        default=None,
        help="JAX platform for likelihood and sampling (default: auto; falls back to cpu when no GPU is visible)",
    )
    args = parser.parse_args(argv)

    if args.harada_bsa:
        args.dataset = "harada_bsa"
        args.use_m = False
        args.use_m2 = False
        args.use_m4 = False
        args.use_chi = False
        args.use_binder = True
        if args.obs_sigma_scale is None:
            args.obs_sigma_scale = 1.0

    return fit_dataset(
        args.dataset,
        infer_Tc=args.infer_Tc,
        infer_nu=args.infer_nu,
        infer_beta=args.infer_beta,
        infer_omega=args.infer_omega,
        infer_gp_hyperparams=args.infer_gp_hyperparams,
        use_m=args.use_m,
        use_m2=args.use_m2,
        use_m4=args.use_m4,
        use_binder=args.use_binder,
        use_chi=args.use_chi,
        correction_m=args.correction_m,
        correction_m2=args.correction_m2,
        correction_m4=args.correction_m4,
        correction_binder=args.correction_binder,
        correction_chi=args.correction_chi,
        discrepancy_m=args.discrepancy_m,
        discrepancy_m2=args.discrepancy_m2,
        discrepancy_m4=args.discrepancy_m4,
        discrepancy_binder=args.discrepancy_binder,
        discrepancy_chi=args.discrepancy_chi,
        discrepancy_form=args.discrepancy_form,
        infer_discrepancy=args.infer_discrepancy,
        infer_disc_t0=args.infer_disc_t0,
        infer_disc_L0=args.infer_disc_L0,
        infer_disc_p=args.infer_disc_p,
        infer_disc_q=args.infer_disc_q,
        infer_disc_sigma_model=args.infer_disc_sigma_model,
        fixed_disc_t0=args.fixed_disc_t0,
        fixed_disc_L0=args.fixed_disc_L0,
        fixed_disc_p=args.fixed_disc_p,
        fixed_disc_q=args.fixed_disc_q,
        fixed_disc_sigma_model=args.fixed_disc_sigma_model,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        cores=args.cores,
        target_accept=args.target_accept,
        seed=args.seed,
        max_eta_hours=None if args.max_eta_hours <= 0 else args.max_eta_hours,
        data=args.data,
        out=args.out,
        gp_ell_factor=args.gp_ell_factor,
        gp_ell=args.gp_ell,
        gp_eta=args.gp_eta,
        correction_gp_ell_factor=args.correction_gp_ell_factor,
        correction_gp_ell=args.correction_gp_ell,
        correction_gp_eta=args.correction_gp_eta,
        gp_kernel=args.gp_kernel,
        universal_kernel=args.universal_kernel,
        max_abs_z=args.max_abs_z,
        obs_sigma_scale=args.obs_sigma_scale,
        use_log_m=args.use_log_m,
        profile_mle_init=args.profile_mle_init,
        nuts_init=args.nuts_init,
        reparametrize_Tc=args.reparametrize_Tc,
        tc_delta_scale=args.tc_delta_scale,
        tc_prior_lower=args.tc_prior_lower,
        tc_prior_upper=args.tc_prior_upper,
        tc_init=args.tc_init,
        sampler_backend=args.sampler_backend,
        jax_platform=args.jax_platform,
    )


if __name__ == "__main__":
    main()
