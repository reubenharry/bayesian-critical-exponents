#!/usr/bin/env python3
"""Simulate → profile diagnostics → fit → plot for FSS models."""

from __future__ import annotations

from . import fit, plot_posterior, simulate
from .datasets import resolve_paths
from .observables import ensure_harada_bsa_observables_csv
from .profile_likelihood import (
    CORRECTION_BINDER,
    CORRECTION_CHI,
    CORRECTION_GP_ELL_FACTOR,
    CORRECTION_GP_ETA,
    CORRECTION_M,
    CORRECTION_M2,
    CORRECTION_M4,
    FssProfileConfig,
    GP_ELL_FACTOR,
    GP_ETA,
    OBS_SIGMA_SCALE,
    USE_BINDER,
    USE_CHI,
    USE_LOG_M,
    USE_M,
    USE_M2,
    USE_M4,
    run_profile_diagnostics,
)

# --- edit these ---
DATASET = "harada_square"
MODEL = "fss"

INFER_TC = True
INFER_NU = True
INFER_BETA = True

INFER_OMEGA = False
INFER_GP_HYPERPARAMS = False

REDO_SIMULATE = False  # harada_bsa uses published Binder table, not MC simulation
REDO_PROFILE = True
REDO_FIT = True
REDO_PLOT = True

DRAWS = 100
TUNE = 200
CHAINS = 2000
MAX_ETA_HOURS = 0.0  # 0 disables the sampling time guard (PyMC only)

SAMPLER_BACKEND = "laps"
JAX_PLATFORM = None  # auto | cpu | gpu | tpu

# FSS channel / GP flags live in profile_likelihood.py (imported above).

# TEMPORARY: start chains at joint profile MLE + dense mass-matrix adaptation.
# Uses the data to pick starting points — remove before trusting out-of-sample runs.
PROFILE_MLE_INIT = True

# T_c = TC_EXACT + T_c_delta / TC_DELTA_SCALE so NUTS sees O(1) delta steps.
REPARAMETRIZE_TC = True
TC_DELTA_SCALE = 1000.0


def _profile_config() -> FssProfileConfig:
    """Profile-likelihood settings mirrored from the flags above."""
    return FssProfileConfig(
        use_m=USE_M,
        use_m2=USE_M2,
        use_m4=USE_M4,
        use_binder=USE_BINDER,
        use_chi=USE_CHI,
        correction_m=CORRECTION_M,
        correction_m2=CORRECTION_M2,
        correction_m4=CORRECTION_M4,
        correction_binder=CORRECTION_BINDER,
        correction_chi=CORRECTION_CHI,
        use_log_m=USE_LOG_M,
        gp_ell_factor=GP_ELL_FACTOR,
        gp_eta=GP_ETA,
        correction_gp_ell_factor=CORRECTION_GP_ELL_FACTOR,
        correction_gp_eta=CORRECTION_GP_ETA,
        obs_sigma_scale=OBS_SIGMA_SCALE,
        infer_Tc=INFER_TC,
        infer_nu=INFER_NU,
        infer_beta=INFER_BETA,
        reparametrize_Tc=REPARAMETRIZE_TC,
        tc_delta_scale=TC_DELTA_SCALE,
    )


def main() -> None:
    data_path, posterior_path, plots_path = resolve_paths(DATASET, model=MODEL)

    if DATASET == "harada_bsa":
        ensure_harada_bsa_observables_csv(data_path)
        print(f"Using published Harada BSA Binder data at {data_path}")
    elif REDO_SIMULATE or not data_path.exists():
        simulate.simulate_dataset(DATASET, out=data_path)
    else:
        print(f"Skipping simulate ({data_path} exists, redo=False)")

    profile_marker = plots_path / "profile_exponent_joint_likelihoods.png"
    if REDO_PROFILE or not profile_marker.exists():
        run_profile_diagnostics(
            DATASET,
            data=data_path,
            out=plots_path,
            config=_profile_config(),
        )
    else:
        print(f"Skipping profile diagnostics ({profile_marker} exists, redo=False)")

    if REDO_FIT or not posterior_path.exists():
        fit.fit_dataset(
            DATASET,
            model=MODEL,
            infer_Tc=INFER_TC,
            infer_nu=INFER_NU,
            infer_beta=INFER_BETA,
            infer_omega=INFER_OMEGA,
            infer_gp_hyperparams=INFER_GP_HYPERPARAMS,
            use_m=USE_M,
            use_m2=USE_M2,
            use_m4=USE_M4,
            use_binder=USE_BINDER,
            use_chi=USE_CHI,
            correction_m=CORRECTION_M,
            correction_m2=CORRECTION_M2,
            correction_m4=CORRECTION_M4,
            correction_binder=CORRECTION_BINDER,
            correction_chi=CORRECTION_CHI,
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            max_eta_hours=None if MAX_ETA_HOURS <= 0 else MAX_ETA_HOURS,
            data=data_path,
            out=posterior_path,
            gp_ell_factor=GP_ELL_FACTOR,
            gp_eta=GP_ETA,
            correction_gp_ell_factor=CORRECTION_GP_ELL_FACTOR,
            correction_gp_eta=CORRECTION_GP_ETA,
            obs_sigma_scale=OBS_SIGMA_SCALE,
            use_log_m=USE_LOG_M,
            profile_mle_init=PROFILE_MLE_INIT,
            reparametrize_Tc=REPARAMETRIZE_TC,
            tc_delta_scale=TC_DELTA_SCALE,
            sampler_backend=SAMPLER_BACKEND,
            jax_platform=JAX_PLATFORM,
        )
    else:
        print(f"Skipping fit ({posterior_path} exists, redo=False)")

    plot_marker = plots_path / "posterior_exponents.png"
    if REDO_PLOT or not plot_marker.exists():
        plot_posterior.plot_dataset(
            DATASET,
            model=MODEL,
            data=data_path,
            posterior=posterior_path,
            out=plots_path,
        )
    else:
        print(f"Skipping plot ({plot_marker} exists, redo=False)")


if __name__ == "__main__":
    main()
