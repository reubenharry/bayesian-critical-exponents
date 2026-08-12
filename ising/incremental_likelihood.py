"""Incremental FSS log-likelihood for hypothetical new observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .constants import SPATIAL_DIMENSION
from .fss_likelihood import (
    FssLikelihoodArrays,
    _chi_gp_ell_scales,
    _config_gp_kernel,
    collapse_z,
    extract_likelihood_arrays,
    fss_gp_scales,
)
from .gp_utils import (
    correction_gp_log_predictive_density,
    correction_gp_posterior_predictive,
    gp_log_predictive_density,
    gp_posterior_predictive,
)
from .profile_likelihood import FssProfileConfig
from .scaling_function import GP_JITTER

YMode = Literal["given", "predictive_mean"]


@dataclass(frozen=True)
class CandidateObservation:
    """Hypothetical observation used for acquisition scoring."""

    L: float
    T: float
    y_by_channel: dict[str, float]
    sigma_by_channel: dict[str, float]


def _single_gp_delta_log_likelihood(
    z_train: np.ndarray,
    y_train: np.ndarray,
    sigma_train: np.ndarray,
    *,
    z_new: float,
    y_new: float,
    sigma_new: float,
    length_scale: float,
    amplitude: float,
    kernel: str,
) -> float:
    return float(
        gp_log_predictive_density(
            z_train,
            y_train,
            sigma_train,
            np.array([z_new]),
            np.array([y_new]),
            sigma_new=sigma_new,
            length_scale=length_scale,
            amplitude=amplitude,
            kernel=kernel,
            jitter=GP_JITTER,
        )[0]
    )


def _single_correction_gp_delta_log_likelihood(
    z_train: np.ndarray,
    L_train: np.ndarray,
    y_train: np.ndarray,
    sigma_train: np.ndarray,
    *,
    z_new: float,
    L_new: float,
    y_new: float,
    sigma_new: float,
    omega: float,
    gp_ell: float,
    gp_eta: float,
    correction_gp_ell: float,
    correction_gp_eta: float,
    kernel: str,
) -> float:
    return float(
        correction_gp_log_predictive_density(
            z_train,
            L_train,
            y_train,
            sigma_train,
            np.array([z_new]),
            np.array([L_new]),
            np.array([y_new]),
            sigma_test=sigma_new,
            omega=omega,
            gp_ell=gp_ell,
            gp_eta=gp_eta,
            correction_gp_ell=correction_gp_ell,
            correction_gp_eta=correction_gp_eta,
            kernel=kernel,
            jitter=GP_JITTER,
        )[0]
    )


def _gp_predictive_mean_y(
    z_train: np.ndarray,
    y_train: np.ndarray,
    sigma_train: np.ndarray,
    *,
    z_new: float,
    length_scale: float,
    amplitude: float,
    kernel: str,
) -> float:
    mean, _ = gp_posterior_predictive(
        z_train,
        y_train,
        sigma_train,
        np.array([z_new]),
        length_scale=length_scale,
        amplitude=amplitude,
        kernel=kernel,
        jitter=GP_JITTER,
    )
    return float(mean[0])


def _correction_gp_predictive_mean_y(
    z_train: np.ndarray,
    L_train: np.ndarray,
    y_train: np.ndarray,
    sigma_train: np.ndarray,
    *,
    z_new: float,
    L_new: float,
    omega: float,
    gp_ell: float,
    gp_eta: float,
    correction_gp_ell: float,
    correction_gp_eta: float,
    kernel: str,
) -> float:
    mean, _ = correction_gp_posterior_predictive(
        z_train,
        L_train,
        y_train,
        sigma_train,
        np.array([z_new]),
        np.array([L_new]),
        omega=omega,
        gp_ell=gp_ell,
        gp_eta=gp_eta,
        correction_gp_ell=correction_gp_ell,
        correction_gp_eta=correction_gp_eta,
        kernel=kernel,
        jitter=GP_JITTER,
    )
    return float(mean[0])


def _resolve_channel_y(
    *,
    y_mode: YMode,
    channel: str,
    candidate: CandidateObservation,
    predictive_mean: float,
) -> float:
    if y_mode == "predictive_mean":
        return predictive_mean
    if channel not in candidate.y_by_channel:
        raise KeyError(f"Missing channel {channel!r} in candidate.y_by_channel")
    return float(candidate.y_by_channel[channel])


def fss_delta_log_likelihood(
    arrays: FssLikelihoodArrays,
    config: FssProfileConfig,
    candidate: CandidateObservation,
    *,
    T_c: float,
    nu: float,
    beta: float,
    y_mode: YMode = "given",
) -> float:
    """Sum of channel log predictive densities for one candidate at fixed exponents."""
    scales = fss_gp_scales(
        arrays.L,
        arrays.T,
        gp_ell_factor=config.gp_ell_factor,
        gp_eta=config.gp_eta,
        correction_gp_ell_factor=config.correction_gp_ell_factor,
        correction_gp_eta=config.correction_gp_eta,
        gp_ell=getattr(config, "gp_ell", None),
        correction_gp_ell=getattr(config, "correction_gp_ell", None),
    )
    z = collapse_z(arrays.L, arrays.T, T_c=T_c, nu=nu)
    z_new = float(collapse_z(np.array([candidate.L]), np.array([candidate.T]), T_c=T_c, nu=nu)[0])
    omega = float(config.omega_fixed)
    chi_gp_ell, chi_correction_gp_ell = _chi_gp_ell_scales(
        z=z, config=config, scales=scales
    )
    gp_kernel = _config_gp_kernel(config)
    L_new = float(candidate.L)
    log_ml = 0.0

    if config.use_m:
        assert arrays.magnetization is not None
        assert arrays.sigma_m is not None
        assert arrays.log_m is not None
        assert arrays.sigma_log is not None
        if config.use_log_m and not config.correction_m:
            log_phi_m = arrays.log_m + (beta / nu) * np.log(arrays.L)
            sigma_new = candidate.sigma_by_channel["log_m"]
            mu = _gp_predictive_mean_y(
                z,
                log_phi_m,
                arrays.sigma_log,
                z_new=z_new,
                length_scale=scales.base_gp_ell,
                amplitude=scales.base_gp_eta,
                kernel=gp_kernel,
            )
            y_new = _resolve_channel_y(
                y_mode=y_mode,
                channel="log_m",
                candidate=candidate,
                predictive_mean=mu,
            )
            log_ml += _single_gp_delta_log_likelihood(
                z,
                log_phi_m,
                arrays.sigma_log,
                z_new=z_new,
                y_new=y_new,
                sigma_new=sigma_new,
                length_scale=scales.base_gp_ell,
                amplitude=scales.base_gp_eta,
                kernel=gp_kernel,
            )
        elif config.correction_m:
            phi_m = arrays.magnetization * arrays.L ** (beta / nu)
            sigma_phi_m = arrays.sigma_m * arrays.L ** (beta / nu)
            sigma_new = candidate.sigma_by_channel["m"]
            mu = _correction_gp_predictive_mean_y(
                z,
                arrays.L,
                phi_m,
                sigma_phi_m,
                z_new=z_new,
                L_new=L_new,
                omega=omega,
                gp_ell=scales.base_gp_ell,
                gp_eta=scales.base_gp_eta,
                correction_gp_ell=scales.correction_gp_ell,
                correction_gp_eta=scales.correction_gp_eta,
                kernel=gp_kernel,
            )
            y_new = _resolve_channel_y(
                y_mode=y_mode,
                channel="m",
                candidate=candidate,
                predictive_mean=mu,
            )
            log_ml += _single_correction_gp_delta_log_likelihood(
                z,
                arrays.L,
                phi_m,
                sigma_phi_m,
                z_new=z_new,
                L_new=L_new,
                y_new=y_new,
                sigma_new=sigma_new,
                omega=omega,
                gp_ell=scales.base_gp_ell,
                gp_eta=scales.base_gp_eta,
                correction_gp_ell=scales.correction_gp_ell,
                correction_gp_eta=scales.correction_gp_eta,
                kernel=gp_kernel,
            )
        else:
            phi_m = arrays.magnetization * arrays.L ** (beta / nu)
            sigma_phi_m = arrays.sigma_m * arrays.L ** (beta / nu)
            sigma_new = candidate.sigma_by_channel["m"]
            mu = _gp_predictive_mean_y(
                z,
                phi_m,
                sigma_phi_m,
                z_new=z_new,
                length_scale=scales.base_gp_ell,
                amplitude=scales.base_gp_eta,
                kernel=gp_kernel,
            )
            y_new = _resolve_channel_y(
                y_mode=y_mode,
                channel="m",
                candidate=candidate,
                predictive_mean=mu,
            )
            log_ml += _single_gp_delta_log_likelihood(
                z,
                phi_m,
                sigma_phi_m,
                z_new=z_new,
                y_new=y_new,
                sigma_new=sigma_new,
                length_scale=scales.base_gp_ell,
                amplitude=scales.base_gp_eta,
                kernel=gp_kernel,
            )

    if config.use_m2:
        assert arrays.m2 is not None and arrays.sigma_m2 is not None
        power = 2.0 * beta / nu
        phi = arrays.m2 * arrays.L**power
        sigma_phi = arrays.sigma_m2 * arrays.L**power
        sigma_new = candidate.sigma_by_channel["m2"]
        if config.correction_m2:
            mu = _correction_gp_predictive_mean_y(
                z,
                arrays.L,
                phi,
                sigma_phi,
                z_new=z_new,
                L_new=L_new,
                omega=omega,
                gp_ell=scales.base_gp_ell,
                gp_eta=scales.base_gp_eta,
                correction_gp_ell=scales.correction_gp_ell,
                correction_gp_eta=scales.correction_gp_eta,
                kernel=gp_kernel,
            )
            y_new = _resolve_channel_y(
                y_mode=y_mode,
                channel="m2",
                candidate=candidate,
                predictive_mean=mu,
            )
            log_ml += _single_correction_gp_delta_log_likelihood(
                z,
                arrays.L,
                phi,
                sigma_phi,
                z_new=z_new,
                L_new=L_new,
                y_new=y_new,
                sigma_new=sigma_new,
                omega=omega,
                gp_ell=scales.base_gp_ell,
                gp_eta=scales.base_gp_eta,
                correction_gp_ell=scales.correction_gp_ell,
                correction_gp_eta=scales.correction_gp_eta,
                kernel=gp_kernel,
            )
        else:
            mu = _gp_predictive_mean_y(
                z,
                phi,
                sigma_phi,
                z_new=z_new,
                length_scale=scales.base_gp_ell,
                amplitude=scales.base_gp_eta,
                kernel=gp_kernel,
            )
            y_new = _resolve_channel_y(
                y_mode=y_mode,
                channel="m2",
                candidate=candidate,
                predictive_mean=mu,
            )
            log_ml += _single_gp_delta_log_likelihood(
                z,
                phi,
                sigma_phi,
                z_new=z_new,
                y_new=y_new,
                sigma_new=sigma_new,
                length_scale=scales.base_gp_ell,
                amplitude=scales.base_gp_eta,
                kernel=gp_kernel,
            )

    if config.use_m4:
        assert arrays.m4 is not None and arrays.sigma_m4 is not None
        power = 4.0 * beta / nu
        phi = arrays.m4 * arrays.L**power
        sigma_phi = arrays.sigma_m4 * arrays.L**power
        sigma_new = candidate.sigma_by_channel["m4"]
        if config.correction_m4:
            mu = _correction_gp_predictive_mean_y(
                z,
                arrays.L,
                phi,
                sigma_phi,
                z_new=z_new,
                L_new=L_new,
                omega=omega,
                gp_ell=scales.base_gp_ell,
                gp_eta=scales.base_gp_eta,
                correction_gp_ell=scales.correction_gp_ell,
                correction_gp_eta=scales.correction_gp_eta,
                kernel=gp_kernel,
            )
            y_new = _resolve_channel_y(
                y_mode=y_mode,
                channel="m4",
                candidate=candidate,
                predictive_mean=mu,
            )
            log_ml += _single_correction_gp_delta_log_likelihood(
                z,
                arrays.L,
                phi,
                sigma_phi,
                z_new=z_new,
                L_new=L_new,
                y_new=y_new,
                sigma_new=sigma_new,
                omega=omega,
                gp_ell=scales.base_gp_ell,
                gp_eta=scales.base_gp_eta,
                correction_gp_ell=scales.correction_gp_ell,
                correction_gp_eta=scales.correction_gp_eta,
                kernel=gp_kernel,
            )
        else:
            mu = _gp_predictive_mean_y(
                z,
                phi,
                sigma_phi,
                z_new=z_new,
                length_scale=scales.base_gp_ell,
                amplitude=scales.base_gp_eta,
                kernel=gp_kernel,
            )
            y_new = _resolve_channel_y(
                y_mode=y_mode,
                channel="m4",
                candidate=candidate,
                predictive_mean=mu,
            )
            log_ml += _single_gp_delta_log_likelihood(
                z,
                phi,
                sigma_phi,
                z_new=z_new,
                y_new=y_new,
                sigma_new=sigma_new,
                length_scale=scales.base_gp_ell,
                amplitude=scales.base_gp_eta,
                kernel=gp_kernel,
            )

    if config.use_binder:
        assert arrays.binder is not None and arrays.sigma_binder is not None
        sigma_new = candidate.sigma_by_channel["binder"]
        if config.correction_binder:
            mu = _correction_gp_predictive_mean_y(
                z,
                arrays.L,
                arrays.binder,
                arrays.sigma_binder,
                z_new=z_new,
                L_new=L_new,
                omega=omega,
                gp_ell=scales.base_gp_ell,
                gp_eta=scales.base_gp_eta,
                correction_gp_ell=scales.correction_gp_ell,
                correction_gp_eta=scales.correction_gp_eta,
                kernel=gp_kernel,
            )
            y_new = _resolve_channel_y(
                y_mode=y_mode,
                channel="binder",
                candidate=candidate,
                predictive_mean=mu,
            )
            log_ml += _single_correction_gp_delta_log_likelihood(
                z,
                arrays.L,
                arrays.binder,
                arrays.sigma_binder,
                z_new=z_new,
                L_new=L_new,
                y_new=y_new,
                sigma_new=sigma_new,
                omega=omega,
                gp_ell=scales.base_gp_ell,
                gp_eta=scales.base_gp_eta,
                correction_gp_ell=scales.correction_gp_ell,
                correction_gp_eta=scales.correction_gp_eta,
                kernel=gp_kernel,
            )
        else:
            mu = _gp_predictive_mean_y(
                z,
                arrays.binder,
                arrays.sigma_binder,
                z_new=z_new,
                length_scale=scales.base_gp_ell,
                amplitude=scales.base_gp_eta,
                kernel=gp_kernel,
            )
            y_new = _resolve_channel_y(
                y_mode=y_mode,
                channel="binder",
                candidate=candidate,
                predictive_mean=mu,
            )
            log_ml += _single_gp_delta_log_likelihood(
                z,
                arrays.binder,
                arrays.sigma_binder,
                z_new=z_new,
                y_new=y_new,
                sigma_new=sigma_new,
                length_scale=scales.base_gp_ell,
                amplitude=scales.base_gp_eta,
                kernel=gp_kernel,
            )

    if config.use_chi:
        assert arrays.chi is not None and arrays.sigma_chi is not None
        gamma = SPATIAL_DIMENSION * nu - 2.0 * beta
        phi_chi = arrays.chi * arrays.L ** (-gamma / nu)
        sigma_phi_chi = arrays.sigma_chi * arrays.L ** (-gamma / nu)
        sigma_new = candidate.sigma_by_channel["chi"]
        if config.correction_chi:
            mu = _correction_gp_predictive_mean_y(
                z,
                arrays.L,
                phi_chi,
                sigma_phi_chi,
                z_new=z_new,
                L_new=L_new,
                omega=omega,
                gp_ell=chi_gp_ell,
                gp_eta=scales.base_gp_eta,
                correction_gp_ell=chi_correction_gp_ell,
                correction_gp_eta=scales.correction_gp_eta,
                kernel=gp_kernel,
            )
            y_new = _resolve_channel_y(
                y_mode=y_mode,
                channel="chi",
                candidate=candidate,
                predictive_mean=mu,
            )
            log_ml += _single_correction_gp_delta_log_likelihood(
                z,
                arrays.L,
                phi_chi,
                sigma_phi_chi,
                z_new=z_new,
                L_new=L_new,
                y_new=y_new,
                sigma_new=sigma_new,
                omega=omega,
                gp_ell=chi_gp_ell,
                gp_eta=scales.base_gp_eta,
                correction_gp_ell=chi_correction_gp_ell,
                correction_gp_eta=scales.correction_gp_eta,
                kernel=gp_kernel,
            )
        else:
            mu = _gp_predictive_mean_y(
                z,
                phi_chi,
                sigma_phi_chi,
                z_new=z_new,
                length_scale=chi_gp_ell,
                amplitude=scales.base_gp_eta,
                kernel=gp_kernel,
            )
            y_new = _resolve_channel_y(
                y_mode=y_mode,
                channel="chi",
                candidate=candidate,
                predictive_mean=mu,
            )
            log_ml += _single_gp_delta_log_likelihood(
                z,
                phi_chi,
                sigma_phi_chi,
                z_new=z_new,
                y_new=y_new,
                sigma_new=sigma_new,
                length_scale=chi_gp_ell,
                amplitude=scales.base_gp_eta,
                kernel=gp_kernel,
            )

    return float(log_ml)


def fss_delta_log_likelihood_batch(
    arrays: FssLikelihoodArrays,
    config: FssProfileConfig,
    candidate: CandidateObservation,
    *,
    T_c: np.ndarray,
    nu: np.ndarray,
    beta: np.ndarray,
    y_mode: YMode = "given",
) -> np.ndarray:
    """Vectorised delta log-likelihood over posterior draws."""
    T_c = np.asarray(T_c, dtype=np.float64).ravel()
    nu = np.asarray(nu, dtype=np.float64).ravel()
    beta = np.asarray(beta, dtype=np.float64).ravel()
    if not (T_c.size == nu.size == beta.size):
        raise ValueError("T_c, nu, and beta must have the same length")
    out = np.empty(T_c.size, dtype=np.float64)
    for i in range(T_c.size):
        out[i] = fss_delta_log_likelihood(
            arrays,
            config,
            candidate,
            T_c=float(T_c[i]),
            nu=float(nu[i]),
            beta=float(beta[i]),
            y_mode=y_mode,
        )
    return out
