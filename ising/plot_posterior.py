#!/usr/bin/env python3
"""Plot exponent posteriors and a data-collapse diagnostic."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import erfinv

from .analytics import magnetization_infinite_L
from .constants import (
    BETA_EXACT,
    GAMMA_EXACT,
    NU_EXACT,
    OMEGA_EXACT,
    SPATIAL_DIMENSION,
    TC_EXACT,
)
from .model_scaling_nu import (
    CORRECTION_BASE_GP_ETA_FIXED,
    CORRECTION_GP_ELL_FACTOR,
    CORRECTION_GP_ETA_FIXED,
    GP_ELL_STIFF_FACTOR,
)
from .datasets import list_datasets, resolve_paths
from .observables import validate_observables_df
from .gp_utils import (
    correction_gp_latent_conditional,
    correction_gp_posterior_predictive,
    gp_latent_conditional,
    gp_posterior_predictive,
)
from .observables import resolve_log_m_observables
from .scaling import log_phi_from_log_m, z_from_LT
from .scaling_function import DEFAULT_SPLINE_DEGREE, eval_scaling_curve_np

EXACT_VALUES = {
    "T_c": TC_EXACT,
    "nu": NU_EXACT,
    "beta": BETA_EXACT,
    "gamma": GAMMA_EXACT,
    "omega": OMEGA_EXACT,
}


def _posterior_has(idata: az.InferenceData, name: str) -> bool:
    return name in idata.posterior


def _infer_gp_hyperparams(idata: az.InferenceData) -> bool:
    raw = idata.posterior.attrs.get("infer_gp_hyperparams")
    if raw is not None:
        return str(raw).lower() in {"true", "1", "yes"}
    return _posterior_has(idata, "gp_ell") and not _posterior_has(idata, "gp_ell_m")


def _inferred_gp_var_names(idata: az.InferenceData) -> tuple[str, ...]:
    names: list[str] = []
    for name in ("gp_ell", "gp_eta", "correction_gp_ell", "correction_gp_eta"):
        if _posterior_has(idata, name):
            names.append(name)
    return tuple(names)


def _posterior_gp_means(idata: az.InferenceData) -> dict[str, float]:
    """Posterior means for directly sampled GP hyperparameters (``gp_ell``, etc.)."""
    if not _infer_gp_hyperparams(idata):
        return {}
    out: dict[str, float] = {}
    for name in _inferred_gp_var_names(idata):
        out[name] = float(idata.posterior[name].mean().values)
    return out


def _filter_posterior_vars(
    idata: az.InferenceData,
    names: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    return tuple(name for name in names if _posterior_has(idata, name))


def _posterior_mean_or_exact(idata: az.InferenceData, name: str) -> float:
    if _posterior_has(idata, name):
        return float(idata.posterior[name].mean().values)
    return EXACT_VALUES[name]


def _posterior_draw_or_exact(
    idata: az.InferenceData,
    name: str,
    chain: int,
    draw: int,
) -> float:
    if _posterior_has(idata, name):
        return float(idata.posterior[name].values[chain, draw])
    return EXACT_VALUES[name]


def _append_plots(paths: list[Path], *plot_paths: Path | None) -> None:
    paths.extend(path for path in plot_paths if path is not None)


def _posterior_model(idata: az.InferenceData) -> str:
    return str(idata.posterior.attrs.get("model", "fss"))


def _is_tl_gp_model(idata: az.InferenceData) -> bool:
    return _posterior_model(idata) in ("scaling_tl", "scaling_tl_tc")


def _is_correction_nu_model(idata: az.InferenceData) -> bool:
    return _posterior_model(idata) in (
        "scaling_nu_corr",
        "scaling_nu_corr_hyper",
        "scaling_nu_corr_hyper_omega2",
        "scaling_corr_hyper_omega2_tc",
        "scaling_corr_hyper_omega2_fss",
        "scaling_corr_hyper_fss",
    )


def _is_fss_gp_model(idata: az.InferenceData) -> bool:
    """Correction-GP FSS model from model_fss (current or legacy posteriors)."""
    model = _posterior_model(idata)
    if model in ("fss_nu", "fss_binder"):
        return True
    return model == "fss" and idata.posterior.attrs.get("fss_correction") is not None


def _is_fss_tc_model(idata: az.InferenceData) -> bool:
    return _is_fss_gp_model(idata)


def _fss_include_m(idata: az.InferenceData) -> bool:
    raw = idata.posterior.attrs.get("fss_m")
    if raw is not None:
        return str(raw).lower() in {"true", "1", "yes"}
    return _posterior_has(idata, "log_phi_m")


def _fss_include_m2(idata: az.InferenceData) -> bool:
    raw = idata.posterior.attrs.get("fss_m2")
    if raw is not None:
        return str(raw).lower() in {"true", "1", "yes"}
    return _posterior_has(idata, "log_phi_m2")


def _fss_include_m4(idata: az.InferenceData) -> bool:
    raw = idata.posterior.attrs.get("fss_m4")
    if raw is not None:
        return str(raw).lower() in {"true", "1", "yes"}
    return _posterior_has(idata, "log_phi_m4")


def _fss_use_log_m(idata: az.InferenceData) -> bool:
    raw = idata.posterior.attrs.get("use_log_m")
    if raw is None:
        return True
    return str(raw).lower() in {"true", "1", "yes"}


def _fss_include_binder(idata: az.InferenceData) -> bool:
    if _posterior_model(idata) == "fss_binder":
        return True
    raw = idata.posterior.attrs.get("fss_binder")
    return raw is not None and str(raw).lower() in {"true", "1", "yes"}


def _fss_binder_only(idata: az.InferenceData) -> bool:
    """True when Binder is the sole active FSS channel."""
    return _fss_include_binder(idata) and not (
        _fss_include_m(idata)
        or _fss_include_m2(idata)
        or _fss_include_m4(idata)
        or _fss_include_chi(idata)
    )


def _uses_marginal_gp_fss_plot(idata: az.InferenceData) -> bool:
    """JAX/LAPS FSS fit: marginal GP with fixed hyperparameters, no z traces."""
    if _posterior_model(idata) != "fss" or _uses_fss_correction_gp_plot(idata):
        return False
    if _posterior_has_scaling_trace(idata):
        return False
    return idata.posterior.attrs.get("gp_ell_fixed") is not None


def _uses_marginal_gp_binder_plot(idata: az.InferenceData) -> bool:
    """Binder-only subset of marginal GP FSS plots."""
    return _uses_marginal_gp_fss_plot(idata) and _fss_binder_only(idata)


def _fss_include_chi(idata: az.InferenceData) -> bool:
    raw = idata.posterior.attrs.get("fss_chi")
    if raw is not None:
        return str(raw).lower() in {"true", "1", "yes"}
    return _posterior_has(idata, "log_phi_chi")


def _fss_active_log_target(idata: az.InferenceData) -> str:
    """Posterior variable for log Phi on the primary fitted FSS channel."""
    for name in ("log_phi_m", "log_phi_m2", "log_phi_m4", "log_phi_chi"):
        if _posterior_has(idata, name):
            return name
    return "log_phi_m"


def _draw_log_phi_values(
    draw: dict[str, float | np.ndarray],
    idata: az.InferenceData,
) -> np.ndarray:
    if "log_phi" in draw:
        return np.asarray(draw["log_phi"], dtype=np.float64)
    target = _scaling_log_target(idata)
    return np.asarray(draw[target], dtype=np.float64)


def _scaling_function_log_label(idata: az.InferenceData) -> str:
    if _fss_binder_only(idata):
        return r"$U_4(z)$"
    if _fss_include_chi(idata) and not (
        _fss_include_m(idata) or _fss_include_m2(idata) or _fss_include_m4(idata)
    ):
        return r"$\log\Phi_\chi(z)$"
    if _fss_include_m2(idata) and not _fss_include_m(idata):
        return r"$\log\Phi_{m^2}(z)$"
    if _fss_include_m4(idata) and not (_fss_include_m(idata) or _fss_include_m2(idata)):
        return r"$\log\Phi_{m^4}(z)$"
    return r"$\log\Phi_m(z)$"


def _fss_channel_included(idata: az.InferenceData, channel: str) -> bool:
    if channel == "m":
        return _fss_include_m(idata)
    if channel == "m2":
        return _fss_include_m2(idata)
    if channel == "m4":
        return _fss_include_m4(idata)
    if channel == "binder":
        return _fss_include_binder(idata)
    if channel == "chi":
        return _fss_include_chi(idata)
    raise ValueError(f"Unknown FSS channel {channel!r}")


def _fss_fixed_Tc(idata: az.InferenceData) -> bool:
    return _is_fss_tc_model(idata) and not _posterior_has(idata, "T_c")


def _fss_channel_correction(idata: az.InferenceData, channel: str) -> bool:
    """Whether an active channel uses f0 + L^-omega f1 (vs plain f0 only)."""
    if not _is_fss_tc_model(idata) or not _fss_channel_included(idata, channel):
        return False
    attr_map = {
        "m": "fss_correction_m",
        "m2": "fss_correction_m2",
        "m4": "fss_correction_m4",
        "binder": "fss_correction_binder",
        "chi": "fss_correction_chi",
    }
    raw = idata.posterior.attrs.get(attr_map[channel])
    if raw is None:
        return False
    return str(raw).lower() in {"true", "1", "yes"}


def _is_fss_correction(idata: az.InferenceData) -> bool:
    if not _is_fss_tc_model(idata):
        return False
    for channel in ("m", "m2", "m4", "binder", "chi"):
        if _fss_channel_correction(idata, channel):
            return True
    return "correction_gp_ell_m" in idata.posterior


def _is_fss_binder_correction(idata: az.InferenceData) -> bool:
    return _fss_channel_correction(idata, "m")


def _uses_fss_correction_gp_plot(idata: az.InferenceData) -> bool:
    return any(
        _fss_channel_correction(idata, channel)
        for channel in ("m", "m2", "m4", "binder", "chi")
    )


def _primary_fss_correction_channel(idata: az.InferenceData) -> str:
    for channel in ("m", "m2", "m4", "chi", "binder"):
        if _fss_channel_correction(idata, channel):
            return channel
    raise ValueError("No active FSS correction GP channel")


def _fss_m_uses_phi_gp(idata: az.InferenceData) -> bool:
    """Whether the m channel GP likelihood is on Phi_m (linear) rather than log Phi_m."""
    if not _fss_include_m(idata):
        return False
    return not _fss_use_log_m(idata) or _fss_channel_correction(idata, "m")


def _omega_from_draw(
    idata: az.InferenceData,
    draw: dict[str, float | np.ndarray],
) -> float:
    if "omega" in draw:
        return float(draw["omega"])
    fixed = idata.posterior.attrs.get("omega_fixed")
    if fixed is not None:
        return float(fixed)
    return OMEGA_EXACT


def _mc_sigma(df: pd.DataFrame) -> np.ndarray:
    return df["magnetization_std"].to_numpy(dtype=np.float64) / np.sqrt(
        np.maximum(df["n_eff"].to_numpy(dtype=np.float64), 1.0)
    )


def _obs_sigma_scale(idata: az.InferenceData) -> float:
    raw = idata.posterior.attrs.get("obs_sigma_scale")
    if raw is None:
        return 1.0
    return float(raw)


def _correction_gp_hyperparams(
    idata: az.InferenceData,
    draw: dict[str, float | np.ndarray] | None = None,
) -> dict[str, float]:
    if draw is not None and "gp_ell" in draw:
        return {
            "gp_ell": float(draw["gp_ell"]),
            "gp_eta": float(draw["gp_eta"]),
            "correction_gp_ell": float(draw["correction_gp_ell"]),
            "correction_gp_eta": float(draw["correction_gp_eta"]),
        }
    attrs = idata.posterior.attrs
    out: dict[str, float] = {}
    for name, fixed_key in (
        ("gp_ell", "gp_ell_fixed"),
        ("gp_eta", "gp_eta_fixed"),
        ("correction_gp_ell", "correction_gp_ell_fixed"),
        ("correction_gp_eta", "correction_gp_eta_fixed"),
    ):
        if _posterior_has(idata, name):
            out[name] = float(idata.posterior[name].mean().values)
        elif fixed_key in attrs:
            out[name] = float(attrs[fixed_key])
    return out


def _m_phi_training_from_df(
    df: pd.DataFrame,
    nu: float,
    *,
    beta: float = BETA_EXACT,
    T_c: float = TC_EXACT,
    obs_sigma_scale: float = 1.0,
    use_log_m: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (z, L, phi, sigma_phi) for the m channel."""
    L = df["L"].to_numpy(dtype=np.float64)
    T = df["T"].to_numpy(dtype=np.float64)
    t = (T - T_c) / T_c
    z = t * L ** (1.0 / nu)
    scale = L ** (beta / nu)
    if use_log_m:
        log_m, sigma_log = resolve_log_m_observables(
            df["log_magnetization"].to_numpy(dtype=np.float64),
            df["log_magnetization_std"].to_numpy(dtype=np.float64),
            df["n_eff_log"].to_numpy(dtype=np.float64),
        )
        phi = np.exp(log_m) * scale
        sigma_phi = phi * sigma_log * obs_sigma_scale
    else:
        m = df["magnetization"].to_numpy(dtype=np.float64)
        sigma_m = _mc_sigma(df) * obs_sigma_scale
        phi = m * scale
        sigma_phi = sigma_m * scale
    return z, L, phi, sigma_phi


def _correction_training_from_df(
    df: pd.DataFrame,
    nu: float,
    *,
    beta: float = BETA_EXACT,
    T_c: float = TC_EXACT,
    obs_sigma_scale: float = 1.0,
    use_log_m: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _m_phi_training_from_df(
        df,
        nu,
        beta=beta,
        T_c=T_c,
        obs_sigma_scale=obs_sigma_scale,
        use_log_m=use_log_m,
    )


def _moment_phi_training_from_df(
    df: pd.DataFrame,
    nu: float,
    *,
    moment: int,
    beta: float = BETA_EXACT,
    T_c: float = TC_EXACT,
    obs_sigma_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (z, L, phi, sigma_phi) for m^2 or m^4 scaling fields."""
    if moment not in (2, 4):
        raise ValueError(f"moment must be 2 or 4, got {moment}")
    value_col = f"m{moment}"
    std_col = f"m{moment}_std"
    n_eff_col = f"n_eff_m{moment}"
    L = df["L"].to_numpy(dtype=np.float64)
    T = df["T"].to_numpy(dtype=np.float64)
    t = (T - T_c) / T_c
    z = t * L ** (1.0 / nu)
    scale = L ** (moment * beta / nu)
    values = df[value_col].to_numpy(dtype=np.float64)
    sigma = df[std_col].to_numpy(dtype=np.float64) / np.sqrt(
        np.maximum(df[n_eff_col].to_numpy(dtype=np.float64), 1.0)
    )
    sigma = sigma * obs_sigma_scale
    phi = values * scale
    sigma_phi = sigma * scale
    return z, L, phi, sigma_phi


def _correction_gp_predict_mean_std(
    df: pd.DataFrame,
    idata: az.InferenceData,
    draw: dict[str, float | np.ndarray],
    z_test: np.ndarray,
    L_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    nu = float(draw["nu"])
    omega = _omega_from_draw(idata, draw)
    beta = float(draw.get("beta", BETA_EXACT))
    T_c = float(draw.get("T_c", TC_EXACT))
    hp = _correction_gp_hyperparams(idata, draw)
    z_train, L_train, phi_train, sigma_phi = _correction_training_from_df(
        df,
        nu,
        beta=beta,
        T_c=T_c,
        obs_sigma_scale=_obs_sigma_scale(idata),
        use_log_m=_fss_use_log_m(idata),
    )
    return correction_gp_posterior_predictive(
        z_train,
        L_train,
        phi_train,
        sigma_phi,
        z_test,
        L_test,
        omega=omega,
        **hp,
    )


def _correction_gp_predict(
    df: pd.DataFrame,
    idata: az.InferenceData,
    draw: dict[str, float | np.ndarray],
    z_test: np.ndarray,
    L_test: np.ndarray,
) -> np.ndarray:
    mean, _ = _correction_gp_predict_mean_std(df, idata, draw, z_test, L_test)
    return mean


def _fss_binder_channel_hyperparams(
    idata: az.InferenceData,
    draw: dict[str, float | np.ndarray],
    channel: str,
) -> dict[str, float]:
    suffix = "_m" if channel == "m" else "_u"
    attrs = idata.posterior.attrs

    def _value(name: str, attr_key: str) -> float:
        draw_name = f"{name}{suffix}"
        if draw_name in draw:
            return float(draw[draw_name])
        if attr_key in attrs:
            return float(attrs[attr_key])
        raise KeyError(f"Missing fss_binder GP hyperparameter {draw_name!r}")

    return {
        "gp_ell": _value("gp_ell", "gp_ell_fixed"),
        "gp_eta": _value("gp_eta", "gp_eta_fixed"),
        "correction_gp_ell": _value("correction_gp_ell", "correction_gp_ell_fixed"),
        "correction_gp_eta": _value("correction_gp_eta", "correction_gp_eta_fixed"),
    }


def _binder_correction_training_from_df(
    df: pd.DataFrame,
    nu: float,
    *,
    T_c: float = TC_EXACT,
    obs_sigma_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    L = df["L"].to_numpy(dtype=np.float64)
    T = df["T"].to_numpy(dtype=np.float64)
    u4 = df["binder_cumulant"].to_numpy(dtype=np.float64)
    sigma = df["binder_cumulant_std"].to_numpy(dtype=np.float64) / np.sqrt(
        np.maximum(df["n_eff_binder"].to_numpy(dtype=np.float64), 1.0)
    )
    sigma = sigma * obs_sigma_scale
    t = (T - T_c) / T_c
    z = t * L ** (1.0 / nu)
    return z, L, u4, sigma


def _fss_binder_m_correction_predict_mean_std(
    df: pd.DataFrame,
    idata: az.InferenceData,
    draw: dict[str, float | np.ndarray],
    z_test: np.ndarray,
    L_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    nu = float(draw["nu"])
    omega = _omega_from_draw(idata, draw)
    beta = float(draw.get("beta", BETA_EXACT))
    T_c = float(draw.get("T_c", TC_EXACT))
    hp = _fss_binder_channel_hyperparams(idata, draw, "m")
    z_train, L_train, phi_train, sigma_phi = _correction_training_from_df(
        df,
        nu,
        beta=beta,
        T_c=T_c,
        obs_sigma_scale=_obs_sigma_scale(idata),
        use_log_m=_fss_use_log_m(idata),
    )
    return correction_gp_posterior_predictive(
        z_train,
        L_train,
        phi_train,
        sigma_phi,
        z_test,
        L_test,
        omega=omega,
        **hp,
    )


def _fss_binder_m_correction_predict(
    df: pd.DataFrame,
    idata: az.InferenceData,
    draw: dict[str, float | np.ndarray],
    z_test: np.ndarray,
    L_test: np.ndarray,
) -> np.ndarray:
    mean, _ = _fss_binder_m_correction_predict_mean_std(df, idata, draw, z_test, L_test)
    return mean


def _fss_moment_correction_predict(
    df: pd.DataFrame,
    idata: az.InferenceData,
    draw: dict[str, float | np.ndarray],
    z_test: np.ndarray,
    L_test: np.ndarray,
    *,
    moment: int,
) -> np.ndarray:
    nu = float(draw["nu"])
    omega = _omega_from_draw(idata, draw)
    beta = float(draw.get("beta", BETA_EXACT))
    T_c = float(draw.get("T_c", TC_EXACT))
    hp = _correction_gp_hyperparams(idata, draw)
    z_train, L_train, phi_train, sigma_phi = _moment_phi_training_from_df(
        df,
        nu,
        moment=moment,
        beta=beta,
        T_c=T_c,
        obs_sigma_scale=_obs_sigma_scale(idata),
    )
    mean, _ = correction_gp_posterior_predictive(
        z_train,
        L_train,
        phi_train,
        sigma_phi,
        z_test,
        L_test,
        omega=omega,
        **hp,
    )
    return mean


def _chi_correction_training_from_df(
    df: pd.DataFrame,
    nu: float,
    *,
    beta: float = BETA_EXACT,
    T_c: float = TC_EXACT,
    obs_sigma_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    L = df["L"].to_numpy(dtype=np.float64)
    T = df["T"].to_numpy(dtype=np.float64)
    t = (T - T_c) / T_c
    z = t * L ** (1.0 / nu)
    chi = df["susceptibility"].to_numpy(dtype=np.float64)
    sigma_chi = df["susceptibility_std"].to_numpy(dtype=np.float64) / np.sqrt(
        np.maximum(df["n_eff_susceptibility"].to_numpy(dtype=np.float64), 1.0)
    )
    scale = L ** (-SPATIAL_DIMENSION + 2.0 * beta / nu)
    phi = chi * scale
    sigma_phi = sigma_chi * scale * obs_sigma_scale
    return z, L, phi, sigma_phi


def _fss_chi_gp_hyperparams(
    z_train: np.ndarray,
    idata: az.InferenceData,
) -> dict[str, float]:
    attrs = idata.posterior.attrs
    z_span = float(np.max(z_train) - np.min(z_train))
    return {
        "gp_ell": float(attrs.get("gp_ell_fixed", GP_ELL_STIFF_FACTOR * z_span)),
        "gp_eta": float(attrs.get("gp_eta_fixed", CORRECTION_BASE_GP_ETA_FIXED)),
        "correction_gp_ell": float(
            attrs.get("correction_gp_ell_fixed", CORRECTION_GP_ELL_FACTOR * z_span)
        ),
        "correction_gp_eta": float(
            attrs.get("correction_gp_eta_fixed", CORRECTION_GP_ETA_FIXED)
        ),
    }


def _fss_chi_correction_predict(
    df: pd.DataFrame,
    idata: az.InferenceData,
    draw: dict[str, float | np.ndarray],
    z_test: np.ndarray,
    L_test: np.ndarray,
) -> np.ndarray:
    nu = float(draw["nu"])
    omega = _omega_from_draw(idata, draw)
    beta = float(draw.get("beta", BETA_EXACT))
    T_c = float(draw.get("T_c", TC_EXACT))
    z_train, L_train, phi_train, sigma_phi = _chi_correction_training_from_df(
        df, nu, beta=beta, T_c=T_c, obs_sigma_scale=_obs_sigma_scale(idata)
    )
    hp = _fss_chi_gp_hyperparams(z_train, idata)
    mean, _ = correction_gp_posterior_predictive(
        z_train,
        L_train,
        phi_train,
        sigma_phi,
        z_test,
        L_test,
        omega=omega,
        **hp,
    )
    return mean


def _fss_correction_log_phi_predict(
    df: pd.DataFrame,
    idata: az.InferenceData,
    draw: dict[str, float | np.ndarray],
    z_test: np.ndarray,
    L_test: np.ndarray,
) -> np.ndarray:
    df = _df_for_fss_plot(df, idata)
    channel = _primary_fss_correction_channel(idata)
    if channel == "m":
        phi = _fss_binder_m_correction_predict(df, idata, draw, z_test, L_test)
    elif channel == "m2":
        phi = _fss_moment_correction_predict(
            df, idata, draw, z_test, L_test, moment=2
        )
    elif channel == "m4":
        phi = _fss_moment_correction_predict(
            df, idata, draw, z_test, L_test, moment=4
        )
    elif channel == "chi":
        phi = _fss_chi_correction_predict(df, idata, draw, z_test, L_test)
    elif channel == "binder":
        phi = _fss_binder_u_correction_predict(df, idata, draw, z_test, L_test)
    else:
        raise ValueError(f"Unsupported FSS correction GP channel {channel!r}")
    return np.log(np.maximum(phi, 1e-12))


def _fss_binder_f0_predict(
    df: pd.DataFrame,
    idata: az.InferenceData,
    draw: dict[str, float | np.ndarray],
    z_test: np.ndarray,
) -> np.ndarray:
    """GP posterior mean of the universal Binder cumulant U_4(z) (f0 only)."""
    df = _df_for_fss_plot(df, idata)
    nu = float(draw["nu"])
    T_c = float(draw.get("T_c", TC_EXACT))
    z_train, _, u4_train, sigma_u = _binder_correction_training_from_df(
        df, nu, T_c=T_c, obs_sigma_scale=_obs_sigma_scale(idata)
    )
    gp_hp = _gp_params_from_attrs(idata)
    if not gp_hp:
        raise ValueError("GP hyperparameters required for marginal binder GP plot")
    mean, _ = gp_posterior_predictive(
        z_train,
        u4_train,
        sigma_u,
        z_test,
        length_scale=gp_hp["gp_ell"],
        amplitude=gp_hp["gp_eta"],
        kernel=_universal_kernel_from_attrs(idata),
    )
    return mean


def _fss_binder_u_correction_predict(
    df: pd.DataFrame,
    idata: az.InferenceData,
    draw: dict[str, float | np.ndarray],
    z_test: np.ndarray,
    L_test: np.ndarray,
) -> np.ndarray:
    nu = float(draw["nu"])
    omega = _omega_from_draw(idata, draw)
    T_c = float(draw.get("T_c", TC_EXACT))
    hp = _fss_binder_channel_hyperparams(idata, draw, "u")
    z_train, L_train, u4_train, sigma_u = _binder_correction_training_from_df(
        df, nu, T_c=T_c, obs_sigma_scale=_obs_sigma_scale(idata)
    )
    mean, _ = correction_gp_posterior_predictive(
        z_train,
        L_train,
        u4_train,
        sigma_u,
        z_test,
        L_test,
        omega=omega,
        **hp,
    )
    return mean


def _predict_log_m_tl_family(
    idata: az.InferenceData,
    df: pd.DataFrame,
    draw: dict[str, float | np.ndarray],
    L_new: np.ndarray,
    T_new: np.ndarray,
) -> np.ndarray:
    if _posterior_model(idata) == "scaling_tl_tc":
        return predict_log_m_tl_tc_gp(df, draw, L_new, T_new)
    return predict_log_m_tl_gp(df, draw, L_new, T_new)


def _posterior_variant(idata: az.InferenceData) -> str:
    return str(idata.posterior.attrs.get("variant", "plain"))


def _gp_params_from_attrs(idata: az.InferenceData) -> dict[str, float]:
    out: dict[str, float] = {}
    ell_fixed = idata.posterior.attrs.get("gp_ell_fixed")
    if ell_fixed is not None:
        out["gp_ell"] = float(ell_fixed)
    eta_fixed = idata.posterior.attrs.get("gp_eta_fixed")
    if eta_fixed is not None:
        out["gp_eta"] = float(eta_fixed)
    return out


def _universal_kernel_from_attrs(idata: az.InferenceData) -> str:
    from .gp_kernels import DEFAULT_GP_KERNEL, normalize_gp_kernel

    raw = idata.posterior.attrs.get("universal_kernel")
    if raw is None:
        raw = idata.posterior.attrs.get("gp_kernel", DEFAULT_GP_KERNEL)
    return normalize_gp_kernel(str(raw))


def _max_abs_z_from_attrs(idata: az.InferenceData) -> float | None:
    raw = idata.posterior.attrs.get("max_abs_z")
    if raw is None:
        return None
    value = float(raw)
    if value <= 0.0:
        raise ValueError(f"max_abs_z must be positive, got {value}")
    return value


def _fit_window_mask(
    df: pd.DataFrame,
    max_abs_z: float | None,
) -> np.ndarray | None:
    """Boolean mask of rows in the fit window (provisional z at exact T_c, nu)."""
    if max_abs_z is None:
        return None
    L = df["L"].to_numpy(dtype=np.float64)
    T = df["T"].to_numpy(dtype=np.float64)
    z_ref = z_from_LT(L, T, T_c=TC_EXACT, nu=NU_EXACT)
    return np.abs(z_ref) <= max_abs_z


def _z_grid_for_fit_window(max_abs_z: float, *, n_points: int = 200) -> np.ndarray:
    pad = 0.05 * (2.0 * max_abs_z)
    return np.linspace(-max_abs_z - pad, max_abs_z + pad, n_points)


def _fit_window_title_suffix(max_abs_z: float | None) -> str:
    if max_abs_z is None:
        return ""
    return rf" ($|z|\leq {max_abs_z:g}$ fit window)"


def _df_for_fss_plot(df: pd.DataFrame, idata: az.InferenceData) -> pd.DataFrame:
    """Observables rows included in the FSS fit (respects max_abs_z)."""
    mask = _fit_window_mask(df, _max_abs_z_from_attrs(idata))
    if mask is None:
        return df
    return df.loc[mask].reset_index(drop=True)


def _is_crossover_model(idata: az.InferenceData) -> bool:
    return _posterior_model(idata) in ("scaling_crossover", "scaling_crossover_tc", "scaling_crossover_fss")


def _crossover_gp_hyperparams(idata: az.InferenceData) -> dict[str, float]:
    attrs = idata.posterior.attrs
    return {
        "gp_ell": float(attrs["gp_ell_fixed"]),
        "gp_eta": float(attrs["gp_eta_fixed"]),
        "correction_gp_ell": float(attrs["correction_gp_ell_fixed"]),
        "correction_gp_eta": float(attrs["correction_gp_eta_fixed"]),
        "bulk_lo_gp_ell": float(attrs["bulk_lo_gp_ell_fixed"]),
        "bulk_lo_gp_eta": float(attrs["bulk_lo_gp_eta_fixed"]),
        "bulk_hi_gp_ell": float(attrs["bulk_hi_gp_ell_fixed"]),
        "bulk_hi_gp_eta": float(attrs["bulk_hi_gp_eta_fixed"]),
    }


def _crossover_Tc_nu(
    idata: az.InferenceData,
    chain: int,
    draw: int,
) -> tuple[float, float]:
    if "T_c" in idata.posterior:
        T_c = float(idata.posterior["T_c"].values[chain, draw])
    else:
        T_c = TC_EXACT
    if "nu" in idata.posterior:
        nu = float(idata.posterior["nu"].values[chain, draw])
    else:
        nu = NU_EXACT
    return T_c, nu


def _crossover_r_value(idata: az.InferenceData) -> float:
    if "crossover_r" in idata.posterior:
        return float(idata.posterior["crossover_r"].mean().values)
    raw = idata.posterior.attrs.get("crossover_r")
    if raw is not None:
        return float(raw)
    return 1.0


def _crossover_sharpness(idata: az.InferenceData) -> float:
    raw = idata.posterior.attrs.get("crossover_sharpness")
    if raw is not None:
        return float(raw)
    return 0.2


def _crossover_bulk_log_m(
    T: np.ndarray,
    *,
    T_lo_unique: np.ndarray,
    log_m_lo: np.ndarray,
    bulk_lo_ell: float,
    bulk_lo_eta: float,
    T_hi_unique: np.ndarray,
    log_m_hi: np.ndarray,
    bulk_hi_ell: float,
    bulk_hi_eta: float,
) -> np.ndarray:
    """Bulk log m(T) from below/above T_c GPs."""
    T = np.asarray(T, dtype=np.float64)
    out = np.zeros_like(T)
    lo = T < TC_EXACT
    hi = ~lo
    if lo.any():
        out[lo] = gp_posterior_mean(
            T_lo_unique,
            log_m_lo,
            np.zeros_like(log_m_lo),
            T[lo],
            length_scale=bulk_lo_ell,
            amplitude=bulk_lo_eta,
        )
    if hi.any():
        out[hi] = gp_posterior_mean(
            T_hi_unique,
            log_m_hi,
            np.zeros_like(log_m_hi),
            T[hi],
            length_scale=bulk_hi_ell,
            amplitude=bulk_hi_eta,
        )
    return out


def _crossover_predict_components(
    idata: az.InferenceData,
    chain: int,
    draw: int,
    df: pd.DataFrame,
    T: np.ndarray,
    L: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (Phi_m, |m|, m_scal, m_bulk, w) at (T, L) from one forward pass.

    Scaling: GP conditional mean of phi0/phi1 from z_ref knots to z(T, L; nu).
    Bulk: GP conditional mean in T from bulk knot values.
    Mixture: m = w m_scal + (1 - w) m_bulk with the same rule at every query point.
    """
    T = np.asarray(T, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    omega = _omega_from_draw(idata, {})
    sharpness = _crossover_sharpness(idata)
    r = float(idata.posterior["crossover_r"].values[chain, draw])

    z_ref, _, _ = collapsed_at_exact_exponents(
        df["L"].to_numpy(dtype=np.float64),
        df["T"].to_numpy(dtype=np.float64),
        df["magnetization"].to_numpy(dtype=np.float64),
    )
    phi0_train = idata.posterior["phi0"].values[chain, draw].astype(np.float64)
    phi1_train = idata.posterior["phi1"].values[chain, draw].astype(np.float64)

    T_c, nu = _crossover_Tc_nu(idata, chain, draw)
    if "beta" in idata.posterior:
        beta = float(idata.posterior["beta"].values[chain, draw])
    else:
        beta = BETA_EXACT
    t = (T - T_c) / T_c
    z = t * L ** (1.0 / nu)
    hp = _crossover_gp_hyperparams(idata)
    gp_ell = hp["gp_ell"]
    gp_eta = hp["gp_eta"]
    corr_ell = hp["correction_gp_ell"]
    corr_eta = hp["correction_gp_eta"]

    phi0 = gp_posterior_mean(
        z_ref,
        phi0_train,
        np.zeros_like(phi0_train),
        z,
        length_scale=gp_ell,
        amplitude=gp_eta,
    )
    phi1 = gp_posterior_mean(
        z_ref,
        phi1_train,
        np.zeros_like(phi1_train),
        z,
        length_scale=corr_ell,
        amplitude=corr_eta,
    )

    T_df = df["T"].to_numpy(dtype=np.float64)
    lo_mask = T_df < TC_EXACT
    hi_mask = ~lo_mask
    T_lo_unique = np.unique(T_df[lo_mask])
    T_hi_unique = np.unique(T_df[hi_mask])
    log_m_lo = idata.posterior["log_m_bulk_lo"].values[chain, draw].astype(np.float64)
    log_m_hi = idata.posterior["log_m_bulk_hi"].values[chain, draw].astype(np.float64)
    log_m_bulk = _crossover_bulk_log_m(
        T,
        T_lo_unique=T_lo_unique,
        log_m_lo=log_m_lo,
        bulk_lo_ell=hp["bulk_lo_gp_ell"],
        bulk_lo_eta=hp["bulk_lo_gp_eta"],
        T_hi_unique=T_hi_unique,
        log_m_hi=log_m_hi,
        bulk_hi_ell=hp["bulk_hi_gp_ell"],
        bulk_hi_eta=hp["bulk_hi_gp_eta"],
    )

    phi_scal = phi0 + L ** (-omega) * phi1
    m_scal = phi_scal * L ** (-beta / nu)
    m_bulk = np.exp(log_m_bulk)
    w = crossover_weight_grid(np.abs(z), r, sharpness=sharpness)
    m = w * m_scal + (1.0 - w) * m_bulk
    phi_m = m * L ** (beta / nu)
    return phi_m, m, m_scal, m_bulk, w


def _crossover_predict(
    idata: az.InferenceData,
    chain: int,
    draw: int,
    df: pd.DataFrame,
    T: np.ndarray,
    L: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (Phi_m, |m|) from the crossover model at (T, L)."""
    phi_m, m, _, _, _ = _crossover_predict_components(idata, chain, draw, df, T, L)
    return phi_m, m


def _crossover_T_slice_from_posterior(
    idata: az.InferenceData,
    T_grid: np.ndarray,
    L: int,
    sample_idx: np.ndarray,
    *,
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Posterior samples of |m|, m_scal, m_bulk, and w at fixed L."""
    n_samples = sample_idx.size
    n_T = T_grid.size
    m_samples = np.empty((n_samples, n_T), dtype=np.float64)
    m_scal_samples = np.empty((n_samples, n_T), dtype=np.float64)
    m_bulk_samples = np.empty((n_samples, n_T), dtype=np.float64)
    w_samples = np.empty((n_samples, n_T), dtype=np.float64)
    L_test = np.full_like(T_grid, float(L))

    for row, flat_idx in enumerate(sample_idx):
        chain = flat_idx // idata.posterior.sizes["draw"]
        draw = flat_idx % idata.posterior.sizes["draw"]
        _, m, m_scal, m_bulk, w = _crossover_predict_components(
            idata, chain, draw, df, T_grid, L_test
        )
        m_samples[row] = m
        m_scal_samples[row] = m_scal
        m_bulk_samples[row] = m_bulk
        w_samples[row] = w
    return m_samples, m_scal_samples, m_bulk_samples, w_samples


def _shade_crossover_regimes(
    ax: plt.Axes,
    T_grid: np.ndarray,
    w_median: np.ndarray,
    *,
    y_lo: float,
    y_hi: float,
) -> None:
    """Background tint: green = scaling (w~1), purple = bulk (w~0)."""
    cmap = plt.cm.RdYlGn
    for i in range(T_grid.size - 1):
        ax.axvspan(
            T_grid[i],
            T_grid[i + 1],
            ymin=0.0,
            ymax=1.0,
            facecolor=cmap(float(w_median[i])),
            alpha=0.18,
            zorder=0,
        )
    ax.set_ylim(y_lo, y_hi)


def _crossover_T_boundaries(
    L: float,
    r: float,
    *,
    T_c: float = TC_EXACT,
    nu: float = NU_EXACT,
) -> tuple[float, float]:
    """Temperatures where |z| = r at fixed L (w = 1/2 contour)."""
    delta_t = r / max(L ** (1.0 / nu), 1e-12)
    return T_c * (1.0 - delta_t), T_c * (1.0 + delta_t)


def plot_crossover_magnetization_regimes(
    df: pd.DataFrame,
    idata: az.InferenceData,
    out_dir: Path,
    *,
    n_samples: int = 300,
    hdi_prob: float = 0.94,
    seed: int = 0,
    T_grid_points: int = 250,
) -> Path:
    """|m|(T) at each L with shaded scaling vs bulk crossover regions."""
    L_values = sorted(int(L) for L in df["L"].unique())
    n_L = len(L_values)
    fig, axes = plt.subplots(
        1, n_L, figsize=(5.5 * n_L, 4.5), constrained_layout=True, sharey=True
    )
    if n_L == 1:
        axes = [axes]

    r = _crossover_r_value(idata)
    means = _exponent_values(idata)
    nu = means["nu"]
    T_c_mean = _Tc_posterior_summary(idata, hdi_prob=hdi_prob)["mean"]
    sample_idx = _draw_indices(idata, n_samples, seed)
    palette = {"scal": "#55a868", "bulk": "#8172b3", "mix": "#c44e52"}

    for panel, (ax, L) in enumerate(zip(axes, L_values)):
        T_grid = _T_grid_for_L(df, L, n_points=T_grid_points)
        m_s, m_scal_s, m_bulk_s, w_s = _crossover_T_slice_from_posterior(
            idata, T_grid, L, sample_idx, df=df
        )
        m_lo, m_hi, m_mean = _posterior_mean_band(m_s, hdi_prob)
        _, _, m_scal_med = _percentile_band(m_scal_s, hdi_prob)
        _, _, m_bulk_med = _percentile_band(m_bulk_s, hdi_prob)
        _, _, w_med = _percentile_band(w_s, hdi_prob)

        y_lo = 0.0
        y_hi = float(max(m_hi.max(), m_scal_med.max(), m_bulk_med.max()) * 1.05)
        _shade_crossover_regimes(ax, T_grid, w_med, y_lo=y_lo, y_hi=y_hi)

        T_lo, T_hi = _crossover_T_boundaries(float(L), r, T_c=T_c_mean, nu=nu)
        tc_label = (
            rf"$T_c={T_c_mean:.3f}$ (posterior mean)"
            if panel == 0
            else None
        )
        w_half_label = r"$|z|=r$ ($w=1/2$)" if panel == 0 else None
        ax.axvline(
            T_c_mean,
            color="#333333",
            linestyle=":",
            linewidth=1.5,
            label=tc_label,
            zorder=1,
        )
        ax.axvline(
            T_lo,
            color=palette["scal"],
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            label=w_half_label,
            zorder=1,
        )
        ax.axvline(
            T_hi,
            color=palette["scal"],
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            zorder=1,
        )

        _plot_posterior_band(
            ax,
            T_grid,
            m_lo,
            m_hi,
            m_mean,
            hdi_prob=hdi_prob,
            label_band=True,
            central_label="posterior mean",
        )
        ax.plot(
            T_grid,
            m_scal_med,
            color=palette["scal"],
            linestyle="--",
            linewidth=1.8,
            label=r"$m_{\mathrm{scal}}(T)$",
            zorder=3,
        )
        ax.plot(
            T_grid,
            m_bulk_med,
            color=palette["bulk"],
            linestyle="--",
            linewidth=1.8,
            label=r"$m_{\mathrm{bulk}}(T)$",
            zorder=3,
        )

        obs_idx = _obs_indices_for_L(df, L)
        T_obs = df["T"].to_numpy(dtype=np.float64)[obs_idx]
        m_obs = df["magnetization"].to_numpy(dtype=np.float64)[obs_idx]
        m_sigma = df["magnetization_std"].to_numpy(dtype=np.float64)[obs_idx] / np.sqrt(
            np.maximum(df["n_eff"].to_numpy(dtype=np.float64)[obs_idx], 1.0)
        )
        ax.errorbar(
            T_obs,
            m_obs,
            yerr=m_sigma,
            fmt="o",
            color="#4c72b0",
            ecolor="#4c72b0",
            capsize=3,
            markersize=7,
            label="Ising MC",
            zorder=5,
        )

        ax.set_xlabel(r"$T$")
        ax.set_ylabel(r"$|m|$")
        ax.set_title(rf"$L={L}$: crossover model")
        ax.legend(loc="best", fontsize=7)

    fig.suptitle(
        rf"Magnetization vs $T$ with scaling (green) / bulk (red) regions "
        rf"($r={r:.2f}$, $T_c={T_c_mean:.3f}$, $\nu={nu:.3f}$)",
        fontsize=11,
        y=1.02,
    )
    path = out_dir / "crossover_magnetization_regimes.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _crossover_phi_summary_at_obs(
    idata: az.InferenceData,
    df: pd.DataFrame,
    obs_idx: np.ndarray,
    *,
    hdi_prob: float = 0.94,
) -> dict[str, np.ndarray]:
    L = df["L"].to_numpy(dtype=np.float64)[obs_idx]
    T = df["T"].to_numpy(dtype=np.float64)[obs_idx]
    n_chain = idata.posterior.sizes["chain"]
    n_draw = idata.posterior.sizes["draw"]
    phi = np.empty((n_chain, n_draw, obs_idx.size), dtype=np.float64)
    for c in range(n_chain):
        for d in range(n_draw):
            phi[c, d], _ = _crossover_predict(idata, c, d, df, T, L)
    alpha = (1.0 - hdi_prob) / 2.0
    lo_pct, hi_pct = 100.0 * alpha, 100.0 * (1.0 - alpha)
    return {
        "phi_mean": phi.mean(axis=(0, 1)),
        "phi_lo": np.percentile(phi, lo_pct, axis=(0, 1)),
        "phi_hi": np.percentile(phi, hi_pct, axis=(0, 1)),
    }


def _exponent_values(idata: az.InferenceData) -> dict[str, float]:
    if _posterior_model(idata) in ("scaling_only", "scaling_tl"):
        return {"T_c": TC_EXACT, "nu": NU_EXACT, "beta": BETA_EXACT}
    if _posterior_model(idata) == "scaling_nu":
        return {
            "T_c": TC_EXACT,
            "nu": float(idata.posterior["nu"].mean().values),
            "beta": BETA_EXACT,
        }
    if _posterior_model(idata) == "scaling_nu_corr":
        return {
            "T_c": TC_EXACT,
            "nu": float(idata.posterior["nu"].mean().values),
            "beta": BETA_EXACT,
        }
    if _posterior_model(idata) in ("scaling_nu_corr_hyper", "scaling_nu_corr_hyper_omega2"):
        return {
            "T_c": TC_EXACT,
            "nu": float(idata.posterior["nu"].mean().values),
            "beta": BETA_EXACT,
        }
    if _posterior_model(idata) == "scaling_corr_hyper_omega2_tc":
        return {
            "T_c": float(idata.posterior["T_c"].mean().values),
            "nu": float(idata.posterior["nu"].mean().values),
            "beta": BETA_EXACT,
        }
    if _posterior_model(idata) in (
        "scaling_corr_hyper_omega2_fss",
        "scaling_corr_hyper_fss",
    ):
        return {
            "T_c": float(idata.posterior["T_c"].mean().values),
            "nu": _posterior_mean_or_exact(idata, "nu"),
            "beta": _posterior_mean_or_exact(idata, "beta"),
        }
    if _posterior_model(idata) == "scaling_crossover":
        return {
            "T_c": TC_EXACT,
            "nu": NU_EXACT,
            "beta": BETA_EXACT,
            "crossover_r": _crossover_r_value(idata),
        }
    if _posterior_model(idata) == "scaling_crossover_tc":
        return {
            "T_c": TC_EXACT,
            "nu": float(idata.posterior["nu"].mean().values),
            "beta": BETA_EXACT,
            "crossover_r": _crossover_r_value(idata),
        }
    if _posterior_model(idata) == "scaling_crossover_fss":
        return {
            "T_c": TC_EXACT,
            "nu": _posterior_mean_or_exact(idata, "nu"),
            "beta": _posterior_mean_or_exact(idata, "beta"),
            "crossover_r": _crossover_r_value(idata),
        }
    if _posterior_model(idata) in ("scaling_tc", "scaling_tl_tc"):
        return {
            "T_c": float(idata.posterior["T_c"].mean().values),
            "nu": NU_EXACT,
            "beta": BETA_EXACT,
        }
    if _is_fss_tc_model(idata):
        T_c = (
            TC_EXACT
            if _fss_fixed_Tc(idata)
            else float(idata.posterior["T_c"].mean().values)
        )
        return {
            "T_c": T_c,
            "nu": _posterior_mean_or_exact(idata, "nu"),
            "beta": _posterior_mean_or_exact(idata, "beta"),
            **(
                {"gamma": float(idata.posterior["gamma"].mean().values)}
                if _posterior_has(idata, "gamma")
                else {}
            ),
        }
    return {
        "T_c": float(idata.posterior["T_c"].mean().values),
        "nu": _posterior_mean_or_exact(idata, "nu"),
        "beta": _posterior_mean_or_exact(idata, "beta"),
    }


def _log_phi_fixed_at_obs(idata: az.InferenceData) -> bool:
    """Whether collapsed log_phi at MC points is independent of the posterior draw."""
    return _posterior_model(idata) in (
        "scaling_only",
        "scaling_tc",
        "scaling_tl",
        "scaling_tl_tc",
    )


def _posterior_scaling(idata: az.InferenceData) -> ScalingBackend:
    return str(idata.posterior.attrs.get("scaling", "gp"))  # type: ignore[return-value]


def _is_singular_model(idata: az.InferenceData) -> bool:
    return _posterior_variant(idata) == "singular"


def _scaling_log_target(idata: az.InferenceData) -> str:
    if _is_fss_tc_model(idata):
        return _fss_active_log_target(idata)
    return "log_psi" if _is_singular_model(idata) else "log_phi"


def _z_knots_from_attrs(idata: az.InferenceData) -> np.ndarray | None:
    raw = idata.posterior.attrs.get("z_knots")
    if raw is None:
        return None
    return np.asarray(json.loads(str(raw)), dtype=np.float64)


def _spline_degree_from_attrs(idata: az.InferenceData) -> int:
    raw = idata.posterior.attrs.get("spline_degree")
    if raw is None:
        return DEFAULT_SPLINE_DEGREE
    return int(raw)


def _log_phi_on_grid(
    z_grid: np.ndarray,
    log_psi_grid: np.ndarray,
    beta: float,
) -> np.ndarray:
    return singular_log_phi(z_grid, beta) + log_psi_grid


def _posterior_means(idata: az.InferenceData) -> dict[str, float]:
    out = _exponent_values(idata)
    if _is_tl_gp_model(idata):
        out["gp_ell_L"] = float(idata.posterior["gp_ell_L"].mean().values)
        out["gp_eta"] = float(idata.posterior["gp_eta"].mean().values)
        axis_name = "gp_ell_t" if _posterior_model(idata) == "scaling_tl_tc" else "gp_ell_T"
        out[axis_name] = float(idata.posterior[axis_name].mean().values)
    elif _is_fss_tc_model(idata):
        if "gp_ell_fixed" in idata.posterior.attrs:
            out["gp_ell_m"] = float(idata.posterior.attrs["gp_ell_fixed"])
            out["gp_eta_m"] = float(idata.posterior.attrs["gp_eta_fixed"])
            if _fss_include_binder(idata):
                out["gp_ell_u"] = float(idata.posterior.attrs["gp_ell_fixed"])
                out["gp_eta_u"] = float(idata.posterior.attrs["gp_eta_fixed"])
        m_names = ("gp_ell_m", "gp_eta_m", "correction_gp_ell_m", "correction_gp_eta_m")
        u_names = ("gp_ell_u", "gp_eta_u", "correction_gp_ell_u", "correction_gp_eta_u")
        for name in m_names + (u_names if _fss_include_binder(idata) else ()):
            if _posterior_has(idata, name):
                out[name] = float(idata.posterior[name].mean().values)
        out.update(_posterior_gp_means(idata))
        if (
            not _infer_gp_hyperparams(idata)
            and _is_fss_correction(idata)
            and "correction_gp_ell_fixed" in idata.posterior.attrs
        ):
            corr_ell = float(idata.posterior.attrs["correction_gp_ell_fixed"])
            corr_eta = float(idata.posterior.attrs["correction_gp_eta_fixed"])
            out["correction_gp_ell_m"] = corr_ell
            out["correction_gp_eta_m"] = corr_eta
            if _fss_include_binder(idata):
                out["correction_gp_ell_u"] = corr_ell
                out["correction_gp_eta_u"] = corr_eta
    elif _posterior_scaling(idata) == "gp" and "gp_ell" in idata.posterior:
        out["gp_ell"] = float(idata.posterior["gp_ell"].mean().values)
        out["gp_eta"] = float(idata.posterior["gp_eta"].mean().values)
    else:
        out.update(_posterior_gp_means(idata))
    if _is_crossover_model(idata):
        out["crossover_r"] = _crossover_r_value(idata)
        out.update(_crossover_gp_hyperparams(idata))
    return out


def _posterior_exponent_panel_specs(
    idata: az.InferenceData,
) -> list[tuple[str, str, float | None, str]]:
    """Return (var_name, title, reference_value, reference_label) for histogram panels."""
    specs: list[tuple[str, str, float | None, str]] = []
    for name in ("T_c", "nu", "beta", "gamma"):
        if _posterior_has(idata, name):
            ref = EXACT_VALUES.get(name)
            label = "exact" if ref is not None else ""
            specs.append((name, name, ref, label))

    if _posterior_has(idata, "omega"):
        specs.append(("omega", r"$\omega$", OMEGA_EXACT, "exact"))
    elif _posterior_has(idata, "disc_q"):
        form = str(idata.posterior.attrs.get("discrepancy_form", ""))
        if form == "additive_gp_fss":
            specs.append(("disc_q", r"$\omega$", OMEGA_EXACT, "exact"))
        else:
            specs.append(("disc_q", r"$q$", None, ""))

    gp_specs = (
        ("gp_ell", r"$\ell_0$", "gp_ell_fixed", "gp_ell_prior_mu"),
        ("gp_eta", r"$\eta_0$", "gp_eta_fixed", None),
        ("correction_gp_ell", r"$\ell_1$", "correction_gp_ell_fixed", "correction_gp_ell_prior_mu"),
        ("correction_gp_eta", r"$\eta_1$", "correction_gp_eta_fixed", None),
    )
    for name, title, fixed_attr, prior_attr in gp_specs:
        if not _posterior_has(idata, name):
            continue
        ref = _lognormal_prior_median(idata, prior_attr) if prior_attr else None
        if ref is None and fixed_attr in idata.posterior.attrs:
            ref = float(idata.posterior.attrs[fixed_attr])
        label = "prior median" if prior_attr and ref is not None else "nominal"
        specs.append((name, title, ref, label))
    return specs


def _plot_tc_nu_joint_heatmap(
    ax: plt.Axes,
    idata: az.InferenceData,
    *,
    fig: plt.Figure,
) -> None:
    """Joint posterior density of ``T_c`` and ``nu`` on ``ax``."""
    tc = idata.posterior["T_c"].values.reshape(-1)
    nu = idata.posterior["nu"].values.reshape(-1)
    _, _, _, mesh = ax.hist2d(tc, nu, bins=100, cmap="Blues", density=True)
    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04, label="density")
    tc_ex = EXACT_VALUES["T_c"]
    nu_ex = EXACT_VALUES["nu"]
    ax.plot(
        tc_ex,
        nu_ex,
        marker="*",
        color="black",
        markersize=10,
        linestyle="none",
        label="exact",
    )
    ax.set_xlabel(r"$T_c$")
    ax.set_ylabel(r"$\nu$")
    ax.set_title(r"$T_c$ vs $\nu$")
    ax.legend(loc="best", fontsize=8)


def plot_exponent_histograms(idata: az.InferenceData, out_dir: Path) -> Path | None:
    specs = _posterior_exponent_panel_specs(idata)
    if not specs:
        return None
    show_joint = _posterior_has(idata, "T_c") and _posterior_has(idata, "nu")
    n = len(specs) + (1 if show_joint else 0)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 3.5 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    axes_flat = axes.ravel()
    panel = 0
    if show_joint:
        _plot_tc_nu_joint_heatmap(axes_flat[0], idata, fig=fig)
        panel = 1
    for ax, (name, title, ref, ref_label) in zip(axes_flat[panel:], specs):
        samples = idata.posterior[name].values.reshape(-1)
        ax.hist(samples, bins=100, color="#4c72b0", alpha=0.85, density=True)
        if ref is not None:
            ax.axvline(
                ref,
                color="black",
                linestyle="--",
                linewidth=1.5,
                label=ref_label,
            )
            ax.legend(loc="best", fontsize=8)
        ax.set_title(title)
        ax.set_xlabel(name)
        ax.set_ylabel("density")
    for ax in axes_flat[n:]:
        ax.axis("off")
    path = out_dir / "posterior_exponents.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _lognormal_prior_median(idata: az.InferenceData, attr: str) -> float | None:
    raw = idata.posterior.attrs.get(attr)
    if raw is None:
        return None
    return float(np.exp(float(raw)))


def plot_fss_gp_hyperparameter_histograms(
    idata: az.InferenceData,
    out_dir: Path,
) -> Path | None:
    """Posterior histograms for fss_nu / fss_binder GP hyperparameters."""
    if not _is_fss_correction(idata):
        return None
    specs = (
        ("gp_ell_m", "gp_ell_prior_mu", r"$\Phi_m\,\ell_0$"),
        ("gp_eta_m", "gp_eta_prior_mu", r"$\Phi_m\,\eta_0$"),
    )
    if _fss_include_m(idata) and _fss_channel_correction(idata, "m"):
        specs = specs + (
            ("correction_gp_ell_m", "correction_gp_ell_prior_mu", r"$\Phi_m\,\ell_1$"),
            ("correction_gp_eta_m", "correction_gp_eta_prior_mu", r"$\Phi_m\,\eta_1$"),
        )
    if _fss_include_binder(idata):
        specs = specs + (
            ("gp_ell_u", "gp_ell_prior_mu", r"$U_4\,\ell_0$"),
            ("gp_eta_u", "gp_eta_prior_mu", r"$U_4\,\eta_0$"),
        )
        if _fss_channel_correction(idata, "binder"):
            specs = specs + (
                ("correction_gp_ell_u", "correction_gp_ell_prior_mu", r"$U_4\,\ell_1$"),
                ("correction_gp_eta_u", "correction_gp_eta_prior_mu", r"$U_4\,\eta_1$"),
            )
    present = [
        (name, prior_attr, label)
        for name, prior_attr, label in specs
        if _posterior_has(idata, name)
    ]
    if not present:
        return None
    ncols = 4
    nrows = int(np.ceil(len(present) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, (name, prior_attr, label) in zip(axes_flat, present):
        samples = idata.posterior[name].values.reshape(-1)
        ax.hist(samples, bins=100, color="#4c72b0", alpha=0.85, density=True)
        prior_median = _lognormal_prior_median(idata, prior_attr)
        if prior_median is not None:
            ax.axvline(
                prior_median,
                color="black",
                linestyle="--",
                linewidth=1.5,
                label="prior median",
            )
        ax.set_title(label)
        ax.set_xlabel(name)
        ax.set_ylabel("density")
    for ax in axes_flat[len(present) :]:
        ax.axis("off")
    path = out_dir / "gp_hyperparameters.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_gp_hyperparameter_histograms(idata: az.InferenceData, out_dir: Path) -> Path:
    """Posterior histograms for GP length scales and amplitudes."""
    specs = (
        ("gp_ell", "gp_ell_prior_mu", r"$\ell_0$"),
        ("gp_eta", "gp_eta_prior_mu", r"$\eta_0$"),
        ("correction_gp_ell", "correction_gp_ell_prior_mu", r"$\ell_1$"),
        ("correction_gp_eta", "correction_gp_eta_prior_mu", r"$\eta_1$"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    for ax, (name, prior_attr, label) in zip(axes.ravel(), specs):
        samples = idata.posterior[name].values.reshape(-1)
        ax.hist(samples, bins=100, color="#4c72b0", alpha=0.85, density=True)
        prior_median = _lognormal_prior_median(idata, prior_attr)
        if prior_median is not None:
            ax.axvline(
                prior_median,
                color="black",
                linestyle="--",
                linewidth=1.5,
                label="prior median",
            )
        ax.set_title(label)
        ax.set_xlabel(name)
        ax.set_ylabel("density")
        if prior_median is not None:
            ax.legend(loc="best", fontsize=8)
    path = out_dir / "posterior_gp_hyperparams.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_crossover_histogram(idata: az.InferenceData, out_dir: Path) -> Path:
    """Posterior histogram for crossover scale r in |z|-space."""
    samples = idata.posterior["crossover_r"].values.reshape(-1)
    prior_mu = idata.posterior.attrs.get("crossover_r_prior_mu")
    fig, ax = plt.subplots(figsize=(4.5, 3.5), constrained_layout=True)
    ax.hist(samples, bins=100, color="#55a868", alpha=0.85, density=True)
    ax.axvline(float(np.median(samples)), color="black", linestyle="-", linewidth=1.5, label="median")
    if prior_mu is not None:
        ax.axvline(
            float(np.exp(float(prior_mu))),
            color="#999999",
            linestyle="--",
            linewidth=1.5,
            label="prior median",
        )
    ax.set_title(r"Crossover location $r$")
    ax.set_xlabel(r"$r$  ($|z|=r$ gives $w=1/2$)")
    ax.set_ylabel("density")
    ax.legend(loc="best", fontsize=8)
    path = out_dir / "crossover_location.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_crossover_window(idata: az.InferenceData, out_dir: Path) -> Path:
    """Logistic crossover window w(|z|) with posterior uncertainty on r."""
    sharpness = _crossover_sharpness(idata)
    r_samples = idata.posterior["crossover_r"].values.reshape(-1)
    abs_z_max = max(float(np.percentile(r_samples, 97.5)) * 1.75, 1.0)
    abs_z_grid = np.linspace(0.0, abs_z_max, 250)

    curves = np.array(
        [
            crossover_weight_grid(abs_z_grid, float(r), sharpness=sharpness)
            for r in r_samples[:: max(1, len(r_samples) // 200)]
        ]
    )
    median_curve = np.median(curves, axis=0)
    lo = np.percentile(curves, 2.5, axis=0)
    hi = np.percentile(curves, 97.5, axis=0)
    r_med = float(np.median(r_samples))

    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    ax.fill_between(abs_z_grid, lo, hi, color="#55a868", alpha=0.25, label="94% band")
    ax.plot(abs_z_grid, median_curve, color="#55a868", linewidth=2.0, label="posterior median")
    ax.axvline(r_med, color="black", linestyle="--", linewidth=1.5, label=rf"$r={r_med:.2f}$")
    ax.axhline(0.5, color="#999999", linestyle=":", linewidth=1.0)
    ax.set_xlim(0.0, abs_z_max)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r"$|z|$")
    ax.set_ylabel(r"$w(|z|)$")
    ax.set_title("Scaling / bulk crossover window")
    ax.legend(loc="best", fontsize=8)
    path = out_dir / "crossover_window.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_crossover_map(
    df: pd.DataFrame,
    idata: az.InferenceData,
    out_dir: Path,
) -> Path:
    """Data grid in (T, L) with weights and the |z|=r crossover contour."""
    means = _exponent_values(idata)
    T_c = means["T_c"]
    nu = means["nu"]
    r = _crossover_r_value(idata)
    sharpness = _crossover_sharpness(idata)

    L = df["L"].to_numpy(dtype=np.float64)
    T = df["T"].to_numpy(dtype=np.float64)
    t = (T - T_c) / T_c
    z = t * L ** (1.0 / nu)
    abs_z = np.abs(z)
    w = crossover_weight_grid(abs_z, r, sharpness=sharpness)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    sc = axes[0].scatter(T, L, c=w, cmap="viridis", s=70, edgecolors="k", linewidths=0.4)
    plt.colorbar(sc, ax=axes[0], label=r"$w(t,L)$")
    L_line = np.linspace(float(L.min()), float(L.max()), 200)
    delta_t = r / np.maximum(L_line ** (1.0 / nu), 1e-12)
    T_upper = T_c * (1.0 + delta_t)
    T_lower = T_c * (1.0 - delta_t)
    axes[0].plot(T_upper, L_line, "r--", linewidth=1.5, label=r"$|z|=r$")
    axes[0].plot(T_lower, L_line, "r--", linewidth=1.5)
    axes[0].axvline(T_c, color="#999999", linestyle=":", linewidth=1.0)
    axes[0].set_xlabel(r"$T$")
    axes[0].set_ylabel(r"$L$")
    axes[0].set_title(r"Crossover weight on data grid")
    axes[0].legend(loc="best", fontsize=8)

    axes[1].scatter(abs_z, w, c=L, cmap="tab10", s=70, edgecolors="k", linewidths=0.4)
    axes[1].axvline(r, color="red", linestyle="--", linewidth=1.5, label=rf"$r={r:.2f}$")
    abs_z_line = np.linspace(0.0, max(abs_z.max(), r) * 1.2, 200)
    axes[1].plot(
        abs_z_line,
        crossover_weight_grid(abs_z_line, r, sharpness=sharpness),
        color="black",
        linewidth=1.5,
        label=r"$w(|z|)$",
    )
    axes[1].set_xlabel(r"$|z|$")
    axes[1].set_ylabel(r"$w$")
    axes[1].set_title(r"Weight vs scaling distance")
    axes[1].legend(loc="best", fontsize=8)

    path = out_dir / "crossover_map.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_bulk_magnetization_gp(
    df: pd.DataFrame,
    idata: az.InferenceData,
    out_dir: Path,
) -> Path:
    """Posterior bulk magnetization GPs below and above T_c."""
    T = df["T"].to_numpy(dtype=np.float64)
    lo_mask = T < TC_EXACT
    hi_mask = ~lo_mask

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    panels = (
        (axes[0], lo_mask, "log_m_bulk_lo", "bulk_lo_gp_ell", "bulk_lo_gp_eta", "Bulk below T_c"),
        (axes[1], hi_mask, "log_m_bulk_hi", "bulk_hi_gp_ell", "bulk_hi_gp_eta", "Bulk above T_c"),
    )
    for ax, mask, latent_name, ell_name, eta_name, title in panels:
        T_train = np.unique(T[mask])
        latent = idata.posterior[latent_name].mean(dim=("chain", "draw")).values.astype(np.float64)
        hp = _crossover_gp_hyperparams(idata)
        ell = hp[ell_name]
        eta = hp[eta_name]
        T_grid = np.linspace(float(T_train.min()), float(T_train.max()), 200)
        mean_log, std_log = gp_posterior_predictive(
            T_train,
            latent,
            np.full(T_train.shape, 1e-3),
            T_grid,
            length_scale=ell,
            amplitude=eta,
        )
        m_mean = np.exp(mean_log)
        m_lo = np.exp(mean_log - 1.96 * std_log)
        m_hi = np.exp(mean_log + 1.96 * std_log)
        ax.fill_between(T_grid, m_lo, m_hi, color="#8172b3", alpha=0.25)
        ax.plot(T_grid, m_mean, color="#8172b3", linewidth=2.0, label="GP bulk")
        if "below" in title:
            ax.plot(
                T_grid,
                magnetization_infinite_L(T_grid),
                color="black",
                linestyle="--",
                linewidth=1.2,
                label=r"Onsager $m_\infty(T)$",
            )
        ax.scatter(
            T_train,
            np.exp(latent),
            s=50,
            color="#4c72b0",
            edgecolors="k",
            linewidths=0.4,
            zorder=4,
            label="latent at data",
        )
        ax.set_xlabel(r"$T$")
        ax.set_ylabel(r"$|m|$")
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)

    path = out_dir / "bulk_magnetization_gp.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_crossover_gp_hyperparameter_histograms(
    idata: az.InferenceData, out_dir: Path
) -> Path:
    specs = (
        ("gp_ell", "gp_ell_fixed", r"$\Phi_0\,\ell$"),
        ("gp_eta", "gp_eta_fixed", r"$\Phi_0\,\eta$"),
        ("correction_gp_ell", "correction_gp_ell_fixed", r"$\Phi_1\,\ell$"),
        ("correction_gp_eta", "correction_gp_eta_fixed", r"$\Phi_1\,\eta$"),
        ("bulk_lo_gp_ell", "bulk_lo_gp_ell_fixed", r"bulk$_{-}\,\ell$"),
        ("bulk_lo_gp_eta", "bulk_lo_gp_eta_fixed", r"bulk$_{-}\,\eta$"),
        ("bulk_hi_gp_ell", "bulk_hi_gp_ell_fixed", r"bulk$_{+}\,\ell$"),
        ("bulk_hi_gp_eta", "bulk_hi_gp_eta_fixed", r"bulk$_{+}\,\eta$"),
    )
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    for ax, (name, fixed_attr, label) in zip(axes.ravel(), specs):
        fixed = idata.posterior.attrs.get(fixed_attr)
        if fixed is None:
            ax.set_visible(False)
            continue
        fixed_val = float(fixed)
        ax.axvline(
            fixed_val,
            color="#4c72b0",
            linewidth=2.0,
            label=f"fixed = {fixed_val:.4g}",
        )
        ax.set_xlim(0.5 * fixed_val, 1.5 * fixed_val)
        ax.set_title(label)
        ax.set_xlabel(name)
        ax.set_ylabel("")
        ax.legend(loc="best", fontsize=7)
    path = out_dir / "posterior_gp_hyperparams.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


_COLLAPSE_CHANNELS: tuple[tuple[str, str, int], ...] = (
    ("magnetization", r"$L^{\beta/\nu}\,|m|$", 1),
    ("m2", r"$L^{2\beta/\nu}\,\langle m^2\rangle$", 2),
    ("m4", r"$L^{4\beta/\nu}\,\langle m^4\rangle$", 4),
)


def _available_collapse_channels(df: pd.DataFrame) -> list[tuple[str, str, int]]:
    return [spec for spec in _COLLAPSE_CHANNELS if spec[0] in df.columns]


def _plot_data_collapse_scatter(
    df: pd.DataFrame,
    *,
    T_c: float,
    nu: float,
    beta: float,
    title: str,
    path: Path,
    channels: list[tuple[str, str, int]] | None = None,
) -> Path:
    if channels is None:
        channels = _available_collapse_channels(df)
        if not channels:
            raise ValueError("DataFrame has no collapse channels (magnetization, m2, m4)")

    t = (df["T"].to_numpy() - T_c) / T_c
    L = df["L"].to_numpy(dtype=np.float64)
    z = t * L ** (1.0 / nu)

    n_panels = len(channels)
    if n_panels == 1:
        fig, axes = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
        axes_list = [axes]
    else:
        fig, axes = plt.subplots(
            1,
            n_panels,
            figsize=(5.5 * n_panels, 4.5),
            constrained_layout=True,
            sharex=True,
        )
        axes_list = list(np.atleast_1d(axes))

    for ax, (column, ylabel, moment) in zip(axes_list, channels):
        values = df[column].to_numpy(dtype=np.float64)
        scaled = values * L ** (moment * beta / nu)
        for L_value in sorted(df["L"].unique()):
            mask = df["L"] == L_value
            ax.scatter(
                z[mask],
                scaled[mask],
                label=f"L={int(L_value)}",
                s=60,
                alpha=0.85,
            )
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    axes_list[-1].set_xlabel(r"$t L^{1/\nu}$")
    if n_panels > 1:
        fig.suptitle(title, fontsize=12)
    else:
        axes_list[0].set_title(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_data_collapse(
    df: pd.DataFrame,
    means: dict[str, float],
    out_dir: Path,
) -> Path:
    return _plot_data_collapse_scatter(
        df,
        T_c=means["T_c"],
        nu=means["nu"],
        beta=means["beta"],
        title="Data collapse at posterior mean exponents",
        path=out_dir / "data_collapse.png",
        channels=[("magnetization", r"$L^{\beta/\nu}\,|m|$", 1)],
    )


def plot_data_collapse_exact(df: pd.DataFrame, out_dir: Path) -> Path:
    """Empirical collapse using exact 2D Ising exponents (no fit)."""
    channels = _available_collapse_channels(df)
    return _plot_data_collapse_scatter(
        df,
        T_c=TC_EXACT,
        nu=NU_EXACT,
        beta=BETA_EXACT,
        title=(
            r"Data collapse at exact exponents "
            rf"($T_c={TC_EXACT:.4f}$, $\nu={NU_EXACT}$, $\beta={BETA_EXACT}$)"
        ),
        path=out_dir / "data_collapse_exact.png",
        channels=channels,
    )


def _draw_indices(idata: az.InferenceData, n_samples: int, seed: int) -> np.ndarray:
    total = idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"]
    rng = np.random.default_rng(seed)
    return rng.choice(total, size=min(n_samples, total), replace=False)


def _select_draw(idata: az.InferenceData, flat_index: int) -> dict[str, float | np.ndarray]:
    n_draw = idata.posterior.sizes["draw"]
    chain = flat_index // n_draw
    draw = flat_index % n_draw
    out: dict[str, float | np.ndarray] = dict(_exponent_values(idata))
    model = _posterior_model(idata)
    if model == "fss" and not _is_fss_gp_model(idata):
        for name in ("T_c", "nu", "beta"):
            out[name] = _posterior_draw_or_exact(idata, name, chain, draw)
        for name in _inferred_gp_var_names(idata):
            out[name] = float(idata.posterior[name].values[chain, draw])
    elif model in ("scaling_tc", "scaling_tl_tc"):
        out["T_c"] = float(idata.posterior["T_c"].values[chain, draw])
    elif _is_fss_gp_model(idata):
        out["T_c"] = (
            float(idata.posterior["T_c"].values[chain, draw])
            if _posterior_has(idata, "T_c")
            else TC_EXACT
        )
        out["nu"] = _posterior_draw_or_exact(idata, "nu", chain, draw)
        out["beta"] = _posterior_draw_or_exact(idata, "beta", chain, draw)
        channel_names = (
            "gp_ell_m",
            "gp_eta_m",
            "correction_gp_ell_m",
            "correction_gp_eta_m",
        )
        if _fss_include_binder(idata):
            channel_names = channel_names + (
                "gp_ell_u",
                "gp_eta_u",
                "correction_gp_ell_u",
                "correction_gp_eta_u",
            )
        for name in channel_names:
            if name in idata.posterior:
                out[name] = float(idata.posterior[name].values[chain, draw])
        for name in _inferred_gp_var_names(idata):
            out[name] = float(idata.posterior[name].values[chain, draw])
        attrs = idata.posterior.attrs
        if not _infer_gp_hyperparams(idata) and "gp_ell_fixed" in attrs:
            out["gp_ell_m"] = float(attrs["gp_ell_fixed"])
            out["gp_eta_m"] = float(attrs["gp_eta_fixed"])
            out["gp_ell"] = float(attrs["gp_ell_fixed"])
            out["gp_eta"] = float(attrs["gp_eta_fixed"])
            if _fss_include_binder(idata):
                out["gp_ell_u"] = float(attrs["gp_ell_fixed"])
                out["gp_eta_u"] = float(attrs["gp_eta_fixed"])
        if not _infer_gp_hyperparams(idata) and "correction_gp_ell_fixed" in attrs:
            corr_ell = float(attrs["correction_gp_ell_fixed"])
            corr_eta = float(attrs["correction_gp_eta_fixed"])
            out["correction_gp_ell_m"] = corr_ell
            out["correction_gp_eta_m"] = corr_eta
            out["correction_gp_ell"] = corr_ell
            out["correction_gp_eta"] = corr_eta
            if _fss_include_binder(idata):
                out["correction_gp_ell_u"] = corr_ell
                out["correction_gp_eta_u"] = corr_eta
        elif "gp_ell_m" in out:
            out["gp_ell"] = float(out["gp_ell_m"])
            out["gp_eta"] = float(out["gp_eta_m"])
        if "correction_gp_ell_m" in out:
            out["correction_gp_ell"] = float(out["correction_gp_ell_m"])
            out["correction_gp_eta"] = float(out["correction_gp_eta_m"])
    elif model in (
        "scaling_nu",
        "scaling_nu_corr",
        "scaling_nu_corr_hyper",
        "scaling_nu_corr_hyper_omega2",
        "scaling_corr_hyper_omega2_tc",
    ):
        out["nu"] = float(idata.posterior["nu"].values[chain, draw])
        if model == "scaling_corr_hyper_omega2_tc":
            out["T_c"] = float(idata.posterior["T_c"].values[chain, draw])
        if "omega" in idata.posterior:
            out["omega"] = float(idata.posterior["omega"].values[chain, draw])
        else:
            out["omega"] = _omega_from_draw(idata, out)
        if "gp_ell" in idata.posterior:
            out["gp_ell"] = float(idata.posterior["gp_ell"].values[chain, draw])
            out["gp_eta"] = float(idata.posterior["gp_eta"].values[chain, draw])
            out["correction_gp_ell"] = float(
                idata.posterior["correction_gp_ell"].values[chain, draw]
            )
            out["correction_gp_eta"] = float(
                idata.posterior["correction_gp_eta"].values[chain, draw]
            )
        else:
            out.update(_gp_params_from_attrs(idata))
    elif model == "scaling_corr_hyper_omega2_fss":
        for name in ("T_c", "nu", "beta"):
            if name == "T_c":
                out[name] = float(idata.posterior[name].values[chain, draw])
            else:
                out[name] = _posterior_draw_or_exact(idata, name, chain, draw)
        out["omega"] = _omega_from_draw(idata, out)
        out["gp_ell"] = float(idata.posterior["gp_ell"].values[chain, draw])
        out["gp_eta"] = float(idata.posterior["gp_eta"].values[chain, draw])
        out["correction_gp_ell"] = float(
            idata.posterior["correction_gp_ell"].values[chain, draw]
        )
        out["correction_gp_eta"] = float(
            idata.posterior["correction_gp_eta"].values[chain, draw]
        )
    elif model == "scaling_corr_hyper_fss":
        for name in ("T_c", "nu", "beta", "omega"):
            if name in ("T_c", "omega"):
                out[name] = float(idata.posterior[name].values[chain, draw])
            else:
                out[name] = _posterior_draw_or_exact(idata, name, chain, draw)
        out["gp_ell"] = float(idata.posterior["gp_ell"].values[chain, draw])
        out["gp_eta"] = float(idata.posterior["gp_eta"].values[chain, draw])
        out["correction_gp_ell"] = float(
            idata.posterior["correction_gp_ell"].values[chain, draw]
        )
        out["correction_gp_eta"] = float(
            idata.posterior["correction_gp_eta"].values[chain, draw]
        )
    elif model == "scaling_crossover":
        out["crossover_r"] = float(idata.posterior["crossover_r"].values[chain, draw])
        out.update(_crossover_gp_hyperparams(idata))
        out["T_c"] = TC_EXACT
        out["nu"] = NU_EXACT
        out["beta"] = BETA_EXACT
        out["omega"] = _omega_from_draw(idata, out)
    elif model == "scaling_crossover_tc":
        out["crossover_r"] = float(idata.posterior["crossover_r"].values[chain, draw])
        out["nu"] = float(idata.posterior["nu"].values[chain, draw])
        out.update(_crossover_gp_hyperparams(idata))
        out["T_c"] = TC_EXACT
        out["beta"] = BETA_EXACT
        out["omega"] = _omega_from_draw(idata, out)
    elif model == "scaling_crossover_fss":
        out["crossover_r"] = float(idata.posterior["crossover_r"].values[chain, draw])
        out["nu"] = _posterior_draw_or_exact(idata, "nu", chain, draw)
        out["beta"] = _posterior_draw_or_exact(idata, "beta", chain, draw)
        out.update(_crossover_gp_hyperparams(idata))
        out["T_c"] = TC_EXACT
        out["omega"] = _omega_from_draw(idata, out)
    if _is_tl_gp_model(idata):
        out["gp_ell_L"] = float(idata.posterior["gp_ell_L"].values[chain, draw])
        out["gp_eta"] = float(idata.posterior["gp_eta"].values[chain, draw])
        axis_name = "gp_ell_t" if model == "scaling_tl_tc" else "gp_ell_T"
        out[axis_name] = float(idata.posterior[axis_name].values[chain, draw])
    elif _posterior_scaling(idata) == "gp" and not _is_fss_tc_model(idata):
        if "gp_ell" in idata.posterior:
            out["gp_ell"] = float(idata.posterior["gp_ell"].values[chain, draw])
        if "gp_eta" in idata.posterior:
            out["gp_eta"] = float(idata.posterior["gp_eta"].values[chain, draw])
        out.update(_gp_params_from_attrs(idata))
    elif "spline_coeff" in idata.posterior:
        out["spline_coeff"] = idata.posterior["spline_coeff"].values[chain, draw].astype(
            np.float64
        )
    if "z" in idata.posterior:
        out["z"] = idata.posterior["z"].values[chain, draw].astype(np.float64)
    if "gamma" in idata.posterior:
        out["gamma"] = float(idata.posterior["gamma"].values[chain, draw])
    target = _scaling_log_target(idata)
    if target in idata.posterior:
        target_values = idata.posterior[target].values[chain, draw].astype(np.float64)
        out[target] = target_values
        if target == "log_psi":
            out["log_phi"] = _log_phi_on_grid(out["z"], target_values, float(out["beta"]))
        else:
            out["log_phi"] = target_values
    return out


def _posterior_has_scaling_trace(idata: az.InferenceData) -> bool:
    return "z" in idata.posterior


def _obs_scaling_at_points(
    df: pd.DataFrame,
    idata: az.InferenceData,
    draw: dict[str, float | np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Collapsed z and log Phi(z) at each observables row for one posterior draw."""
    L = df["L"].to_numpy(dtype=np.float64)
    T = df["T"].to_numpy(dtype=np.float64)
    z = z_from_LT(L, T, T_c=float(draw["T_c"]), nu=float(draw["nu"]))
    if _uses_marginal_gp_fss_plot(idata):
        if _fss_binder_only(idata):
            u4 = df["binder_cumulant"].to_numpy(dtype=np.float64)
            return z, u4
        log_phi = _marginal_gp_fss_log_target_predict(df, idata, draw, z, L_test=L)
        return z, log_phi
    if _uses_fss_correction_gp_plot(idata):
        log_phi = _fss_correction_log_phi_predict(df, idata, draw, z, L)
    else:
        target = _scaling_log_target(idata)
        if target not in draw:
            raise ValueError(
                "Posterior has no scaling trace (z, log_phi); "
                "use a correction-GP FSS fit or PyMC sampling."
            )
        log_phi = np.asarray(draw[target], dtype=np.float64)
    return z, log_phi


def _z_grid_for_posterior(
    idata: az.InferenceData,
    df: pd.DataFrame,
    *,
    n_points: int = 200,
) -> np.ndarray:
    max_abs_z = _max_abs_z_from_attrs(idata)
    if max_abs_z is not None:
        return _z_grid_for_fit_window(max_abs_z, n_points=n_points)
    if _posterior_has_scaling_trace(idata):
        z_obs = idata.posterior["z"].values.mean(axis=(0, 1))
    else:
        means = _exponent_values(idata)
        L = df["L"].to_numpy(dtype=np.float64)
        T = df["T"].to_numpy(dtype=np.float64)
        z_obs = z_from_LT(L, T, T_c=means["T_c"], nu=means["nu"])
    return _z_grid_from_obs(z_obs, n_points=n_points)


def _obs_posterior_summary(
    idata: az.InferenceData,
    *,
    df: pd.DataFrame | None = None,
    hdi_prob: float = 0.94,
) -> dict[str, np.ndarray]:
    if _posterior_has_scaling_trace(idata):
        target = _scaling_log_target(idata)
        z = idata.posterior["z"].values
        log_phi = idata.posterior[target].values
    else:
        if df is None:
            raise ValueError(
                "df is required when the posterior omits z / scaling-function traces "
                "(e.g. JAX exponent-only sampling)"
            )
        n_chain = idata.posterior.sizes["chain"]
        n_draw = idata.posterior.sizes["draw"]
        n_obs = len(df)
        z = np.empty((n_chain, n_draw, n_obs), dtype=np.float64)
        log_phi = np.empty((n_chain, n_draw, n_obs), dtype=np.float64)
        for flat_idx in range(n_chain * n_draw):
            chain = flat_idx // n_draw
            draw_idx = flat_idx % n_draw
            z_draw, log_phi_draw = _obs_scaling_at_points(
                df, idata, _select_draw(idata, flat_idx)
            )
            z[chain, draw_idx] = z_draw
            log_phi[chain, draw_idx] = log_phi_draw
    z_summary = _hdi_summary(z, hdi_prob, axis=(0, 1))
    log_phi_summary = _hdi_summary(log_phi, hdi_prob, axis=(0, 1))
    return {
        "z_mean": z_summary["center"],
        "log_phi_mean": log_phi_summary["center"],
        "z_lo": z_summary["lo"],
        "z_hi": z_summary["hi"],
        "log_phi_lo": log_phi_summary["lo"],
        "log_phi_hi": log_phi_summary["hi"],
    }


def _z_grid_from_obs(z_obs: np.ndarray, n_points: int = 200) -> np.ndarray:
    pad = 0.05 * (float(z_obs.max()) - float(z_obs.min()))
    return np.linspace(float(z_obs.min()) - pad, float(z_obs.max()) + pad, n_points)


def _reference_L(df: pd.DataFrame, L: int = 32) -> float:
    L_vals = np.unique(df["L"].to_numpy(dtype=int))
    if L in L_vals:
        return float(L)
    return float(L_vals[np.argmin(np.abs(L_vals - L))])


def _log_phi_curve_at_fixed_L(
    df: pd.DataFrame,
    draw: dict[str, float | np.ndarray],
    z_grid: np.ndarray,
    *,
    L_ref: float,
    idata: az.InferenceData | None = None,
) -> np.ndarray:
    T_span = df["T"].to_numpy(dtype=np.float64)
    T_fine = np.linspace(float(T_span.min()), float(T_span.max()), z_grid.size)
    if idata is not None:
        log_m = _predict_log_m_tl_family(
            idata, df, draw, np.full_like(T_fine, L_ref), T_fine
        )
    else:
        log_m = predict_log_m_tl_gp(
            df, draw, np.full_like(T_fine, L_ref), T_fine
        )
    log_phi = log_phi_from_log_m(log_m, L_ref)
    T_c = float(draw.get("T_c", TC_EXACT))
    z_vals = z_from_LT(L_ref, T_fine, T_c=T_c)
    order = np.argsort(z_vals)
    return np.interp(
        z_grid,
        z_vals[order],
        log_phi[order],
        left=np.nan,
        right=np.nan,
    )


def _eval_log_target_on_grid(
    idata: az.InferenceData,
    draw: dict[str, float | np.ndarray],
    z_grid: np.ndarray,
    *,
    df: pd.DataFrame | None = None,
    L_ref: float | None = None,
) -> np.ndarray:
    if _is_tl_gp_model(idata):
        if df is None:
            raise ValueError("df is required for separable (L,T) GP curve evaluation")
        return _log_phi_curve_at_fixed_L(
            df, draw, z_grid, L_ref=L_ref or _reference_L(df), idata=idata
        )
    if _uses_fss_correction_gp_plot(idata):
        if df is None:
            raise ValueError("df is required for FSS correction GP curve evaluation")
        L_test = np.full_like(z_grid, L_ref or _reference_L(df))
        return _fss_correction_log_phi_predict(df, idata, draw, z_grid, L_test)
    if _uses_marginal_gp_fss_plot(idata):
        if df is None:
            raise ValueError("df is required for marginal GP curve evaluation")
        return _marginal_gp_fss_log_target_predict(
            df, idata, draw, z_grid, L_ref=L_ref
        )
    if not _posterior_has_scaling_trace(idata):
        raise ValueError(
            "Posterior has no scaling trace (z, log_phi); "
            "provide df for marginal GP evaluation."
        )
    target = _scaling_log_target(idata)
    log_target_grid = eval_scaling_curve_np(
        draw["z"],
        draw[target],
        z_grid,
        backend=_posterior_scaling(idata),
        draw=draw,
        knots=_z_knots_from_attrs(idata),
        spline_degree=_spline_degree_from_attrs(idata),
    )
    if target == "log_psi":
        return _log_phi_on_grid(z_grid, log_target_grid, float(draw["beta"]))
    return log_target_grid


def _scaling_curves_from_posterior(
    idata: az.InferenceData,
    z_grid: np.ndarray,
    sample_idx: np.ndarray,
    *,
    df: pd.DataFrame | None = None,
    L_ref: float | None = None,
) -> np.ndarray:
    target = _scaling_log_target(idata)
    backend = _posterior_scaling(idata)
    knots = _z_knots_from_attrs(idata)
    degree = _spline_degree_from_attrs(idata)
    curves = np.empty((sample_idx.size, z_grid.size), dtype=np.float64)
    for row, flat_idx in enumerate(sample_idx):
        draw = _select_draw(idata, int(flat_idx))
        if _is_tl_gp_model(idata):
            curves[row] = _eval_log_target_on_grid(
                idata, draw, z_grid, df=df, L_ref=L_ref
            )
            continue
        if _uses_fss_correction_gp_plot(idata):
            if df is None:
                raise ValueError("df is required for FSS correction scaling curves")
            L_test = np.full_like(z_grid, L_ref or _reference_L(df))
            curves[row] = _fss_correction_log_phi_predict(
                df, idata, draw, z_grid, L_test
            )
            continue
        if _uses_marginal_gp_fss_plot(idata):
            if df is None:
                raise ValueError("df is required for marginal GP scaling curves")
            curves[row] = _marginal_gp_fss_log_target_predict(
                df, idata, draw, z_grid, L_ref=L_ref
            )
            continue
        if not _posterior_has_scaling_trace(idata):
            raise ValueError(
                "Posterior has no scaling trace (z, log_phi); "
                "provide df for marginal GP evaluation."
            )
        log_target_grid = eval_scaling_curve_np(
            draw["z"],
            draw[target],
            z_grid,
            backend=backend,
            draw=draw,
            knots=knots,
            spline_degree=degree,
        )
        if target == "log_psi":
            curves[row] = _log_phi_on_grid(
                z_grid, log_target_grid, float(draw["beta"])
            )
        else:
            curves[row] = log_target_grid
    return curves


def plot_scaling_function_posterior(
    df: pd.DataFrame,
    idata: az.InferenceData,
    out_dir: Path,
    *,
    n_samples: int = 300,
    hdi_prob: float = 0.94,
    seed: int = 0,
    z_grid_points: int = 200,
) -> Path:
    """Plot log Phi_m(z) from the joint PyMC posterior (no post-hoc GP refit)."""
    obs = _obs_posterior_summary(idata, df=df, hdi_prob=hdi_prob)
    sample_idx = _draw_indices(idata, n_samples, seed)
    max_abs_z = _max_abs_z_from_attrs(idata)
    fit_mask = _fit_window_mask(df, max_abs_z)
    z_grid = _z_grid_for_posterior(idata, df, n_points=z_grid_points)
    L_ref = _reference_L(df)
    curve_samples = _scaling_curves_from_posterior(
        idata, z_grid, sample_idx, df=df, L_ref=L_ref
    )

    alpha = (1.0 - hdi_prob) / 2.0
    lower_pct, upper_pct = 100.0 * alpha, 100.0 * (1.0 - alpha)
    hdi_lower = np.percentile(curve_samples, lower_pct, axis=0)
    hdi_upper = np.percentile(curve_samples, upper_pct, axis=0)
    median_curve = np.percentile(curve_samples, 50.0, axis=0)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.fill_between(
        z_grid,
        hdi_lower,
        hdi_upper,
        color="#c44e52",
        alpha=0.25,
        label=f"{100 * hdi_prob:.0f}% posterior band",
        zorder=1,
    )
    ax.plot(
        z_grid,
        median_curve,
        color="#c44e52",
        linewidth=2.0,
        label="posterior median",
        zorder=3,
    )

    palette = {"16": "#1f77b4", "32": "#ff7f0e", "48": "#2ca02c"}
    for L_value in sorted(df["L"].unique()):
        mask = (df["L"] == L_value).to_numpy()
        if fit_mask is not None:
            mask = mask & fit_mask
        if not np.any(mask):
            continue
        color = palette.get(str(int(L_value)), "#4c72b0")
        if _log_phi_fixed_at_obs(idata):
            xerr = None
            if _posterior_model(idata) in ("scaling_tc", "scaling_tl_tc"):
                xerr = _asymmetric_error_stack(
                    obs["z_mean"][mask],
                    obs["z_lo"][mask],
                    obs["z_hi"][mask],
                )
            ax.errorbar(
                obs["z_mean"][mask],
                obs["log_phi_mean"][mask],
                xerr=xerr,
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=1.2,
                capsize=3,
                markersize=7,
                label=f"MC grid, $L={int(L_value)}$",
                zorder=4,
            )
        else:
            xerr = _asymmetric_error_stack(
                obs["z_mean"][mask],
                obs["z_lo"][mask],
                obs["z_hi"][mask],
            )
            yerr = _asymmetric_error_stack(
                obs["log_phi_mean"][mask],
                obs["log_phi_lo"][mask],
                obs["log_phi_hi"][mask],
            )
            ax.errorbar(
                obs["z_mean"][mask],
                obs["log_phi_mean"][mask],
                xerr=xerr,
                yerr=yerr,
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=1.2,
                capsize=3,
                markersize=7,
                label=f"MC grid, $L={int(L_value)}$",
                zorder=4,
            )

    ax.set_xlabel(r"$z = t L^{1/\nu}$")
    ax.set_ylabel(_scaling_function_log_label(idata))
    model = _posterior_model(idata)
    if model == "scaling_only":
        title = r"Scaling function with fixed exact exponents"
    elif model == "scaling_tl":
        title = rf"Scaling function recovered at $L={int(L_ref)}$ (separable GP on $(L,T)$)"
    elif model == "scaling_tl_tc":
        title = rf"Scaling function recovered at $L={int(L_ref)}$ (separable GP on $(L,t)$, $T_c \sim \mathrm{{Unif}}(1,3)$)"
    elif model == "scaling_tc":
        title = r"Scaling function with Gaussian prior on $T_c$ only"
    elif model == "scaling_nu":
        title = r"Stiff GP on $\log\Phi_m(z)$ with $\nu \sim \mathrm{Unif}(0.5, 1.5)$"
    elif model == "scaling_nu_corr":
        title = (
            r"Stiff GP on $\log\Phi_m(z)$ with $L^{-\omega}$ correction GP"
        )
    else:
        title = r"Scaling function from joint posterior on $\log\Phi_m$, $T_c$, $\nu$, $\beta$"
    title += _fit_window_title_suffix(max_abs_z)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    path = out_dir / "scaling_function_posterior.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _obs_indices_for_L(df: pd.DataFrame, L: int) -> np.ndarray:
    return np.flatnonzero(df["L"].to_numpy(dtype=int) == L)


def _obs_indices_for_T(df: pd.DataFrame, T: float, *, rtol: float = 1e-6) -> np.ndarray:
    return np.flatnonzero(np.isclose(df["T"].to_numpy(dtype=np.float64), T, rtol=rtol))


def _nearest_T_in_data(df: pd.DataFrame, T_target: float) -> float:
    T_vals = np.unique(df["T"].to_numpy(dtype=np.float64))
    return float(T_vals[np.argmin(np.abs(T_vals - T_target))])


def _L_grid_from_df(df: pd.DataFrame, *, n_points: int = 200) -> np.ndarray:
    L_vals = np.unique(df["L"].to_numpy(dtype=np.float64))
    pad = 0.05 * (float(L_vals.max()) - float(L_vals.min()))
    return np.linspace(float(L_vals.min()) - pad, float(L_vals.max()) + pad, n_points)


def _percentile_band(
    samples: np.ndarray,
    hdi_prob: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha = (1.0 - hdi_prob) / 2.0
    lower_pct, upper_pct = 100.0 * alpha, 100.0 * (1.0 - alpha)
    return (
        np.percentile(samples, lower_pct, axis=0),
        np.percentile(samples, upper_pct, axis=0),
        np.percentile(samples, 50.0, axis=0),
    )


def _hdi_summary(
    samples: np.ndarray,
    hdi_prob: float,
    axis: int | tuple[int, ...],
) -> dict[str, np.ndarray]:
    """Posterior median with asymmetric HDI bounds."""
    alpha = (1.0 - hdi_prob) / 2.0
    lo_pct, hi_pct = 100.0 * alpha, 100.0 * (1.0 - alpha)
    return {
        "center": np.median(samples, axis=axis),
        "lo": np.percentile(samples, lo_pct, axis=axis),
        "hi": np.percentile(samples, hi_pct, axis=axis),
    }


def _asymmetric_error_stack(
    center: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
) -> np.ndarray:
    return np.vstack(
        (
            np.maximum(center - lo, 0.0),
            np.maximum(hi - center, 0.0),
        )
    )


def _magnetization_log_sigma(df: pd.DataFrame, obs_idx: np.ndarray) -> np.ndarray:
    """Standard error of log |m| at obs_idx (matching the fit likelihood)."""
    _, sigma_log = _log_m_obs_at(df, obs_idx)
    return sigma_log


def _lognormal_error_stack(
    center: np.ndarray,
    sigma_log: np.ndarray,
    *,
    interval_prob: float = 0.682689492137086,
) -> np.ndarray:
    """Asymmetric |m| error bars from log-normal MC uncertainty.

    ``interval_prob`` sets the total probability mass in the interval, e.g.
    0.94 for bars comparable to the posterior HDI shading.
    """
    center = np.maximum(np.asarray(center, dtype=np.float64), 1e-12)
    sigma_log = np.asarray(sigma_log, dtype=np.float64)
    z = math.sqrt(2.0) * float(erfinv(interval_prob))
    return np.vstack(
        (
            center * np.maximum(1.0 - np.exp(-z * sigma_log), 0.0),
            center * np.maximum(np.exp(z * sigma_log) - 1.0, 0.0),
        )
    )


def _m_obs_at(
    df: pd.DataFrame,
    obs_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """MC mean |m| and its standard error at obs_idx."""
    m = df["magnetization"].to_numpy(dtype=np.float64)[obs_idx]
    sigma_m = _mc_sigma(df)[obs_idx]
    return m, sigma_m


def _gaussian_error_stack(
    sigma: np.ndarray,
    *,
    interval_prob: float = 0.682689492137086,
) -> np.ndarray:
    """Symmetric error bars from Gaussian MC uncertainty."""
    z = math.sqrt(2.0) * float(erfinv(interval_prob))
    half = z * np.asarray(sigma, dtype=np.float64)
    return np.vstack((half, half))


def _log_m_obs_at(
    df: pd.DataFrame,
    obs_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """MC log |m| and its standard error at obs_idx."""
    return resolve_log_m_observables(
        df["log_magnetization"].to_numpy(dtype=np.float64)[obs_idx],
        df["log_magnetization_std"].to_numpy(dtype=np.float64)[obs_idx],
        df["n_eff_log"].to_numpy(dtype=np.float64)[obs_idx],
    )


def _log_training_from_df(
    df: pd.DataFrame,
    nu: float,
    *,
    beta: float = BETA_EXACT,
    T_c: float = TC_EXACT,
    obs_idx: np.ndarray | None = None,
    obs_sigma_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (z, L, log_m, log_phi, sigma_log) for all rows or obs_idx."""
    if obs_idx is None:
        obs_idx = np.arange(len(df), dtype=int)
    L = df["L"].to_numpy(dtype=np.float64)[obs_idx]
    T = df["T"].to_numpy(dtype=np.float64)[obs_idx]
    log_m, sigma_log = _log_m_obs_at(df, obs_idx)
    sigma_log = sigma_log * obs_sigma_scale
    log_phi = log_phi_from_log_m(log_m, L, beta=beta, nu=nu)
    z = z_from_LT(L, T, T_c=T_c, nu=nu)
    return z, L, log_m, log_phi, sigma_log


def _exp_percentile_band(
    log_samples: np.ndarray,
    hdi_prob: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Percentile band on a positive quantity stored in log space."""
    lo_log, hi_log, med_log = _percentile_band(log_samples, hdi_prob)
    return np.exp(lo_log), np.exp(hi_log), np.exp(med_log)


def _log_m_log_phi_gp_conditional(
    df: pd.DataFrame,
    idata: az.InferenceData,
    draw: dict[str, float | np.ndarray],
    z_test: np.ndarray,
    L_test: np.ndarray,
    *,
    rng: np.random.Generator | None = None,
    include_gp_epistemic_std: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """GP conditional on MC log targets; returns (log_phi, log_m) at test points."""
    df = _df_for_fss_plot(df, idata)
    nu = float(draw["nu"])
    beta = float(draw["beta"])
    T_c = float(draw.get("T_c", TC_EXACT))
    L_test = np.asarray(L_test, dtype=np.float64).ravel()
    z_test = np.asarray(z_test, dtype=np.float64).ravel()

    if _is_tl_gp_model(idata):
        raise ValueError("tl_gp models should call _predict_log_m_tl_family directly")

    if _fss_m_uses_phi_gp(idata) or _is_correction_nu_model(idata):
        use_log_m = _fss_use_log_m(idata)
        z_train, L_train, phi_train, sigma_phi_train = _m_phi_training_from_df(
            df,
            nu,
            beta=beta,
            T_c=T_c,
            obs_sigma_scale=_obs_sigma_scale(idata),
            use_log_m=use_log_m,
        )
        if _fss_channel_correction(idata, "m") or _is_correction_nu_model(idata):
            if _is_fss_binder_correction(idata):
                hp = _fss_binder_channel_hyperparams(idata, draw, "m")
            else:
                hp = _correction_gp_hyperparams(idata, draw)
            omega = _omega_from_draw(idata, draw)
            phi_mean, phi_std = correction_gp_latent_conditional(
                z_train,
                L_train,
                phi_train,
                sigma_phi_train,
                z_test,
                L_test,
                omega=omega,
                **hp,
            )
        else:
            gp_hp = _gp_params_from_attrs(idata)
            if not gp_hp and "gp_ell" in draw:
                gp_hp = {
                    "gp_ell": float(draw["gp_ell"]),
                    "gp_eta": float(draw["gp_eta"]),
                }
            if not gp_hp:
                raise ValueError("GP hyperparameters required for linear Phi_m model")
            phi_mean, phi_std = gp_posterior_predictive(
                z_train,
                phi_train,
                sigma_phi_train,
                z_test,
                length_scale=gp_hp["gp_ell"],
                amplitude=gp_hp["gp_eta"],
                kernel=_universal_kernel_from_attrs(idata),
            )
        if include_gp_epistemic_std and rng is not None:
            phi = np.maximum(phi_mean + rng.normal(0.0, phi_std), 1e-12)
        else:
            phi = np.maximum(phi_mean, 1e-12)
        log_phi = np.log(phi)
        log_m = log_phi - (beta / nu) * np.log(L_test)
        return log_phi, log_m

    z_train, L_train, log_m_train, log_phi_train, sigma_log_train = _log_training_from_df(
        df, nu, beta=beta, T_c=T_c, obs_sigma_scale=_obs_sigma_scale(idata)
    )

    gp_hp = _gp_params_from_attrs(idata)
    if not gp_hp and "gp_ell" in draw:
        gp_hp = {"gp_ell": float(draw["gp_ell"]), "gp_eta": float(draw["gp_eta"])}

    if gp_hp:
        log_phi_mean, log_phi_std = gp_latent_conditional(
            z_train,
            log_phi_train,
            z_test,
            sigma_log_train=sigma_log_train,
            length_scale=gp_hp["gp_ell"],
            amplitude=gp_hp["gp_eta"],
            kernel=_universal_kernel_from_attrs(idata),
        )
        if include_gp_epistemic_std and rng is not None:
            log_phi = log_phi_mean + rng.normal(0.0, log_phi_std)
        else:
            log_phi = log_phi_mean
        log_m = log_phi - (beta / nu) * np.log(L_test)
        return log_phi, log_m

    log_phi = _eval_log_target_on_grid(idata, draw, z_test, df=df, L_ref=float(L_test[0]))
    log_m = log_phi - (beta / nu) * np.log(L_test)
    return log_phi, log_m


def _marginal_gp_fss_log_target_predict(
    df: pd.DataFrame,
    idata: az.InferenceData,
    draw: dict[str, float | np.ndarray],
    z_test: np.ndarray,
    *,
    L_test: np.ndarray | None = None,
    L_ref: float | None = None,
) -> np.ndarray:
    """GP posterior mean of the primary scaling target for exponent-only FSS fits."""
    if _fss_binder_only(idata):
        return _fss_binder_f0_predict(df, idata, draw, z_test)
    if L_test is None:
        L_ref = L_ref or _reference_L(df)
        L_test = np.full_like(z_test, L_ref)
    log_phi, _ = _log_m_log_phi_gp_conditional(df, idata, draw, z_test, L_test)
    return log_phi


def _posterior_mean_band(
    samples: np.ndarray,
    hdi_prob: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HDI bounds and posterior mean (for plotting m(T) from posterior function samples)."""
    lo, hi, _ = _percentile_band(samples, hdi_prob)
    return lo, hi, samples.mean(axis=0)


def _plot_posterior_band(
    ax: plt.Axes,
    x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    central: np.ndarray,
    *,
    hdi_prob: float,
    label_band: bool = True,
    central_label: str = "posterior median",
    color: str = "#c44e52",
    band_alpha: float = 0.25,
) -> None:
    band_label = f"{100 * hdi_prob:.0f}% posterior band" if label_band else None
    ax.fill_between(x, lower, upper, color=color, alpha=band_alpha, label=band_label, zorder=1)
    ax.plot(x, central, color=color, linewidth=2.0, label=central_label, zorder=3)


def _color_for_L(L: int) -> str:
    palette = {16: "#1f77b4", 32: "#ff7f0e", 48: "#2ca02c", 64: "#d62728"}
    return palette.get(L, "#4c72b0")


def _T_grid_for_L(
    df: pd.DataFrame,
    L: int,
    *,
    n_points: int = 200,
) -> np.ndarray:
    T_vals = df.loc[df["L"] == L, "T"].to_numpy(dtype=np.float64)
    pad = 0.05 * (float(T_vals.max()) - float(T_vals.min()))
    return np.linspace(
        float(T_vals.min()) - pad,
        float(T_vals.max()) + pad,
        n_points,
    )


def _Tc_posterior_summary(
    idata: az.InferenceData,
    *,
    hdi_prob: float = 0.94,
) -> dict[str, float]:
    if _posterior_model(idata) in (
        "scaling_only",
        "scaling_tl",
        "scaling_nu",
        "scaling_nu_corr",
        "scaling_nu_corr_hyper",
        "scaling_nu_corr_hyper_omega2",
        "scaling_crossover",
        "scaling_crossover_tc",
        "scaling_crossover_fss",
    ) or _fss_fixed_Tc(idata):
        return {"median": TC_EXACT, "mean": TC_EXACT, "lo": TC_EXACT, "hi": TC_EXACT}
    T_c = idata.posterior["T_c"].values.reshape(-1)
    alpha = (1.0 - hdi_prob) / 2.0
    lo_pct, hi_pct = 100.0 * alpha, 100.0 * (1.0 - alpha)
    return {
        "median": float(np.median(T_c)),
        "mean": float(T_c.mean()),
        "lo": float(np.percentile(T_c, lo_pct)),
        "hi": float(np.percentile(T_c, hi_pct)),
    }


def _add_Tc_markers(
    ax: plt.Axes,
    T_c_summary: dict[str, float],
    *,
    hdi_prob: float,
    label_band: bool,
) -> None:
    band_label = f"{100 * hdi_prob:.0f}% $T_c$ HDI" if label_band else None
    ax.axvspan(
        T_c_summary["lo"],
        T_c_summary["hi"],
        color="#888888",
        alpha=0.2,
        label=band_label,
        zorder=0,
    )
    ax.axvline(
        T_c_summary["median"],
        color="#333333",
        linestyle="--",
        linewidth=1.5,
        label=r"median $T_c$",
        zorder=1,
    )


def _correction_phi_summary_at_obs(
    idata: az.InferenceData,
    df: pd.DataFrame,
    obs_idx: np.ndarray,
    sample_idx: np.ndarray,
    *,
    hdi_prob: float = 0.94,
) -> dict[str, np.ndarray]:
    phi = np.empty((sample_idx.size, obs_idx.size), dtype=np.float64)
    for row, flat_idx in enumerate(sample_idx):
        draw = _select_draw(idata, int(flat_idx))
        nu = float(draw["nu"])
        T_c = float(draw["T_c"])
        beta = float(draw["beta"])
        L = df["L"].to_numpy(dtype=np.float64)[obs_idx]
        T = df["T"].to_numpy(dtype=np.float64)[obs_idx]
        t = (T - T_c) / T_c
        z = t * L ** (1.0 / nu)
        phi[row] = _correction_gp_predict(df, idata, draw, z, L)
    summary = _hdi_summary(phi, hdi_prob, axis=0)
    return {
        "phi_mean": summary["center"],
        "phi_lo": summary["lo"],
        "phi_hi": summary["hi"],
    }


def _fss_binder_m_correction_summary_at_obs(
    idata: az.InferenceData,
    df: pd.DataFrame,
    obs_idx: np.ndarray,
    sample_idx: np.ndarray,
    *,
    hdi_prob: float = 0.94,
) -> dict[str, np.ndarray]:
    phi = np.empty((sample_idx.size, obs_idx.size), dtype=np.float64)
    for row, flat_idx in enumerate(sample_idx):
        draw = _select_draw(idata, int(flat_idx))
        nu = float(draw["nu"])
        T_c = float(draw["T_c"])
        L = df["L"].to_numpy(dtype=np.float64)[obs_idx]
        T = df["T"].to_numpy(dtype=np.float64)[obs_idx]
        t = (T - T_c) / T_c
        z = t * L ** (1.0 / nu)
        phi[row] = _fss_binder_m_correction_predict(df, idata, draw, z, L)
    summary = _hdi_summary(phi, hdi_prob, axis=0)
    return {
        "phi_mean": summary["center"],
        "phi_lo": summary["lo"],
        "phi_hi": summary["hi"],
    }


def _fss_correction_summary_at_obs(
    idata: az.InferenceData,
    df: pd.DataFrame,
    obs_idx: np.ndarray,
    sample_idx: np.ndarray,
    *,
    hdi_prob: float = 0.94,
) -> dict[str, np.ndarray]:
    channel = _primary_fss_correction_channel(idata)
    phi = np.empty((sample_idx.size, obs_idx.size), dtype=np.float64)
    for row, flat_idx in enumerate(sample_idx):
        draw = _select_draw(idata, int(flat_idx))
        nu = float(draw["nu"])
        T_c = float(draw["T_c"])
        L = df["L"].to_numpy(dtype=np.float64)[obs_idx]
        T = df["T"].to_numpy(dtype=np.float64)[obs_idx]
        t = (T - T_c) / T_c
        z = t * L ** (1.0 / nu)
        if channel == "m":
            values = _fss_binder_m_correction_predict(df, idata, draw, z, L)
        elif channel == "m2":
            values = _fss_moment_correction_predict(
                df, idata, draw, z, L, moment=2
            )
        elif channel == "m4":
            values = _fss_moment_correction_predict(
                df, idata, draw, z, L, moment=4
            )
        elif channel == "chi":
            values = _fss_chi_correction_predict(df, idata, draw, z, L)
        elif channel == "binder":
            values = _fss_binder_u_correction_predict(df, idata, draw, z, L)
        else:
            raise ValueError(f"Unsupported FSS correction GP channel {channel!r}")
        phi[row] = values
    summary = _hdi_summary(phi, hdi_prob, axis=0)
    return {
        "phi_mean": summary["center"],
        "phi_lo": summary["lo"],
        "phi_hi": summary["hi"],
    }


def _latent_phi_summary_at_obs(
    idata: az.InferenceData,
    obs_idx: np.ndarray,
    *,
    hdi_prob: float = 0.94,
) -> dict[str, np.ndarray]:
    target = _scaling_log_target(idata)
    log_phi = idata.posterior[target].values[:, :, obs_idx]
    phi = np.exp(log_phi)
    summary = _hdi_summary(phi, hdi_prob, axis=(0, 1))
    return {
        "phi_mean": summary["center"],
        "phi_lo": summary["lo"],
        "phi_hi": summary["hi"],
    }


def _L_slice_curves_from_posterior(
    idata: az.InferenceData,
    L_grid: np.ndarray,
    T_fixed: float,
    sample_idx: np.ndarray,
    *,
    df: pd.DataFrame | None = None,
    rng: np.random.Generator | None = None,
    include_gp_epistemic_std: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior samples of log Phi_m(L) and log |m|(L) at fixed T."""
    log_phi_samples = np.empty((sample_idx.size, L_grid.size), dtype=np.float64)
    log_m_samples = np.empty((sample_idx.size, L_grid.size), dtype=np.float64)

    for row, flat_idx in enumerate(sample_idx):
        draw = _select_draw(idata, int(flat_idx))
        T_c = float(draw["T_c"])
        nu = float(draw["nu"])
        beta = float(draw["beta"])
        if _is_tl_gp_model(idata):
            if df is None:
                raise ValueError("df is required for separable (L,T) GP slice curves")
            log_m = _predict_log_m_tl_family(
                idata, df, draw, L_grid, np.full_like(L_grid, T_fixed)
            )
            log_m_samples[row] = log_m
            log_phi_samples[row] = log_phi_from_log_m(log_m, L_grid, beta=beta, nu=nu)
            continue
        if _is_crossover_model(idata):
            if df is None:
                raise ValueError("df is required for crossover slice curves")
            chain = flat_idx // idata.posterior.sizes["draw"]
            d = flat_idx % idata.posterior.sizes["draw"]
            T_test = np.full_like(L_grid, T_fixed)
            phi, m = _crossover_predict(idata, chain, d, df, T_test, L_grid)
            log_phi_samples[row] = np.log(np.maximum(phi, 1e-12))
            log_m_samples[row] = np.log(np.maximum(m, 1e-12))
            continue
        if df is None:
            raise ValueError("df is required for GP slice curves")
        t = (T_fixed - T_c) / T_c
        z_grid = t * L_grid ** (1.0 / nu)
        log_phi, log_m = _log_m_log_phi_gp_conditional(
            df,
            idata,
            draw,
            z_grid,
            L_grid,
            rng=rng,
            include_gp_epistemic_std=include_gp_epistemic_std,
        )
        log_phi_samples[row] = log_phi
        log_m_samples[row] = log_m
    return log_phi_samples, log_m_samples


def _T_slice_curves_from_posterior(
    idata: az.InferenceData,
    T_grid: np.ndarray,
    L: int,
    sample_idx: np.ndarray,
    *,
    df: pd.DataFrame | None = None,
    rng: np.random.Generator | None = None,
    include_gp_epistemic_std: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior samples of log Phi_m(T) and log |m|(T) at fixed L."""
    log_phi_samples = np.empty((sample_idx.size, T_grid.size), dtype=np.float64)
    log_m_samples = np.empty((sample_idx.size, T_grid.size), dtype=np.float64)
    L_f = float(L)

    for row, flat_idx in enumerate(sample_idx):
        draw = _select_draw(idata, int(flat_idx))
        nu = float(draw["nu"])
        beta = float(draw["beta"])
        if _is_tl_gp_model(idata):
            if df is None:
                raise ValueError("df is required for separable (L,T) GP slice curves")
            log_m = _predict_log_m_tl_family(
                idata, df, draw, np.full_like(T_grid, L_f), T_grid
            )
            log_m_samples[row] = log_m
            log_phi_samples[row] = log_phi_from_log_m(log_m, L_f, beta=beta, nu=nu)
            continue
        if _is_crossover_model(idata):
            if df is None:
                raise ValueError("df is required for crossover slice curves")
            chain = flat_idx // idata.posterior.sizes["draw"]
            d = flat_idx % idata.posterior.sizes["draw"]
            L_test = np.full_like(T_grid, L_f)
            phi, m = _crossover_predict(idata, chain, d, df, T_grid, L_test)
            log_phi_samples[row] = np.log(np.maximum(phi, 1e-12))
            log_m_samples[row] = np.log(np.maximum(m, 1e-12))
            continue
        if df is None:
            raise ValueError("df is required for GP slice curves")
        T_c = float(draw["T_c"])
        t_grid = (T_grid - T_c) / T_c
        z_grid = t_grid * L_f ** (1.0 / nu)
        L_test = np.full_like(T_grid, L_f)
        log_phi, log_m = _log_m_log_phi_gp_conditional(
            df,
            idata,
            draw,
            z_grid,
            L_test,
            rng=rng,
            include_gp_epistemic_std=include_gp_epistemic_std,
        )
        log_phi_samples[row] = log_phi
        log_m_samples[row] = log_m
    return log_phi_samples, log_m_samples


def _phi_summary_at_obs(
    idata: az.InferenceData,
    df: pd.DataFrame,
    obs_idx: np.ndarray,
    sample_idx: np.ndarray,
    *,
    hdi_prob: float = 0.94,
) -> dict[str, np.ndarray]:
    if _is_correction_nu_model(idata):
        return _correction_phi_summary_at_obs(
            idata, df, obs_idx, sample_idx, hdi_prob=hdi_prob
        )
    if _uses_fss_correction_gp_plot(idata):
        return _fss_correction_summary_at_obs(
            idata, df, obs_idx, sample_idx, hdi_prob=hdi_prob
        )
    if _is_crossover_model(idata):
        return _crossover_phi_summary_at_obs(idata, df, obs_idx, hdi_prob=hdi_prob)
    return _latent_phi_summary_at_obs(idata, obs_idx, hdi_prob=hdi_prob)


def plot_t_slices(
    df: pd.DataFrame,
    idata: az.InferenceData,
    out_dir: Path,
    *,
    L_values: list[int] | None = None,
    T_fixed: float | None = None,
    n_samples: int = 300,
    hdi_prob: float = 0.94,
    seed: int = 0,
    T_grid_points: int = 200,
    L_grid_points: int = 200,
) -> list[Path]:
    """Phi_m and |m| vs T (one curve per L) and vs L at fixed T on a single figure."""
    if L_values is None:
        L_values = sorted(int(L) for L in df["L"].unique())
    if not L_values:
        raise ValueError("No lattice sizes to plot")

    if T_fixed is None:
        T_fixed = _nearest_T_in_data(df, TC_EXACT)
    obs_idx_T = _obs_indices_for_T(df, T_fixed)
    if obs_idx_T.size == 0:
        raise ValueError(f"No observations at T={T_fixed} in data")

    L_grid = _L_grid_from_df(df, n_points=L_grid_points)
    T_c_summary = _Tc_posterior_summary(idata, hdi_prob=hdi_prob)
    sample_idx = _draw_indices(idata, n_samples, seed)
    curve_rng = np.random.default_rng(seed)

    log_phi_L, log_m_L = _L_slice_curves_from_posterior(
        idata, L_grid, T_fixed, sample_idx, df=df, rng=curve_rng
    )
    phi_L_lo, phi_L_hi, phi_L_med = _exp_percentile_band(log_phi_L, hdi_prob)
    m_L_lo, m_L_hi, m_L_med = _exp_percentile_band(log_m_L, hdi_prob)

    L_at_T = df["L"].to_numpy(dtype=np.float64)[obs_idx_T]
    beta_plot = _posterior_mean_or_exact(idata, "beta")
    nu_plot = _posterior_mean_or_exact(idata, "nu")
    use_log_m_plot = _fss_use_log_m(idata)
    if use_log_m_plot:
        log_m_obs_T, sigma_log_T = _log_m_obs_at(df, obs_idx_T)
        m_obs_T = np.exp(log_m_obs_T)
        phi_obs_T = np.exp(log_phi_from_log_m(log_m_obs_T, L_at_T, beta=beta_plot, nu=nu_plot))
        m_err_T = _lognormal_error_stack(m_obs_T, sigma_log_T, interval_prob=hdi_prob)
        phi_err_T = _lognormal_error_stack(phi_obs_T, sigma_log_T, interval_prob=hdi_prob)
    else:
        m_obs_T, sigma_m_T = _m_obs_at(df, obs_idx_T)
        phi_obs_T = m_obs_T * L_at_T ** (beta_plot / nu_plot)
        sigma_phi_T = sigma_m_T * L_at_T ** (beta_plot / nu_plot)
        m_err_T = _gaussian_error_stack(sigma_m_T, interval_prob=hdi_prob)
        phi_err_T = _gaussian_error_stack(sigma_phi_T, interval_prob=hdi_prob)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    ax_phi_T = axes[0, 0]
    ax_m_T = axes[0, 1]
    _add_Tc_markers(ax_phi_T, T_c_summary, hdi_prob=hdi_prob, label_band=True)
    _add_Tc_markers(ax_m_T, T_c_summary, hdi_prob=hdi_prob, label_band=False)

    m_T_ylo = np.inf
    m_T_yhi = -np.inf

    for idx, L in enumerate(L_values):
        obs_idx_L = _obs_indices_for_L(df, L)
        if obs_idx_L.size == 0:
            raise ValueError(f"No observations for L={L} in data")

        color = _color_for_L(L)
        T_grid = _T_grid_for_L(df, L, n_points=T_grid_points)
        log_phi_T, log_m_T = _T_slice_curves_from_posterior(
            idata, T_grid, L, sample_idx, df=df, rng=curve_rng
        )
        phi_T_lo, phi_T_hi, phi_T_med = _exp_percentile_band(log_phi_T, hdi_prob)
        if _is_crossover_model(idata):
            m_T_lo, m_T_hi, _ = _exp_percentile_band(log_m_T, hdi_prob)
            m_T_central = np.exp(log_m_T.mean(axis=0))
        else:
            m_T_lo, m_T_hi, m_T_central = _exp_percentile_band(log_m_T, hdi_prob)

        T_at_L = df["T"].to_numpy(dtype=np.float64)[obs_idx_L]
        if use_log_m_plot:
            log_m_obs_L, sigma_log_L = _log_m_obs_at(df, obs_idx_L)
            m_obs_L = np.exp(log_m_obs_L)
            phi_obs_L = np.exp(
                log_phi_from_log_m(log_m_obs_L, float(L), beta=beta_plot, nu=nu_plot)
            )
            m_err_L = _lognormal_error_stack(
                m_obs_L, sigma_log_L, interval_prob=hdi_prob
            )
            phi_err_L = _lognormal_error_stack(
                phi_obs_L, sigma_log_L, interval_prob=hdi_prob
            )
        else:
            m_obs_L, sigma_m_L = _m_obs_at(df, obs_idx_L)
            phi_obs_L = m_obs_L * float(L) ** (beta_plot / nu_plot)
            sigma_phi_L = sigma_m_L * float(L) ** (beta_plot / nu_plot)
            m_err_L = _gaussian_error_stack(sigma_m_L, interval_prob=hdi_prob)
            phi_err_L = _gaussian_error_stack(sigma_phi_L, interval_prob=hdi_prob)
        L_label = rf"$L={L}$"

        _plot_posterior_band(
            ax_phi_T,
            T_grid,
            phi_T_lo,
            phi_T_hi,
            phi_T_med,
            hdi_prob=hdi_prob,
            label_band=idx == 0,
            central_label=L_label,
            color=color,
            band_alpha=0.30,
        )
        ax_phi_T.errorbar(
            T_at_L,
            phi_obs_L,
            yerr=phi_err_L,
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=7,
            label=f"MC, {L_label}",
            zorder=4,
        )

        _plot_posterior_band(
            ax_m_T,
            T_grid,
            m_T_lo,
            m_T_hi,
            m_T_central,
            hdi_prob=hdi_prob,
            label_band=idx == 0,
            central_label=L_label,
            color=color,
            band_alpha=0.30,
        )
        m_T_ylo = min(m_T_ylo, float(np.min(m_obs_L - m_err_L[0])))
        m_T_yhi = max(m_T_yhi, float(np.max(m_obs_L + m_err_L[1])))
        ax_m_T.errorbar(
            T_at_L,
            m_obs_L,
            yerr=m_err_L,
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.4,
            capsize=4,
            capthick=1.2,
            markersize=7,
            label=f"MC, {L_label}",
            zorder=5,
        )

    m_inf_T_grid = _T_grid_for_L(df, L_values[0], n_points=T_grid_points)
    ax_m_T.plot(
        m_inf_T_grid,
        magnetization_infinite_L(m_inf_T_grid),
        color="#333333",
        linestyle="--",
        linewidth=2.0,
        label=r"$L=\infty$ exact",
        zorder=2,
    )
    m_T_pad = 0.05 * (m_T_yhi - m_T_ylo)
    ax_m_T.set_ylim(max(0.0, m_T_ylo - m_T_pad), m_T_yhi + m_T_pad)

    ax_phi_T.set_xlabel(r"$T$")
    ax_phi_T.set_ylabel(r"$\Phi_m(T)$")
    ax_phi_T.set_title("Scaling function vs $T$")
    ax_phi_T.legend(loc="best", fontsize=8)

    ax_m_T.set_xlabel(r"$T$")
    ax_m_T.set_ylabel(r"$|m|$")
    ax_m_T.set_title("Magnetization vs $T$")
    ax_m_T.legend(loc="best", fontsize=8)

    ax = axes[1, 0]
    _plot_posterior_band(
        ax, L_grid, phi_L_lo, phi_L_hi, phi_L_med, hdi_prob=hdi_prob, band_alpha=0.30
    )
    ax.errorbar(
        L_at_T,
        phi_obs_T,
        yerr=phi_err_T,
        fmt="o",
        color="#4c72b0",
        ecolor="#4c72b0",
        capsize=3,
        markersize=7,
        label="MC",
        zorder=4,
    )
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"$\Phi_m(L)$")
    ax.set_title(rf"Scaling function at $T={T_fixed:.4f}$")
    ax.legend(loc="best", fontsize=8)

    ax = axes[1, 1]
    _plot_posterior_band(
        ax, L_grid, m_L_lo, m_L_hi, m_L_med, hdi_prob=hdi_prob, label_band=True, band_alpha=0.30
    )
    ax.errorbar(
        L_at_T,
        m_obs_T,
        yerr=m_err_T,
        fmt="o",
        color="#4c72b0",
        ecolor="#4c72b0",
        elinewidth=1.4,
        capsize=4,
        capthick=1.2,
        markersize=7,
        label="Ising MC",
        zorder=4,
    )
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"$|m|$")
    ax.set_title(rf"Magnetization at $T={T_fixed:.4f}$")
    ax.legend(loc="best", fontsize=8)

    path = out_dir / "t_slice.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return [path]


def plot_t_slice(
    df: pd.DataFrame,
    idata: az.InferenceData,
    out_dir: Path,
    *,
    L: int = 32,
    T_fixed: float | None = None,
    **kwargs,
) -> Path:
    """Backward-compatible wrapper around :func:`plot_t_slices` for a single L."""
    return plot_t_slices(
        df, idata, out_dir, L_values=[L], T_fixed=T_fixed, **kwargs
    )[0]


def _trace_var_names(idata: az.InferenceData) -> list[str]:
    model = _posterior_model(idata)
    if model == "scaling_only":
        return ["gp_ell", "gp_eta"] if _posterior_scaling(idata) == "gp" else ["spline_coeff"]
    if model == "scaling_tl":
        return ["gp_ell_L", "gp_ell_T", "gp_eta"]
    if model == "scaling_tl_tc":
        return ["T_c", "gp_ell_L", "gp_ell_t", "gp_eta"]
    if model == "scaling_tc":
        return ["T_c", "gp_ell", "gp_eta"]
    if model == "scaling_nu":
        return ["nu"]
    if model == "scaling_nu_corr_hyper":
        return [
            "nu",
            "omega",
            "gp_ell",
            "gp_eta",
            "correction_gp_ell",
            "correction_gp_eta",
        ]
    if model == "scaling_nu_corr_hyper_omega2":
        return [
            "nu",
            "gp_ell",
            "gp_eta",
            "correction_gp_ell",
            "correction_gp_eta",
        ]
    if model == "scaling_corr_hyper_omega2_tc":
        return [
            "T_c",
            "nu",
            "gp_ell",
            "gp_eta",
            "correction_gp_ell",
            "correction_gp_eta",
        ]
    if model == "scaling_corr_hyper_omega2_fss":
        return [
            "T_c",
            "nu",
            "beta",
            "gp_ell",
            "gp_eta",
            "correction_gp_ell",
            "correction_gp_eta",
        ]
    if model == "scaling_corr_hyper_fss":
        return [
            "T_c",
            "nu",
            "beta",
            "omega",
            "gp_ell",
            "gp_eta",
            "correction_gp_ell",
            "correction_gp_eta",
        ]
    if model == "scaling_crossover":
        return ["crossover_r"]
    if model == "scaling_crossover_tc":
        return ["nu", "crossover_r"]
    if model == "scaling_crossover_fss":
        return ["nu", "beta", "crossover_r"]
    if model in ("fss", "fss_nu", "fss_binder") and _is_fss_gp_model(idata):
        names = [
            name
            for name in ("T_c_delta", "T_c", "nu", "beta", "gamma")
            if _posterior_has(idata, name)
        ]
        names.extend(_inferred_gp_var_names(idata))
        if _posterior_has(idata, "omega"):
            names.append("omega")
        for name in ("disc_q", "disc_t0", "disc_sigma_model"):
            if _posterior_has(idata, name):
                names.append(name)
        return names
    return ["T_c", "nu", "beta", *_inferred_gp_var_names(idata)]


def plot_posterior_traces(idata: az.InferenceData, out_dir: Path) -> Path | None:
    var_names = list(_filter_posterior_vars(idata, _trace_var_names(idata)))
    if not var_names:
        return None
    axes = az.plot_trace(
        idata,
        var_names=var_names,
        figsize=(12, 8),
        combined=False,
    )
    fig = axes.flatten()[0].figure
    path = out_dir / "posterior_traces.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_posterior_sample_collapses(
    df: pd.DataFrame,
    idata: az.InferenceData,
    out_dir: Path,
    *,
    n_samples: int = 6,
    seed: int = 0,
) -> Path:
    """Overlay data-collapse using a handful of posterior draws."""
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

    L = df["L"].to_numpy(dtype=np.float64)
    T = df["T"].to_numpy(dtype=np.float64)
    m = df["magnetization"].to_numpy(dtype=np.float64)

    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, n_samples))
    for i, (color, flat_idx) in enumerate(
        zip(cmap, _draw_indices(idata, n_samples, seed))
    ):
        params = _select_draw(idata, int(flat_idx))
        t = (T - params["T_c"]) / params["T_c"]
        z = t * L ** (1.0 / params["nu"])
        scaled_m = m * L ** (params["beta"] / params["nu"])
        label = "posterior draws" if i == 0 else None
        ax.scatter(
            z,
            scaled_m,
            s=35,
            alpha=0.45,
            color=color,
            edgecolors="none",
            label=label,
        )

    for L_value, color in zip(
        sorted(df["L"].unique()), ("#1f77b4", "#ff7f0e", "#2ca02c")
    ):
        mask = df["L"] == L_value
        t_ref = (T[mask] - TC_EXACT) / TC_EXACT
        z_ref = t_ref * L[mask] ** (1.0 / NU_EXACT)
        scaled_ref = m[mask] * L[mask] ** (BETA_EXACT / NU_EXACT)
        ax.scatter(
            z_ref,
            scaled_ref,
            s=80,
            marker="x",
            linewidths=1.5,
            color=color,
            label=f"exact exponents, L={int(L_value)}",
        )

    ax.set_xlabel(r"$t L^{1/\nu}$")
    ax.set_ylabel(r"$L^{\beta/\nu}\,|m|$")
    ax.set_title(f"Data collapse: {n_samples} posterior draws vs exact exponents")
    ax.legend(loc="best", fontsize=8)
    path = out_dir / "posterior_sample_collapses.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_posterior_sample_scaling_curves(
    idata: az.InferenceData,
    out_dir: Path,
    *,
    df: pd.DataFrame | None = None,
    n_samples: int = 6,
    seed: int = 0,
) -> Path:
    """Plot a few joint-posterior scaling curves log Phi_m(z)."""
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    cmap = plt.cm.plasma(np.linspace(0.15, 0.85, n_samples))
    max_abs_z = _max_abs_z_from_attrs(idata)
    fit_mask = _fit_window_mask(df, max_abs_z) if df is not None else None
    z_grid = _z_grid_for_posterior(idata, df, n_points=200)

    for color, flat_idx in zip(cmap, _draw_indices(idata, n_samples, seed)):
        draw = _select_draw(idata, int(flat_idx))
        curve = _eval_log_target_on_grid(idata, draw, z_grid, df=df)
        ax.plot(z_grid, curve, color=color, alpha=0.8, linewidth=1.5)
        if df is not None:
            z_obs, log_phi_obs = _obs_scaling_at_points(df, idata, draw)
            if fit_mask is not None:
                z_obs = z_obs[fit_mask]
                log_phi_obs = log_phi_obs[fit_mask]
            ax.scatter(
                z_obs,
                log_phi_obs,
                s=35,
                color=color,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.5,
                zorder=5,
            )

    ax.set_xlabel(r"$z = t L^{1/\nu}$")
    ax.set_ylabel(_scaling_function_log_label(idata))
    title_channel = "Binder cumulant" if _fss_binder_only(idata) else "scaling functions"
    ax.set_title(
        f"{n_samples} joint-posterior {title_channel}"
        + _fit_window_title_suffix(max_abs_z)
    )
    path = out_dir / "posterior_sample_scaling_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_dataset(
    dataset: str,
    *,
    data: Path | None = None,
    posterior: Path | None = None,
    out: Path | None = None,
    L: int | list[int] | None = None,
    T_fixed: float | None = None,
    # Accepted for backward compatibility; ignored.
    model: str = "fss",
    method: str = "marginal",
    variant: str = "plain",
    scaling: str = "gp",
) -> list[Path]:
    if model != "fss":
        raise ValueError(f"Only model='fss' is supported, got {model!r}")

    data_path, posterior_path, plots_path = resolve_paths(
        dataset,
        model="fss",
        method=method,
        variant="plain",
        scaling="gp",
        data=data,
        posterior=posterior,
        plots=out,
    )
    if not posterior_path.exists():
        raise FileNotFoundError(
            f"Posterior not found at {posterior_path}. "
            f"Run: python -m ising.fit --dataset {dataset}"
        )

    idata = az.from_netcdf(posterior_path)
    df = pd.read_csv(data_path)
    validate_observables_df(df)
    means = _posterior_means(idata)
    if L is None:
        t_slice_L_values: list[int] | None = None
    elif isinstance(L, int):
        t_slice_L_values = [L]
    else:
        t_slice_L_values = list(L)

    plots_path.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    exponent_path = plot_exponent_histograms(idata, plots_path)
    if exponent_path is not None:
        paths.append(exponent_path)
    _append_plots(
        paths,
        plot_posterior_traces(idata, plots_path),
    )
    if _fss_include_m(idata):
        _append_plots(
            paths,
            plot_data_collapse_exact(df, plots_path),
            plot_data_collapse(df, means, plots_path),
            plot_posterior_sample_collapses(df, idata, plots_path),
        )
    if _fss_include_binder(idata):
        from .plot_binder import plot_binder_dataset_diagnostics

        paths.extend(
            plot_binder_dataset_diagnostics(
                df,
                plots_path,
                T_c_post=means.get("T_c"),
                nu_post=means.get("nu"),
                dataset=dataset,
            )
        )
    _append_plots(
        paths,
        plot_posterior_sample_scaling_curves(idata, plots_path, df=df),
        plot_scaling_function_posterior(df, idata, plots_path),
        plot_fss_gp_hyperparameter_histograms(idata, plots_path),
    )
    if _fss_include_m(idata):
        paths.extend(
            plot_t_slices(df, idata, plots_path, L_values=t_slice_L_values, T_fixed=T_fixed)
        )
    for path in paths:
        print(f"[{dataset}] Wrote {path}")
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=list_datasets(),
        default="medium",
        help="Named dataset preset",
    )
    parser.add_argument(
        "--harada-bsa",
        action="store_true",
        help="Use published Harada BSA Binder data (sets --dataset harada_bsa)",
    )
    parser.add_argument("--posterior", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--L",
        type=int,
        default=None,
        help="Lattice size for fixed-L panels (default: all sizes in the dataset)",
    )
    parser.add_argument(
        "--T",
        type=float,
        default=None,
        help="Temperature for fixed-T panels (default: grid point nearest exact T_c)",
    )
    args = parser.parse_args(argv)
    if args.harada_bsa:
        args.dataset = "harada_bsa"

    plot_dataset(
        args.dataset,
        data=args.data,
        posterior=args.posterior,
        out=args.out,
        L=args.L,
        T_fixed=args.T,
    )


if __name__ == "__main__":
    main()
