"""Profile log-marginal-likelihood slices for FSS exponent diagnostics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .constants import BETA_EXACT, NU_EXACT, OMEGA_EXACT, TC_EXACT
from .datasets import resolve_paths
from .fss_likelihood import (
    CompiledFssLikelihoodJax,
    LikelihoodBackend,
    compile_fss_log_marginal_likelihood_jax,
)
from .model_fss import beta_affects_fss_likelihood, compile_fss_exponent_log_likelihood
from .model_fss import TC_DELTA_SCALE
from .model_fss import (
    BETA_PRIOR_LOWER,
    BETA_PRIOR_UPPER,
    NU_PRIOR_LOWER,
    NU_PRIOR_UPPER,
)
from .model_scaling_nu import (
    TC_PRIOR_LOWER,
    TC_PRIOR_UPPER,
)
from .gp_kernels import DEFAULT_GP_KERNEL, GpKernelKind, normalize_gp_kernel
from .observables import read_harada_binder_observables, read_observables_for_fss

# --- edit these (run.py imports these for profile + fit) ---
USE_M = True
USE_M2 = True
USE_M4 = True

USE_BINDER = True
CORRECTION_BINDER = False

CORRECTION_M = False
CORRECTION_M2 = False
CORRECTION_M4 = False

USE_CHI = False
CORRECTION_CHI = False

# Optional discrepancy outside the scaling window.
# Form: "noise" inflates σ; "additive_gp" uses raw = L^{-β/ν} f(z) + a g(z) + ε
# (collapsed: Φ = f + a L^{β/ν} g + ε_Φ; analogous L-powers for m², …).
# "additive_gp_z_threshold" uses collapsed a = 1_{|z|>Z_DISC_THRESHOLD} (else 0).
DISCREPANCY_M = False
DISCREPANCY_M2 = False
DISCREPANCY_M4 = False
DISCREPANCY_BINDER = False
DISCREPANCY_CHI = False
DISCREPANCY_FORM: Literal["noise", "additive_gp", "additive_gp_z_threshold"] = "noise"
# Gate for additive_gp_z_threshold: a=1 for |z| > this, else 0 (collapsed Φ-space).
Z_DISC_THRESHOLD = 10.0

GP_ELL_FACTOR = 0.15
# When set, absolute GP length scales (ignore gp_*_ell_factor × z_span).
GP_ELL: float | None = None
GP_ETA = 1.0
CORRECTION_GP_ELL_FACTOR = 0.15
CORRECTION_GP_ELL: float | None = None
CORRECTION_GP_ETA = 1.0
GP_KERNEL: GpKernelKind = DEFAULT_GP_KERNEL

OBS_SIGMA_SCALE = 1.0
USE_LOG_M = False
PROFILE_1D_Y_FLOOR = -10.0
PROFILE_1D_N_POINTS = 240
PROFILE_JOINT_MLE_N_COARSE = 35
PROFILE_JOINT_MLE_N_REFINE = 25
PROFILE_JOINT_MLE_N_STARTS = 12
PROFILE_JOINT_MLE_MAX_REFINE_PASSES = 6
LIKELIHOOD_BACKEND: LikelihoodBackend = "jax"

ProfileSweep = Literal["T_c", "nu", "beta"]
JointPair = Literal["T_c_nu", "T_c_beta", "nu_beta"]

JOINT_PAIR_SPECS: dict[JointPair, tuple[ProfileSweep, ProfileSweep, ProfileSweep]] = {
    "T_c_nu": ("T_c", "nu", "beta"),
    "T_c_beta": ("T_c", "beta", "nu"),
    "nu_beta": ("nu", "beta", "T_c"),
}

EXACT_BY_SWEEP: dict[ProfileSweep, float] = {
    "T_c": TC_EXACT,
    "nu": NU_EXACT,
    "beta": BETA_EXACT,
}

GRID_BOUNDS: dict[ProfileSweep, tuple[float, float]] = {
    "T_c": (TC_PRIOR_LOWER, TC_PRIOR_UPPER),
    "nu": (NU_PRIOR_LOWER, NU_PRIOR_UPPER),
    "beta": (BETA_PRIOR_LOWER, BETA_PRIOR_UPPER),
}


@dataclass(frozen=True)
class FssProfileConfig:
    """FSS model options mirrored from fit/run for profile likelihood evaluation."""

    use_m: bool = USE_M
    use_m2: bool = USE_M2
    use_m4: bool = USE_M4
    use_binder: bool = USE_BINDER
    use_chi: bool = USE_CHI
    correction_m: bool = CORRECTION_M
    correction_m2: bool = CORRECTION_M2
    correction_m4: bool = CORRECTION_M4
    correction_binder: bool = CORRECTION_BINDER
    correction_chi: bool = CORRECTION_CHI
    discrepancy_m: bool = DISCREPANCY_M
    discrepancy_m2: bool = DISCREPANCY_M2
    discrepancy_m4: bool = DISCREPANCY_M4
    discrepancy_binder: bool = DISCREPANCY_BINDER
    discrepancy_chi: bool = DISCREPANCY_CHI
    discrepancy_form: Literal[
        "noise", "additive_gp", "additive_gp_z_threshold"
    ] = DISCREPANCY_FORM
    z_disc_threshold: float = Z_DISC_THRESHOLD
    use_log_m: bool = USE_LOG_M
    omega_fixed: float = OMEGA_EXACT
    gp_ell_factor: float = GP_ELL_FACTOR
    gp_ell: float | None = GP_ELL
    gp_eta: float = GP_ETA
    correction_gp_ell_factor: float = CORRECTION_GP_ELL_FACTOR
    correction_gp_ell: float | None = CORRECTION_GP_ELL
    correction_gp_eta: float = CORRECTION_GP_ETA
    gp_kernel: GpKernelKind = GP_KERNEL
    obs_sigma_scale: float = OBS_SIGMA_SCALE
    infer_Tc: bool = True
    infer_nu: bool = True
    infer_beta: bool = True
    reparametrize_Tc: bool = False
    tc_delta_scale: float = TC_DELTA_SCALE
    likelihood_backend: LikelihoodBackend = LIKELIHOOD_BACKEND


def default_profile_config() -> FssProfileConfig:
    """Return profile settings from the module-level defaults above."""
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
        discrepancy_m=DISCREPANCY_M,
        discrepancy_m2=DISCREPANCY_M2,
        discrepancy_m4=DISCREPANCY_M4,
        discrepancy_binder=DISCREPANCY_BINDER,
        discrepancy_chi=DISCREPANCY_CHI,
        discrepancy_form=DISCREPANCY_FORM,
        z_disc_threshold=Z_DISC_THRESHOLD,
        use_log_m=USE_LOG_M,
        gp_ell_factor=GP_ELL_FACTOR,
        gp_ell=GP_ELL,
        gp_eta=GP_ETA,
        correction_gp_ell_factor=CORRECTION_GP_ELL_FACTOR,
        correction_gp_ell=CORRECTION_GP_ELL,
        correction_gp_eta=CORRECTION_GP_ETA,
        gp_kernel=GP_KERNEL,
        obs_sigma_scale=OBS_SIGMA_SCALE,
        infer_Tc=True,
        infer_nu=True,
        infer_beta=True,
        reparametrize_Tc=False,
        tc_delta_scale=TC_DELTA_SCALE,
        likelihood_backend=LIKELIHOOD_BACKEND,
    )


def _resolve_profile_config(
    config: FssProfileConfig | None,
) -> FssProfileConfig:
    return config or default_profile_config()


def _validate_profile_config(config: FssProfileConfig) -> None:
    if not (
        config.use_m
        or config.use_m2
        or config.use_m4
        or config.use_binder
        or config.use_chi
    ):
        raise ValueError(
            "At least one of use_m, use_m2, use_m4, use_binder, or use_chi must be True"
        )


ProfileObservables = pd.DataFrame


def _resolve_profile_observables(
    *,
    data: Path | None = None,
    dataset: str = "medium",
    observables: ProfileObservables | None = None,
    config: FssProfileConfig | None = None,
) -> pd.DataFrame:
    if observables is not None:
        return observables
    config = _resolve_profile_config(config)
    if dataset == "harada_bsa":
        return read_harada_binder_observables()
    if data is None:
        data, _, _ = resolve_paths(dataset, model="fss")
    return read_observables_for_fss(
        data,
        use_m=config.use_m,
        use_m2=config.use_m2,
        use_m4=config.use_m4,
        use_binder=config.use_binder,
        use_chi=config.use_chi,
    )


def _load_profile_observables(
    *,
    data: Path | None = None,
    dataset: str = "medium",
    config: FssProfileConfig | None = None,
) -> pd.DataFrame:
    return _resolve_profile_observables(data=data, dataset=dataset, config=config)


def _profile_build_kwargs(config: FssProfileConfig) -> dict[str, object]:
    infer_beta = config.infer_beta and beta_affects_fss_likelihood(
        use_m=config.use_m,
        use_m2=config.use_m2,
        use_m4=config.use_m4,
        use_chi=config.use_chi,
    )
    return {
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
        "discrepancy_m": config.discrepancy_m,
        "discrepancy_m2": config.discrepancy_m2,
        "discrepancy_m4": config.discrepancy_m4,
        "discrepancy_binder": config.discrepancy_binder,
        "discrepancy_chi": config.discrepancy_chi,
        "discrepancy_form": config.discrepancy_form,
        "z_disc_threshold": config.z_disc_threshold,
        "use_log_m": config.use_log_m,
        "omega_fixed": config.omega_fixed,
        "gp_ell_factor": config.gp_ell_factor,
        "gp_ell": config.gp_ell,
        "gp_eta": config.gp_eta,
        "correction_gp_ell_factor": config.correction_gp_ell_factor,
        "correction_gp_ell": config.correction_gp_ell,
        "correction_gp_eta": config.correction_gp_eta,
        "gp_kernel": config.gp_kernel,
        "obs_sigma_scale": config.obs_sigma_scale,
        "infer_Tc": config.infer_Tc,
        "infer_nu": config.infer_nu,
        "infer_beta": infer_beta,
        "reparametrize_Tc": config.reparametrize_Tc,
        "tc_delta_scale": config.tc_delta_scale,
    }


def _observables_cache_key(obs: pd.DataFrame, config: FssProfileConfig) -> tuple:
    arrays = tuple(obs[col].to_numpy().tobytes() for col in obs.columns)
    return (arrays, config, config.likelihood_backend)


_LIKELIHOOD_EVALUATORS: dict[
    tuple, Callable[[float, float, float], float]
] = {}
_COMPILED_JAX_LIKELIHOODS: dict[tuple, CompiledFssLikelihoodJax] = {}


def _get_compiled_jax_likelihood(
    observables: pd.DataFrame,
    config: FssProfileConfig,
) -> CompiledFssLikelihoodJax:
    key = _observables_cache_key(observables, config)
    compiled = _COMPILED_JAX_LIKELIHOODS.get(key)
    if compiled is None:
        compiled = compile_fss_log_marginal_likelihood_jax(observables, config)
        _COMPILED_JAX_LIKELIHOODS[key] = compiled
        _LIKELIHOOD_EVALUATORS[key] = compiled.evaluate
    return compiled


def _get_log_likelihood_evaluator(
    observables: pd.DataFrame,
    config: FssProfileConfig,
) -> Callable[[float, float, float], float]:
    key = _observables_cache_key(observables, config)
    evaluator = _LIKELIHOOD_EVALUATORS.get(key)
    if evaluator is None:
        if config.likelihood_backend == "jax":
            evaluator = _get_compiled_jax_likelihood(observables, config).evaluate
        elif config.likelihood_backend == "pymc":
            _, evaluator = compile_fss_exponent_log_likelihood(
                observables,
                **_profile_build_kwargs(config),
            )
        else:
            raise ValueError(
                f"likelihood_backend must be 'jax' or 'pymc', "
                f"got {config.likelihood_backend!r}"
            )
        _LIKELIHOOD_EVALUATORS[key] = evaluator
    return evaluator


def fss_log_marginal_likelihood(
    *,
    observables: pd.DataFrame,
    T_c: float,
    nu: float,
    beta: float,
    config: FssProfileConfig | None = None,
) -> float:
    """Total log marginal GP likelihood with fixed exponents (JAX or PyMC)."""
    cfg = _resolve_profile_config(config)
    _validate_profile_config(cfg)
    evaluator = _get_log_likelihood_evaluator(observables, cfg)
    return evaluator(float(T_c), float(nu), float(beta))


def profile_exponent_log_marginal_likelihood(
    grid: np.ndarray,
    *,
    sweep: ProfileSweep,
    T_c: float = TC_EXACT,
    nu: float = NU_EXACT,
    beta: float = BETA_EXACT,
    data: Path | None = None,
    dataset: str = "medium",
    observables: ProfileObservables | None = None,
    config: FssProfileConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate log marginal likelihood while sweeping one exponent."""
    config = _resolve_profile_config(config)
    obs = _resolve_profile_observables(
        data=data, dataset=dataset, observables=observables, config=config
    )
    grid = np.asarray(grid, dtype=np.float64).ravel()

    if config.likelihood_backend == "jax":
        compiled = _get_compiled_jax_likelihood(obs, config)
        if sweep == "T_c":
            log_ml = compiled.profile_tc(grid, nu, beta)
        elif sweep == "nu":
            log_ml = compiled.profile_nu(grid, T_c, beta)
        elif sweep == "beta":
            log_ml = compiled.profile_beta(grid, T_c, nu)
        else:
            raise ValueError(f"Unsupported sweep {sweep!r}")
        return grid, log_ml

    log_ml = np.empty(grid.size, dtype=np.float64)

    for i, value in enumerate(grid):
        params = dict(T_c=T_c, nu=nu, beta=beta)
        if sweep == "T_c":
            params["T_c"] = float(value)
        elif sweep == "nu":
            params["nu"] = float(value)
        elif sweep == "beta":
            params["beta"] = float(value)
        else:
            raise ValueError(f"Unsupported sweep {sweep!r}")
        log_ml[i] = fss_log_marginal_likelihood(
            observables=obs,
            config=config,
            **params,
        )
    return grid, log_ml


def _default_grid(sweep: ProfileSweep, n_points: int) -> np.ndarray:
    lo, hi = GRID_BOUNDS[sweep]
    grid = np.linspace(lo, hi, n_points)
    exact = EXACT_BY_SWEEP[sweep]
    if lo <= exact <= hi and not np.any(np.isclose(grid, exact, rtol=0.0, atol=1e-10)):
        grid = np.sort(np.append(grid, exact))
    return grid


def _profile_max_log_ml(
    log_ml: np.ndarray,
    *,
    exact_log_ml: float | None = None,
) -> float:
    """Maximum profile log-likelihood, including an off-grid exact evaluation."""
    profile_max = float(np.max(log_ml))
    if exact_log_ml is not None:
        profile_max = max(profile_max, float(exact_log_ml))
    return profile_max


def _exact_profile_delta_log_p(
    log_ml: np.ndarray,
    exact_log_ml: float,
    *,
    normalize: bool = True,
) -> float:
    """Δlog p at the exact exponent relative to the profile slice maximum."""
    if not normalize:
        return float(exact_log_ml)
    return float(exact_log_ml) - _profile_max_log_ml(log_ml, exact_log_ml=exact_log_ml)


def _exponent_values(
    *,
    T_c: float,
    nu: float,
    beta: float,
) -> dict[ProfileSweep, float]:
    return {"T_c": T_c, "nu": nu, "beta": beta}


@dataclass(frozen=True)
class ExactJointLikelihoodResult:
    """Joint log marginal likelihood at the exact exponents vs the joint MLE."""

    T_c_exact: float
    nu_exact: float
    beta_exact: float
    log_ml_exact: float
    T_c_mle: float
    nu_mle: float
    beta_mle: float
    log_ml_mle: float

    @property
    def delta_log_p(self) -> float:
        return self.log_ml_exact - self.log_ml_mle

    @property
    def relative_likelihood(self) -> float:
        return float(np.exp(self.delta_log_p))


def log_marginal_likelihood_at_exponents(
    *,
    T_c: float,
    nu: float,
    beta: float,
    data: Path | None = None,
    dataset: str = "medium",
    observables: ProfileObservables | None = None,
    config: FssProfileConfig | None = None,
) -> float:
    """Log marginal likelihood with all three exponents set simultaneously."""
    config = _resolve_profile_config(config)
    obs = _resolve_profile_observables(
        data=data, dataset=dataset, observables=observables, config=config
    )
    return fss_log_marginal_likelihood(
        observables=obs,
        T_c=T_c,
        nu=nu,
        beta=beta,
        config=config,
    )


def find_joint_mle_log_marginal_likelihood(
    *,
    data: Path | None = None,
    dataset: str = "medium",
    observables: ProfileObservables | None = None,
    config: FssProfileConfig | None = None,
    T_c: float = TC_EXACT,
    nu: float = NU_EXACT,
    beta: float = BETA_EXACT,
    n_points: int = PROFILE_1D_N_POINTS,
    max_iter: int = 8,
) -> tuple[float, float, float, float]:
    """Coordinate-ascent MLE for (T_c, nu, beta) on the joint log marginal likelihood."""
    config = _resolve_profile_config(config)
    obs = _resolve_profile_observables(
        data=data, dataset=dataset, observables=observables, config=config
    )

    def _eval(tc: float, nu_val: float, beta_val: float) -> float:
        return log_marginal_likelihood_at_exponents(
            T_c=tc,
            nu=nu_val,
            beta=beta_val,
            observables=obs,
            config=config,
        )

    candidates: list[tuple[float, float, float]] = [
        (float(T_c), float(nu), float(beta)),
    ]
    for sweep in ("T_c", "nu", "beta"):
        grid, ll = profile_exponent_log_marginal_likelihood(
            _default_grid(sweep, n_points),
            sweep=sweep,
            T_c=T_c,
            nu=nu,
            beta=beta,
            observables=obs,
            config=config,
        )
        params = _exponent_values(T_c=T_c, nu=nu, beta=beta)
        params[sweep] = float(grid[int(np.argmax(ll))])
        candidates.append((params["T_c"], params["nu"], params["beta"]))

    tc, nu_val, beta_val = float(T_c), float(nu), float(beta)
    candidates.append((tc, nu_val, beta_val))

    for _ in range(max_iter):
        prev = (tc, nu_val, beta_val)
        for sweep in ("T_c", "nu", "beta"):
            grid, ll = profile_exponent_log_marginal_likelihood(
                _default_grid(sweep, n_points),
                sweep=sweep,
                T_c=tc,
                nu=nu_val,
                beta=beta_val,
                observables=obs,
                config=config,
            )
            value = float(grid[int(np.argmax(ll))])
            if sweep == "T_c":
                tc = value
            elif sweep == "nu":
                nu_val = value
            else:
                beta_val = value
        candidates.append((tc, nu_val, beta_val))
        if (tc, nu_val, beta_val) == prev:
            break

    best_tc, best_nu, best_beta = candidates[0]
    best_ll = _eval(*candidates[0])
    for tc_c, nu_c, beta_c in candidates[1:]:
        ll = _eval(tc_c, nu_c, beta_c)
        if ll > best_ll:
            best_ll = ll
            best_tc, best_nu, best_beta = tc_c, nu_c, beta_c

    return best_tc, best_nu, best_beta, best_ll


def _refine_axis_grid(
    sweep: ProfileSweep,
    center: float,
    *,
    n_coarse: int,
    n_points: int,
) -> np.ndarray:
    lo, hi = GRID_BOUNDS[sweep]
    if n_coarse <= 1:
        half = 0.5 * (hi - lo)
    else:
        half = 0.5 * (hi - lo) / (n_coarse - 1)
    return np.linspace(
        max(lo, center - 2.0 * half),
        min(hi, center + 2.0 * half),
        n_points,
    )


def _normalized_triple_distance(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> float:
    spans = {
        "T_c": GRID_BOUNDS["T_c"][1] - GRID_BOUNDS["T_c"][0],
        "nu": GRID_BOUNDS["nu"][1] - GRID_BOUNDS["nu"][0],
        "beta": GRID_BOUNDS["beta"][1] - GRID_BOUNDS["beta"][0],
    }
    d_tc = abs(a[0] - b[0]) / spans["T_c"]
    d_nu = abs(a[1] - b[1]) / spans["nu"]
    d_beta = abs(a[2] - b[2]) / spans["beta"]
    return float(max(d_tc, d_nu, d_beta))


def _dedupe_triple_centers(
    centers: list[tuple[float, float, float]],
    *,
    decimals: tuple[int, int, int] = (6, 6, 6),
) -> list[tuple[float, float, float]]:
    seen: set[tuple[float, float, float]] = set()
    unique: list[tuple[float, float, float]] = []
    for tc, nu_val, beta_val in centers:
        key = (
            round(float(tc), decimals[0]),
            round(float(nu_val), decimals[1]),
            round(float(beta_val), decimals[2]),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append((float(tc), float(nu_val), float(beta_val)))
    return unique


def _anchor_slice_grid_centers(
    *,
    observables: ProfileObservables,
    config: FssProfileConfig,
    n_points: int = PROFILE_1D_N_POINTS,
) -> list[tuple[float, float, float]]:
    """1D grid maxima with the other two exponents fixed at the exact values."""
    centers: list[tuple[float, float, float]] = [
        (float(TC_EXACT), float(NU_EXACT), float(BETA_EXACT)),
    ]
    for sweep in ("T_c", "nu", "beta"):
        grid, ll = profile_exponent_log_marginal_likelihood(
            _default_grid(sweep, n_points),
            sweep=sweep,
            T_c=TC_EXACT,
            nu=NU_EXACT,
            beta=BETA_EXACT,
            observables=observables,
            config=config,
        )
        params = _exponent_values(T_c=TC_EXACT, nu=NU_EXACT, beta=BETA_EXACT)
        params[sweep] = float(grid[int(np.argmax(ll))])
        centers.append((params["T_c"], params["nu"], params["beta"]))
    return centers


def _coarse_topk_grid_centers(
    eval_fn: Callable[[float, float, float], float],
    *,
    n_coarse: int,
    n_starts: int,
    min_sep: float = 0.12,
    batch_eval_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    | None = None,
) -> list[tuple[float, float, float]]:
    """Top coarse 3D grid points by likelihood, greedily spaced in normalized space."""
    tc_grid = _default_grid("T_c", n_coarse)
    nu_grid = _default_grid("nu", n_coarse)
    beta_grid = _default_grid("beta", n_coarse)
    ranked: list[tuple[float, float, float, float]] = []
    if batch_eval_fn is not None:
        ll_cube = batch_eval_fn(tc_grid, nu_grid, beta_grid)
        for i, tc in enumerate(tc_grid):
            for j, nu_val in enumerate(nu_grid):
                for k, beta_val in enumerate(beta_grid):
                    ranked.append(
                        (float(ll_cube[i, j, k]), float(tc), float(nu_val), float(beta_val))
                    )
    else:
        for tc in tc_grid:
            for nu_val in nu_grid:
                for beta_val in beta_grid:
                    ll = eval_fn(float(tc), float(nu_val), float(beta_val))
                    ranked.append((ll, float(tc), float(nu_val), float(beta_val)))
    ranked.sort(key=lambda item: item[0], reverse=True)

    picked: list[tuple[float, float, float]] = []
    for _, tc, nu_val, beta_val in ranked:
        candidate = (tc, nu_val, beta_val)
        if all(
            _normalized_triple_distance(candidate, center) >= min_sep for center in picked
        ):
            picked.append(candidate)
        if len(picked) >= n_starts:
            break
    return picked


def _refine_joint_grid_center(
    tc: float,
    nu_val: float,
    beta_val: float,
    eval_fn: Callable[[float, float, float], float],
    *,
    n_coarse: int,
    n_refine: int,
    max_passes: int,
) -> tuple[float, float, float, float]:
    """Iterative local 3D grid refinement around a starting triple."""
    best_tc, best_nu, best_beta = float(tc), float(nu_val), float(beta_val)
    best_ll = eval_fn(best_tc, best_nu, best_beta)
    for _ in range(max_passes):
        improved = False
        for tc_r in _refine_axis_grid("T_c", best_tc, n_coarse=n_coarse, n_points=n_refine):
            for nu_r in _refine_axis_grid(
                "nu", best_nu, n_coarse=n_coarse, n_points=n_refine
            ):
                for beta_r in _refine_axis_grid(
                    "beta", best_beta, n_coarse=n_coarse, n_points=n_refine
                ):
                    ll = eval_fn(float(tc_r), float(nu_r), float(beta_r))
                    if ll > best_ll:
                        best_ll = ll
                        best_tc, best_nu, best_beta = float(tc_r), float(nu_r), float(beta_r)
                        improved = True
        if not improved:
            break
    return best_tc, best_nu, best_beta, best_ll


def find_joint_argmax_log_marginal_likelihood(
    *,
    data: Path | None = None,
    dataset: str = "medium",
    observables: ProfileObservables | None = None,
    config: FssProfileConfig | None = None,
    n_coarse: int = PROFILE_JOINT_MLE_N_COARSE,
    n_refine: int = PROFILE_JOINT_MLE_N_REFINE,
    n_starts: int = PROFILE_JOINT_MLE_N_STARTS,
    max_refine_passes: int = PROFILE_JOINT_MLE_MAX_REFINE_PASSES,
) -> tuple[float, float, float, float]:
    """Global argmax of (T_c, nu, beta) via multi-start coarse grid + local 3D refinement."""
    config = _resolve_profile_config(config)
    obs = _resolve_profile_observables(
        data=data, dataset=dataset, observables=observables, config=config
    )
    n_coarse = max(n_coarse, 5)
    n_refine = max(n_refine, 5)
    n_starts = max(n_starts, 1)
    max_refine_passes = max(max_refine_passes, 1)

    def _eval(tc: float, nu_val: float, beta_val: float) -> float:
        return log_marginal_likelihood_at_exponents(
            T_c=tc,
            nu=nu_val,
            beta=beta_val,
            observables=obs,
            config=config,
        )

    batch_eval_fn = None
    if config.likelihood_backend == "jax":
        batch_eval_fn = _get_compiled_jax_likelihood(obs, config).profile_triple_grid

    centers = _dedupe_triple_centers(
        [
            *_anchor_slice_grid_centers(
                observables=obs,
                config=config,
            ),
            *_coarse_topk_grid_centers(
                _eval,
                n_coarse=n_coarse,
                n_starts=n_starts,
                batch_eval_fn=batch_eval_fn,
            ),
        ]
    )

    best_tc, best_nu, best_beta = centers[0]
    best_ll = _eval(*centers[0])
    for tc, nu_val, beta_val in centers:
        tc_r, nu_r, beta_r, ll = _refine_joint_grid_center(
            tc,
            nu_val,
            beta_val,
            _eval,
            n_coarse=n_coarse,
            n_refine=n_refine,
            max_passes=max_refine_passes,
        )
        if ll > best_ll:
            best_ll = ll
            best_tc, best_nu, best_beta = tc_r, nu_r, beta_r

    return best_tc, best_nu, best_beta, best_ll


def evaluate_exact_joint_log_marginal_likelihood(
    *,
    data: Path | None = None,
    dataset: str = "medium",
    observables: ProfileObservables | None = None,
    config: FssProfileConfig | None = None,
    T_c_exact: float = TC_EXACT,
    nu_exact: float = NU_EXACT,
    beta_exact: float = BETA_EXACT,
    n_coarse: int = PROFILE_JOINT_MLE_N_COARSE,
    n_refine: int = PROFILE_JOINT_MLE_N_REFINE,
    n_starts: int = PROFILE_JOINT_MLE_N_STARTS,
    max_refine_passes: int = PROFILE_JOINT_MLE_MAX_REFINE_PASSES,
) -> ExactJointLikelihoodResult:
    """Joint log marginal likelihood at the exact triple and at the global joint MLE."""
    config = _resolve_profile_config(config)
    obs = _resolve_profile_observables(
        data=data, dataset=dataset, observables=observables, config=config
    )
    log_ml_exact = log_marginal_likelihood_at_exponents(
        T_c=T_c_exact,
        nu=nu_exact,
        beta=beta_exact,
        observables=obs,
        config=config,
    )
    T_c_mle, nu_mle, beta_mle, log_ml_mle = find_joint_argmax_log_marginal_likelihood(
        observables=obs,
        config=config,
        n_coarse=n_coarse,
        n_refine=n_refine,
        n_starts=n_starts,
        max_refine_passes=max_refine_passes,
    )
    return ExactJointLikelihoodResult(
        T_c_exact=T_c_exact,
        nu_exact=nu_exact,
        beta_exact=beta_exact,
        log_ml_exact=log_ml_exact,
        T_c_mle=T_c_mle,
        nu_mle=nu_mle,
        beta_mle=beta_mle,
        log_ml_mle=log_ml_mle,
    )


def format_exact_joint_likelihood_summary(result: ExactJointLikelihoodResult) -> str:
    """Human-readable summary of the exact triple joint log marginal likelihood."""
    ratio_str = _format_relative_likelihood(result.delta_log_p)
    return (
        "Exact joint log marginal likelihood "
        f"(T_c={result.T_c_exact:.6g}, nu={result.nu_exact:.6g}, "
        f"beta={result.beta_exact:.6g}):\n"
        f"  log p(data | exact) = {result.log_ml_exact:.6g}\n"
        f"  global joint MLE: T_c={result.T_c_mle:.6g}, nu={result.nu_mle:.6g}, "
        f"beta={result.beta_mle:.6g}, log p = {result.log_ml_mle:.6g}\n"
        f"  Δlog p (exact − MLE) = {result.delta_log_p:.6g}\n"
        f"  p(exact) / p(MLE) = {ratio_str}"
    )


def profile_joint_exponent_log_marginal_likelihood(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    *,
    pair: JointPair,
    T_c: float = TC_EXACT,
    nu: float = NU_EXACT,
    beta: float = BETA_EXACT,
    data: Path | None = None,
    dataset: str = "medium",
    observables: ProfileObservables | None = None,
    config: FssProfileConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate log marginal likelihood on a 2D exponent grid."""
    config = _resolve_profile_config(config)
    obs = _resolve_profile_observables(
        data=data, dataset=dataset, observables=observables, config=config
    )
    x_grid = np.asarray(x_grid, dtype=np.float64).ravel()
    y_grid = np.asarray(y_grid, dtype=np.float64).ravel()
    sweep_x, sweep_y, fixed_sweep = JOINT_PAIR_SPECS[pair]
    values = _exponent_values(T_c=T_c, nu=nu, beta=beta)

    if config.likelihood_backend == "jax":
        compiled = _get_compiled_jax_likelihood(obs, config)
        log_ml = compiled.profile_joint(
            x_grid,
            y_grid,
            pair,
            values["T_c"],
            values["nu"],
            values["beta"],
        )
        return x_grid, y_grid, log_ml

    log_ml = np.empty((y_grid.size, x_grid.size), dtype=np.float64)

    for j, y_val in enumerate(y_grid):
        for i, x_val in enumerate(x_grid):
            params = dict(values)
            params[sweep_x] = float(x_val)
            params[sweep_y] = float(y_val)
            log_ml[j, i] = fss_log_marginal_likelihood(
                observables=obs,
                T_c=params["T_c"],
                nu=params["nu"],
                beta=params["beta"],
                config=config,
            )
    return x_grid, y_grid, log_ml


def _format_relative_likelihood(log_ratio: float) -> str:
    """Format p/p_max = exp(log_ratio) as a plain decimal string."""
    ratio = min(float(np.exp(log_ratio)), 1.0)
    if not np.isfinite(ratio) or ratio <= 0.0:
        return "0"
    if ratio >= 1.0:
        return format(ratio, ".8f").rstrip("0").rstrip(".")
    if ratio < 1e-4:
        return format(ratio, ".4e")
    decimals = min(16, max(4, int(np.ceil(-np.log10(ratio))) + 3))
    return format(ratio, f".{decimals}f").rstrip("0").rstrip(".")


def _plot_joint_heatmap_axis(
    ax: plt.Axes,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    log_ml: np.ndarray,
    *,
    sweep_x: ProfileSweep,
    sweep_y: ProfileSweep,
    fixed_sweep: ProfileSweep,
    fixed_value: float,
    joint_mle: dict[ProfileSweep, float] | None = None,
    exact_delta_log_p: float | None = None,
    normalize: bool = True,
) -> None:
    z = log_ml.copy()
    log_ml_max = float(np.max(log_ml))
    if normalize:
        z = z - log_ml_max

    labels = {
        "T_c": r"$T_c$",
        "nu": r"$\nu$",
        "beta": r"$\beta$",
    }
    xx, yy = np.meshgrid(x_grid, y_grid)
    mesh = ax.pcolormesh(
        xx,
        yy,
        z,
        shading="auto",
        cmap="viridis",
        vmin=max(float(z.min()), -80.0),
        vmax=0.0,
    )
    exact_x = EXACT_BY_SWEEP[sweep_x]
    exact_y = EXACT_BY_SWEEP[sweep_y]
    ax.plot(exact_x, exact_y, "w*", markersize=12, markeredgecolor="black", label="exact")
    if joint_mle is not None:
        mle_x = joint_mle[sweep_x]
        mle_y = joint_mle[sweep_y]
        mle_label = "joint MLE"
    else:
        mle_idx = np.unravel_index(int(np.argmax(log_ml)), log_ml.shape)
        mle_x = float(x_grid[mle_idx[1]])
        mle_y = float(y_grid[mle_idx[0]])
        mle_label = "2D MLE"
    ax.plot(
        mle_x,
        mle_y,
        "x",
        color="#ffb000",
        markersize=10,
        mew=2.0,
        label=mle_label,
    )
    if exact_delta_log_p is not None:
        ratio_str = _format_relative_likelihood(exact_delta_log_p)
        ax.annotate(
            rf"exact: $p/p_{{\max}} = {ratio_str}$",
            xy=(exact_x, exact_y),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=8,
            color="white",
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
            zorder=5,
        )
    ax.set_xlabel(labels[sweep_x])
    ax.set_ylabel(labels[sweep_y])
    ax.set_title(
        rf"Joint profile LL ({labels[fixed_sweep]}={fixed_value:g} fixed)"
    )
    ax.legend(loc="best", fontsize=7, framealpha=0.85)
    plt.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04, label=r"$\Delta\log p$")


def _profile_xlim_zoomed(
    grid: np.ndarray,
    y: np.ndarray,
    *,
    exact_x: float,
    mle_x: float,
    y_floor: float = PROFILE_1D_Y_FLOOR,
    pad_fraction: float = 0.08,
    min_span_fraction: float = 0.04,
) -> tuple[float, float]:
    """X limits covering MLE, exact, and grid points with Δlog p above y_floor."""
    mask = y >= y_floor
    if np.any(mask):
        x_lo = float(np.min(grid[mask]))
        x_hi = float(np.max(grid[mask]))
    else:
        x_lo = min(exact_x, mle_x)
        x_hi = max(exact_x, mle_x)
    x_lo = min(x_lo, exact_x, mle_x)
    x_hi = max(x_hi, exact_x, mle_x)
    full_span = float(np.max(grid) - np.min(grid))
    min_span = max(min_span_fraction * full_span, 1e-6)
    center = 0.5 * (exact_x + mle_x)
    span = max(x_hi - x_lo, min_span)
    x_lo = min(center - 0.5 * span, x_lo, exact_x, mle_x)
    x_hi = max(center + 0.5 * span, x_hi, exact_x, mle_x)
    span = max(x_hi - x_lo, min_span)
    if span < min_span:
        center = 0.5 * (x_lo + x_hi)
        x_lo = center - 0.5 * min_span
        x_hi = center + 0.5 * min_span
    pad = pad_fraction * span
    return x_lo - pad, x_hi + pad


def _annotate_1d_exact_relative_likelihood(
    ax: plt.Axes,
    *,
    exact_x: float,
    x_lo: float,
    x_hi: float,
    y_floor: float,
    ratio_str: str,
) -> None:
    """Place p/p_max label inside the axes, away from clipped panel edges."""
    x_span = max(x_hi - x_lo, 1e-6)
    x_frac = (exact_x - x_lo) / x_span
    y_text = y_floor + 0.10 * (0.05 - y_floor)
    if x_frac < 0.25:
        x_text = x_lo + 0.04 * x_span
        ha = "left"
    elif x_frac > 0.75:
        x_text = x_hi - 0.04 * x_span
        ha = "right"
    else:
        x_text = exact_x
        ha = "center"
    ax.text(
        x_text,
        y_text,
        rf"exact: $p/p_{{\max}} = {ratio_str}$",
        fontsize=8,
        ha=ha,
        va="bottom",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            alpha=0.85,
            edgecolor="0.7",
        ),
        zorder=5,
        clip_on=False,
    )


def _plot_1d_profile_axis(
    ax: plt.Axes,
    grid: np.ndarray,
    log_ml: np.ndarray,
    *,
    sweep: ProfileSweep,
    fixed_a: ProfileSweep,
    fixed_b: ProfileSweep,
    values: dict[ProfileSweep, float],
    labels: dict[ProfileSweep, str],
    exact_delta_log_p: float,
    normalize: bool = True,
    y_floor: float = PROFILE_1D_Y_FLOOR,
    profile_max: float | None = None,
) -> None:
    profile_max = (
        float(np.max(log_ml)) if profile_max is None else float(profile_max)
    )
    y = log_ml - profile_max if normalize else log_ml
    exact_x = EXACT_BY_SWEEP[sweep]
    mle_x = float(grid[int(np.argmax(log_ml))])
    coincident = bool(np.isclose(exact_x, mle_x, rtol=0.0, atol=1e-9))

    ax.plot(grid, y, color="#4c72b0", linewidth=2.0, zorder=1)
    ax.axvline(
        mle_x,
        color="#ffb000",
        linestyle="-",
        linewidth=1.5,
        label="1D MLE" if not coincident else "exact (= 1D MLE)",
        zorder=2,
    )
    if not coincident:
        ax.axvline(
            exact_x,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="exact",
            zorder=3,
        )
    else:
        ax.axvline(
            exact_x,
            color="black",
            linestyle="--",
            linewidth=2.0,
            label="_nolegend_",
            zorder=3,
        )
    ax.set_xlabel(labels[sweep])
    ax.set_ylabel(r"$\Delta\log p$" if normalize else r"$\log p$")
    ax.set_title(
        rf"Profile LL ({labels[fixed_a]}={values[fixed_a]:g}, "
        rf"{labels[fixed_b]}={values[fixed_b]:g} fixed)"
    )
    ax.set_ylim(y_floor, 0.05)
    x_lo, x_hi = _profile_xlim_zoomed(
        grid,
        y,
        exact_x=exact_x,
        mle_x=mle_x,
        y_floor=y_floor,
    )
    ax.set_xlim(x_lo, x_hi)
    ratio_str = _format_relative_likelihood(exact_delta_log_p)
    _annotate_1d_exact_relative_likelihood(
        ax,
        exact_x=exact_x,
        x_lo=x_lo,
        x_hi=x_hi,
        y_floor=y_floor,
        ratio_str=ratio_str,
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)


def plot_exponent_joint_profile_likelihoods(
    dataset: str,
    out_dir: Path,
    *,
    T_c: float = TC_EXACT,
    nu: float = NU_EXACT,
    beta: float = BETA_EXACT,
    data: Path | None = None,
    config: FssProfileConfig | None = None,
    normalize: bool = True,
    n_points_1d: int = PROFILE_1D_N_POINTS,
    n_points_2d: int = 55,
    exact_joint: ExactJointLikelihoodResult | None = None,
) -> Path:
    """Six-panel figure: three joint heatmaps and three 1D profile slices."""
    config = _resolve_profile_config(config)
    obs = _resolve_profile_observables(data=data, dataset=dataset, config=config)
    values = _exponent_values(T_c=T_c, nu=nu, beta=beta)
    exact_joint = exact_joint or evaluate_exact_joint_log_marginal_likelihood(
        data=data,
        dataset=dataset,
        observables=obs,
        config=config,
        T_c_exact=TC_EXACT,
        nu_exact=NU_EXACT,
        beta_exact=BETA_EXACT,
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    labels = {"T_c": r"$T_c$", "nu": r"$\nu$", "beta": r"$\beta$"}

    pairs: tuple[JointPair, ...] = ("T_c_nu", "T_c_beta", "nu_beta")
    for ax, pair in zip(axes[0], pairs):
        sweep_x, sweep_y, fixed_sweep = JOINT_PAIR_SPECS[pair]
        x_grid = _default_grid(sweep_x, n_points_2d)
        y_grid = _default_grid(sweep_y, n_points_2d)
        _, _, log_ml = profile_joint_exponent_log_marginal_likelihood(
            x_grid,
            y_grid,
            pair=pair,
            T_c=values["T_c"],
            nu=values["nu"],
            beta=values["beta"],
            observables=obs,
            config=config,
        )
        exact_at_star = _exponent_values(
            T_c=values["T_c"],
            nu=values["nu"],
            beta=values["beta"],
        )
        exact_at_star[sweep_x] = EXACT_BY_SWEEP[sweep_x]
        exact_at_star[sweep_y] = EXACT_BY_SWEEP[sweep_y]
        exact_log_ml = fss_log_marginal_likelihood(
            observables=obs,
            T_c=exact_at_star["T_c"],
            nu=exact_at_star["nu"],
            beta=exact_at_star["beta"],
            config=config,
        )
        exact_delta = _exact_profile_delta_log_p(
            log_ml,
            exact_log_ml,
            normalize=normalize,
        )
        _plot_joint_heatmap_axis(
            ax,
            x_grid,
            y_grid,
            log_ml,
            sweep_x=sweep_x,
            sweep_y=sweep_y,
            fixed_sweep=fixed_sweep,
            fixed_value=values[fixed_sweep],
            exact_delta_log_p=exact_delta,
            normalize=normalize,
        )

    marginal_specs: tuple[tuple[ProfileSweep, ProfileSweep, ProfileSweep], ...] = (
        ("T_c", "nu", "beta"),
        ("beta", "T_c", "nu"),
        ("nu", "T_c", "beta"),
    )
    for ax, (sweep, fixed_a, fixed_b) in zip(axes[1], marginal_specs):
        grid = _default_grid(sweep, n_points_1d)
        _, log_ml = profile_exponent_log_marginal_likelihood(
            grid,
            sweep=sweep,
            T_c=values["T_c"],
            nu=values["nu"],
            beta=values["beta"],
            observables=obs,
            config=config,
        )
        exact_at = _exponent_values(
            T_c=values["T_c"],
            nu=values["nu"],
            beta=values["beta"],
        )
        exact_at[sweep] = EXACT_BY_SWEEP[sweep]
        exact_log_ml = fss_log_marginal_likelihood(
            observables=obs,
            T_c=exact_at["T_c"],
            nu=exact_at["nu"],
            beta=exact_at["beta"],
            config=config,
        )
        profile_max = _profile_max_log_ml(log_ml, exact_log_ml=exact_log_ml)
        exact_delta = _exact_profile_delta_log_p(
            log_ml,
            exact_log_ml,
            normalize=normalize,
        )
        _plot_1d_profile_axis(
            ax,
            grid,
            log_ml,
            sweep=sweep,
            fixed_a=fixed_a,
            fixed_b=fixed_b,
            values=values,
            labels=labels,
            exact_delta_log_p=exact_delta,
            normalize=normalize,
            profile_max=profile_max,
        )

    fig.suptitle(
        "Profile log marginal likelihood: joint heatmaps and 1D slices "
        f"(obs_sigma_scale={config.obs_sigma_scale:g})\n"
        rf"Global joint MLE: $T_c={exact_joint.T_c_mle:.6g}$, $\nu={exact_joint.nu_mle:.5g}$, "
        rf"$\beta={exact_joint.beta_mle:.5g}$, $\log p={exact_joint.log_ml_mle:.4g}$; "
        rf"exact triple $\log p={exact_joint.log_ml_exact:.4g}$, "
        rf"$p_{{\mathrm{{exact}}}}/p_{{\mathrm{{MLE}}}} = "
        f"{_format_relative_likelihood(exact_joint.delta_log_p)}$ "
        rf"(heatmaps: third exponent fixed at exact; $\times$ = 2D MLE on each slice)",
        fontsize=11,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "profile_exponent_joint_likelihoods.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def profile_tc_log_marginal_likelihood(
    tc_grid: np.ndarray,
    *,
    nu: float = NU_EXACT,
    beta: float = BETA_EXACT,
    data: Path | None = None,
    dataset: str = "medium",
    config: FssProfileConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the FSS log marginal likelihood over a grid of T_c values."""
    return profile_exponent_log_marginal_likelihood(
        tc_grid,
        sweep="T_c",
        T_c=TC_EXACT,
        nu=nu,
        beta=beta,
        data=data,
        dataset=dataset,
        config=config,
    )


def profile_nu_log_marginal_likelihood(
    nu_grid: np.ndarray,
    *,
    T_c: float = TC_EXACT,
    beta: float = BETA_EXACT,
    data: Path | None = None,
    dataset: str = "medium",
    config: FssProfileConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the FSS log marginal likelihood over a grid of nu values."""
    return profile_exponent_log_marginal_likelihood(
        nu_grid,
        sweep="nu",
        T_c=T_c,
        nu=NU_EXACT,
        beta=beta,
        data=data,
        dataset=dataset,
        config=config,
    )


def profile_beta_log_marginal_likelihood(
    beta_grid: np.ndarray,
    *,
    T_c: float = TC_EXACT,
    nu: float = NU_EXACT,
    data: Path | None = None,
    dataset: str = "medium",
    config: FssProfileConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the FSS log marginal likelihood over a grid of beta values."""
    return profile_exponent_log_marginal_likelihood(
        beta_grid,
        sweep="beta",
        T_c=T_c,
        nu=nu,
        beta=BETA_EXACT,
        data=data,
        dataset=dataset,
        config=config,
    )


def _plot_profile_curve(
    grid: np.ndarray,
    log_ml: np.ndarray,
    *,
    sweep: ProfileSweep,
    fixed_label: str,
    exact_value: float,
    out_path: Path,
    normalize: bool = True,
) -> Path:
    y = log_ml.copy()
    if normalize:
        y = y - np.max(y)

    labels = {
        "T_c": (r"$T_c$", r"$\log p(\mathrm{data}\mid T_c, \nu, \beta)$", "T_c"),
        "nu": (r"$\nu$", r"$\log p(\mathrm{data}\mid \nu, T_c, \beta)$", r"\nu"),
        "beta": (r"$\beta$", r"$\log p(\mathrm{data}\mid \beta, T_c, \nu)$", r"\beta"),
    }
    xlabel, ylabel_base, tex_name = labels[sweep]
    ylabel = ylabel_base + (" (shifted to max = 0)" if normalize else "")

    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    ax.plot(grid, y, color="#4c72b0", linewidth=2.0)
    ax.axvline(
        exact_value,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=rf"exact ${tex_name}$ = {exact_value:.4g}",
    )
    mle = float(grid[np.argmax(log_ml)])
    ax.axvline(
        mle,
        color="#c44e52",
        linestyle="-.",
        linewidth=1.5,
        label=rf"MLE ${tex_name}$ = {mle:.4g}",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Profile log marginal likelihood ({fixed_label} fixed)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.25)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_tc_marginal_likelihood(
    dataset: str,
    out_dir: Path,
    *,
    nu: float = NU_EXACT,
    beta: float = BETA_EXACT,
    data: Path | None = None,
    tc_grid: np.ndarray | None = None,
    config: FssProfileConfig | None = None,
    normalize: bool = True,
    n_points: int = PROFILE_1D_N_POINTS,
) -> Path:
    """Plot log marginal likelihood vs T_c with nu and beta fixed."""
    if tc_grid is None:
        tc_grid = np.linspace(TC_PRIOR_LOWER, TC_PRIOR_UPPER, n_points)
    tc_grid, log_ml = profile_tc_log_marginal_likelihood(
        tc_grid,
        nu=nu,
        beta=beta,
        data=data,
        dataset=dataset,
        config=config,
    )
    return _plot_profile_curve(
        tc_grid,
        log_ml,
        sweep="T_c",
        fixed_label=rf"$\nu={nu:g}$, $\beta={beta:g}$",
        exact_value=TC_EXACT,
        out_path=out_dir / "profile_tc_marginal_likelihood.png",
        normalize=normalize,
    )


def plot_nu_marginal_likelihood(
    dataset: str,
    out_dir: Path,
    *,
    T_c: float = TC_EXACT,
    beta: float = BETA_EXACT,
    data: Path | None = None,
    nu_grid: np.ndarray | None = None,
    config: FssProfileConfig | None = None,
    normalize: bool = True,
    n_points: int = PROFILE_1D_N_POINTS,
) -> Path:
    """Plot log marginal likelihood vs nu with T_c and beta fixed."""
    if nu_grid is None:
        nu_grid = np.linspace(NU_PRIOR_LOWER, NU_PRIOR_UPPER, n_points)
    nu_grid, log_ml = profile_nu_log_marginal_likelihood(
        nu_grid,
        T_c=T_c,
        beta=beta,
        data=data,
        dataset=dataset,
        config=config,
    )
    return _plot_profile_curve(
        nu_grid,
        log_ml,
        sweep="nu",
        fixed_label=rf"$T_c={T_c:g}$, $\beta={beta:g}$",
        exact_value=NU_EXACT,
        out_path=out_dir / "profile_nu_marginal_likelihood.png",
        normalize=normalize,
    )


def plot_beta_marginal_likelihood(
    dataset: str,
    out_dir: Path,
    *,
    T_c: float = TC_EXACT,
    nu: float = NU_EXACT,
    data: Path | None = None,
    beta_grid: np.ndarray | None = None,
    config: FssProfileConfig | None = None,
    normalize: bool = True,
    n_points: int = PROFILE_1D_N_POINTS,
) -> Path:
    """Plot log marginal likelihood vs beta with T_c and nu fixed."""
    if beta_grid is None:
        beta_grid = np.linspace(BETA_PRIOR_LOWER, BETA_PRIOR_UPPER, n_points)
    beta_grid, log_ml = profile_beta_log_marginal_likelihood(
        beta_grid,
        T_c=T_c,
        nu=nu,
        data=data,
        dataset=dataset,
        config=config,
    )
    return _plot_profile_curve(
        beta_grid,
        log_ml,
        sweep="beta",
        fixed_label=rf"$T_c={T_c:g}$, $\nu={nu:g}$",
        exact_value=BETA_EXACT,
        out_path=out_dir / "profile_beta_marginal_likelihood.png",
        normalize=normalize,
    )


def plot_exponent_profile_likelihoods(
    dataset: str,
    out_dir: Path,
    *,
    T_c: float = TC_EXACT,
    nu: float = NU_EXACT,
    beta: float = BETA_EXACT,
    data: Path | None = None,
    config: FssProfileConfig | None = None,
    normalize: bool = True,
    n_points: int = PROFILE_1D_N_POINTS,
) -> list[Path]:
    """Plot profile log marginal likelihood slices for T_c, nu, and beta."""
    return [
        plot_tc_marginal_likelihood(
            dataset,
            out_dir,
            nu=nu,
            beta=beta,
            data=data,
            config=config,
            normalize=normalize,
            n_points=n_points,
        ),
        plot_nu_marginal_likelihood(
            dataset,
            out_dir,
            T_c=T_c,
            beta=beta,
            data=data,
            config=config,
            normalize=normalize,
            n_points=n_points,
        ),
        plot_beta_marginal_likelihood(
            dataset,
            out_dir,
            T_c=T_c,
            nu=nu,
            data=data,
            config=config,
            normalize=normalize,
            n_points=n_points,
        ),
    ]


def add_fss_profile_arguments(parser: argparse.ArgumentParser) -> None:
    """Register FSS channel / GP flags (mirrors fit.py and run.py)."""
    parser.add_argument(
        "--use-m",
        action=argparse.BooleanOptionalAction,
        default=USE_M,
        help="Include magnetization |m| in the profile likelihood",
    )
    parser.add_argument(
        "--use-m2",
        action=argparse.BooleanOptionalAction,
        default=USE_M2,
        help="Include mean m^2 (Phi_m2 = m^2 L^(2 beta/nu))",
    )
    parser.add_argument(
        "--use-m4",
        action=argparse.BooleanOptionalAction,
        default=USE_M4,
        help="Include mean m^4 (Phi_m4 = m^4 L^(4 beta/nu))",
    )
    parser.add_argument(
        "--use-binder",
        action=argparse.BooleanOptionalAction,
        default=USE_BINDER,
        help="Include Binder cumulant U_4",
    )
    parser.add_argument(
        "--use-chi",
        action=argparse.BooleanOptionalAction,
        default=USE_CHI,
        help="Include susceptibility chi",
    )
    parser.add_argument(
        "--correction-m",
        action=argparse.BooleanOptionalAction,
        default=CORRECTION_M,
        help="Use L^-omega f1(z) correction for magnetization",
    )
    parser.add_argument(
        "--correction-m2",
        action=argparse.BooleanOptionalAction,
        default=CORRECTION_M2,
        help="Use L^-omega f1(z) correction for m^2",
    )
    parser.add_argument(
        "--correction-m4",
        action=argparse.BooleanOptionalAction,
        default=CORRECTION_M4,
        help="Use L^-omega f1(z) correction for m^4",
    )
    parser.add_argument(
        "--correction-binder",
        action=argparse.BooleanOptionalAction,
        default=CORRECTION_BINDER,
        help="Use L^-omega f1(z) correction for Binder cumulant",
    )
    parser.add_argument(
        "--correction-chi",
        action=argparse.BooleanOptionalAction,
        default=CORRECTION_CHI,
        help="Use L^-omega f1(z) correction for susceptibility",
    )
    parser.add_argument(
        "--gp-ell-factor",
        type=float,
        default=GP_ELL_FACTOR,
        help="f0(z) GP length scale as factor * z_span",
    )
    parser.add_argument(
        "--gp-eta",
        type=float,
        default=GP_ETA,
        help="f0(z) GP amplitude eta",
    )
    parser.add_argument(
        "--correction-gp-ell-factor",
        type=float,
        default=CORRECTION_GP_ELL_FACTOR,
        help="f1(z) GP length scale as factor * z_span",
    )
    parser.add_argument(
        "--correction-gp-eta",
        type=float,
        default=CORRECTION_GP_ETA,
        help="f1(z) GP amplitude eta",
    )
    parser.add_argument(
        "--obs-sigma-scale",
        type=float,
        default=OBS_SIGMA_SCALE,
        help="Multiplier on MC observation std devs in the GP likelihood",
    )
    parser.add_argument(
        "--gp-kernel",
        choices=("matern52", "gaussian", "harada"),
        default=GP_KERNEL,
        help=(
            "GP kernel on the scaling variable z: matern52 (default) or "
            "gaussian/harada (squared-exponential, Harada PRE 84)"
        ),
    )
    parser.add_argument(
        "--use-log-m",
        action=argparse.BooleanOptionalAction,
        default=USE_LOG_M,
        help="Use batch mean(log|m|) for the m channel when not using correction_m",
    )
    parser.add_argument(
        "--likelihood-backend",
        choices=("jax", "pymc"),
        default=LIKELIHOOD_BACKEND,
        help="Backend for log marginal likelihood evaluation (default: jax)",
    )


def fss_profile_config_from_args(args: argparse.Namespace) -> FssProfileConfig:
    config = FssProfileConfig(
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
        gp_ell_factor=args.gp_ell_factor,
        gp_eta=args.gp_eta,
        correction_gp_ell_factor=args.correction_gp_ell_factor,
        correction_gp_eta=args.correction_gp_eta,
        gp_kernel=normalize_gp_kernel(args.gp_kernel),
        obs_sigma_scale=args.obs_sigma_scale,
        use_log_m=args.use_log_m,
        likelihood_backend=args.likelihood_backend,
    )
    _validate_profile_config(config)
    return config


def _print_profile_config(config: FssProfileConfig) -> None:
    print(
        f"  use_m={config.use_m}, use_m2={config.use_m2}, use_m4={config.use_m4}, "
        f"use_binder={config.use_binder}, use_chi={config.use_chi}, "
        f"correction_m={config.correction_m}, correction_m2={config.correction_m2}, "
        f"correction_m4={config.correction_m4}, correction_binder={config.correction_binder}, "
        f"correction_chi={config.correction_chi}, use_log_m={config.use_log_m}"
    )
    print(
        f"  GP: kernel={config.gp_kernel}, gp_ell_factor={config.gp_ell_factor:g}, "
        f"gp_eta={config.gp_eta:g}, "
        f"correction_gp_ell_factor={config.correction_gp_ell_factor:g}, "
        f"correction_gp_eta={config.correction_gp_eta:g}, "
        f"obs_sigma_scale={config.obs_sigma_scale:g}, "
        f"likelihood_backend={config.likelihood_backend}"
    )


def run_profile_diagnostics(
    dataset: str,
    *,
    data: Path | None = None,
    out: Path | None = None,
    config: FssProfileConfig | None = None,
    T_c: float = TC_EXACT,
    nu: float = NU_EXACT,
    beta: float = BETA_EXACT,
    n_points_1d: int = PROFILE_1D_N_POINTS,
    n_points_2d: int = 55,
    n_coarse: int = PROFILE_JOINT_MLE_N_COARSE,
    n_refine: int = PROFILE_JOINT_MLE_N_REFINE,
    n_starts: int = PROFILE_JOINT_MLE_N_STARTS,
    max_refine_passes: int = PROFILE_JOINT_MLE_MAX_REFINE_PASSES,
    print_summary: bool = True,
) -> list[Path]:
    """Profile log-ML slices and joint figure (same flags as fit/run)."""
    config = _resolve_profile_config(config)
    _validate_profile_config(config)
    _, _, plots_path = resolve_paths(dataset, model="fss")
    out_dir = out or plots_path

    if print_summary:
        print(f"Profile likelihood for dataset {dataset!r}:")
        _print_profile_config(config)

    exact_joint = evaluate_exact_joint_log_marginal_likelihood(
        data=data,
        dataset=dataset,
        config=config,
        T_c_exact=TC_EXACT,
        nu_exact=NU_EXACT,
        beta_exact=BETA_EXACT,
        n_coarse=n_coarse,
        n_refine=n_refine,
        n_starts=n_starts,
        max_refine_passes=max_refine_passes,
    )
    if print_summary:
        print(format_exact_joint_likelihood_summary(exact_joint))

    paths: list[Path] = [
        plot_exponent_joint_profile_likelihoods(
            dataset,
            out_dir,
            T_c=T_c,
            nu=nu,
            beta=beta,
            data=data,
            config=config,
            n_points_1d=n_points_1d,
            n_points_2d=n_points_2d,
            exact_joint=exact_joint,
        ),
        *plot_exponent_profile_likelihoods(
            dataset,
            out_dir,
            T_c=T_c,
            nu=nu,
            beta=beta,
            data=data,
            config=config,
            n_points=n_points_1d,
        ),
    ]
    for path in paths:
        print(f"Wrote {path}")
    return paths


def main(argv: list[str] | None = None) -> list[Path]:
    parser = argparse.ArgumentParser(
        description="Plot profile log marginal likelihood slices for T_c, nu, and beta."
    )
    parser.add_argument("--dataset", default="medium")
    parser.add_argument(
        "--harada-bsa",
        action="store_true",
        help="Use published Harada BSA Binder data (sets --dataset harada_bsa)",
    )
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--T-c", type=float, default=TC_EXACT, dest="T_c")
    parser.add_argument("--nu", type=float, default=NU_EXACT)
    parser.add_argument("--beta", type=float, default=BETA_EXACT)
    parser.add_argument(
        "--n-points",
        type=int,
        default=PROFILE_1D_N_POINTS,
        help="Grid resolution for 1D profile slices",
    )
    parser.add_argument(
        "--only",
        choices=("T_c", "nu", "beta", "joint", "all"),
        default="all",
        help="Which profile slice(s) to plot (default: all, includes joint figure)",
    )
    parser.add_argument(
        "--grid-points-2d",
        type=int,
        default=55,
        help="Grid resolution per axis for joint heatmaps",
    )
    parser.add_argument(
        "--joint-mle-coarse",
        type=int,
        default=PROFILE_JOINT_MLE_N_COARSE,
        help="Coarse grid resolution per axis for global joint MLE search",
    )
    parser.add_argument(
        "--joint-mle-refine",
        type=int,
        default=PROFILE_JOINT_MLE_N_REFINE,
        help="Local refinement grid resolution per axis for global joint MLE search",
    )
    parser.add_argument(
        "--joint-mle-starts",
        type=int,
        default=PROFILE_JOINT_MLE_N_STARTS,
        help="Number of diversified coarse-grid seeds for global joint MLE search",
    )
    parser.add_argument(
        "--joint-mle-refine-passes",
        type=int,
        default=PROFILE_JOINT_MLE_MAX_REFINE_PASSES,
        help="Maximum local 3D refinement passes per seed for global joint MLE search",
    )
    add_fss_profile_arguments(parser)
    args = parser.parse_args(argv)
    if args.harada_bsa:
        args.dataset = "harada_bsa"
        if args.obs_sigma_scale == OBS_SIGMA_SCALE:
            args.obs_sigma_scale = 1.0
        if args.use_m or args.use_m2 or args.use_m4 or args.use_chi:
            parser.error(
                "--harada-bsa supports binder-only profiles; disable other channels "
                "(--no-use-m --no-use-m2 --no-use-m4 --no-use-chi --use-binder)"
            )
    config = fss_profile_config_from_args(args)

    _, _, plots_path = resolve_paths(args.dataset, model="fss")
    out_dir = args.out or plots_path

    if args.only == "all":
        return run_profile_diagnostics(
            args.dataset,
            data=args.data,
            out=out_dir,
            config=config,
            T_c=args.T_c,
            nu=args.nu,
            beta=args.beta,
            n_points_1d=args.n_points,
            n_points_2d=args.grid_points_2d,
            n_coarse=args.joint_mle_coarse,
            n_refine=args.joint_mle_refine,
            n_starts=args.joint_mle_starts,
            max_refine_passes=args.joint_mle_refine_passes,
        )

    print(f"Profile likelihood for dataset {args.dataset!r}:")
    _print_profile_config(config)

    if args.only == "joint":
        exact_joint = evaluate_exact_joint_log_marginal_likelihood(
            data=args.data,
            dataset=args.dataset,
            config=config,
            T_c_exact=TC_EXACT,
            nu_exact=NU_EXACT,
            beta_exact=BETA_EXACT,
            n_coarse=args.joint_mle_coarse,
            n_refine=args.joint_mle_refine,
            n_starts=args.joint_mle_starts,
            max_refine_passes=args.joint_mle_refine_passes,
        )
        print(format_exact_joint_likelihood_summary(exact_joint))
        path = plot_exponent_joint_profile_likelihoods(
            args.dataset,
            out_dir,
            T_c=args.T_c,
            nu=args.nu,
            beta=args.beta,
            data=args.data,
            config=config,
            n_points_1d=args.n_points,
            n_points_2d=args.grid_points_2d,
            exact_joint=exact_joint,
        )
        print(f"Wrote {path}")
        return [path]

    if args.only == "T_c":
        paths = [
            plot_tc_marginal_likelihood(
                args.dataset,
                out_dir,
                nu=args.nu,
                beta=args.beta,
                data=args.data,
                config=config,
                n_points=args.n_points,
            )
        ]
    elif args.only == "nu":
        paths = [
            plot_nu_marginal_likelihood(
                args.dataset,
                out_dir,
                T_c=args.T_c,
                beta=args.beta,
                data=args.data,
                config=config,
                n_points=args.n_points,
            )
        ]
    else:
        paths = [
            plot_beta_marginal_likelihood(
                args.dataset,
                out_dir,
                T_c=args.T_c,
                nu=args.nu,
                data=args.data,
                config=config,
                n_points=args.n_points,
            )
        ]

    for path in paths:
        print(f"Wrote {path}")
    return paths


if __name__ == "__main__":
    main()
