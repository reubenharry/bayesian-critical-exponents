"""Active learning: simulate posterior updates and rank candidate observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from ._deps import az
from .constants import BETA_EXACT, NU_EXACT, TC_EXACT
from .fss_likelihood import extract_likelihood_arrays
from .importance_weights import (
    UncertaintyMetric,
    pareto_smooth_log_weights,
    uncertainty_scalar,
    weighted_uncertainty,
)
from .incremental_likelihood import (
    CandidateObservation,
    fss_delta_log_likelihood_batch,
)
from .observables import (
    df_binder_sigmas,
    df_chi_sigmas,
    df_magnetization_sigmas,
    df_moment_sigmas,
    precision_updated_sigma,
    sigma_from_mc_batch,
)
from .profile_likelihood import FssProfileConfig

CandidateKind = Literal["extra_sweeps", "sample_chunk"]


@dataclass(frozen=True)
class CandidateSpec:
    """One acquisition candidate."""

    L: int
    T: float
    kind: CandidateKind = "extra_sweeps"
    extra_sweeps: int = 0
    # For sample_chunk: chunks already allocated at this point before the action.
    chunks_before: int = 0
    chunk_size: int = 10_000
    n_chunks_max: int = 8


@dataclass(frozen=True)
class PosteriorDraws:
    """Flat posterior samples of exponents."""

    T_c: np.ndarray
    nu: np.ndarray
    beta: np.ndarray

    @property
    def n_samples(self) -> int:
        return int(self.T_c.size)

    def as_matrix(
        self,
        *,
        infer_Tc: bool,
        infer_nu: bool,
        infer_beta: bool,
    ) -> np.ndarray:
        cols: list[np.ndarray] = []
        if infer_Tc:
            cols.append(self.T_c)
        if infer_nu:
            cols.append(self.nu)
        if infer_beta:
            cols.append(self.beta)
        if not cols:
            raise ValueError("At least one exponent must be inferred")
        return np.column_stack(cols)


@dataclass(frozen=True)
class CandidateScore:
    """Acquisition result for one candidate."""

    candidate: CandidateSpec
    delta_uncertainty: float
    uncertainty_before: float
    uncertainty_after: float
    ess: float
    pareto_k: float
    row_index: int | None = None


@dataclass(frozen=True)
class CounterfactualResult:
    """IS preview of adding one excluded observables row to the likelihood."""

    row_index: int
    L: int
    T: float
    tr_cov_before: float
    tr_cov_after: float
    delta_tr_cov: float
    ess: float
    pareto_k: float
    tc_samples: np.ndarray
    nu_samples: np.ndarray
    weights: np.ndarray


def load_posterior_draws(
    idata: az.InferenceData,
    *,
    infer_Tc: bool = True,
    infer_nu: bool = True,
    infer_beta: bool = True,
    max_draws: int | None = None,
    seed: int = 0,
) -> PosteriorDraws:
    """Stack chain/draw arrays into flat exponent vectors."""
    post = idata.posterior
    n = int(np.asarray(post["nu" if infer_nu else "T_c"]).reshape(-1).size)
    if infer_nu:
        n = int(np.asarray(post["nu"]).reshape(-1).size)
    elif infer_Tc:
        n = int(np.asarray(post["T_c"]).reshape(-1).size)
    elif infer_beta and "beta" in post:
        n = int(np.asarray(post["beta"]).reshape(-1).size)
    else:
        n = 1

    T_c = (
        np.asarray(post["T_c"]).reshape(-1)
        if infer_Tc
        else np.full(n, TC_EXACT, dtype=np.float64)
    )
    nu = (
        np.asarray(post["nu"]).reshape(-1)
        if infer_nu
        else np.full(n, NU_EXACT, dtype=np.float64)
    )
    if infer_beta and "beta" in post:
        beta = np.asarray(post["beta"]).reshape(-1)
    else:
        beta = np.full(n, BETA_EXACT, dtype=np.float64)

    if not (T_c.size == nu.size == beta.size):
        raise ValueError(
            f"Posterior draw counts differ: T_c={T_c.size}, nu={nu.size}, beta={beta.size}"
        )

    if max_draws is not None and n > max_draws:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_draws, replace=False)
        T_c = T_c[idx]
        nu = nu[idx]
        beta = beta[idx]
    return PosteriorDraws(T_c=T_c, nu=nu, beta=beta)


def sample_prior_draws(
    n: int,
    *,
    seed: int = 0,
    infer_Tc: bool = True,
    infer_nu: bool = True,
    infer_beta: bool = False,
) -> PosteriorDraws:
    """Draw exponent samples from the uniform prior box used by LAPS."""
    from .model_fss import (
        BETA_PRIOR_LOWER,
        BETA_PRIOR_UPPER,
        NU_PRIOR_LOWER,
        NU_PRIOR_UPPER,
    )
    from .model_scaling_nu import TC_PRIOR_LOWER, TC_PRIOR_UPPER

    if n < 2:
        raise ValueError(f"n must be >= 2 for covariance estimates, got {n}")
    rng = np.random.default_rng(seed)
    T_c = (
        rng.uniform(TC_PRIOR_LOWER, TC_PRIOR_UPPER, size=n)
        if infer_Tc
        else np.full(n, TC_EXACT, dtype=np.float64)
    )
    nu = (
        rng.uniform(NU_PRIOR_LOWER, NU_PRIOR_UPPER, size=n)
        if infer_nu
        else np.full(n, NU_EXACT, dtype=np.float64)
    )
    beta = (
        rng.uniform(BETA_PRIOR_LOWER, BETA_PRIOR_UPPER, size=n)
        if infer_beta
        else np.full(n, BETA_EXACT, dtype=np.float64)
    )
    return PosteriorDraws(T_c=T_c, nu=nu, beta=beta)


def planned_chunk_n_eff(
    n_eff_full: float,
    n_chunks: int,
    *,
    n_chunks_max: int = 8,
) -> float:
    """Score-time n_eff after ``n_chunks`` allocations (``n_eff_full / n_chunks_max`` each)."""
    if n_chunks_max < 1:
        raise ValueError(f"n_chunks_max must be >= 1, got {n_chunks_max}")
    if n_chunks < 0:
        raise ValueError(f"n_chunks must be >= 0, got {n_chunks}")
    return float(n_chunks) * float(n_eff_full) / float(n_chunks_max)


def _planned_n_eff_full_from_planner(
    planner: object,
    L: float,
    T: float,
    *,
    chunk_size: int,
    n_chunks_max: int,
) -> float:
    """Predicted full-budget Binder n_eff from an online NeffRatePlanner."""
    from .neff_scaling import NeffRatePlanner

    if not isinstance(planner, NeffRatePlanner):
        raise TypeError(f"neff_planner must be NeffRatePlanner, got {type(planner)!r}")
    n_sweeps_full = float(chunk_size) * float(n_chunks_max)
    return float(
        planner.predict_n_eff_full(L, T, n_sweeps_full=n_sweeps_full)
    )


def _find_row_index(df: pd.DataFrame, L: int, T: float) -> int | None:
    L_arr = df["L"].to_numpy(dtype=np.float64)
    T_arr = df["T"].to_numpy(dtype=np.float64)
    mask = (L_arr == float(L)) & np.isclose(T_arr, T, rtol=0.0, atol=1e-9)
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else None


def _channel_values_for_row(
    row: pd.Series,
    config: FssProfileConfig,
    *,
    beta: float,
    nu: float,
    obs_sigma_scale: float,
    extra_sweeps: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Build plug-in y and sigma for each enabled channel at one grid point."""
    y: dict[str, float] = {}
    sigma: dict[str, float] = {}
    scale = float(obs_sigma_scale)

    if config.use_m:
        m = float(row["magnetization"])
        std_m = float(row["magnetization_std"])
        n_eff = float(row["n_eff"])
        sig_m = float(
            precision_updated_sigma(
                sigma_from_mc_batch(np.array([std_m]), np.array([n_eff]))[0] * scale,
                n_eff,
                extra_sweeps,
            )[0]
        )
        if config.use_log_m and not config.correction_m:
            log_m = float(row["log_magnetization"])
            std_log = float(row["log_magnetization_std"])
            n_eff_log = float(row["n_eff_log"])
            y["log_m"] = log_m + (beta / nu) * np.log(float(row["L"]))
            sigma["log_m"] = float(
                precision_updated_sigma(
                    sigma_from_mc_batch(np.array([std_log]), np.array([n_eff_log]))[0]
                    * scale,
                    n_eff_log,
                    extra_sweeps,
                )[0]
            )
        else:
            phi_m = m * float(row["L"]) ** (beta / nu)
            y["m"] = phi_m
            sigma["m"] = sig_m * float(row["L"]) ** (beta / nu)

    if config.use_m2:
        m2, sig = df_moment_sigmas(
            pd.DataFrame([row]), value_col="m2", std_col="m2_std", n_eff_col="n_eff_m2"
        )
        n_eff = float(row["n_eff_m2"])
        sig2 = float(
            precision_updated_sigma(sig * scale, np.array([n_eff]), extra_sweeps)[0]
        )
        y["m2"] = float(m2[0] * float(row["L"]) ** (2.0 * beta / nu))
        sigma["m2"] = float(sig2 * float(row["L"]) ** (2.0 * beta / nu))

    if config.use_m4:
        m4, sig = df_moment_sigmas(
            pd.DataFrame([row]), value_col="m4", std_col="m4_std", n_eff_col="n_eff_m4"
        )
        n_eff = float(row["n_eff_m4"])
        sig2 = float(
            precision_updated_sigma(sig * scale, np.array([n_eff]), extra_sweeps)[0]
        )
        y["m4"] = float(m4[0] * float(row["L"]) ** (4.0 * beta / nu))
        sigma["m4"] = float(sig2 * float(row["L"]) ** (4.0 * beta / nu))

    if config.use_binder:
        binder, sig = df_binder_sigmas(pd.DataFrame([row]))
        n_eff = float(row["n_eff_binder"])
        sig2 = float(
            precision_updated_sigma(sig * scale, np.array([n_eff]), extra_sweeps)[0]
        )
        y["binder"] = float(binder[0])
        sigma["binder"] = sig2

    if config.use_chi:
        chi, sig = df_chi_sigmas(pd.DataFrame([row]))
        n_eff = float(row["n_eff_susceptibility"])
        gamma = 2.0 * nu - 2.0 * beta
        sig2 = float(
            precision_updated_sigma(sig * scale, np.array([n_eff]), extra_sweeps)[0]
        )
        y["chi"] = float(chi[0] * float(row["L"]) ** (-gamma / nu))
        sigma["chi"] = float(sig2 * float(row["L"]) ** (-gamma / nu))

    return y, sigma


def _channel_sigmas_for_row(
    row: pd.Series,
    config: FssProfileConfig,
    *,
    beta: float,
    nu: float,
    obs_sigma_scale: float,
    extra_sweeps: int,
) -> dict[str, float]:
    """Planned MC sigmas for acquisition at one grid point (no plug-in y)."""
    _, sigma = _channel_values_for_row(
        row,
        config,
        beta=beta,
        nu=nu,
        obs_sigma_scale=obs_sigma_scale,
        extra_sweeps=extra_sweeps,
    )
    return sigma


def _acquisition_candidate_from_row(
    row: pd.Series,
    config: FssProfileConfig,
    *,
    extra_sweeps: int = 0,
    beta_ref: float = BETA_EXACT,
    nu_ref: float = NU_EXACT,
) -> CandidateObservation:
    """Candidate location and planned noise only; y comes from the GP predictive mean."""
    sigma = _channel_sigmas_for_row(
        row,
        config,
        beta=beta_ref,
        nu=nu_ref,
        obs_sigma_scale=config.obs_sigma_scale,
        extra_sweeps=extra_sweeps,
    )
    return CandidateObservation(
        L=float(row["L"]),
        T=float(row["T"]),
        y_by_channel={},
        sigma_by_channel=sigma,
    )


def build_candidate_observation(
    observables: pd.DataFrame,
    config: FssProfileConfig,
    candidate: CandidateSpec,
    *,
    beta_ref: float = BETA_EXACT,
    nu_ref: float = NU_EXACT,
) -> tuple[CandidateObservation, int | None]:
    """Construct plug-in candidate observation for extra-sweeps acquisition."""
    if candidate.kind != "extra_sweeps":
        raise ValueError(f"Unsupported candidate kind: {candidate.kind!r}")
    row_index = _find_row_index(observables, candidate.L, candidate.T)
    if row_index is None:
        raise ValueError(
            f"No observables row for L={candidate.L}, T={candidate.T:.6g}"
        )
    row = observables.iloc[row_index]
    y, sigma = _channel_values_for_row(
        row,
        config,
        beta=beta_ref,
        nu=nu_ref,
        obs_sigma_scale=config.obs_sigma_scale,
        extra_sweeps=candidate.extra_sweeps,
    )
    return (
        CandidateObservation(
            L=float(candidate.L),
            T=float(candidate.T),
            y_by_channel=y,
            sigma_by_channel=sigma,
        ),
        row_index,
    )


def simulate_posterior_update(
    exponent_samples: np.ndarray,
    log_weights: np.ndarray,
    *,
    metric: UncertaintyMetric = "trace_cov",
) -> tuple[float, float, float]:
    """Return (uncertainty, ess, pareto_k) after PSIS reweighting."""
    iw = pareto_smooth_log_weights(log_weights)
    uncertainty = weighted_uncertainty(exponent_samples, iw.weights, metric=metric)
    return uncertainty, iw.ess, iw.pareto_k


def counterfactual_include_point(
    included_observables: pd.DataFrame,
    config: FssProfileConfig,
    draws: PosteriorDraws,
    row: pd.Series,
    *,
    row_index: int,
    infer_Tc: bool = True,
    infer_nu: bool = True,
    infer_beta: bool = False,
) -> CounterfactualResult:
    """Preview posterior update from adding one row to the GP likelihood.

    Acquisition uses the GP predictive mean at each exponent draw for ``y``,
    with planned MC noise ``sigma`` from the candidate row.
    """
    arrays = extract_likelihood_arrays(included_observables, config)
    cand_obs = _acquisition_candidate_from_row(row, config)
    exponent_samples = draws.as_matrix(
        infer_Tc=infer_Tc,
        infer_nu=infer_nu,
        infer_beta=infer_beta,
    )
    u_before = uncertainty_scalar(
        np.cov(exponent_samples, rowvar=False),
        metric="trace_cov",
    )
    log_w = fss_delta_log_likelihood_batch(
        arrays,
        config,
        cand_obs,
        T_c=draws.T_c,
        nu=draws.nu,
        beta=draws.beta,
        y_mode="predictive_mean",
    )
    iw = pareto_smooth_log_weights(log_w)
    u_after = weighted_uncertainty(
        exponent_samples,
        iw.weights,
        metric="trace_cov",
    )
    return CounterfactualResult(
        row_index=int(row_index),
        L=int(row["L"]),
        T=float(row["T"]),
        tr_cov_before=float(u_before),
        tr_cov_after=float(u_after),
        delta_tr_cov=float(u_before - u_after),
        ess=float(iw.ess),
        pareto_k=float(iw.pareto_k),
        tc_samples=draws.T_c.copy(),
        nu_samples=draws.nu.copy(),
        weights=iw.weights.copy(),
    )


def score_candidate(
    observables: pd.DataFrame,
    config: FssProfileConfig,
    draws: PosteriorDraws,
    candidate: CandidateSpec,
    *,
    infer_Tc: bool = True,
    infer_nu: bool = True,
    infer_beta: bool = True,
    metric: UncertaintyMetric = "trace_cov",
    n_y_samples: int = 8,
    seed: int = 0,
    beta_ref: float | None = None,
    nu_ref: float | None = None,
) -> CandidateScore:
    """Expected uncertainty reduction for one candidate."""
    arrays = extract_likelihood_arrays(observables, config)
    _, row_index = build_candidate_observation(
        observables,
        config,
        candidate,
        beta_ref=BETA_EXACT if beta_ref is None else beta_ref,
        nu_ref=NU_EXACT if nu_ref is None else nu_ref,
    )
    row = observables.iloc[int(row_index)]
    cand_obs = _acquisition_candidate_from_row(
        row,
        config,
        extra_sweeps=candidate.extra_sweeps,
        beta_ref=BETA_EXACT if beta_ref is None else beta_ref,
        nu_ref=NU_EXACT if nu_ref is None else nu_ref,
    )
    exponent_samples = draws.as_matrix(
        infer_Tc=infer_Tc,
        infer_nu=infer_nu,
        infer_beta=infer_beta,
    )
    u_before = uncertainty_scalar(np.cov(exponent_samples, rowvar=False), metric=metric)

    log_w = fss_delta_log_likelihood_batch(
        arrays,
        config,
        cand_obs,
        T_c=draws.T_c,
        nu=draws.nu,
        beta=draws.beta,
        y_mode="predictive_mean",
    )
    u_after, ess, k = simulate_posterior_update(
        exponent_samples,
        log_w,
        metric=metric,
    )

    return CandidateScore(
        candidate=candidate,
        delta_uncertainty=float(u_before - u_after),
        uncertainty_before=float(u_before),
        uncertainty_after=float(u_after),
        ess=float(ess),
        pareto_k=float(k),
        row_index=row_index,
    )


def enumerate_extra_sweep_candidates(
    observables: pd.DataFrame,
    extra_sweeps_grid: tuple[int, ...],
) -> list[CandidateSpec]:
    """All (L, T) rows crossed with extra-sweep counts."""
    candidates: list[CandidateSpec] = []
    for _, row in observables.iterrows():
        L = int(row["L"])
        T = float(row["T"])
        for extra in extra_sweeps_grid:
            if extra <= 0:
                continue
            candidates.append(
                CandidateSpec(
                    L=L,
                    T=T,
                    kind="extra_sweeps",
                    extra_sweeps=int(extra),
                )
            )
    return candidates


def rank_candidates(
    observables: pd.DataFrame,
    config: FssProfileConfig,
    draws: PosteriorDraws,
    candidates: list[CandidateSpec],
    *,
    infer_Tc: bool = True,
    infer_nu: bool = True,
    infer_beta: bool = True,
    metric: UncertaintyMetric = "trace_cov",
    n_y_samples: int = 8,
    seed: int = 0,
) -> pd.DataFrame:
    """Score and rank candidates by expected uncertainty reduction."""
    rows: list[dict[str, object]] = []
    for j, candidate in enumerate(candidates):
        score = score_candidate(
            observables,
            config,
            draws,
            candidate,
            infer_Tc=infer_Tc,
            infer_nu=infer_nu,
            infer_beta=infer_beta,
            metric=metric,
            n_y_samples=n_y_samples,
            seed=seed + j,
        )
        rows.append(
            {
                "L": candidate.L,
                "T": candidate.T,
                "extra_sweeps": candidate.extra_sweeps,
                "delta_uncertainty": score.delta_uncertainty,
                "uncertainty_before": score.uncertainty_before,
                "uncertainty_after": score.uncertainty_after,
                "ess": score.ess,
                "pareto_k": score.pareto_k,
                "row_index": score.row_index,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("delta_uncertainty", ascending=False).reset_index(drop=True)


def _score_row_with_planned_neff(
    ref_row: pd.Series,
    *,
    n_eff_full_planned: float,
    n_chunks: int,
    n_chunks_max: int,
    included_observables: pd.DataFrame | None = None,
) -> pd.Series:
    """Binder score-time row: planned ESS; std from current if available else ref."""
    row = ref_row.copy()
    row["n_eff_binder"] = planned_chunk_n_eff(
        n_eff_full_planned, n_chunks, n_chunks_max=n_chunks_max
    )
    if included_observables is not None and not included_observables.empty:
        idx = _find_row_index(
            included_observables, int(ref_row["L"]), float(ref_row["T"])
        )
        if idx is not None:
            cur = included_observables.iloc[idx]
            if "binder_cumulant_std" in cur.index and np.isfinite(
                float(cur["binder_cumulant_std"])
            ):
                row["binder_cumulant_std"] = float(cur["binder_cumulant_std"])
    return row


def score_sample_chunk(
    included_observables: pd.DataFrame,
    config: FssProfileConfig,
    draws: PosteriorDraws,
    ref_row: pd.Series,
    *,
    row_index: int,
    chunks_before: int,
    n_chunks_max: int = 8,
    chunk_size: int = 10_000,
    neff_planner: object | None = None,
    infer_Tc: bool = True,
    infer_nu: bool = True,
    infer_beta: bool = False,
) -> CandidateScore:
    """Score allocating one more sample chunk at ``ref_row``'s (L, T).

    Score-time Binder ESS comes from ``neff_planner`` (online dynamical GP or
    cold-start rate), never from oracle full-chain ``n_eff_binder``. Acquisition
    ``y`` is the FSS GP predictive mean.
    """
    if chunks_before < 0 or chunks_before >= n_chunks_max:
        raise ValueError(
            f"chunks_before must be in 0..{n_chunks_max - 1}, got {chunks_before}"
        )
    if neff_planner is None:
        raise ValueError("neff_planner is required for sample-chunk scoring")
    candidate = CandidateSpec(
        L=int(ref_row["L"]),
        T=float(ref_row["T"]),
        kind="sample_chunk",
        chunks_before=chunks_before,
        chunk_size=chunk_size,
        n_chunks_max=n_chunks_max,
        extra_sweeps=0,
    )
    n_eff_full = _planned_n_eff_full_from_planner(
        neff_planner,
        float(ref_row["L"]),
        float(ref_row["T"]),
        chunk_size=chunk_size,
        n_chunks_max=n_chunks_max,
    )
    n_eff_one = planned_chunk_n_eff(n_eff_full, 1, n_chunks_max=n_chunks_max)

    if chunks_before == 0:
        # Virgin point: counterfactual include at score-time σ for one chunk.
        score_row = _score_row_with_planned_neff(
            ref_row,
            n_eff_full_planned=n_eff_full,
            n_chunks=1,
            n_chunks_max=n_chunks_max,
            included_observables=included_observables,
        )
        result = counterfactual_include_point(
            included_observables,
            config,
            draws,
            score_row,
            row_index=row_index,
            infer_Tc=infer_Tc,
            infer_nu=infer_nu,
            infer_beta=infer_beta,
        )
        return CandidateScore(
            candidate=candidate,
            delta_uncertainty=float(result.delta_tr_cov),
            uncertainty_before=float(result.tr_cov_before),
            uncertainty_after=float(result.tr_cov_after),
            ess=float(result.ess),
            pareto_k=float(result.pareto_k),
            row_index=row_index,
        )

    # Revisit: precision-update score using planned n_eff schedule, not real prefix ESS.
    score_row = _score_row_with_planned_neff(
        ref_row,
        n_eff_full_planned=n_eff_full,
        n_chunks=chunks_before,
        n_chunks_max=n_chunks_max,
        included_observables=included_observables,
    )
    cand_obs = _acquisition_candidate_from_row(
        score_row,
        config,
        extra_sweeps=int(round(n_eff_one)),
    )
    arrays = extract_likelihood_arrays(included_observables, config)
    exponent_samples = draws.as_matrix(
        infer_Tc=infer_Tc,
        infer_nu=infer_nu,
        infer_beta=infer_beta,
    )
    u_before = uncertainty_scalar(
        np.cov(exponent_samples, rowvar=False),
        metric="trace_cov",
    )
    log_w = fss_delta_log_likelihood_batch(
        arrays,
        config,
        cand_obs,
        T_c=draws.T_c,
        nu=draws.nu,
        beta=draws.beta,
        y_mode="predictive_mean",
    )
    u_after, ess, k = simulate_posterior_update(
        exponent_samples,
        log_w,
        metric="trace_cov",
    )
    return CandidateScore(
        candidate=candidate,
        delta_uncertainty=float(u_before - u_after),
        uncertainty_before=float(u_before),
        uncertainty_after=float(u_after),
        ess=float(ess),
        pareto_k=float(k),
        row_index=row_index,
    )


def min_chunk_count(n_draws: np.ndarray, *, chunk_size: int) -> int:
    """Minimum number of completed chunks across the grid."""
    n_draws_arr = np.asarray(n_draws, dtype=np.int64).ravel()
    if n_draws_arr.size == 0:
        return 0
    return int(np.min(n_draws_arr // int(chunk_size)))


def enumerate_sample_chunk_candidates(
    n_draws: np.ndarray,
    *,
    chunk_size: int = 10_000,
    n_chunks_max: int = 8,
    L: np.ndarray | None = None,
    T: np.ndarray | None = None,
    require_uniform_coverage: bool = True,
) -> list[CandidateSpec]:
    """Candidates that still have remaining sample budget.

    When ``require_uniform_coverage`` is True (default), only points at the
    current minimum chunk count are eligible — no point may receive chunk
    ``k+1`` until every grid point has at least ``k`` chunks (e.g. no 20k
    until every point has 10k).
    """
    n_draws_arr = np.asarray(n_draws, dtype=np.int64).ravel()
    min_chunks = (
        min_chunk_count(n_draws_arr, chunk_size=chunk_size)
        if require_uniform_coverage
        else 0
    )
    candidates: list[CandidateSpec] = []
    for i, n_i in enumerate(n_draws_arr):
        chunks_before = int(n_i) // int(chunk_size)
        if chunks_before >= n_chunks_max:
            continue
        if require_uniform_coverage and chunks_before > min_chunks:
            continue
        Li = int(L[i]) if L is not None else 0
        Ti = float(T[i]) if T is not None else 0.0
        candidates.append(
            CandidateSpec(
                L=Li,
                T=Ti,
                kind="sample_chunk",
                chunks_before=chunks_before,
                chunk_size=chunk_size,
                n_chunks_max=n_chunks_max,
            )
        )
    return candidates


def rank_sample_chunk_candidates(
    included_observables: pd.DataFrame,
    full_ref: pd.DataFrame,
    n_draws: np.ndarray,
    config: FssProfileConfig,
    draws: PosteriorDraws,
    *,
    neff_planner: object,
    chunk_size: int = 10_000,
    n_chunks_max: int = 8,
    L_ref: float = 64.0,
    require_uniform_coverage: bool = True,
    allowed_L: int | None = None,
    infer_Tc: bool = True,
    infer_nu: bool = True,
    infer_beta: bool = False,
) -> pd.DataFrame:
    """Score feasible sample-chunk actions by Δ tr(cov) / (L/L_ref)^2.

    With ``require_uniform_coverage=True`` (default), only points at the
    current minimum chunk depth are ranked (breadth-first over the grid).

    If ``allowed_L`` is set, only candidates with that lattice size are scored.
    """
    from .neff_scaling import chunk_compute_cost

    n_draws_arr = np.asarray(n_draws, dtype=np.int64).ravel()
    if n_draws_arr.size != len(full_ref):
        raise ValueError("n_draws length must match full_ref rows")
    min_chunks = (
        min_chunk_count(n_draws_arr, chunk_size=chunk_size)
        if require_uniform_coverage
        else 0
    )
    rows: list[dict[str, object]] = []
    for i in range(len(full_ref)):
        chunks_before = int(n_draws_arr[i]) // int(chunk_size)
        if chunks_before >= n_chunks_max:
            continue
        if require_uniform_coverage and chunks_before > min_chunks:
            continue
        ref_row = full_ref.iloc[i]
        L_i = int(ref_row["L"])
        if allowed_L is not None and L_i != int(allowed_L):
            continue
        T_i = float(ref_row["T"])
        n_eff_full = _planned_n_eff_full_from_planner(
            neff_planner,
            float(L_i),
            T_i,
            chunk_size=chunk_size,
            n_chunks_max=n_chunks_max,
        )
        score = score_sample_chunk(
            included_observables,
            config,
            draws,
            ref_row,
            row_index=i,
            chunks_before=chunks_before,
            n_chunks_max=n_chunks_max,
            chunk_size=chunk_size,
            neff_planner=neff_planner,
            infer_Tc=infer_Tc,
            infer_nu=infer_nu,
            infer_beta=infer_beta,
        )
        cost = chunk_compute_cost(L_i, L_ref=L_ref)
        delta = float(score.delta_uncertainty)
        rows.append(
            {
                "row_index": i,
                "L": L_i,
                "T": T_i,
                "chunks_before": chunks_before,
                "n_draws_before": int(n_draws_arr[i]),
                "delta_tr_cov": delta,
                "cost": float(cost),
                "delta_tr_cov_per_cost": float(delta / cost),
                "tr_cov_before": float(score.uncertainty_before),
                "tr_cov_after": float(score.uncertainty_after),
                "ess": float(score.ess),
                "pareto_k": float(score.pareto_k),
                "predicted_n_eff_full": float(n_eff_full),
                "planned_n_eff_after": planned_chunk_n_eff(
                    n_eff_full,
                    chunks_before + 1,
                    n_chunks_max=n_chunks_max,
                ),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        "delta_tr_cov_per_cost", ascending=False
    ).reset_index(drop=True)
