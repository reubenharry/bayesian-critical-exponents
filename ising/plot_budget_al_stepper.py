#!/usr/bin/env python3
"""Interactive HTML stepper for sample-budget active learning runs.

Each frame shows:
- Top-left: joint posterior (or prior) heatmap from ``iter_XX/posterior.nc``
- Top-middle: Binder grid with marker size ∝ draws spent so far; optional softmax
  P(choose) coloring (see ``_COLOR_CANDIDATES_BY_SOFTMAX``); gold star = chosen
  next chunk.
- Top-right: online Binder integrated autocorrelation time
  ``τ_int ≈ N/(2 n_eff)`` vs reduced temperature ``t`` (all L overlaid), with
  GP mean curves when the dynamical fit is available.
- Bottom: FSS Binder GP predictive mean ``Û(T)`` ± 1σ, plotted vs raw ``T``.
  The band mixes the GP posterior over posterior/prior draws of ``(T_c, ν)``
  (law of total variance). Dashed curves: all-data GP at exact exponents.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .active_learning import sample_prior_draws
from .constants import NU_EXACT, TC_EXACT
from .datasets import ISING_DIR, observables_path, samples_path
from .fss_likelihood import collapse_z, extract_likelihood_arrays, fss_gp_scales
from .gp_utils import gp_posterior_predictive
from .greedy_active_learning import build_al_config
from .importance_weights import uncertainty_scalar
from .neff_scaling import (
    DEFAULT_ONLINE_MIN_POINTS,
    NeffScalingFit,
    neff_column,
    neff_fit_at_fixed_params,
    parse_neff_fit_summary,
    predict_neff,
)
from .plot_greedy_al_evolution import _shared_xlim
from .plot_greedy_al_stepper import (
    _choice_prob_by_index,
    _overlay_traces,
    _prob_colormap_limits,
    show_stepper_in_notebook,
)
from .scaling_function import GP_JITTER

DEFAULT_RUN_DIR = ISING_DIR / "plots" / "budget_al_harada_square"

_BINDER_STYLE: dict[int, dict[str, object]] = {
    64: {"color": "#1f77b4", "symbol": "circle", "size_base": 8, "label": "L=64 (budget)"},
    128: {"color": "#2ca02c", "symbol": "circle", "size_base": 7, "label": "L=128 (budget)"},
    256: {"color": "#d62728", "symbol": "circle", "size_base": 8, "label": "L=256 (budget)"},
}

_NEFF_L_STYLE: dict[int, dict[str, object]] = {
    64: {"color": "#1f77b4", "symbol": "circle", "label": "L=64"},
    128: {"color": "#2ca02c", "symbol": "x", "label": "L=128"},
    256: {"color": "#d62728", "symbol": "cross", "label": "L=256"},
}

# Softmax P(choose) Turbo colormap + colorbar on binder candidates.
# Kept implemented below; flip to True to restore.
_COLOR_CANDIDATES_BY_SOFTMAX = False
_CANDIDATE_FLAT_COLOR = "#9aa0a6"

_NEFF_CURVE_POINTS = 80
_BINDER_GP_CURVE_POINTS = 100
# Posterior / prior draws of (T_c, ν) mixed into the Binder GP predictive band.
_BINDER_GP_PARAM_DRAWS = 32


def _tau_int_from_neff(n_eff: np.ndarray, n_sweeps: np.ndarray) -> np.ndarray:
    """Integrated autocorrelation time: n_eff ≈ N / (2 τ_int)."""
    n_eff = np.asarray(n_eff, dtype=np.float64)
    n_sweeps = np.asarray(n_sweeps, dtype=np.float64)
    return n_sweeps / (2.0 * np.maximum(n_eff, 1.0))


@dataclass(frozen=True)
class NeffPanelState:
    """Per-iteration online mixing diagnostic for the stepper's third panel.

    Stores Binder ``τ_int ≈ N/(2 n_eff)`` (budget-invariant) vs reduced ``t``.
    """

    kind: str  # "gp" | "cold_start" | "none"
    L_values: tuple[int, ...]
    L: np.ndarray
    t: np.ndarray
    tau_int: np.ndarray
    curve_t: dict[int, np.ndarray] = field(default_factory=dict)
    curve_tau: dict[int, np.ndarray] = field(default_factory=dict)
    z_dyn_mean: float | None = None
    n_train: int = 0

    @staticmethod
    def empty(L_values: tuple[int, ...]) -> NeffPanelState:
        return NeffPanelState(
            kind="none",
            L_values=L_values,
            L=np.array([], dtype=np.float64),
            t=np.array([], dtype=np.float64),
            tau_int=np.array([], dtype=np.float64),
        )


@dataclass(frozen=True)
class BinderGpPanelState:
    """Per-iteration FSS Binder GP predictive curves ``Û(t)`` at mean ``(T_c, ν)``.

    The GP is trained in collapsed ``z = t L^{1/ν}`` space (universal without an
    ``L^{-ω}`` correction), but curves and points are plotted against the raw
    temperature ``T`` so markers stay put as belief in ``T_c`` shifts.
    Training uses currently budgeted grid points. ``curve_U_std`` is the mixture
    predictive std over posterior/prior draws of ``(T_c, ν)``:
    ``√(E[σ²|θ] + Var[μ|θ])``.
    """

    kind: str  # "gp" | "none"
    L_values: tuple[int, ...]
    L: np.ndarray
    T: np.ndarray
    U: np.ndarray
    curve_T: dict[int, np.ndarray] = field(default_factory=dict)
    curve_U: dict[int, np.ndarray] = field(default_factory=dict)
    curve_U_std: dict[int, np.ndarray] = field(default_factory=dict)
    # Reference: GP on all grid data at exact (T_c, ν); dashed overlay.
    exact_curve_T: dict[int, np.ndarray] = field(default_factory=dict)
    exact_curve_U: dict[int, np.ndarray] = field(default_factory=dict)
    n_train: int = 0

    @staticmethod
    def empty(
        L_values: tuple[int, ...],
        *,
        exact_curve_T: dict[int, np.ndarray] | None = None,
        exact_curve_U: dict[int, np.ndarray] | None = None,
    ) -> BinderGpPanelState:
        return BinderGpPanelState(
            kind="none",
            L_values=L_values,
            L=np.array([], dtype=np.float64),
            T=np.array([], dtype=np.float64),
            U=np.array([], dtype=np.float64),
            exact_curve_T=exact_curve_T or {},
            exact_curve_U=exact_curve_U or {},
        )


@dataclass(frozen=True)
class BudgetStepperFrame:
    iteration: int
    n_points: int
    total_draws: int
    n_draws: np.ndarray
    next_index: int
    next_delta_tr_cov: float
    chunks_before: int
    tr_cov: float
    z: np.ndarray
    tc_centers: np.ndarray
    nu_centers: np.ndarray
    mean_tc: float
    mean_nu: float
    tc_std: float
    ranking: pd.DataFrame
    has_posterior: bool
    neff_panel: NeffPanelState
    binder_gp_panel: BinderGpPanelState


def _load_run_meta(run_dir: Path) -> dict:
    meta_path = run_dir / "run_meta.json"
    if not meta_path.is_file():
        return {}
    return json.loads(meta_path.read_text())


def _load_full_grid(run_dir: Path) -> pd.DataFrame:
    """Full (L,T,U4) grid for the Binder panel.

    Prefer ``observables_harada_square.csv``; fall back to any ``iter_*/budget.csv``
    merged with that table's Binder values if present.
    """
    csv_path = observables_path("harada_square")
    if csv_path.is_file():
        df = pd.read_csv(csv_path)
        required = {"L", "T", "binder_cumulant"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")
        return df.reset_index(drop=True)

    # Reconstruct L,T from first available budget file.
    for iter_dir in sorted(run_dir.glob("iter_*")):
        budget_path = iter_dir / "budget.csv"
        if budget_path.is_file():
            budget = pd.read_csv(budget_path)
            if {"L", "T"}.issubset(budget.columns):
                out = budget[["L", "T"]].copy()
                out["binder_cumulant"] = 0.5
                return out.reset_index(drop=True)
    raise FileNotFoundError(
        f"Need {csv_path} or iter_*/budget.csv under {run_dir} for the Binder grid"
    )


def _reduced_t(T: np.ndarray, *, T_c: float = TC_EXACT) -> np.ndarray:
    return (np.asarray(T, dtype=np.float64) - float(T_c)) / float(T_c)


def _load_neff_train_df(
    run_dir: Path,
    iteration: int,
    n_draws: np.ndarray,
    *,
    stored_cache: dict[str, object],
    allow_materialize: bool = False,
) -> pd.DataFrame:
    """Load prefix n_eff rows for one iteration.

    Prefer ``iter_XX/neff_train.csv``. Optionally recompute from raw samples when
    ``allow_materialize`` is True (expensive; used by backfill).
    """
    _ = n_draws
    train_path = run_dir / f"iter_{iteration:02d}" / "neff_train.csv"
    col = neff_column("binder")
    if train_path.is_file():
        df = pd.read_csv(train_path)
        if col in df.columns and not df.empty:
            return df.reset_index(drop=True)
        return pd.DataFrame(columns=["L", "T", col, "n_sweeps"])

    if not allow_materialize:
        return pd.DataFrame(columns=["L", "T", col, "n_sweeps"])

    sample_path = samples_path("harada_square")
    if not sample_path.is_file():
        return pd.DataFrame(columns=["L", "T", col, "n_sweeps"])
    if "stored" not in stored_cache:
        from .samples import load_samples

        stored_cache["stored"] = load_samples(sample_path)
    from .budget_active_learning import materialize_budget

    current = materialize_budget(stored_cache["stored"], n_draws)
    if current.empty or col not in current.columns:
        return pd.DataFrame(columns=["L", "T", col, "n_sweeps"])
    cols = ["L", "T", col]
    if "n_sweeps" in current.columns:
        cols.append("n_sweeps")
    return current.loc[:, cols].reset_index(drop=True)


def backfill_neff_train_csvs(
    run_dir: Path,
    *,
    chunk_size: int | None = None,
    force: bool = False,
) -> int:
    """Write missing ``iter_XX/neff_train.csv`` from prefix materialization.

    Computes each ``(grid point, chunk count)`` prefix once, then assembles
    per-iteration tables (much cheaper than rematerializing every iter).

    Returns the number of files written. Skips iters that already have a csv
    unless ``force`` is True.
    """
    from .neff_scaling import write_neff_train_csv
    from .samples import load_samples, observables_from_budget

    run_dir = Path(run_dir)
    meta = _load_run_meta(run_dir)
    if chunk_size is None:
        chunk_size = int(meta.get("chunk_size", 10_000))
    n_chunks_max = int(meta.get("n_chunks_max", 8))
    sample_path = samples_path("harada_square")
    if not sample_path.is_file():
        raise FileNotFoundError(f"Raw samples not found: {sample_path}")
    stored = load_samples(sample_path)
    n_points = int(stored.n_points)
    summary = load_budget_summary(run_dir, chunk_size=chunk_size, n_points=n_points)

    # Discover which (point, chunks) pairs are needed.
    needed: set[tuple[int, int]] = set()
    n_draws_by_iter: dict[int, np.ndarray] = {}
    for row in summary.itertuples(index=False):
        iteration = int(row.iteration)
        out_path = run_dir / f"iter_{iteration:02d}" / "neff_train.csv"
        if out_path.is_file() and not force:
            continue
        n_draws = n_draws_before_iteration(
            summary, n_points, iteration, chunk_size=chunk_size
        )
        budget_path = run_dir / f"iter_{iteration:02d}" / "n_draws.npy"
        if budget_path.is_file():
            after = np.asarray(np.load(budget_path), dtype=np.int64).ravel()
            if after.size == n_points:
                before = after.copy()
                before[int(row.added_row_index)] = max(
                    0, int(before[int(row.added_row_index)]) - chunk_size
                )
                n_draws = before
        n_draws_by_iter[iteration] = n_draws
        for i, n_i in enumerate(n_draws.tolist()):
            if n_i > 0:
                needed.add((i, int(n_i)))

    if not n_draws_by_iter:
        return 0

    # Materialize unique budgets: one vector with a single point active.
    print(
        f"Backfilling n_eff train CSVs: {len(needed)} unique prefixes, "
        f"{len(n_draws_by_iter)} iterations",
        flush=True,
    )
    row_cache: dict[tuple[int, int], dict[str, float | int]] = {}
    col = neff_column("binder")
    for idx, (point_i, n_i) in enumerate(sorted(needed)):
        budget = np.zeros(n_points, dtype=np.int64)
        budget[point_i] = n_i
        df = observables_from_budget(stored, budget)
        if df.empty:
            continue
        row = df.iloc[0]
        row_cache[(point_i, n_i)] = {
            "L": int(row["L"]),
            "T": float(row["T"]),
            col: float(row[col]),
            "n_sweeps": int(row["n_sweeps"]),
        }
        if (idx + 1) % 25 == 0 or idx + 1 == len(needed):
            print(f"  prefixes {idx + 1}/{len(needed)}", flush=True)
    _ = n_chunks_max

    written = 0
    for iteration, n_draws in n_draws_by_iter.items():
        rows = []
        for i, n_i in enumerate(n_draws.tolist()):
            if n_i <= 0:
                continue
            cached = row_cache.get((i, int(n_i)))
            if cached is not None:
                rows.append(cached)
        out_path = run_dir / f"iter_{iteration:02d}" / "neff_train.csv"
        write_neff_train_csv(pd.DataFrame(rows), out_path, channel="binder")
        written += 1
    print(f"Wrote {written} neff_train.csv files under {run_dir}", flush=True)
    return written


def _fit_from_summary_and_train(
    train: pd.DataFrame,
    summary: dict[str, str],
) -> NeffScalingFit | None:
    """Replay online GP at stored posterior-mean params (no NUTS)."""
    if not summary or summary.get("kind") == "cold_start":
        return None
    if "z_dyn_mean" not in summary or "noise_sigma_mean" not in summary:
        return None
    col = neff_column("binder")
    if train.empty or col not in train.columns:
        return None
    work = train.copy()
    if "n_sweeps" not in work.columns:
        return None
    mask = (
        np.isfinite(work[col].to_numpy(dtype=np.float64))
        & (work[col].to_numpy(dtype=np.float64) > 0)
        & (work["n_sweeps"].to_numpy(dtype=np.float64) > 0)
    )
    work = work.loc[mask].reset_index(drop=True)
    if len(work) < DEFAULT_ONLINE_MIN_POINTS:
        return None
    try:
        return neff_fit_at_fixed_params(
            work,
            z_dyn=float(summary["z_dyn_mean"]),
            noise_sigma=float(summary["noise_sigma_mean"]),
            length_scale=(
                float(summary["length_scale_fixed"])
                if "length_scale_fixed" in summary
                else None
            ),
            amplitude=(
                float(summary["amplitude_fixed"])
                if "amplitude_fixed" in summary
                else None
            ),
            channel="binder",
            T_c=float(summary.get("T_c", TC_EXACT)),
            nu=float(summary.get("nu", NU_EXACT)),
            kernel=summary.get("kernel", "gaussian"),
        )
    except Exception:
        return None


def _build_neff_panel(
    run_dir: Path,
    iteration: int,
    n_draws: np.ndarray,
    *,
    L_values: tuple[int, ...],
    stored_cache: dict[str, object],
) -> NeffPanelState:
    """Assemble scatter + optional GP curves for τ_int vs t."""
    iter_dir = run_dir / f"iter_{iteration:02d}"
    summary = parse_neff_fit_summary(iter_dir / "neff_scaling_fit.txt")
    train = _load_neff_train_df(
        run_dir, iteration, n_draws, stored_cache=stored_cache
    )
    col = neff_column("binder")
    if train.empty or col not in train.columns or "n_sweeps" not in train.columns:
        kind = "cold_start" if summary.get("kind") == "cold_start" else "none"
        return NeffPanelState(
            kind=kind,
            L_values=L_values,
            L=np.array([], dtype=np.float64),
            t=np.array([], dtype=np.float64),
            tau_int=np.array([], dtype=np.float64),
            n_train=int(summary.get("n_train", 0) or 0),
        )

    L = train["L"].to_numpy(dtype=np.float64)
    T = train["T"].to_numpy(dtype=np.float64)
    n_eff = train[col].to_numpy(dtype=np.float64)
    n_sweeps = train["n_sweeps"].to_numpy(dtype=np.float64)
    tau_int = _tau_int_from_neff(n_eff, n_sweeps)
    t = _reduced_t(T)
    n_train = int(len(train))

    fit = _fit_from_summary_and_train(train, summary)
    curve_t: dict[int, np.ndarray] = {}
    curve_tau: dict[int, np.ndarray] = {}
    z_dyn_mean: float | None = None
    kind = "cold_start"
    if fit is not None:
        kind = "gp"
        z_dyn_mean = float(fit.z_dyn_mean)
        for L_val in L_values:
            mask = np.isclose(L, float(L_val))
            if not mask.any():
                continue
            t_pts = t[mask]
            t_lo, t_hi = float(t_pts.min()), float(t_pts.max())
            if t_hi <= t_lo:
                t_grid = np.array([t_lo], dtype=np.float64)
            else:
                t_grid = np.linspace(t_lo, t_hi, _NEFF_CURVE_POINTS)
            T_grid = TC_EXACT * (1.0 + t_grid)
            L_grid = np.full(T_grid.shape, float(L_val))
            # N=1 → predict_neff returns the rate n_eff/N; τ = 1/(2 rate).
            rate_hat, _ = predict_neff(fit, L_grid, T_grid, n_sweeps=1.0)
            curve_t[int(L_val)] = t_grid
            curve_tau[int(L_val)] = 1.0 / (2.0 * np.maximum(rate_hat, 1e-12))
    elif summary.get("kind") == "cold_start" or n_train < DEFAULT_ONLINE_MIN_POINTS:
        kind = "cold_start"
    else:
        kind = "none"

    return NeffPanelState(
        kind=kind,
        L_values=L_values,
        L=L,
        t=t,
        tau_int=tau_int,
        curve_t=curve_t,
        curve_tau=curve_tau,
        z_dyn_mean=z_dyn_mean,
        n_train=n_train,
    )


def _hex_to_rgba(color: str, alpha: float) -> str:
    """Convert ``#rrggbb`` (or a named fallback) to an ``rgba(...)`` string."""
    c = str(color).strip()
    if c.startswith("#") and len(c) == 7:
        r = int(c[1:3], 16)
        g = int(c[3:5], 16)
        b = int(c[5:7], 16)
        return f"rgba({r},{g},{b},{alpha})"
    return f"rgba(0,0,0,{alpha})"


def _subsample_param_draws(
    tc: np.ndarray,
    nu: np.ndarray,
    *,
    n_max: int = _BINDER_GP_PARAM_DRAWS,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample ``(T_c, ν)`` draws for the Binder GP mixture predictive."""
    tc = np.asarray(tc, dtype=np.float64).ravel()
    nu = np.asarray(nu, dtype=np.float64).ravel()
    if tc.size != nu.size:
        raise ValueError(f"tc/nu size mismatch: {tc.size} vs {nu.size}")
    if tc.size == 0:
        return tc, nu
    if tc.size <= n_max:
        return tc, nu
    rng = np.random.default_rng(seed)
    idx = rng.choice(tc.size, size=int(n_max), replace=False)
    return tc[idx], nu[idx]


def _binder_T_grids(
    full_df: pd.DataFrame,
    L_values: tuple[int, ...],
) -> dict[int, np.ndarray]:
    """Fixed temperature grids per L for Binder GP curves."""
    grids: dict[int, np.ndarray] = {}
    for L_val in L_values:
        mask_L = full_df["L"].to_numpy(dtype=np.int64) == int(L_val)
        if not mask_L.any():
            continue
        T_L = full_df.loc[mask_L, "T"].to_numpy(dtype=np.float64)
        T_lo = float(T_L.min())
        T_hi = float(T_L.max())
        if T_hi <= T_lo:
            grids[int(L_val)] = np.array([T_lo], dtype=np.float64)
        else:
            grids[int(L_val)] = np.linspace(T_lo, T_hi, _BINDER_GP_CURVE_POINTS)
    return grids


def _predict_binder_gp_curves(
    train_df: pd.DataFrame,
    full_df: pd.DataFrame,
    *,
    T_c: float,
    nu: float,
    L_values: tuple[int, ...],
) -> tuple[
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    int,
]:
    """Fit Binder GP on ``train_df`` at a single ``(T_c, ν)``; predict vs ``T``.

    Returns ``(curve_T, curve_U, curve_U_std, n_train)``. Empty dicts on failure.
    """
    return _predict_binder_gp_mixture(
        train_df,
        full_df,
        tc_draws=np.asarray([T_c], dtype=np.float64),
        nu_draws=np.asarray([nu], dtype=np.float64),
        L_values=L_values,
    )


def _predict_binder_gp_mixture(
    train_df: pd.DataFrame,
    full_df: pd.DataFrame,
    *,
    tc_draws: np.ndarray,
    nu_draws: np.ndarray,
    L_values: tuple[int, ...],
) -> tuple[
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    int,
]:
    """Mixture Binder GP predictive over ``(T_c, ν)`` draws, plotted vs raw ``T``.

    For each draw ``θ = (T_c, ν)`` the GP is fit in collapsed ``z``-space. The
    reported mean/std are the law of total expectation/variance:
        μ = E[μ(T|θ)],  σ² = E[σ²(T|θ)] + Var[μ(T|θ)].
    """
    empty: tuple[
        dict[int, np.ndarray],
        dict[int, np.ndarray],
        dict[int, np.ndarray],
        int,
    ] = ({}, {}, {}, 0)
    required = {"L", "T", "binder_cumulant", "binder_cumulant_std", "n_eff_binder"}
    if not required.issubset(train_df.columns) or train_df.empty:
        return empty

    tc_draws = np.asarray(tc_draws, dtype=np.float64).ravel()
    nu_draws = np.asarray(nu_draws, dtype=np.float64).ravel()
    ok = (
        np.isfinite(tc_draws)
        & (tc_draws > 0.0)
        & np.isfinite(nu_draws)
        & (nu_draws > 0.0)
    )
    tc_draws = tc_draws[ok]
    nu_draws = nu_draws[ok]
    if tc_draws.size == 0:
        return empty

    config = build_al_config()
    try:
        arrays = extract_likelihood_arrays(train_df, config)
    except Exception:
        return empty
    if arrays.binder is None or arrays.sigma_binder is None or arrays.binder.size < 1:
        return empty

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
    kernel = str(config.gp_kernel)
    T_grids = _binder_T_grids(full_df, L_values)
    if not T_grids:
        return empty

    means: dict[int, list[np.ndarray]] = {L: [] for L in T_grids}
    vars_: dict[int, list[np.ndarray]] = {L: [] for L in T_grids}

    for T_c, nu in zip(tc_draws, nu_draws, strict=True):
        z_train = collapse_z(arrays.L, arrays.T, T_c=float(T_c), nu=float(nu))
        for L_val, T_grid in T_grids.items():
            L_grid = np.full(T_grid.shape, float(L_val))
            z_test = collapse_z(L_grid, T_grid, T_c=float(T_c), nu=float(nu))
            try:
                u_mean, u_std = gp_posterior_predictive(
                    z_train,
                    arrays.binder,
                    arrays.sigma_binder,
                    z_test,
                    length_scale=scales.base_gp_ell,
                    amplitude=scales.base_gp_eta,
                    kernel=kernel,
                    jitter=GP_JITTER,
                )
            except Exception:
                continue
            means[L_val].append(np.asarray(u_mean, dtype=np.float64))
            vars_[L_val].append(np.asarray(u_std, dtype=np.float64) ** 2)

    curve_T: dict[int, np.ndarray] = {}
    curve_U: dict[int, np.ndarray] = {}
    curve_U_std: dict[int, np.ndarray] = {}
    for L_val, T_grid in T_grids.items():
        if not means[L_val]:
            continue
        mu_stack = np.stack(means[L_val], axis=0)
        var_stack = np.stack(vars_[L_val], axis=0)
        mu = mu_stack.mean(axis=0)
        # Law of total variance: E[Var] + Var[E]
        mixed_var = var_stack.mean(axis=0) + mu_stack.var(axis=0)
        curve_T[int(L_val)] = T_grid
        curve_U[int(L_val)] = mu
        curve_U_std[int(L_val)] = np.sqrt(np.maximum(mixed_var, 0.0))

    if not curve_T:
        return empty
    return curve_T, curve_U, curve_U_std, int(arrays.binder.size)


def _build_exact_binder_gp_curves(
    full_df: pd.DataFrame,
    L_values: tuple[int, ...],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Binder GP on the full grid at exact ``(T_c, ν)`` — dashed reference overlay."""
    curve_T, curve_U, _, _ = _predict_binder_gp_curves(
        full_df.reset_index(drop=True),
        full_df,
        T_c=float(TC_EXACT),
        nu=float(NU_EXACT),
        L_values=L_values,
    )
    return curve_T, curve_U


def _build_binder_gp_panel(
    full_df: pd.DataFrame,
    n_draws: np.ndarray,
    *,
    tc_draws: np.ndarray,
    nu_draws: np.ndarray,
    L_values: tuple[int, ...],
    exact_curve_T: dict[int, np.ndarray] | None = None,
    exact_curve_U: dict[int, np.ndarray] | None = None,
    param_draw_seed: int = 0,
) -> BinderGpPanelState:
    """FSS Binder GP mixture predictive ``Û(T)`` ± 1σ over ``(T_c, ν)`` draws."""
    exact_T = exact_curve_T or {}
    exact_U = exact_curve_U or {}
    required = {"L", "T", "binder_cumulant", "binder_cumulant_std", "n_eff_binder"}
    if not required.issubset(full_df.columns):
        return BinderGpPanelState.empty(
            L_values, exact_curve_T=exact_T, exact_curve_U=exact_U
        )

    active = np.asarray(n_draws, dtype=np.int64) > 0
    if not active.any():
        return BinderGpPanelState.empty(
            L_values, exact_curve_T=exact_T, exact_curve_U=exact_U
        )

    obs = full_df.loc[active].reset_index(drop=True)
    tc_sub, nu_sub = _subsample_param_draws(
        tc_draws, nu_draws, seed=param_draw_seed
    )
    curve_T, curve_U, curve_U_std, n_train = _predict_binder_gp_mixture(
        obs,
        full_df,
        tc_draws=tc_sub,
        nu_draws=nu_sub,
        L_values=L_values,
    )
    if not curve_T:
        return BinderGpPanelState.empty(
            L_values, exact_curve_T=exact_T, exact_curve_U=exact_U
        )

    return BinderGpPanelState(
        kind="gp",
        L_values=L_values,
        L=obs["L"].to_numpy(dtype=np.float64),
        T=obs["T"].to_numpy(dtype=np.float64),
        U=obs["binder_cumulant"].to_numpy(dtype=np.float64),
        curve_T=curve_T,
        curve_U=curve_U,
        curve_U_std=curve_U_std,
        exact_curve_T=exact_T,
        exact_curve_U=exact_U,
        n_train=n_train,
    )


def n_draws_before_iteration(
    summary: pd.DataFrame,
    n_points: int,
    iteration: int,
    *,
    chunk_size: int,
) -> np.ndarray:
    """Budget vector at the start of ``iteration`` (before that step's allocation)."""
    n_draws = np.zeros(n_points, dtype=np.int64)
    for row in summary.itertuples(index=False):
        if int(row.iteration) >= iteration:
            break
        n_draws[int(row.added_row_index)] += int(chunk_size)
    return n_draws


def _iter_dir_complete(iter_dir: Path) -> bool:
    """True when an iteration finished allocation (ranking + after-budget on disk)."""
    return (iter_dir / "ranking.csv").is_file() and (iter_dir / "n_draws.npy").is_file()


def discover_completed_iterations(run_dir: Path) -> list[int]:
    """Sorted iteration indices that have finished scoring + allocation."""
    import re

    pat = re.compile(r"^iter_(\d+)$")
    out: list[int] = []
    if not run_dir.is_dir():
        return out
    for path in sorted(run_dir.iterdir()):
        if not path.is_dir():
            continue
        m = pat.match(path.name)
        if m is None:
            continue
        if _iter_dir_complete(path):
            out.append(int(m.group(1)))
    return out


def reconstruct_summary_from_iters(
    run_dir: Path,
    *,
    chunk_size: int,
    n_points: int,
) -> pd.DataFrame:
    """Build a summary table from completed ``iter_XX/`` dirs (partial-run safe).

    Derives the chosen index from consecutive ``n_draws.npy`` diffs. Looks up
    Δtr(cov) from that iter's ``ranking.csv`` when available.
    """
    completed = discover_completed_iterations(run_dir)
    if not completed:
        return pd.DataFrame()

    prev = np.zeros(n_points, dtype=np.int64)
    rows: list[dict[str, object]] = []
    for iteration in completed:
        iter_dir = run_dir / f"iter_{iteration:02d}"
        after = np.asarray(np.load(iter_dir / "n_draws.npy"), dtype=np.int64).ravel()
        if after.size != n_points:
            continue
        diff = after - prev
        added = np.flatnonzero(diff > 0)
        if added.size == 0:
            continue
        chunk_hits = added[diff[added] == chunk_size]
        chosen_idx = int(chunk_hits[0]) if chunk_hits.size else int(added[np.argmax(diff[added])])

        ranking_path = iter_dir / "ranking.csv"
        delta = float("nan")
        chunks_before = int(prev[chosen_idx] // max(chunk_size, 1))
        tr_before = float("nan")
        tr_after = float("nan")
        ess = float("nan")
        pareto_k = float("nan")
        planned = float("nan")
        if ranking_path.is_file():
            ranking = pd.read_csv(ranking_path)
            if "row_index" in ranking.columns:
                hit = ranking.loc[ranking["row_index"].astype(int) == chosen_idx]
                if not hit.empty:
                    r0 = hit.iloc[0]
                    if "delta_tr_cov" in hit.columns:
                        delta = float(r0["delta_tr_cov"])
                    if "chunks_before" in hit.columns:
                        chunks_before = int(r0["chunks_before"])
                    if "tr_cov_before" in hit.columns:
                        tr_before = float(r0["tr_cov_before"])
                    if "tr_cov_after" in hit.columns:
                        tr_after = float(r0["tr_cov_after"])
                    if "ess" in hit.columns:
                        ess = float(r0["ess"])
                    if "pareto_k" in hit.columns:
                        pareto_k = float(r0["pareto_k"])
                    if "planned_n_eff_after" in hit.columns:
                        planned = float(r0["planned_n_eff_after"])

        budget_path = iter_dir / "budget.csv"
        added_L = -1
        added_T = float("nan")
        if budget_path.is_file():
            budget = pd.read_csv(budget_path)
            if chosen_idx < len(budget):
                added_L = int(budget.iloc[chosen_idx]["L"])
                added_T = float(budget.iloc[chosen_idx]["T"])

        rows.append(
            {
                "iteration": iteration,
                "n_points": int(np.sum(prev > 0)),
                "total_draws_before": int(prev.sum()),
                "added_row_index": chosen_idx,
                "added_L": added_L,
                "added_T": added_T,
                "chunks_before": chunks_before,
                "chunks_after": chunks_before + 1,
                "n_draws_after_point": int(after[chosen_idx]),
                "delta_tr_cov": delta,
                "tr_cov_before": tr_before,
                "tr_cov_after": tr_after,
                "ess": ess,
                "pareto_k": pareto_k,
                "planned_n_eff_after": planned,
            }
        )
        prev = after
    return pd.DataFrame(rows)


def load_budget_summary(
    run_dir: Path,
    *,
    chunk_size: int,
    n_points: int,
) -> pd.DataFrame:
    """Load ``summary.csv``, or reconstruct from completed iters if missing/stale.

    If both exist, prefer whichever covers more completed iterations.
    """
    completed = discover_completed_iterations(run_dir)
    reconstructed = reconstruct_summary_from_iters(
        run_dir, chunk_size=chunk_size, n_points=n_points
    )
    summary_path = run_dir / "summary.csv"
    if summary_path.is_file():
        summary = pd.read_csv(summary_path).sort_values("iteration").reset_index(drop=True)
        if completed and "iteration" in summary.columns:
            summary = summary.loc[summary["iteration"].astype(int).isin(completed)].reset_index(
                drop=True
            )
        if len(summary) >= len(reconstructed):
            return summary
    if reconstructed.empty:
        raise FileNotFoundError(
            f"No completed budget-AL iterations under {run_dir} "
            "(need iter_XX/ranking.csv + n_draws.npy)"
        )
    return reconstructed


def _load_tc_nu_for_iteration(
    run_dir: Path,
    iteration: int,
    *,
    max_prior_draws: int = 500,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return (T_c, nu, has_posterior) for one iteration."""
    from ._deps import az

    nc_path = run_dir / f"iter_{iteration:02d}" / "posterior.nc"
    if nc_path.is_file():
        idata = az.from_netcdf(nc_path)
        tc = np.asarray(idata.posterior["T_c"].values.reshape(-1), dtype=np.float64)
        nu = np.asarray(idata.posterior["nu"].values.reshape(-1), dtype=np.float64)
        return tc, nu, True

    draws = sample_prior_draws(
        max_prior_draws,
        seed=seed + iteration,
        infer_Tc=True,
        infer_nu=True,
        infer_beta=False,
    )
    return draws.T_c, draws.nu, False


def build_budget_stepper_frames(
    run_dir: Path,
    *,
    n_bins: int = 40,
    max_prior_draws: int = 500,
    max_frames: int | None = None,
    heatmap_percentiles: tuple[float, float] = (0.5, 99.5),
    heatmap_pad_frac: float = 0.08,
    heatmap_posterior_only_limits: bool = False,
) -> tuple[pd.DataFrame, list[BudgetStepperFrame], tuple[float, float], tuple[float, float]]:
    """Load budget-AL artifacts into animation frames (partial runs supported).

    If ``max_frames`` is set, only the earliest that many iterations are loaded
    (useful for GIF exports of the first ~100 steps).

    ``heatmap_percentiles`` / ``heatmap_pad_frac`` control the shared ``(T_c, ν)``
    axis window. Set ``heatmap_posterior_only_limits=True`` to ignore prior-only
    iterations when choosing that window (tighter zoom for GIFs).
    """
    meta = _load_run_meta(run_dir)
    chunk_size = int(meta.get("chunk_size", 10_000))
    n_chunks_max = int(meta.get("n_chunks_max", 8))
    full_df = _load_full_grid(run_dir)
    n_points = len(full_df)
    L_values = tuple(int(x) for x in sorted(full_df["L"].unique()))
    stored_cache: dict[str, object] = {}

    summary = load_budget_summary(run_dir, chunk_size=chunk_size, n_points=n_points)
    if summary.empty:
        raise ValueError(f"No complete iterations to display under {run_dir}")
    summary = summary.sort_values("iteration").reset_index(drop=True)
    if max_frames is not None:
        summary = summary.head(int(max_frames)).reset_index(drop=True)

    # Shared heatmap grid from all available posteriors (+ prior draws for empty iters).
    tc_all: list[np.ndarray] = []
    nu_all: list[np.ndarray] = []
    tc_post: list[np.ndarray] = []
    nu_post: list[np.ndarray] = []
    samples_by_iter: dict[int, tuple[np.ndarray, np.ndarray, bool]] = {}
    for iteration in summary["iteration"].astype(int).tolist():
        tc, nu, has_post = _load_tc_nu_for_iteration(
            run_dir,
            int(iteration),
            max_prior_draws=max_prior_draws,
            seed=int(meta.get("seed", 0)),
        )
        samples_by_iter[int(iteration)] = (tc, nu, has_post)
        tc_all.append(tc)
        nu_all.append(nu)
        if has_post:
            tc_post.append(tc)
            nu_post.append(nu)

    tc_for_lim = tc_post if (heatmap_posterior_only_limits and tc_post) else tc_all
    nu_for_lim = nu_post if (heatmap_posterior_only_limits and nu_post) else nu_all
    # For zoomed GIFs, average per-frame percentile windows from early posteriors.
    # Pooling all early draws is too wide (outliers); late-only pooling collapses
    # onto a tiny blob.
    if heatmap_posterior_only_limits and len(tc_for_lim) > 1:
        n_early = max(1, min(15, len(tc_for_lim)))
        p_lo, p_hi = heatmap_percentiles
        tc_los = [float(np.percentile(tc, p_lo)) for tc in tc_for_lim[:n_early]]
        tc_his = [float(np.percentile(tc, p_hi)) for tc in tc_for_lim[:n_early]]
        nu_los = [float(np.percentile(nu, p_lo)) for nu in nu_for_lim[:n_early]]
        nu_his = [float(np.percentile(nu, p_hi)) for nu in nu_for_lim[:n_early]]
        tc_lo, tc_hi = float(np.mean(tc_los)), float(np.mean(tc_his))
        nu_lo, nu_hi = float(np.mean(nu_los)), float(np.mean(nu_his))
        tc_pad = max(tc_hi - tc_lo, 1e-6) * heatmap_pad_frac
        nu_pad = max(nu_hi - nu_lo, 1e-6) * heatmap_pad_frac
        tc_lim = (tc_lo - tc_pad, tc_hi + tc_pad)
        nu_lim = (nu_lo - nu_pad, nu_hi + nu_pad)
    else:
        tc_lim = _shared_xlim(
            tc_for_lim, pad_frac=heatmap_pad_frac, percentiles=heatmap_percentiles
        )
        nu_lim = _shared_xlim(
            nu_for_lim, pad_frac=heatmap_pad_frac, percentiles=heatmap_percentiles
        )
    # Always keep the exact (T_c, ν) marker inside the shared window.
    tc_span = max(tc_lim[1] - tc_lim[0], 1e-6)
    nu_span = max(nu_lim[1] - nu_lim[0], 1e-6)
    tc_lim = (
        min(tc_lim[0], float(TC_EXACT) - 0.12 * tc_span),
        max(tc_lim[1], float(TC_EXACT) + 0.12 * tc_span),
    )
    nu_lim = (
        min(nu_lim[0], float(NU_EXACT) - 0.12 * nu_span),
        max(nu_lim[1], float(NU_EXACT) + 0.12 * nu_span),
    )
    tc_edges = np.linspace(tc_lim[0], tc_lim[1], n_bins + 1)
    nu_edges = np.linspace(nu_lim[0], nu_lim[1], n_bins + 1)
    tc_centers = 0.5 * (tc_edges[:-1] + tc_edges[1:])
    nu_centers = 0.5 * (nu_edges[:-1] + nu_edges[1:])

    frames: list[BudgetStepperFrame] = []
    exact_curve_T, exact_curve_U = _build_exact_binder_gp_curves(full_df, L_values)
    for row in summary.itertuples(index=False):
        iteration = int(row.iteration)
        tc, nu, has_post = samples_by_iter[iteration]
        mat = np.column_stack([tc, nu])
        tr_cov = uncertainty_scalar(np.cov(mat, rowvar=False), metric="trace_cov")
        z = np.histogram2d(tc, nu, bins=[tc_edges, nu_edges], density=True)[0].T

        n_draws = n_draws_before_iteration(
            summary, n_points, iteration, chunk_size=chunk_size
        )
        # Prefer on-disk budget if present (after state); convert to before.
        budget_path = run_dir / f"iter_{iteration:02d}" / "n_draws.npy"
        if budget_path.is_file():
            after = np.asarray(np.load(budget_path), dtype=np.int64).ravel()
            if after.size == n_points:
                before = after.copy()
                before[int(row.added_row_index)] = max(
                    0, int(before[int(row.added_row_index)]) - chunk_size
                )
                n_draws = before

        rank_path = run_dir / f"iter_{iteration:02d}" / "ranking.csv"
        if rank_path.is_file():
            ranking = pd.read_csv(rank_path)
        else:
            ranking = pd.DataFrame(columns=["row_index", "delta_tr_cov"])

        neff_panel = _build_neff_panel(
            run_dir,
            iteration,
            n_draws,
            L_values=L_values,
            stored_cache=stored_cache,
        )
        binder_gp_panel = _build_binder_gp_panel(
            full_df,
            n_draws,
            tc_draws=tc,
            nu_draws=nu,
            L_values=L_values,
            exact_curve_T=exact_curve_T,
            exact_curve_U=exact_curve_U,
            param_draw_seed=iteration,
        )

        frames.append(
            BudgetStepperFrame(
                iteration=iteration,
                n_points=int(np.sum(n_draws > 0)),
                total_draws=int(n_draws.sum()),
                n_draws=n_draws,
                next_index=int(row.added_row_index),
                next_delta_tr_cov=float(row.delta_tr_cov),
                chunks_before=int(
                    getattr(
                        row,
                        "chunks_before",
                        n_draws[int(row.added_row_index)] // chunk_size,
                    )
                ),
                tr_cov=float(tr_cov),
                z=z,
                tc_centers=tc_centers,
                nu_centers=nu_centers,
                mean_tc=float(np.mean(tc)),
                mean_nu=float(np.mean(nu)),
                tc_std=float(np.std(tc, ddof=0)),
                ranking=ranking,
                has_posterior=has_post,
                neff_panel=neff_panel,
                binder_gp_panel=binder_gp_panel,
            )
        )

    if not frames:
        raise ValueError(f"No stepper frames could be built from {run_dir}")
    _ = n_chunks_max
    return full_df, frames, tc_lim, nu_lim


def _marker_size(n_draws: np.ndarray, *, chunk_size: int, n_chunks_max: int) -> np.ndarray:
    """Marker diameter in px; grows clearly with samples collected at that (L,T).

    Each allocated chunk adds a fixed step so 10k / 40k / 80k reads as small /
    medium / large. Untouched candidates stay small (~5px).
    """
    _ = n_chunks_max
    chunks = np.maximum(n_draws.astype(np.float64) / max(float(chunk_size), 1.0), 0.0)
    return 5.0 + 4.5 * chunks


def _heatmap_trace(frame: BudgetStepperFrame) -> go.Heatmap:
    title = "density" if frame.has_posterior else "prior dens."
    return go.Heatmap(
        x=frame.tc_centers,
        y=frame.nu_centers,
        z=frame.z,
        colorscale="Blues",
        showscale=True,
        colorbar=dict(
            title=title,
            x=0.30,
            xanchor="right",
            y=0.78,
            len=0.42,
            thickness=12,
        ),
        zmin=0.0,
        hovertemplate=r"$T_c$=%{x:.5f}<br>$\nu$=%{y:.4f}<br>d=%{z:.3g}<extra></extra>",
    )


def _binder_traces(
    full_df: pd.DataFrame,
    frame: BudgetStepperFrame,
    *,
    chunk_size: int,
    n_chunks_max: int,
    temperature_scale: float = 2.0,
) -> list[go.Scatter]:
    L_arr = full_df["L"].to_numpy(dtype=np.int64)
    T_arr = full_df["T"].to_numpy(dtype=np.float64)
    u4 = full_df["binder_cumulant"].to_numpy(dtype=np.float64)
    n_draws = frame.n_draws
    sizes = _marker_size(n_draws, chunk_size=chunk_size, n_chunks_max=n_chunks_max)
    next_idx = frame.next_index
    score_by_idx = {
        int(r.row_index): float(r.delta_tr_cov) for r in frame.ranking.itertuples(index=False)
    }
    prob_by_idx = _choice_prob_by_index(frame.ranking, temperature_scale=temperature_scale)
    max_draws = chunk_size * n_chunks_max

    traces: list[go.Scatter] = []
    # Layer 1: points with budget already spent (size ∝ draws).
    for L in sorted(full_df["L"].unique()):
        style = _BINDER_STYLE.get(
            int(L),
            {"color": "black", "symbol": "circle", "size_base": 7, "label": f"L={L}"},
        )
        active = (n_draws > 0) & (L_arr == L)
        if not active.any():
            traces.append(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(color=style["color"], symbol=style["symbol"], size=style["size_base"]),
                    name=str(style["label"]),
                    xaxis="x2",
                    yaxis="y2",
                )
            )
            continue
        idx = np.arange(len(full_df))[active]
        traces.append(
            go.Scatter(
                x=T_arr[active],
                y=u4[active],
                mode="markers",
                marker=dict(
                    color=style["color"],
                    symbol=style["symbol"],
                    size=sizes[active],
                    line=dict(width=1.0, color="white"),
                    opacity=0.9,
                ),
                name=str(style["label"]),
                hovertemplate=(
                    "idx=%{customdata[0]}<br>T=%{x:.4f}<br>U4=%{y:.4f}<br>"
                    "n_draws=%{customdata[1]}<extra></extra>"
                ),
                customdata=np.column_stack([idx, n_draws[active]]),
                xaxis="x2",
                yaxis="y2",
            )
        )

    # Layer 2: remaining candidates from ranking (softmax P in hover; optional color).
    if not frame.ranking.empty:
        cand_idx = frame.ranking["row_index"].astype(int).to_numpy()
        # Keep only still-feasible rows (safety).
        cand_idx = cand_idx[n_draws[cand_idx] < max_draws]
        if cand_idx.size:
            deltas = np.array([score_by_idx.get(int(i), np.nan) for i in cand_idx], dtype=np.float64)
            probs = np.array([prob_by_idx.get(int(i), np.nan) for i in cand_idx], dtype=np.float64)
            if _COLOR_CANDIDATES_BY_SOFTMAX:
                # Color by softmax P(choose) this step (Turbo + right-hand colorbar).
                prob_cmin, prob_cmax = _prob_colormap_limits(probs)
                marker = dict(
                    size=sizes[cand_idx],
                    color=probs,
                    colorscale="Turbo",
                    cmin=prob_cmin,
                    cmax=prob_cmax,
                    showscale=True,
                    colorbar=dict(
                        title=r"P(choose)<br>(this step)",
                        x=1.02,
                        xanchor="left",
                        len=0.85,
                        thickness=14,
                        tickformat=".1%",
                    ),
                    line=dict(width=0.6, color="rgba(255,255,255,0.85)"),
                    symbol="circle",
                    opacity=0.85,
                )
                cand_name = "softmax P(choose)"
            else:
                marker = dict(
                    size=sizes[cand_idx],
                    color=_CANDIDATE_FLAT_COLOR,
                    line=dict(width=0.6, color="rgba(255,255,255,0.85)"),
                    symbol="circle",
                    opacity=0.55,
                )
                cand_name = "candidates"
            traces.append(
                go.Scatter(
                    x=T_arr[cand_idx],
                    y=u4[cand_idx],
                    mode="markers",
                    name=cand_name,
                    marker=marker,
                    customdata=np.column_stack(
                        [cand_idx, deltas, probs, n_draws[cand_idx]]
                    ),
                    hovertemplate=(
                        "idx=%{customdata[0]}<br>T=%{x:.4f}<br>U4=%{y:.4f}<br>"
                        "Δtr(cov)=%{customdata[1]:.4g}<br>P(choose)=%{customdata[2]:.2%}<br>"
                        "n_draws=%{customdata[3]}<extra></extra>"
                    ),
                    xaxis="x2",
                    yaxis="y2",
                )
            )

    next_row = full_df.iloc[next_idx]
    chosen_p = prob_by_idx.get(next_idx, float("nan"))
    traces.append(
        go.Scatter(
            x=[float(next_row["T"])],
            y=[float(next_row["binder_cumulant"])],
            mode="markers",
            marker=dict(symbol="star", size=18, color="#e6a817", line=dict(width=1.2, color="black")),
            name="chosen next +10k",
            hovertemplate=(
                f"chosen idx={next_idx}<br>L={int(next_row['L'])}<br>T=%{{x:.4f}}<br>"
                f"U4=%{{y:.4f}}<br>chunks_before={frame.chunks_before}<br>"
                f"Δtr(cov)={frame.next_delta_tr_cov:.4g}<br>"
                f"P(choose)={chosen_p:.2%}<extra></extra>"
            ),
            xaxis="x2",
            yaxis="y2",
        )
    )
    return traces


def _neff_style(L: int) -> dict[str, object]:
    return dict(
        _NEFF_L_STYLE.get(
            int(L),
            {"color": "black", "symbol": "circle", "label": f"L={L}"},
        )
    )


def _neff_traces(frame: BudgetStepperFrame) -> list[go.Scatter]:
    """Fixed-length trace list: scatter + curve per L (Plotly frame-safe)."""
    panel = frame.neff_panel
    traces: list[go.Scatter] = []
    for L in panel.L_values:
        style = _neff_style(int(L))
        mask = np.isclose(panel.L, float(L)) if panel.L.size else np.array([], dtype=bool)
        if mask.size and mask.any():
            traces.append(
                go.Scatter(
                    x=panel.t[mask],
                    y=panel.tau_int[mask],
                    mode="markers",
                    marker=dict(
                        color=style["color"],
                        symbol=style["symbol"],
                        size=9,
                        line=dict(width=0.8, color="white"),
                    ),
                    name=str(style["label"]),
                    legendgroup=f"neff-L{L}",
                    hovertemplate=(
                        f"L={int(L)}<br>t=%{{x:.4f}}<br>"
                        r"τ_int≈%{y:.2f}<extra></extra>"
                    ),
                    xaxis="x3",
                    yaxis="y3",
                )
            )
        else:
            traces.append(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(color=style["color"], symbol=style["symbol"], size=9),
                    name=str(style["label"]),
                    legendgroup=f"neff-L{L}",
                    showlegend=True,
                    xaxis="x3",
                    yaxis="y3",
                )
            )

        t_c = panel.curve_t.get(int(L))
        tau_c = panel.curve_tau.get(int(L))
        if t_c is not None and tau_c is not None and t_c.size:
            traces.append(
                go.Scatter(
                    x=t_c,
                    y=tau_c,
                    mode="lines",
                    line=dict(color=style["color"], width=1.6),
                    name=f"{style['label']} fit",
                    legendgroup=f"neff-L{L}",
                    showlegend=False,
                    hovertemplate=(
                        f"L={int(L)} fit<br>t=%{{x:.4f}}<br>"
                        r"τ̂_int≈%{y:.2f}<extra></extra>"
                    ),
                    xaxis="x3",
                    yaxis="y3",
                )
            )
        else:
            traces.append(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    line=dict(color=style["color"], width=1.6),
                    name=f"{style['label']} fit",
                    legendgroup=f"neff-L{L}",
                    showlegend=False,
                    xaxis="x3",
                    yaxis="y3",
                )
            )
    return traces


def _binder_gp_traces(frame: BudgetStepperFrame) -> list[go.Scatter]:
    """Fixed-length: budgeted scatter + ±1σ band + GP mean + exact dashed per L."""
    panel = frame.binder_gp_panel
    traces: list[go.Scatter] = []
    for L in panel.L_values:
        style = _neff_style(int(L))
        color = str(style["color"])
        mask = np.isclose(panel.L, float(L)) if panel.L.size else np.array([], dtype=bool)
        label = f"Û L={int(L)}"
        if mask.size and mask.any():
            traces.append(
                go.Scatter(
                    x=panel.T[mask],
                    y=panel.U[mask],
                    mode="markers",
                    marker=dict(
                        color=color,
                        symbol=style["symbol"],
                        size=9,
                        line=dict(width=0.8, color="white"),
                    ),
                    name=label,
                    legendgroup=f"ugp-L{L}",
                    showlegend=False,
                    hovertemplate=(
                        f"L={int(L)}<br>T=%{{x:.4f}}<br>"
                        r"U₄=%{y:.4f}<extra></extra>"
                    ),
                    xaxis="x4",
                    yaxis="y4",
                )
            )
        else:
            traces.append(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(color=color, symbol=style["symbol"], size=9),
                    name=label,
                    legendgroup=f"ugp-L{L}",
                    showlegend=False,
                    xaxis="x4",
                    yaxis="y4",
                )
            )

        T_curve = panel.curve_T.get(int(L))
        u_c = panel.curve_U.get(int(L))
        u_std = panel.curve_U_std.get(int(L))
        if (
            T_curve is not None
            and u_c is not None
            and u_std is not None
            and T_curve.size
            and u_c.size == T_curve.size
            and u_std.size == T_curve.size
        ):
            u_lo = u_c - u_std
            u_hi = u_c + u_std
            fillcolor = _hex_to_rgba(color, 0.22)
            traces.append(
                go.Scatter(
                    x=T_curve,
                    y=u_lo,
                    mode="lines",
                    line=dict(width=0, color=color),
                    name=label,
                    legendgroup=f"ugp-L{L}",
                    showlegend=False,
                    hoverinfo="skip",
                    xaxis="x4",
                    yaxis="y4",
                )
            )
            traces.append(
                go.Scatter(
                    x=T_curve,
                    y=u_hi,
                    mode="lines",
                    line=dict(width=0, color=color),
                    fill="tonexty",
                    fillcolor=fillcolor,
                    name=label,
                    legendgroup=f"ugp-L{L}",
                    showlegend=False,
                    hovertemplate=(
                        f"L={int(L)} GP+params ±1σ<br>T=%{{x:.4f}}<br>"
                        r"Û₄≈%{y:.4f}<extra></extra>"
                    ),
                    xaxis="x4",
                    yaxis="y4",
                )
            )
            traces.append(
                go.Scatter(
                    x=T_curve,
                    y=u_c,
                    mode="lines",
                    line=dict(color=color, width=1.8),
                    name=label,
                    legendgroup=f"ugp-L{L}",
                    showlegend=True,
                    hovertemplate=(
                        f"L={int(L)} GP<br>T=%{{x:.4f}}<br>"
                        r"Û₄≈%{y:.4f}<extra></extra>"
                    ),
                    xaxis="x4",
                    yaxis="y4",
                )
            )
        else:
            for _ in range(2):
                traces.append(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="lines",
                        line=dict(width=0, color=color),
                        name=label,
                        legendgroup=f"ugp-L{L}",
                        showlegend=False,
                        xaxis="x4",
                        yaxis="y4",
                    )
                )
            traces.append(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    line=dict(color=color, width=1.8),
                    name=label,
                    legendgroup=f"ugp-L{L}",
                    showlegend=True,
                    xaxis="x4",
                    yaxis="y4",
                )
            )

        exact_label = f"exact L={int(L)}"
        T_ex = panel.exact_curve_T.get(int(L))
        U_ex = panel.exact_curve_U.get(int(L))
        if (
            T_ex is not None
            and U_ex is not None
            and T_ex.size
            and U_ex.size == T_ex.size
        ):
            traces.append(
                go.Scatter(
                    x=T_ex,
                    y=U_ex,
                    mode="lines",
                    line=dict(color=color, width=1.6, dash="dash"),
                    name=exact_label,
                    legendgroup=f"ugp-exact-L{L}",
                    showlegend=True,
                    hovertemplate=(
                        f"L={int(L)} exact (all data)<br>T=%{{x:.4f}}<br>"
                        r"Û₄≈%{y:.4f}<extra></extra>"
                    ),
                    xaxis="x4",
                    yaxis="y4",
                )
            )
        else:
            traces.append(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    line=dict(color=color, width=1.6, dash="dash"),
                    name=exact_label,
                    legendgroup=f"ugp-exact-L{L}",
                    showlegend=True,
                    xaxis="x4",
                    yaxis="y4",
                )
            )
    # Legend entry for the inferred-Tc shapes (updated per frame via layout.shapes).
    traces.append(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="rgba(60,60,60,0.95)", width=1.6, dash="dot"),
            name=r"⟨Tc⟩ ±1σ",
            legendgroup="ugp-tc",
            showlegend=True,
            hoverinfo="skip",
            xaxis="x4",
            yaxis="y4",
        )
    )
    return traces


def _reference_shapes(frame: BudgetStepperFrame) -> list[dict]:
    """Static + per-frame reference lines/bands (must be set on every Frame).

    Includes exact markers on the Binder / τ_int / GP panels, plus the inferred
    ``⟨T_c⟩ ± 1σ`` band on the GP panel. All shapes are listed every frame so
    Plotly animation does not wipe earlier ``add_vline`` shapes.
    """
    tc = float(frame.mean_tc)
    tc_std = float(frame.tc_std)
    if np.isfinite(tc_std) and tc_std > 0.0:
        tc_lo = tc - tc_std
        tc_hi = tc + tc_std
    else:
        tc_lo = tc
        tc_hi = tc

    return [
        # Exact T_c on Binder grid (top-middle).
        dict(
            type="line",
            xref="x2",
            yref="y2 domain",
            x0=float(TC_EXACT),
            x1=float(TC_EXACT),
            y0=0,
            y1=1,
            line=dict(color="red", width=1, dash="dashdot"),
            layer="below",
        ),
        # Exact t=0 on τ_int panel (top-right).
        dict(
            type="line",
            xref="x3",
            yref="y3 domain",
            x0=0.0,
            x1=0.0,
            y0=0,
            y1=1,
            line=dict(color="red", width=1, dash="dashdot"),
            layer="below",
        ),
        # Exact T_c on Binder GP panel (bottom).
        dict(
            type="line",
            xref="x4",
            yref="y4 domain",
            x0=float(TC_EXACT),
            x1=float(TC_EXACT),
            y0=0,
            y1=1,
            line=dict(color="red", width=1, dash="dashdot"),
            layer="below",
        ),
        # Inferred ⟨T_c⟩ ± 1σ band on Binder GP panel.
        dict(
            type="rect",
            xref="x4",
            yref="y4 domain",
            x0=tc_lo,
            x1=tc_hi,
            y0=0,
            y1=1,
            fillcolor="rgba(60,60,60,0.16)",
            line=dict(width=0),
            layer="below",
        ),
        # Inferred ⟨T_c⟩ mean.
        dict(
            type="line",
            xref="x4",
            yref="y4 domain",
            x0=tc,
            x1=tc,
            y0=0,
            y1=1,
            line=dict(color="rgba(60,60,60,0.95)", width=1.6, dash="dot"),
            layer="below",
        ),
    ]


def _frame_title(
    run_dir: Path,
    frame: BudgetStepperFrame,
    *,
    temperature_scale: float,
) -> str:
    meta = _load_run_meta(run_dir)
    selection = meta.get("selection", "greedy")
    prob_by_idx = _choice_prob_by_index(frame.ranking, temperature_scale=temperature_scale)
    chosen_p = prob_by_idx.get(frame.next_index, float("nan"))
    post_tag = "LAPS" if frame.has_posterior else "prior"
    neff = frame.neff_panel
    if neff.kind == "gp" and neff.z_dyn_mean is not None:
        neff_tag = f"n_eff GP z_dyn={neff.z_dyn_mean:.3f} (n={neff.n_train})"
    elif neff.kind == "cold_start":
        neff_tag = f"n_eff cold_start (n={neff.n_train})"
    else:
        neff_tag = "n_eff n/a"
    ugp = frame.binder_gp_panel
    ugp_tag = (
        f"U GP n={ugp.n_train}"
        if ugp.kind == "gp"
        else "U GP n/a"
    )
    return (
        f"iter {frame.iteration:02d} ({selection}, {post_tag}): "
        f"{frame.n_points} pts / {frame.total_draws} draws, "
        f"next idx={frame.next_index} (+chunk), "
        f"Δtr(cov)={frame.next_delta_tr_cov:.4g}, "
        f"P(choose)={chosen_p:.1%}, tr(cov)={frame.tr_cov:.4g}, "
        f"{neff_tag}, {ugp_tag}"
    )


def _frame_traces(
    full_df: pd.DataFrame,
    frame: BudgetStepperFrame,
    *,
    chunk_size: int,
    n_chunks_max: int,
    temperature_scale: float,
) -> tuple[
    list[go.Heatmap | go.Scatter],
    list[go.Scatter],
    list[go.Scatter],
    list[go.Scatter],
]:
    # _overlay_traces expects StepperFrame-like mean_tc/mean_nu — BudgetStepperFrame has them.
    left: list[go.Heatmap | go.Scatter] = [
        _heatmap_trace(frame),
        *_overlay_traces(frame),  # type: ignore[arg-type]
    ]
    middle = _binder_traces(
        full_df,
        frame,
        chunk_size=chunk_size,
        n_chunks_max=n_chunks_max,
        temperature_scale=temperature_scale,
    )
    right = _neff_traces(frame)
    bottom = _binder_gp_traces(frame)
    return left, middle, right, bottom


def build_budget_stepper_figure(
    run_dir: Path,
    *,
    n_bins: int = 40,
    softmax_temperature_scale: float = 2.0,
) -> go.Figure:
    """Build the Plotly stepper figure for a budget-AL run directory."""
    meta = _load_run_meta(run_dir)
    chunk_size = int(meta.get("chunk_size", 10_000))
    n_chunks_max = int(meta.get("n_chunks_max", 8))
    full_df, frames, tc_lim, nu_lim = build_budget_stepper_frames(run_dir, n_bins=n_bins)

    fig = make_subplots(
        rows=2,
        cols=3,
        column_widths=[0.32, 0.34, 0.34],
        row_heights=[0.55, 0.45],
        specs=[
            [{}, {}, {}],
            [{"colspan": 3}, None, None],
        ],
        subplot_titles=(
            r"Posterior / prior $(T_c, \nu)$",
            (
                r"Binder (size ∝ draws; color = softmax P)"
                if _COLOR_CANDIDATES_BY_SOFTMAX
                else r"Binder (size ∝ draws)"
            ),
            r"$\tau_{\mathrm{int}}\approx N/(2n_{\mathrm{eff}})$ vs $t$",
            r"FSS Binder GP $\hat U_4(T)\pm 1\sigma$; gray = $\langle T_c\rangle\pm 1\sigma$",
        ),
        horizontal_spacing=0.08,
        # Large gap so the iteration slider can sit between the two rows.
        vertical_spacing=0.22,
    )

    f0_left, f0_mid, f0_right, f0_bottom = _frame_traces(
        full_df,
        frames[0],
        chunk_size=chunk_size,
        n_chunks_max=n_chunks_max,
        temperature_scale=softmax_temperature_scale,
    )
    for tr in f0_left:
        fig.add_trace(tr, row=1, col=1)
    for tr in f0_mid:
        fig.add_trace(tr, row=1, col=2)
    for tr in f0_right:
        fig.add_trace(tr, row=1, col=3)
    for tr in f0_bottom:
        fig.add_trace(tr, row=2, col=1)

    plotly_frames = []
    for frame in frames:
        left, mid, right, bottom = _frame_traces(
            full_df,
            frame,
            chunk_size=chunk_size,
            n_chunks_max=n_chunks_max,
            temperature_scale=softmax_temperature_scale,
        )
        plotly_frames.append(
            go.Frame(
                name=str(frame.iteration),
                data=[*left, *mid, *right, *bottom],
                layout=go.Layout(
                    title_text=_frame_title(
                        run_dir, frame, temperature_scale=softmax_temperature_scale
                    ),
                    shapes=_reference_shapes(frame),
                ),
            )
        )
    fig.frames = plotly_frames

    slider_steps = [
        {
            "args": [
                [str(frame.iteration)],
                {
                    "frame": {"duration": 0, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 0},
                },
            ],
            "label": f"{frame.iteration:02d}",
            "method": "animate",
        }
        for frame in frames
    ]

    fig.update_layout(
        title_text=_frame_title(
            run_dir, frames[0], temperature_scale=softmax_temperature_scale
        ),
        template="plotly_white",
        width=1580,
        height=900,
        margin=dict(l=56, r=120, t=88, b=40),
        showlegend=True,
        legend=dict(font=dict(size=9), x=1.01, y=0.72),
        # Explicit domains: top row above the controls, bottom row below.
        # Gap ≈ [0.44, 0.54] in paper coords hosts Prev/Next + slider.
        yaxis=dict(domain=[0.54, 1.0]),
        yaxis2=dict(domain=[0.54, 1.0]),
        yaxis3=dict(domain=[0.54, 1.0]),
        yaxis4=dict(domain=[0.0, 0.42]),
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.01,
                "y": 0.48,
                "xanchor": "left",
                "yanchor": "middle",
                "direction": "right",
                "pad": {"r": 8, "t": 0, "b": 0},
                "buttons": [
                    {
                        "label": "Prev",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                                "fromcurrent": True,
                                "direction": "backward",
                            },
                        ],
                    },
                    {
                        "label": "Next",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                                "fromcurrent": True,
                                "direction": "forward",
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.12,
                "y": 0.48,
                "len": 0.78,
                "xanchor": "left",
                "yanchor": "middle",
                "pad": {"t": 0, "b": 0},
                "currentvalue": {
                    "prefix": "Iteration: ",
                    "font": {"size": 12},
                    "xanchor": "right",
                    "offset": -15,
                },
                "font": {"size": 9},
                "steps": slider_steps,
            }
        ],
    )

    fig.update_xaxes(title_text=r"$T_c$", range=list(tc_lim), row=1, col=1)
    fig.update_yaxes(title_text=r"$\nu$", range=list(nu_lim), row=1, col=1)
    fig.update_xaxes(title_text=r"$T$", row=1, col=2)
    fig.update_yaxes(title_text=r"$U_4$", row=1, col=2)
    fig.update_xaxes(title_text=r"$t=(T-T_c)/T_c$", row=1, col=3)
    fig.update_yaxes(title_text=r"$\tau_{\mathrm{int}}$ (sweeps)", type="log", row=1, col=3)
    fig.update_xaxes(title_text=r"$T$", row=2, col=1)
    fig.update_yaxes(title_text=r"$\hat U_4$", row=2, col=1)
    # Re-assert domains after axis title updates (Plotly can reset them).
    fig.update_yaxes(domain=[0.54, 1.0], row=1, col=1)
    fig.update_yaxes(domain=[0.54, 1.0], row=1, col=2)
    fig.update_yaxes(domain=[0.54, 1.0], row=1, col=3)
    fig.update_yaxes(domain=[0.0, 0.42], row=2, col=1)
    fig.update_layout(shapes=_reference_shapes(frames[0]))
    return fig


def budget_stepper_keyboard_post_script(frames: list[BudgetStepperFrame]) -> str:
    frame_names = [str(frame.iteration) for frame in frames]
    return f"""
var frameNames = {json.dumps(frame_names)};
var stepIdx = 0;
function gotoStep(idx) {{
  stepIdx = Math.max(0, Math.min(idx, frameNames.length - 1));
  Plotly.animate('budget-al-stepper', [frameNames[stepIdx]], {{
    frame: {{duration: 0, redraw: true}},
    mode: 'immediate',
    transition: {{duration: 0}}
  }});
}}
document.addEventListener('keydown', function(e) {{
  if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {{
    gotoStep(stepIdx + 1);
    e.preventDefault();
  }} else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {{
    gotoStep(stepIdx - 1);
    e.preventDefault();
  }}
}});
var gd = document.getElementById('budget-al-stepper');
if (gd) {{
  gd.on('plotly_sliderchange', function(e) {{
    stepIdx = e.step._index;
  }});
}}
"""


def _mpl_binder_style(L: int) -> dict[str, object]:
    style = _BINDER_STYLE.get(
        int(L),
        {"color": "black", "symbol": "circle", "size_base": 7, "label": f"L={L}"},
    )
    return dict(style)


def _draw_budget_stepper_gif_frame(
    axes: tuple[object, object, object],
    full_df: pd.DataFrame,
    frame: BudgetStepperFrame,
    *,
    tc_lim: tuple[float, float],
    nu_lim: tuple[float, float],
    chunk_size: int,
    n_chunks_max: int,
    u4_lim: tuple[float, float],
    T_lim: tuple[float, float],
    run_dir: Path,
) -> None:
    """Render heatmap + Binder grid + Binder GP into the three matplotlib axes."""
    ax_heat, ax_bind, ax_gp = axes
    for ax in axes:
        ax.clear()

    # --- Heatmap (T_c, ν) ---
    # Per-frame √-scaled density so early diffuse posteriors keep visible mass
    # (a shared vmax across the GIF washes them out against late peaked frames).
    # Truncate Blues so low-density bins stay readable; keep exact zeros white.
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.ndimage import gaussian_filter

    extent = [tc_lim[0], tc_lim[1], nu_lim[0], nu_lim[1]]
    z = np.asarray(frame.z, dtype=np.float64)
    # Stronger blur when the histogram is sparse (early iterations).
    n_hit = int(np.count_nonzero(z > 0))
    sigma = 1.6 if n_hit < 0.08 * z.size else 0.9
    z = gaussian_filter(z, sigma=sigma)
    z_peak = float(np.max(z)) if z.size else 0.0
    if z_peak > 0.0:
        z_show = np.sqrt(np.clip(z / z_peak, 0.0, 1.0))
    else:
        z_show = z
    blues = plt.get_cmap("Blues")
    heat_cmap = LinearSegmentedColormap.from_list(
        "blues_hi",
        blues(np.linspace(0.22, 1.0, 256)),
    )
    heat_cmap.set_bad("white")
    z_masked = np.ma.masked_where(z_show <= 1e-3, z_show)
    ax_heat.imshow(
        z_masked,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=heat_cmap,
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
    )

    ax_heat.plot(TC_EXACT, NU_EXACT, marker="*", color="black", markersize=11, linestyle="none")
    ax_heat.plot(
        frame.mean_tc,
        frame.mean_nu,
        marker="x",
        color="#c44e52",
        markersize=9,
        linestyle="none",
    )
    dens = "posterior" if frame.has_posterior else "prior"
    ax_heat.set_title(f"(T_c, ν) {dens}")
    ax_heat.set_xlabel(r"$T_c$")
    ax_heat.set_ylabel(r"$\nu$")
    ax_heat.set_xlim(*tc_lim)
    ax_heat.set_ylim(*nu_lim)

    # --- Binder grid ---
    L_arr = full_df["L"].to_numpy(dtype=np.int64)
    T_arr = full_df["T"].to_numpy(dtype=np.float64)
    u4 = full_df["binder_cumulant"].to_numpy(dtype=np.float64)
    n_draws = frame.n_draws
    sizes = _marker_size(n_draws, chunk_size=chunk_size, n_chunks_max=n_chunks_max)
    max_draws = chunk_size * n_chunks_max

    for L in sorted(full_df["L"].unique()):
        style = _mpl_binder_style(int(L))
        active = (n_draws > 0) & (L_arr == int(L))
        if not active.any():
            continue
        ax_bind.scatter(
            T_arr[active],
            u4[active],
            s=(sizes[active] ** 2) * 0.35,
            c=style["color"],
            edgecolors="white",
            linewidths=0.6,
            alpha=0.9,
            label=str(style["label"]),
            zorder=2,
        )

    if not frame.ranking.empty:
        cand_idx = frame.ranking["row_index"].astype(int).to_numpy()
        cand_idx = cand_idx[n_draws[cand_idx] < max_draws]
        if cand_idx.size:
            ax_bind.scatter(
                T_arr[cand_idx],
                u4[cand_idx],
                s=(sizes[cand_idx] ** 2) * 0.35,
                c=_CANDIDATE_FLAT_COLOR,
                edgecolors="white",
                linewidths=0.4,
                alpha=0.45,
                label="candidates",
                zorder=1,
            )

    next_row = full_df.iloc[frame.next_index]
    ax_bind.scatter(
        [float(next_row["T"])],
        [float(next_row["binder_cumulant"])],
        s=220,
        marker="*",
        c="#e6a817",
        edgecolors="black",
        linewidths=0.8,
        label="chosen +10k",
        zorder=3,
    )
    ax_bind.axvline(TC_EXACT, color="red", ls="-.", lw=1.0, alpha=0.8)
    ax_bind.set_title("Binder grid")
    ax_bind.set_xlabel(r"$T$")
    ax_bind.set_ylabel(r"$U_4$")
    ax_bind.set_xlim(*T_lim)
    ax_bind.set_ylim(*u4_lim)
    ax_bind.legend(loc="best", fontsize=7, frameon=False)

    # --- Binder GP ---
    panel = frame.binder_gp_panel
    for L in panel.L_values:
        style = _neff_style(int(L))
        color = str(style["color"])
        mask = np.isclose(panel.L, float(L)) if panel.L.size else np.array([], dtype=bool)
        if mask.size and mask.any():
            ax_gp.scatter(
                panel.T[mask],
                panel.U[mask],
                s=28,
                c=color,
                edgecolors="white",
                linewidths=0.5,
                zorder=3,
                label=f"budget L={int(L)}",
            )
        T_curve = panel.curve_T.get(int(L))
        u_c = panel.curve_U.get(int(L))
        u_std = panel.curve_U_std.get(int(L))
        if (
            T_curve is not None
            and u_c is not None
            and u_std is not None
            and T_curve.size
            and u_c.size == T_curve.size
            and u_std.size == T_curve.size
        ):
            ax_gp.fill_between(
                T_curve,
                u_c - u_std,
                u_c + u_std,
                color=color,
                alpha=0.2,
                linewidth=0,
                zorder=1,
            )
            ax_gp.plot(T_curve, u_c, color=color, lw=1.8, zorder=2, label=rf"$\hat U$ L={int(L)}")
        T_ex = panel.exact_curve_T.get(int(L))
        U_ex = panel.exact_curve_U.get(int(L))
        if T_ex is not None and U_ex is not None and T_ex.size and U_ex.size == T_ex.size:
            ax_gp.plot(
                T_ex,
                U_ex,
                color=color,
                lw=1.4,
                ls="--",
                alpha=0.85,
                zorder=2,
                label=f"exact L={int(L)}",
            )

    tc = float(frame.mean_tc)
    tc_std = float(frame.tc_std)
    if np.isfinite(tc_std) and tc_std > 0.0:
        ax_gp.axvspan(tc - tc_std, tc + tc_std, color="0.55", alpha=0.18, zorder=0)
    ax_gp.axvline(tc, color="0.25", ls=":", lw=1.4, zorder=1, label=r"$\langle T_c\rangle\pm1\sigma$")
    ax_gp.axvline(TC_EXACT, color="red", ls="-.", lw=1.0, alpha=0.8)
    ax_gp.set_title(r"Binder GP $\hat U_4(T)$ (budgeted pts)")
    ax_gp.set_xlabel(r"$T$")
    ax_gp.set_ylabel(r"$U_4$")
    ax_gp.set_xlim(*T_lim)
    ax_gp.set_ylim(*u4_lim)
    ax_gp.legend(loc="best", fontsize=7, ncol=2, frameon=False)

    meta = _load_run_meta(run_dir)
    selection = str(meta.get("selection", "budget-AL"))
    post_tag = "LAPS" if frame.has_posterior else "prior"
    ax_heat.figure.suptitle(
        (
            f"iter {frame.iteration:02d} ({selection}, {post_tag}): "
            f"{frame.n_points} pts / {frame.total_draws} draws, "
            f"next idx={frame.next_index}, Δtr(cov)={frame.next_delta_tr_cov:.4g}, "
            f"tr(cov)={frame.tr_cov:.4g}"
        ),
        fontsize=11,
    )


def export_budget_stepper_gif(
    run_dir: Path,
    out_path: Path | None = None,
    *,
    max_frames: int = 50,
    n_bins: int = 120,
    fps: int = 10,
    dpi: int = 130,
) -> Path:
    """Write a GIF of the first ``max_frames`` budget-AL steps.

    Each frame shows the joint ``(T_c, ν)`` heatmap, Binder grid, and Binder GP
    panel (the three views from the interactive stepper, omitting τ_int).
    Default ``n_bins=120`` gives a finer heatmap than the interactive stepper.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    run_dir = Path(run_dir)
    if out_path is None:
        out_path = run_dir / f"evolution_stepper_first{int(max_frames)}.gif"
    out_path = Path(out_path)

    meta = _load_run_meta(run_dir)
    chunk_size = int(meta.get("chunk_size", 10_000))
    n_chunks_max = int(meta.get("n_chunks_max", 8))

    print(
        f"Building up to {max_frames} stepper frames from {run_dir} …",
        flush=True,
    )
    full_df, frames, tc_lim, nu_lim = build_budget_stepper_frames(
        run_dir,
        n_bins=n_bins,
        max_frames=max_frames,
        # Zoom onto the posterior mass so bins resolve the blob, not empty prior tails.
        heatmap_percentiles=(15.0, 85.0),
        heatmap_pad_frac=0.35,
        heatmap_posterior_only_limits=True,
    )
    if not frames:
        raise ValueError(f"No frames available under {run_dir}")
    print(f"Rendering GIF with {len(frames)} frames → {out_path}", flush=True)
    print(
        f"Heatmap window: T_c∈[{tc_lim[0]:.4f},{tc_lim[1]:.4f}], "
        f"ν∈[{nu_lim[0]:.4f},{nu_lim[1]:.4f}]",
        flush=True,
    )

    T_arr = full_df["T"].to_numpy(dtype=np.float64)
    u4 = full_df["binder_cumulant"].to_numpy(dtype=np.float64)
    T_pad = 0.02 * (float(T_arr.max()) - float(T_arr.min()) + 1e-9)
    u_pad = 0.05 * (float(u4.max()) - float(u4.min()) + 1e-9)
    T_lim = (float(T_arr.min()) - T_pad, float(T_arr.max()) + T_pad)
    u4_lim = (float(u4.min()) - u_pad, float(u4.max()) + u_pad)

    fig = plt.figure(figsize=(11.5, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0])
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_bind = fig.add_subplot(gs[0, 1])
    ax_gp = fig.add_subplot(gs[1, :])
    axes = (ax_heat, ax_bind, ax_gp)

    def _update(i: int) -> tuple[object, ...]:
        _draw_budget_stepper_gif_frame(
            axes,
            full_df,
            frames[i],
            tc_lim=tc_lim,
            nu_lim=nu_lim,
            chunk_size=chunk_size,
            n_chunks_max=n_chunks_max,
            u4_lim=u4_lim,
            T_lim=T_lim,
            run_dir=run_dir,
        )
        return axes

    anim = FuncAnimation(fig, _update, frames=len(frames), blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return out_path


def export_budget_stepper_html(
    run_dir: Path,
    out_path: Path,
    *,
    n_bins: int = 40,
    inline_plotlyjs: bool = False,
) -> Path:
    """Write a self-contained HTML viewer (arrow keys + slider)."""
    _, frames, _, _ = build_budget_stepper_frames(run_dir, n_bins=n_bins)
    fig = build_budget_stepper_figure(run_dir, n_bins=n_bins)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        out_path,
        div_id="budget-al-stepper",
        include_plotlyjs=True if inline_plotlyjs else "cdn",
        auto_open=False,
        post_script=budget_stepper_keyboard_post_script(frames),
    )
    return out_path


def export_budget_stepper_notebook(run_dir: Path, out_path: Path | None = None) -> Path:
    """Write a notebook that displays the budget-AL stepper."""
    run_dir = run_dir.resolve()
    if out_path is None:
        out_path = run_dir / "evolution_stepper.ipynb"
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Sample-budget active-learning stepper\n",
                    "\n",
                    "Interactive view of each budget-AL iteration:\n",
                    "- **Top-left:** joint `(T_c, ν)` heatmap — LAPS posterior when available, "
                    "else prior draws (iteration 0)\n",
                    "- **Top-middle:** Binder grid — marker size ∝ draws spent; gold star = next +10k "
                    "chunk\n",
                    "- **Top-right:** Binder integrated autocorrelation "
                    r"`τ_int ≈ N/(2 n_eff)` vs reduced temperature "
                    "`t=(T-T_c)/T_c` (all L overlaid; GP curves once the dynamical fit is live)\n",
                    "- **Bottom:** FSS Binder GP predictive mean "
                    r"`Û₄(t)` ± 1σ at the current mean `(T_c, ν)`, "
                    "one curve per L (trained on currently budgeted points)\n",
                    "\n",
                    "Use the **slider** or **Prev / Next** buttons under the figure.\n",
                    "If the right panel is empty for an older run, regenerate with "
                    "`--backfill-neff` once.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import sys\n",
                    "from pathlib import Path\n",
                    "\n",
                    "repo_root = Path.cwd()\n",
                    "while repo_root != repo_root.parent and not (repo_root / \"pyproject.toml\").exists():\n",
                    "    repo_root = repo_root.parent\n",
                    "if str(repo_root) not in sys.path:\n",
                    "    sys.path.insert(0, str(repo_root))\n",
                    "\n",
                    "from ising.plot_budget_al_stepper import (\n",
                    "    build_budget_stepper_figure,\n",
                    "    discover_completed_iterations,\n",
                    "    show_stepper_in_notebook,\n",
                    ")\n",
                    f"RUN_DIR = Path({str(run_dir)!r})\n",
                    "\n",
                    "assert discover_completed_iterations(RUN_DIR), (\n",
                    "    f\"No completed iters under {RUN_DIR} \"\n",
                    "    \"(need iter_XX/ranking.csv + n_draws.npy)\"\n",
                    ")\n",
                    "print(f\"{len(discover_completed_iterations(RUN_DIR))} completed iterations\")\n",
                    "RUN_DIR\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "fig = build_budget_stepper_figure(RUN_DIR, softmax_temperature_scale=0.1)\n",
                    "show_stepper_in_notebook(fig)\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from ising.plot_budget_al_stepper import export_budget_stepper_gif\n",
                    "\n",
                    "gif_path = export_budget_stepper_gif(\n",
                    "    RUN_DIR,\n",
                    "    max_frames=50,\n",
                    "    n_bins=120,\n",
                    "    fps=10,\n",
                    ")\n",
                    "print(gif_path)\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(nb, indent=1) + "\n")
    return out_path


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Budget AL output directory",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output HTML path (default: RUN_DIR/evolution_stepper.html)",
    )
    parser.add_argument("--bins", type=int, default=40, help="Heatmap bin count")
    parser.add_argument(
        "--inline-plotlyjs",
        action="store_true",
        help="Embed plotly.js in HTML (larger file; offline-friendly)",
    )
    parser.add_argument(
        "--notebook",
        type=Path,
        default=None,
        nargs="?",
        const=True,
        metavar="PATH",
        help="Also write a Jupyter notebook (default: RUN_DIR/evolution_stepper.ipynb)",
    )
    parser.add_argument(
        "--backfill-neff",
        action="store_true",
        help=(
            "Write missing iter_XX/neff_train.csv from raw sample prefixes "
            "(needed once for runs started before this artifact existed)"
        ),
    )
    parser.add_argument(
        "--gif",
        type=Path,
        default=None,
        nargs="?",
        const=True,
        metavar="PATH",
        help=(
            "Also write a GIF of the first --gif-frames steps "
            "(default: RUN_DIR/evolution_stepper_firstN.gif)"
        ),
    )
    parser.add_argument(
        "--gif-frames",
        type=int,
        default=50,
        help="Number of early iterations to include in the GIF (default: 50)",
    )
    parser.add_argument(
        "--gif-fps",
        type=int,
        default=10,
        help="GIF frames per second (default: 10)",
    )
    args = parser.parse_args(argv)
    if args.backfill_neff:
        n_written = backfill_neff_train_csvs(args.run_dir)
        print(f"Backfilled {n_written} neff_train.csv files", flush=True)
    out_path = args.out if args.out is not None else args.run_dir / "evolution_stepper.html"
    path = export_budget_stepper_html(
        args.run_dir,
        out_path,
        n_bins=args.bins,
        inline_plotlyjs=args.inline_plotlyjs,
    )
    print(f"Wrote {path}", flush=True)
    if args.notebook is not None:
        nb_out = (
            args.run_dir / "evolution_stepper.ipynb"
            if args.notebook is True
            else Path(args.notebook)
        )
        nb_path = export_budget_stepper_notebook(args.run_dir, nb_out)
        print(f"Wrote notebook {nb_path}", flush=True)
    if args.gif is not None:
        gif_out = (
            None
            if args.gif is True
            else Path(args.gif)
        )
        gif_path = export_budget_stepper_gif(
            args.run_dir,
            gif_out,
            max_frames=args.gif_frames,
            n_bins=args.bins,
            fps=args.gif_fps,
        )
        print(f"Wrote GIF {gif_path}", flush=True)
    return path


if __name__ == "__main__":
    main()
