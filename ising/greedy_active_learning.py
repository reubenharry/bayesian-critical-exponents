#!/usr/bin/env python3
"""Greedy / worst-case active learning on harada_bsa Binder points."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from ._deps import az
from .active_learning import (
    counterfactual_include_point,
    load_posterior_draws,
)
from .constants import TC_EXACT
from .datasets import ISING_DIR, observables_path
from .explorer_inference import (
    WIDGET_LAPS_CHAINS,
    WIDGET_LAPS_DRAWS,
    WIDGET_LAPS_TUNE,
    posterior_heatmap_from_idata,
    run_widget_laps_inference,
    save_posterior_heatmap_png,
    save_weighted_preview_png,
)
from .jax_config import configure_jax, jax_device_summary
from .observables import ensure_harada_bsa_observables_csv, read_observables_for_fss
from .plot_binder import save_binder_selection_png
from .profile_likelihood import FssProfileConfig, GP_ETA

SelectionStrategy = Literal["greedy", "worst"]

DEFAULT_OUT_DIR = ISING_DIR / "plots" / "greedy_al_harada_bsa"
DEFAULT_WORST_OUT_DIR = ISING_DIR / "plots" / "worst_al_harada_bsa"
HARADA_BSA_L_VALUES = (64, 128, 256)


def build_al_config() -> FssProfileConfig:
    """Binder-only FSS config: no correction, RBF (gaussian), ell=0.15."""
    return FssProfileConfig(
        use_m=False,
        use_m2=False,
        use_m4=False,
        use_binder=True,
        use_chi=False,
        correction_m=False,
        correction_m2=False,
        correction_m4=False,
        correction_binder=False,
        correction_chi=False,
        use_log_m=False,
        gp_ell_factor=0.15,
        gp_eta=GP_ETA,
        gp_kernel="gaussian",
        obs_sigma_scale=1.0,
        infer_Tc=True,
        infer_nu=True,
        infer_beta=False,
    )


def select_initial_subset(
    df: pd.DataFrame,
    n: int = 10,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Stratified mask: ~n/3 per L, preferring T near T_c."""
    if n <= 0 or n > len(df):
        raise ValueError(f"n must be in 1..{len(df)}, got {n}")

    rng = np.random.default_rng(seed)
    mask = np.zeros(len(df), dtype=bool)
    L_values = sorted(int(L) for L in df["L"].unique())
    if not L_values:
        raise ValueError("observables dataframe has no L column")

    base = n // len(L_values)
    remainder = n % len(L_values)
    counts = {L: base for L in L_values}
    for L in rng.permutation(L_values)[:remainder]:
        counts[int(L)] += 1

    for L in L_values:
        L_mask = df["L"].to_numpy(dtype=np.int64) == L
        idx = np.flatnonzero(L_mask)
        if idx.size == 0:
            continue
        T = df.iloc[idx]["T"].to_numpy(dtype=np.float64)
        order = np.argsort(np.abs(T - TC_EXACT))
        take = min(counts.get(int(L), 0), order.size)
        mask[idx[order[:take]]] = True

    if int(mask.sum()) < n:
        remaining = np.flatnonzero(~mask)
        need = n - int(mask.sum())
        if remaining.size < need:
            raise ValueError("could not select enough initial points")
        extra = rng.choice(remaining, size=need, replace=False)
        mask[extra] = True

    return mask


def rank_excluded_points(
    included_df: pd.DataFrame,
    full_df: pd.DataFrame,
    included_mask: np.ndarray,
    config: FssProfileConfig,
    idata: az.InferenceData,
    *,
    infer_Tc: bool = True,
    infer_nu: bool = True,
    infer_beta: bool = False,
    max_is_draws: int | None = 500,
    seed: int = 0,
) -> pd.DataFrame:
    """Score every excluded row by IS counterfactual Δ tr(cov)."""
    draws = load_posterior_draws(
        idata,
        infer_Tc=infer_Tc,
        infer_nu=infer_nu,
        infer_beta=infer_beta,
        max_draws=max_is_draws,
        seed=seed,
    )
    excluded_idx = np.flatnonzero(~included_mask)
    rows: list[dict[str, object]] = []
    for j, idx in enumerate(excluded_idx):
        row = full_df.iloc[int(idx)]
        result = counterfactual_include_point(
            included_df,
            config,
            draws,
            row,
            row_index=int(idx),
            infer_Tc=infer_Tc,
            infer_nu=infer_nu,
            infer_beta=infer_beta,
        )
        rows.append(
            {
                "row_index": int(idx),
                "L": int(result.L),
                "T": float(result.T),
                "delta_tr_cov": float(result.delta_tr_cov),
                "tr_cov_before": float(result.tr_cov_before),
                "tr_cov_after": float(result.tr_cov_after),
                "ess": float(result.ess),
                "pareto_k": float(result.pareto_k),
            }
        )
        if (j + 1) % 10 == 0:
            print(f"  scored {j + 1}/{excluded_idx.size} excluded points", flush=True)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("delta_tr_cov", ascending=False).reset_index(drop=True)


def select_ranked_point(ranked: pd.DataFrame, selection: SelectionStrategy) -> pd.Series:
    """Pick the next point from a Δ tr(cov)-sorted ranking table."""
    if ranked.empty:
        raise ValueError("cannot select from empty ranking table")
    if selection == "greedy":
        return ranked.iloc[0]
    if selection == "worst":
        return ranked.iloc[-1]
    raise ValueError(f"Unknown selection strategy: {selection!r}")


def default_out_dir(selection: SelectionStrategy) -> Path:
    if selection == "worst":
        return DEFAULT_WORST_OUT_DIR
    return DEFAULT_OUT_DIR


def run_greedy_pipeline(
    *,
    out_dir: Path,
    n_init: int = 10,
    n_iters: int = 50,
    seed: int = 1,
    max_is_draws: int = 500,
    save_netcdf: bool = True,
    selection: SelectionStrategy = "greedy",
    infer_Tc: bool = True,
    infer_nu: bool = True,
    infer_beta: bool = False,
) -> pd.DataFrame:
    """Run point-addition active learning and write artifacts."""
    configure_jax()
    print(jax_device_summary(), flush=True)
    print(f"Selection strategy: {selection}", flush=True)

    data_path = ensure_harada_bsa_observables_csv(observables_path("harada_bsa"))
    full_df = read_observables_for_fss(
        data_path,
        use_m=False,
        use_m2=False,
        use_m4=False,
        use_binder=True,
        use_chi=False,
    )
    config = build_al_config()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_meta = {
        "selection": selection,
        "n_init": n_init,
        "n_iters": n_iters,
        "seed": seed,
        "max_is_draws": max_is_draws,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2) + "\n")

    included_mask = select_initial_subset(full_df, n=n_init, seed=seed)
    init_log = full_df.copy()
    init_log["included"] = included_mask
    init_log.to_csv(out_dir / "init_mask.csv", index=True)
    print(f"Initial subset: {int(included_mask.sum())} / {len(full_df)} points", flush=True)

    summary_rows: list[dict[str, object]] = []

    for iteration in range(n_iters):
        iter_dir = out_dir / f"iter_{iteration:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        n_included = int(included_mask.sum())
        included_df = full_df.loc[included_mask].reset_index(drop=True)
        print(f"\n=== Iteration {iteration}: {n_included} included points ===", flush=True)

        t0 = time.perf_counter()
        idata = run_widget_laps_inference(
            included_df,
            config,
            infer_Tc=infer_Tc,
            infer_nu=infer_nu,
            infer_beta=infer_beta,
            chains=WIDGET_LAPS_CHAINS,
            tune=WIDGET_LAPS_TUNE,
            draws=WIDGET_LAPS_DRAWS,
            seed=seed + iteration,
        )
        laps_elapsed = time.perf_counter() - t0

        hm = posterior_heatmap_from_idata(
            idata,
            infer_Tc=infer_Tc,
            infer_nu=infer_nu,
            infer_beta=infer_beta,
        )
        posterior_title = (
            f"Iter {iteration}: {n_included} pts, "
            f"LAPS {WIDGET_LAPS_CHAINS} ch, tr(cov)={hm.tr_cov:.5g}"
        )
        save_posterior_heatmap_png(hm, iter_dir / "posterior.png", title=posterior_title)
        if save_netcdf:
            idata.to_netcdf(iter_dir / "posterior.nc")
            print(f"  saved {iter_dir / 'posterior.nc'}", flush=True)

        excluded_count = int((~included_mask).sum())
        if excluded_count == 0:
            print("No excluded points left; stopping early.", flush=True)
            break

        t1 = time.perf_counter()
        ranked = rank_excluded_points(
            included_df,
            full_df,
            included_mask,
            config,
            idata,
            infer_Tc=infer_Tc,
            infer_nu=infer_nu,
            infer_beta=infer_beta,
            max_is_draws=max_is_draws,
            seed=seed + 1000 + iteration,
        )
        rank_elapsed = time.perf_counter() - t1
        ranked.to_csv(iter_dir / "ranking.csv", index=False)

        best = ranked.iloc[0]
        chosen = select_ranked_point(ranked, selection)
        chosen_idx = int(chosen["row_index"])
        row = full_df.iloc[chosen_idx]
        draws = load_posterior_draws(
            idata,
            infer_Tc=infer_Tc,
            infer_nu=infer_nu,
            infer_beta=infer_beta,
            max_draws=max_is_draws,
            seed=seed + 2000 + iteration,
        )
        preview = counterfactual_include_point(
            included_df,
            config,
            draws,
            row,
            row_index=chosen_idx,
            infer_Tc=infer_Tc,
            infer_nu=infer_nu,
            infer_beta=infer_beta,
        )
        choice_label = "best" if selection == "greedy" else "worst"
        preview_title = (
            f"If added ({choice_label}): L={preview.L}, T={preview.T:.4f}, "
            f"Δtr(cov)={preview.delta_tr_cov:.5g}, ESS={preview.ess:.0f}"
        )
        save_weighted_preview_png(hm, preview, iter_dir / "preview_add.png", title=preview_title)
        binder_title = (
            f"Iter {iteration} ({selection}): {n_included} included, "
            f"next L={int(chosen['L'])} T={float(chosen['T']):.4f} (idx={chosen_idx})"
        )
        save_binder_selection_png(
            full_df,
            included_mask,
            iter_dir / "binder.png",
            next_index=chosen_idx,
            title=binder_title,
        )

        included_mask[chosen_idx] = True
        total_elapsed = time.perf_counter() - t0
        rank_best = float(best["delta_tr_cov"])
        rank_worst = float(ranked.iloc[-1]["delta_tr_cov"])
        summary_rows.append(
            {
                "iteration": iteration,
                "selection": selection,
                "n_included": n_included,
                "added_row_index": chosen_idx,
                "added_L": int(chosen["L"]),
                "added_T": float(chosen["T"]),
                "delta_tr_cov": float(chosen["delta_tr_cov"]),
                "rank_best_delta_tr_cov": rank_best,
                "rank_worst_delta_tr_cov": rank_worst,
                "rank_delta_tr_cov_spread": rank_best - rank_worst,
                "tr_cov_before": float(chosen["tr_cov_before"]),
                "tr_cov_after": float(chosen["tr_cov_after"]),
                "ess": float(chosen["ess"]),
                "pareto_k": float(chosen["pareto_k"]),
                "posterior_tr_cov": float(hm.tr_cov),
                "laps_elapsed_s": float(laps_elapsed),
                "rank_elapsed_s": float(rank_elapsed),
                "total_elapsed_s": float(total_elapsed),
            }
        )
        print(
            f"Added ({selection}) idx={chosen_idx} L={int(chosen['L'])} T={float(chosen['T']):.4f} "
            f"Δtr(cov)={float(chosen['delta_tr_cov']):.5g} "
            f"[best={rank_best:.5g}, worst={rank_worst:.5g}] "
            f"ESS={float(chosen['ess']):.0f} ({total_elapsed:.1f}s total)",
            flush=True,
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"\nWrote summary to {out_dir / 'summary.csv'}", flush=True)
    return summary


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: greedy_al_harada_bsa or worst_al_harada_bsa)",
    )
    parser.add_argument("--n-init", type=int, default=10, help="Initial included points")
    parser.add_argument("--n-iters", type=int, default=50, help="Active-learning iterations")
    parser.add_argument(
        "--selection",
        choices=("greedy", "worst"),
        default="greedy",
        help="greedy: max Δ tr(cov); worst: min Δ tr(cov) each step",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--max-is-draws",
        type=int,
        default=500,
        help="Posterior draws subsampled for IS scoring",
    )
    parser.add_argument(
        "--save-netcdf",
        action="store_true",
        default=True,
        help="Save full LAPS posterior as iter_XX/posterior.nc (default: on)",
    )
    parser.add_argument(
        "--no-save-netcdf",
        action="store_false",
        dest="save_netcdf",
        help="Skip saving posterior.nc each iteration",
    )
    parser.add_argument("--infer-Tc", action="store_true", default=True)
    parser.add_argument("--no-infer-Tc", action="store_false", dest="infer_Tc")
    parser.add_argument("--infer-nu", action="store_true", default=True)
    parser.add_argument("--no-infer-nu", action="store_false", dest="infer_nu")
    parser.add_argument("--infer-beta", action="store_true", default=False)
    args = parser.parse_args(argv)
    out_dir = args.out if args.out is not None else default_out_dir(args.selection)

    run_greedy_pipeline(
        out_dir=out_dir,
        n_init=args.n_init,
        n_iters=args.n_iters,
        seed=args.seed,
        max_is_draws=args.max_is_draws,
        save_netcdf=args.save_netcdf,
        selection=args.selection,
        infer_Tc=args.infer_Tc,
        infer_nu=args.infer_nu,
        infer_beta=args.infer_beta,
    )
    return out_dir


if __name__ == "__main__":
    main()
