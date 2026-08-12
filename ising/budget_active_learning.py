#!/usr/bin/env python3
"""Sample-budget active learning on harada_square stored chains.

Each iteration allocates ``chunk_size`` (default 10_000) draws at one ``(L, T)``
from ``samples_harada_square.npz``. Score-time Binder ESS comes from an online
dynamical n_eff GP fit on materialized prefixes (cold-start rate until enough
points); candidates are ranked by Δtr(cov) / (L/64)^2. After selection,
observables are materialized with real prefix mean/std/n_eff.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .active_learning import (
    load_posterior_draws,
    rank_sample_chunk_candidates,
    sample_prior_draws,
    score_sample_chunk,
)
from .datasets import ISING_DIR, observables_path, samples_path
from .explorer_inference import (
    WIDGET_LAPS_CHAINS,
    WIDGET_LAPS_DRAWS,
    WIDGET_LAPS_TUNE,
    posterior_heatmap_from_idata,
    run_widget_laps_inference,
    save_posterior_heatmap_png,
)
from .greedy_active_learning import build_al_config, select_ranked_point
from .jax_config import configure_jax, jax_device_summary
from .neff_scaling import (
    COST_L_REF,
    DEFAULT_COLD_START_NEFF_RATE,
    DEFAULT_ONLINE_MIN_POINTS,
    build_online_neff_planner,
    write_neff_fit_summary,
    write_neff_train_csv,
)
from .observables import validate_observables_df
from .plot_binder import save_binder_selection_png
from .samples import load_samples, observables_from_budget, observables_from_samples

SelectionStrategy = Literal["greedy", "worst"]

DEFAULT_CHUNK_SIZE = 10_000
DEFAULT_N_CHUNKS_MAX = 8
DEFAULT_OUT_DIR = ISING_DIR / "plots" / "budget_al_harada_square"
DEFAULT_WORST_OUT_DIR = ISING_DIR / "plots" / "worst_budget_al_harada_square"

_EMPTY_OBS_COLUMNS = [
    "L",
    "T",
    "magnetization",
    "magnetization_std",
    "n_eff",
    "log_magnetization",
    "log_magnetization_std",
    "n_eff_log",
    "susceptibility",
    "susceptibility_std",
    "n_eff_susceptibility",
    "log_susceptibility",
    "log_susceptibility_std",
    "n_eff_log_susceptibility",
    "binder_cumulant",
    "binder_cumulant_std",
    "n_eff_binder",
    "m2",
    "m2_std",
    "n_eff_m2",
    "m4",
    "m4_std",
    "n_eff_m4",
    "n_sweeps",
    "n_thermalize",
    "seed",
]


def default_out_dir(selection: SelectionStrategy) -> Path:
    if selection == "worst":
        return DEFAULT_WORST_OUT_DIR
    return DEFAULT_OUT_DIR


def prepare_binder_observables(df: pd.DataFrame) -> pd.DataFrame:
    """Validate/filter a full MC observables table for binder-only AL."""
    if df.empty:
        return pd.DataFrame(columns=_EMPTY_OBS_COLUMNS)

    validate_observables_df(
        df,
        use_m=False,
        use_m2=False,
        use_m4=False,
        use_binder=True,
        use_chi=False,
    )
    return df.reset_index(drop=True)


def load_full_reference(
    *,
    samples_file: Path | None = None,
    observables_csv: Path | None = None,
) -> tuple[object, pd.DataFrame]:
    """Load stored chains + full-chain binder table (std scale / grid; not ESS oracle)."""
    sample_path = samples_file or samples_path("harada_square")
    if not sample_path.is_file():
        raise FileNotFoundError(f"Raw samples not found: {sample_path}")
    stored = load_samples(sample_path)

    csv_path = observables_csv or observables_path("harada_square")
    if csv_path.is_file():
        full_ref = pd.read_csv(csv_path)
    else:
        full_ref = observables_from_samples(stored)
    full_ref = prepare_binder_observables(full_ref)
    if len(full_ref) != stored.n_points:
        raise ValueError(
            f"full_ref rows ({len(full_ref)}) != stored.n_points ({stored.n_points})"
        )
    # Align row order to stored sample grid.
    key = list(zip(stored.L.tolist(), np.round(stored.T, decimals=10).tolist()))
    ref_key = list(
        zip(
            full_ref["L"].astype(int).tolist(),
            np.round(full_ref["T"].to_numpy(dtype=np.float64), decimals=10).tolist(),
        )
    )
    if key != ref_key:
        # Reorder CSV rows to match NPZ order.
        order = []
        ref_map = {k: i for i, k in enumerate(ref_key)}
        for k in key:
            if k not in ref_map:
                raise ValueError(f"Missing (L,T)={k} in full reference observables")
            order.append(ref_map[k])
        full_ref = full_ref.iloc[order].reset_index(drop=True)
    return stored, full_ref


def materialize_budget(
    stored,
    n_draws: np.ndarray,
) -> pd.DataFrame:
    """Real prefix aggregates for the current budget vector."""
    return prepare_binder_observables(observables_from_budget(stored, n_draws))


def budget_included_mask(full_ref: pd.DataFrame, current: pd.DataFrame) -> np.ndarray:
    """Boolean mask over full_ref for points present in ``current``."""
    if current.empty:
        return np.zeros(len(full_ref), dtype=bool)
    cur_keys = {
        (int(r.L), float(np.round(r.T, decimals=10)))
        for r in current.itertuples(index=False)
    }
    mask = np.zeros(len(full_ref), dtype=bool)
    for i, row in full_ref.iterrows():
        key = (int(row["L"]), float(np.round(float(row["T"]), decimals=10)))
        mask[i] = key in cur_keys
    return mask


def _discover_iter_indices(out_dir: Path) -> list[int]:
    import re

    pat = re.compile(r"^iter_(\d+)$")
    out: list[int] = []
    if not out_dir.is_dir():
        return out
    for path in out_dir.iterdir():
        if not path.is_dir():
            continue
        m = pat.match(path.name)
        if m is None:
            continue
        if (path / "n_draws.npy").is_file():
            out.append(int(m.group(1)))
    return sorted(out)


def load_resume_state(
    out_dir: Path,
    n_points: int,
    *,
    chunk_size: int,
) -> tuple[np.ndarray, int, list[dict[str, object]]]:
    """Load ``n_draws``, next iteration index, and prior summary rows from ``out_dir``."""
    n_draws_path = out_dir / "n_draws_final.npy"
    if n_draws_path.is_file():
        n_draws = np.asarray(np.load(n_draws_path), dtype=np.int64).ravel()
    else:
        done = _discover_iter_indices(out_dir)
        if not done:
            raise FileNotFoundError(
                f"Cannot resume {out_dir}: no n_draws_final.npy or iter_*/n_draws.npy"
            )
        n_draws = np.asarray(
            np.load(out_dir / f"iter_{done[-1]:02d}" / "n_draws.npy"),
            dtype=np.int64,
        ).ravel()

    if n_draws.size != n_points:
        raise ValueError(
            f"Resume n_draws length {n_draws.size} != n_points {n_points}"
        )

    summary_path = out_dir / "summary.csv"
    summary_rows: list[dict[str, object]] = []
    if summary_path.is_file():
        summary_df = pd.read_csv(summary_path)
        if not summary_df.empty:
            summary_rows = summary_df.to_dict(orient="records")
            start_iteration = int(summary_df["iteration"].max()) + 1
        else:
            start_iteration = int(n_draws.sum() // max(chunk_size, 1))
    else:
        done = _discover_iter_indices(out_dir)
        start_iteration = (max(done) + 1) if done else int(n_draws.sum() // max(chunk_size, 1))

    return n_draws, start_iteration, summary_rows


def remaining_chunk_slots(
    n_draws: np.ndarray,
    *,
    chunk_size: int,
    n_chunks_max: int,
) -> int:
    """How many chunk allocations are still possible."""
    cap = chunk_size * n_chunks_max
    return int(np.sum((cap - n_draws) // chunk_size))


def parse_l_cycle(spec: str | None) -> tuple[int, ...] | None:
    """Parse ``--l-cycle`` like ``64,128,256`` into a non-empty int tuple."""
    if spec is None:
        return None
    text = str(spec).strip()
    if not text:
        return None
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError(f"Invalid --l-cycle {spec!r}: expected comma-separated ints")
    if any(v <= 0 for v in values):
        raise ValueError(f"Invalid --l-cycle {spec!r}: lattice sizes must be positive")
    return values


def load_l_cycle_index(out_dir: Path, *, l_cycle: tuple[int, ...] | None) -> int:
    """Resume cycle cursor from ``run_meta.json`` when present."""
    if l_cycle is None:
        return 0
    meta_path = out_dir / "run_meta.json"
    if not meta_path.is_file():
        return 0
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    raw = meta.get("l_cycle_index", 0)
    try:
        return int(raw) % len(l_cycle)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0


def run_budget_pipeline(
    *,
    out_dir: Path,
    n_iters: int = 40,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    n_chunks_max: int = DEFAULT_N_CHUNKS_MAX,
    seed: int = 1,
    max_is_draws: int = 500,
    save_netcdf: bool = True,
    selection: SelectionStrategy = "greedy",
    infer_Tc: bool = True,
    infer_nu: bool = True,
    infer_beta: bool = False,
    samples_file: Path | None = None,
    observables_csv: Path | None = None,
    resume: bool = False,
    until_exhausted: bool = False,
    l_cycle: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Run sample-budget active learning and write artifacts.

    Parameters
    ----------
    resume
        Continue from ``out_dir`` (``n_draws_final.npy`` + ``summary.csv``).
    until_exhausted
        Keep allocating until every grid point has ``n_chunks_max`` chunks
        (or ranking is empty). ``n_iters`` is then only a safety cap on
        *additional* steps (use a large value or leave at default).
    l_cycle
        Optional round-robin lattice sizes, e.g. ``(64, 128, 256)``. Each
        iteration must allocate at the next L in the cycle (skipping any L
        with no eligible candidates).
    """
    configure_jax()
    print(jax_device_summary(), flush=True)
    print(f"Selection strategy: {selection}", flush=True)
    if l_cycle is not None:
        print(f"L-cycle: {','.join(str(L) for L in l_cycle)}", flush=True)

    stored, full_ref = load_full_reference(
        samples_file=samples_file,
        observables_csv=observables_csv,
    )
    max_draws_available = int(stored.m_signed.shape[1])
    if chunk_size * n_chunks_max > max_draws_available:
        raise ValueError(
            f"chunk_size * n_chunks_max = {chunk_size * n_chunks_max} exceeds "
            f"stored chain length {max_draws_available}"
        )

    config = build_al_config()
    out_dir.mkdir(parents=True, exist_ok=True)

    if resume:
        n_draws, start_iteration, summary_rows = load_resume_state(
            out_dir, stored.n_points, chunk_size=chunk_size
        )
        print(
            f"Resuming from iteration {start_iteration}: "
            f"{int((n_draws > 0).sum())} points, {int(n_draws.sum())} draws spent, "
            f"{remaining_chunk_slots(n_draws, chunk_size=chunk_size, n_chunks_max=n_chunks_max)} "
            f"chunks remaining",
            flush=True,
        )
    else:
        n_draws = np.zeros(stored.n_points, dtype=np.int64)
        start_iteration = 0
        summary_rows = []

    l_cycle_index = load_l_cycle_index(out_dir, l_cycle=l_cycle) if resume else 0

    remaining_slots = remaining_chunk_slots(
        n_draws, chunk_size=chunk_size, n_chunks_max=n_chunks_max
    )
    if until_exhausted:
        max_additional = remaining_slots
        print(
            f"until-exhausted: will run up to {max_additional} more allocations",
            flush=True,
        )
    else:
        max_additional = int(n_iters)

    def write_run_meta() -> None:
        run_meta = {
            "selection": selection,
            "n_iters": n_iters,
            "chunk_size": chunk_size,
            "n_chunks_max": n_chunks_max,
            "seed": seed,
            "max_is_draws": max_is_draws,
            "dataset": "harada_square",
            "n_grid_points": int(stored.n_points),
            "score_noise": (
                "online dynamical n_eff GP on prefixes "
                f"(cold_start rate={DEFAULT_COLD_START_NEFF_RATE} until "
                f"{DEFAULT_ONLINE_MIN_POINTS} pts); sigma = std / sqrt(planned n_eff)"
            ),
            "cost_metric": f"(L/{COST_L_REF:g})^2",
            "rank_by": "delta_tr_cov_per_cost",
            "l_cycle": list(l_cycle) if l_cycle is not None else None,
            "l_cycle_index": int(l_cycle_index),
            "uniform_coverage": (
                "no point may receive chunk k+1 until every point has >= k chunks"
            ),
            "execute_noise": "recomputed prefix ESS",
            "resume": resume,
            "until_exhausted": until_exhausted,
            "start_iteration": start_iteration,
            "max_additional": max_additional,
        }
        (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2) + "\n")

    write_run_meta()

    if remaining_slots == 0:
        print("No remaining sample budget; nothing to do.", flush=True)
        summary = pd.DataFrame(summary_rows)
        if not summary.empty:
            summary.to_csv(out_dir / "summary.csv", index=False)
        return summary

    for step in range(max_additional):
        iteration = start_iteration + step
        iter_dir = out_dir / f"iter_{iteration:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        current = materialize_budget(stored, n_draws)
        n_points = len(current)
        total_draws = int(n_draws.sum())
        print(
            f"\n=== Iteration {iteration}: {n_points} points, "
            f"{total_draws} total draws "
            f"(step {step + 1}/{max_additional}) ===",
            flush=True,
        )

        t0 = time.perf_counter()
        if n_points == 0:
            draws = sample_prior_draws(
                max_is_draws,
                seed=seed + iteration,
                infer_Tc=infer_Tc,
                infer_nu=infer_nu,
                infer_beta=infer_beta,
            )
            idata = None
            hm = None
            laps_elapsed = 0.0
            posterior_tr_cov = float(
                np.trace(np.cov(draws.as_matrix(
                    infer_Tc=infer_Tc, infer_nu=infer_nu, infer_beta=infer_beta
                ), rowvar=False))
            )
            print(
                f"  empty dataset: scoring from {max_is_draws} prior draws "
                f"(tr(cov)={posterior_tr_cov:.5g})",
                flush=True,
            )
        else:
            idata = run_widget_laps_inference(
                current,
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
            draws = load_posterior_draws(
                idata,
                infer_Tc=infer_Tc,
                infer_nu=infer_nu,
                infer_beta=infer_beta,
                max_draws=max_is_draws,
                seed=seed + 1000 + iteration,
            )
            hm = posterior_heatmap_from_idata(
                idata,
                infer_Tc=infer_Tc,
                infer_nu=infer_nu,
                infer_beta=infer_beta,
            )
            posterior_tr_cov = float(hm.tr_cov)
            posterior_title = (
                f"Iter {iteration}: {n_points} pts / {total_draws} draws, "
                f"LAPS {WIDGET_LAPS_CHAINS} ch, tr(cov)={hm.tr_cov:.5g}"
            )
            save_posterior_heatmap_png(
                hm, iter_dir / "posterior.png", title=posterior_title
            )
            if save_netcdf:
                idata.to_netcdf(iter_dir / "posterior.nc")
                print(f"  saved {iter_dir / 'posterior.nc'}", flush=True)

        remaining = int(np.sum(n_draws < chunk_size * n_chunks_max))
        if remaining == 0:
            print("No remaining sample budget; stopping.", flush=True)
            break

        t_neff = time.perf_counter()
        neff_planner = build_online_neff_planner(
            current,
            channel="binder",
            min_points=DEFAULT_ONLINE_MIN_POINTS,
            cold_rate=DEFAULT_COLD_START_NEFF_RATE,
            seed=seed + 10_000 + iteration,
        )
        neff_fit_elapsed = time.perf_counter() - t_neff
        write_neff_train_csv(current, iter_dir / "neff_train.csv", channel="binder")
        if neff_planner.kind == "gp" and neff_planner.fit is not None:
            write_neff_fit_summary(
                neff_planner.fit, iter_dir / "neff_scaling_fit.txt"
            )
            print(
                f"  online n_eff GP: {neff_planner.n_train} pts, "
                f"z_dyn={neff_planner.fit.z_dyn_mean:.3f}±{neff_planner.fit.z_dyn_std:.3f} "
                f"({neff_fit_elapsed:.1f}s)",
                flush=True,
            )
        else:
            (iter_dir / "neff_scaling_fit.txt").write_text(
                f"kind=cold_start\n"
                f"n_train={neff_planner.n_train}\n"
                f"cold_rate={neff_planner.cold_rate:.6g}\n"
                f"min_points={DEFAULT_ONLINE_MIN_POINTS}\n",
                encoding="utf-8",
            )
            print(
                f"  online n_eff: cold_start "
                f"(n_train={neff_planner.n_train}<{DEFAULT_ONLINE_MIN_POINTS}, "
                f"rate={neff_planner.cold_rate:.3g})",
                flush=True,
            )

        t1 = time.perf_counter()
        target_L: int | None = None
        ranked = pd.DataFrame()
        if l_cycle is None:
            ranked = rank_sample_chunk_candidates(
                current,
                full_ref,
                n_draws,
                config,
                draws,
                neff_planner=neff_planner,
                chunk_size=chunk_size,
                n_chunks_max=n_chunks_max,
                L_ref=COST_L_REF,
                infer_Tc=infer_Tc,
                infer_nu=infer_nu,
                infer_beta=infer_beta,
            )
        else:
            # Round-robin by L; skip any size with no eligible candidates.
            for offset in range(len(l_cycle)):
                candidate_L = int(l_cycle[(l_cycle_index + offset) % len(l_cycle)])
                ranked_L = rank_sample_chunk_candidates(
                    current,
                    full_ref,
                    n_draws,
                    config,
                    draws,
                    neff_planner=neff_planner,
                    chunk_size=chunk_size,
                    n_chunks_max=n_chunks_max,
                    L_ref=COST_L_REF,
                    allowed_L=candidate_L,
                    infer_Tc=infer_Tc,
                    infer_nu=infer_nu,
                    infer_beta=infer_beta,
                )
                if ranked_L.empty:
                    print(
                        f"  L-cycle: no eligible candidates for L={candidate_L}; skipping",
                        flush=True,
                    )
                    continue
                target_L = candidate_L
                ranked = ranked_L
                l_cycle_index = (l_cycle_index + offset + 1) % len(l_cycle)
                write_run_meta()
                break
        rank_elapsed = time.perf_counter() - t1
        ranked.to_csv(iter_dir / "ranking.csv", index=False)

        if ranked.empty:
            print("Empty ranking; stopping.", flush=True)
            break

        best = ranked.iloc[0]
        # Ranking is already sorted by delta_tr_cov_per_cost (cost-aware).
        chosen = select_ranked_point(ranked, selection)
        chosen_idx = int(chosen["row_index"])
        choice_label = "best" if selection == "greedy" else "worst"
        if target_L is not None:
            print(f"  L-cycle target L={target_L}", flush=True)

        preview = score_sample_chunk(
            current,
            config,
            draws,
            full_ref.iloc[chosen_idx],
            row_index=chosen_idx,
            chunks_before=int(chosen["chunks_before"]),
            n_chunks_max=n_chunks_max,
            chunk_size=chunk_size,
            neff_planner=neff_planner,
            infer_Tc=infer_Tc,
            infer_nu=infer_nu,
            infer_beta=infer_beta,
        )
        print(
            f"  choose ({choice_label}) idx={chosen_idx} "
            f"L={int(chosen['L'])} T={float(chosen['T']):.4f} "
            f"chunks {int(chosen['chunks_before'])}->{int(chosen['chunks_before']) + 1} "
            f"Δtr(cov)={float(chosen['delta_tr_cov']):.5g} "
            f"Δ/cost={float(chosen['delta_tr_cov_per_cost']):.5g} "
            f"cost={float(chosen['cost']):.3g} "
            f"ESS={float(chosen['ess']):.0f} "
            f"(preview Δ={preview.delta_uncertainty:.5g})",
            flush=True,
        )

        included_mask = budget_included_mask(full_ref, current)
        binder_title = (
            f"Iter {iteration} ({selection}): {n_points} pts, "
            f"next L={int(chosen['L'])} T={float(chosen['T']):.4f} "
            f"(+{chunk_size} draws, idx={chosen_idx})"
        )
        save_binder_selection_png(
            full_ref,
            included_mask,
            iter_dir / "binder.png",
            next_index=chosen_idx,
            title=binder_title,
        )

        n_draws[chosen_idx] += int(chunk_size)
        np.save(iter_dir / "n_draws.npy", n_draws)
        pd.DataFrame(
            {
                "row_index": np.arange(len(n_draws)),
                "L": full_ref["L"].to_numpy(dtype=np.int64),
                "T": full_ref["T"].to_numpy(dtype=np.float64),
                "n_draws": n_draws,
            }
        ).to_csv(iter_dir / "budget.csv", index=False)

        after = materialize_budget(stored, n_draws)
        after_row = after[
            (after["L"].astype(int) == int(chosen["L"]))
            & np.isclose(after["T"].to_numpy(dtype=np.float64), float(chosen["T"]))
        ]
        actual_n_eff = (
            float(after_row.iloc[0]["n_eff_binder"]) if len(after_row) else float("nan")
        )

        total_elapsed = time.perf_counter() - t0
        rank_best = float(best["delta_tr_cov_per_cost"])
        rank_worst = float(ranked.iloc[-1]["delta_tr_cov_per_cost"])
        summary_rows.append(
            {
                "iteration": iteration,
                "selection": selection,
                "n_points": n_points,
                "total_draws_before": total_draws,
                "added_row_index": chosen_idx,
                "added_L": int(chosen["L"]),
                "added_T": float(chosen["T"]),
                "target_L": target_L,
                "l_cycle_index_after": int(l_cycle_index) if l_cycle is not None else None,
                "chunks_before": int(chosen["chunks_before"]),
                "chunks_after": int(chosen["chunks_before"]) + 1,
                "n_draws_after_point": int(n_draws[chosen_idx]),
                "delta_tr_cov": float(chosen["delta_tr_cov"]),
                "cost": float(chosen["cost"]),
                "delta_tr_cov_per_cost": float(chosen["delta_tr_cov_per_cost"]),
                "rank_best_delta_tr_cov_per_cost": rank_best,
                "rank_worst_delta_tr_cov_per_cost": rank_worst,
                "rank_delta_tr_cov_per_cost_spread": rank_best - rank_worst,
                "tr_cov_before": float(chosen["tr_cov_before"]),
                "tr_cov_after": float(chosen["tr_cov_after"]),
                "ess": float(chosen["ess"]),
                "pareto_k": float(chosen["pareto_k"]),
                "neff_planner_kind": neff_planner.kind,
                "predicted_n_eff_full": float(chosen["predicted_n_eff_full"]),
                "planned_n_eff_after": float(chosen["planned_n_eff_after"]),
                "actual_n_eff_after": actual_n_eff,
                "neff_fit_elapsed_s": float(neff_fit_elapsed),
                "posterior_tr_cov": posterior_tr_cov,
                "laps_elapsed_s": float(laps_elapsed),
                "rank_elapsed_s": float(rank_elapsed),
                "total_elapsed_s": float(total_elapsed),
            }
        )
        pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)
        np.save(out_dir / "n_draws_final.npy", n_draws)
        print(
            f"  allocated +{chunk_size} -> n_draws={int(n_draws[chosen_idx])}, "
            f"actual n_eff_binder={actual_n_eff:.1f} "
            f"(planned={float(chosen['planned_n_eff_after']):.1f}) "
            f"({total_elapsed:.1f}s total)",
            flush=True,
        )

    summary = pd.DataFrame(summary_rows)
    print(f"\nWrote summary to {out_dir / 'summary.csv'}", flush=True)
    return summary


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--n-iters",
        type=int,
        default=40,
        help="Additional AL iterations (ignored as a stop reason when --until-exhausted)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from out_dir (n_draws_final.npy + summary.csv)",
    )
    parser.add_argument(
        "--until-exhausted",
        action="store_true",
        help="Keep allocating until every (L,T) has n_chunks_max chunks",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Draws allocated per iteration at the chosen (L,T)",
    )
    parser.add_argument(
        "--n-chunks-max",
        type=int,
        default=DEFAULT_N_CHUNKS_MAX,
        help="Max chunks per grid point (80k/10k = 8)",
    )
    parser.add_argument(
        "--selection",
        choices=("greedy", "worst"),
        default="greedy",
        help="greedy: max Δ tr(cov); worst: min Δ tr(cov)",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--max-is-draws",
        type=int,
        default=500,
        help="Prior/posterior draws subsampled for IS scoring",
    )
    parser.add_argument(
        "--save-netcdf",
        action="store_true",
        default=True,
        help="Save LAPS posterior as iter_XX/posterior.nc when data exists",
    )
    parser.add_argument(
        "--no-save-netcdf",
        action="store_false",
        dest="save_netcdf",
    )
    parser.add_argument("--infer-Tc", action="store_true", default=True)
    parser.add_argument("--no-infer-Tc", action="store_false", dest="infer_Tc")
    parser.add_argument("--infer-nu", action="store_true", default=True)
    parser.add_argument("--no-infer-nu", action="store_false", dest="infer_nu")
    parser.add_argument("--infer-beta", action="store_true", default=False)
    parser.add_argument(
        "--samples",
        type=Path,
        default=None,
        help="Override path to samples_harada_square.npz",
    )
    parser.add_argument(
        "--observables",
        type=Path,
        default=None,
        help="Override full-chain observables CSV (n_eff_full reference)",
    )
    parser.add_argument(
        "--l-cycle",
        type=str,
        default=None,
        help=(
            "Round-robin lattice sizes, e.g. '64,128,256'. Each iteration "
            "allocates at the next L (skipping sizes with no eligible candidates)."
        ),
    )
    args = parser.parse_args(argv)
    out_dir = args.out if args.out is not None else default_out_dir(args.selection)
    l_cycle = parse_l_cycle(args.l_cycle)

    run_budget_pipeline(
        out_dir=out_dir,
        n_iters=args.n_iters,
        chunk_size=args.chunk_size,
        n_chunks_max=args.n_chunks_max,
        seed=args.seed,
        max_is_draws=args.max_is_draws,
        save_netcdf=args.save_netcdf,
        selection=args.selection,
        infer_Tc=args.infer_Tc,
        infer_nu=args.infer_nu,
        infer_beta=args.infer_beta,
        samples_file=args.samples,
        observables_csv=args.observables,
        resume=args.resume,
        until_exhausted=args.until_exhausted,
        l_cycle=l_cycle,
    )
    return out_dir


if __name__ == "__main__":
    main()
