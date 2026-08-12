#!/usr/bin/env python3
"""Run ``n_chains`` independent φ⁴ HMC chains per (L, λ) and compute R-hat on |m|.

Writes (under ``phi4/data/``; zarr stores are gitignored):
  - ``samples_<dataset>_nchains.zarr/`` — per-point groups ``L*_T*`` with
    chunked ``m_signed`` ``(n_chains, n_draws)`` for streaming / resume
  - ``rhat_<dataset>.csv`` — per-point R-hat / ESS diagnostics

Use a single process on one GPU (``--jobs 1``). For R-hat the chains need only
be independent, not wall-clock concurrent.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from phi4.datasets import (
    DATA_DIR,
    get_dataset,
    observables_path,
)
from phi4.sampler import run_phi4

_DRAW_CHUNK = 8192


@dataclass(frozen=True)
class _ChainTask:
    L: int
    lam: float
    chain_id: int
    seed: int
    n_thermalize: int
    n_sweeps: int


@dataclass(frozen=True)
class _PointTask:
    L: int
    lam: float
    base_seed: int
    n_chains: int
    n_thermalize: int
    n_sweeps: int


def _rhat_csv_path(dataset: str) -> Path:
    return DATA_DIR / f"rhat_{dataset}.csv"


def _samples_zarr_path(dataset: str) -> Path:
    return DATA_DIR / f"samples_{dataset}_nchains.zarr"


def _point_key(L: int, lam: float) -> str:
    return f"L{int(L)}_T{float(lam):.4f}"


def _open_samples_store(path: Path, *, mode: str = "a") -> zarr.Group:
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "w" and path.exists():
        shutil.rmtree(path)
    return zarr.open_group(path, mode=mode)


def _run_one_chain(task: _ChainTask) -> tuple[int, np.ndarray]:
    """Return ``(chain_id, m_signed)`` for one independent HMC chain."""
    _, raw = run_phi4(
        task.L,
        task.lam,
        n_thermalize=task.n_thermalize,
        n_sweeps=task.n_sweeps,
        seed=task.seed,
        store_samples=True,
    )
    return task.chain_id, np.asarray(raw.m_signed, dtype=np.float64).ravel()


def _diagnostics_row(
    task: _PointTask, stacked: np.ndarray
) -> dict[str, float | int]:
    from blackjax.diagnostics import effective_sample_size, potential_scale_reduction

    m_abs = np.abs(stacked)
    rhat = float(np.asarray(potential_scale_reduction(m_abs)).item())
    ess = float(np.asarray(effective_sample_size(m_abs)).item())
    ess_per_chain = [
        float(np.asarray(effective_sample_size(m_abs[c : c + 1])).item())
        for c in range(task.n_chains)
    ]
    means = [float(np.mean(np.abs(ch))) for ch in stacked]
    row: dict[str, float | int] = {
        "L": int(task.L),
        "T": float(task.lam),
        "n_chains": int(task.n_chains),
        "n_sweeps": int(task.n_sweeps),
        "n_thermalize": int(task.n_thermalize),
        "seed": int(task.base_seed),
        "rhat_abs_m": rhat,
        "ess_abs_m": ess,
        "ess_abs_m_min_chain": float(min(ess_per_chain)),
        "mean_abs_m": float(np.mean(means)),
        "mean_abs_m_chain_std": float(np.std(means, ddof=1)) if task.n_chains > 1 else 0.0,
    }
    for c, (mu, e) in enumerate(zip(means, ess_per_chain)):
        row[f"mean_abs_m_c{c}"] = mu
        row[f"ess_abs_m_c{c}"] = e
    return row


def _run_point(task: _PointTask) -> tuple[dict[str, float | int], np.ndarray]:
    """Run ``n_chains`` sequential independent chains for one (L, λ)."""
    chains: list[np.ndarray] = [np.array([])] * task.n_chains
    for c in range(task.n_chains):
        chain_id, m = _run_one_chain(
            _ChainTask(
                L=task.L,
                lam=task.lam,
                chain_id=c,
                seed=task.base_seed + 10_000 * c,
                n_thermalize=task.n_thermalize,
                n_sweeps=task.n_sweeps,
            )
        )
        chains[chain_id] = m
    stacked = np.stack(chains, axis=0)  # (n_chains, n_draws)
    return _diagnostics_row(task, stacked), stacked


def _sweeps_from_existing_csv(dataset: str) -> dict[tuple[int, float], int]:
    path = observables_path(dataset)
    if not path.is_file():
        return {}
    df = pd.read_csv(path)
    out: dict[tuple[int, float], int] = {}
    for _, r in df.iterrows():
        out[(int(r["L"]), round(float(r["T"]), 6))] = int(r["n_sweeps"])
    return out


def _row_from_attrs(attrs: dict) -> dict[str, float | int]:
    skip = {"point_key"}
    row: dict[str, float | int] = {}
    for k, v in attrs.items():
        if k in skip:
            continue
        row[k] = v
    row["L"] = int(row["L"])
    row["T"] = float(row["T"])
    row["n_chains"] = int(row["n_chains"])
    row["n_sweeps"] = int(row["n_sweeps"])
    row["n_thermalize"] = int(row["n_thermalize"])
    row["seed"] = int(row["seed"])
    return row


def _load_point_zarr(
    root: zarr.Group, task: _PointTask
) -> tuple[dict[str, float | int], np.ndarray] | None:
    key = _point_key(task.L, task.lam)
    if key not in root:
        return None
    grp = root[key]
    if "m_signed" not in grp:
        return None
    arr = grp["m_signed"]
    if tuple(arr.shape) != (task.n_chains, task.n_sweeps):
        return None
    attrs = dict(grp.attrs)
    if int(attrs.get("n_chains", -1)) != task.n_chains:
        return None
    if int(attrs.get("n_sweeps", -1)) != task.n_sweeps:
        return None
    # Stream into memory only for diagnostics resume; chunked on disk.
    chains = np.asarray(arr[:], dtype=np.float64)
    return _row_from_attrs(attrs), chains


def _save_point_zarr(
    root: zarr.Group,
    task: _PointTask,
    row: dict[str, float | int],
    chains: np.ndarray,
) -> None:
    key = _point_key(task.L, task.lam)
    if key in root:
        del root[key]
    grp = root.create_group(key)
    n_chains, n_draws = chains.shape
    chunk_draws = min(_DRAW_CHUNK, max(n_draws, 1))
    arr = grp.create_array(
        "m_signed",
        shape=(n_chains, n_draws),
        chunks=(1, chunk_draws),
        dtype=np.float64,
    )
    # Write one chain at a time so a single chunk stream stays small.
    for c in range(n_chains):
        arr[c, :] = chains[c]
    for k, v in row.items():
        grp.attrs[k] = v
    grp.attrs["point_key"] = key


def _write_rhat_csv(rows: list[dict[str, float | int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["L", "T"], ignore_index=True)
    df.to_csv(path, index=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="L64_128")
    parser.add_argument("--n-chains", type=int, default=4)
    parser.add_argument(
        "--n-sweeps",
        type=int,
        default=None,
        help="Fixed post-warmup draws per chain. Default: use n_sweeps from "
        "existing observables CSV per point, else dataset.n_sweeps.",
    )
    parser.add_argument(
        "--n-sweeps-cap",
        type=int,
        default=None,
        help="Optional upper bound on per-point n_sweeps (for quicker diagnostics).",
    )
    parser.add_argument("--n-thermalize", type=int, default=None)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel (L,λ) points. Keep 1 on a single GPU.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore existing per-point zarr groups and rerun.",
    )
    parser.add_argument(
        "--lam-min",
        type=float,
        default=None,
        help="Only run points with λ >= this (e.g. 4.2 near criticality).",
    )
    parser.add_argument(
        "--lam-max",
        type=float,
        default=None,
        help="Only run points with λ <= this.",
    )
    args = parser.parse_args(argv)

    if args.n_chains < 2:
        parser.error("--n-chains must be >= 2 for R-hat")

    spec = get_dataset(args.dataset)
    existing_sweeps = _sweeps_from_existing_csv(args.dataset)
    n_thermalize = int(args.n_thermalize or spec.n_thermalize)

    tasks: list[_PointTask] = []
    for offset, (L, lam) in enumerate(spec.grid()):
        if args.lam_min is not None and float(lam) < float(args.lam_min):
            continue
        if args.lam_max is not None and float(lam) > float(args.lam_max):
            continue
        key = (int(L), round(float(lam), 6))
        if args.n_sweeps is not None:
            n_sweeps = int(args.n_sweeps)
        else:
            n_sweeps = int(existing_sweeps.get(key, spec.n_sweeps))
        if args.n_sweeps_cap is not None:
            n_sweeps = min(n_sweeps, int(args.n_sweeps_cap))
        tasks.append(
            _PointTask(
                L=int(L),
                lam=float(lam),
                base_seed=spec.seed + offset,
                n_chains=int(args.n_chains),
                n_thermalize=n_thermalize,
                n_sweeps=n_sweeps,
            )
        )
    if not tasks:
        parser.error("No (L, λ) points left after --lam-min/--lam-max filters")

    samples_path = _samples_zarr_path(args.dataset)
    root = _open_samples_store(samples_path, mode="a")
    root.attrs["dataset"] = args.dataset
    root.attrs["n_chains"] = int(args.n_chains)
    root.attrs["n_thermalize"] = n_thermalize

    print(
        f"Multi-chain R-hat for {args.dataset}: {len(tasks)} points × "
        f"{args.n_chains} chains, jobs={args.jobs}\n"
        f"  samples zarr -> {samples_path}",
        flush=True,
    )

    rows: list[dict[str, float | int]] = []
    rhat_path = _rhat_csv_path(args.dataset)

    def _record(task: _PointTask, row: dict[str, float | int], chains: np.ndarray) -> None:
        _save_point_zarr(root, task, row, chains)
        rows.append(row)
        _write_rhat_csv(rows, rhat_path)
        print(
            f"Progress: {len(rows)}/{len(tasks)}  L={task.L} lam={task.lam:g}  "
            f"rhat(|m|)={float(row['rhat_abs_m']):.3f}  "
            f"ess(|m|)={float(row['ess_abs_m']):.1f}  "
            f"draws/chain={task.n_sweeps}",
            flush=True,
        )

    pending: list[_PointTask] = []
    for task in tasks:
        if not args.overwrite:
            loaded = _load_point_zarr(root, task)
            if loaded is not None:
                row, chains = loaded
                rows.append(row)
                _write_rhat_csv(rows, rhat_path)
                print(
                    f"Resume: {len(rows)}/{len(tasks)}  L={task.L} lam={task.lam:g}  "
                    f"rhat(|m|)={float(row['rhat_abs_m']):.3f}  "
                    f"(loaded from zarr)",
                    flush=True,
                )
                continue
        pending.append(task)

    if not pending:
        print("All points already in zarr; refreshed rhat CSV.", flush=True)
    elif args.jobs <= 1:
        for task in pending:
            row, chains = _run_point(task)
            _record(task, row, chains)
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx) as pool:
            futs = {pool.submit(_run_point, t): t for t in pending}
            for fut in as_completed(futs):
                task = futs[fut]
                row, chains = fut.result()
                _record(task, row, chains)

    _write_rhat_csv(rows, rhat_path)

    df = pd.DataFrame(rows)
    print(f"Wrote {rhat_path}")
    print(f"Wrote {samples_path}  (gitignored; stream e.g. root['L64_T4.2500']['m_signed'][c])")
    print(
        f"rhat(|m|): min={df.rhat_abs_m.min():.3f}, "
        f"median={df.rhat_abs_m.median():.3f}, max={df.rhat_abs_m.max():.3f}"
    )
    bad = df[df.rhat_abs_m > 1.1].sort_values("rhat_abs_m", ascending=False)
    if len(bad):
        print("Points with rhat(|m|) > 1.1:")
        print(bad[["L", "T", "rhat_abs_m", "ess_abs_m", "n_sweeps"]].to_string(index=False))
    else:
        print("All points have rhat(|m|) <= 1.1")


if __name__ == "__main__":
    main()
