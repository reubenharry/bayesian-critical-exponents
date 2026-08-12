#!/usr/bin/env python3
"""Extend ``L64_128``: fill missing (L, λ) points and top up ESS(m) on existing ones."""

from __future__ import annotations

import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ising.sampler import IsingRawSamples as Phi4RawSamples
from phi4.datasets import (
    get_dataset,
    observables_path,
    samples_path,
    write_manifest,
)
from phi4.sampler import (
    _magnetization_ess,
    result_to_row,
    run_phi4_until_ess,
    summarize_signed_m_samples,
)
from phi4.samples import load_samples, save_samples


@dataclass(frozen=True)
class _Task:
    L: int
    lam: float
    seed: int
    min_ess_m: float
    n_thermalize: int
    n_sweeps_min: int
    n_sweeps_max: int
    chunk_sweeps: int
    existing_m: np.ndarray | None  # finite draws already collected


def _run_task(task: _Task) -> tuple[dict[str, float | int], np.ndarray]:
    existing = (
        None
        if task.existing_m is None
        else np.asarray(task.existing_m, dtype=np.float64).ravel()
    )
    existing_ess = 0.0 if existing is None or existing.size < 2 else _magnetization_ess(existing)
    if existing is not None and existing_ess >= task.min_ess_m:
        result = summarize_signed_m_samples(
            existing,
            L=task.L,
            T=task.lam,
            n_thermalize=task.n_thermalize,
            n_sweeps=int(existing.size),
            seed=task.seed,
        )
        return result_to_row(result), existing

    # How much extra ESS we still need (rough): run until new chain alone covers the gap
    # or until combined ESS hits the target.
    need = max(task.min_ess_m - existing_ess, 20.0)
    extra, raw = run_phi4_until_ess(
        task.L,
        task.lam,
        min_ess_m=need if existing is None else max(need * 0.6, 25.0),
        n_thermalize=task.n_thermalize,
        n_sweeps_min=task.n_sweeps_min,
        n_sweeps_max=task.n_sweeps_max,
        chunk_sweeps=task.chunk_sweeps,
        seed=task.seed,
        store_samples=True,
    )
    del extra
    combined = raw.m_signed if existing is None else np.concatenate([existing, raw.m_signed])
    # If still short, keep appending independent chains until target or budget exhausted.
    budget_left = task.n_sweeps_max
    attempt = 1
    while _magnetization_ess(combined) < task.min_ess_m and budget_left > task.n_sweeps_min:
        attempt += 1
        more, raw2 = run_phi4_until_ess(
            task.L,
            task.lam,
            min_ess_m=30.0,
            n_thermalize=task.n_thermalize,
            n_sweeps_min=min(task.n_sweeps_min, budget_left),
            n_sweeps_max=budget_left,
            chunk_sweeps=task.chunk_sweeps,
            seed=task.seed + 1000 * attempt,
            store_samples=True,
        )
        del more
        combined = np.concatenate([combined, raw2.m_signed])
        budget_left = max(budget_left - int(raw2.m_signed.size), 0)

    result = summarize_signed_m_samples(
        combined,
        L=task.L,
        T=task.lam,
        n_thermalize=task.n_thermalize,
        n_sweeps=int(combined.size),
        seed=task.seed,
    )
    if result.n_eff + 1e-9 < task.min_ess_m:
        print(
            f"  WARNING L={task.L} lam={task.lam:g}: n_eff(m)={result.n_eff:.1f} "
            f"< target {task.min_ess_m:g} after {combined.size} draws",
            flush=True,
        )
    return result_to_row(result), combined


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="L64_128")
    parser.add_argument("--min-ess-m", type=float, default=250.0)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--max-sweeps", type=int, default=250_000)
    parser.add_argument("--chunk-sweeps", type=int, default=8000)
    parser.add_argument("--n-sweeps-min", type=int, default=12000)
    args = parser.parse_args(argv)

    spec = get_dataset(args.dataset)
    csv_path = observables_path(args.dataset)
    sample_path = samples_path(args.dataset)

    if csv_path.is_file():
        df_old = pd.read_csv(csv_path)
    else:
        df_old = pd.DataFrame()

    stored = None
    if sample_path.is_file():
        stored = load_samples(sample_path)

    existing_map: dict[tuple[int, float], np.ndarray] = {}
    if stored is not None:
        for i in range(stored.n_points):
            key = (int(stored.L[i]), round(float(stored.T[i]), 6))
            m = np.asarray(stored.m_signed[i], dtype=np.float64)
            existing_map[key] = m[np.isfinite(m)]

    tasks: list[_Task] = []
    for offset, (L, lam) in enumerate(spec.grid()):
        key = (int(L), round(float(lam), 6))
        existing = existing_map.get(key)
        tasks.append(
            _Task(
                L=int(L),
                lam=float(lam),
                seed=spec.seed + offset,
                min_ess_m=float(args.min_ess_m),
                n_thermalize=spec.n_thermalize,
                n_sweeps_min=args.n_sweeps_min,
                n_sweeps_max=args.max_sweeps,
                chunk_sweeps=args.chunk_sweeps,
                existing_m=existing,
            )
        )

    print(
        f"Extending {args.dataset}: {len(tasks)} points, "
        f"min_ess_m={args.min_ess_m:g}, jobs={args.jobs}",
        flush=True,
    )

    rows: list[dict[str, float | int]] = []
    chains: dict[tuple[int, float], np.ndarray] = {}

    def _record(task: _Task, row: dict[str, float | int], chain: np.ndarray) -> None:
        rows.append(row)
        chains[(task.L, round(task.lam, 6))] = chain
        print(
            f"Progress: {len(rows)}/{len(tasks)}  L={task.L} lam={task.lam:g}  "
            f"n_eff(m)={float(row['n_eff']):.1f}  draws={chain.size}",
            flush=True,
        )

    if args.jobs <= 1:
        for task in tasks:
            row, chain = _run_task(task)
            _record(task, row, chain)
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx) as pool:
            futs = {pool.submit(_run_task, t): t for t in tasks}
            for fut in as_completed(futs):
                task = futs[fut]
                row, chain = fut.result()
                _record(task, row, chain)

    df = pd.DataFrame(rows).sort_values(["L", "T"], ignore_index=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    write_manifest(spec)

    max_len = max(ch.size for ch in chains.values())
    items: list[Phi4RawSamples] = []
    for _, row in df.iterrows():
        key = (int(row["L"]), round(float(row["T"]), 6))
        m = chains[key]
        pad = np.full(max_len, np.nan, dtype=np.float64)
        pad[: m.size] = m
        items.append(
            Phi4RawSamples(
                L=int(row["L"]),
                T=float(row["T"]),
                m_signed=pad,
                n_thermalize=spec.n_thermalize,
                n_sweeps=max_len,
                sample_stride=1,
                seed=int(row["seed"]),
            )
        )
    save_samples(sample_path, items)

    neff = df["n_eff"].to_numpy(dtype=float)
    print(f"Wrote {csv_path} ({len(df)} points)")
    print(f"Wrote {sample_path} (padded to {max_len} draws)")
    print(
        f"n_eff(m): min={neff.min():.1f}, median={float(np.median(neff)):.1f}, "
        f"max={neff.max():.1f}"
    )
    short = df[df["n_eff"] < args.min_ess_m]
    if len(short):
        print("Still below target:")
        print(short[["L", "T", "n_eff", "n_sweeps"]].to_string(index=False))


if __name__ == "__main__":
    main()
