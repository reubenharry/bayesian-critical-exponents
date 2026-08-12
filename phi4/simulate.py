#!/usr/bin/env python3
"""Run a (L, λ) grid of 2D φ⁴ HMC simulations for a named dataset.

CSV column ``T`` stores the quartic coupling λ (FSS schema compatibility).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .datasets import (
    DATASETS,
    get_dataset,
    list_datasets,
    observables_path,
    samples_path,
    write_manifest,
)
from .sampler import Phi4RawSamples, result_to_row


@dataclass(frozen=True)
class _GridTask:
    L: int
    T: float  # λ
    n_thermalize: int
    n_sweeps: int
    seed: int
    store_samples: bool
    checkpoint_every: int | None
    checkpoint_dir: Path | None
    min_ess_m: float
    n_sweeps_max: int
    chunk_sweeps: int


@dataclass(frozen=True)
class _GridPointResult:
    row: dict[str, float | int]
    samples: Phi4RawSamples | None


def _make_checkpoint_callback(task: _GridTask):
    if task.checkpoint_every is None or task.checkpoint_dir is None:
        return None

    from .samples import PointCheckpoint, point_checkpoint_path, save_point_checkpoint

    path = point_checkpoint_path(task.checkpoint_dir, task.L, task.T)

    def on_checkpoint(n_sweeps_done: int, m_signed: object) -> None:
        save_point_checkpoint(
            path,
            PointCheckpoint(
                L=task.L,
                T=task.T,
                m_signed=m_signed,  # type: ignore[arg-type]
                seed=task.seed,
                n_thermalize=task.n_thermalize,
                n_sweeps_target=task.n_sweeps_max if task.min_ess_m > 0 else task.n_sweeps,
                n_sweeps_done=n_sweeps_done,
                sample_stride=1,
            ),
        )
        print(
            f"  checkpoint L={task.L} lam={task.T:.6g}: "
            f"{n_sweeps_done} sweeps -> {path.name}",
            flush=True,
        )

    return on_checkpoint


def _run_grid_task(task: _GridTask) -> _GridPointResult:
    from .sampler import run_phi4_until_ess

    on_checkpoint = _make_checkpoint_callback(task)
    output = run_phi4_until_ess(
        task.L,
        task.T,
        min_ess_m=task.min_ess_m,
        n_thermalize=task.n_thermalize,
        n_sweeps_min=task.n_sweeps,
        n_sweeps_max=task.n_sweeps_max,
        chunk_sweeps=task.chunk_sweeps,
        seed=task.seed,
        store_samples=task.store_samples,
        checkpoint_every=task.checkpoint_every,
        on_checkpoint=on_checkpoint,
    )
    if task.store_samples:
        result, raw = output
        return _GridPointResult(row=result_to_row(result), samples=raw)
    return _GridPointResult(row=result_to_row(output), samples=None)


def _write_partial_observables(rows: list[dict[str, float | int]], path: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values(["L", "T"], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _report_grid_point_progress(
    n_done: int,
    n_total: int,
    row: dict[str, float | int],
    *,
    partial_out: Path | None,
) -> None:
    sweeps = row.get("n_sweeps", row.get("n_draws", "?"))
    u4 = row.get("binder_cumulant")
    u4_str = f"{float(u4):.4f}" if u4 is not None else "n/a"
    neff = row.get("n_eff")
    neff_str = f"{float(neff):.1f}" if neff is not None else "n/a"
    msg = (
        f"Progress: {n_done}/{n_total}  "
        f"L={row['L']} lam={row['T']:.6g}  "
        f"U4={u4_str}  n_eff(m)={neff_str}  sweeps={sweeps}"
    )
    if partial_out is not None:
        msg += f"  (partial -> {partial_out})"
    print(msg, flush=True)


def _record_grid_point(
    item: _GridPointResult,
    *,
    rows: list[dict[str, float | int]],
    raw_samples: list[Phi4RawSamples],
    n_total: int,
    partial_out: Path | None,
) -> None:
    rows.append(item.row)
    if item.samples is not None:
        raw_samples.append(item.samples)
    _report_grid_point_progress(len(rows), n_total, item.row, partial_out=partial_out)
    if partial_out is not None:
        _write_partial_observables(rows, partial_out)


def simulate_grid(
    grid: list[tuple[int, float]],
    *,
    n_thermalize: int,
    n_sweeps: int,
    seed: int,
    n_jobs: int = 1,
    store_samples: bool = False,
    checkpoint_every: int | None = None,
    checkpoint_dir: Path | None = None,
    partial_out: Path | None = None,
    min_ess_m: float = 0.0,
    n_sweeps_max: int | None = None,
    chunk_sweeps: int = 4000,
) -> tuple[pd.DataFrame, list[Phi4RawSamples]]:
    if checkpoint_every is not None and checkpoint_dir is None:
        raise ValueError("checkpoint_dir is required when checkpoint_every is set")
    sweeps_max = int(n_sweeps if n_sweeps_max is None else n_sweeps_max)

    tasks = [
        _GridTask(
            L,
            lam,
            n_thermalize,
            n_sweeps,
            seed + offset,
            store_samples,
            checkpoint_every,
            checkpoint_dir,
            float(min_ess_m),
            sweeps_max,
            int(chunk_sweeps),
        )
        for offset, (L, lam) in enumerate(grid)
    ]
    rows: list[dict[str, float | int]] = []
    raw_samples: list[Phi4RawSamples] = []

    if n_jobs <= 1 or len(tasks) <= 1:
        for task in tasks:
            item = _run_grid_task(task)
            _record_grid_point(
                item,
                rows=rows,
                raw_samples=raw_samples,
                n_total=len(tasks),
                partial_out=partial_out,
            )
    else:
        workers = min(n_jobs, len(tasks))
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futures = {pool.submit(_run_grid_task, task): task for task in tasks}
            for future in as_completed(futures):
                item = future.result()
                _record_grid_point(
                    item,
                    rows=rows,
                    raw_samples=raw_samples,
                    n_total=len(tasks),
                    partial_out=partial_out,
                )

    if store_samples and len(raw_samples) != len(tasks):
        raise RuntimeError("expected raw samples for every grid point")
    return pd.DataFrame(rows).sort_values(["L", "T"], ignore_index=True), raw_samples


def partial_observables_path(name: str) -> Path:
    return observables_path(name).with_name(f"observables_{name}.partial.csv")


def simulate_dataset(
    name: str,
    *,
    out: Path | None = None,
    n_jobs: int = 1,
    store_samples: bool = False,
    samples_out: Path | None = None,
    checkpoint_every: int | None = None,
    checkpoint_dir: Path | None = None,
    min_ess_m: float = 0.0,
    n_sweeps_max: int | None = None,
    chunk_sweeps: int = 4000,
) -> Path:
    spec = get_dataset(name)
    ckpt_dir = checkpoint_dir
    if checkpoint_every is not None and ckpt_dir is None:
        from .samples import default_checkpoint_dir

        ckpt_dir = default_checkpoint_dir(name)

    partial_out = partial_observables_path(name) if checkpoint_every is not None else None
    sweeps_max = n_sweeps_max
    if sweeps_max is None and min_ess_m > 0:
        sweeps_max = max(spec.n_sweeps * 20, 80_000)
    df, raw_samples = simulate_grid(
        spec.grid(),
        n_thermalize=spec.n_thermalize,
        n_sweeps=spec.n_sweeps,
        seed=spec.seed,
        n_jobs=n_jobs,
        store_samples=store_samples,
        checkpoint_every=checkpoint_every,
        checkpoint_dir=ckpt_dir,
        partial_out=partial_out,
        min_ess_m=min_ess_m,
        n_sweeps_max=sweeps_max,
        chunk_sweeps=chunk_sweeps,
    )
    path = out or observables_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    write_manifest(spec)
    print(f"Dataset {name!r}: {len(df)} points -> {path}")
    if store_samples and raw_samples:
        from .samples import save_samples

        # Adaptive ESS yields unequal chain lengths; pad with NaN to a common width.
        max_len = max(int(s.m_signed.size) for s in raw_samples)
        padded: list[Phi4RawSamples] = []
        for s in raw_samples:
            if s.m_signed.size == max_len:
                padded.append(s)
                continue
            m = np.full(max_len, np.nan, dtype=np.float64)
            m[: s.m_signed.size] = s.m_signed
            padded.append(
                Phi4RawSamples(
                    L=s.L,
                    T=s.T,
                    m_signed=m,
                    n_thermalize=s.n_thermalize,
                    n_sweeps=max_len,
                    sample_stride=s.sample_stride,
                    seed=s.seed,
                )
            )
        sample_file = save_samples(samples_out or samples_path(name), padded)
        print(f"  raw chains -> {sample_file} (padded to {max_len} draws)")
    if checkpoint_every is not None:
        assert ckpt_dir is not None
        assert partial_out is not None
        print(
            f"  checkpoints every {checkpoint_every} sweeps -> {ckpt_dir} "
            f"(partial observables -> {partial_out})"
        )
    if n_jobs > 1:
        print(f"  parallel workers: {min(n_jobs, len(df))}")
    print(
        f"  L in {spec.L}, {len(spec.T) if spec.T else spec.n_T} λ values, "
        f"min_sweeps={spec.n_sweeps}, thermalize={spec.n_thermalize}"
        + (f", min_ess_m={min_ess_m:g}" if min_ess_m > 0 else "")
    )
    if min_ess_m > 0:
        neff = df["n_eff"].to_numpy(dtype=float)
        print(
            f"  n_eff(m): min={neff.min():.1f}, median={float(np.median(neff)):.1f}, "
            f"max={neff.max():.1f}"
        )
    return path


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=list_datasets(),
        default="small",
        help=f"Named preset ({', '.join(list_datasets())})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Override output CSV (default: data/observables_<dataset>.csv)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help="Parallel workers for independent (L, λ) points (default: 1)",
    )
    parser.add_argument(
        "--store-samples",
        action="store_true",
        help="Also save raw m chains to data/samples_<dataset>.npz",
    )
    parser.add_argument(
        "--samples-out",
        type=Path,
        default=None,
        help="Override raw-sample NPZ path (used with --store-samples)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        metavar="N",
        help="Save partial chains every N HMC steps to data/checkpoints/<dataset>/",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Override checkpoint directory (used with --checkpoint-every)",
    )
    parser.add_argument(
        "--aggregate-samples",
        type=Path,
        default=None,
        metavar="NPZ",
        help="Rebuild observables CSV from a stored samples NPZ (no simulation)",
    )
    parser.add_argument(
        "--aggregate-checkpoints",
        type=Path,
        default=None,
        metavar="DIR",
        help="Rebuild observables CSV from partial point checkpoints (no simulation)",
    )
    parser.add_argument(
        "--max-draws",
        type=int,
        default=None,
        metavar="N",
        help="When aggregating samples, use only the first N draws per chain",
    )
    parser.add_argument(
        "--thin",
        type=int,
        default=1,
        metavar="K",
        help="When aggregating samples, keep every K-th draw (default: 1)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print dataset presets and exit",
    )
    parser.add_argument(
        "--min-ess-m",
        type=float,
        default=0.0,
        metavar="N",
        help="Keep sampling until blackjax ESS of |m| reaches N (0 disables)",
    )
    parser.add_argument(
        "--max-sweeps",
        type=int,
        default=None,
        metavar="N",
        help="Cap post-warmup HMC sweeps when using --min-ess-m",
    )
    parser.add_argument(
        "--chunk-sweeps",
        type=int,
        default=4000,
        metavar="N",
        help="Adaptive ESS chunk size (default: 4000)",
    )
    args = parser.parse_args(argv)

    if args.list:
        for name, spec in DATASETS.items():
            print(
                f"{name:18s}  {spec.n_points:3d} pts  "
                f"sweeps={spec.n_sweeps:5d}  lam={spec.n_T}  {spec.description}"
            )
        return observables_path("small")

    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    if args.thin < 1:
        parser.error("--thin must be >= 1")
    if args.max_draws is not None and args.max_draws < 1:
        parser.error("--max-draws must be >= 1")
    if args.checkpoint_every is not None and args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be >= 1")

    if args.aggregate_samples is not None:
        from .samples import aggregate_samples_file

        out = args.out or observables_path(args.dataset)
        aggregate_samples_file(
            args.aggregate_samples,
            out=out,
            max_draws=args.max_draws,
            thin=args.thin,
        )
        label = f"max_draws={args.max_draws}" if args.max_draws is not None else "all draws"
        if args.thin > 1:
            label = f"{label}, thin={args.thin}"
        print(f"Aggregated {args.aggregate_samples} ({label}) -> {out}")
        return out

    if args.aggregate_checkpoints is not None:
        from .samples import write_checkpoint_observables

        out = args.out or partial_observables_path(args.dataset)
        write_checkpoint_observables(args.aggregate_checkpoints, out=out)
        print(f"Aggregated checkpoints in {args.aggregate_checkpoints} -> {out}")
        return out

    return simulate_dataset(
        args.dataset,
        out=args.out,
        n_jobs=args.jobs,
        store_samples=args.store_samples,
        samples_out=args.samples_out,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=args.checkpoint_dir,
        min_ess_m=args.min_ess_m,
        n_sweeps_max=args.max_sweeps,
        chunk_sweeps=args.chunk_sweeps,
    )


if __name__ == "__main__":
    main()
