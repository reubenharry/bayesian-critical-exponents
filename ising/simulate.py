#!/usr/bin/env python3
"""Run a (L, T) grid of 2D Ising simulations for a named dataset."""

from __future__ import annotations

import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .datasets import (
    DATASETS,
    get_dataset,
    list_datasets,
    observables_path,
    samples_path,
    write_manifest,
)
from .sampler import IsingRawSamples, result_to_row, run_ising


def append_extra_sweeps(
    observables: pd.DataFrame,
    *,
    L: int,
    T: float,
    extra_sweeps: int,
    n_thermalize: int = 0,
    seed: int = 0,
) -> pd.DataFrame:
    """Run Wolff MC at ``(L, T)`` and merge into the observables table."""
    from .observables import merge_observation_row

    result = run_ising(
        L,
        T,
        n_thermalize=n_thermalize,
        n_sweeps=extra_sweeps,
        seed=seed,
    )
    row = result_to_row(result)
    row["n_sweeps"] = extra_sweeps
    return merge_observation_row(observables, L=L, T=T, new_row=row)


@dataclass(frozen=True)
class _GridTask:
    L: int
    T: float
    n_thermalize: int
    n_sweeps: int
    seed: int
    store_samples: bool
    checkpoint_every: int | None
    checkpoint_dir: Path | None


@dataclass(frozen=True)
class _GridPointResult:
    row: dict[str, float | int]
    samples: IsingRawSamples | None


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
                n_sweeps_target=task.n_sweeps,
                n_sweeps_done=n_sweeps_done,
                sample_stride=1,
            ),
        )
        print(
            f"  checkpoint L={task.L} T={task.T:.6g}: "
            f"{n_sweeps_done}/{task.n_sweeps} sweeps -> {path.name}",
            flush=True,
        )

    return on_checkpoint


def _run_grid_task(task: _GridTask) -> _GridPointResult:
    # Import inside the worker so spawn/fork children stay clear of parent imports.
    from .sampler import run_ising

    on_checkpoint = _make_checkpoint_callback(task)
    output = run_ising(
        task.L,
        task.T,
        n_thermalize=task.n_thermalize,
        n_sweeps=task.n_sweeps,
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
    target = row.get("n_sweeps_target", sweeps)
    u4 = row.get("binder_cumulant")
    u4_str = f"{float(u4):.4f}" if u4 is not None else "n/a"
    msg = (
        f"Progress: {n_done}/{n_total}  "
        f"L={row['L']} T={row['T']:.6g}  "
        f"U4={u4_str}  sweeps={sweeps}/{target}"
    )
    if partial_out is not None:
        msg += f"  (partial -> {partial_out})"
    print(msg, flush=True)


def _record_grid_point(
    item: _GridPointResult,
    *,
    rows: list[dict[str, float | int]],
    raw_samples: list[IsingRawSamples],
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
) -> tuple[pd.DataFrame, list[IsingRawSamples]]:
    if checkpoint_every is not None and checkpoint_dir is None:
        raise ValueError("checkpoint_dir is required when checkpoint_every is set")

    tasks = [
        _GridTask(
            L,
            T,
            n_thermalize,
            n_sweeps,
            seed + offset,
            store_samples,
            checkpoint_every,
            checkpoint_dir,
        )
        for offset, (L, T) in enumerate(grid)
    ]
    rows: list[dict[str, float | int]] = []
    raw_samples: list[IsingRawSamples] = []

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
) -> Path:
    spec = get_dataset(name)
    ckpt_dir = checkpoint_dir
    if checkpoint_every is not None and ckpt_dir is None:
        from .samples import default_checkpoint_dir

        ckpt_dir = default_checkpoint_dir(name)

    partial_out = partial_observables_path(name) if checkpoint_every is not None else None
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
    )
    path = out or observables_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    write_manifest(spec)
    print(f"Dataset {name!r}: {len(df)} points -> {path}")
    if store_samples:
        from .samples import save_samples

        sample_file = save_samples(samples_out or samples_path(name), raw_samples)
        print(f"  raw chains -> {sample_file}")
    if checkpoint_every is not None:
        assert ckpt_dir is not None
        assert partial_out is not None
        print(
            f"  checkpoints every {checkpoint_every} sweeps -> {ckpt_dir} "
            f"(partial observables -> {partial_out})"
        )
    if n_jobs > 1:
        print(f"  parallel workers: {min(n_jobs, len(df))}")
    if spec.z_values is not None:
        print(
            f"  L in {spec.L}, {len(spec.z_values)} z-values (|z|<={max(map(abs, spec.z_values))}), "
            f"{spec.n_sweeps} sweeps, {spec.n_thermalize} thermalize"
        )
    else:
        print(
            f"  L in {spec.L}, {len(spec.T)} temperatures, "
            f"{spec.n_sweeps} sweeps, {spec.n_thermalize} thermalize"
        )
    return path


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=list_datasets(),
        default="medium",
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
        help="Parallel workers for independent (L, T) points (default: 1)",
    )
    parser.add_argument(
        "--store-samples",
        action="store_true",
        help="Also save raw per-sweep m chains to data/samples_<dataset>.npz",
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
        help=(
            "Save partial Wolff chains every N sweeps to "
            "data/checkpoints/<dataset>/ (also writes observables_<dataset>.partial.csv "
            "as grid points finish)"
        ),
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
    args = parser.parse_args(argv)

    if args.list:
        for name, spec in DATASETS.items():
            if spec.z_values is not None:
                temp_label = f"z={len(spec.z_values)}"
            else:
                temp_label = f"T={len(spec.T)}"
            print(
                f"{name:18s}  {spec.n_points:3d} pts  "
                f"sweeps={spec.n_sweeps:5d}  {temp_label}  {spec.description}"
            )
        return observables_path("medium")

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
    )


if __name__ == "__main__":
    main()
