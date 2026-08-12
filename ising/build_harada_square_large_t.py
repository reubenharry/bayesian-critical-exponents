#!/usr/bin/env python3
"""Build ``harada_square_large_t`` by merging Harada CSV with new large-|t| MC."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .datasets import (
    _harada_square_large_t_extra_points,
    get_dataset,
    observables_path,
    plots_dir,
    write_manifest,
)
from .plot_binder import plot_binder_collapse, plot_binder_collapse_z
from .simulate import simulate_grid


def build_observables(*, n_jobs: int = 24, force_resimulate: bool = False) -> Path:
    name = "harada_square_large_t"
    spec = get_dataset(name)
    extra = list(_harada_square_large_t_extra_points())
    base_path = observables_path("harada_square")
    out_path = observables_path(name)
    extra_cache = out_path.with_name("observables_harada_square_large_t_extra.csv")

    if not base_path.exists():
        raise FileNotFoundError(
            f"Need existing Harada observables at {base_path}; "
            "run simulate --dataset harada_square first."
        )
    base = pd.read_csv(base_path)
    print(f"base harada_square: {len(base)} rows from {base_path}", flush=True)

    if extra_cache.exists() and not force_resimulate:
        print(f"reusing cached extras from {extra_cache}", flush=True)
        extra_df = pd.read_csv(extra_cache)
    else:
        print(
            f"simulating {len(extra)} large-|t| points "
            f"({spec.n_sweeps} sweeps, jobs={n_jobs})",
            flush=True,
        )
        t0 = time.time()
        extra_df, _ = simulate_grid(
            extra,
            n_thermalize=spec.n_thermalize,
            n_sweeps=spec.n_sweeps,
            seed=spec.seed + 10_000,
            n_jobs=n_jobs,
        )
        print(f"extra simulation done in {time.time() - t0:.1f}s", flush=True)
        extra_df.to_csv(extra_cache, index=False)
        print(f"cached extras -> {extra_cache}", flush=True)

    base = base.copy()
    base["_k"] = list(zip(base["L"].astype(int), base["T"].round(6)))
    extra_df = extra_df.copy()
    extra_df["_k"] = list(zip(extra_df["L"].astype(int), extra_df["T"].round(6)))
    overlap = set(base["_k"]) & set(extra_df["_k"])
    if overlap:
        print(f"warning: dropping {len(overlap)} overlapping keys from extra", flush=True)
        extra_df = extra_df[~extra_df["_k"].isin(overlap)]

    merged = pd.concat(
        [base.drop(columns="_k"), extra_df.drop(columns="_k")],
        ignore_index=True,
    )
    merged = merged.sort_values(["L", "T"], ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    write_manifest(spec)
    print(f"wrote {len(merged)} rows -> {out_path}", flush=True)
    print(merged.groupby("L").size().to_string(), flush=True)
    return out_path


def write_collapse_plots(observables: Path | None = None) -> list[Path]:
    name = "harada_square_large_t"
    path = observables or observables_path(name)
    df = pd.read_csv(path)
    out_dir = plots_dir(name)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    z_path = out_dir / "binder_collapse_z_breakdown.png"
    fig = plot_binder_collapse_z(df, out=z_path, t_scale_max=0.10)
    plt.close(fig)
    paths.append(z_path)

    x_path = out_dir / "binder_collapse_harada_x.png"
    fig = plot_binder_collapse(
        df,
        out=x_path,
        title=(
            r"Binder Harada-$x$ collapse with large-$|t|$ extension "
            r"(exact $T_c$, $\nu$)"
        ),
    )
    plt.close(fig)
    paths.append(x_path)

    for p in paths:
        print(f"wrote {p}", flush=True)
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=24)
    parser.add_argument(
        "--force-resimulate",
        action="store_true",
        help="Ignore cached large-|t| extras CSV and resimulate",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip simulation; only regenerate collapse plots",
    )
    args = parser.parse_args(argv)

    if not args.plot_only:
        build_observables(n_jobs=args.jobs, force_resimulate=args.force_resimulate)
    write_collapse_plots()


if __name__ == "__main__":
    main()
