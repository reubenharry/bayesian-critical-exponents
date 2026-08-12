#!/usr/bin/env python3
"""Build ``harada_square_outlier`` = Harada square + one (L=8, T=8) point."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .datasets import (
    HARADA_SQUARE_OUTLIER_POINT,
    get_dataset,
    observables_path,
    write_manifest,
)
from .simulate import simulate_grid


def build_observables(*, force_resimulate: bool = False) -> Path:
    name = "harada_square_outlier"
    spec = get_dataset(name)
    base_path = observables_path("harada_square")
    out_path = observables_path(name)
    L_out, T_out = HARADA_SQUARE_OUTLIER_POINT

    if not base_path.exists():
        raise FileNotFoundError(
            f"Need existing Harada observables at {base_path}; "
            "run simulate --dataset harada_square first."
        )
    base = pd.read_csv(base_path)
    print(f"base harada_square: {len(base)} rows from {base_path}", flush=True)

    # Prefer the cached large-t extra row if present (same (L,T) already simulated).
    extra_cache = observables_path("harada_square_large_t").with_name(
        "observables_harada_square_large_t_extra.csv"
    )
    outlier: pd.DataFrame | None = None
    if extra_cache.exists() and not force_resimulate:
        extras = pd.read_csv(extra_cache)
        hit = extras[(extras["L"].astype(int) == L_out) & (extras["T"] == T_out)]
        if len(hit) == 1:
            outlier = hit.copy()
            print(f"reusing outlier row from {extra_cache}", flush=True)

    if outlier is None:
        print(
            f"simulating outlier L={L_out}, T={T_out} "
            f"({spec.n_sweeps} sweeps)",
            flush=True,
        )
        outlier, _ = simulate_grid(
            [(L_out, T_out)],
            n_thermalize=spec.n_thermalize,
            n_sweeps=spec.n_sweeps,
            seed=spec.seed + 20_000,
            n_jobs=1,
        )

    merged = pd.concat([base, outlier], ignore_index=True)
    merged = merged.sort_values(["L", "T"], ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    write_manifest(spec)
    print(f"wrote {len(merged)} rows -> {out_path}", flush=True)
    print(merged.groupby("L").size().to_string(), flush=True)
    return out_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-resimulate",
        action="store_true",
        help="Ignore cached large-t extras; re-run Wolff for the outlier",
    )
    args = parser.parse_args(argv)
    build_observables(force_resimulate=args.force_resimulate)


if __name__ == "__main__":
    main()
