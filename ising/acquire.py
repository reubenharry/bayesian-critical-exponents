#!/usr/bin/env python3
"""Rank and optionally execute active-learning acquisition candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ._deps import az
from .active_learning import (
    CandidateSpec,
    enumerate_extra_sweep_candidates,
    load_posterior_draws,
    rank_candidates,
)
from .datasets import get_dataset, list_datasets, resolve_paths
from .fit import fit_dataset
from .jax_inference import profile_config_from_fit_kwargs
from .observables import ensure_harada_bsa_observables_csv, read_observables_for_fss
from .simulate import append_extra_sweeps


def _parse_extra_sweeps_grid(raw: str) -> tuple[int, ...]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("extra-sweeps grid must contain at least one positive integer")
    grid = tuple(int(p) for p in parts)
    if any(x <= 0 for x in grid):
        raise ValueError("extra-sweeps grid values must be positive")
    return grid


def _default_extra_sweeps_grid(dataset: str) -> tuple[int, ...]:
    spec = get_dataset(dataset)
    base = max(spec.n_sweeps // 10, 1_000)
    return (base, base * 5, base * 10)


def acquire_dataset(
    dataset: str,
    *,
    dry_run: bool = True,
    execute_top: int = 0,
    extra_sweeps_grid: tuple[int, ...] | None = None,
    top_k: int = 10,
    max_is_draws: int | None = 500,
    n_y_samples: int = 8,
    seed: int = 1,
    data: Path | None = None,
    posterior: Path | None = None,
    infer_Tc: bool = True,
    infer_nu: bool = True,
    infer_beta: bool = False,
    use_m: bool = False,
    use_m2: bool = False,
    use_m4: bool = False,
    use_binder: bool = True,
    use_chi: bool = False,
    correction_m: bool = True,
    correction_m2: bool = True,
    correction_m4: bool = True,
    correction_binder: bool = False,
    correction_chi: bool = False,
    use_log_m: bool = False,
    gp_ell_factor: float | None = None,
    gp_eta: float | None = None,
    correction_gp_ell_factor: float | None = None,
    correction_gp_eta: float | None = None,
    gp_kernel: str | None = None,
    obs_sigma_scale: float | None = None,
    refit: bool = True,
    n_thermalize: int | None = None,
    sampler_backend: str = "jax",
    draws: int = 500,
    tune: int = 500,
    chains: int | None = None,
) -> pd.DataFrame:
    """Rank candidates and optionally run the top acquisition."""
    data_path, posterior_path, _ = resolve_paths(dataset, data=data, posterior=posterior)
    if dataset == "harada_bsa":
        data_path = ensure_harada_bsa_observables_csv(data_path)
    if not posterior_path.exists():
        raise FileNotFoundError(f"Posterior not found: {posterior_path}")

    observables = read_observables_for_fss(
        data_path,
        use_m=use_m,
        use_m2=use_m2,
        use_m4=use_m4,
        use_binder=use_binder,
        use_chi=use_chi,
    )
    config = profile_config_from_fit_kwargs(
        use_m=use_m,
        use_m2=use_m2,
        use_m4=use_m4,
        use_binder=use_binder,
        use_chi=use_chi,
        correction_m=correction_m,
        correction_m2=correction_m2,
        correction_m4=correction_m4,
        correction_binder=correction_binder,
        correction_chi=correction_chi,
        use_log_m=use_log_m,
        gp_ell_factor=gp_ell_factor,
        gp_eta=gp_eta,
        correction_gp_ell_factor=correction_gp_ell_factor,
        correction_gp_eta=correction_gp_eta,
        gp_kernel=gp_kernel,
        obs_sigma_scale=obs_sigma_scale,
        infer_Tc=infer_Tc,
        infer_nu=infer_nu,
        infer_beta=infer_beta,
    )
    idata = az.from_netcdf(posterior_path)
    draws_obj = load_posterior_draws(
        idata,
        infer_Tc=infer_Tc,
        infer_nu=infer_nu,
        infer_beta=infer_beta,
        max_draws=max_is_draws,
        seed=seed,
    )

    grid = extra_sweeps_grid or _default_extra_sweeps_grid(dataset)
    candidates = enumerate_extra_sweep_candidates(observables, grid)
    ranked = rank_candidates(
        observables,
        config,
        draws_obj,
        candidates,
        infer_Tc=infer_Tc,
        infer_nu=infer_nu,
        infer_beta=infer_beta,
        n_y_samples=n_y_samples,
        seed=seed,
    )

    print(f"Dataset: {dataset!r} ({len(observables)} points)")
    print(f"Posterior: {posterior_path}")
    print(f"IS draws: {draws_obj.n_samples}, extra-sweeps grid: {grid}")
    if ranked.empty:
        print("No candidates scored.")
        return ranked

    show = ranked.head(top_k)
    print("\nTop candidates by expected uncertainty reduction:")
    print(
        show.to_string(
            index=False,
            float_format=lambda x: f"{x: .6g}",
        )
    )

    if dry_run or execute_top <= 0:
        return ranked

    best = ranked.iloc[0]
    candidate = CandidateSpec(
        L=int(best["L"]),
        T=float(best["T"]),
        kind="extra_sweeps",
        extra_sweeps=int(best["extra_sweeps"]),
    )
    spec = get_dataset(dataset)
    thermalize = spec.n_thermalize if n_thermalize is None else n_thermalize
    print(
        f"\nExecuting top candidate: L={candidate.L} T={candidate.T:.6g} "
        f"+{candidate.extra_sweeps} sweeps"
    )
    updated = append_extra_sweeps(
        observables,
        L=candidate.L,
        T=candidate.T,
        extra_sweeps=candidate.extra_sweeps,
        n_thermalize=thermalize,
        seed=seed + 1,
    )
    updated.to_csv(data_path, index=False)
    print(f"Updated observables -> {data_path}")

    if refit:
        fit_dataset(
            dataset,
            data=data_path,
            infer_Tc=infer_Tc,
            infer_nu=infer_nu,
            infer_beta=infer_beta,
            use_m=use_m,
            use_m2=use_m2,
            use_m4=use_m4,
            use_binder=use_binder,
            use_chi=use_chi,
            correction_m=correction_m,
            correction_m2=correction_m2,
            correction_m4=correction_m4,
            correction_binder=correction_binder,
            correction_chi=correction_chi,
            use_log_m=use_log_m,
            gp_ell_factor=gp_ell_factor,
            gp_eta=gp_eta,
            correction_gp_ell_factor=correction_gp_ell_factor,
            correction_gp_eta=correction_gp_eta,
            gp_kernel=gp_kernel,
            obs_sigma_scale=obs_sigma_scale,
            sampler_backend=sampler_backend,
            draws=draws,
            tune=tune,
            chains=chains,
            seed=seed + 2,
        )
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Active learning: rank extra-sweep candidates for FSS Ising datasets"
    )
    parser.add_argument(
        "--dataset",
        choices=list_datasets(),
        required=True,
        help="Named dataset",
    )
    parser.add_argument("--data", type=Path, default=None, help="Observables CSV path")
    parser.add_argument(
        "--posterior",
        type=Path,
        default=None,
        help="Posterior NetCDF path (default: posteriors/posterior_{dataset}.nc)",
    )
    parser.add_argument(
        "--extra-sweeps",
        type=str,
        default=None,
        help="Comma-separated extra sweep counts (default: dataset-based grid)",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Rows to print")
    parser.add_argument(
        "--max-is-draws",
        type=int,
        default=500,
        help="Subsample posterior to this many draws for scoring (0 = all)",
    )
    parser.add_argument(
        "--n-y-samples",
        type=int,
        default=8,
        help="Predictive MC samples per candidate for E[uncertainty]",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--execute-top",
        type=int,
        default=0,
        help="If >0, run Wolff for the top candidate and merge observables",
    )
    parser.add_argument(
        "--no-refit",
        action="store_true",
        help="Skip posterior refit after executing acquisition",
    )
    parser.add_argument("--infer-Tc", action="store_true", default=True)
    parser.add_argument("--no-infer-Tc", action="store_false", dest="infer_Tc")
    parser.add_argument("--infer-nu", action="store_true", default=True)
    parser.add_argument("--no-infer-nu", action="store_false", dest="infer_nu")
    parser.add_argument("--infer-beta", action="store_true", default=False)
    parser.add_argument("--no-infer-beta", action="store_false", dest="infer_beta")
    parser.add_argument("--use-m", action="store_true", default=False)
    parser.add_argument("--use-m2", action="store_true", default=False)
    parser.add_argument("--use-m4", action="store_true", default=False)
    parser.add_argument("--use-binder", action="store_true", default=True)
    parser.add_argument("--use-chi", action="store_true", default=False)
    parser.add_argument("--correction-binder", action="store_true", default=False)
    parser.add_argument("--obs-sigma-scale", type=float, default=None)
    parser.add_argument("--gp-kernel", type=str, default=None)
    parser.add_argument("--sampler-backend", choices=("jax", "laps", "pymc"), default="jax")
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--tune", type=int, default=500)
    parser.add_argument("--chains", type=int, default=None)
    args = parser.parse_args()

    grid = _parse_extra_sweeps_grid(args.extra_sweeps) if args.extra_sweeps else None
    acquire_dataset(
        args.dataset,
        dry_run=args.execute_top <= 0,
        execute_top=args.execute_top,
        extra_sweeps_grid=grid,
        top_k=args.top_k,
        max_is_draws=None if args.max_is_draws <= 0 else args.max_is_draws,
        n_y_samples=args.n_y_samples,
        seed=args.seed,
        data=args.data,
        posterior=args.posterior,
        infer_Tc=args.infer_Tc,
        infer_nu=args.infer_nu,
        infer_beta=args.infer_beta,
        use_m=args.use_m,
        use_m2=args.use_m2,
        use_m4=args.use_m4,
        use_binder=args.use_binder,
        use_chi=args.use_chi,
        correction_binder=args.correction_binder,
        obs_sigma_scale=args.obs_sigma_scale,
        gp_kernel=args.gp_kernel,
        refit=not args.no_refit,
        sampler_backend=args.sampler_backend,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
    )


if __name__ == "__main__":
    main()
