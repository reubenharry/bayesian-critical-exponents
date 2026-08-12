#!/usr/bin/env python3
"""Binder diagnostics + binder-only FSS fit visualizations for a φ⁴ dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ising.plot_binder import plot_binder_dataset_diagnostics
from ising.plot_posterior import (
    plot_exponent_histograms,
    plot_posterior_traces,
)
from phi4.constants import BETA_EXACT, LAM_C, NU_EXACT
from phi4.datasets import (
    observables_path,
    plots_dir,
    posterior_path,
)
from phi4.fss_io import TC_PRIOR_LOWER, TC_PRIOR_UPPER


def _plot_joint_and_marginals(idata: az.InferenceData, out_dir: Path, *, label: str) -> list[Path]:
    lam = np.asarray(idata.posterior["T_c"]).ravel()
    nu = np.asarray(idata.posterior["nu"]).ravel()
    lam_mean, lam_sd = float(lam.mean()), float(lam.std(ddof=1))
    nu_mean, nu_sd = float(nu.mean()), float(nu.std(ddof=1))
    hdi_lam = az.hdi(lam, hdi_prob=0.94)
    hdi_nu = az.hdi(nu, hdi_prob=0.94)
    paths: list[Path] = []

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    for ax, samples, truth, name, mean, sd, hdi in (
        (axes[0], lam, LAM_C, r"$\lambda_c$", lam_mean, lam_sd, hdi_lam),
        (axes[1], nu, NU_EXACT, r"$\nu$", nu_mean, nu_sd, hdi_nu),
    ):
        ax.hist(samples, bins=40, density=True, color="0.75", edgecolor="0.45", lw=0.6)
        ax.axvline(truth, color="C3", ls="--", lw=1.4, label=f"truth {truth:g}")
        ax.axvline(mean, color="C0", ls="-", lw=1.4, label=f"mean {mean:.3f}")
        ax.axvspan(hdi[0], hdi[1], color="C0", alpha=0.15, label="94% HDI")
        ax.set_xlabel(name)
        ax.set_ylabel("density")
        ax.legend(frameon=False, fontsize=8)
        ax.set_title(rf"{name}: ${mean:.3f}\pm{sd:.3f}$")
    fig.suptitle(rf"φ⁴ {label} · Binder-only posterior ($\beta$ fixed $=1/8$)")
    fig.tight_layout()
    marg = out_dir / "posterior_marginals_lam_nu.png"
    fig.savefig(marg, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(marg)

    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.scatter(lam, nu, s=6, alpha=0.25, c="0.35", edgecolors="none", rasterized=True)
    H, xedges, yedges = np.histogram2d(lam, nu, bins=35)
    ax.contour(
        0.5 * (xedges[:-1] + xedges[1:]),
        0.5 * (yedges[:-1] + yedges[1:]),
        H.T,
        levels=6,
        colors="C0",
        linewidths=0.9,
        alpha=0.85,
    )
    ax.axvline(LAM_C, color="C3", ls="--", lw=1.2, label=rf"truth $\lambda_c={LAM_C:g}$")
    ax.axhline(NU_EXACT, color="C3", ls=":", lw=1.2, label=rf"truth $\nu={NU_EXACT:g}$")
    ax.plot([lam_mean], [nu_mean], "o", color="C0", ms=6, label="posterior mean")
    ax.set_xlabel(r"$\lambda_c$ (FSS $T_c$)")
    ax.set_ylabel(r"$\nu$")
    ax.set_title(f"Joint posterior: Binder · φ⁴ {label}")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    joint = out_dir / "posterior_joint_lam_nu.png"
    fig.savefig(joint, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(joint)
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="L64_128")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--posterior", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--fit", action="store_true", help="Also run binder-only JAX fit")
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--tune", type=int, default=500)
    parser.add_argument("--chains", type=int, default=4)
    args = parser.parse_args(argv)

    data_path = args.data or observables_path(args.dataset)
    post_path = args.posterior or posterior_path(args.dataset, variant="binder")
    # Use plain name postfix via out dir
    out_dir = args.out or plots_dir(args.dataset, variant="binder")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} points from {data_path}")
    print(
        f"  n_eff(m): min={df['n_eff'].min():.1f}, "
        f"median={df['n_eff'].median():.1f}, max={df['n_eff'].max():.1f}"
    )

    binder_paths = plot_binder_dataset_diagnostics(
        df,
        out_dir,
        T_c_exact=LAM_C,
        nu_exact=NU_EXACT,
        L_ref=128.0,
        dataset=f"phi4/{args.dataset}",
    )
    print("Binder plots:")
    for p in binder_paths:
        print(f"  {p}")

    if args.fit or not post_path.is_file():
        from ising.fit import fit_dataset

        post_path.parent.mkdir(parents=True, exist_ok=True)
        # Reuse Ising 'medium' name only for path helpers; --data/--out override.
        fit_dataset(
            "medium",
            infer_Tc=True,
            infer_nu=True,
            infer_beta=False,
            use_m=False,
            use_binder=True,
            data=data_path,
            out=post_path,
            tc_prior_lower=TC_PRIOR_LOWER,
            tc_prior_upper=TC_PRIOR_UPPER,
            tc_init=LAM_C,
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            sampler_backend="jax",
            jax_platform="cpu",
            seed=1,
        )

    idata = az.from_netcdf(post_path)
    print(az.summary(idata, var_names=["T_c", "nu", "gamma"], kind="all", hdi_prob=0.94))
    print(f"truth: lam_c={LAM_C}, nu={NU_EXACT}, beta={BETA_EXACT}")

    means = {
        "T_c": float(np.asarray(idata.posterior["T_c"]).mean()),
        "nu": float(np.asarray(idata.posterior["nu"]).mean()),
    }
    post_binder = plot_binder_dataset_diagnostics(
        df,
        out_dir,
        T_c_post=means["T_c"],
        nu_post=means["nu"],
        T_c_exact=LAM_C,
        nu_exact=NU_EXACT,
        L_ref=128.0,
        dataset=f"phi4/{args.dataset}",
    )
    exp_path = plot_exponent_histograms(idata, out_dir)
    trace_paths = plot_posterior_traces(idata, out_dir)
    if isinstance(trace_paths, Path):
        trace_list = [trace_paths]
    elif trace_paths is None:
        trace_list = []
    else:
        trace_list = list(trace_paths)
    joint_paths = _plot_joint_and_marginals(idata, out_dir, label=args.dataset)

    print("Posterior / standard plots:")
    for p in [*post_binder, exp_path, *trace_list, *joint_paths]:
        if p is not None:
            print(f"  {p}")


if __name__ == "__main__":
    main()
