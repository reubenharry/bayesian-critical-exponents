#!/usr/bin/env python3
"""LAPS discrepancy demo on ``harada_square_large_t`` (2000 chains).

Samples with JAX LAPS, plots posteriors of (t0, L0, p, q), and annotates the
Binder collapse with a(·, L) curves per lattice size.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

from .constants import NU_EXACT, TC_EXACT
from .datasets import observables_path, plots_dir, posterior_path
from .discrepancy import discrepancy_amplitude, effective_obs_sigma
from .fit import fit_dataset
from .observables import read_observables_for_fss, sigma_from_mc_batch


def _plot_disc_posteriors(idata: az.InferenceData, out: Path) -> None:
    names = ["disc_t0", "disc_L0", "disc_p", "disc_q"]
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.0), constrained_layout=True)
    for ax, name in zip(axes.ravel(), names):
        samples = idata.posterior[name].values.ravel()
        ax.hist(samples, bins=40, density=True, color="C0", alpha=0.85, edgecolor="none")
        mean = float(np.mean(samples))
        ax.axvline(mean, color="C3", linestyle="--", linewidth=1.2, label=rf"mean={mean:.3g}")
        ax.set_xlabel(name.replace("disc_", ""))
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
    fig.suptitle("Discrepancy-parameter posteriors (LAPS)")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_binder_with_a(
    df,
    idata: az.InferenceData,
    *,
    out: Path,
) -> None:
    post = idata.posterior
    Tc = float(post["T_c"].mean()) if "T_c" in post else TC_EXACT
    nu = float(post["nu"].mean()) if "nu" in post else NU_EXACT
    t0 = float(post["disc_t0"].mean())
    L0 = float(post["disc_L0"].mean())
    p = float(post["disc_p"].mean())
    q = float(post["disc_q"].mean())

    L_arr = df["L"].to_numpy(dtype=np.float64)
    T_arr = df["T"].to_numpy(dtype=np.float64)
    u4 = df["binder_cumulant"].to_numpy(dtype=np.float64)
    t = (T_arr - Tc) / Tc
    z = t * L_arr ** (1.0 / nu)
    a_pts = discrepancy_amplitude(np.abs(t), L_arr, t0=t0, L0=L0, p=p, q=q)
    # Visual weight: opaque near a~0, faded when discrepancy inflates noise.
    alpha_pts = 1.0 / (1.0 + a_pts)
    size_pts = 18.0 + 40.0 * a_pts / (1.0 + a_pts)

    colors = {
        8: "C4",
        12: "C5",
        16: "C6",
        24: "C8",
        32: "C9",
        64: "C0",
        128: "C2",
        256: "C3",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), constrained_layout=True)

    ax = axes[0]
    for Lval in sorted(df["L"].unique()):
        mask = L_arr == Lval
        a_L0 = float(
            discrepancy_amplitude(0.0, float(Lval), t0=t0, L0=L0, p=p, q=q)
        )
        ax.scatter(
            z[mask],
            u4[mask],
            c=colors.get(int(Lval), "k"),
            s=size_pts[mask],
            alpha=np.clip(alpha_pts[mask], 0.15, 0.95),
            edgecolors="none",
            label=rf"$L={int(Lval)}$ ($a(0)={a_L0:.2g}$)",
        )
    ax.axvline(0.0, color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel(r"$z = t L^{1/\nu}$")
    ax.set_ylabel(r"$U_4$")
    ax.set_title(r"Binder vs $z$ (marker size$\propto a$, opacity$\propto 1/(1+a)$)")
    ax.legend(fontsize=6, ncol=2, loc="best")
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    abs_t = np.linspace(0.0, 1.5, 300)
    for Lval in sorted(df["L"].unique()):
        a = discrepancy_amplitude(abs_t, float(Lval), t0=t0, L0=L0, p=p, q=q)
        ax.plot(
            abs_t,
            a,
            color=colors.get(int(Lval), "k"),
            label=rf"$L={int(Lval)}$",
        )
    ax.axvline(0.1, color="gray", linestyle=":", alpha=0.7, label=r"$|t|=0.1$")
    ax.set_xlabel(r"$|t|$")
    ax.set_ylabel(r"$a(|t|, L)$")
    ax.set_title(
        rf"Posterior-mean $a$: $t_0={t0:.3g}$, $L_0={L0:.3g}$, $p={p:.3g}$, $q={q:.3g}$"
    )
    ax.legend(fontsize=6, ncol=2, loc="best")
    ax.grid(True, alpha=0.25)

    fig.suptitle(
        rf"Discrepancy on binder ($T_c={Tc:.4f}$, $\nu={nu:.3g}$, LAPS)"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _disc_posterior_means(idata: az.InferenceData) -> dict[str, float]:
    post = idata.posterior
    return {
        "Tc": float(post["T_c"].mean()) if "T_c" in post else TC_EXACT,
        "nu": float(post["nu"].mean()) if "nu" in post else NU_EXACT,
        "t0": float(post["disc_t0"].mean()),
        "L0": float(post["disc_L0"].mean()),
        "p": float(post["disc_p"].mean()),
        "q": float(post["disc_q"].mean()),
        "sigma_model": float(post["disc_sigma_model"].mean()),
    }


def _plot_binder_errors_with_a(
    df,
    idata: az.InferenceData,
    *,
    out: Path,
) -> None:
    """Binder vs z with MC-only vs σ_eff = sqrt(σ_MC² + (σ_model a)²) error bars."""
    m = _disc_posterior_means(idata)
    L_arr = df["L"].to_numpy(dtype=np.float64)
    T_arr = df["T"].to_numpy(dtype=np.float64)
    u4 = df["binder_cumulant"].to_numpy(dtype=np.float64)
    t = (T_arr - m["Tc"]) / m["Tc"]
    z = t * L_arr ** (1.0 / m["nu"])
    sigma_mc = sigma_from_mc_batch(
        df["binder_cumulant_std"].to_numpy(dtype=np.float64),
        df["n_eff_binder"].to_numpy(dtype=np.float64),
    )
    sigma_eff = effective_obs_sigma(
        sigma_mc,
        t,
        L_arr,
        t0=m["t0"],
        L0=m["L0"],
        p=m["p"],
        q=m["q"],
        sigma_model=m["sigma_model"],
    )

    colors = {
        8: "C4",
        12: "C5",
        16: "C6",
        24: "C8",
        32: "C9",
        64: "C0",
        128: "C2",
        256: "C3",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=True, constrained_layout=True)
    panels = [
        (axes[0], sigma_mc, r"MC only: $\sigma_{\mathrm{MC}}$"),
        (
            axes[1],
            sigma_eff,
            r"with $a$: $\sigma_{\mathrm{eff}}=\sqrt{\sigma_{\mathrm{MC}}^2+(\sigma_{\mathrm{model}}a)^2}$",
        ),
    ]
    # Keep both panels on the physical U_4 scale so inflation is visible as
    # bars that blow past the data range (rather than rescaling the left panel).
    y_lo = float(np.min(u4 - sigma_mc)) - 0.05
    y_hi = float(np.max(u4 + sigma_mc)) + 0.05
    y_lo = min(y_lo, -0.15)
    y_hi = max(y_hi, 0.75)
    for ax, sigma, title in panels:
        for Lval in sorted(df["L"].unique()):
            mask = L_arr == Lval
            order = np.argsort(z[mask])
            ax.errorbar(
                z[mask][order],
                u4[mask][order],
                yerr=sigma[mask][order],
                fmt="o",
                color=colors.get(int(Lval), "k"),
                markersize=3.5,
                alpha=0.85,
                elinewidth=0.9,
                capsize=1.5,
                label=rf"$L={int(Lval)}$",
            )
        ax.axvline(0.0, color="gray", linestyle=":", alpha=0.7)
        ax.set_xlabel(r"$z = t L^{1/\nu}$")
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=6, ncol=2, loc="best")
    axes[0].set_ylabel(r"$U_4$")
    # Annotate how many σ_eff bars exceed the displayed window.
    n_clip = int(np.sum((u4 - sigma_eff < y_lo) | (u4 + sigma_eff > y_hi)))
    axes[1].text(
        0.02,
        0.02,
        rf"{n_clip}/{len(u4)} points have $\sigma_{{\mathrm{{eff}}}}$ bars clipped by this $y$-window",
        transform=axes[1].transAxes,
        fontsize=7,
        va="bottom",
        ha="left",
        color="0.3",
    )
    fig.suptitle(
        rf"Binder errors with discrepancy "
        rf"($T_c={m['Tc']:.4f}$, $\nu={m['nu']:.3g}$, "
        rf"$t_0={m['t0']:.3g}$, $L_0={m['L0']:.3g}$, "
        rf"$p={m['p']:.3g}$, $q={m['q']:.3g}$, "
        rf"$\sigma_{{\mathrm{{model}}}}={m['sigma_model']:.3g}$)"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def run_demo(
    *,
    chains: int = 2000,
    tune: int = 200,
    draws: int = 50,
    seed: int = 1,
) -> list[Path]:
    dataset = "harada_square_large_t"
    data_path = observables_path(dataset)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}")

    out_nc = posterior_path(dataset).with_name(
        f"posterior_{dataset}_discrepancy_laps.nc"
    )
    print(
        f"LAPS fit: {chains} chains, tune={tune}, draws={draws} -> {out_nc}",
        flush=True,
    )
    fit_dataset(
        dataset,
        data=data_path,
        out=out_nc,
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
        discrepancy_binder=True,
        infer_Tc=True,
        infer_nu=True,
        infer_beta=False,
        reparametrize_Tc=True,
        sampler_backend="laps",
        draws=draws,
        tune=tune,
        chains=chains,
        seed=seed,
        max_eta_hours=None,
        profile_mle_init=False,
    )

    idata = az.from_netcdf(out_nc)
    print(az.summary(idata, var_names=["T_c", "nu", "disc_t0", "disc_L0", "disc_p", "disc_q", "disc_sigma_model"]))

    df = read_observables_for_fss(
        data_path,
        use_m=False,
        use_m2=False,
        use_m4=False,
        use_binder=True,
        use_chi=False,
    )
    out_dir = plots_dir(dataset)
    paths = [
        out_dir / "disc_param_posteriors_laps.png",
        out_dir / "binder_with_a_laps.png",
        out_dir / "binder_errors_with_a_laps.png",
    ]
    _plot_disc_posteriors(idata, paths[0])
    _plot_binder_with_a(df, idata, out=paths[1])
    _plot_binder_errors_with_a(df, idata, out=paths[2])
    for p in paths:
        print(f"wrote {p}", flush=True)
    return paths


def plot_from_posterior(
    *,
    posterior: Path | None = None,
    data: Path | None = None,
) -> list[Path]:
    """Regenerate discrepancy plots from an existing LAPS posterior."""
    dataset = "harada_square_large_t"
    out_nc = posterior or posterior_path(dataset).with_name(
        f"posterior_{dataset}_discrepancy_laps.nc"
    )
    data_path = data or observables_path(dataset)
    idata = az.from_netcdf(out_nc)
    df = read_observables_for_fss(
        data_path,
        use_m=False,
        use_m2=False,
        use_m4=False,
        use_binder=True,
        use_chi=False,
    )
    out_dir = plots_dir(dataset)
    paths = [
        out_dir / "disc_param_posteriors_laps.png",
        out_dir / "binder_with_a_laps.png",
        out_dir / "binder_errors_with_a_laps.png",
    ]
    _plot_disc_posteriors(idata, paths[0])
    _plot_binder_with_a(df, idata, out=paths[1])
    _plot_binder_errors_with_a(df, idata, out=paths[2])
    for p in paths:
        print(f"wrote {p}", flush=True)
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chains", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=200)
    parser.add_argument("--draws", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip sampling; regenerate plots from existing LAPS posterior.",
    )
    args = parser.parse_args(argv)
    if args.plot_only:
        plot_from_posterior()
    else:
        run_demo(chains=args.chains, tune=args.tune, draws=args.draws, seed=args.seed)


if __name__ == "__main__":
    main()
