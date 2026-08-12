#!/usr/bin/env python3
"""Fit and plot dynamical n_eff(L, t) scaling on an observables table."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .constants import NU_EXACT, TC_EXACT
from .neff_scaling import (
    NeffScalingFit,
    fit_neff_scaling,
    neff_column,
    predict_collapse_gp,
    predict_neff,
)
from .observables import read_observables

_HARADA_L_STYLE: dict[int, dict[str, object]] = {
    64: {"color": "C0", "marker": "*", "markersize": 9, "label": r"$L=64$"},
    128: {"color": "C2", "marker": "x", "markersize": 7, "label": r"$L=128$"},
    256: {"color": "C3", "marker": "+", "markersize": 9, "label": r"$L=256$"},
}


def _style_for_L(L: int) -> dict[str, object]:
    if L in _HARADA_L_STYLE:
        return dict(_HARADA_L_STYLE[L])
    return {
        "color": f"C{int(L) % 10}",
        "marker": "o",
        "markersize": 6,
        "label": rf"$L={L}$",
    }


def plot_neff_collapse(
    fit: NeffScalingFit,
    *,
    out: Path | None = None,
    figsize: tuple[float, float] = (7.0, 4.5),
) -> plt.Figure:
    """Collapsed log(n_eff/N) + z_dyn log L vs z_coll, with GP mean band."""
    fig, ax = plt.subplots(figsize=figsize)
    z_grid = np.linspace(
        float(fit.z_coll_train.min()), float(fit.z_coll_train.max()), 400
    )
    g_mean, g_std = predict_collapse_gp(fit, z_grid)
    ax.fill_between(
        z_grid,
        g_mean - 2.0 * g_std,
        g_mean + 2.0 * g_std,
        color="0.75",
        alpha=0.5,
        label=r"GP $\pm 2\sigma$",
    )
    ax.plot(z_grid, g_mean, color="0.2", lw=1.5, label=r"GP mean $g(z)$")

    for L_val in sorted(np.unique(fit.L_train)):
        mask = fit.L_train == L_val
        style = _style_for_L(int(L_val))
        ax.plot(
            fit.z_coll_train[mask],
            fit.y_collapse_train[mask],
            linestyle="none",
            **style,
        )

    ax.set_xlabel(r"$z = t L^{1/\nu}$")
    ax.set_ylabel(
        rf"$\log(n_{{\mathrm{{eff}}}}/N) + z_{{\mathrm{{dyn}}}}\log L$"
        + "\n"
        + rf"(mean $z_{{\mathrm{{dyn}}}}={fit.z_dyn_mean:.3f}$)"
    )
    ax.set_title(f"Dynamical n_eff collapse ({fit.channel})")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    return fig


def plot_z_dyn_histogram(
    fit: NeffScalingFit,
    *,
    out: Path | None = None,
    figsize: tuple[float, float] = (6.5, 4.2),
    bins: int = 60,
) -> plt.Figure:
    """Histogram of NUTS posterior draws of the dynamical exponent z_dyn."""
    fig, ax = plt.subplots(figsize=figsize)
    samples = fit.z_dyn_samples
    ax.hist(
        samples,
        bins=bins,
        density=True,
        color="C0",
        edgecolor="white",
        linewidth=0.4,
        alpha=0.85,
        label=f"NUTS draws ({fit.sampler})",
    )

    ax.axvline(
        fit.z_dyn_map,
        color="C3",
        ls="-",
        lw=1.4,
        label=rf"MAP $= {fit.z_dyn_map:.3f}$",
    )
    ax.axvline(
        fit.z_dyn_mean,
        color="C2",
        ls="--",
        lw=1.2,
        label=rf"mean $= {fit.z_dyn_mean:.3f}$",
    )
    ax.axvspan(
        fit.z_dyn_hdi_low,
        fit.z_dyn_hdi_high,
        color="C1",
        alpha=0.18,
        label=(
            rf"94% HDI $[{fit.z_dyn_hdi_low:.3f},"
            rf"\,{fit.z_dyn_hdi_high:.3f}]$"
        ),
    )
    ax.set_xlabel(r"$z_{\mathrm{dyn}}$")
    ax.set_ylabel("density")
    ax.set_title(
        rf"Dynamical exponent posterior "
        rf"($\mathrm{{std}}={fit.z_dyn_std:.3f}$, $n={samples.size}$)"
    )
    pad = max(4.0 * fit.z_dyn_std, 0.05)
    ax.set_xlim(fit.z_dyn_mean - pad, fit.z_dyn_mean + pad)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    return fig


def plot_noise_histogram(
    fit: NeffScalingFit,
    *,
    out: Path | None = None,
    figsize: tuple[float, float] = (6.5, 4.2),
    bins: int = 60,
) -> plt.Figure:
    """Histogram of NUTS posterior draws of shared observation noise σ."""
    fig, ax = plt.subplots(figsize=figsize)
    samples = fit.noise_sigma_samples
    ax.hist(
        samples,
        bins=bins,
        density=True,
        color="C0",
        edgecolor="white",
        linewidth=0.4,
        alpha=0.85,
        label="NUTS draws",
    )
    ax.axvline(
        fit.noise_sigma_map,
        color="C3",
        ls="-",
        lw=1.4,
        label=rf"MAP $= {fit.noise_sigma_map:.3f}$",
    )
    ax.axvline(
        fit.noise_sigma_mean,
        color="C2",
        ls="--",
        lw=1.2,
        label=rf"mean $= {fit.noise_sigma_mean:.3f}$",
    )
    ax.axvspan(
        fit.noise_sigma_hdi_low,
        fit.noise_sigma_hdi_high,
        color="C1",
        alpha=0.18,
        label=(
            rf"94% HDI $[{fit.noise_sigma_hdi_low:.3f},"
            rf"\,{fit.noise_sigma_hdi_high:.3f}]$"
        ),
    )
    ax.axvline(
        fit.noise_prior_scale,
        color="0.4",
        ls=":",
        lw=1.2,
        label=rf"HalfNormal scale $= {fit.noise_prior_scale:.3f}$",
    )
    ax.set_xlabel(r"observation noise $\sigma$ (log $n_{\mathrm{eff}}$ scale)")
    ax.set_ylabel("density")
    ax.set_title(
        rf"Observation-noise posterior "
        rf"($\mathrm{{std}}={fit.noise_sigma_std:.3f}$, $n={samples.size}$)"
    )
    pad = max(4.0 * fit.noise_sigma_std, 0.01)
    ax.set_xlim(
        max(0.0, fit.noise_sigma_mean - pad),
        fit.noise_sigma_mean + pad,
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    return fig


def plot_neff_vs_temperature(
    fit: NeffScalingFit,
    *,
    out: Path | None = None,
    figsize: tuple[float, float] = (7.0, 4.5),
    n_curve: int = 200,
) -> plt.Figure:
    """Plot n_eff vs T for each L, with model prediction curves."""
    fig, ax = plt.subplots(figsize=figsize)

    for L_val in sorted(np.unique(fit.L_train)):
        style = _style_for_L(int(L_val))
        mask = fit.L_train == L_val
        order = np.argsort(fit.T_train[mask])
        T_pts = fit.T_train[mask][order]
        n_pts = fit.n_eff_train[mask][order]
        # Use typical n_sweeps for this L (constant on harada_square).
        n_sweeps_L = float(np.median(fit.n_sweeps_train[mask]))

        ax.plot(
            T_pts,
            n_pts,
            linestyle="none",
            zorder=3,
            **style,
        )
        # Curve only over the observed T span for this L (avoid empty-band extrapolations).
        T_grid = np.linspace(float(T_pts.min()), float(T_pts.max()), n_curve)
        L_grid = np.full(T_grid.shape, float(L_val))
        n_hat, n_std = predict_neff(fit, L_grid, T_grid, n_sweeps=n_sweeps_L)
        line_color = style["color"]
        ax.plot(
            T_grid,
            n_hat,
            color=line_color,
            lw=1.4,
            alpha=0.9,
            zorder=2,
        )
        ax.fill_between(
            T_grid,
            np.maximum(n_hat - 2.0 * n_std, 1.0),
            n_hat + 2.0 * n_std,
            color=line_color,
            alpha=0.15,
            zorder=1,
        )

    ax.axvline(fit.T_c, color="0.35", ls="--", lw=1.0, label=r"$T_c$")
    ax.set_xlabel(r"$T$")
    ax.set_ylabel(r"$n_{\mathrm{eff}}$")
    ax.set_title(f"$n_{{\\mathrm{{eff}}}}$ vs $T$ ({fit.channel})")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    return fig


def plot_neff_predicted_vs_actual(
    fit: NeffScalingFit,
    *,
    out: Path | None = None,
    figsize: tuple[float, float] = (6.5, 5.5),
) -> plt.Figure:
    """Predicted vs actual log n_eff and residuals by L."""
    n_eff_hat, _ = predict_neff(
        fit, fit.L_train, fit.T_train, n_sweeps=fit.n_sweeps_train
    )
    log_actual = np.log(np.maximum(fit.n_eff_train, 1.0))
    log_pred = np.log(np.maximum(n_eff_hat, 1.0))
    resid = log_pred - log_actual

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    ax0, ax1 = axes

    lo = float(min(log_actual.min(), log_pred.min()))
    hi = float(max(log_actual.max(), log_pred.max()))
    ax0.plot([lo, hi], [lo, hi], color="0.5", ls="--", lw=1.0)
    for L_val in sorted(np.unique(fit.L_train)):
        mask = fit.L_train == L_val
        style = _style_for_L(int(L_val))
        ax0.plot(log_actual[mask], log_pred[mask], linestyle="none", **style)
    ax0.set_xlabel(r"$\log n_{\mathrm{eff}}$ (actual)")
    ax0.set_ylabel(r"$\log n_{\mathrm{eff}}$ (predicted)")
    ax0.set_title("Predicted vs actual")
    ax0.legend(loc="best", fontsize=8)
    ax0.grid(True, alpha=0.3)

    for L_val in sorted(np.unique(fit.L_train)):
        mask = fit.L_train == L_val
        style = _style_for_L(int(L_val))
        ax1.plot(fit.z_coll_train[mask], resid[mask], linestyle="none", **style)
    ax1.axhline(0.0, color="0.4", ls="--", lw=1.0)
    ax1.set_xlabel(r"$z = t L^{1/\nu}$")
    ax1.set_ylabel(r"$\log\hat n_{\mathrm{eff}} - \log n_{\mathrm{eff}}$")
    ax1.set_title("Residuals vs collapse $z$")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True, alpha=0.3)

    fig.suptitle(
        rf"$z_{{\mathrm{{dyn}}}}={fit.z_dyn_mean:.3f}\pm{fit.z_dyn_std:.3f}$, "
        rf"$\sigma={fit.noise_sigma_mean:.3f}\pm{fit.noise_sigma_std:.3f}$, "
        rf"ℓ={fit.length_scale:.3g}, η={fit.amplitude:.3g} (fixed)"
    )
    fig.tight_layout()
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    return fig


def write_fit_summary(fit: NeffScalingFit, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_eff_hat, _ = predict_neff(
        fit, fit.L_train, fit.T_train, n_sweeps=fit.n_sweeps_train
    )
    log_resid = np.log(np.maximum(n_eff_hat, 1.0)) - np.log(
        np.maximum(fit.n_eff_train, 1.0)
    )
    lines = [
        f"channel={fit.channel}",
        f"n_points={fit.z_coll_train.size}",
        f"sampler={fit.sampler}",
        f"T_c={fit.T_c:.8f}",
        f"nu={fit.nu:.8f}",
        f"z_dyn_mean={fit.z_dyn_mean:.6f}",
        f"z_dyn_std={fit.z_dyn_std:.6f}",
        f"z_dyn_map={fit.z_dyn_map:.6f}",
        f"z_dyn_hdi94_low={fit.z_dyn_hdi_low:.6f}",
        f"z_dyn_hdi94_high={fit.z_dyn_hdi_high:.6f}",
        f"noise_sigma_mean={fit.noise_sigma_mean:.6f}",
        f"noise_sigma_std={fit.noise_sigma_std:.6f}",
        f"noise_sigma_map={fit.noise_sigma_map:.6f}",
        f"noise_sigma_hdi94_low={fit.noise_sigma_hdi_low:.6f}",
        f"noise_sigma_hdi94_high={fit.noise_sigma_hdi_high:.6f}",
        f"noise_prior_scale={fit.noise_prior_scale:.6f}",
        f"n_posterior_samples={fit.z_dyn_samples.size}",
        f"length_scale_fixed={fit.length_scale:.6g}",
        f"amplitude_fixed={fit.amplitude:.6g}",
        f"kernel={fit.kernel}",
        f"lml_at_mean_params={fit.log_marginal_likelihood:.6f}",
        f"z_coll_min={fit.z_coll_train.min():.6g}",
        f"z_coll_max={fit.z_coll_train.max():.6g}",
        f"rmse_log_neff={float(np.sqrt(np.mean(log_resid**2))):.6g}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Observables CSV (default: data/observables_harada_square.csv)",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default="binder",
        choices=sorted({"binder", "m", "m2", "m4", "susceptibility", "log_m"}),
        help="Which n_eff column to model (default: binder)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for plots (default: plots/<data-stem>/)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=60,
        help="Number of histogram bins for z_dyn posterior (default: 60)",
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=4,
        help="NUTS chains (default: 4)",
    )
    parser.add_argument(
        "--tune",
        type=int,
        default=500,
        help="NUTS warmup steps (default: 500)",
    )
    parser.add_argument(
        "--draws",
        type=int,
        default=1000,
        help="NUTS draws per chain (default: 1000)",
    )
    parser.add_argument(
        "--z-dyn-max",
        type=float,
        default=2.0,
        help="Upper bound of z_dyn prior (default: 2)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save figures only; do not open an interactive window",
    )
    args = parser.parse_args(argv)

    from .datasets import DATA_DIR, PLOTS_DIR

    data_path = args.data or (DATA_DIR / "observables_harada_square.csv")
    df = read_observables(data_path)
    col = neff_column(args.channel)  # type: ignore[arg-type]
    if col not in df.columns:
        raise SystemExit(f"Column {col!r} missing in {data_path}")

    fit = fit_neff_scaling(
        df,
        channel=args.channel,  # type: ignore[arg-type]
        T_c=TC_EXACT,
        nu=NU_EXACT,
        z_dyn_max=args.z_dyn_max,
        chains=args.chains,
        tune=args.tune,
        draws=args.draws,
    )
    stem = data_path.stem.replace(".partial", "")
    out_dir = args.out_dir or (PLOTS_DIR / stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    collapse_path = out_dir / f"neff_{args.channel}_collapse.png"
    hist_path = out_dir / f"neff_{args.channel}_z_dyn_hist.png"
    noise_path = out_dir / f"neff_{args.channel}_noise_hist.png"
    vs_T_path = out_dir / f"neff_{args.channel}_vs_T.png"
    pred_path = out_dir / f"neff_{args.channel}_pred_vs_actual.png"
    summary_path = out_dir / f"neff_{args.channel}_fit.txt"

    plot_neff_collapse(fit, out=collapse_path)
    plot_z_dyn_histogram(fit, out=hist_path, bins=args.bins)
    plot_noise_histogram(fit, out=noise_path, bins=args.bins)
    plot_neff_vs_temperature(fit, out=vs_T_path)
    plot_neff_predicted_vs_actual(fit, out=pred_path)
    write_fit_summary(fit, summary_path)

    print(
        f"z_dyn mean={fit.z_dyn_mean:.4f}±{fit.z_dyn_std:.4f}  "
        f"MAP={fit.z_dyn_map:.4f}  "
        f"94% HDI=[{fit.z_dyn_hdi_low:.4f}, {fit.z_dyn_hdi_high:.4f}]  "
        f"({fit.sampler})"
    )
    print(
        f"sigma mean={fit.noise_sigma_mean:.4f}±{fit.noise_sigma_std:.4f}  "
        f"MAP={fit.noise_sigma_map:.4f}  "
        f"94% HDI=[{fit.noise_sigma_hdi_low:.4f}, {fit.noise_sigma_hdi_high:.4f}]  "
        f"(HalfNormal scale={fit.noise_prior_scale:.4f})"
    )
    print(
        f"fixed kernel: ell={fit.length_scale:.4g}  eta={fit.amplitude:.4g}"
    )
    print(f"Wrote {collapse_path}")
    print(f"Wrote {hist_path}")
    print(f"Wrote {noise_path}")
    print(f"Wrote {vs_T_path}")
    print(f"Wrote {pred_path}")
    print(f"Wrote {summary_path}")

    if args.no_show:
        plt.close("all")
    else:
        plt.show()
    return out_dir


if __name__ == "__main__":
    main()
