#!/usr/bin/env python3
"""Plot posterior evolution across greedy active-learning iterations."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._deps import az
from .constants import NU_EXACT, TC_EXACT
from .datasets import ISING_DIR
from .importance_weights import uncertainty_scalar

DEFAULT_RUN_DIR = ISING_DIR / "plots" / "greedy_al_harada_bsa"
_ITER_DIR_RE = re.compile(r"^iter_(\d+)$")


@dataclass(frozen=True)
class IterationPosterior:
    iteration: int
    n_included: int
    tc: np.ndarray
    nu: np.ndarray
    path: Path


def discover_iteration_posteriors(run_dir: Path) -> list[IterationPosterior]:
    """Load ``posterior.nc`` from each ``iter_XX/`` subdirectory."""
    summary_path = run_dir / "summary.csv"
    n_included_by_iter: dict[int, int] = {}
    if summary_path.is_file():
        summary = pd.read_csv(summary_path)
        for row in summary.itertuples(index=False):
            n_included_by_iter[int(row.iteration)] = int(row.n_included)

    out: list[IterationPosterior] = []
    for iter_dir in sorted(run_dir.iterdir()):
        if not iter_dir.is_dir():
            continue
        match = _ITER_DIR_RE.match(iter_dir.name)
        if match is None:
            continue
        nc_path = iter_dir / "posterior.nc"
        if not nc_path.is_file():
            continue
        iteration = int(match.group(1))
        idata = az.from_netcdf(nc_path)
        tc = np.asarray(idata.posterior["T_c"].values.reshape(-1), dtype=np.float64)
        nu = np.asarray(idata.posterior["nu"].values.reshape(-1), dtype=np.float64)
        n_included = n_included_by_iter.get(iteration, iteration + 10)
        out.append(
            IterationPosterior(
                iteration=iteration,
                n_included=n_included,
                tc=tc,
                nu=nu,
                path=nc_path,
            )
        )
    if not out:
        raise FileNotFoundError(f"No iter_XX/posterior.nc files found under {run_dir}")
    return out


def uncertainty_metrics_table(iterations: list[IterationPosterior]) -> pd.DataFrame:
    """Compute tr(cov), Var(T_c), and Var(nu) from saved posterior samples."""
    rows: list[dict[str, float | int]] = []
    for it in iterations:
        mat = np.column_stack([it.tc, it.nu])
        cov = np.cov(mat, rowvar=False)
        rows.append(
            {
                "iteration": it.iteration,
                "n_included": it.n_included,
                "tr_cov": uncertainty_scalar(cov, metric="trace_cov"),
                "var_tc": float(np.var(it.tc)),
                "var_nu": float(np.var(it.nu)),
            }
        )
    return pd.DataFrame(rows).sort_values("iteration").reset_index(drop=True)


def build_prediction_discrepancy_table(run_dir: Path) -> pd.DataFrame:
    """Compare predicted vs actual Δ tr(cov) after each chosen inclusion.

    Predicted values come from the incremental IS model at selection time
    (``summary.csv``). Actual values are measured from consecutive LAPS
    posteriors saved in ``iter_XX/posterior.nc``.
    """
    summary_path = run_dir / "summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing {summary_path}")
    summary = pd.read_csv(summary_path).sort_values("iteration").reset_index(drop=True)
    iterations = discover_iteration_posteriors(run_dir)
    metrics = uncertainty_metrics_table(iterations)
    table = summary.merge(
        metrics[["iteration", "tr_cov"]].rename(columns={"tr_cov": "actual_tr_cov"}),
        on="iteration",
        how="left",
    )
    table["predicted_delta_tr_cov"] = table["delta_tr_cov"].astype(float)
    table["actual_tr_cov_after"] = table["actual_tr_cov"].shift(-1)
    table["actual_delta_tr_cov"] = table["actual_tr_cov"] - table["actual_tr_cov_after"]
    table["discrepancy"] = table["actual_delta_tr_cov"] - table["predicted_delta_tr_cov"]
    pred = table["predicted_delta_tr_cov"].to_numpy(dtype=np.float64)
    table["relative_discrepancy"] = np.where(
        np.abs(pred) > 0.0,
        table["discrepancy"].to_numpy(dtype=np.float64) / pred,
        np.nan,
    )
    return table.loc[table["actual_tr_cov_after"].notna()].reset_index(drop=True)


def plot_prediction_discrepancy(table: pd.DataFrame, path: Path) -> None:
    """Plot predicted vs actual Δ tr(cov) and their residual over AL iterations."""
    it = table["iteration"].to_numpy(dtype=np.int64)
    pred = table["predicted_delta_tr_cov"].to_numpy(dtype=np.float64)
    actual = table["actual_delta_tr_cov"].to_numpy(dtype=np.float64)
    disc = table["discrepancy"].to_numpy(dtype=np.float64)

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.5), constrained_layout=True, sharex=True)

    ax = axes[0]
    ax.plot(it, pred, marker="o", markersize=4, linewidth=1.5, color="C0", label="predicted")
    ax.plot(it, actual, marker="s", markersize=4, linewidth=1.5, color="C1", label="actual")
    ax.set_ylabel(r"$\Delta\,\mathrm{tr}(\mathrm{cov})$")
    ax.set_title("Incremental model vs LAPS posterior after each inclusion")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    ax.axhline(0.0, color="0.55", linewidth=0.9, linestyle="--")
    ax.plot(it, disc, marker="o", markersize=4, linewidth=1.5, color="C3")
    ax.fill_between(it, 0.0, disc, alpha=0.18, color="C3")
    ax.set_ylabel(r"actual $-$ predicted")
    ax.set_xlabel("iteration")
    ax.grid(True, alpha=0.25)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_uncertainty_evolution(metrics: pd.DataFrame, path: Path) -> None:
    """Plot tr(cov) and marginal variances vs number of included points."""
    x = metrics["n_included"].to_numpy(dtype=np.float64)
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 8.0), constrained_layout=True, sharex=True)

    specs = (
        ("tr_cov", r"$\mathrm{tr}(\mathrm{cov})$", "C0"),
        ("var_tc", r"$\mathrm{Var}(T_c)$", "C1"),
        ("var_nu", r"$\mathrm{Var}(\nu)$", "C2"),
    )
    for ax, (col, ylabel, color) in zip(axes, specs):
        y = metrics[col].to_numpy(dtype=np.float64)
        y = np.maximum(y, np.finfo(np.float64).tiny)
        ax.plot(x, y, marker="o", markersize=4, color=color, linewidth=1.5)
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25, which="both")

    axes[-1].set_xlabel("included points")
    axes[0].set_title("Posterior uncertainty vs active-learning progress")
    stride = max(1, len(metrics) // 8)
    for row in metrics.iloc[::stride].itertuples():
        axes[0].annotate(
            f"iter {int(row.iteration):02d}",
            (float(row.n_included), float(row.tr_cov)),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=6,
            color="0.35",
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _shared_xlim(samples_by_iter: list[np.ndarray], *, pad_frac: float = 0.08) -> tuple[float, float]:
    pooled = np.concatenate(samples_by_iter)
    lo, hi = np.percentile(pooled, [0.5, 99.5])
    span = max(float(hi - lo), 1e-6)
    pad = span * pad_frac
    return float(lo - pad), float(hi + pad)


def _ridge_scale(hist: np.ndarray, *, target_peak: float = 0.85) -> np.ndarray:
    peak = float(hist.max()) if hist.size else 0.0
    if peak <= 0.0:
        return hist
    return hist * (target_peak / peak)


def plot_marginal_ridge(
    iterations: list[IterationPosterior],
    *,
    param: str,
    exact: float,
    xlabel: str,
    title: str,
    path: Path,
    n_bins: int = 60,
) -> None:
    """Ridge plot of marginal histograms, one row per iteration."""
    samples_key = "tc" if param == "T_c" else "nu"
    samples_by_iter = [getattr(it, samples_key) for it in iterations]
    xlim = _shared_xlim(samples_by_iter)
    bin_edges = np.linspace(xlim[0], xlim[1], n_bins + 1)

    n = len(iterations)
    fig_h = max(4.5, 0.28 * n + 1.5)
    fig, ax = plt.subplots(figsize=(7.0, fig_h), constrained_layout=True)
    cmap = plt.get_cmap("viridis")

    for row, it in enumerate(reversed(iterations)):
        samples = getattr(it, samples_key)
        hist, edges = np.histogram(samples, bins=bin_edges, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ridge = _ridge_scale(hist)
        color = cmap(row / max(n - 1, 1))
        y0 = float(row)
        ax.fill_between(centers, y0, y0 + ridge, color=color, alpha=0.75, linewidth=0)
        ax.plot(centers, y0 + ridge, color=color, linewidth=0.9, alpha=0.95)
        mean = float(np.mean(samples))
        ax.scatter([mean], [y0 + 0.02], s=12, c="white", edgecolors="0.2", linewidths=0.4, zorder=3)
        ax.text(
            xlim[0],
            y0 + 0.42,
            f"iter {it.iteration:02d}  ({it.n_included} pts)",
            fontsize=7,
            va="center",
            ha="left",
            color="0.25",
        )

    ax.axvline(exact, color="black", linestyle="--", linewidth=1.2, label=f"exact = {exact:g}")
    ax.set_xlim(xlim)
    ax.set_ylim(-0.2, float(n) + 0.3)
    ax.set_yticks([])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="x", alpha=0.25)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _subsample_iterations(
    iterations: list[IterationPosterior],
    *,
    max_panels: int,
) -> list[IterationPosterior]:
    if len(iterations) <= max_panels:
        return iterations
    idx = np.linspace(0, len(iterations) - 1, max_panels, dtype=int)
    return [iterations[i] for i in idx]


def plot_joint_heatmap_grid(
    iterations: list[IterationPosterior],
    path: Path,
    *,
    max_panels: int = 25,
    n_bins: int = 40,
) -> None:
    """Small-multiples grid of joint (T_c, nu) posterior heatmaps."""
    panels = _subsample_iterations(iterations, max_panels=max_panels)
    tc_all = np.concatenate([it.tc for it in iterations])
    nu_all = np.concatenate([it.nu for it in iterations])
    tc_lim = _shared_xlim([tc_all])
    nu_lim = _shared_xlim([nu_all])
    tc_edges = np.linspace(tc_lim[0], tc_lim[1], n_bins + 1)
    nu_edges = np.linspace(nu_lim[0], nu_lim[1], n_bins + 1)

    n = len(panels)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.2 * ncols, 2.8 * nrows),
        constrained_layout=True,
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    for ax, it in zip(axes.ravel(), panels):
        z, _, _ = np.histogram2d(it.tc, it.nu, bins=[tc_edges, nu_edges], density=True)
        extent = [tc_edges[0], tc_edges[-1], nu_edges[0], nu_edges[-1]]
        ax.imshow(
            z.T,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="Blues",
            interpolation="nearest",
        )
        mean_tc = float(np.mean(it.tc))
        mean_nu = float(np.mean(it.nu))
        ax.plot(TC_EXACT, NU_EXACT, marker="*", color="black", markersize=6, linestyle="none")
        ax.plot(mean_tc, mean_nu, marker="x", color="#c44e52", markersize=5, linestyle="none")
        ax.set_title(f"iter {it.iteration:02d} ({it.n_included} pts)", fontsize=8)
        ax.tick_params(labelsize=6)

    for ax in axes.ravel()[n:]:
        ax.axis("off")

    for ax in axes[-1, :]:
        ax.set_xlabel(r"$T_c$", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\nu$", fontsize=8)

    fig.suptitle("Joint posterior evolution (subsampled iterations)", fontsize=11, y=1.02)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_joint_evolution_animation(
    iterations: list[IterationPosterior],
    path: Path,
    *,
    n_bins: int = 40,
    fps: int = 4,
) -> None:
    """Write an MP4 cycling through joint posterior heatmaps."""
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

    tc_all = np.concatenate([it.tc for it in iterations])
    nu_all = np.concatenate([it.nu for it in iterations])
    tc_lim = _shared_xlim([tc_all])
    nu_lim = _shared_xlim([nu_all])

    tc_edges = np.linspace(tc_lim[0], tc_lim[1], n_bins + 1)
    nu_edges = np.linspace(nu_lim[0], nu_lim[1], n_bins + 1)
    heatmaps = [
        np.histogram2d(it.tc, it.nu, bins=[tc_edges, nu_edges], density=True)[0].T
        for it in iterations
    ]
    z_max = max(float(z.max()) for z in heatmaps) if heatmaps else 1.0

    fig, ax = plt.subplots(figsize=(5.0, 4.2), constrained_layout=True)
    extent = [tc_lim[0], tc_lim[1], nu_lim[0], nu_lim[1]]
    mesh = ax.imshow(
        heatmaps[0],
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="Blues",
        interpolation="nearest",
        vmin=0.0,
        vmax=z_max,
    )
    star = ax.plot(TC_EXACT, NU_EXACT, marker="*", color="black", markersize=10, linestyle="none")[0]
    mean_pt = ax.plot(
        float(np.mean(iterations[0].tc)),
        float(np.mean(iterations[0].nu)),
        marker="x",
        color="#c44e52",
        markersize=8,
        linestyle="none",
    )[0]
    title = ax.set_title("")
    ax.set_xlabel(r"$T_c$")
    ax.set_ylabel(r"$\nu$")
    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04, label="density")

    def _update(frame: int) -> tuple[object, ...]:
        z = heatmaps[frame]
        it = iterations[frame]
        mesh.set_data(z)
        mean_pt.set_data([float(np.mean(it.tc))], [float(np.mean(it.nu))])
        title.set_text(f"iter {it.iteration:02d}: {it.n_included} included points")
        return mesh, star, mean_pt, title

    anim = FuncAnimation(fig, _update, frames=len(iterations), blit=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".gif":
        anim.save(path, writer=PillowWriter(fps=fps))
    else:
        try:
            anim.save(path, writer=FFMpegWriter(fps=fps))
        except Exception:
            gif_path = path.with_suffix(".gif")
            anim.save(gif_path, writer=PillowWriter(fps=fps))
            path = gif_path
    plt.close(fig)


def plot_greedy_al_evolution(
    run_dir: Path,
    *,
    out_dir: Path | None = None,
    max_joint_panels: int = 25,
    make_animation: bool = False,
) -> dict[str, Path]:
    """Generate ridge and joint-evolution figures from saved iteration posteriors."""
    out_dir = run_dir if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    iterations = discover_iteration_posteriors(run_dir)
    print(f"Loaded {len(iterations)} iterations from {run_dir}", flush=True)

    paths: dict[str, Path] = {}
    paths["nu_ridge"] = out_dir / "evolution_nu_ridge.png"
    paths["tc_ridge"] = out_dir / "evolution_tc_ridge.png"
    paths["joint_grid"] = out_dir / "evolution_joint_grid.png"
    paths["uncertainty"] = out_dir / "evolution_uncertainty.png"
    paths["uncertainty_csv"] = out_dir / "evolution_uncertainty.csv"
    paths["prediction_discrepancy"] = out_dir / "evolution_prediction_discrepancy.png"
    paths["prediction_discrepancy_csv"] = out_dir / "evolution_prediction_discrepancy.csv"

    metrics = uncertainty_metrics_table(iterations)
    metrics.to_csv(paths["uncertainty_csv"], index=False)
    plot_uncertainty_evolution(metrics, paths["uncertainty"])

    discrepancy = build_prediction_discrepancy_table(run_dir)
    discrepancy.to_csv(paths["prediction_discrepancy_csv"], index=False)
    plot_prediction_discrepancy(discrepancy, paths["prediction_discrepancy"])

    plot_marginal_ridge(
        iterations,
        param="nu",
        exact=NU_EXACT,
        xlabel=r"$\nu$",
        title=rf"Marginal $\nu$ posterior evolution ({len(iterations)} iterations)",
        path=paths["nu_ridge"],
    )
    plot_marginal_ridge(
        iterations,
        param="T_c",
        exact=TC_EXACT,
        xlabel=r"$T_c$",
        title=rf"Marginal $T_c$ posterior evolution ({len(iterations)} iterations)",
        path=paths["tc_ridge"],
    )
    plot_joint_heatmap_grid(
        iterations,
        paths["joint_grid"],
        max_panels=max_joint_panels,
    )

    if make_animation:
        anim_path = out_dir / "evolution_joint.mp4"
        plot_joint_evolution_animation(iterations, anim_path)
        paths["joint_animation"] = anim_path

    for key, path in paths.items():
        print(f"  {key}: {path}", flush=True)
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Greedy AL output directory containing iter_XX/posterior.nc",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Directory for evolution plots (default: same as --run-dir)",
    )
    parser.add_argument(
        "--max-joint-panels",
        type=int,
        default=25,
        help="Max joint heatmap panels in the grid (subsampled evenly)",
    )
    parser.add_argument(
        "--animation",
        action="store_true",
        help="Also write evolution_joint.mp4 (or .gif if ffmpeg unavailable)",
    )
    args = parser.parse_args(argv)
    plot_greedy_al_evolution(
        args.run_dir,
        out_dir=args.out,
        max_joint_panels=args.max_joint_panels,
        make_animation=args.animation,
    )


if __name__ == "__main__":
    main()
