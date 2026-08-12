"""Posterior visualization and LAPS helpers for the fss_explorer widget."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ._deps import az
from .active_learning import (
    CounterfactualResult,
    counterfactual_include_point,
    load_posterior_draws,
)
from .constants import NU_EXACT, TC_EXACT
from .importance_weights import uncertainty_scalar
from .jax_inference import sample_fss_posterior_laps
from .model_fss import NU_PRIOR_LOWER, NU_PRIOR_UPPER
from .model_scaling_nu import TC_PRIOR_LOWER, TC_PRIOR_UPPER
from .profile_likelihood import FssProfileConfig

POSTERIOR_HIST_BINS = 50
WIDGET_LAPS_CHAINS = 2000
WIDGET_LAPS_TUNE = 200
WIDGET_LAPS_DRAWS = 100
PREVIEW_RESAMPLE_SIZE = 8000


@dataclass(frozen=True)
class PosteriorHeatmapData:
    """Joint (T_c, nu) samples and histogram summaries."""

    tc_samples: np.ndarray
    nu_samples: np.ndarray
    tc_centers: np.ndarray
    nu_centers: np.ndarray
    tc_edges: np.ndarray
    nu_edges: np.ndarray
    z: np.ndarray
    tr_cov: float
    mean_tc: float
    mean_nu: float
    x_range: tuple[float, float]
    y_range: tuple[float, float]


def _posterior_tc_nu_samples(idata: az.InferenceData) -> tuple[np.ndarray, np.ndarray]:
    tc = np.asarray(idata.posterior["T_c"].values.reshape(-1), dtype=np.float64)
    nu = np.asarray(idata.posterior["nu"].values.reshape(-1), dtype=np.float64)
    return tc, nu


def _axis_ranges(
    tc: np.ndarray,
    nu: np.ndarray,
    *,
    pad_frac: float = 0.12,
    min_span_tc: float = 0.002,
    min_span_nu: float = 0.02,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Tight axis limits around posterior mass (percentile padding)."""
    lo_tc, hi_tc = np.percentile(tc, [1.0, 99.0])
    lo_nu, hi_nu = np.percentile(nu, [1.0, 99.0])
    span_tc = max(float(hi_tc - lo_tc), min_span_tc)
    span_nu = max(float(hi_nu - lo_nu), min_span_nu)
    pad_tc = span_tc * pad_frac
    pad_nu = span_nu * pad_frac
    return (lo_tc - pad_tc, hi_tc + pad_tc), (lo_nu - pad_nu, hi_nu + pad_nu)


def posterior_heatmap_from_idata(
    idata: az.InferenceData,
    *,
    infer_Tc: bool = True,
    infer_nu: bool = True,
    infer_beta: bool = False,
    n_bins: int = POSTERIOR_HIST_BINS,
) -> PosteriorHeatmapData:
    """Histogram density of LAPS posterior (T_c, nu) draws."""
    if not infer_Tc or not infer_nu:
        raise ValueError("Joint T_c–nu heatmap requires infer_Tc and infer_nu")
    if "T_c" not in idata.posterior or "nu" not in idata.posterior:
        raise ValueError("Posterior must contain T_c and nu")

    tc, nu = _posterior_tc_nu_samples(idata)
    hist, tc_edges, nu_edges = np.histogram2d(tc, nu, bins=n_bins, density=True)
    tc_centers = 0.5 * (tc_edges[:-1] + tc_edges[1:])
    nu_centers = 0.5 * (nu_edges[:-1] + nu_edges[1:])

    draws = load_posterior_draws(
        idata,
        infer_Tc=infer_Tc,
        infer_nu=infer_nu,
        infer_beta=infer_beta,
    )
    mat = draws.as_matrix(
        infer_Tc=infer_Tc,
        infer_nu=infer_nu,
        infer_beta=infer_beta,
    )
    tr_cov = uncertainty_scalar(np.cov(mat, rowvar=False))
    x_range, y_range = _axis_ranges(tc, nu)

    return PosteriorHeatmapData(
        tc_samples=tc,
        nu_samples=nu,
        tc_centers=tc_centers,
        nu_centers=nu_centers,
        tc_edges=tc_edges,
        nu_edges=nu_edges,
        z=np.asarray(hist.T, dtype=np.float64),
        tr_cov=float(tr_cov),
        mean_tc=float(np.mean(tc)),
        mean_nu=float(np.mean(nu)),
        x_range=x_range,
        y_range=y_range,
    )


def weighted_posterior_histogram(
    tc: np.ndarray,
    nu: np.ndarray,
    weights: np.ndarray,
    tc_edges: np.ndarray,
    nu_edges: np.ndarray,
) -> np.ndarray:
    """Weighted 2D histogram on fixed bin edges (Heatmap z: rows=nu, cols=T_c)."""
    hist, _, _ = np.histogram2d(
        tc,
        nu,
        bins=[tc_edges, nu_edges],
        weights=weights,
        density=True,
    )
    return np.asarray(hist.T, dtype=np.float64)


def resample_importance_posterior(
    tc: np.ndarray,
    nu: np.ndarray,
    weights: np.ndarray,
    *,
    size: int = PREVIEW_RESAMPLE_SIZE,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw synthetic samples from PSIS-normalized importance weights."""
    w = np.asarray(weights, dtype=np.float64).ravel()
    w = w / np.sum(w)
    n = int(size)
    rng = np.random.default_rng(seed)
    idx = rng.choice(w.size, size=n, replace=True, p=w)
    return tc[idx], nu[idx]


def _single_posterior_figure_widget(
    *,
    width: int,
    height: int,
    title: str,
    colorscale: str,
    colorbar_title: str,
) -> go.FigureWidget:
    empty = np.array([], dtype=np.float64)
    return go.FigureWidget(
        data=[
            go.Histogram2d(
                x=empty,
                y=empty,
                nbinsx=POSTERIOR_HIST_BINS,
                nbinsy=POSTERIOR_HIST_BINS,
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(title=colorbar_title, len=0.85, x=1.02),
            ),
            go.Scatter(
                x=[TC_EXACT],
                y=[NU_EXACT],
                mode="markers",
                marker=dict(
                    symbol="star",
                    size=11,
                    color="white",
                    line=dict(width=1.2, color="black"),
                ),
                name="exact",
            ),
            go.Scatter(
                x=[TC_EXACT],
                y=[NU_EXACT],
                mode="markers",
                marker=dict(
                    symbol="x",
                    size=12,
                    color="#c44e52",
                    line=dict(width=1.5, color="white"),
                ),
                name="mean",
            ),
        ],
        layout=go.Layout(
            title=dict(text=title, font=dict(size=12)),
            xaxis_title=r"$T_c$",
            yaxis_title=r"$\nu$",
            width=width,
            height=height,
            template="plotly_white",
            margin=dict(l=48, r=72, t=44, b=40),
            legend=dict(font=dict(size=9), yanchor="top", y=0.99, xanchor="left", x=0.01),
        ),
    )


def build_posterior_figure_widget(
    *,
    width: int,
    height: int,
) -> go.FigureWidget:
    """Figure for the current LAPS posterior."""
    return _single_posterior_figure_widget(
        width=width,
        height=height,
        title="Current posterior (T_c, nu) — run inference",
        colorscale="Blues",
        colorbar_title="density",
    )


def build_preview_posterior_figure_widget(
    *,
    width: int,
    height: int,
) -> go.FigureWidget:
    """Figure for the IS-weighted counterfactual posterior (fixed bin grid)."""
    z0 = np.zeros((POSTERIOR_HIST_BINS, POSTERIOR_HIST_BINS), dtype=np.float64)
    tc0 = np.linspace(TC_PRIOR_LOWER, TC_PRIOR_UPPER, POSTERIOR_HIST_BINS)
    nu0 = np.linspace(NU_PRIOR_LOWER, NU_PRIOR_UPPER, POSTERIOR_HIST_BINS)
    return go.FigureWidget(
        data=[
            go.Heatmap(
                x=tc0,
                y=nu0,
                z=z0,
                colorscale="Oranges",
                showscale=True,
                colorbar=dict(title="weighted density", len=0.85, x=1.02),
            ),
            go.Scatter(
                x=[TC_EXACT],
                y=[NU_EXACT],
                mode="markers",
                marker=dict(
                    symbol="star",
                    size=11,
                    color="white",
                    line=dict(width=1.2, color="black"),
                ),
                name="exact",
            ),
            go.Scatter(
                x=[TC_EXACT],
                y=[NU_EXACT],
                mode="markers",
                marker=dict(
                    symbol="x",
                    size=12,
                    color="#c44e52",
                    line=dict(width=1.5, color="white"),
                ),
                name="IS mean",
            ),
        ],
        layout=go.Layout(
            title=dict(
                text="Predicted posterior if point added — preview a grey point",
                font=dict(size=12),
            ),
            xaxis_title=r"$T_c$",
            yaxis_title=r"$\nu$",
            width=width,
            height=height,
            template="plotly_white",
            margin=dict(l=48, r=72, t=44, b=40),
            legend=dict(font=dict(size=9), yanchor="top", y=0.99, xanchor="left", x=0.01),
        ),
    )


def _apply_axis_ranges(
    fig: go.FigureWidget,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> None:
    fig.update_layout(
        xaxis=dict(range=list(x_range)),
        yaxis=dict(range=list(y_range)),
    )


def update_posterior_figure_base(
    fig: go.FigureWidget,
    hm: PosteriorHeatmapData,
    *,
    n_included: int,
    chains: int,
) -> None:
    """Render the LAPS posterior as a Histogram2d from raw chain draws."""
    with fig.batch_update():
        fig.data[0].update(
            x=hm.tc_samples,
            y=hm.nu_samples,
            visible=True,
        )
        fig.data[1].update(x=[TC_EXACT], y=[NU_EXACT])
        fig.data[2].update(x=[hm.mean_tc], y=[hm.mean_nu])
        fig.update_layout(
            title=dict(
                text=(
                    f"Current posterior — {n_included} pts, "
                    f"LAPS {chains} ch, tr(cov)={hm.tr_cov:.5g}"
                ),
                font=dict(size=12),
            ),
        )
        _apply_axis_ranges(fig, hm.x_range, hm.y_range)


def update_preview_posterior_figure(
    fig: go.FigureWidget,
    hm: PosteriorHeatmapData,
    result: CounterfactualResult,
) -> None:
    """Render IS-weighted counterfactual posterior on the current posterior bin grid."""
    z = weighted_posterior_histogram(
        result.tc_samples,
        result.nu_samples,
        result.weights,
        hm.tc_edges,
        hm.nu_edges,
    )
    mean_tc = float(np.average(result.tc_samples, weights=result.weights))
    mean_nu = float(np.average(result.nu_samples, weights=result.weights))

    with fig.batch_update():
        fig.data[0].update(
            x=hm.tc_centers.tolist(),
            y=hm.nu_centers.tolist(),
            z=z.tolist(),
            visible=True,
        )
        fig.data[1].update(x=[TC_EXACT], y=[NU_EXACT])
        fig.data[2].update(x=[mean_tc], y=[mean_nu], visible=True)
        fig.update_layout(
            title=dict(
                text=(
                    f"If added: L={result.L}, T={result.T:.4f} — "
                    f"Δtr(cov)={result.delta_tr_cov:.5g}, "
                    f"IS mean T_c={mean_tc:.5f}, nu={mean_nu:.5f}"
                ),
                font=dict(size=11),
            ),
        )
        _apply_axis_ranges(fig, hm.x_range, hm.y_range)


def clear_preview_posterior_figure(fig: go.FigureWidget) -> None:
    """Reset the preview figure to its empty placeholder state."""
    z0 = np.zeros((POSTERIOR_HIST_BINS, POSTERIOR_HIST_BINS), dtype=np.float64)
    tc0 = np.linspace(TC_PRIOR_LOWER, TC_PRIOR_UPPER, POSTERIOR_HIST_BINS)
    nu0 = np.linspace(NU_PRIOR_LOWER, NU_PRIOR_UPPER, POSTERIOR_HIST_BINS)
    with fig.batch_update():
        fig.data[0].update(x=tc0.tolist(), y=nu0.tolist(), z=z0.tolist())
        fig.data[1].update(x=[TC_EXACT], y=[NU_EXACT])
        fig.data[2].update(x=[TC_EXACT], y=[NU_EXACT], visible=False)
        fig.update_layout(
            title=dict(
                text="Predicted posterior if point added — preview a grey point",
                font=dict(size=12),
            ),
        )


def preview_status_html(result: CounterfactualResult) -> str:
    warn = ""
    if result.pareto_k > 0.7:
        warn = " <span style='color:#c44e52'>(high Pareto-k)</span>"
    elif result.ess < 50:
        warn = " <span style='color:#c44e52'>(low ESS)</span>"
    mean_tc = float(np.average(result.tc_samples, weights=result.weights))
    mean_nu = float(np.average(result.nu_samples, weights=result.weights))
    return (
        f"Preview add idx={result.row_index}: L={result.L}, T={result.T:.4f} — "
        f"tr(cov) {result.tr_cov_before:.5g} → {result.tr_cov_after:.5g} "
        f"(Δ={result.delta_tr_cov:.5g}), "
        f"IS mean T_c={mean_tc:.5f}, nu={mean_nu:.5f}, "
        f"ESS={result.ess:.0f}, k={result.pareto_k:.2f}{warn}"
    )


def run_widget_laps_inference(
    observables: pd.DataFrame,
    config: FssProfileConfig,
    *,
    infer_Tc: bool,
    infer_nu: bool,
    infer_beta: bool,
    chains: int = WIDGET_LAPS_CHAINS,
    tune: int = WIDGET_LAPS_TUNE,
    draws: int = WIDGET_LAPS_DRAWS,
    seed: int = 1,
) -> az.InferenceData:
    """Run LAPS on masked observables for the explorer widget."""
    return sample_fss_posterior_laps(
        observables,
        config,
        infer_Tc=infer_Tc,
        infer_nu=infer_nu,
        infer_beta=infer_beta,
        infer_gp_hyperparams=False,
        tune=tune,
        draws=draws,
        chains=chains,
        seed=seed,
        progress_bar=True,
    )


def preview_include_point(
    included_observables: pd.DataFrame,
    full_observables: pd.DataFrame,
    config: FssProfileConfig,
    idata: az.InferenceData,
    row_index: int,
    *,
    infer_Tc: bool,
    infer_nu: bool,
    infer_beta: bool,
) -> CounterfactualResult:
    """IS counterfactual for adding one excluded row back into the likelihood."""
    draws = load_posterior_draws(
        idata,
        infer_Tc=infer_Tc,
        infer_nu=infer_nu,
        infer_beta=infer_beta,
    )
    row = full_observables.iloc[row_index]
    return counterfactual_include_point(
        included_observables,
        config,
        draws,
        row,
        row_index=row_index,
        infer_Tc=infer_Tc,
        infer_nu=infer_nu,
        infer_beta=infer_beta,
    )


def _save_heatmap_png(
    z: np.ndarray,
    tc_centers: np.ndarray,
    nu_centers: np.ndarray,
    path: Path,
    *,
    title: str,
    cmap: str,
    colorbar_label: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    mean_tc: float,
    mean_nu: float,
    mean_label: str = "posterior mean",
) -> None:
    """Write a (T_c, nu) density heatmap to PNG."""
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.0, 4.2), constrained_layout=True)
    extent = [float(tc_centers[0]), float(tc_centers[-1]), float(nu_centers[0]), float(nu_centers[-1])]
    mesh = ax.imshow(
        np.asarray(z, dtype=np.float64),
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=cmap,
        interpolation="nearest",
    )
    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)
    ax.plot(TC_EXACT, NU_EXACT, marker="*", color="black", markersize=10, linestyle="none", label="exact")
    ax.plot(
        mean_tc,
        mean_nu,
        marker="x",
        color="#c44e52",
        markersize=10,
        linestyle="none",
        label=mean_label,
    )
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xlabel(r"$T_c$")
    ax.set_ylabel(r"$\nu$")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_posterior_heatmap_png(
    hm: PosteriorHeatmapData,
    path: Path,
    *,
    title: str,
) -> None:
    """Save current posterior joint density as PNG."""
    _save_heatmap_png(
        hm.z,
        hm.tc_centers,
        hm.nu_centers,
        path,
        title=title,
        cmap="Blues",
        colorbar_label="density",
        x_range=hm.x_range,
        y_range=hm.y_range,
        mean_tc=hm.mean_tc,
        mean_nu=hm.mean_nu,
    )


def save_weighted_preview_png(
    hm: PosteriorHeatmapData,
    result: CounterfactualResult,
    path: Path,
    *,
    title: str,
) -> None:
    """Save IS-weighted counterfactual posterior on the current posterior bin grid."""
    z = weighted_posterior_histogram(
        result.tc_samples,
        result.nu_samples,
        result.weights,
        hm.tc_edges,
        hm.nu_edges,
    )
    mean_tc = float(np.average(result.tc_samples, weights=result.weights))
    mean_nu = float(np.average(result.nu_samples, weights=result.weights))
    _save_heatmap_png(
        z,
        hm.tc_centers,
        hm.nu_centers,
        path,
        title=title,
        cmap="Oranges",
        colorbar_label="weighted density",
        x_range=hm.x_range,
        y_range=hm.y_range,
        mean_tc=mean_tc,
        mean_nu=mean_nu,
        mean_label="IS mean",
    )
