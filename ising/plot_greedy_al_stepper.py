#!/usr/bin/env python3
"""Interactive HTML stepper: joint posterior heatmap + binder selection per AL step."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .constants import NU_EXACT, TC_EXACT
from .datasets import ISING_DIR
from .importance_weights import uncertainty_scalar
from .plot_greedy_al_evolution import (
    DEFAULT_RUN_DIR,
    discover_iteration_posteriors,
    _shared_xlim,
)

_BINDER_STYLE: dict[int, dict[str, object]] = {
    64: {"color": "#1f77b4", "symbol": "star", "size": 10, "label": "L=64"},
    128: {"color": "#2ca02c", "symbol": "x", "size": 8, "label": "L=128"},
    256: {"color": "#d62728", "symbol": "cross", "size": 10, "label": "L=256"},
}


@dataclass(frozen=True)
class StepperFrame:
    iteration: int
    n_included: int
    next_index: int
    next_delta_tr_cov: float
    tr_cov: float
    z: np.ndarray
    tc_centers: np.ndarray
    nu_centers: np.ndarray
    mean_tc: float
    mean_nu: float
    included_mask: np.ndarray
    ranking: pd.DataFrame


def _load_run_table(run_dir: Path) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    init_path = run_dir / "init_mask.csv"
    summary_path = run_dir / "summary.csv"
    if not init_path.is_file():
        raise FileNotFoundError(f"Missing {init_path}")
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing {summary_path}")
    full_df = pd.read_csv(init_path, index_col=0)
    initial_mask = full_df["included"].astype(bool).to_numpy()
    full_df = full_df.drop(columns=["included"])
    summary = pd.read_csv(summary_path)
    return full_df, initial_mask, summary


def _joint_heatmap_grid(
    iterations: list,
    *,
    n_bins: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tc_all = np.concatenate([it.tc for it in iterations])
    nu_all = np.concatenate([it.nu for it in iterations])
    tc_lim = _shared_xlim([tc_all])
    nu_lim = _shared_xlim([nu_all])
    tc_edges = np.linspace(tc_lim[0], tc_lim[1], n_bins + 1)
    nu_edges = np.linspace(nu_lim[0], nu_lim[1], n_bins + 1)
    tc_centers = 0.5 * (tc_edges[:-1] + tc_edges[1:])
    nu_centers = 0.5 * (nu_edges[:-1] + nu_edges[1:])
    return tc_edges, nu_edges, tc_centers, nu_centers


def build_stepper_frames(
    run_dir: Path,
    *,
    n_bins: int = 40,
) -> tuple[pd.DataFrame, list[StepperFrame], tuple[float, float], tuple[float, float]]:
    full_df, initial_mask, summary = _load_run_table(run_dir)
    iterations = discover_iteration_posteriors(run_dir)
    tc_edges, nu_edges, tc_centers, nu_centers = _joint_heatmap_grid(iterations, n_bins=n_bins)
    tc_lim = (float(tc_edges[0]), float(tc_edges[-1]))
    nu_lim = (float(nu_edges[0]), float(nu_edges[-1]))

    frames: list[StepperFrame] = []
    for it in iterations:
        row = summary.loc[summary["iteration"] == it.iteration]
        if row.empty:
            continue
        row = row.iloc[0]
        mask = initial_mask.copy()
        for prev in summary.loc[summary["iteration"] < it.iteration].itertuples():
            mask[int(prev.added_row_index)] = True

        mat = np.column_stack([it.tc, it.nu])
        tr_cov = uncertainty_scalar(np.cov(mat, rowvar=False), metric="trace_cov")
        z = np.histogram2d(it.tc, it.nu, bins=[tc_edges, nu_edges], density=True)[0].T

        rank_path = run_dir / f"iter_{it.iteration:02d}" / "ranking.csv"
        if rank_path.is_file():
            ranking = pd.read_csv(rank_path)
        else:
            ranking = pd.DataFrame(columns=["row_index", "delta_tr_cov"])

        frames.append(
            StepperFrame(
                iteration=int(it.iteration),
                n_included=int(row["n_included"]),
                next_index=int(row["added_row_index"]),
                next_delta_tr_cov=float(row["delta_tr_cov"]),
                tr_cov=float(tr_cov),
                z=z,
                tc_centers=tc_centers,
                nu_centers=nu_centers,
                mean_tc=float(np.mean(it.tc)),
                mean_nu=float(np.mean(it.nu)),
                included_mask=mask,
                ranking=ranking,
            )
        )

    if not frames:
        raise ValueError(f"No stepper frames could be built from {run_dir}")
    return full_df, frames, tc_lim, nu_lim


def softmax_choice_probs(
    utilities: np.ndarray,
    *,
    temperature_scale: float = 2.0,
) -> np.ndarray:
    """Softmax choice probabilities from utilities (higher utility → higher P).

    Temperature is ``temperature_scale * (max - min)`` at this step so the map
    adapts to each iteration's Δ tr(cov) spread.
    """
    u = np.asarray(utilities, dtype=np.float64)
    if u.size == 0:
        return u
    if u.size == 1:
        return np.array([1.0])
    spread = float(u.max() - u.min())
    if spread <= 0.0:
        return np.full(u.size, 1.0 / u.size)
    tau = max(spread * temperature_scale, 1e-15)
    z = (u - u.max()) / tau
    w = np.exp(z)
    return w / w.sum()


def _choice_prob_by_index(
    ranking: pd.DataFrame,
    *,
    temperature_scale: float,
) -> dict[int, float]:
    if ranking.empty:
        return {}
    idx = ranking["row_index"].astype(int).to_numpy()
    delta = ranking["delta_tr_cov"].astype(float).to_numpy()
    probs = softmax_choice_probs(delta, temperature_scale=temperature_scale)
    return {int(i): float(p) for i, p in zip(idx, probs)}


def _prob_colormap_limits(probs: np.ndarray) -> tuple[float, float]:
    """Color scale bounds for one step's softmax probabilities."""
    valid = probs[np.isfinite(probs)]
    if valid.size == 0:
        return 0.0, 1.0
    pmin = float(valid.min())
    pmax = float(valid.max())
    if pmin >= pmax:
        pad = max(abs(pmin) * 0.1, 1e-6)
        return max(0.0, pmin - pad), min(1.0, pmax + pad)
    return pmin, pmax


def _heatmap_trace(frame: StepperFrame) -> go.Heatmap:
    return go.Heatmap(
        x=frame.tc_centers,
        y=frame.nu_centers,
        z=frame.z,
        colorscale="Blues",
        showscale=True,
        colorbar=dict(
            title="density",
            x=0.44,
            xanchor="right",
            len=0.85,
            thickness=14,
        ),
        zmin=0.0,
        hovertemplate=r"$T_c$=%{x:.5f}<br>$\nu$=%{y:.4f}<br>d=%{z:.3g}<extra></extra>",
    )


def _binder_traces(
    full_df: pd.DataFrame,
    frame: StepperFrame,
    *,
    temperature_scale: float = 2.0,
    show_score_colorbar: bool = False,
) -> list[go.Scatter]:
    L_arr = full_df["L"].to_numpy(dtype=np.int64)
    T_arr = full_df["T"].to_numpy(dtype=np.float64)
    u4 = full_df["binder_cumulant"].to_numpy(dtype=np.float64)
    mask = frame.included_mask
    next_idx = frame.next_index
    score_by_idx = {
        int(r.row_index): float(r.delta_tr_cov) for r in frame.ranking.itertuples(index=False)
    }
    prob_by_idx = _choice_prob_by_index(frame.ranking, temperature_scale=temperature_scale)

    traces: list[go.Scatter] = []
    for L in sorted(full_df["L"].unique()):
        style = _BINDER_STYLE.get(int(L), {"color": "black", "symbol": "circle", "size": 7, "label": f"L={L}"})
        inc = mask & (L_arr == L)
        if not inc.any():
            traces.append(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(color=style["color"], symbol=style["symbol"], size=style["size"]),
                    name=str(style["label"]),
                    xaxis="x2",
                    yaxis="y2",
                )
            )
            continue
        idx = np.arange(len(full_df))[inc]
        traces.append(
            go.Scatter(
                x=T_arr[inc],
                y=u4[inc],
                mode="markers",
                marker=dict(
                    color=style["color"],
                    symbol=style["symbol"],
                    size=style["size"],
                    line=dict(width=1.0, color="white"),
                ),
                name=str(style["label"]),
                hovertemplate="idx=%{customdata}<br>T=%{x:.4f}<br>U4=%{y:.4f}<extra></extra>",
                customdata=idx,
                xaxis="x2",
                yaxis="y2",
            )
        )

    excluded_idx = np.flatnonzero(~mask)
    if excluded_idx.size:
        deltas = np.array([score_by_idx.get(int(i), np.nan) for i in excluded_idx], dtype=np.float64)
        probs = np.array([prob_by_idx.get(int(i), np.nan) for i in excluded_idx], dtype=np.float64)
        prob_cmin, prob_cmax = _prob_colormap_limits(probs)
        traces.append(
            go.Scatter(
                x=T_arr[excluded_idx],
                y=u4[excluded_idx],
                mode="markers",
                name="softmax P(choose)",
                marker=dict(
                    size=11,
                    color=probs,
                    colorscale="Turbo",
                    cmin=prob_cmin,
                    cmax=prob_cmax,
                    showscale=show_score_colorbar,
                    colorbar=dict(
                        title=r"P(choose)<br>(this step)",
                        x=1.02,
                        xanchor="left",
                        len=0.85,
                        thickness=14,
                        tickformat=".1%",
                    ),
                    line=dict(width=0.6, color="rgba(255,255,255,0.85)"),
                ),
                customdata=np.column_stack([excluded_idx, deltas, probs]),
                hovertemplate=(
                    "idx=%{customdata[0]}<br>T=%{x:.4f}<br>U4=%{y:.4f}<br>"
                    "Δtr(cov)=%{customdata[1]:.4g}<br>P(choose)=%{customdata[2]:.2%}<extra></extra>"
                ),
                xaxis="x2",
                yaxis="y2",
            )
        )

    next_row = full_df.iloc[next_idx]
    chosen_p = prob_by_idx.get(next_idx, float("nan"))
    traces.append(
        go.Scatter(
            x=[float(next_row["T"])],
            y=[float(next_row["binder_cumulant"])],
            mode="markers",
            marker=dict(symbol="star", size=18, color="#e6a817", line=dict(width=1.2, color="black")),
            name="chosen next",
            hovertemplate=(
                f"chosen idx={next_idx}<br>L={int(next_row['L'])}<br>T=%{{x:.4f}}<br>"
                f"U4=%{{y:.4f}}<br>Δtr(cov)={frame.next_delta_tr_cov:.4g}<br>"
                f"P(choose)={chosen_p:.2%}<extra></extra>"
            ),
            xaxis="x2",
            yaxis="y2",
        )
    )
    return traces


def _overlay_traces(frame: StepperFrame) -> list[go.Scatter]:
    return [
        go.Scatter(
            x=[TC_EXACT],
            y=[NU_EXACT],
            mode="markers",
            marker=dict(symbol="star", size=12, color="black"),
            name="exact",
            hovertemplate=f"exact T_c={TC_EXACT:.5f}, nu={NU_EXACT}<extra></extra>",
        ),
        go.Scatter(
            x=[frame.mean_tc],
            y=[frame.mean_nu],
            mode="markers",
            marker=dict(symbol="x", size=10, color="#c44e52"),
            name="posterior mean",
            hovertemplate=r"mean $T_c$=%{x:.5f}<br>mean $\nu$=%{y:.4f}<extra></extra>",
        ),
    ]


def _frame_title(
    run_dir: Path,
    frame: StepperFrame,
    *,
    temperature_scale: float,
) -> str:
    selection = "greedy"
    meta_path = run_dir / "run_meta.json"
    if meta_path.is_file():
        selection = json.loads(meta_path.read_text()).get("selection", selection)
    prob_by_idx = _choice_prob_by_index(frame.ranking, temperature_scale=temperature_scale)
    chosen_p = prob_by_idx.get(frame.next_index, float("nan"))
    return (
        f"iter {frame.iteration:02d} ({selection}): {frame.n_included} included, "
        f"next idx={frame.next_index}, Δtr(cov)={frame.next_delta_tr_cov:.4g}, "
        f"P(choose)={chosen_p:.1%}, tr(cov)={frame.tr_cov:.4g}"
    )


def _frame_traces(
    full_df: pd.DataFrame,
    frame: StepperFrame,
    *,
    temperature_scale: float,
) -> tuple[list[go.Heatmap | go.Scatter], list[go.Scatter]]:
    """Left (heatmap) and right (binder) traces for one animation frame."""
    left: list[go.Heatmap | go.Scatter] = [_heatmap_trace(frame), *_overlay_traces(frame)]
    right = _binder_traces(
        full_df, frame, temperature_scale=temperature_scale, show_score_colorbar=False
    )
    if len(right) >= 2 and int((~frame.included_mask).sum()) > 0:
        right[-2].marker.showscale = True  # type: ignore[union-attr]
    return left, right


def build_stepper_figure(
    run_dir: Path,
    *,
    n_bins: int = 40,
    softmax_temperature_scale: float = 2.0,
) -> go.Figure:
    """Build the Plotly stepper figure (heatmap + binder, slider/frames)."""
    full_df, frames, tc_lim, nu_lim = build_stepper_frames(run_dir, n_bins=n_bins)

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.46, 0.46],
        subplot_titles=(
            r"Posterior $(T_c, \nu)$",
            r"Binder (softmax P(choose); colors scaled to step range)",
        ),
        horizontal_spacing=0.14,
    )

    f0_left, f0_right = _frame_traces(
        full_df, frames[0], temperature_scale=softmax_temperature_scale
    )
    for tr in f0_left:
        fig.add_trace(tr, row=1, col=1)
    for tr in f0_right:
        fig.add_trace(tr, row=1, col=2)

    plotly_frames = []
    for frame in frames:
        left, right = _frame_traces(
            full_df, frame, temperature_scale=softmax_temperature_scale
        )
        plotly_frames.append(
            go.Frame(
                name=str(frame.iteration),
                data=[*left, *right],
                layout=go.Layout(
                    title_text=_frame_title(
                        run_dir, frame, temperature_scale=softmax_temperature_scale
                    )
                ),
            )
        )
    fig.frames = plotly_frames

    slider_steps = [
        {
            "args": [
                [str(frame.iteration)],
                {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}},
            ],
            "label": f"{frame.iteration:02d}",
            "method": "animate",
        }
        for frame in frames
    ]

    fig.update_layout(
        title_text=_frame_title(
            run_dir, frames[0], temperature_scale=softmax_temperature_scale
        ),
        template="plotly_white",
        width=1180,
        height=540,
        margin=dict(l=56, r=150, t=88, b=110),
        showlegend=True,
        legend=dict(font=dict(size=9), x=1.14, y=0.5),
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.05,
                "y": -0.08,
                "xanchor": "left",
                "buttons": [
                    {
                        "label": "Prev",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                                "fromcurrent": True,
                                "direction": "backward",
                            },
                        ],
                    },
                    {
                        "label": "Next",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                                "fromcurrent": True,
                                "direction": "forward",
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Iteration: ", "font": {"size": 12}},
                "pad": {"t": 40},
                "steps": slider_steps,
            }
        ],
    )

    fig.update_xaxes(title_text=r"$T_c$", range=list(tc_lim), row=1, col=1)
    fig.update_yaxes(title_text=r"$\nu$", range=list(nu_lim), row=1, col=1)
    fig.update_xaxes(title_text=r"$T$", row=1, col=2)
    fig.update_yaxes(title_text=r"$U_4$", row=1, col=2)
    fig.add_vline(x=TC_EXACT, line=dict(color="red", dash="dashdot", width=1), row=1, col=2)
    return fig


def show_stepper_in_notebook(fig: go.Figure) -> None:
    """Display the stepper inline in Jupyter / Cursor (HTML embed, no nbformat needed)."""
    from IPython.display import HTML, display

    display(HTML(fig.to_html(include_plotlyjs="cdn", full_html=False)))


def stepper_keyboard_post_script(frames: list[StepperFrame]) -> str:
    frame_names = [str(frame.iteration) for frame in frames]
    return f"""
var frameNames = {json.dumps(frame_names)};
var stepIdx = 0;
function gotoStep(idx) {{
  stepIdx = Math.max(0, Math.min(idx, frameNames.length - 1));
  Plotly.animate('greedy-al-stepper', [frameNames[stepIdx]], {{
    frame: {{duration: 0, redraw: true}},
    mode: 'immediate',
    transition: {{duration: 0}}
  }});
}}
document.addEventListener('keydown', function(e) {{
  if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {{
    gotoStep(stepIdx + 1);
    e.preventDefault();
  }} else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {{
    gotoStep(stepIdx - 1);
    e.preventDefault();
  }}
}});
var gd = document.getElementById('greedy-al-stepper');
if (gd) {{
  gd.on('plotly_sliderchange', function(e) {{
    stepIdx = e.step._index;
  }});
}}
"""


def export_stepper_html(
    run_dir: Path,
    out_path: Path,
    *,
    n_bins: int = 40,
    inline_plotlyjs: bool = False,
) -> Path:
    """Write a self-contained HTML viewer (arrow keys + slider)."""
    _, frames, _, _ = build_stepper_frames(run_dir, n_bins=n_bins)
    fig = build_stepper_figure(run_dir, n_bins=n_bins)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        out_path,
        div_id="greedy-al-stepper",
        include_plotlyjs=True if inline_plotlyjs else "cdn",
        auto_open=False,
        post_script=stepper_keyboard_post_script(frames),
    )
    return out_path


def export_stepper_notebook(run_dir: Path, out_path: Path | None = None) -> Path:
    """Write a minimal notebook that displays the stepper via ``fig.show()``."""
    run_dir = run_dir.resolve()
    if out_path is None:
        out_path = ISING_DIR / "greedy_al_stepper.ipynb"
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Active-learning stepper\n",
                    "\n",
                    "Interactive view of each greedy/worst-case iteration:\n",
                    "- **Left:** joint posterior `(T_c, ν)` heatmap\n",
                    "- **Right:** binder plot — included points by `L`, excluded candidates "
                    "colored by **Δ tr(cov)** from `iter_XX/ranking.csv`, gold star = chosen next point\n",
                    "\n",
                    "Use the **slider** or **Prev / Next** buttons under the figure to step through "
                    "iterations (works in Jupyter / Cursor — no port forwarding needed).\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import sys\n",
                    "from pathlib import Path\n",
                    "\n",
                    "repo_root = Path.cwd()\n",
                    "while repo_root != repo_root.parent and not (repo_root / \"pyproject.toml\").exists():\n",
                    "    repo_root = repo_root.parent\n",
                    "if str(repo_root) not in sys.path:\n",
                    "    sys.path.insert(0, str(repo_root))\n",
                    "\n",
                    "from ising.plot_greedy_al_stepper import (\n",
                    "    build_stepper_figure,\n",
                    "    show_stepper_in_notebook,\n",
                    ")\n",
                    f"RUN_DIR = Path({str(run_dir)!r})\n",
                    "\n",
                    "assert (RUN_DIR / \"summary.csv\").is_file(), f\"Missing AL outputs under {{RUN_DIR}}\"\n",
                    "RUN_DIR\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "fig = build_stepper_figure(RUN_DIR)\n",
                    "show_stepper_in_notebook(fig)\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(nb, indent=1) + "\n")
    return out_path


def serve_stepper(
    html_path: Path,
    *,
    port: int = 8765,
    host: str = "127.0.0.1",
) -> None:
    """Serve the HTML file locally (use SSH port forwarding from your laptop)."""
    import http.server
    import socketserver

    html_path = html_path.resolve()
    if not html_path.is_file():
        raise FileNotFoundError(html_path)

    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(html_path.parent), **kwargs)

    url_path = html_path.name
    print(f"Serving {html_path.parent} at http://{host}:{port}/", flush=True)
    print(f"Stepper URL: http://{host}:{port}/{url_path}", flush=True)
    if host == "127.0.0.1":
        print(
            f"\nFrom your laptop (new terminal):\n"
            f"  ssh -N -L {port}:127.0.0.1:{port} $USER@$(hostname -f)\n"
            f"Then open: http://localhost:{port}/{url_path}\n"
            f"(Cursor/VS Code may also auto-forward port {port} — check the Ports panel.)\n",
            flush=True,
        )
    with socketserver.TCPServer((host, port), _Handler) as httpd:
        httpd.serve_forever()


def main(argv: list[str] | None = None) -> Path | None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Greedy / worst AL output directory",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output HTML path (default: RUN_DIR/evolution_stepper.html)",
    )
    parser.add_argument("--bins", type=int, default=40, help="Heatmap bin count")
    parser.add_argument(
        "--inline-plotlyjs",
        action="store_true",
        help="Embed plotly.js in HTML (larger file; no CDN needed when serving offline)",
    )
    parser.add_argument(
        "--notebook",
        type=Path,
        default=None,
        nargs="?",
        const=ISING_DIR / "greedy_al_stepper.ipynb",
        metavar="PATH",
        help="Write a Jupyter notebook (default: ising/greedy_al_stepper.ipynb)",
    )
    parser.add_argument(
        "--serve",
        type=int,
        default=None,
        metavar="PORT",
        help="Serve the HTML on PORT after writing (default host 127.0.0.1; use with SSH -L)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Bind address for --serve (default: 127.0.0.1)",
    )
    args = parser.parse_args(argv)
    out_path = args.out if args.out is not None else args.run_dir / "evolution_stepper.html"
    path = export_stepper_html(
        args.run_dir,
        out_path,
        n_bins=args.bins,
        inline_plotlyjs=args.inline_plotlyjs,
    )
    print(f"Wrote {path}", flush=True)
    if args.notebook is not None:
        nb_path = export_stepper_notebook(args.run_dir, args.notebook)
        print(f"Wrote notebook {nb_path}", flush=True)
    if args.serve is not None:
        serve_stepper(path, port=args.serve, host=args.host)
        return path
    print(
        "Open ising/greedy_al_stepper.ipynb in Cursor/Jupyter "
        "and run both cells (no port forwarding needed).",
        flush=True,
    )
    return path


if __name__ == "__main__":
    main()
