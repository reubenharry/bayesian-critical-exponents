#!/usr/bin/env python3
"""Plot Binder cumulant U_4(L, T) from an observables table (including partial checkpoints)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .constants import NU_EXACT, TC_EXACT
from .datasets import HARADA_BINDER_DAT, read_harada_binder_df
from .observables import read_observables, sigma_from_mc_batch

# Paper figure styling (Harada PRE 84, 056704).
_HARADA_BSA_STYLE: dict[int, dict[str, object]] = {
    64: {"color": "C0", "marker": "*", "markersize": 9, "label": r"$L=64$"},
    128: {"color": "C2", "marker": "x", "markersize": 7, "label": r"$L=128$"},
    256: {"color": "C3", "marker": "+", "markersize": 9, "label": r"$L=256$"},
}


def _binder_sigma_series(df: pd.DataFrame) -> pd.Series:
    if "binder_cumulant_std" in df.columns and "n_eff_binder" in df.columns:
        sigma = sigma_from_mc_batch(
            df["binder_cumulant_std"].to_numpy(dtype=np.float64),
            df["n_eff_binder"].to_numpy(dtype=np.float64),
        )
        return pd.Series(sigma, index=df.index)
    raise ValueError(
        "DataFrame needs binder_cumulant_std and n_eff_binder for error bars"
    )


def _chain_progress(df: pd.DataFrame) -> pd.Series | None:
    if "n_draws" in df.columns and "n_sweeps_target" in df.columns:
        draws = df["n_draws"].to_numpy(dtype=np.float64)
        target = df["n_sweeps_target"].to_numpy(dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            return pd.Series(np.where(target > 0, draws / target, np.nan), index=df.index)
    return None


def _L_label(L: int, progress: float | None) -> str:
    if progress is None or not np.isfinite(progress):
        return f"L={L}"
    if progress >= 0.999:
        return f"L={L}"
    return f"L={L} ({100.0 * progress:.0f}%)"


def harada_collapse_x(
    L: np.ndarray,
    T: np.ndarray,
    *,
    T_c: float = TC_EXACT,
    nu: float = NU_EXACT,
    L_ref: float = 256.0,
) -> np.ndarray:
    """Harada scaling abscissa: ``(1/T - 1/T_c) (L/L_ref)^{1/\\nu}``."""
    L = np.asarray(L, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    return (1.0 / T - 1.0 / T_c) * (L / L_ref) ** (1.0 / nu)


def plot_binder_vs_temperature(
    df: pd.DataFrame,
    *,
    out: Path | None = None,
    T_c: float = TC_EXACT,
    show_exact_tc: bool = True,
    title: str | None = None,
    figsize: tuple[float, float] = (7.0, 4.5),
) -> plt.Figure:
    """Plot ``U_4`` vs ``T`` with one curve per lattice size."""
    required = {"L", "T", "binder_cumulant"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {sorted(missing)}")

    sigma = _binder_sigma_series(df)
    progress = _chain_progress(df)
    plot_df = df.assign(_sigma_binder=sigma)
    if progress is not None:
        plot_df = plot_df.assign(_progress=progress)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    for L in sorted(plot_df["L"].unique()):
        g = plot_df[plot_df["L"] == L].sort_values("T")
        prog = float(g["_progress"].iloc[0]) if progress is not None else None
        alpha = 1.0 if prog is None or prog >= 0.999 else 0.45 + 0.55 * prog
        ax.errorbar(
            g["T"],
            g["binder_cumulant"],
            yerr=g["_sigma_binder"],
            fmt="o-",
            capsize=3,
            markersize=5,
            alpha=alpha,
            label=_L_label(int(L), prog),
        )

    if show_exact_tc:
        ax.axvline(T_c, color="red", linestyle="-.", linewidth=1.2, label=rf"exact $T_c={T_c:.4f}$")

    ax.set_xlabel(r"$T$")
    ax.set_ylabel(r"$U_4 = 1 - \langle m^4\rangle / (3\langle m^2\rangle^2)$")
    if title is None:
        title = "Binder cumulant vs temperature"
        if progress is not None and (progress < 0.999).any():
            title += " (partial chains: faded = in progress)"
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.25)

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    return fig


def save_binder_selection_png(
    df: pd.DataFrame,
    included_mask: np.ndarray,
    path: Path,
    *,
    next_index: int | None = None,
    title: str | None = None,
    T_c: float = TC_EXACT,
    figsize: tuple[float, float] = (7.0, 4.5),
) -> None:
    """Save Binder vs T with included points highlighted and one candidate marked."""
    required = {"L", "T", "binder_cumulant"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {sorted(missing)}")
    if len(included_mask) != len(df):
        raise ValueError(
            f"included_mask length {len(included_mask)} != dataframe length {len(df)}"
        )

    mask = np.asarray(included_mask, dtype=bool)
    has_sigma = "binder_cumulant_std" in df.columns and "n_eff_binder" in df.columns
    sigma = _binder_sigma_series(df) if has_sigma else None

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    L_arr = df["L"].to_numpy(dtype=np.int64)
    T_arr = df["T"].to_numpy(dtype=np.float64)
    u4 = df["binder_cumulant"].to_numpy(dtype=np.float64)

    excluded = ~mask
    if next_index is not None:
        excluded = excluded & (np.arange(len(df)) != int(next_index))

    if excluded.any():
        ax.scatter(
            T_arr[excluded],
            u4[excluded],
            c="#bbbbbb",
            s=28,
            alpha=0.55,
            marker="o",
            linewidths=0.6,
            edgecolors="0.45",
            label="excluded",
            zorder=1,
        )

    legend_L: set[int] = set()
    for L in sorted(df["L"].unique()):
        style = _HARADA_BSA_STYLE.get(
            int(L),
            {"color": "C0", "marker": "o", "markersize": 6, "label": f"L={L}"},
        )
        inc_L = mask & (L_arr == L)
        if next_index is not None:
            inc_L = inc_L & (np.arange(len(df)) != int(next_index))
        if not inc_L.any():
            continue
        label = str(style["label"]) if int(L) not in legend_L else None
        if label is not None:
            legend_L.add(int(L))
        kwargs: dict[str, object] = {
            "color": style["color"],
            "fmt": style["marker"],
            "linestyle": "none",
            "markersize": style["markersize"],
            "label": label,
            "capsize": 2,
            "zorder": 2,
        }
        if sigma is not None:
            ax.errorbar(T_arr[inc_L], u4[inc_L], yerr=sigma.to_numpy()[inc_L], **kwargs)
        else:
            ax.plot(T_arr[inc_L], u4[inc_L], **kwargs)

    if next_index is not None:
        j = int(next_index)
        row = df.iloc[j]
        ax.scatter(
            [float(row["T"])],
            [float(row["binder_cumulant"])],
            marker="*",
            s=220,
            c="#e6a817",
            edgecolors="black",
            linewidths=0.8,
            label=rf"next add (L={int(row['L'])}, idx={j})",
            zorder=4,
        )

    ax.axvline(T_c, color="red", linestyle="dashdot", linewidth=1.2, label=rf"exact $T_c={T_c:.4f}$")
    ax.set_xlabel(r"$T$")
    ax.set_ylabel(r"$U_4$")
    if title is None:
        n_inc = int(mask.sum())
        title = f"Binder selection ({n_inc} included"
        if next_index is not None:
            title += f", next idx={int(next_index)}"
        title += ")"
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.25)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_binder_collapse(
    df: pd.DataFrame,
    *,
    out: Path | None = None,
    T_c: float = TC_EXACT,
    nu: float = NU_EXACT,
    L_ref: float = 256.0,
    title: str | None = None,
    figsize: tuple[float, float] = (7.0, 4.5),
) -> plt.Figure:
    """Plot collapsed ``U_4`` vs Harada ``x = (1/T - 1/T_c)(L/L_ref)^{1/\\nu}``."""
    required = {"L", "T", "binder_cumulant"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {sorted(missing)}")

    sigma = _binder_sigma_series(df)
    L_arr = df["L"].to_numpy(dtype=np.float64)
    T_arr = df["T"].to_numpy(dtype=np.float64)
    u4 = df["binder_cumulant"].to_numpy(dtype=np.float64)
    x = harada_collapse_x(L_arr, T_arr, T_c=T_c, nu=nu, L_ref=L_ref)
    progress = _chain_progress(df)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    for Lval in sorted(df["L"].unique()):
        mask = L_arr == Lval
        order = np.argsort(x[mask])
        x_L = x[mask][order]
        u4_L = u4[mask][order]
        sigma_L = sigma.to_numpy()[mask][order]
        if progress is not None:
            prog = float(progress.to_numpy()[mask][order[0]])
        else:
            prog = None
        alpha = 1.0 if prog is None or prog >= 0.999 else 0.45 + 0.55 * prog
        ax.errorbar(
            x_L,
            u4_L,
            yerr=sigma_L,
            fmt="o",
            capsize=3,
            markersize=5,
            alpha=alpha,
            label=_L_label(int(Lval), prog),
        )

    ax.axvline(0.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.set_xlabel(
        rf"$(1/T - 1/T_c)\,(L/{int(L_ref)})^{{1/\nu}}$"
    )
    ax.set_ylabel(r"$U_4$")
    if title is None:
        title = rf"Binder collapse ($T_c={T_c:.4f}$, $\nu={nu:.3g}$, $L_{{\mathrm{{ref}}}}={int(L_ref)}$)"
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.25)

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    return fig


def plot_binder_collapse_z(
    df: pd.DataFrame,
    *,
    out: Path | None = None,
    T_c: float = TC_EXACT,
    nu: float = NU_EXACT,
    t_scale_max: float = 0.10,
    title: str | None = None,
    figsize: tuple[float, float] = (10.0, 4.6),
) -> plt.Figure:
    """Two-panel Binder collapse in ``z = t L^{1/\\nu}`` highlighting large-|t| failure.

    Left: near-critical window where data of different ``L`` should collapse.
    Right: full ``z`` range; points with ``|t| > t_scale_max`` use open markers.
    """
    required = {"L", "T", "binder_cumulant"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {sorted(missing)}")

    sigma = _binder_sigma_series(df).to_numpy(dtype=np.float64)
    L_arr = df["L"].to_numpy(dtype=np.float64)
    T_arr = df["T"].to_numpy(dtype=np.float64)
    u4 = df["binder_cumulant"].to_numpy(dtype=np.float64)
    t = (T_arr - T_c) / T_c
    z = t * np.power(L_arr, 1.0 / nu)
    in_window = np.abs(t) <= float(t_scale_max)

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
    markers_in = {
        8: "v",
        12: "^",
        16: "<",
        24: ">",
        32: "P",
        64: "o",
        128: "s",
        256: "D",
    }
    # Left panel: only Harada-scale L (collapse should work).
    # Right panel: all L, with small-L / large-|t| marked.
    harada_L = {64, 128, 256}
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    for ax, xlim, panel, L_filter, show_large_t in (
        (
            axes[0],
            (-8.0, 8.0),
            "Harada L (near-critical window)",
            harada_L,
            False,
        ),
        (
            axes[1],
            None,
            "full range incl. small L × large |t|",
            None,
            True,
        ),
    ):
        for Lval in sorted(df["L"].unique()):
            Lval_i = int(Lval)
            if L_filter is not None and Lval_i not in L_filter:
                continue
            mask = L_arr == Lval
            color = colors.get(Lval_i, "C0")
            marker = markers_in.get(Lval_i, "o")

            mask_in = mask & in_window
            if np.any(mask_in):
                order = np.argsort(z[mask_in])
                ax.errorbar(
                    z[mask_in][order],
                    u4[mask_in][order],
                    yerr=sigma[mask_in][order],
                    fmt=marker,
                    color=color,
                    linestyle="none",
                    capsize=2,
                    markersize=5,
                    alpha=0.95,
                    label=rf"$L={Lval_i}$",
                )

            mask_out = mask & ~in_window
            if show_large_t and np.any(mask_out):
                order = np.argsort(z[mask_out])
                ax.errorbar(
                    z[mask_out][order],
                    u4[mask_out][order],
                    yerr=sigma[mask_out][order],
                    fmt="x",
                    color=color,
                    linestyle="none",
                    capsize=2,
                    markersize=6,
                    alpha=0.75,
                    label=rf"$L={Lval_i}$ (large $|t|$)",
                )

        ax.axvline(0.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.set_xlabel(rf"$z = t\,L^{{1/\nu}}$")
        ax.set_ylabel(r"$U_4$")
        ax.set_title(panel)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=6, loc="best", ncol=2 if L_filter is None else 1)
        if xlim is not None:
            ax.set_xlim(*xlim)

    if title is None:
        title = (
            rf"Binder scaling collapse at exact $T_c={T_c:.4f}$, $\nu={nu:.3g}$ "
            rf"(× marks: $|t|>{t_scale_max:g}$)"
        )
    fig.suptitle(title, fontsize=11)

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    return fig


def plot_binder_vs_inv_t(
    df: pd.DataFrame,
    *,
    out: Path | None = None,
    T_c: float = TC_EXACT,
    show_exact_tc: bool = True,
    title: str | None = None,
    figsize: tuple[float, float] = (6.5, 5.0),
    xlim: tuple[float, float] = (0.40, 0.48),
    ylim: tuple[float, float] = (0.0, 1.0),
    inset: bool = True,
    inset_xlim: tuple[float, float] = (0.435, 0.445),
    inset_ylim: tuple[float, float] = (0.80, 0.98),
) -> plt.Figure:
    """Plot ``U_4`` vs ``1/T`` in the style of Harada PRE 84, Fig. (Binder)."""
    required = {"L", "binder_cumulant"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {sorted(missing)}")

    if "inv_T" in df.columns:
        inv_t = df["inv_T"].to_numpy(dtype=np.float64)
    elif "T" in df.columns:
        inv_t = 1.0 / df["T"].to_numpy(dtype=np.float64)
    else:
        raise ValueError("DataFrame needs inv_T or T for the abscissa")

    sigma = _binder_sigma_series(df) if "binder_cumulant_std" in df.columns else None
    u4 = df["binder_cumulant"].to_numpy(dtype=np.float64)
    L_arr = df["L"].to_numpy(dtype=np.int64)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    for L in sorted(df["L"].unique()):
        mask = L_arr == L
        order = np.argsort(inv_t[mask])
        style = _HARADA_BSA_STYLE.get(int(L), {"color": "k", "marker": "o", "markersize": 6, "label": f"L={L}"})
        kwargs: dict[str, object] = {
            "color": style["color"],
            "fmt": style["marker"],
            "linestyle": "none",
            "markersize": style["markersize"],
            "label": style["label"],
            "capsize": 2,
        }
        x_L = inv_t[mask][order]
        u4_L = u4[mask][order]
        if sigma is not None:
            ax.errorbar(x_L, u4_L, yerr=sigma.to_numpy()[mask][order], **kwargs)
        else:
            ax.plot(x_L, u4_L, **kwargs)

    if show_exact_tc:
        inv_tc = 1.0 / T_c
        ax.axvline(inv_tc, color="gray", linestyle=":", linewidth=1.0, alpha=0.8)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"$1/T$")
    ax.set_ylabel(r"$U(T,L)$")
    if title is None:
        title = r"Binder cumulant, square-lattice Ising (Harada BSA data)"
    ax.set_title(title)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.25)

    if inset:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

        axins = inset_axes(ax, width="38%", height="38%", loc="upper left", borderpad=1.0)
        for L in sorted(df["L"].unique()):
            mask = L_arr == L
            order = np.argsort(inv_t[mask])
            style = _HARADA_BSA_STYLE.get(int(L), {"color": "k", "marker": "o", "markersize": 5, "label": f"L={L}"})
            kwargs = {
                "color": style["color"],
                "fmt": style["marker"],
                "linestyle": "none",
                "markersize": max(5, int(style["markersize"]) - 2),
                "capsize": 1,
            }
            x_L = inv_t[mask][order]
            u4_L = u4[mask][order]
            if sigma is not None:
                axins.errorbar(x_L, u4_L, yerr=sigma.to_numpy()[mask][order], **kwargs)
            else:
                axins.plot(x_L, u4_L, **kwargs)
        axins.set_xlim(*inset_xlim)
        axins.set_ylim(*inset_ylim)
        axins.tick_params(labelsize=7)
        axins.grid(True, alpha=0.2)
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.45", lw=0.8)

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    return fig


def plot_harada_bsa_binder(
    *,
    path: Path | None = None,
    out: Path | None = None,
    **kwargs: object,
) -> plt.Figure:
    """Reproduce Harada's published Binder figure from the bundled BSA ``.dat`` file."""
    df = read_harada_binder_df(path)
    return plot_binder_vs_inv_t(df, out=out, **kwargs)  # type: ignore[arg-type]


def plot_binder_dataset_diagnostics(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    T_c_post: float | None = None,
    nu_post: float | None = None,
    T_c_exact: float = TC_EXACT,
    nu_exact: float = NU_EXACT,
    L_ref: float = 256.0,
    dataset: str | None = None,
) -> list[Path]:
    """Write Binder vs 1/T and Harada-collapse figures for a dataset run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    label = dataset or "dataset"
    paths: list[Path] = []

    inv_path = out_dir / "binder_vs_inv_T.png"
    fig = plot_binder_vs_inv_t(
        df,
        out=inv_path,
        T_c=T_c_exact,
        title=rf"Binder cumulant vs $1/T$ — {label}",
    )
    plt.close(fig)
    paths.append(inv_path)

    exact_path = out_dir / "binder_collapse_exact.png"
    fig = plot_binder_collapse(
        df,
        out=exact_path,
        T_c=T_c_exact,
        nu=nu_exact,
        L_ref=L_ref,
        title=(
            rf"Binder collapse at exact exponents "
            rf"($T_c={T_c_exact:.4f}$, $\nu={nu_exact:.3g}$, "
            rf"$L_{{\mathrm{{ref}}}}={int(L_ref)}$) — {label}"
        ),
    )
    plt.close(fig)
    paths.append(exact_path)

    if T_c_post is not None and nu_post is not None:
        post_path = out_dir / "binder_collapse.png"
        fig = plot_binder_collapse(
            df,
            out=post_path,
            T_c=T_c_post,
            nu=nu_post,
            L_ref=L_ref,
            title=(
                rf"Binder collapse at posterior mean "
                rf"($T_c={T_c_post:.4f}$, $\nu={nu_post:.3g}$, "
                rf"$L_{{\mathrm{{ref}}}}={int(L_ref)}$) — {label}"
            ),
        )
        plt.close(fig)
        paths.append(post_path)

    return paths


def load_observables_table(path: Path) -> pd.DataFrame:
    """Load observables CSV; validate when possible, else require binder columns."""
    try:
        return read_observables(path)
    except ValueError:
        df = pd.read_csv(path)
        required = {
            "L",
            "T",
            "binder_cumulant",
            "binder_cumulant_std",
            "n_eff_binder",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"{path} missing binder columns: {sorted(missing)}"
            ) from None
        return df


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data",
        type=Path,
        nargs="?",
        default=None,
        help="Observables CSV (default: harada_square partial if present)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: plots/<stem>/binder_vs_T.png)",
    )
    parser.add_argument(
        "--harada-bsa",
        action="store_true",
        help="Plot Harada published Binder data (BSA Ising-square-Binder.dat)",
    )
    parser.add_argument(
        "--harada-dat",
        type=Path,
        default=None,
        help="Path to Harada BSA .dat file (with --harada-bsa)",
    )
    parser.add_argument(
        "--collapse",
        action="store_true",
        help="Also write collapsed U_4 vs z plot",
    )
    parser.add_argument(
        "--inv-t",
        action="store_true",
        help="Plot U_4 vs 1/T instead of T",
    )
    parser.add_argument(
        "--T-c",
        type=float,
        default=TC_EXACT,
        dest="T_c",
        help=f"Reference T_c for vertical line / collapse (default: {TC_EXACT:.6f})",
    )
    parser.add_argument(
        "--nu",
        type=float,
        default=NU_EXACT,
        help=f"ν for Harada collapse (default: {NU_EXACT})",
    )
    parser.add_argument(
        "--L-ref",
        type=float,
        default=256.0,
        dest="L_ref",
        help="Reference lattice size in Harada abscissa (default: 256)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window (save only)",
    )
    args = parser.parse_args(argv)

    from .datasets import DATA_DIR, PLOTS_DIR

    if args.harada_bsa:
        dat_path = args.harada_dat or HARADA_BINDER_DAT
        df = read_harada_binder_df(dat_path)
        out_dir = PLOTS_DIR / "harada_bsa"
        out_path = args.out or out_dir / "binder_vs_inv_T.png"
        plot_binder_vs_inv_t(df, out=out_path, T_c=args.T_c)
        print(f"Wrote {out_path}")
        if args.collapse:
            out_z = out_path.with_name("binder_collapse.png")
            plot_binder_collapse(
                df, out=out_z, T_c=args.T_c, nu=args.nu, L_ref=args.L_ref
            )
            print(f"Wrote {out_z}")
        if not args.no_show:
            plt.show()
        else:
            plt.close("all")
        return out_path

    if args.data is None:
        partial = DATA_DIR / "observables_harada_square.partial.csv"
        default = DATA_DIR / "observables_harada_square.csv"
        args.data = partial if partial.is_file() else default

    df = load_observables_table(args.data)
    stem = args.data.stem.replace(".partial", "")
    out_dir = DATA_DIR.parent / "plots" / stem

    if args.inv_t:
        out_path = args.out or out_dir / "binder_vs_inv_T.png"
        plot_binder_vs_inv_t(df, out=out_path, T_c=args.T_c)
        print(f"Wrote {out_path}")
    else:
        out_path = args.out or out_dir / "binder_vs_T.png"
        plot_binder_vs_temperature(df, out=out_path, T_c=args.T_c)
        print(f"Wrote {out_path}")

    if args.collapse:
        out_z = out_path.with_name("binder_collapse.png")
        plot_binder_collapse(
            df, out=out_z, T_c=args.T_c, nu=args.nu, L_ref=args.L_ref
        )
        print(f"Wrote {out_z}")

    if not args.no_show:
        plt.show()
    else:
        plt.close("all")

    return out_path


if __name__ == "__main__":
    main()
