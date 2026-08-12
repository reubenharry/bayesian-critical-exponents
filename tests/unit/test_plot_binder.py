"""Tests for Binder cumulant plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import numpy as np
from ising.constants import TC_EXACT
from ising.datasets import read_harada_binder_df
from ising.observables import make_observables_df
from ising.plot_binder import (
    harada_collapse_x,
    load_observables_table,
    plot_binder_collapse,
    plot_binder_vs_inv_t,
    plot_binder_vs_temperature,
    plot_harada_bsa_binder,
)


def test_plot_binder_from_partial_like_df(tmp_path: Path):
    n = 6
    L = [16, 16, 16, 32, 32, 32]
    T = [2.25, TC_EXACT, 2.30, 2.25, TC_EXACT, 2.30]
    u4 = [0.62, 0.66, 0.60, 0.61, 0.655, 0.59]
    df = make_observables_df(
        L=L,
        T=T,
        magnetization=[0.5] * n,
        binder_cumulant=u4,
    )
    df["n_draws"] = [80000, 40000, 80000, 80000, 20000, 80000]
    df["n_sweeps_target"] = 80000

    out = tmp_path / "binder.png"
    fig = plot_binder_vs_temperature(df, out=out, show_exact_tc=True)
    assert out.is_file()
    assert fig.axes[0].get_legend() is not None
    matplotlib.pyplot.close(fig)

    out_z = tmp_path / "collapse.png"
    fig2 = plot_binder_collapse(df, out=out_z)
    assert out_z.is_file()
    matplotlib.pyplot.close(fig2)


def test_harada_collapse_x_at_tc():
    x = harada_collapse_x(
        np.array([64.0, 128.0, 256.0]),
        np.array([TC_EXACT, TC_EXACT, TC_EXACT]),
        L_ref=256.0,
    )
    assert np.allclose(x, 0.0)
    x256 = harada_collapse_x(
        np.array([256.0]),
        np.array([1.0 / (1.0 / TC_EXACT + 0.01)]),
        L_ref=256.0,
    )
    assert np.allclose(x256, 0.01, rtol=1e-3)


def test_load_observables_table_partial_csv(tmp_path: Path):
    path = tmp_path / "partial.csv"
    df = make_observables_df(
        L=[16],
        T=[TC_EXACT],
        magnetization=[0.5],
        binder_cumulant=[0.65],
    )
    df["n_draws"] = [1000]
    df["n_sweeps_target"] = [80000]
    df.to_csv(path, index=False)
    loaded = load_observables_table(path)
    assert len(loaded) == 1


def test_read_harada_binder_df():
    df = read_harada_binder_df()
    assert len(df) == 86
    assert set(df["L"].unique()) == {64, 128, 256}
    assert "inv_T" in df.columns
    assert df["binder_cumulant"].between(0.0, 1.0).all()


def test_plot_harada_bsa_binder(tmp_path: Path):
    out = tmp_path / "harada_binder.png"
    fig = plot_harada_bsa_binder(out=out)
    assert out.is_file()
    ax = fig.axes[0]
    assert ax.get_xlabel() == r"$1/T$"
    assert len(ax.get_legend().get_texts()) == 3
    matplotlib.pyplot.close(fig)
