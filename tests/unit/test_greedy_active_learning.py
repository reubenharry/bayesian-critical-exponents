"""Tests for greedy active-learning pipeline helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from ising.constants import NU_EXACT, TC_EXACT
from ising.greedy_active_learning import (
    build_al_config,
    rank_excluded_points,
    select_initial_subset,
    select_ranked_point,
)
from ising.observables import make_observables_df
from ising.plot_binder import save_binder_selection_png

az = pytest.importorskip("arviz")


def _harada_like_df() -> pd.DataFrame:
    """Three L strata with T spread around T_c."""
    rows: list[dict[str, float]] = []
    for L in (64, 128, 256):
        for i, delta in enumerate(np.linspace(-0.02, 0.02, 30)):
            T = TC_EXACT * (1.0 + delta)
            rows.append(
                {
                    "L": float(L),
                    "T": float(T),
                    "magnetization": 0.3,
                    "binder_cumulant": 0.6 - 0.01 * abs(delta),
                    "binder_cumulant_std": 0.01 + 0.001 * i,
                }
            )
    L = np.array([r["L"] for r in rows])
    T = np.array([r["T"] for r in rows])
    u4 = np.array([r["binder_cumulant"] for r in rows])
    sigma_u = np.array([r["binder_cumulant_std"] for r in rows])
    return make_observables_df(
        L=L,
        T=T,
        magnetization=np.full(len(rows), 0.3),
        binder_cumulant=u4,
        binder_cumulant_std=sigma_u,
        n_eff_binder=np.full(len(rows), 200.0),
    )


def test_build_al_config_binder_only_rbf():
    cfg = build_al_config()
    assert cfg.use_binder is True
    assert cfg.use_m is False
    assert cfg.correction_binder is False
    assert cfg.gp_kernel == "gaussian"
    assert cfg.gp_ell_factor == pytest.approx(0.15)
    assert cfg.infer_Tc is True
    assert cfg.infer_nu is True
    assert cfg.infer_beta is False


def test_select_initial_subset_stratified_near_tc():
    df = _harada_like_df()
    mask = select_initial_subset(df, n=10, seed=0)
    assert mask.shape == (len(df),)
    assert int(mask.sum()) == 10
    included = df.loc[mask]
    assert set(included["L"].astype(int).unique()) == {64, 128, 256}
    for L in (64, 128, 256):
        sub = included[included["L"] == L]
        assert 2 <= len(sub) <= 4
    # Selected points should be closer to T_c than typical excluded ones.
    excluded = df.loc[~mask]
    inc_dist = np.abs(included["T"].to_numpy() - TC_EXACT).mean()
    exc_dist = np.abs(excluded["T"].to_numpy() - TC_EXACT).mean()
    assert inc_dist < exc_dist


def test_rank_excluded_points_returns_finite_argmax():
    from ising.profile_likelihood import FssProfileConfig

    n = 6
    L = np.full(n, 48.0)
    T = np.linspace(2.265, 2.275, n)
    u4 = np.linspace(0.62, 0.58, n)
    sigma_u = np.linspace(0.02, 0.005, n)
    full_df = make_observables_df(
        L=L,
        T=T,
        magnetization=np.linspace(0.4, 0.15, n),
        binder_cumulant=u4,
        binder_cumulant_std=sigma_u,
        n_eff_binder=np.full(n, 200.0),
    )
    included_mask = np.array([True, True, True, False, False, False])
    included_df = full_df.loc[included_mask].reset_index(drop=True)
    config = FssProfileConfig(
        use_m=False,
        use_m2=False,
        use_m4=False,
        use_binder=True,
        use_chi=False,
        correction_binder=False,
        infer_Tc=True,
        infer_nu=True,
        infer_beta=False,
    )
    idata = az.from_dict(
        posterior={
            "T_c": np.random.default_rng(0).normal(TC_EXACT, 1e-4, (2, 20)),
            "nu": np.random.default_rng(1).normal(NU_EXACT, 0.01, (2, 20)),
        }
    )
    ranked = rank_excluded_points(
        included_df,
        full_df,
        included_mask,
        config,
        idata,
        max_is_draws=40,
        seed=0,
    )
    assert len(ranked) == 3
    assert ranked["delta_tr_cov"].notna().all()
    assert np.isfinite(ranked.iloc[0]["delta_tr_cov"])
    assert ranked.iloc[0]["delta_tr_cov"] >= ranked.iloc[1]["delta_tr_cov"]


def test_select_ranked_point_greedy_vs_worst():
    ranked = pd.DataFrame(
        {
            "row_index": [1, 2, 3],
            "delta_tr_cov": [0.3, 0.1, -0.05],
            "L": [64, 128, 256],
            "T": [2.27, 2.28, 2.26],
        }
    )
    greedy = select_ranked_point(ranked, "greedy")
    worst = select_ranked_point(ranked, "worst")
    assert int(greedy["row_index"]) == 1
    assert int(worst["row_index"]) == 3
    assert float(greedy["delta_tr_cov"]) == pytest.approx(0.3)
    assert float(worst["delta_tr_cov"]) == pytest.approx(-0.05)


def test_save_binder_selection_png_writes_file(tmp_path):
    df = _harada_like_df()
    mask = select_initial_subset(df, n=10, seed=0)
    next_idx = int(np.flatnonzero(~mask)[0])
    out = tmp_path / "binder.png"
    save_binder_selection_png(df, mask, out, next_index=next_idx)
    assert out.is_file()
    assert out.stat().st_size > 0


def test_plot_prediction_discrepancy_writes_file(tmp_path):
    from ising.plot_greedy_al_evolution import (
        plot_prediction_discrepancy,
    )

    table = pd.DataFrame(
        {
            "iteration": [0, 1, 2],
            "predicted_delta_tr_cov": [1e-4, 8e-5, 5e-5],
            "actual_delta_tr_cov": [9e-5, 7e-5, 6e-5],
            "discrepancy": [-1e-5, -1e-5, 1e-5],
        }
    )
    out = tmp_path / "prediction_discrepancy.png"
    plot_prediction_discrepancy(table, out)
    assert out.is_file()
    assert out.stat().st_size > 0
