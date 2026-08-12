"""Tests for sample-budget active learning helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from ising.active_learning import (
    PosteriorDraws,
    planned_chunk_n_eff,
    rank_sample_chunk_candidates,
    sample_prior_draws,
    score_sample_chunk,
)
from ising.budget_active_learning import (
    parse_l_cycle,
    prepare_binder_observables,
)
from ising.constants import (
    BETA_EXACT,
    NU_EXACT,
    TC_EXACT,
)
from ising.greedy_active_learning import build_al_config
from ising.neff_scaling import (
    COST_L_REF,
    DEFAULT_COLD_START_NEFF_RATE,
    NeffRatePlanner,
    build_online_neff_planner,
    chunk_compute_cost,
)
from ising.observables import make_observables_df
from ising.samples import (
    StoredSamples,
    observables_from_budget,
)


def _tiny_stored(n_points: int = 3, n_sweeps: int = 800) -> StoredSamples:
    rng = np.random.default_rng(0)
    L = np.array([64, 128, 256], dtype=np.int32)[:n_points]
    T = np.array([2.26, 2.27, 2.28], dtype=np.float64)[:n_points]
    # Mild autocorrelation via AR(1)-like signed m.
    m = np.zeros((n_points, n_sweeps), dtype=np.float64)
    for i in range(n_points):
        x = rng.normal(0.0, 0.2, size=n_sweeps)
        for t in range(1, n_sweeps):
            x[t] = 0.8 * x[t - 1] + 0.2 * x[t]
        m[i] = x
    return StoredSamples(
        m_signed=m,
        L=L,
        T=T,
        seed=np.arange(n_points, dtype=np.int64),
        n_thermalize=10,
        n_sweeps=n_sweeps,
        sample_stride=1,
    )


def _full_ref_from_stored(stored: StoredSamples) -> pd.DataFrame:
    from ising.samples import observables_from_samples

    return prepare_binder_observables(observables_from_samples(stored))


def test_planned_chunk_n_eff_scales_linearly():
    assert planned_chunk_n_eff(8000.0, 1, n_chunks_max=8) == pytest.approx(1000.0)
    assert planned_chunk_n_eff(8000.0, 2, n_chunks_max=8) == pytest.approx(2000.0)
    assert planned_chunk_n_eff(8000.0, 8, n_chunks_max=8) == pytest.approx(8000.0)


def test_chunk_compute_cost_scales_as_L_squared():
    assert chunk_compute_cost(64, L_ref=COST_L_REF) == pytest.approx(1.0)
    assert chunk_compute_cost(128, L_ref=COST_L_REF) == pytest.approx(4.0)
    assert chunk_compute_cost(256, L_ref=COST_L_REF) == pytest.approx(16.0)


def test_build_online_neff_planner_cold_start_when_empty():
    planner = build_online_neff_planner(pd.DataFrame(), min_points=6)
    assert planner.kind == "cold_start"
    assert planner.n_train == 0
    n_full = planner.predict_n_eff_full(64, 2.27, n_sweeps_full=80_000)
    assert n_full == pytest.approx(DEFAULT_COLD_START_NEFF_RATE * 80_000)


def test_observables_from_budget_unequal_prefixes_real_neff():
    stored = _tiny_stored(n_points=3, n_sweeps=400)
    n_draws = np.array([100, 0, 200], dtype=np.int64)
    df = observables_from_budget(stored, n_draws)
    assert len(df) == 2
    assert set(df["L"].astype(int)) == {64, 256}
    assert int(df.loc[df["L"] == 64, "n_sweeps"].iloc[0]) == 100
    assert int(df.loc[df["L"] == 256, "n_sweeps"].iloc[0]) == 200
    # Real ESS recomputed on prefixes — positive and finite.
    assert float(df.loc[df["L"] == 64, "n_eff_binder"].iloc[0]) > 0
    assert float(df.loc[df["L"] == 256, "n_eff_binder"].iloc[0]) > 0


def test_observables_from_budget_empty():
    stored = _tiny_stored()
    df = observables_from_budget(stored, np.zeros(stored.n_points, dtype=np.int64))
    assert df.empty
    assert "n_eff_binder" in df.columns


def test_sample_prior_draws_in_bounds():
    draws = sample_prior_draws(
        50, seed=1, infer_Tc=True, infer_nu=True, infer_beta=False
    )
    assert draws.n_samples == 50
    assert draws.T_c.min() >= 2.0
    assert draws.T_c.max() <= 2.5
    assert draws.nu.min() >= 0.5
    assert draws.nu.max() <= 1.5
    assert np.allclose(draws.beta, BETA_EXACT)


def test_parse_l_cycle():
    assert parse_l_cycle(None) is None
    assert parse_l_cycle("") is None
    assert parse_l_cycle("64,128,256") == (64, 128, 256)
    assert parse_l_cycle(" 64, 128 ") == (64, 128)
    with pytest.raises(ValueError):
        parse_l_cycle("64,-1")


def test_rank_sample_chunks_allowed_L_filter():
    stored = _tiny_stored(n_points=3, n_sweeps=400)
    full_ref = _full_ref_from_stored(stored)
    chunk_size = 100
    n_chunks_max = 2
    n_draws = np.zeros(3, dtype=np.int64)
    included = observables_from_budget(stored, n_draws)
    draws = PosteriorDraws(
        T_c=TC_EXACT + np.linspace(-1e-3, 1e-3, 40),
        nu=NU_EXACT + np.linspace(-0.02, 0.02, 40),
        beta=np.full(40, BETA_EXACT),
    )
    planner = NeffRatePlanner(kind="cold_start", cold_rate=0.2, n_train=0)
    ranked = rank_sample_chunk_candidates(
        included,
        full_ref,
        n_draws,
        build_al_config(),
        draws,
        neff_planner=planner,
        chunk_size=chunk_size,
        n_chunks_max=n_chunks_max,
        allowed_L=128,
        infer_Tc=True,
        infer_nu=True,
        infer_beta=False,
    )
    assert len(ranked) == 1
    assert int(ranked.iloc[0]["L"]) == 128
    assert int(ranked.iloc[0]["row_index"]) == 1


def test_rank_sample_chunks_excludes_exhausted_and_uses_planner_neff():
    stored = _tiny_stored(n_points=3, n_sweeps=400)
    full_ref = _full_ref_from_stored(stored)
    # Two chunks available in this tiny fixture setup via n_chunks_max=2, chunk_size=100
    chunk_size = 100
    n_chunks_max = 2
    n_draws = np.array([0, 100, 200], dtype=np.int64)  # point 2 exhausted
    included = observables_from_budget(stored, n_draws)
    draws = PosteriorDraws(
        T_c=TC_EXACT + np.linspace(-1e-3, 1e-3, 40),
        nu=NU_EXACT + np.linspace(-0.02, 0.02, 40),
        beta=np.full(40, BETA_EXACT),
    )
    planner = NeffRatePlanner(kind="cold_start", cold_rate=0.2, n_train=0)
    ranked = rank_sample_chunk_candidates(
        included,
        full_ref,
        n_draws,
        build_al_config(),
        draws,
        neff_planner=planner,
        chunk_size=chunk_size,
        n_chunks_max=n_chunks_max,
        infer_Tc=True,
        infer_nu=True,
        infer_beta=False,
    )
    # Uniform coverage: min chunks = 0, so only virgin point 0 is eligible
    # (point 1 already has 1 chunk; point 2 is exhausted).
    assert set(ranked["row_index"].astype(int)) == {0}
    assert "delta_tr_cov_per_cost" in ranked.columns
    assert "cost" in ranked.columns
    # Planned n_eff from planner, not oracle full_ref n_eff_binder.
    n_sweeps_full = chunk_size * n_chunks_max
    n_eff_full_planned = 0.2 * n_sweeps_full
    planned = float(ranked.loc[ranked["row_index"] == 0, "planned_n_eff_after"].iloc[0])
    assert planned == pytest.approx(
        planned_chunk_n_eff(n_eff_full_planned, 1, n_chunks_max=2)
    )
    # Must not match oracle table ESS schedule.
    n_eff_oracle = float(full_ref.iloc[0]["n_eff_binder"])
    assert planned != pytest.approx(
        planned_chunk_n_eff(n_eff_oracle, 1, n_chunks_max=2)
    )


def test_uniform_coverage_blocks_second_chunk_until_all_have_first():
    from ising.active_learning import (
        enumerate_sample_chunk_candidates,
        min_chunk_count,
    )

    chunk_size = 10_000
    n_draws = np.array([10_000, 0, 10_000, 0], dtype=np.int64)
    assert min_chunk_count(n_draws, chunk_size=chunk_size) == 0
    cands = enumerate_sample_chunk_candidates(
        n_draws, chunk_size=chunk_size, n_chunks_max=8
    )
    # Only the two zeros may receive the first chunk.
    assert {c.chunks_before for c in cands} == {0}
    assert len(cands) == 2

    n_draws_full_first = np.array([10_000, 10_000, 10_000, 10_000], dtype=np.int64)
    assert min_chunk_count(n_draws_full_first, chunk_size=chunk_size) == 1
    cands2 = enumerate_sample_chunk_candidates(
        n_draws_full_first, chunk_size=chunk_size, n_chunks_max=8
    )
    assert len(cands2) == 4
    assert all(c.chunks_before == 1 for c in cands2)


def test_score_virgin_from_empty_dataset():
    """Iteration-0 path: empty D + prior draws + virgin chunk score."""
    n = 4
    L = np.full(n, 64.0)
    T = np.linspace(2.26, 2.28, n)
    full_ref = make_observables_df(
        L=L,
        T=T,
        magnetization=np.full(n, 0.3),
        binder_cumulant=np.linspace(0.61, 0.58, n),
        binder_cumulant_std=np.full(n, 0.02),
        n_eff_binder=np.full(n, 8000.0),
    )
    empty = prepare_binder_observables(pd.DataFrame())
    draws = sample_prior_draws(
        60, seed=0, infer_Tc=True, infer_nu=True, infer_beta=False
    )
    planner = NeffRatePlanner(kind="cold_start", cold_rate=0.1, n_train=0)
    score = score_sample_chunk(
        empty,
        build_al_config(),
        draws,
        full_ref.iloc[1],
        row_index=1,
        chunks_before=0,
        n_chunks_max=8,
        chunk_size=10_000,
        neff_planner=planner,
        infer_Tc=True,
        infer_nu=True,
        infer_beta=False,
    )
    assert np.isfinite(score.delta_uncertainty)
    assert score.ess > 0
    assert planned_chunk_n_eff(0.1 * 80_000, 1, n_chunks_max=8) == pytest.approx(1000.0)


def test_gp_empty_prior_predictive():
    from ising.gp_utils import gp_posterior_predictive

    mean, std = gp_posterior_predictive(
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([0.0, 1.0]),
        length_scale=0.5,
        amplitude=1.2,
    )
    assert np.allclose(mean, 0.0)
    assert np.allclose(std, 1.2)
