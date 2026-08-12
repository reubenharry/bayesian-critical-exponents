"""Tests for budget-AL stepper helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ising.plot_budget_al_stepper import (
    _marker_size,
    build_budget_stepper_frames,
    n_draws_before_iteration,
)


def test_marker_size_grows_with_draws():
    sizes = _marker_size(
        np.array([0, 10_000, 40_000, 80_000]),
        chunk_size=10_000,
        n_chunks_max=8,
    )
    assert sizes[0] < sizes[1] < sizes[2] < sizes[3]
    assert sizes[0] == pytest.approx(5.0)
    assert sizes[3] == pytest.approx(5.0 + 4.5 * 8)


def test_n_draws_before_iteration_accumulates():
    summary = pd.DataFrame(
        {
            "iteration": [0, 1, 2],
            "added_row_index": [1, 1, 0],
            "delta_tr_cov": [0.1, 0.05, 0.02],
        }
    )
    assert np.array_equal(
        n_draws_before_iteration(summary, 3, 0, chunk_size=10_000),
        np.array([0, 0, 0]),
    )
    assert np.array_equal(
        n_draws_before_iteration(summary, 3, 1, chunk_size=10_000),
        np.array([0, 10_000, 0]),
    )
    assert np.array_equal(
        n_draws_before_iteration(summary, 3, 2, chunk_size=10_000),
        np.array([0, 20_000, 0]),
    )


def _write_mini_run(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "budget_al"
    run_dir.mkdir()
    (run_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "selection": "greedy",
                "chunk_size": 100,
                "n_chunks_max": 2,
                "seed": 0,
            }
        )
        + "\n"
    )
    summary = pd.DataFrame(
        {
            "iteration": [0, 1],
            "added_row_index": [0, 1],
            "added_L": [64, 128],
            "added_T": [2.26, 2.27],
            "chunks_before": [0, 0],
            "delta_tr_cov": [0.01, 0.008],
            "n_points": [0, 1],
            "total_draws_before": [0, 100],
        }
    )
    summary.to_csv(run_dir / "summary.csv", index=False)

    # Minimal observables grid for Binder panel.
    obs = pd.DataFrame(
        {
            "L": [64, 128, 256],
            "T": [2.26, 2.27, 2.28],
            "binder_cumulant": [0.61, 0.60, 0.59],
            "binder_cumulant_std": [0.01, 0.01, 0.01],
            "n_eff_binder": [100.0, 100.0, 100.0],
            "magnetization": [0.3, 0.3, 0.3],
            "magnetization_std": [0.01, 0.01, 0.01],
            "n_eff": [100.0, 100.0, 100.0],
            "log_magnetization": [-1.0, -1.0, -1.0],
            "log_magnetization_std": [0.01, 0.01, 0.01],
            "n_eff_log": [100.0, 100.0, 100.0],
            "susceptibility": [1.0, 1.0, 1.0],
            "susceptibility_std": [0.01, 0.01, 0.01],
            "n_eff_susceptibility": [100.0, 100.0, 100.0],
            "log_susceptibility": [0.0, 0.0, 0.0],
            "log_susceptibility_std": [0.01, 0.01, 0.01],
            "n_eff_log_susceptibility": [100.0, 100.0, 100.0],
            "m2": [0.1, 0.1, 0.1],
            "m2_std": [0.01, 0.01, 0.01],
            "n_eff_m2": [100.0, 100.0, 100.0],
            "m4": [0.01, 0.01, 0.01],
            "m4_std": [0.001, 0.001, 0.001],
            "n_eff_m4": [100.0, 100.0, 100.0],
            "n_sweeps": [100, 100, 100],
            "n_thermalize": [0, 0, 0],
            "seed": [0, 1, 2],
        }
    )
    obs_path = run_dir / "observables_harada_square.csv"
    obs.to_csv(obs_path, index=False)

    for it, n_after in ((0, np.array([100, 0, 0])), (1, np.array([100, 100, 0]))):
        iter_dir = run_dir / f"iter_{it:02d}"
        iter_dir.mkdir()
        np.save(iter_dir / "n_draws.npy", n_after)
        pd.DataFrame(
            {
                "row_index": [0, 1, 2],
                "L": [64, 128, 256],
                "T": [2.26, 2.27, 2.28],
                "n_draws": n_after,
            }
        ).to_csv(iter_dir / "budget.csv", index=False)
        ranking = pd.DataFrame(
            {
                "row_index": [0, 1, 2],
                "L": [64, 128, 256],
                "T": [2.26, 2.27, 2.28],
                "delta_tr_cov": [0.01 - 0.001 * i for i in range(3)],
                "tr_cov_before": [0.1, 0.1, 0.1],
                "tr_cov_after": [0.09, 0.09, 0.09],
                "ess": [50.0, 50.0, 50.0],
                "pareto_k": [0.1, 0.1, 0.1],
                "chunks_before": [0, 0, 0],
                "n_draws_before": [0, 0, 0],
                "planned_n_eff_after": [50.0, 50.0, 50.0],
            }
        )
        ranking.to_csv(iter_dir / "ranking.csv", index=False)
        # No posterior.nc — frames should fall back to prior draws.

    return run_dir, obs_path


def test_build_budget_stepper_frames_prior_fallback(tmp_path, monkeypatch):
    run_dir, obs_path = _write_mini_run(tmp_path)

    # Point the grid loader at our temp observables.
    import ising.plot_budget_al_stepper as mod

    monkeypatch.setattr(mod, "observables_path", lambda name: obs_path)

    full_df, frames, tc_lim, nu_lim = build_budget_stepper_frames(run_dir, n_bins=20)
    assert len(full_df) == 3
    assert len(frames) == 2
    assert frames[0].has_posterior is False
    assert frames[0].total_draws == 0
    assert frames[0].next_index == 0
    assert np.array_equal(frames[0].n_draws, np.array([0, 0, 0]))
    assert frames[1].total_draws == 100
    assert np.array_equal(frames[1].n_draws, np.array([100, 0, 0]))
    assert frames[1].next_index == 1
    assert tc_lim[0] < tc_lim[1]
    assert nu_lim[0] < nu_lim[1]


def test_reconstruct_summary_without_summary_csv(tmp_path, monkeypatch):
    """Partial runs with only iter_XX artifacts should still step."""
    run_dir, obs_path = _write_mini_run(tmp_path)
    # No summary.csv — only iter artifacts (like the in-flight job before flush).
    summary_path = run_dir / "summary.csv"
    if summary_path.is_file():
        summary_path.unlink()

    import ising.plot_budget_al_stepper as mod

    monkeypatch.setattr(mod, "observables_path", lambda name: obs_path)

    from ising.plot_budget_al_stepper import (
        discover_completed_iterations,
        reconstruct_summary_from_iters,
    )

    assert discover_completed_iterations(run_dir) == [0, 1]
    summary = reconstruct_summary_from_iters(run_dir, chunk_size=100, n_points=3)
    assert list(summary["iteration"]) == [0, 1]
    assert int(summary.iloc[0]["added_row_index"]) == 0
    assert int(summary.iloc[1]["added_row_index"]) == 1

    _, frames, _, _ = build_budget_stepper_frames(run_dir, n_bins=16)
    assert len(frames) == 2


def test_incomplete_last_iter_is_skipped(tmp_path, monkeypatch):
    run_dir, obs_path = _write_mini_run(tmp_path)
    (run_dir / "summary.csv").unlink(missing_ok=True)
    # In-flight iter_02: ranking only, no n_draws yet.
    incomplete = run_dir / "iter_02"
    incomplete.mkdir()
    pd.DataFrame({"row_index": [0], "delta_tr_cov": [0.001]}).to_csv(
        incomplete / "ranking.csv", index=False
    )

    import ising.plot_budget_al_stepper as mod

    monkeypatch.setattr(mod, "observables_path", lambda name: obs_path)

    from ising.plot_budget_al_stepper import (
        discover_completed_iterations,
    )

    assert discover_completed_iterations(run_dir) == [0, 1]
    _, frames, _, _ = build_budget_stepper_frames(run_dir, n_bins=16)
    assert [f.iteration for f in frames] == [0, 1]
