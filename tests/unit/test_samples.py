"""Tests for raw Ising sample storage and re-aggregation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from ising.constants import TC_EXACT
from ising.sampler import run_ising
from ising.samples import (
    aggregate_samples_file,
    load_samples,
    observables_from_samples,
    save_samples,
)
from ising.simulate import simulate_grid


def test_samples_roundtrip_and_truncation(tmp_path: Path):
    _, raw_samples = simulate_grid(
        [(16, TC_EXACT * 0.96), (16, TC_EXACT * 1.04)],
        n_thermalize=20,
        n_sweeps=120,
        seed=4,
        store_samples=True,
    )
    path = tmp_path / "samples.npz"
    save_samples(path, raw_samples)
    stored = load_samples(path)
    full_df = observables_from_samples(stored)
    short_df = observables_from_samples(stored, max_draws=40)

    assert len(full_df) == 2
    pd.testing.assert_series_equal(full_df["L"], short_df["L"])
    assert short_df["magnetization_std"].iloc[0] >= full_df["magnetization_std"].iloc[0]

    out = tmp_path / "obs.csv"
    aggregate_samples_file(path, out=out, max_draws=40)
    rebuilt = pd.read_csv(out)
    pd.testing.assert_frame_equal(short_df, rebuilt)


def test_run_ising_store_samples_matches_summary():
    full = run_ising(16, TC_EXACT * 0.97, n_thermalize=30, n_sweeps=200, seed=1)
    result, raw = run_ising(
        16,
        TC_EXACT * 0.97,
        n_thermalize=30,
        n_sweeps=200,
        seed=1,
        store_samples=True,
    )
    assert raw.m_signed.size == 200
    assert result.magnetization == pytest.approx(full.magnetization)
    assert result.binder_cumulant == pytest.approx(full.binder_cumulant)


def test_run_ising_checkpoint_callback():
    seen: list[tuple[int, int]] = []

    def on_checkpoint(n_sweeps_done: int, m_signed) -> None:
        seen.append((n_sweeps_done, int(m_signed.size)))

    run_ising(
        16,
        TC_EXACT * 0.97,
        n_thermalize=10,
        n_sweeps=250,
        seed=2,
        checkpoint_every=100,
        on_checkpoint=on_checkpoint,
    )
    assert seen == [(100, 100), (200, 200), (250, 250)]


def test_point_checkpoint_roundtrip(tmp_path: Path):
    from ising.samples import (
        PointCheckpoint,
        load_point_checkpoint,
        observables_from_point_checkpoint,
        save_point_checkpoint,
    )

    _, raw = run_ising(
        16,
        TC_EXACT * 0.98,
        n_thermalize=10,
        n_sweeps=120,
        seed=3,
        store_samples=True,
    )
    path = tmp_path / "point.npz"
    ckpt = PointCheckpoint(
        L=raw.L,
        T=raw.T,
        m_signed=raw.m_signed[:60],
        seed=raw.seed,
        n_thermalize=raw.n_thermalize,
        n_sweeps_target=raw.n_sweeps,
        n_sweeps_done=60,
        sample_stride=raw.sample_stride,
    )
    save_point_checkpoint(path, ckpt)
    loaded = load_point_checkpoint(path)
    assert loaded.n_draws == 60
    assert loaded.n_sweeps_target == 120
    row = observables_from_point_checkpoint(loaded)
    assert row["n_draws"] == 60
    assert row["n_sweeps_target"] == 120


def test_simulate_grid_checkpoint_writes_partial_csv(tmp_path: Path):
    ckpt_dir = tmp_path / "ckpt"
    partial = tmp_path / "partial.csv"
    grid = [(16, TC_EXACT * 0.96), (16, TC_EXACT * 1.04)]
    df, _ = simulate_grid(
        grid,
        n_thermalize=10,
        n_sweeps=120,
        seed=1,
        n_jobs=1,
        checkpoint_every=40,
        checkpoint_dir=ckpt_dir,
        partial_out=partial,
    )
    assert len(df) == 2
    assert partial.is_file()
    partial_df = pd.read_csv(partial)
    assert len(partial_df) == 2
    assert len(list(ckpt_dir.glob("L*_T*.npz"))) == 2
