"""Tests for parallel Ising grid simulation."""

from __future__ import annotations

import pandas as pd
from ising.constants import TC_EXACT
from ising.simulate import simulate_grid


def test_simulate_grid_parallel_matches_serial():
    grid = [(16, TC_EXACT * 0.95), (16, TC_EXACT * 1.05)]
    kwargs = dict(n_thermalize=20, n_sweeps=80, seed=3)
    serial, _ = simulate_grid(grid, n_jobs=1, **kwargs)
    parallel, _ = simulate_grid(grid, n_jobs=2, **kwargs)
    pd.testing.assert_frame_equal(serial, parallel)
