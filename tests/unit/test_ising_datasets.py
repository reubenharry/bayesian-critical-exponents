"""Tests for Ising dataset grid presets."""

from __future__ import annotations

from collections import Counter

import pytest
from ising.constants import NU_EXACT, TC_EXACT
from ising.datasets import (
    HARADA_BINDER_DAT,
    T_from_harada_x_at_L,
    T_from_z,
    get_dataset,
    parse_harada_binder_dat,
)


def test_T_from_z_at_criticality():
    assert T_from_z(64, 0.0) == round(TC_EXACT, 4)


def test_z_grid_keeps_reduced_temperature_constant():
    spec = get_dataset("critical_large_z")
    grid = spec.grid()
    assert len(grid) == spec.n_points == 21
    for L, T in grid:
        z = (T / TC_EXACT - 1.0) * L ** (1.0 / NU_EXACT)
        assert abs(z) <= 0.30 + 1e-9
        assert abs(z - round(z, 2)) < 0.06 or z == 0.0


def test_harada_binder_dat_point_counts():
    points = parse_harada_binder_dat(HARADA_BINDER_DAT)
    counts = Counter(L for L, _ in points)
    assert counts[64] == 28
    assert counts[128] == 29
    assert counts[256] == 29
    assert len(points) == 86


def test_harada_square_matches_bsa_grid():
    spec = get_dataset("harada_square")
    assert spec.n_points == 86
    assert spec.points is not None
    assert spec.grid() == list(spec.points)
    counts = Counter(L for L, _ in spec.grid())
    assert counts == {64: 28, 128: 29, 256: 29}


def test_harada_square_L64_spans_wide_inv_t():
    spec = get_dataset("harada_square")
    inv_t = sorted(1.0 / T for L, T in spec.grid() if L == 64)
    assert inv_t[0] == pytest.approx(0.40)
    assert inv_t[-1] == pytest.approx(0.475)
    assert len(inv_t) == 28


def test_harada_square_large_t_extends_beyond_harada():
    base = get_dataset("harada_square")
    large = get_dataset("harada_square_large_t")
    base_set = set(base.grid())
    large_set = set(large.grid())
    assert base_set <= large_set
    extra = large_set - base_set
    assert len(extra) == 5 * 13  # L=8,12,16,24,32 × 13 temperatures
    assert max(T for _, T in extra) == pytest.approx(8.0)
    assert min(T for _, T in extra) == pytest.approx(1.70)
    assert max(L for L, _ in extra) == 32
    assert min(L for L, _ in extra) == 8
    assert (8, 8.0) in large_set
    assert (32, 1.70) in large_set
    # Extras are small-L only; Harada sizes stay on the published grid.
    assert (64, 8.0) not in large_set
    assert (256, 1.70) not in large_set


def test_harada_square_outlier_is_harada_plus_one():
    from ising.datasets import (
        HARADA_SQUARE_OUTLIER_POINT,
    )

    base = get_dataset("harada_square")
    outlier = get_dataset("harada_square_outlier")
    base_set = set(base.grid())
    out_set = set(outlier.grid())
    assert base_set <= out_set
    assert out_set - base_set == {HARADA_SQUARE_OUTLIER_POINT}
    assert HARADA_SQUARE_OUTLIER_POINT == (8, 8.0)


def test_harada_bsa_binder_observables():
    from ising.observables import (
        read_harada_binder_observables,
        required_observable_columns,
    )

    df = read_harada_binder_observables()
    assert len(df) == 86
    required = required_observable_columns(
        use_m=False,
        use_m2=False,
        use_m4=False,
        use_binder=True,
        use_chi=False,
    )
    assert required <= set(df.columns)


def test_harada_bsa_observables_table_pads_inactive_channels():
    from ising.observables import (
        REQUIRED_OBSERVABLE_COLUMNS,
        harada_bsa_observables_table,
    )

    df = harada_bsa_observables_table()
    assert len(df) == 86
    assert set(REQUIRED_OBSERVABLE_COLUMNS) <= set(df.columns)
    assert df["magnetization"].isna().all()
    assert df["binder_cumulant"].notna().all()


def test_ensure_harada_bsa_observables_csv_writes_cache(tmp_path):
    from ising.observables import (
        ensure_harada_bsa_observables_csv,
    )

    path = tmp_path / "observables_harada_bsa.csv"
    out = ensure_harada_bsa_observables_csv(path)
    assert out == path
    assert path.exists()
    assert ensure_harada_bsa_observables_csv(path) == path


def test_T_from_harada_x_at_L_differs_by_size():
    """Fixed-x collapse grid gives different T per L (unlike legacy shared-T)."""
    x = -0.01
    t64 = T_from_harada_x_at_L(x, 64)
    t256 = T_from_harada_x_at_L(x, 256)
    assert t64 != t256
