"""Named simulation datasets and path helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .constants import NU_EXACT, TC_EXACT

ISING_DIR = Path(__file__).resolve().parent
DATA_DIR = ISING_DIR / "data"
POSTERIORS_DIR = ISING_DIR / "posteriors"
PLOTS_DIR = ISING_DIR / "plots"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    L: tuple[int, ...]
    T: tuple[float, ...] = ()
    n_thermalize: int = 0
    n_sweeps: int = 0
    seed: int = 0
    description: str = ""
    # When set, temperatures are T(L, z) = T_c (1 + z L^{-1/nu}) per (L, z) pair
    # instead of the Cartesian product L × T.
    z_values: tuple[float, ...] | None = None
    nu_grid: float = NU_EXACT
    # Explicit (L, T) pairs (e.g. Harada BSA Binder table); overrides L × T.
    points: tuple[tuple[int, float], ...] | None = None

    @property
    def n_T(self) -> int:
        if self.points is not None:
            return max(
                sum(1 for L, _ in self.points if L == Lval) for Lval in self.L
            )
        if self.z_values is not None:
            return len(self.z_values)
        return len(self.T)

    @property
    def n_points(self) -> int:
        if self.points is not None:
            return len(self.points)
        return len(self.L) * self.n_T

    def grid(self) -> list[tuple[int, float]]:
        if self.points is not None:
            return list(self.points)
        if self.z_values is not None:
            return [
                (L, T_from_z(L, z, nu=self.nu_grid))
                for L in self.L
                for z in self.z_values
            ]
        if not self.T:
            raise ValueError(
                f"Dataset {self.name!r}: provide T, z_values, or points for the (L, T) grid."
            )
        return [(L, T) for L in self.L for T in self.T]


def _T_near_tc() -> tuple[float, ...]:
    return (
        2.00,
        2.05,
        2.10,
        2.15,
        2.20,
        2.24,
        TC_EXACT,
        2.28,
        2.30,
        2.32,
    )


def _T_dense() -> tuple[float, ...]:
    coarse = np.linspace(2.00, 2.34, 13)
    near_tc = np.linspace(2.22, 2.30, 9)
    merged = sorted(set(np.round(np.concatenate([coarse, near_tc]), 4).tolist()))
    return tuple(float(T) for T in merged)


def _T_critical() -> tuple[float, ...]:
    """Symmetric window ±~0.015 around T_c (|z| <~ 0.4 for L <= 64, nu=1)."""
    deltas = (-0.015, -0.010, -0.005, 0.0, 0.005, 0.010, 0.015)
    return tuple(round(TC_EXACT * (1.0 + d), 4) for d in deltas)


def T_from_z(L: int, z: float, *, nu: float = NU_EXACT) -> float:
    """Temperature at lattice size L for reduced scaling variable z = t L^{1/nu}."""
    t = z * float(L) ** (-1.0 / nu)
    return round(TC_EXACT * (1.0 + t), 4)


def _Z_near_tc() -> tuple[float, ...]:
    """Reduced temperatures spanning the scaling regime (|z| <= 0.3)."""
    return (-0.30, -0.15, -0.05, 0.0, 0.05, 0.15, 0.30)


def _Z_fss_dense() -> tuple[float, ...]:
    """Eleven z-points in [-0.3, 0.3], denser near z=0 for T_c interpolation."""
    return (-0.30, -0.20, -0.12, -0.06, -0.03, 0.0, 0.03, 0.06, 0.12, 0.20, 0.30)


def T_from_harada_x(
    x: float,
    *,
    T_c: float = TC_EXACT,
) -> float:
    """Temperature at ``L_ref=256`` for Harada collapse variable ``x = 1/T - 1/T_c``."""
    inv_t = 1.0 / T_c + x
    if inv_t <= 0.0:
        raise ValueError(f"Harada x={x} gives non-positive 1/T")
    return round(1.0 / inv_t, 4)


def T_from_harada_x_at_L(
    x: float,
    L: int,
    *,
    T_c: float = TC_EXACT,
    nu: float = NU_EXACT,
    L_ref: int = 256,
) -> float:
    """``T(L,x)`` so ``x = (1/T - 1/T_c)(L/L_ref)^{1/\\nu}`` (fixed-x collapse grid)."""
    scale = (float(L) / float(L_ref)) ** (1.0 / nu)
    inv_t = 1.0 / T_c + x / scale
    if inv_t <= 0.0:
        raise ValueError(f"Harada x={x} at L={L} gives non-positive 1/T")
    return round(1.0 / inv_t, 6)


def T_from_inv_t(inv_t: float) -> float:
    """Convert inverse temperature ``1/T`` (Harada BSA format) to ``T``."""
    if inv_t <= 0.0:
        raise ValueError(f"1/T must be positive, got {inv_t}")
    return round(1.0 / inv_t, 6)


HARADA_BINDER_DAT = DATA_DIR / "harada_ising_square_binder.dat"


def parse_harada_binder_dat(
    path: Path | None = None,
) -> tuple[tuple[int, float], ...]:
    """Parse BSA ``Ising-square-Binder.dat``: ``L  1/T  U4  err`` → ``(L, T)``."""
    df = read_harada_binder_df(path)
    return tuple((int(L), float(T)) for L, T in zip(df["L"], df["T"]))


def read_harada_binder_df(path: Path | None = None) -> pd.DataFrame:
    """Load Harada BSA Binder table as a DataFrame.

    Columns: ``L``, ``inv_T``, ``T``, ``binder_cumulant``, ``binder_cumulant_std``,
    ``n_eff_binder``. The ``Error`` column in the ``.dat`` file is treated as the
    inferential uncertainty on ``U_4`` (``n_eff_binder=1`` so ``sigma = Error``).
    """
    import pandas as pd

    path = path or HARADA_BINDER_DAT
    rows: list[dict[str, float | int]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        L = int(parts[0])
        inv_t = float(parts[1])
        u4 = float(parts[2])
        err = float(parts[3])
        rows.append(
            {
                "L": L,
                "inv_T": inv_t,
                "T": T_from_inv_t(inv_t),
                "binder_cumulant": u4,
                "binder_cumulant_std": err,
                "n_eff_binder": 1.0,
            }
        )
    if not rows:
        raise ValueError(f"no Binder rows found in {path}")
    return pd.DataFrame(rows)


def _harada_binder_points() -> tuple[tuple[int, float], ...]:
    return parse_harada_binder_dat(HARADA_BINDER_DAT)


def _harada_square_large_t_extra_points() -> tuple[tuple[int, float], ...]:
    """Small-L × large-|t| points that leave the FSS asymptotic window.

    Harada's published grid is already at large L (64–256) and modest |t|.
    Breakdown of a pure scaling form is clearest when *both* limits fail:
    small L and temperatures well away from T_c.
    """
    small_L = (8, 12, 16, 24, 32)
    # |t| from ~0.25 (T=1.70) up to ~2.5 (T=8), both sides of T_c.
    T_vals = (
        1.70,
        1.85,
        1.95,
        2.05,
        2.55,
        2.70,
        2.90,
        3.20,
        3.60,
        4.20,
        5.00,
        6.50,
        8.00,
    )
    return tuple((L, float(T)) for L in small_L for T in T_vals)


def _harada_square_large_t_points() -> tuple[tuple[int, float], ...]:
    """Harada Binder grid plus small-L / large-|t| extension."""
    base = _harada_binder_points()
    base_set = {(int(L), round(float(T), 6)) for L, T in base}
    extra = [
        (L, T)
        for L, T in _harada_square_large_t_extra_points()
        if (int(L), round(float(T), 6)) not in base_set
    ]
    return base + tuple(extra)


# Single large-|t| contaminant used by ``harada_square_outlier``.
HARADA_SQUARE_OUTLIER_POINT: tuple[int, float] = (8, 8.0)


def _harada_square_outlier_points() -> tuple[tuple[int, float], ...]:
    """Harada Binder grid plus one (L=8, T=8) outlier."""
    return _harada_binder_points() + (HARADA_SQUARE_OUTLIER_POINT,)


def _X_harada_square() -> tuple[float, ...]:
    """Eleven x-points in [-0.01, 0.01] at L_ref=256 (collapse figure only)."""
    return (-0.010, -0.007, -0.005, -0.003, -0.0015, 0.0, 0.0015, 0.003, 0.005, 0.007, 0.010)


def _T_harada_square_legacy_shared() -> tuple[float, ...]:
    """Legacy: same T for all L (L=256 x-grid only). Prefer ``harada_binder_points``."""
    return tuple(T_from_harada_x(x) for x in _X_harada_square())


def _small_grid() -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Sparse (L, T) grid for fast inference and crossover smoke tests."""
    return (
        (16, 32),
        (2.00, 2.15, 2.20, TC_EXACT, 2.30, 2.32),
    )


_SMALL_L, _SMALL_T = _small_grid()
_SMALL_LONG_T = _SMALL_T + (2.40, 2.50)


DATASETS: dict[str, DatasetSpec] = {
    "small": DatasetSpec(
        name="small",
        L=_SMALL_L,
        T=_SMALL_T,
        n_thermalize=500,
        n_sweeps=2000,
        seed=0,
        description=(
            "Smoke-test grid: 2 lattice sizes, 6 temperatures "
            "(bulk-ordered, subcritical, T_c, above T_c)."
        ),
    ),
    "small_long": DatasetSpec(
        name="small_long",
        L=_SMALL_L,
        T=_SMALL_LONG_T,
        n_thermalize=1000,
        n_sweeps=8000,
        seed=0,
        description=(
            "Sparse grid like small plus T=2.40, 2.50 (16 points) with long chains "
            "for lower Monte Carlo noise while keeping inference cost low."
        ),
    ),
    "medium": DatasetSpec(
        name="medium",
        L=(12, 16, 24, 32, 40, 48),
        T=_T_near_tc(),
        n_thermalize=500,
        n_sweeps=4000,
        seed=0,
        description="Default production grid: 6 lattice sizes, 10 temperatures.",
    ),
    "medium_short": DatasetSpec(
        name="medium_short",
        L=(12, 16, 24, 32, 40, 48),
        T=_T_near_tc(),
        n_thermalize=500,
        n_sweeps=800,
        seed=0,
        description=(
            "Same (L, T) grid as medium with n_sweeps=800 (~1/5 of medium) "
            "for noisier MC errors and wider exponent posteriors."
        ),
    ),
    "medium_veryshort": DatasetSpec(
        name="medium_veryshort",
        L=(12, 16, 24, 32, 40, 48),
        T=_T_near_tc(),
        n_thermalize=500,
        n_sweeps=22,
        seed=0,
        description=(
            "Same (L, T) grid as medium with n_sweeps=22 (~1/182 of medium) "
            "for ~10 effective magnetization samples per point."
        ),
    ),
    "large": DatasetSpec(
        name="large",
        L=(8, 12, 16, 20, 24, 28, 32, 40, 48),
        T=_T_dense(),
        n_thermalize=1000,
        n_sweeps=8000,
        seed=0,
        description="High-statistics grid: 9 lattice sizes, dense temperature sampling, long chains.",
    ),
    "critical": DatasetSpec(
        name="critical",
        L=(16, 32, 48, 64),
        T=_T_critical(),
        n_thermalize=1000,
        n_sweeps=12000,
        seed=0,
        description=(
            "Near-T_c grid for joint m + Binder FSS: 4 lattice sizes, 7 symmetric "
            "temperatures (|t|<=0.015), long chains for U_4 moments."
        ),
    ),
    "critical_large_z": DatasetSpec(
        name="critical_large_z",
        L=(48, 64, 96),
        z_values=_Z_near_tc(),
        n_thermalize=3000,
        n_sweeps=30000,
        seed=0,
        description=(
            "Large-L scaling-regime grid: T(L) chosen so |z|<=0.3 is constant "
            "across L, 7 z-points, very long Wolff chains for low MC noise."
        ),
    ),
    "critical_fss": DatasetSpec(
        name="critical_fss",
        L=(64, 96, 128, 192, 256),
        z_values=_Z_fss_dense(),
        n_thermalize=4000,
        n_sweeps=40000,
        seed=0,
        description=(
            "FSS-focused grid: five lattice sizes (64–256, ratio 4), eleven z-points "
            "denser near z=0, long Wolff chains for classical Binder/log-log analysis."
        ),
    ),
    "harada_bsa": DatasetSpec(
        name="harada_bsa",
        L=(64, 128, 256),
        points=_harada_binder_points(),
        description=(
            "Published Harada BSA Ising-square-Binder.dat (86 points): "
            "use for binder-only FSS/profile; no MC simulation required."
        ),
    ),
    "harada_square": DatasetSpec(
        name="harada_square",
        L=(64, 128, 256),
        n_thermalize=8000,
        n_sweeps=80000,
        seed=0,
        points=_harada_binder_points(),
        description=(
            "Harada PRE 84 056704 square-lattice Binder MC grid from BSA "
            "Ising-square-Binder.dat: 28/29/29 (1/T) points at L=64/128/256 "
            "(86 total), denser near 1/T_c≈0.4407; 80k Wolff sweeps per point."
        ),
    ),
    "harada_square_large_t": DatasetSpec(
        name="harada_square_large_t",
        L=(8, 12, 16, 24, 32, 64, 128, 256),
        n_thermalize=2000,
        n_sweeps=20000,
        seed=0,
        points=_harada_square_large_t_points(),
        description=(
            "Harada square Binder grid (L=64/128/256) plus small-L×large-|t| "
            "points (L=8–32, T∈[1.70,8]) to expose FSS collapse failure away "
            "from the joint t→0, L→∞ limit."
        ),
    ),
    "harada_square_outlier": DatasetSpec(
        name="harada_square_outlier",
        L=(8, 64, 128, 256),
        n_thermalize=2000,
        n_sweeps=20000,
        seed=0,
        points=_harada_square_outlier_points(),
        description=(
            "Exact Harada square Binder grid (L=64/128/256) plus a single "
            "outlier at L=8, T=8 (|t|≈2.5) to probe discrepancy response to "
            "one large-|t|, small-L contaminant."
        ),
    ),
}


def list_datasets() -> list[str]:
    return list(DATASETS.keys())


def get_dataset(name: str) -> DatasetSpec:
    try:
        return DATASETS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown dataset {name!r}. Available: {', '.join(list_datasets())}"
        ) from exc


def _variant_suffix(variant: str) -> str:
    return "" if variant == "plain" else f"_{variant}"


def _scaling_suffix(scaling: str) -> str:
    return "" if scaling == "gp" else f"_{scaling}"


def _model_suffix(model: str) -> str:
    return "" if model == "fss" else f"_{model}"


def observables_path(name: str) -> Path:
    return DATA_DIR / f"observables_{name}.csv"


def samples_path(name: str) -> Path:
    return DATA_DIR / f"samples_{name}.npz"


def manifest_path(name: str) -> Path:
    return DATA_DIR / f"manifest_{name}.json"


def posterior_path(
    name: str,
    *,
    model: str = "fss",
    method: str = "marginal",
    variant: str = "plain",
    scaling: str = "gp",
) -> Path:
    if model == "fss":
        stem = f"posterior_{name}{_variant_suffix(variant)}{_scaling_suffix(scaling)}"
    else:
        stem = (
            f"posterior_{name}{_model_suffix(model)}"
            f"{_variant_suffix(variant)}{_scaling_suffix(scaling)}"
        )
    return POSTERIORS_DIR / f"{stem}.nc"


def plots_dir(
    name: str,
    *,
    model: str = "fss",
    variant: str = "plain",
    scaling: str = "gp",
) -> Path:
    if model == "fss":
        return PLOTS_DIR / f"{name}{_variant_suffix(variant)}{_scaling_suffix(scaling)}"
    return (
        PLOTS_DIR
        / f"{name}{_model_suffix(model)}{_variant_suffix(variant)}{_scaling_suffix(scaling)}"
    )


def write_manifest(spec: DatasetSpec, *, path: Path | None = None) -> Path:
    out = path or manifest_path(spec.name)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(spec)
    payload["n_points"] = spec.n_points
    payload["observables_path"] = str(observables_path(spec.name))
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return out


def resolve_paths(
    dataset: str,
    *,
    model: str = "fss",
    method: str = "marginal",
    variant: str = "plain",
    scaling: str = "gp",
    data: Path | None = None,
    posterior: Path | None = None,
    plots: Path | None = None,
) -> tuple[Path, Path, Path]:
    """Return (observables_csv, posterior_nc, plots_dir) for a named dataset."""
    get_dataset(dataset)  # validate
    return (
        data or observables_path(dataset),
        posterior or posterior_path(
            dataset, model=model, method=method, variant=variant, scaling=scaling
        ),
        plots or plots_dir(dataset, model=model, variant=variant, scaling=scaling),
    )
