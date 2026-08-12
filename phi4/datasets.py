"""Named φ⁴ simulation datasets and path helpers.

Control parameter is the quartic coupling λ, stored in CSV column ``T`` for
FSS schema compatibility. ``TC_EXACT`` / grid helpers use λ_c ≈ 4.25.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .constants import LAM_C, NU_EXACT, TC_EXACT

PHI4_DIR = Path(__file__).resolve().parent
DATA_DIR = PHI4_DIR / "data"
POSTERIORS_DIR = PHI4_DIR / "posteriors"
PLOTS_DIR = PHI4_DIR / "plots"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    L: tuple[int, ...]
    T: tuple[float, ...] = ()  # λ values (column name T for FSS drop-in)
    n_thermalize: int = 0
    n_sweeps: int = 0
    seed: int = 0
    description: str = ""
    z_values: tuple[float, ...] | None = None
    nu_grid: float = NU_EXACT
    points: tuple[tuple[int, float], ...] | None = None

    @property
    def n_T(self) -> int:
        if self.points is not None:
            return max(sum(1 for L, _ in self.points if L == Lval) for Lval in self.L)
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
                (L, lam_from_z(L, z, nu=self.nu_grid))
                for L in self.L
                for z in self.z_values
            ]
        if not self.T:
            raise ValueError(
                f"Dataset {self.name!r}: provide T (λ), z_values, or points."
            )
        return [(L, lam) for L in self.L for lam in self.T]


def lam_from_z(L: int, z: float, *, nu: float = NU_EXACT, lam_c: float = LAM_C) -> float:
    """Coupling at lattice size L for reduced scaling variable z = t L^{1/nu}."""
    t = z * float(L) ** (-1.0 / nu)
    return round(lam_c * (1.0 + t), 4)


def _lam_near_c() -> tuple[float, ...]:
    """λ grid spanning ordered / critical / disordered around λ_c ≈ 4.25."""
    return (
        3.50,
        3.80,
        4.00,
        4.15,
        LAM_C,
        4.35,
        4.50,
        4.80,
        5.20,
        5.50,
    )


def _lam_small() -> tuple[float, ...]:
    return (3.50, 4.00, LAM_C, 4.50, 5.00, 5.50)


DATASETS: dict[str, DatasetSpec] = {
    "small": DatasetSpec(
        name="small",
        L=(16, 32),
        T=_lam_small(),
        n_thermalize=200,
        n_sweeps=500,
        seed=0,
        description=(
            "Smoke-test φ⁴ HMC grid: L=16,32 and 6 λ values around λ_c=4.25 "
            "(short warmup/production)."
        ),
    ),
    "medium": DatasetSpec(
        name="medium",
        L=(12, 16, 24, 32, 40, 48),
        T=_lam_near_c(),
        n_thermalize=500,
        n_sweeps=2000,
        seed=0,
        description=(
            "Default φ⁴ production grid: 6 lattice sizes, 10 λ values near λ_c."
        ),
    ),
    "medium_short": DatasetSpec(
        name="medium_short",
        L=(12, 16, 24, 32, 40, 48),
        T=_lam_near_c(),
        n_thermalize=300,
        n_sweeps=400,
        seed=0,
        description="Same (L, λ) grid as medium with shorter HMC chains.",
    ),
    "tiny": DatasetSpec(
        name="tiny",
        L=(8, 12),
        T=(4.00, LAM_C, 4.50),
        n_thermalize=80,
        n_sweeps=120,
        seed=0,
        description="Ultra-short φ⁴ HMC smoke grid (6 points) for pipeline checks.",
    ),
    "L64_128": DatasetSpec(
        name="L64_128",
        L=(64, 128),
        T=(
            4.05,
            4.15,
            4.20,
            4.22,
            4.24,
            4.25,
            4.27,
            4.28,
            4.30,
            4.32,
            4.35,
            4.38,
            4.40,
            4.45,
            4.55,
            4.80,
        ),
        n_thermalize=800,
        n_sweeps=12000,  # minimum; adaptive ESS may extend
        seed=0,
        description=(
            "Large-L φ⁴ grid: L=64,128 with dense λ near/above λ_c=4.25; "
            "simulate with min ESS(m) >= 250 (adaptive HMC)."
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


def observables_path(name: str) -> Path:
    return DATA_DIR / f"observables_{name}.csv"


def samples_path(name: str) -> Path:
    return DATA_DIR / f"samples_{name}.npz"


def manifest_path(name: str) -> Path:
    return DATA_DIR / f"manifest_{name}.json"


def posterior_path(name: str, *, variant: str = "plain", scaling: str = "gp") -> Path:
    suffix = "" if variant == "plain" else f"_{variant}"
    if scaling != "gp":
        suffix += f"_{scaling}"
    return POSTERIORS_DIR / f"posterior_{name}{suffix}.nc"


def plots_dir(name: str, *, variant: str = "plain", scaling: str = "gp") -> Path:
    suffix = "" if variant == "plain" else f"_{variant}"
    if scaling != "gp":
        suffix += f"_{scaling}"
    return PLOTS_DIR / f"{name}{suffix}"


def write_manifest(spec: DatasetSpec, *, path: Path | None = None) -> Path:
    out = path or manifest_path(spec.name)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(spec)
    payload["n_points"] = spec.n_points
    payload["observables_path"] = str(observables_path(spec.name))
    payload["control_parameter"] = "lam"
    payload["lam_c"] = LAM_C
    payload["tc_exact_alias"] = TC_EXACT
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return out
