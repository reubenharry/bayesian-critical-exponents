"""Load φ⁴ observables into the shared Ising FSS / discrepancy stack.

CSV schema matches Ising (column ``T`` holds λ). Use ``TC_EXACT`` / priors from
``phi4.constants`` — not Ising ``T_c ≈ 2.269``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ising.observables import read_observables_for_fss

from .constants import BETA_EXACT, LAM_C, NU_EXACT, TC_EXACT
from .datasets import get_dataset, list_datasets, observables_path

# Priors on the FSS "T_c" parameter (= λ_c for φ⁴).
TC_PRIOR_LOWER = 3.5
TC_PRIOR_UPPER = 5.5


def truth_exponents() -> tuple[float, float, float]:
    """Return ``(λ_c, ν, β)`` used as FSS ground truth for φ⁴."""
    return float(TC_EXACT), float(NU_EXACT), float(BETA_EXACT)


def load_observables(name: str, *, path: Path | None = None) -> pd.DataFrame:
    """Read ``observables_<name>.csv`` for FSS (same columns as Ising)."""
    data_path = path or observables_path(name)
    if not data_path.is_file():
        raise FileNotFoundError(
            f"φ⁴ observables not found: {data_path}. "
            f"Run: python -m phi4.simulate --dataset {name}"
        )
    get_dataset(name)  # validate name
    return read_observables_for_fss(data_path)


def available_fss_datasets() -> list[str]:
    """Named φ⁴ datasets that already have an observables CSV with m and m²."""
    out: list[str] = []
    for name in list_datasets():
        path = observables_path(name)
        if not path.is_file():
            continue
        df = pd.read_csv(path, nrows=1)
        if "magnetization" in df.columns and "m2" in df.columns:
            out.append(name)
    return out


__all__ = [
    "BETA_EXACT",
    "LAM_C",
    "NU_EXACT",
    "TC_EXACT",
    "TC_PRIOR_LOWER",
    "TC_PRIOR_UPPER",
    "available_fss_datasets",
    "load_observables",
    "truth_exponents",
]
