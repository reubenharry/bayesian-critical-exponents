"""Load and validate FSS observables tables (one row per (L, T) grid point)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_OBSERVABLE_COLUMNS = frozenset(
    {
        "L",
        "T",
        "magnetization",
        "magnetization_std",
        "n_eff",
        "log_magnetization",
        "log_magnetization_std",
        "n_eff_log",
        "susceptibility",
        "susceptibility_std",
        "n_eff_susceptibility",
        "log_susceptibility",
        "log_susceptibility_std",
        "n_eff_log_susceptibility",
        "binder_cumulant",
        "binder_cumulant_std",
        "n_eff_binder",
        "m2",
        "m2_std",
        "n_eff_m2",
        "m4",
        "m4_std",
        "n_eff_m4",
    }
)

_BASE_OBSERVABLE_COLUMNS = frozenset({"L", "T"})
_MAGNETIZATION_COLUMNS = frozenset(
    {
        "magnetization",
        "magnetization_std",
        "n_eff",
        "log_magnetization",
        "log_magnetization_std",
        "n_eff_log",
    }
)
_M2_COLUMNS = frozenset({"m2", "m2_std", "n_eff_m2"})
_M4_COLUMNS = frozenset({"m4", "m4_std", "n_eff_m4"})
_BINDER_COLUMNS = frozenset(
    {"binder_cumulant", "binder_cumulant_std", "n_eff_binder"}
)
_CHI_COLUMNS = frozenset(
    {
        "susceptibility",
        "susceptibility_std",
        "n_eff_susceptibility",
        "log_susceptibility",
        "log_susceptibility_std",
        "n_eff_log_susceptibility",
    }
)


def required_observable_columns(
    *,
    use_m: bool = True,
    use_m2: bool = False,
    use_m4: bool = False,
    use_binder: bool = True,
    use_chi: bool = False,
) -> frozenset[str]:
    """Columns required for the active FSS likelihood channels."""
    required = set(_BASE_OBSERVABLE_COLUMNS)
    if use_m:
        required |= _MAGNETIZATION_COLUMNS
    if use_m2:
        required |= _M2_COLUMNS
    if use_m4:
        required |= _M4_COLUMNS
    if use_binder:
        required |= _BINDER_COLUMNS
    if use_chi:
        required |= _CHI_COLUMNS
    return frozenset(required)


def validate_observables_df(
    df: pd.DataFrame,
    *,
    use_m: bool | None = None,
    use_m2: bool | None = None,
    use_m4: bool | None = None,
    use_binder: bool | None = None,
    use_chi: bool | None = None,
) -> None:
    """Validate observables columns for full MC tables or active FSS channels."""
    if (
        use_m is None
        and use_m2 is None
        and use_m4 is None
        and use_binder is None
        and use_chi is None
    ):
        required = REQUIRED_OBSERVABLE_COLUMNS
    else:
        required = required_observable_columns(
            use_m=True if use_m is None else use_m,
            use_m2=False if use_m2 is None else use_m2,
            use_m4=False if use_m4 is None else use_m4,
            use_binder=True if use_binder is None else use_binder,
            use_chi=False if use_chi is None else use_chi,
        )
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Observables table is missing required columns: "
            f"{sorted(missing)}."
        )


def read_observables_for_fss(
    path: Path | str,
    *,
    use_m: bool = True,
    use_m2: bool = False,
    use_m4: bool = False,
    use_binder: bool = True,
    use_chi: bool = False,
) -> pd.DataFrame:
    """Read an observables CSV validating only columns for active FSS channels."""
    df = pd.read_csv(path)
    validate_observables_df(
        df,
        use_m=use_m,
        use_m2=use_m2,
        use_m4=use_m4,
        use_binder=use_binder,
        use_chi=use_chi,
    )
    return df


def _observables_column_order() -> list[str]:
    """Canonical column order for full MC observables tables."""
    return [
        "L",
        "T",
        "magnetization",
        "magnetization_std",
        "n_eff",
        "log_magnetization",
        "log_magnetization_std",
        "n_eff_log",
        "susceptibility",
        "susceptibility_std",
        "n_eff_susceptibility",
        "log_susceptibility",
        "log_susceptibility_std",
        "n_eff_log_susceptibility",
        "binder_cumulant",
        "binder_cumulant_std",
        "n_eff_binder",
        "m2",
        "m2_std",
        "n_eff_m2",
        "m4",
        "m4_std",
        "n_eff_m4",
    ]


def pad_observables_table(df: pd.DataFrame) -> pd.DataFrame:
    """Add missing observables columns as NaN (inactive FSS channels)."""
    out = df.copy()
    n = len(out)
    for col in sorted(REQUIRED_OBSERVABLE_COLUMNS - set(out.columns)):
        out[col] = np.full(n, np.nan, dtype=np.float64)
    return out[_observables_column_order()]


def harada_bsa_observables_table(path: Path | None = None) -> pd.DataFrame:
    """Harada BSA Binder data in the full observables schema (NaN inactive channels)."""
    from .datasets import HARADA_BINDER_DAT, read_harada_binder_df

    binder = read_harada_binder_df(path or HARADA_BINDER_DAT)
    slim = binder[
        ["L", "T", "binder_cumulant", "binder_cumulant_std", "n_eff_binder"]
    ].copy()
    df = pad_observables_table(slim)
    validate_observables_df(df)
    return df


def ensure_harada_bsa_observables_csv(path: Path) -> Path:
    """Write ``observables_harada_bsa.csv`` from the published Binder table if missing."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    harada_bsa_observables_table().to_csv(path, index=False)
    return path


def read_harada_binder_observables(path: Path | None = None) -> pd.DataFrame:
    """Load published Harada BSA Binder data for binder-only FSS."""
    from .datasets import HARADA_BINDER_DAT, read_harada_binder_df

    df = read_harada_binder_df(path or HARADA_BINDER_DAT)
    validate_observables_df(
        df,
        use_m=False,
        use_m2=False,
        use_m4=False,
        use_binder=True,
        use_chi=False,
    )
    return df


def read_observables(path: Path | str) -> pd.DataFrame:
    """Read and validate an observables CSV."""
    df = pd.read_csv(path)
    validate_observables_df(df)
    return df


def sigma_from_mc_batch(
    std: np.ndarray,
    n_eff: np.ndarray,
) -> np.ndarray:
    """Convert MC batch std and effective sample size to inferential sigma."""
    std = np.asarray(std, dtype=np.float64).ravel()
    n_eff = np.asarray(n_eff, dtype=np.float64).ravel()
    return std / np.sqrt(np.maximum(n_eff, 1.0))


def resolve_log_m_observables(
    log_magnetization: np.ndarray,
    log_magnetization_std: np.ndarray,
    n_eff_log: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (log_m, sigma_log) from MC log-|m| batch statistics."""
    log_m = np.asarray(log_magnetization, dtype=np.float64).ravel()
    sigma_log = sigma_from_mc_batch(log_magnetization_std, n_eff_log)
    return log_m, sigma_log


def _as_f64(series: pd.Series) -> np.ndarray:
    return series.to_numpy(dtype=np.float64).ravel()


def df_L_T(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    L = _as_f64(df["L"])
    T = _as_f64(df["T"])
    if L.size != T.size:
        raise ValueError("L and T must have the same length")
    return L, T


def df_magnetization_sigmas(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (magnetization, sigma_m, log_m, sigma_log)."""
    magnetization = _as_f64(df["magnetization"])
    sigma_m = sigma_from_mc_batch(
        _as_f64(df["magnetization_std"]),
        _as_f64(df["n_eff"]),
    )
    log_m, sigma_log = resolve_log_m_observables(
        _as_f64(df["log_magnetization"]),
        _as_f64(df["log_magnetization_std"]),
        _as_f64(df["n_eff_log"]),
    )
    return magnetization, sigma_m, log_m, sigma_log


def df_moment_sigmas(
    df: pd.DataFrame,
    *,
    value_col: str,
    std_col: str,
    n_eff_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    values = _as_f64(df[value_col])
    sigma = sigma_from_mc_batch(_as_f64(df[std_col]), _as_f64(df[n_eff_col]))
    return values, sigma


def df_binder_sigmas(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    binder = _as_f64(df["binder_cumulant"])
    sigma = sigma_from_mc_batch(
        _as_f64(df["binder_cumulant_std"]),
        _as_f64(df["n_eff_binder"]),
    )
    return binder, sigma


def df_chi_sigmas(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    chi = _as_f64(df["susceptibility"])
    sigma_chi = sigma_from_mc_batch(
        _as_f64(df["susceptibility_std"]),
        _as_f64(df["n_eff_susceptibility"]),
    )
    return chi, sigma_chi


def make_observables_df(
    *,
    L: np.ndarray,
    T: np.ndarray,
    magnetization: np.ndarray,
    sigma_m: np.ndarray | None = None,
    magnetization_std: np.ndarray | None = None,
    n_eff: np.ndarray | None = None,
    log_m: np.ndarray | None = None,
    log_magnetization_std: np.ndarray | None = None,
    n_eff_log: np.ndarray | None = None,
    m2: np.ndarray | None = None,
    sigma_m2: np.ndarray | None = None,
    m2_std: np.ndarray | None = None,
    n_eff_m2: np.ndarray | None = None,
    m4: np.ndarray | None = None,
    sigma_m4: np.ndarray | None = None,
    m4_std: np.ndarray | None = None,
    n_eff_m4: np.ndarray | None = None,
    binder_cumulant: np.ndarray | None = None,
    sigma_binder: np.ndarray | None = None,
    binder_cumulant_std: np.ndarray | None = None,
    n_eff_binder: np.ndarray | None = None,
    susceptibility: np.ndarray | None = None,
    sigma_chi: np.ndarray | None = None,
    susceptibility_std: np.ndarray | None = None,
    n_eff_susceptibility: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build a validated observables frame from arrays (primarily for tests)."""
    L = np.asarray(L, dtype=np.float64).ravel()
    T = np.asarray(T, dtype=np.float64).ravel()
    magnetization = np.asarray(magnetization, dtype=np.float64).ravel()
    n = L.size
    if not (T.size == n == magnetization.size):
        raise ValueError("L, T, and magnetization must have the same length")

    if sigma_m is None:
        if magnetization_std is None or n_eff is None:
            magnetization_std = np.full(n, 0.02)
            n_eff = np.full(n, 100.0)
        sigma_m = sigma_from_mc_batch(magnetization_std, n_eff)
    else:
        sigma_m = np.asarray(sigma_m, dtype=np.float64).ravel()
        if magnetization_std is None:
            magnetization_std = sigma_m * np.sqrt(100.0)
        if n_eff is None:
            n_eff = np.full(n, 100.0)

    if log_m is None:
        log_m = np.log(np.maximum(magnetization, 1e-12))
    else:
        log_m = np.asarray(log_m, dtype=np.float64).ravel()
    if log_magnetization_std is None:
        log_magnetization_std = np.full(n, 0.01)
    if n_eff_log is None:
        n_eff_log = np.full(n, 100.0)

    if m2 is None:
        m2 = magnetization**2
    if sigma_m2 is None:
        if m2_std is None or n_eff_m2 is None:
            m2_std = np.full(n, 0.01)
            n_eff_m2 = np.full(n, 100.0)
        sigma_m2 = sigma_from_mc_batch(m2_std, n_eff_m2)
    else:
        sigma_m2 = np.asarray(sigma_m2, dtype=np.float64).ravel()
        if m2_std is None:
            m2_std = sigma_m2 * np.sqrt(100.0)
        if n_eff_m2 is None:
            n_eff_m2 = np.full(n, 100.0)

    if m4 is None:
        m4 = magnetization**4
    if sigma_m4 is None:
        if m4_std is None or n_eff_m4 is None:
            m4_std = np.full(n, 0.005)
            n_eff_m4 = np.full(n, 100.0)
        sigma_m4 = sigma_from_mc_batch(m4_std, n_eff_m4)
    else:
        sigma_m4 = np.asarray(sigma_m4, dtype=np.float64).ravel()
        if m4_std is None:
            m4_std = sigma_m4 * np.sqrt(100.0)
        if n_eff_m4 is None:
            n_eff_m4 = np.full(n, 100.0)

    if binder_cumulant is None:
        binder_cumulant = np.linspace(0.62, 0.58, n)
    if sigma_binder is None:
        if binder_cumulant_std is None or n_eff_binder is None:
            binder_cumulant_std = np.full(n, 0.01)
            n_eff_binder = np.full(n, 100.0)
        sigma_binder = sigma_from_mc_batch(binder_cumulant_std, n_eff_binder)
    else:
        sigma_binder = np.asarray(sigma_binder, dtype=np.float64).ravel()
        if binder_cumulant_std is None:
            binder_cumulant_std = sigma_binder * np.sqrt(100.0)
        if n_eff_binder is None:
            n_eff_binder = np.full(n, 100.0)

    if susceptibility is None:
        susceptibility = np.linspace(80.0, 120.0, n)
    if sigma_chi is None:
        if susceptibility_std is None or n_eff_susceptibility is None:
            susceptibility_std = np.full(n, 10.0)
            n_eff_susceptibility = np.full(n, 100.0)
        sigma_chi = sigma_from_mc_batch(susceptibility_std, n_eff_susceptibility)
    else:
        sigma_chi = np.asarray(sigma_chi, dtype=np.float64).ravel()
        if susceptibility_std is None:
            susceptibility_std = sigma_chi * np.sqrt(100.0)
        if n_eff_susceptibility is None:
            n_eff_susceptibility = np.full(n, 100.0)

    log_chi = np.log(np.maximum(susceptibility, 1e-12))
    log_susceptibility_std = np.full(n, 0.05)
    n_eff_log_susceptibility = np.full(n, 100.0)

    df = pad_observables_table(
        pd.DataFrame(
            {
                "L": L,
                "T": T,
                "magnetization": magnetization,
                "magnetization_std": magnetization_std,
                "n_eff": n_eff,
                "log_magnetization": log_m,
                "log_magnetization_std": log_magnetization_std,
                "n_eff_log": n_eff_log,
                "susceptibility": susceptibility,
                "susceptibility_std": susceptibility_std,
                "n_eff_susceptibility": n_eff_susceptibility,
                "log_susceptibility": log_chi,
                "log_susceptibility_std": log_susceptibility_std,
                "n_eff_log_susceptibility": n_eff_log_susceptibility,
                "binder_cumulant": binder_cumulant,
                "binder_cumulant_std": binder_cumulant_std,
                "n_eff_binder": n_eff_binder,
                "m2": m2,
                "m2_std": m2_std,
                "n_eff_m2": n_eff_m2,
                "m4": m4,
                "m4_std": m4_std,
                "n_eff_m4": n_eff_m4,
            }
        )
    )
    validate_observables_df(df)
    return df


def precision_updated_sigma(
    sigma: np.ndarray | float,
    n_eff: np.ndarray | float,
    extra_sweeps: int | float,
) -> np.ndarray:
    """Inferential sigma after adding ``extra_sweeps`` MC samples at fixed batch std."""
    sigma = np.asarray(sigma, dtype=np.float64)
    n_eff = np.asarray(n_eff, dtype=np.float64)
    extra = max(float(extra_sweeps), 0.0)
    return sigma * np.sqrt(n_eff / np.maximum(n_eff + extra, 1.0))


def _combine_mc_mean_std(
    mean_a: float,
    std_a: float,
    n_eff_a: float,
    mean_b: float,
    std_b: float,
    n_eff_b: float,
) -> tuple[float, float, float]:
    """Precision-weighted merge of two MC batches (same observable)."""
    var_a = (std_a / np.sqrt(max(n_eff_a, 1.0))) ** 2
    var_b = (std_b / np.sqrt(max(n_eff_b, 1.0))) ** 2
    prec_a = 0.0 if var_a <= 0.0 else 1.0 / var_a
    prec_b = 0.0 if var_b <= 0.0 else 1.0 / var_b
    prec_total = prec_a + prec_b
    if prec_total <= 0.0:
        mean = 0.5 * (mean_a + mean_b)
        n_eff = n_eff_a + n_eff_b
        std = np.sqrt(max(mean_a, std_a, std_b, 1e-12))
        return float(mean), float(std), float(n_eff)
    mean = (prec_a * mean_a + prec_b * mean_b) / prec_total
    var_mean = 1.0 / prec_total
    n_eff = n_eff_a + n_eff_b
    std = float(np.sqrt(max(var_mean * n_eff, 1e-24)))
    return float(mean), std, float(n_eff)


def merge_observation_row(
    df: pd.DataFrame,
    *,
    L: int | float,
    T: float,
    new_row: dict[str, float | int],
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> pd.DataFrame:
    """Merge a new MC result into the matching ``(L, T)`` row, or append if missing."""
    out = df.copy()
    L = float(L)
    T = float(T)
    mask = np.isclose(out["L"].to_numpy(dtype=np.float64), L, rtol=0.0, atol=0.0) & np.isclose(
        out["T"].to_numpy(dtype=np.float64), T, rtol=rtol, atol=atol
    )
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        row = {"L": int(L) if L == int(L) else L, "T": T}
        row.update(new_row)
        return pd.concat([out, pd.DataFrame([row])], ignore_index=True)

    i = int(idx[0])
    old = out.iloc[i].to_dict()

    def _merge_field(
        mean_key: str,
        std_key: str,
        n_eff_key: str,
    ) -> None:
        if mean_key not in new_row:
            return
        mean, std, n_eff = _combine_mc_mean_std(
            float(old[mean_key]),
            float(old[std_key]),
            float(old[n_eff_key]),
            float(new_row[mean_key]),
            float(new_row[std_key]),
            float(new_row[n_eff_key]),
        )
        old[mean_key] = mean
        old[std_key] = std
        old[n_eff_key] = n_eff

    for mean_key, std_key, n_eff_key in (
        ("magnetization", "magnetization_std", "n_eff"),
        ("log_magnetization", "log_magnetization_std", "n_eff_log"),
        ("susceptibility", "susceptibility_std", "n_eff_susceptibility"),
        ("log_susceptibility", "log_susceptibility_std", "n_eff_log_susceptibility"),
        ("binder_cumulant", "binder_cumulant_std", "n_eff_binder"),
        ("m2", "m2_std", "n_eff_m2"),
        ("m4", "m4_std", "n_eff_m4"),
    ):
        _merge_field(mean_key, std_key, n_eff_key)

    if "n_sweeps" in new_row:
        old["n_sweeps"] = int(old.get("n_sweeps", 0)) + int(new_row["n_sweeps"])
    out.iloc[i] = old
    validate_observables_df(out)
    return out
