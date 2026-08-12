"""Save/load raw Ising MC chains and rebuild observable tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .datasets import observables_path, samples_path
from .sampler import IsingRawSamples, result_to_row, summarize_signed_m_samples


@dataclass(frozen=True)
class StoredSamples:
    """Raw per-sweep signed magnetization for every point in a dataset grid."""

    m_signed: np.ndarray
    L: np.ndarray
    T: np.ndarray
    seed: np.ndarray
    n_thermalize: int
    n_sweeps: int
    sample_stride: int

    @property
    def n_points(self) -> int:
        return int(self.L.size)


@dataclass(frozen=True)
class PointCheckpoint:
    """Partial Wolff chain for one (L, T) grid point."""

    L: int
    T: float
    m_signed: np.ndarray
    seed: int
    n_thermalize: int
    n_sweeps_target: int
    n_sweeps_done: int
    sample_stride: int

    @property
    def n_draws(self) -> int:
        return int(self.m_signed.size)


def point_checkpoint_path(checkpoint_dir: Path, L: int, T: float) -> Path:
    return checkpoint_dir / f"L{L}_T{T:.4f}.npz"


def default_checkpoint_dir(dataset: str) -> Path:
    from .datasets import DATA_DIR

    return DATA_DIR / "checkpoints" / dataset


def _select_chain_draws(
    chain: np.ndarray,
    *,
    max_draws: int | None,
    thin: int,
) -> np.ndarray:
    x = np.asarray(chain, dtype=np.float64).ravel()
    if thin < 1:
        raise ValueError(f"thin must be >= 1, got {thin}")
    if thin > 1:
        x = x[::thin]
    if max_draws is not None:
        if max_draws < 1:
            raise ValueError(f"max_draws must be >= 1, got {max_draws}")
        x = x[:max_draws]
    if x.size < 1:
        raise ValueError("chain selection left no draws")
    return x


def save_point_checkpoint(path: Path, checkpoint: PointCheckpoint) -> Path:
    """Write a partial chain for one grid point (safe to overwrite while running)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_base = path.with_name(path.stem + "._partial")
    np.savez_compressed(
        str(tmp_base),
        m_signed=np.asarray(checkpoint.m_signed, dtype=np.float64),
        L=np.int32(checkpoint.L),
        T=np.float64(checkpoint.T),
        seed=np.int64(checkpoint.seed),
        n_thermalize=np.int32(checkpoint.n_thermalize),
        n_sweeps_target=np.int32(checkpoint.n_sweeps_target),
        n_sweeps_done=np.int32(checkpoint.n_sweeps_done),
        sample_stride=np.int32(checkpoint.sample_stride),
    )
    Path(f"{tmp_base}.npz").replace(path)
    return path


def load_point_checkpoint(path: Path) -> PointCheckpoint:
    with np.load(path, allow_pickle=False) as archive:
        return PointCheckpoint(
            L=int(archive["L"]),
            T=float(archive["T"]),
            m_signed=np.asarray(archive["m_signed"], dtype=np.float64),
            seed=int(archive["seed"]),
            n_thermalize=int(archive["n_thermalize"]),
            n_sweeps_target=int(archive["n_sweeps_target"]),
            n_sweeps_done=int(archive["n_sweeps_done"]),
            sample_stride=int(archive["sample_stride"]),
        )


def list_point_checkpoints(checkpoint_dir: Path) -> list[Path]:
    if not checkpoint_dir.is_dir():
        return []
    return sorted(checkpoint_dir.glob("L*_T*.npz"))


def observables_from_point_checkpoint(checkpoint: PointCheckpoint) -> dict[str, float | int]:
    result = summarize_signed_m_samples(
        checkpoint.m_signed,
        L=checkpoint.L,
        T=checkpoint.T,
        n_thermalize=checkpoint.n_thermalize,
        n_sweeps=checkpoint.n_sweeps_done,
        seed=checkpoint.seed,
    )
    row = result_to_row(result)
    row["n_sweeps_target"] = checkpoint.n_sweeps_target
    row["n_draws"] = checkpoint.n_draws
    return row


def observables_from_checkpoint_dir(checkpoint_dir: Path) -> pd.DataFrame:
    """Build an observables table from all partial point checkpoints in a directory."""
    rows: list[dict[str, float | int]] = []
    for path in list_point_checkpoints(checkpoint_dir):
        rows.append(observables_from_point_checkpoint(load_point_checkpoint(path)))
    if not rows:
        raise ValueError(f"no point checkpoints found in {checkpoint_dir}")
    df = pd.DataFrame(rows)
    return df.sort_values(["L", "T"], ignore_index=True)


def write_checkpoint_observables(checkpoint_dir: Path, *, out: Path) -> Path:
    df = observables_from_checkpoint_dir(checkpoint_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


def save_samples(path: Path, samples: list[IsingRawSamples]) -> Path:
    """Write a compressed NPZ archive of raw chains for one dataset grid."""
    if not samples:
        raise ValueError("samples must be non-empty")

    n_thermalize = samples[0].n_thermalize
    n_sweeps = samples[0].n_sweeps
    sample_stride = samples[0].sample_stride
    n_draws = samples[0].m_signed.size
    for item in samples:
        if (
            item.n_thermalize != n_thermalize
            or item.n_sweeps != n_sweeps
            or item.sample_stride != sample_stride
            or item.m_signed.size != n_draws
        ):
            raise ValueError("all chains in a batch must share sweep metadata and length")

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        m_signed=np.stack([item.m_signed for item in samples], axis=0),
        L=np.asarray([item.L for item in samples], dtype=np.int32),
        T=np.asarray([item.T for item in samples], dtype=np.float64),
        seed=np.asarray([item.seed for item in samples], dtype=np.int64),
        n_thermalize=np.int32(n_thermalize),
        n_sweeps=np.int32(n_sweeps),
        sample_stride=np.int32(sample_stride),
    )
    return path


def load_samples(path: Path) -> StoredSamples:
    """Load a raw-chain archive written by :func:`save_samples`."""
    with np.load(path, allow_pickle=False) as archive:
        return StoredSamples(
            m_signed=np.asarray(archive["m_signed"], dtype=np.float64),
            L=np.asarray(archive["L"], dtype=np.int32),
            T=np.asarray(archive["T"], dtype=np.float64),
            seed=np.asarray(archive["seed"], dtype=np.int64),
            n_thermalize=int(archive["n_thermalize"]),
            n_sweeps=int(archive["n_sweeps"]),
            sample_stride=int(archive["sample_stride"]),
        )


def observables_from_samples(
    stored: StoredSamples,
    *,
    max_draws: int | None = None,
    thin: int = 1,
) -> pd.DataFrame:
    """Rebuild the observables table from stored chains.

    Use ``max_draws`` to truncate each chain (noisier estimates) or ``thin`` to
    subsample every ``k``-th draw.
    """
    rows: list[dict[str, float | int]] = []
    for i in range(stored.n_points):
        chain = _select_chain_draws(
            stored.m_signed[i],
            max_draws=max_draws,
            thin=thin,
        )
        result = summarize_signed_m_samples(
            chain,
            L=int(stored.L[i]),
            T=float(stored.T[i]),
            n_thermalize=stored.n_thermalize,
            n_sweeps=int(chain.size),
            seed=int(stored.seed[i]),
        )
        rows.append(result_to_row(result))
    return pd.DataFrame(rows)


def observables_from_budget(
    stored: StoredSamples,
    n_draws: np.ndarray | list[int],
    *,
    thin: int = 1,
) -> pd.DataFrame:
    """Aggregate each grid point with its own prefix length.

    Points with ``n_draws[i] <= 0`` are omitted. Means, stds, and ``n_eff`` are
    recomputed on the selected prefix (same summarizer as full-chain aggregates).
    """
    n_draws_arr = np.asarray(n_draws, dtype=np.int64).ravel()
    if n_draws_arr.size != stored.n_points:
        raise ValueError(
            f"n_draws length {n_draws_arr.size} != n_points {stored.n_points}"
        )
    rows: list[dict[str, float | int]] = []
    for i in range(stored.n_points):
        n_i = int(n_draws_arr[i])
        if n_i <= 0:
            continue
        chain = _select_chain_draws(
            stored.m_signed[i],
            max_draws=n_i,
            thin=thin,
        )
        result = summarize_signed_m_samples(
            chain,
            L=int(stored.L[i]),
            T=float(stored.T[i]),
            n_thermalize=stored.n_thermalize,
            n_sweeps=int(chain.size),
            seed=int(stored.seed[i]),
        )
        rows.append(result_to_row(result))
    if not rows:
        empty_cols = [
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
            "n_sweeps",
            "n_thermalize",
            "seed",
        ]
        return pd.DataFrame(columns=empty_cols)
    return pd.DataFrame(rows).reset_index(drop=True)


def aggregate_samples_file(
    samples_file: Path,
    *,
    out: Path,
    max_draws: int | None = None,
    thin: int = 1,
) -> Path:
    stored = load_samples(samples_file)
    df = observables_from_samples(stored, max_draws=max_draws, thin=thin)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


def rebuild_observables_from_samples(
    dataset: str,
    *,
    samples: Path | None = None,
    out: Path | None = None,
    max_draws: int | None = None,
    thin: int = 1,
) -> Path:
    """Rebuild ``observables_<dataset>.csv`` from a stored raw-chain NPZ."""
    sample_file = samples or samples_path(dataset)
    if not sample_file.is_file():
        raise FileNotFoundError(f"Raw samples not found: {sample_file}")
    out_path = out or observables_path(dataset)
    return aggregate_samples_file(
        sample_file,
        out=out_path,
        max_draws=max_draws,
        thin=thin,
    )


def observables_csv_stale_vs_samples(
    dataset: str,
    *,
    data: Path | None = None,
    samples: Path | None = None,
) -> bool:
    """True when raw samples exist and are newer than the observables CSV."""
    data_path = data or observables_path(dataset)
    sample_file = samples or samples_path(dataset)
    if not sample_file.is_file() or not data_path.is_file():
        return False
    return sample_file.stat().st_mtime > data_path.stat().st_mtime


def default_samples_path(dataset: str) -> Path:
    return samples_path(dataset)
