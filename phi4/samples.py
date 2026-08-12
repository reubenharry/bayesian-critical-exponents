"""Save/load raw φ⁴ HMC magnetization chains and rebuild observable tables.

NPZ layout matches Ising (``m_signed``, ``L``, ``T``, …) with ``T`` holding λ.
"""

from __future__ import annotations

from pathlib import Path

from ising.samples import (
    PointCheckpoint,
    StoredSamples,
    aggregate_samples_file,
    list_point_checkpoints,
    load_point_checkpoint,
    load_samples,
    observables_from_budget,
    observables_from_checkpoint_dir,
    observables_from_point_checkpoint,
    observables_from_samples,
    point_checkpoint_path,
    save_point_checkpoint,
    save_samples,
    write_checkpoint_observables,
)

from .datasets import DATA_DIR, observables_path, samples_path

__all__ = [
    "PointCheckpoint",
    "StoredSamples",
    "aggregate_samples_file",
    "default_checkpoint_dir",
    "list_point_checkpoints",
    "load_point_checkpoint",
    "load_samples",
    "observables_from_budget",
    "observables_from_checkpoint_dir",
    "observables_from_point_checkpoint",
    "observables_from_samples",
    "point_checkpoint_path",
    "rebuild_observables_from_samples",
    "save_point_checkpoint",
    "save_samples",
    "write_checkpoint_observables",
]


def default_checkpoint_dir(dataset: str) -> Path:
    return DATA_DIR / "checkpoints" / dataset


def rebuild_observables_from_samples(
    dataset: str,
    *,
    samples: Path | None = None,
    out: Path | None = None,
    max_draws: int | None = None,
    thin: int = 1,
) -> Path:
    """Rebuild ``observables_<dataset>.csv`` under the φ⁴ data directory."""
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
    data_path = data or observables_path(dataset)
    sample_file = samples or samples_path(dataset)
    if not sample_file.is_file() or not data_path.is_file():
        return False
    return sample_file.stat().st_mtime > data_path.stat().st_mtime
