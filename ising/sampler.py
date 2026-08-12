"""2D ferromagnetic Ising model sampler (Wolff cluster updates)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from blackjax.diagnostics import effective_sample_size


@dataclass(frozen=True)
class IsingRawSamples:
    """Per-sweep signed magnetization per spin from one Wolff chain."""

    L: int
    T: float
    m_signed: np.ndarray
    n_thermalize: int
    n_sweeps: int
    sample_stride: int
    seed: int


@dataclass(frozen=True)
class IsingRunResult:
    L: int
    T: float
    magnetization: float
    magnetization_std: float
    n_eff: float
    log_magnetization: float
    log_magnetization_std: float
    n_eff_log: float
    susceptibility: float
    susceptibility_std: float
    n_eff_susceptibility: float
    log_susceptibility: float
    log_susceptibility_std: float
    n_eff_log_susceptibility: float
    binder_cumulant: float
    binder_cumulant_std: float
    n_eff_binder: float
    m2: float
    m2_std: float
    n_eff_m2: float
    m4: float
    m4_std: float
    n_eff_m4: float
    n_sweeps: int
    n_thermalize: int
    seed: int


def result_to_row(result: IsingRunResult) -> dict[str, float | int]:
    return {
        "L": result.L,
        "T": result.T,
        "magnetization": result.magnetization,
        "magnetization_std": result.magnetization_std,
        "n_eff": result.n_eff,
        "log_magnetization": result.log_magnetization,
        "log_magnetization_std": result.log_magnetization_std,
        "n_eff_log": result.n_eff_log,
        "susceptibility": result.susceptibility,
        "susceptibility_std": result.susceptibility_std,
        "n_eff_susceptibility": result.n_eff_susceptibility,
        "log_susceptibility": result.log_susceptibility,
        "log_susceptibility_std": result.log_susceptibility_std,
        "n_eff_log_susceptibility": result.n_eff_log_susceptibility,
        "binder_cumulant": result.binder_cumulant,
        "binder_cumulant_std": result.binder_cumulant_std,
        "n_eff_binder": result.n_eff_binder,
        "m2": result.m2,
        "m2_std": result.m2_std,
        "n_eff_m2": result.n_eff_m2,
        "m4": result.m4,
        "m4_std": result.m4_std,
        "n_eff_m4": result.n_eff_m4,
        "n_sweeps": result.n_sweeps,
        "n_thermalize": result.n_thermalize,
        "seed": result.seed,
    }


def _magnetization_per_spin(spins: np.ndarray) -> float:
    return float(spins.sum()) / spins.size


def _binder_from_moments(m2_mean: float, m4_mean: float) -> float:
    denom = 3.0 * m2_mean**2
    return 1.0 - m4_mean / max(denom, 1e-12)


def _jackknife_binder_se(
    m2_arr: np.ndarray,
    m4_arr: np.ndarray,
    *,
    n_blocks: int = 50,
) -> float:
    """Block-jackknife SE for ``U_4 = 1 - <m^4> / (3 <m^2>^2)``.

    ``U_4`` is a nonlinear function of chain averages, so its uncertainty is
    estimated by leave-one-block-out jackknife (not a delta-method expansion).
    """
    m2_arr = np.asarray(m2_arr, dtype=np.float64).ravel()
    m4_arr = np.asarray(m4_arr, dtype=np.float64).ravel()
    n = int(m2_arr.size)
    if n != int(m4_arr.size):
        raise ValueError("m2_arr and m4_arr must have the same length")
    if n < 2:
        return 0.0

    n_blocks = int(min(max(n_blocks, 2), n))
    usable = (n // n_blocks) * n_blocks
    if usable < 2 * n_blocks:
        # Too short to block: fall back to a single leave-one-draw-out pass on
        # a thinned subset so we still avoid analytic linearization.
        step = max(n // min(n, 200), 1)
        m2_arr = m2_arr[::step]
        m4_arr = m4_arr[::step]
        n = int(m2_arr.size)
        n_blocks = n
        usable = n

    block_m2 = m2_arr[:usable].reshape(n_blocks, -1).mean(axis=1)
    block_m4 = m4_arr[:usable].reshape(n_blocks, -1).mean(axis=1)
    tot_m2 = float(block_m2.sum())
    tot_m4 = float(block_m4.sum())
    denom = float(n_blocks - 1)
    jack = np.empty(n_blocks, dtype=np.float64)
    for k in range(n_blocks):
        m2_j = (tot_m2 - float(block_m2[k])) / denom
        m4_j = (tot_m4 - float(block_m4[k])) / denom
        jack[k] = _binder_from_moments(m2_j, m4_j)
    jack_mean = float(jack.mean())
    return float(np.sqrt(denom / float(n_blocks) * np.sum((jack - jack_mean) ** 2)))


def _wolff_step(spins: np.ndarray, p_add: float, rng: np.random.Generator) -> None:
    L = spins.shape[0]
    i, j = (int(rng.integers(0, L)), int(rng.integers(0, L)))
    seed_spin = spins[i, j]
    cluster: list[tuple[int, int]] = [(i, j)]
    visited = np.zeros((L, L), dtype=bool)
    visited[i, j] = True
    head = 0

    while head < len(cluster):
        x, y = cluster[head]
        head += 1
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nx = (x + dx) % L
            ny = (y + dy) % L
            if visited[nx, ny] or spins[nx, ny] != seed_spin:
                continue
            if rng.random() < p_add:
                visited[nx, ny] = True
                cluster.append((nx, ny))

    for x, y in cluster:
        spins[x, y] *= -1


def _sample_mean_std(samples: np.ndarray) -> tuple[float, float]:
    x = np.asarray(samples, dtype=np.float64).ravel()
    if x.size == 0:
        return 0.0, 0.0
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if x.size > 1 else 0.0
    return mean, std


def _blackjax_effective_sample_size(samples: np.ndarray) -> float:
    """ESS of a single MC chain via blackjax (shape (1, n_draws))."""
    x = np.asarray(samples, dtype=np.float64).ravel()
    if x.size < 2:
        return 1.0
    ess = effective_sample_size(x[None, :])
    return max(float(np.asarray(ess).item()), 1.0)


def summarize_signed_m_samples(
    m_signed: np.ndarray,
    *,
    L: int,
    T: float,
    n_thermalize: int,
    n_sweeps: int,
    seed: int,
) -> IsingRunResult:
    """Aggregate per-sweep signed ``m`` samples into observable means and errors."""
    m_signed_arr = np.asarray(m_signed, dtype=np.float64).ravel()
    if m_signed_arr.size < 1:
        raise ValueError("m_signed must contain at least one draw")

    beta = 1.0 / T
    m_arr = np.abs(m_signed_arr)
    m2_arr = m_signed_arr * m_signed_arr
    m4_arr = m_signed_arr**4

    m_mean, m_std = _sample_mean_std(m_arr)
    n_eff = _blackjax_effective_sample_size(m_arr)

    m2_mean, m2_std = _sample_mean_std(m2_arr)
    n_eff_m2 = _blackjax_effective_sample_size(m2_arr)

    m4_mean, m4_std = _sample_mean_std(m4_arr)
    n_eff_m4 = _blackjax_effective_sample_size(m4_arr)

    log_m_arr = np.log(np.maximum(m_arr, 1e-12))
    log_m_mean, log_m_std = _sample_mean_std(log_m_arr)
    n_eff_log = _blackjax_effective_sample_size(log_m_arr)

    u_mean = _binder_from_moments(m2_mean, m4_mean)
    # Store jackknife SE in the same (std, n_eff) convention as other channels:
    # sigma = std / sqrt(n_eff). Use moment ESS only as that scale factor.
    n_eff_binder = min(n_eff_m2, n_eff_m4)
    se_u = _jackknife_binder_se(m2_arr, m4_arr)
    u_std = se_u * float(np.sqrt(max(n_eff_binder, 1.0)))

    m_signed_mean = float(m_signed_arr.mean())
    chi_arr = beta * float(L**2) * (m_signed_arr - m_signed_mean) ** 2
    chi_mean, chi_std = _sample_mean_std(chi_arr)
    n_eff_chi = _blackjax_effective_sample_size(chi_arr)

    log_chi_arr = np.log(np.maximum(chi_arr, 1e-12))
    log_chi_mean, log_chi_std = _sample_mean_std(log_chi_arr)
    n_eff_log_chi = _blackjax_effective_sample_size(log_chi_arr)

    return IsingRunResult(
        L=L,
        T=T,
        magnetization=m_mean,
        magnetization_std=m_std,
        n_eff=n_eff,
        log_magnetization=log_m_mean,
        log_magnetization_std=log_m_std,
        n_eff_log=n_eff_log,
        susceptibility=chi_mean,
        susceptibility_std=chi_std,
        n_eff_susceptibility=n_eff_chi,
        log_susceptibility=log_chi_mean,
        log_susceptibility_std=log_chi_std,
        n_eff_log_susceptibility=n_eff_log_chi,
        binder_cumulant=u_mean,
        binder_cumulant_std=u_std,
        n_eff_binder=n_eff_binder,
        m2=m2_mean,
        m2_std=m2_std,
        n_eff_m2=n_eff_m2,
        m4=m4_mean,
        m4_std=m4_std,
        n_eff_m4=n_eff_m4,
        n_sweeps=n_sweeps,
        n_thermalize=n_thermalize,
        seed=seed,
    )


def run_ising(
    L: int,
    T: float,
    *,
    n_thermalize: int,
    n_sweeps: int,
    seed: int,
    sample_stride: int = 1,
    store_samples: bool = False,
    checkpoint_every: int | None = None,
    on_checkpoint: Callable[[int, np.ndarray], None] | None = None,
) -> IsingRunResult | tuple[IsingRunResult, IsingRawSamples]:
    """Run a Wolff-sampled 2D Ising chain and estimate <|m|>, χ, and U_4.

    Observables are plain Monte Carlo averages over the post-thermalize chain.
    ``*_std`` is the sample std (ddof=1) of the per-sweep series (or, for ``U_4``,
    a jackknife SE rescaled so ``sigma = std / sqrt(n_eff)``); ``n_eff`` is
    blackjax ESS. The fit uses ``sigma = *_std / sqrt(n_eff)``.
    """
    if L < 2:
        raise ValueError(f"L must be >= 2, got {L}")
    if T <= 0.0:
        raise ValueError(f"T must be positive, got {T}")
    if n_sweeps < 1:
        raise ValueError("n_sweeps must be >= 1")
    if checkpoint_every is not None and checkpoint_every < 1:
        raise ValueError(f"checkpoint_every must be >= 1, got {checkpoint_every}")

    rng = np.random.default_rng(seed)
    beta = 1.0 / T
    p_add = 1.0 - np.exp(-2.0 * beta)

    spins = rng.choice(np.array([-1, 1], dtype=np.int8), size=(L, L))

    for _ in range(n_thermalize):
        _wolff_step(spins, p_add, rng)

    m_signed_samples: list[float] = []
    for sweep in range(n_sweeps):
        _wolff_step(spins, p_add, rng)
        if (sweep + 1) % sample_stride == 0:
            m_signed_samples.append(_magnetization_per_spin(spins))
        if (
            on_checkpoint is not None
            and checkpoint_every is not None
            and (sweep + 1) % checkpoint_every == 0
            and m_signed_samples
        ):
            on_checkpoint(
                sweep + 1,
                np.asarray(m_signed_samples, dtype=np.float64),
            )

    m_signed_arr = np.asarray(m_signed_samples, dtype=np.float64)
    if on_checkpoint is not None and m_signed_samples:
        on_checkpoint(n_sweeps, m_signed_arr)
    result = summarize_signed_m_samples(
        m_signed_arr,
        L=L,
        T=T,
        n_thermalize=n_thermalize,
        n_sweeps=n_sweeps,
        seed=seed,
    )
    if not store_samples:
        return result

    return result, IsingRawSamples(
        L=L,
        T=T,
        m_signed=m_signed_arr,
        n_thermalize=n_thermalize,
        n_sweeps=n_sweeps,
        sample_stride=sample_stride,
        seed=seed,
    )
