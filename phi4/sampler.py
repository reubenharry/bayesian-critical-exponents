"""2D scalar φ⁴ sampler via blackjax HMC."""

from __future__ import annotations

from collections.abc import Callable

from ising.jax_config import configure_jax

configure_jax()

import jax
import jax.numpy as jnp
import numpy as np

from ising.sampler import (
    IsingRawSamples as Phi4RawSamples,
    IsingRunResult as Phi4RunResult,
    result_to_row,
    summarize_signed_m_samples,
)

from .model import make_logdensity_fn, magnetization_per_site

__all__ = [
    "Phi4RawSamples",
    "Phi4RunResult",
    "result_to_row",
    "summarize_signed_m_samples",
    "run_phi4",
    "run_phi4_until_ess",
]


def _magnetization_ess(m_signed: np.ndarray) -> float:
    from blackjax.diagnostics import effective_sample_size

    m_abs = np.abs(np.asarray(m_signed, dtype=np.float64).ravel())
    if m_abs.size < 2:
        return 1.0
    return max(float(np.asarray(effective_sample_size(m_abs[None, :])).item()), 1.0)


def run_phi4(
    L: int,
    lam: float,
    *,
    n_thermalize: int,
    n_sweeps: int,
    seed: int,
    sample_stride: int = 1,
    store_samples: bool = False,
    checkpoint_every: int | None = None,
    on_checkpoint: Callable[[int, np.ndarray], None] | None = None,
    num_integration_steps: int = 64,
    target_acceptance_rate: float = 0.8,
    progress_bar: bool = False,
) -> Phi4RunResult | tuple[Phi4RunResult, Phi4RawSamples]:
    """Run blackjax HMC on 2D φ⁴ at ``(L, lam)`` and estimate magnetization observables.

    ``n_thermalize`` is the window-adaptation warmup length; ``n_sweeps`` is the
    number of post-warmup HMC transitions. The CSV column ``T`` stores ``lam``.

    Observables match the Ising pipeline (``<|m|>``, χ, Binder, m², m⁴) so FSS
    code can load the resulting CSV unchanged. Susceptibility uses the Ising
    summarizer convention ``χ ∝ (1/T) L² Var(m)`` with ``T = lam``.
    """
    output = run_phi4_until_ess(
        L,
        lam,
        min_ess_m=0.0,
        n_thermalize=n_thermalize,
        n_sweeps_min=n_sweeps,
        n_sweeps_max=n_sweeps,
        chunk_sweeps=n_sweeps,
        seed=seed,
        sample_stride=sample_stride,
        store_samples=store_samples,
        checkpoint_every=checkpoint_every,
        on_checkpoint=on_checkpoint,
        num_integration_steps=num_integration_steps,
        target_acceptance_rate=target_acceptance_rate,
        progress_bar=progress_bar,
    )
    return output


def run_phi4_until_ess(
    L: int,
    lam: float,
    *,
    min_ess_m: float = 100.0,
    n_thermalize: int = 500,
    n_sweeps_min: int = 2000,
    n_sweeps_max: int = 120_000,
    chunk_sweeps: int = 4000,
    seed: int = 0,
    sample_stride: int = 1,
    store_samples: bool = False,
    checkpoint_every: int | None = None,
    on_checkpoint: Callable[[int, np.ndarray], None] | None = None,
    num_integration_steps: int = 64,
    target_acceptance_rate: float = 0.8,
    progress_bar: bool = False,
) -> Phi4RunResult | tuple[Phi4RunResult, Phi4RawSamples]:
    """Adapt HMC once, then sample in chunks until ``n_eff(|m|) >= min_ess_m``.

    Always collects at least ``n_sweeps_min`` post-warmup transitions and stops
    by ``n_sweeps_max`` even if the ESS target is not met.
    """
    import blackjax

    if L < 2:
        raise ValueError(f"L must be >= 2, got {L}")
    if lam <= 0.0:
        raise ValueError(f"lam must be positive, got {lam}")
    if n_thermalize < 1:
        raise ValueError("n_thermalize must be >= 1 for HMC adaptation")
    if n_sweeps_min < 1:
        raise ValueError("n_sweeps_min must be >= 1")
    if n_sweeps_max < n_sweeps_min:
        raise ValueError("n_sweeps_max must be >= n_sweeps_min")
    if chunk_sweeps < 1:
        raise ValueError("chunk_sweeps must be >= 1")
    if sample_stride < 1:
        raise ValueError(f"sample_stride must be >= 1, got {sample_stride}")
    if checkpoint_every is not None and checkpoint_every < 1:
        raise ValueError(f"checkpoint_every must be >= 1, got {checkpoint_every}")
    if num_integration_steps < 1:
        raise ValueError(
            f"num_integration_steps must be >= 1, got {num_integration_steps}"
        )

    ndims = L * L
    logdensity = make_logdensity_fn(L, lam)
    key = jax.random.key(seed)
    init_key, adapt_key, sample_key = jax.random.split(key, 3)
    # Broken-symmetry-friendly init helps ordered-phase mixing of |m|.
    position = 0.3 * jnp.ones((ndims,), dtype=jnp.float64)
    position = position + 0.05 * jax.random.normal(
        init_key, shape=(ndims,), dtype=jnp.float64
    )

    warmup = blackjax.window_adaptation(
        blackjax.hmc,
        logdensity,
        target_acceptance_rate=target_acceptance_rate,
        progress_bar=progress_bar,
        num_integration_steps=num_integration_steps,
    )
    adapt_result, _ = warmup.run(adapt_key, position, num_steps=n_thermalize)
    hmc = blackjax.hmc(logdensity, **adapt_result.parameters)

    m_values: list[float] = []
    state = adapt_result.state
    sweeps_done = 0
    target_ess = float(min_ess_m)

    while sweeps_done < n_sweeps_max:
        remaining_for_min = max(n_sweeps_min - sweeps_done, 0)
        if sweeps_done >= n_sweeps_min and m_values:
            ess_now = _magnetization_ess(np.asarray(m_values, dtype=np.float64))
            if ess_now >= target_ess:
                break
        n_step = min(chunk_sweeps, n_sweeps_max - sweeps_done)
        if remaining_for_min > 0:
            n_step = max(n_step, min(remaining_for_min, chunk_sweeps))
        n_step = min(n_step, n_sweeps_max - sweeps_done)
        if checkpoint_every is not None:
            n_step = min(n_step, checkpoint_every)

        sample_key, subkey = jax.random.split(sample_key)

        def _one_step(carry, _):
            st, rng = carry
            rng, step_key = jax.random.split(rng)
            st_new, info = hmc.step(step_key, st)
            del info
            m = magnetization_per_site(st_new.position, L)
            return (st_new, rng), m

        (state, _), m_traj = jax.lax.scan(
            _one_step, (state, subkey), xs=None, length=int(n_step)
        )
        m_np = np.asarray(m_traj, dtype=np.float64)
        for i, m_val in enumerate(m_np):
            sweep_idx = sweeps_done + i + 1
            if sweep_idx % sample_stride == 0:
                m_values.append(float(m_val))
        sweeps_done += int(n_step)
        if on_checkpoint is not None and m_values:
            on_checkpoint(sweeps_done, np.asarray(m_values, dtype=np.float64))

    if not m_values:
        raise RuntimeError("HMC produced no magnetization samples")
    m_signed_arr = np.asarray(m_values, dtype=np.float64)
    if on_checkpoint is not None:
        on_checkpoint(sweeps_done, m_signed_arr)

    result = summarize_signed_m_samples(
        m_signed_arr,
        L=L,
        T=float(lam),
        n_thermalize=n_thermalize,
        n_sweeps=sweeps_done,
        seed=seed,
    )
    if min_ess_m > 0 and result.n_eff + 1e-9 < min_ess_m:
        print(
            f"  WARNING L={L} lam={lam:g}: n_eff(m)={result.n_eff:.1f} "
            f"< target {min_ess_m:g} after {sweeps_done} sweeps",
            flush=True,
        )

    if not store_samples:
        return result

    return result, Phi4RawSamples(
        L=L,
        T=float(lam),
        m_signed=m_signed_arr,
        n_thermalize=n_thermalize,
        n_sweeps=sweeps_done,
        sample_stride=sample_stride,
        seed=seed,
    )
