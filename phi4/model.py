"""2D scalar φ⁴ log-density (vendored from sampler-benchmarks phi4.py).

Action (m² = −4 encoded in nearest-neighbor hopping; scan quartic ``lam``)::

    S[φ] = Σ_x [ λ φ_x⁴ − φ_x Σ_{μ=±ê} φ_{x+μ} ]

Target density ∝ exp(−S).
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp


def phi4_logdensity(x: jnp.ndarray, L: int, lam: float) -> jnp.ndarray:
    """Log-density of the 2D lattice φ⁴ theory at coupling ``lam``."""
    phi = x.reshape(L, L)
    action_density = lam * jnp.power(phi, 4) - phi * (
        jnp.roll(phi, -1, 0)
        + jnp.roll(phi, 1, 0)
        + jnp.roll(phi, -1, 1)
        + jnp.roll(phi, 1, 1)
    )
    return -jnp.sum(action_density)


def make_logdensity_fn(L: int, lam: float) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return a closed-over ``logdensity(x)`` for blackjax."""

    def logdensity(x: jnp.ndarray) -> jnp.ndarray:
        return phi4_logdensity(x, L, lam)

    return logdensity


def magnetization_per_site(x: jnp.ndarray, L: int) -> jnp.ndarray:
    """Signed magnetization per site ``m = L^{-2} Σ φ``."""
    return jnp.mean(x.reshape(L, L))
