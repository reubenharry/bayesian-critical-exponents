"""Scaling-window discrepancy amplitude a(t, L).

Two observation models share the same amplitude::

    a(t, L) = (|t|/t0)^p + (L0/L)^q
    t = (T - T_c) / T_c

**Noise inflation** (default ``discrepancy_form="noise"``)::

    eps_i ~ Normal(0, sigma_MC_i^2 + sigma_model^2 * a(t_i, L_i)^2)

**Additive GP** (``discrepancy_form="additive_gp"``)::

    On the *raw* observable (slide form, Φ₁ dropped, r = g)::

        m(L, t) = L^{-β/ν} f(z) + a(t, L) g(z) + ε_m

    with independent GPs ``f, g`` on the collapse coordinate ``z``.  The
    likelihood is evaluated on the collapsed field
    ``Φ = m L^{β/ν}``, which rearranges to::

        Φ = f(z) + [a(t, L) L^{β/ν}] g(z) + ε_Φ

    so ``a`` itself is *not* part of the scaling function — only the
    product ``a L^{β/ν}`` multiplies ``g`` in Φ-space.  Analogous
    ``L``-powers apply to other channels (``L^{2β/ν}`` for ``m²``, etc.).

**Z-threshold additive GP** (``discrepancy_form="additive_gp_z_threshold"``)::

    Same additive GP marginal likelihood, but the collapsed amplitude is the
    indicator ``a = 1_{|z| > z_disc_threshold}`` (default threshold 10), with
    no ``L^{β/ν}`` rescaling.  Points inside the gate have ``a=0`` (plain GP);
    outside, ``a=1`` so large ``σ_g`` can absorb them.

**FSS additive GP** (``discrepancy_form="additive_gp_fss"``)::

    Collapsed-space amplitude (no ``L^{β/ν}`` rescaling)::

        a = L^{-ω} + κ |t|^{ω ν} = L^{-ω} (1 + κ |z|^{ω ν})

    so Φ = f(z) + a g(z) + ε.  This is the two-term window with the FSS
    identifications q = ω and p = ω ν; overall scale stays in σ_g.

    Interactive / ``evaluate_with_gp`` slot mapping: ``disc_q = ω``,
    ``disc_t0 = κ`` (``disc_L0`` and ``disc_p`` unused).  Inferred ``ω``
    uses Uniform(1, 10).

**Coupling** (``discrepancy_coupling``, GP forms only) controls how the
amplitude enters the collapsed-space GP.  Write ``π`` for a unit-interval
gate: ``π = a/(1+a)`` when ``a`` can exceed 1, and ``π = a`` when ``a`` is
already a 0/1 indicator (z-threshold).

- **additive** (default): ``Φ = f + a g + ε`` (current).
- **mixture**: ``Φ = (1-π) f + π g + ε``, so out-of-window points do not
  train ``f``.
- **weighted**: ``Φ = (1-π) f + ε`` (no second GP).  Equivalent to
  ``K_ij = (1-π_i)(1-π_j) k_f + σ_i² δ_ij``.

In the limit t -> 0 and L -> inf, a -> 0 and the usual FSS + MC noise is recovered.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

DiscrepancyCoupling = Literal["additive", "mixture", "weighted"]
DISCREPANCY_COUPLING: DiscrepancyCoupling = "additive"

# Prior / default ranges for inferred a-parameters (v1).
DISCREPANCY_T0_PRIOR_LOWER = 0.01
DISCREPANCY_T0_PRIOR_UPPER = 10.0
DISCREPANCY_L0_PRIOR_LOWER = 0.1
DISCREPANCY_L0_PRIOR_UPPER = 64.0
DISCREPANCY_P_PRIOR_LOWER = 0.5
DISCREPANCY_P_PRIOR_UPPER = 4.0
DISCREPANCY_Q_PRIOR_LOWER = 0.5
DISCREPANCY_Q_PRIOR_UPPER = 4.0
DISCREPANCY_SIGMA_MODEL_PRIOR_SIGMA = 1.0

# Init / reference values (order-1 near Harada window edge / small-L extras).
DISCREPANCY_T0_INIT = 0.1
DISCREPANCY_L0_INIT = 32.0
DISCREPANCY_P_INIT = 2.0
DISCREPANCY_Q_INIT = 2.0
DISCREPANCY_SIGMA_MODEL_INIT = 0.1

# additive_gp_fss: a = L^{-ω} + κ |t|^{ω ν} (collapsed Φ-space).
DISCREPANCY_KAPPA_INIT = 1.0
DISCREPANCY_KAPPA_PRIOR_LOWER = 0.0
DISCREPANCY_KAPPA_PRIOR_UPPER = 20.0
DISCREPANCY_OMEGA_INIT = 2.0
DISCREPANCY_OMEGA_PRIOR_LOWER = 1.0
DISCREPANCY_OMEGA_PRIOR_UPPER = 10.0


def discrepancy_q_prior_bounds(discrepancy_form: str) -> tuple[float, float]:
    """Prior interval for ``disc_q`` (ω under ``additive_gp_fss``)."""
    if discrepancy_form == "additive_gp_fss":
        return DISCREPANCY_OMEGA_PRIOR_LOWER, DISCREPANCY_OMEGA_PRIOR_UPPER
    return DISCREPANCY_Q_PRIOR_LOWER, DISCREPANCY_Q_PRIOR_UPPER


def discrepancy_amplitude(
    t: np.ndarray | float,
    L: np.ndarray | float,
    *,
    t0: float,
    L0: float,
    p: float,
    q: float,
) -> np.ndarray:
    """Return a(t, L) = (|t|/t0)^p + (L0/L)^q (broadcasts over arrays)."""
    t_arr = np.asarray(t, dtype=np.float64)
    L_arr = np.asarray(L, dtype=np.float64)
    t0_f = max(float(t0), 1e-12)
    L0_f = max(float(L0), 1e-12)
    return (np.abs(t_arr) / t0_f) ** float(p) + (L0_f / np.maximum(L_arr, 1e-12)) ** float(q)


def discrepancy_mix_weight(
    a: np.ndarray | float,
    *,
    squash: bool = True,
) -> np.ndarray:
    """Map amplitude ``a ≥ 0`` to a gate ``π ∈ [0, 1]``.

    ``squash=True`` uses ``π = a / (1 + a)``.  ``squash=False`` clips into
    ``[0, 1]`` (already-unit gates such as ``1_{|z|>threshold}``).
    """
    a_arr = np.maximum(np.asarray(a, dtype=np.float64), 0.0)
    if squash:
        return a_arr / (1.0 + a_arr)
    return np.clip(a_arr, 0.0, 1.0)


def discrepancy_amplitude_fss(
    t: np.ndarray | float,
    L: np.ndarray | float,
    *,
    omega: float,
    kappa: float,
    nu: float,
) -> np.ndarray:
    """Collapsed-space a = L^{-ω} + κ |t|^{ω ν} (broadcasts over arrays)."""
    t_arr = np.asarray(t, dtype=np.float64)
    L_arr = np.maximum(np.asarray(L, dtype=np.float64), 1e-12)
    omega_f = float(omega)
    kappa_f = float(kappa)
    nu_f = max(float(nu), 1e-12)
    return L_arr ** (-omega_f) + kappa_f * np.abs(t_arr) ** (omega_f * nu_f)


def effective_obs_sigma(
    sigma_mc: np.ndarray | float,
    t: np.ndarray | float,
    L: np.ndarray | float,
    *,
    t0: float,
    L0: float,
    p: float,
    q: float,
    sigma_model: float,
) -> np.ndarray:
    """sigma_eff = sqrt(sigma_MC^2 + (sigma_model * a(t,L))^2)."""
    sigma = np.asarray(sigma_mc, dtype=np.float64)
    a = discrepancy_amplitude(t, L, t0=t0, L0=L0, p=p, q=q)
    return np.sqrt(sigma**2 + (float(sigma_model) * a) ** 2)


def point_weight(
    sigma_mc: np.ndarray | float,
    t: np.ndarray | float,
    L: np.ndarray | float,
    *,
    t0: float,
    L0: float,
    p: float,
    q: float,
    sigma_model: float,
) -> np.ndarray:
    """MC-noise fraction of total variance: sigma_MC^2 / sigma_eff^2 in (0, 1]."""
    sigma = np.asarray(sigma_mc, dtype=np.float64)
    sigma_eff = effective_obs_sigma(
        sigma, t, L, t0=t0, L0=L0, p=p, q=q, sigma_model=sigma_model
    )
    return (sigma**2) / np.maximum(sigma_eff**2, 1e-30)
