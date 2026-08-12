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

In the limit t -> 0 and L -> inf, a -> 0 and the usual FSS + MC noise is recovered.
"""

from __future__ import annotations

import numpy as np

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
