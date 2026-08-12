"""Exact critical quantities for the 2D square-lattice Ising model (J=1, k_B=1)."""

from __future__ import annotations

import numpy as np

TC_EXACT = float(2.0 / np.log(1.0 + np.sqrt(2.0)))
NU_EXACT = 1.0
BETA_EXACT = 0.125
GAMMA_EXACT = 1.75
# Spatial dimension for hyperscaling relations (2D square lattice).
SPATIAL_DIMENSION = 2
# Leading correction-to-scaling exponent (y / nu) for 2D Ising, y = 2.
OMEGA_EXACT = 2.0
OMEGA_PRIOR_LOWER = 0.5
OMEGA_PRIOR_UPPER = 4.0


def gamma_from_nu_beta(nu: float, beta: float, *, d: int = SPATIAL_DIMENSION) -> float:
    """γ from Rushbrooke + Josephson hyperscaling: γ = dν − 2β."""
    return float(d * nu - 2.0 * beta)
