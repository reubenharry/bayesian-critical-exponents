"""Exact thermodynamic quantities for the 2D square-lattice Ising model."""

from __future__ import annotations

import numpy as np

from .constants import BETA_EXACT, TC_EXACT


def magnetization_infinite_L(T: np.ndarray | float) -> np.ndarray:
    """Exact spontaneous magnetization per spin in the L -> infinity limit (Onsager/Yang).

    For J = k_B = 1:
        |m|(T) = (1 - sinh(2/T)^{-4})^{beta}   for T < T_c
        |m|(T) = 0                               for T >= T_c
    """
    T_arr = np.asarray(T, dtype=np.float64)
    out = np.zeros_like(T_arr)
    below = T_arr < TC_EXACT
    inner = 1.0 - np.sinh(2.0 / T_arr[below]) ** (-4)
    out[below] = np.maximum(inner, 0.0) ** BETA_EXACT
    return out
