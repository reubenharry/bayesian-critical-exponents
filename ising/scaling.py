"""Scaling-variable helpers for FSS inference."""

from __future__ import annotations

import numpy as np
import pytensor.tensor as pt

from .constants import BETA_EXACT, NU_EXACT, TC_EXACT

Z_EPS = 1e-8


def t_from_T(
    T: np.ndarray,
    *,
    T_c: float = TC_EXACT,
) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    return (T - T_c) / T_c


def z_from_LT(
    L: np.ndarray,
    T: np.ndarray,
    *,
    T_c: float = TC_EXACT,
    nu: float = NU_EXACT,
) -> np.ndarray:
    L = np.asarray(L, dtype=np.float64)
    return t_from_T(T, T_c=T_c) * L ** (1.0 / nu)


def log_phi_from_log_m(
    log_m: np.ndarray,
    L: np.ndarray,
    *,
    beta: float = BETA_EXACT,
    nu: float = NU_EXACT,
) -> np.ndarray:
    L = np.asarray(L, dtype=np.float64)
    return np.asarray(log_m, dtype=np.float64) + (beta / nu) * np.log(L)

