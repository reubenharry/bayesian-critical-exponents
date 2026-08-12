"""Bayesian finite-size scaling on the 2D Ising model."""

from .pytensor_bootstrap import bootstrap_pytensor

bootstrap_pytensor()

from .constants import BETA_EXACT, GAMMA_EXACT, NU_EXACT, TC_EXACT

__all__ = ["TC_EXACT", "NU_EXACT", "BETA_EXACT", "GAMMA_EXACT"]
