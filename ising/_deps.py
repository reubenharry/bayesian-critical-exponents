"""Optional PyMC / ArviZ imports with a single install hint."""

from .pytensor_bootstrap import bootstrap_pytensor

bootstrap_pytensor()

try:
    import arviz as az
    import pymc as pm
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "ising requires pymc and arviz. Install with: "
        'pip install -e ".[dev]" or pip install pymc arviz'
    ) from e

__all__ = ["az", "pm"]
