"""Stationary GP kernels for FSS scaling functions.

Harada (Phys. Rev. E **84**, 056704, 2011) uses the squared-exponential kernel

    k(x, x') = eta^2 exp(-|x - x'|^2 / (2 ell^2))

on the collapsed scaling variable. The default in this codebase is Matérn-5/2.
"""

from __future__ import annotations

from typing import Literal

GpKernelKind = Literal["matern52", "gaussian"]
DEFAULT_GP_KERNEL: GpKernelKind = "gaussian"

_GP_KERNEL_ALIASES: dict[str, GpKernelKind] = {
    "matern52": "matern52",
    "matern5/2": "matern52",
    "matern": "matern52",
    "gaussian": "gaussian",
    "harada": "gaussian",
    "rbf": "gaussian",
    "squaredexponential": "gaussian",
    "squared_exponential": "gaussian",
}


def normalize_gp_kernel(kind: str) -> GpKernelKind:
    """Normalize CLI / config kernel names (``harada`` -> ``gaussian``)."""
    key = kind.strip().lower().replace("-", "").replace(" ", "")
    if key in _GP_KERNEL_ALIASES:
        return _GP_KERNEL_ALIASES[key]
    raise ValueError(
        f"gp_kernel must be one of {sorted(set(_GP_KERNEL_ALIASES.values()))}, "
        f"got {kind!r}"
    )
