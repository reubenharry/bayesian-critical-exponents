"""GP kernels for FSS scaling functions.

Harada (Phys. Rev. E **84**, 056704, 2011) uses the squared-exponential kernel

    k(x, x') = eta^2 exp(-|x - x'|^2 / (2 ell^2))

on the collapsed scaling variable. The default in this codebase is Matérn-5/2.

The ``poly2`` / ``poly4`` kernels are monomial inner products through degree 2 / 4,

    k(x, x') = eta^2 sum_{k=0}^d (x x')^k,

equivalent to marginalizing polynomial coefficients c_k in f(x) = sum_k c_k x^k
with c ~ N(0, eta^2 I).
"""

from __future__ import annotations

from typing import Literal

GpKernelKind = Literal["matern52", "gaussian", "poly2", "poly4"]
DEFAULT_GP_KERNEL: GpKernelKind = "gaussian"
DEFAULT_UNIVERSAL_KERNEL: GpKernelKind | None = None
POLY2_DEGREE = 2
POLY4_DEGREE = 4
POLYNOMIAL_KERNEL_DEGREES: dict[GpKernelKind, int] = {
    "poly2": POLY2_DEGREE,
    "poly4": POLY4_DEGREE,
}

_GP_KERNEL_ALIASES: dict[str, GpKernelKind] = {
    "matern52": "matern52",
    "matern5/2": "matern52",
    "matern": "matern52",
    "gaussian": "gaussian",
    "harada": "gaussian",
    "rbf": "gaussian",
    "squaredexponential": "gaussian",
    "squared_exponential": "gaussian",
    "poly2": "poly2",
    "poly": "poly2",
    "polynomial": "poly2",
    "quadratic": "poly2",
    "poly4": "poly4",
    "quartic": "poly4",
}


def normalize_gp_kernel(kind: str) -> GpKernelKind:
    """Normalize CLI / config kernel names (``harada`` -> ``gaussian``)."""
    key = kind.strip().lower().replace("-", "").replace(" ", "").replace("_", "")
    if key in _GP_KERNEL_ALIASES:
        return _GP_KERNEL_ALIASES[key]
    raise ValueError(
        f"gp_kernel must be one of {sorted(set(_GP_KERNEL_ALIASES.values()))}, "
        f"got {kind!r}"
    )


def is_polynomial_universal_kernel(kind: GpKernelKind | str) -> bool:
    """Return True for marginalized monomial universal kernels (poly2, poly4, ...)."""
    resolved = normalize_gp_kernel(kind) if isinstance(kind, str) else kind
    return resolved in POLYNOMIAL_KERNEL_DEGREES


def polynomial_kernel_degree(kind: GpKernelKind | str) -> int | None:
    resolved = normalize_gp_kernel(kind) if isinstance(kind, str) else kind
    return POLYNOMIAL_KERNEL_DEGREES.get(resolved)


def is_stationary_gp_kernel(kind: GpKernelKind | str) -> bool:
    """Return True for translation-invariant Matérn / Gaussian kernels."""
    resolved = normalize_gp_kernel(kind) if isinstance(kind, str) else kind
    return resolved in ("matern52", "gaussian")


def config_universal_kernel(config) -> GpKernelKind:
    """Kernel for the universal scaling function f_0(z).

    Falls back to ``gp_kernel`` when ``universal_kernel`` is unset.
    """
    universal = getattr(config, "universal_kernel", DEFAULT_UNIVERSAL_KERNEL)
    if universal is None:
        return normalize_gp_kernel(getattr(config, "gp_kernel", DEFAULT_GP_KERNEL))
    return normalize_gp_kernel(universal) if isinstance(universal, str) else universal
