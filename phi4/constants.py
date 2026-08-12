"""Critical quantities for 2D lattice φ⁴ (Ising universality; distinct λ_c).

Exponents ν, β, γ match the 2D Ising model. The critical *control parameter*
is the quartic coupling λ_c (with m² = −4 fixed in the action), not Ising T_c.
"""

from __future__ import annotations

# Quartic coupling at criticality for the sampler-benchmarks action convention
# (m² = −4 fixed). See Fig. 3 of https://arxiv.org/pdf/2207.00283.pdf and
# unreduce_lam(reduced_lam=0, side=L) → 4.25.
LAM_C = 4.25

# Alias for FSS drop-in: inferred "T_c" is λ_c, not Ising temperature ≈ 2.269.
TC_EXACT = LAM_C

NU_EXACT = 1.0
BETA_EXACT = 0.125
GAMMA_EXACT = 1.75
SPATIAL_DIMENSION = 2
OMEGA_EXACT = 2.0


def gamma_from_nu_beta(nu: float, beta: float, *, d: int = SPATIAL_DIMENSION) -> float:
    """γ from Rushbrooke + Josephson hyperscaling: γ = dν − 2β."""
    return float(d * nu - 2.0 * beta)


def unreduce_lam(reduced_lam: float, side: int) -> float:
    """Map reduced λ to physical λ at lattice size ``side``.

    See Fig. 3 in https://arxiv.org/pdf/2207.00283.pdf:
    ``lam = 4.25 * (reduced_lam * side^{-1} + 1)``.
    """
    return float(LAM_C * (float(reduced_lam) * float(side) ** (-1.0) + 1.0))
