"""JAX log-posterior and Blackjax sampling (NUTS, LAPS) for the FSS GP model."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from ._deps import az
from .constants import BETA_EXACT, NU_EXACT, TC_EXACT
from .fss_likelihood import (
    compile_fss_log_marginal_likelihood_jax,
    extract_likelihood_arrays,
    fss_gp_scales,
    uses_fss_correction,
    uses_fss_discrepancy,
)
from .discrepancy import (
    DISCREPANCY_L0_INIT,
    DISCREPANCY_L0_PRIOR_LOWER,
    DISCREPANCY_L0_PRIOR_UPPER,
    DISCREPANCY_P_INIT,
    DISCREPANCY_P_PRIOR_LOWER,
    DISCREPANCY_P_PRIOR_UPPER,
    DISCREPANCY_Q_INIT,
    DISCREPANCY_Q_PRIOR_LOWER,
    DISCREPANCY_Q_PRIOR_UPPER,
    DISCREPANCY_SIGMA_MODEL_INIT,
    DISCREPANCY_SIGMA_MODEL_PRIOR_SIGMA,
    DISCREPANCY_T0_INIT,
    DISCREPANCY_T0_PRIOR_LOWER,
    DISCREPANCY_T0_PRIOR_UPPER,
)
from .jax_config import configure_jax, jax_device_summary
from .model_fss import (
    BETA_PRIOR_LOWER,
    BETA_PRIOR_UPPER,
    NU_PRIOR_LOWER,
    NU_PRIOR_UPPER,
    TC_DELTA_SCALE,
    beta_affects_fss_likelihood,
)
from .model_scaling_nu import TC_PRIOR_LOWER, TC_PRIOR_UPPER
from .profile_likelihood import FssProfileConfig

LAPS_DEFAULT_CHAINS = 200

_LAPS_INIT_JITTER = {
    "T_c": 2.0e-4,
    "nu": 0.02,
    "beta": 0.003,
    "gp_ell": 0.05,
    "gp_eta": 0.05,
    "correction_gp_ell": 0.05,
    "correction_gp_eta": 0.05,
    "disc_t0": 0.05,
    "disc_L0": 2.0,
    "disc_p": 0.1,
    "disc_q": 0.1,
    "disc_sigma_model": 0.05,
}

DEFAULT_GP_ELL_PRIOR_SIGMA = 0.35


@dataclass(frozen=True)
class FssJaxUniformSlot:
    """One inferred scalar with uniform prior on a bounded interval."""

    name: str
    lower: float
    upper: float

    def to_unconstrained(self, value: float) -> float:
        p = (float(value) - self.lower) / (self.upper - self.lower)
        p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
        return float(np.log(p / (1.0 - p)))

    def to_unconstrained_jax(self, value: jnp.ndarray) -> jnp.ndarray:
        p = (value - self.lower) / (self.upper - self.lower)
        p = jnp.clip(p, 1e-12, 1.0 - 1e-12)
        return jnp.log(p / (1.0 - p))

    def from_unconstrained(self, u: jnp.ndarray) -> jnp.ndarray:
        return self.lower + (self.upper - self.lower) * jax.nn.sigmoid(u)

    def log_prior_unconstrained(self, u: jnp.ndarray) -> jnp.ndarray:
        return -jax.nn.softplus(-u) - jax.nn.softplus(u)


@dataclass(frozen=True)
class FssJaxLognormalSlot:
    """Positive scalar with LogNormal prior (standard-normal reparametrization)."""

    name: str
    mu: float
    sigma: float

    def to_unconstrained(self, value: float) -> float:
        value = max(float(value), 1e-12)
        return float((np.log(value) - self.mu) / self.sigma)

    def to_unconstrained_jax(self, value: jnp.ndarray) -> jnp.ndarray:
        value = jnp.maximum(value, 1e-12)
        return (jnp.log(value) - self.mu) / self.sigma

    def from_unconstrained(self, u: jnp.ndarray) -> jnp.ndarray:
        return jnp.exp(self.mu + self.sigma * u)

    def log_prior_unconstrained(self, u: jnp.ndarray) -> jnp.ndarray:
        return -0.5 * u**2 - jnp.log(self.sigma * jnp.sqrt(2.0 * jnp.pi))


@dataclass(frozen=True)
class FssJaxHalfNormalSlot:
    """Positive scalar with HalfNormal(scale=sigma) prior via softplus map."""

    name: str
    sigma: float

    def to_unconstrained(self, value: float) -> float:
        x = max(float(value) / self.sigma, 1e-12)
        return float(np.log(np.expm1(x)))

    def to_unconstrained_jax(self, value: jnp.ndarray) -> jnp.ndarray:
        x = jnp.maximum(value / self.sigma, 1e-12)
        return jnp.log(jnp.expm1(x))

    def from_unconstrained(self, u: jnp.ndarray) -> jnp.ndarray:
        return self.sigma * jax.nn.softplus(u)

    def log_prior_unconstrained(self, u: jnp.ndarray) -> jnp.ndarray:
        x = self.sigma * jax.nn.softplus(u)
        log_p_x = jnp.log(jnp.sqrt(2.0) / (self.sigma * jnp.sqrt(jnp.pi))) - 0.5 * (
            x / self.sigma
        ) ** 2
        log_jac = jnp.log(self.sigma) - jax.nn.softplus(-u)
        return log_p_x + log_jac


FssJaxInferenceSlot = FssJaxUniformSlot | FssJaxLognormalSlot | FssJaxHalfNormalSlot

# Backward-compatible alias: exponent slots are uniform-bounded.
FssJaxParamSlot = FssJaxUniformSlot


@dataclass(frozen=True)
class FssJaxInferenceLayout:
    """Maps between flat unconstrained vectors and physical model parameters."""

    slots: tuple[FssJaxInferenceSlot, ...]
    infer_Tc: bool
    infer_nu: bool
    infer_beta: bool
    infer_gp_hyperparams: bool
    uses_correction: bool
    uses_discrepancy: bool
    reparametrize_Tc: bool
    tc_reparam_ref: float
    tc_delta_scale: float
    fixed_Tc: float
    fixed_nu: float
    fixed_beta: float
    fixed_gp_ell: float
    fixed_gp_eta: float
    fixed_correction_gp_ell: float
    fixed_correction_gp_eta: float
    fixed_disc_t0: float = DISCREPANCY_T0_INIT
    fixed_disc_L0: float = DISCREPANCY_L0_INIT
    fixed_disc_p: float = DISCREPANCY_P_INIT
    fixed_disc_q: float = DISCREPANCY_Q_INIT
    fixed_disc_sigma_model: float = DISCREPANCY_SIGMA_MODEL_INIT
    infer_disc_t0: bool = False
    infer_disc_L0: bool = False
    infer_disc_p: bool = False
    infer_disc_q: bool = False
    infer_disc_sigma_model: bool = False

    @property
    def ndim(self) -> int:
        return len(self.slots)

    def unpack_all(self, unconstrained: jnp.ndarray) -> dict[str, jnp.ndarray]:
        return {
            slot.name: slot.from_unconstrained(unconstrained[i])
            for i, slot in enumerate(self.slots)
        }

    def unpack_physical(
        self, unconstrained: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        values = self.unpack_all(unconstrained)

        if self.infer_Tc:
            if self.reparametrize_Tc:
                tc = self.tc_reparam_ref + values["T_c_delta"] / self.tc_delta_scale
            else:
                tc = values["T_c"]
        else:
            tc = jnp.asarray(self.fixed_Tc, dtype=jnp.float64)

        nu = values["nu"] if self.infer_nu else jnp.asarray(self.fixed_nu, dtype=jnp.float64)
        beta = (
            values["beta"]
            if self.infer_beta
            else jnp.asarray(self.fixed_beta, dtype=jnp.float64)
        )
        return tc, nu, beta

    def unpack_gp_hyperparams(
        self, unconstrained: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        values = self.unpack_all(unconstrained)
        gp_ell = values.get(
            "gp_ell", jnp.asarray(self.fixed_gp_ell, dtype=jnp.float64)
        )
        gp_eta = values.get(
            "gp_eta", jnp.asarray(self.fixed_gp_eta, dtype=jnp.float64)
        )
        correction_gp_ell = values.get(
            "correction_gp_ell",
            jnp.asarray(self.fixed_correction_gp_ell, dtype=jnp.float64),
        )
        correction_gp_eta = values.get(
            "correction_gp_eta",
            jnp.asarray(self.fixed_correction_gp_eta, dtype=jnp.float64),
        )
        return gp_ell, gp_eta, correction_gp_ell, correction_gp_eta

    def unpack_discrepancy(
        self, unconstrained: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        values = self.unpack_all(unconstrained)
        return (
            values.get("disc_t0", jnp.asarray(self.fixed_disc_t0, dtype=jnp.float64)),
            values.get("disc_L0", jnp.asarray(self.fixed_disc_L0, dtype=jnp.float64)),
            values.get("disc_p", jnp.asarray(self.fixed_disc_p, dtype=jnp.float64)),
            values.get("disc_q", jnp.asarray(self.fixed_disc_q, dtype=jnp.float64)),
            values.get(
                "disc_sigma_model",
                jnp.asarray(self.fixed_disc_sigma_model, dtype=jnp.float64),
            ),
        )

    def log_prior(self, unconstrained: jnp.ndarray) -> jnp.ndarray:
        if self.ndim == 0:
            return jnp.array(0.0, dtype=jnp.float64)
        terms = [
            slot.log_prior_unconstrained(unconstrained[i]) for i, slot in enumerate(self.slots)
        ]
        return jnp.sum(jnp.stack(terms))

    def _initial_value_map(
        self,
        *,
        T_c: float,
        nu: float,
        beta: float,
        gp_ell: float,
        gp_eta: float,
        correction_gp_ell: float,
        correction_gp_eta: float,
        disc_t0: float | None = None,
        disc_L0: float | None = None,
        disc_p: float | None = None,
        disc_q: float | None = None,
        disc_sigma_model: float | None = None,
    ) -> dict[str, float]:
        return {
            "T_c": float(T_c),
            "nu": float(nu),
            "beta": float(beta),
            "T_c_delta": (float(T_c) - self.tc_reparam_ref) * self.tc_delta_scale,
            "gp_ell": float(gp_ell),
            "gp_eta": float(gp_eta),
            "correction_gp_ell": float(correction_gp_ell),
            "correction_gp_eta": float(correction_gp_eta),
            "disc_t0": float(self.fixed_disc_t0 if disc_t0 is None else disc_t0),
            "disc_L0": float(self.fixed_disc_L0 if disc_L0 is None else disc_L0),
            "disc_p": float(self.fixed_disc_p if disc_p is None else disc_p),
            "disc_q": float(self.fixed_disc_q if disc_q is None else disc_q),
            "disc_sigma_model": float(
                self.fixed_disc_sigma_model
                if disc_sigma_model is None
                else disc_sigma_model
            ),
        }

    def initial_unconstrained(
        self,
        *,
        T_c: float,
        nu: float,
        beta: float,
        gp_ell: float | None = None,
        gp_eta: float | None = None,
        correction_gp_ell: float | None = None,
        correction_gp_eta: float | None = None,
        disc_t0: float | None = None,
        disc_L0: float | None = None,
        disc_p: float | None = None,
        disc_q: float | None = None,
        disc_sigma_model: float | None = None,
    ) -> jnp.ndarray:
        values = self._initial_value_map(
            T_c=T_c,
            nu=nu,
            beta=beta,
            gp_ell=self.fixed_gp_ell if gp_ell is None else gp_ell,
            gp_eta=self.fixed_gp_eta if gp_eta is None else gp_eta,
            correction_gp_ell=(
                self.fixed_correction_gp_ell
                if correction_gp_ell is None
                else correction_gp_ell
            ),
            correction_gp_eta=(
                self.fixed_correction_gp_eta
                if correction_gp_eta is None
                else correction_gp_eta
            ),
            disc_t0=disc_t0,
            disc_L0=disc_L0,
            disc_p=disc_p,
            disc_q=disc_q,
            disc_sigma_model=disc_sigma_model,
        )
        return jnp.asarray(
            [slot.to_unconstrained(values[slot.name]) for slot in self.slots],
            dtype=jnp.float64,
        )

    def initial_unconstrained_jax(
        self,
        *,
        T_c: jnp.ndarray,
        nu: jnp.ndarray,
        beta: jnp.ndarray,
        gp_ell: jnp.ndarray | None = None,
        gp_eta: jnp.ndarray | None = None,
        correction_gp_ell: jnp.ndarray | None = None,
        correction_gp_eta: jnp.ndarray | None = None,
        disc_t0: jnp.ndarray | None = None,
        disc_L0: jnp.ndarray | None = None,
        disc_p: jnp.ndarray | None = None,
        disc_q: jnp.ndarray | None = None,
        disc_sigma_model: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        values = {
            "T_c": jnp.asarray(T_c, dtype=jnp.float64),
            "nu": jnp.asarray(nu, dtype=jnp.float64),
            "beta": jnp.asarray(beta, dtype=jnp.float64),
            "T_c_delta": (jnp.asarray(T_c, dtype=jnp.float64) - self.tc_reparam_ref)
            * self.tc_delta_scale,
            "gp_ell": (
                jnp.asarray(self.fixed_gp_ell, dtype=jnp.float64)
                if gp_ell is None
                else jnp.asarray(gp_ell, dtype=jnp.float64)
            ),
            "gp_eta": (
                jnp.asarray(self.fixed_gp_eta, dtype=jnp.float64)
                if gp_eta is None
                else jnp.asarray(gp_eta, dtype=jnp.float64)
            ),
            "correction_gp_ell": (
                jnp.asarray(self.fixed_correction_gp_ell, dtype=jnp.float64)
                if correction_gp_ell is None
                else jnp.asarray(correction_gp_ell, dtype=jnp.float64)
            ),
            "correction_gp_eta": (
                jnp.asarray(self.fixed_correction_gp_eta, dtype=jnp.float64)
                if correction_gp_eta is None
                else jnp.asarray(correction_gp_eta, dtype=jnp.float64)
            ),
            "disc_t0": (
                jnp.asarray(self.fixed_disc_t0, dtype=jnp.float64)
                if disc_t0 is None
                else jnp.asarray(disc_t0, dtype=jnp.float64)
            ),
            "disc_L0": (
                jnp.asarray(self.fixed_disc_L0, dtype=jnp.float64)
                if disc_L0 is None
                else jnp.asarray(disc_L0, dtype=jnp.float64)
            ),
            "disc_p": (
                jnp.asarray(self.fixed_disc_p, dtype=jnp.float64)
                if disc_p is None
                else jnp.asarray(disc_p, dtype=jnp.float64)
            ),
            "disc_q": (
                jnp.asarray(self.fixed_disc_q, dtype=jnp.float64)
                if disc_q is None
                else jnp.asarray(disc_q, dtype=jnp.float64)
            ),
            "disc_sigma_model": (
                jnp.asarray(self.fixed_disc_sigma_model, dtype=jnp.float64)
                if disc_sigma_model is None
                else jnp.asarray(disc_sigma_model, dtype=jnp.float64)
            ),
        }
        return jnp.stack(
            [slot.to_unconstrained_jax(values[slot.name]) for slot in self.slots]
        ).astype(jnp.float64)


def build_fss_inference_layout(
    *,
    infer_Tc: bool,
    infer_nu: bool,
    infer_beta: bool,
    infer_gp_hyperparams: bool = False,
    uses_correction: bool = False,
    uses_discrepancy: bool = False,
    infer_discrepancy: bool | None = None,
    infer_disc_t0: bool | None = None,
    infer_disc_L0: bool | None = None,
    infer_disc_p: bool | None = None,
    infer_disc_q: bool | None = None,
    infer_disc_sigma_model: bool | None = None,
    base_gp_ell: float = 1.0,
    base_gp_eta: float = 1.0,
    correction_gp_ell: float = 1.0,
    correction_gp_eta: float = 1.0,
    gp_ell_prior_sigma: float = DEFAULT_GP_ELL_PRIOR_SIGMA,
    reparametrize_Tc: bool = False,
    tc_reparam_ref: float | None = None,
    tc_delta_scale: float = TC_DELTA_SCALE,
    tc_prior_lower: float = TC_PRIOR_LOWER,
    tc_prior_upper: float = TC_PRIOR_UPPER,
    nu_prior_lower: float = NU_PRIOR_LOWER,
    nu_prior_upper: float = NU_PRIOR_UPPER,
    beta_prior_lower: float = BETA_PRIOR_LOWER,
    beta_prior_upper: float = BETA_PRIOR_UPPER,
    fixed_Tc: float = TC_EXACT,
    fixed_nu: float = NU_EXACT,
    fixed_beta: float = BETA_EXACT,
    fixed_disc_t0: float = DISCREPANCY_T0_INIT,
    fixed_disc_L0: float = DISCREPANCY_L0_INIT,
    fixed_disc_p: float = DISCREPANCY_P_INIT,
    fixed_disc_q: float = DISCREPANCY_Q_INIT,
    fixed_disc_sigma_model: float = DISCREPANCY_SIGMA_MODEL_INIT,
) -> FssJaxInferenceLayout:
    tc_ref = float(fixed_Tc) if tc_reparam_ref is None else float(tc_reparam_ref)
    sample_disc_default = uses_discrepancy and (
        True if infer_discrepancy is None else bool(infer_discrepancy)
    )

    def _disc_flag(override: bool | None) -> bool:
        return sample_disc_default if override is None else bool(override)

    free_disc_t0 = _disc_flag(infer_disc_t0)
    free_disc_L0 = _disc_flag(infer_disc_L0)
    free_disc_p = _disc_flag(infer_disc_p)
    free_disc_q = _disc_flag(infer_disc_q)
    free_disc_sigma_model = _disc_flag(infer_disc_sigma_model)
    sample_disc = any(
        (
            free_disc_t0,
            free_disc_L0,
            free_disc_p,
            free_disc_q,
            free_disc_sigma_model,
        )
    )
    slots: list[FssJaxInferenceSlot] = []
    if infer_Tc:
        if reparametrize_Tc:
            delta_lo = (tc_prior_lower - tc_ref) * tc_delta_scale
            delta_hi = (tc_prior_upper - tc_ref) * tc_delta_scale
            slots.append(FssJaxUniformSlot("T_c_delta", delta_lo, delta_hi))
        else:
            slots.append(FssJaxUniformSlot("T_c", tc_prior_lower, tc_prior_upper))
    if infer_nu:
        slots.append(FssJaxUniformSlot("nu", nu_prior_lower, nu_prior_upper))
    if infer_beta:
        slots.append(FssJaxUniformSlot("beta", beta_prior_lower, beta_prior_upper))
    if infer_gp_hyperparams:
        slots.append(
            FssJaxLognormalSlot(
                "gp_ell",
                mu=float(np.log(base_gp_ell)),
                sigma=gp_ell_prior_sigma,
            )
        )
        slots.append(
            FssJaxHalfNormalSlot(
                "gp_eta",
                sigma=max(float(base_gp_eta), 1.0),
            )
        )
        if uses_correction:
            slots.append(
                FssJaxLognormalSlot(
                    "correction_gp_ell",
                    mu=float(np.log(correction_gp_ell)),
                    sigma=gp_ell_prior_sigma,
                )
            )
            slots.append(
                FssJaxHalfNormalSlot(
                    "correction_gp_eta",
                    sigma=max(float(correction_gp_eta), 1.0),
                )
            )
    if free_disc_t0:
        slots.append(
            FssJaxUniformSlot(
                "disc_t0", DISCREPANCY_T0_PRIOR_LOWER, DISCREPANCY_T0_PRIOR_UPPER
            )
        )
    if free_disc_L0:
        slots.append(
            FssJaxUniformSlot(
                "disc_L0", DISCREPANCY_L0_PRIOR_LOWER, DISCREPANCY_L0_PRIOR_UPPER
            )
        )
    if free_disc_p:
        slots.append(
            FssJaxUniformSlot(
                "disc_p", DISCREPANCY_P_PRIOR_LOWER, DISCREPANCY_P_PRIOR_UPPER
            )
        )
    if free_disc_q:
        slots.append(
            FssJaxUniformSlot(
                "disc_q", DISCREPANCY_Q_PRIOR_LOWER, DISCREPANCY_Q_PRIOR_UPPER
            )
        )
    if free_disc_sigma_model:
        slots.append(
            FssJaxHalfNormalSlot(
                "disc_sigma_model",
                sigma=DISCREPANCY_SIGMA_MODEL_PRIOR_SIGMA,
            )
        )
    return FssJaxInferenceLayout(
        slots=tuple(slots),
        infer_Tc=infer_Tc,
        infer_nu=infer_nu,
        infer_beta=infer_beta,
        infer_gp_hyperparams=infer_gp_hyperparams,
        uses_correction=uses_correction,
        # True only when at least one disc param is free (sampled). Fixed disc
        # still enters the likelihood via fixed_disc_* + unpack_discrepancy.
        uses_discrepancy=sample_disc,
        reparametrize_Tc=reparametrize_Tc,
        tc_reparam_ref=tc_ref,
        tc_delta_scale=float(tc_delta_scale),
        fixed_Tc=float(fixed_Tc),
        fixed_nu=float(fixed_nu),
        fixed_beta=float(fixed_beta),
        fixed_gp_ell=float(base_gp_ell),
        fixed_gp_eta=float(base_gp_eta),
        fixed_correction_gp_ell=float(correction_gp_ell),
        fixed_correction_gp_eta=float(correction_gp_eta),
        fixed_disc_t0=float(fixed_disc_t0),
        fixed_disc_L0=float(fixed_disc_L0),
        fixed_disc_p=float(fixed_disc_p),
        fixed_disc_q=float(fixed_disc_q),
        fixed_disc_sigma_model=float(fixed_disc_sigma_model),
        infer_disc_t0=free_disc_t0,
        infer_disc_L0=free_disc_L0,
        infer_disc_p=free_disc_p,
        infer_disc_q=free_disc_q,
        infer_disc_sigma_model=free_disc_sigma_model,
    )


@dataclass(frozen=True)
class CompiledFssLogPosterior:
    layout: FssJaxInferenceLayout
    logdensity: Callable[[jnp.ndarray], jnp.ndarray]
    logdensity_and_grad: Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]
    likelihood: Any

    def initial_position(
        self,
        *,
        T_c: float = TC_EXACT,
        nu: float = NU_EXACT,
        beta: float = BETA_EXACT,
        gp_ell: float | None = None,
        gp_eta: float | None = None,
        correction_gp_ell: float | None = None,
        correction_gp_eta: float | None = None,
    ) -> jnp.ndarray:
        return self.layout.initial_unconstrained(
            T_c=T_c,
            nu=nu,
            beta=beta,
            gp_ell=gp_ell,
            gp_eta=gp_eta,
            correction_gp_ell=correction_gp_ell,
            correction_gp_eta=correction_gp_eta,
        )


def compile_fss_log_posterior(
    observables: pd.DataFrame,
    config: FssProfileConfig,
    *,
    infer_Tc: bool,
    infer_nu: bool,
    infer_beta: bool,
    infer_gp_hyperparams: bool = False,
    infer_discrepancy: bool | None = None,
    infer_disc_t0: bool | None = None,
    infer_disc_L0: bool | None = None,
    infer_disc_p: bool | None = None,
    infer_disc_q: bool | None = None,
    infer_disc_sigma_model: bool | None = None,
    gp_ell_prior_sigma: float = DEFAULT_GP_ELL_PRIOR_SIGMA,
    reparametrize_Tc: bool = False,
    tc_reparam_ref: float | None = None,
    tc_delta_scale: float = TC_DELTA_SCALE,
    tc_prior_lower: float = TC_PRIOR_LOWER,
    tc_prior_upper: float = TC_PRIOR_UPPER,
    fixed_Tc: float = TC_EXACT,
    fixed_nu: float = NU_EXACT,
    fixed_beta: float = BETA_EXACT,
    fixed_disc_t0: float = DISCREPANCY_T0_INIT,
    fixed_disc_L0: float = DISCREPANCY_L0_INIT,
    fixed_disc_p: float = DISCREPANCY_P_INIT,
    fixed_disc_q: float = DISCREPANCY_Q_INIT,
    fixed_disc_sigma_model: float = DISCREPANCY_SIGMA_MODEL_INIT,
) -> CompiledFssLogPosterior:
    """Compile ``u -> log p(data, u)`` with uniform priors on exponents."""
    configure_jax()
    infer_beta_eff = infer_beta and beta_affects_fss_likelihood(
        use_m=config.use_m,
        use_m2=config.use_m2,
        use_m4=config.use_m4,
        use_chi=config.use_chi,
    )
    arrays = extract_likelihood_arrays(observables, config)
    scales = fss_gp_scales(
        arrays.L,
        arrays.T,
        gp_ell_factor=config.gp_ell_factor,
        gp_eta=config.gp_eta,
        correction_gp_ell_factor=config.correction_gp_ell_factor,
        correction_gp_eta=config.correction_gp_eta,
        gp_ell=getattr(config, "gp_ell", None),
        correction_gp_ell=getattr(config, "correction_gp_ell", None),
    )
    uses_correction = uses_fss_correction(config)
    uses_discrepancy = uses_fss_discrepancy(config)
    layout = build_fss_inference_layout(
        infer_Tc=infer_Tc,
        infer_nu=infer_nu,
        infer_beta=infer_beta_eff,
        infer_gp_hyperparams=infer_gp_hyperparams,
        uses_correction=uses_correction,
        uses_discrepancy=uses_discrepancy,
        infer_discrepancy=infer_discrepancy,
        infer_disc_t0=infer_disc_t0,
        infer_disc_L0=infer_disc_L0,
        infer_disc_p=infer_disc_p,
        infer_disc_q=infer_disc_q,
        infer_disc_sigma_model=infer_disc_sigma_model,
        base_gp_ell=scales.base_gp_ell,
        base_gp_eta=scales.base_gp_eta,
        correction_gp_ell=scales.correction_gp_ell,
        correction_gp_eta=scales.correction_gp_eta,
        gp_ell_prior_sigma=gp_ell_prior_sigma,
        reparametrize_Tc=reparametrize_Tc,
        tc_reparam_ref=tc_reparam_ref,
        tc_delta_scale=tc_delta_scale,
        tc_prior_lower=tc_prior_lower,
        tc_prior_upper=tc_prior_upper,
        fixed_Tc=fixed_Tc,
        fixed_nu=fixed_nu,
        fixed_beta=fixed_beta,
        fixed_disc_t0=fixed_disc_t0,
        fixed_disc_L0=fixed_disc_L0,
        fixed_disc_p=fixed_disc_p,
        fixed_disc_q=fixed_disc_q,
        fixed_disc_sigma_model=fixed_disc_sigma_model,
    )
    likelihood = compile_fss_log_marginal_likelihood_jax(observables, config)

    def logdensity(u: jnp.ndarray) -> jnp.ndarray:
        u = jnp.asarray(u, dtype=jnp.float64)
        tc, nu, beta = layout.unpack_physical(u)
        gp_ell, gp_eta, correction_gp_ell, correction_gp_eta = layout.unpack_gp_hyperparams(
            u
        )
        disc_t0, disc_L0, disc_p, disc_q, disc_sigma_model = layout.unpack_discrepancy(u)
        log_like = likelihood.evaluate_with_gp_jax(
            tc,
            nu,
            beta,
            gp_ell,
            gp_eta,
            correction_gp_ell,
            correction_gp_eta,
            disc_t0,
            disc_L0,
            disc_p,
            disc_q,
            disc_sigma_model,
            infer_gp_hyperparams,
        )
        if layout.ndim == 0:
            return log_like
        return log_like + layout.log_prior(u)

    logdensity_jit = jax.jit(logdensity)
    logdensity_and_grad = jax.jit(jax.value_and_grad(logdensity))

    return CompiledFssLogPosterior(
        layout=layout,
        logdensity=lambda u: logdensity_jit(jnp.asarray(u, dtype=jnp.float64)),
        logdensity_and_grad=lambda u: logdensity_and_grad(
            jnp.asarray(u, dtype=jnp.float64)
        ),
        likelihood=likelihood,
    )


def _unpack_chain_positions(
    layout: FssJaxInferenceLayout,
    positions: jnp.ndarray,
) -> dict[str, np.ndarray]:
    """positions shape (chains, draws, ndim) -> posterior arrays."""
    chains, draws, ndim = positions.shape
    flat = positions.reshape(chains * draws, ndim)
    tc_list: list[float] = []
    nu_list: list[float] = []
    beta_list: list[float] = []
    gp_ell_list: list[float] = []
    gp_eta_list: list[float] = []
    correction_gp_ell_list: list[float] = []
    correction_gp_eta_list: list[float] = []
    disc_t0_list: list[float] = []
    disc_L0_list: list[float] = []
    disc_p_list: list[float] = []
    disc_q_list: list[float] = []
    disc_sigma_model_list: list[float] = []
    for row in np.asarray(flat):
        u = jnp.asarray(row)
        tc, nu, beta = layout.unpack_physical(u)
        gp_ell, gp_eta, correction_gp_ell, correction_gp_eta = layout.unpack_gp_hyperparams(
            u
        )
        disc_t0, disc_L0, disc_p, disc_q, disc_sigma_model = layout.unpack_discrepancy(u)
        tc_list.append(float(tc))
        nu_list.append(float(nu))
        beta_list.append(float(beta))
        if layout.infer_gp_hyperparams:
            gp_ell_list.append(float(gp_ell))
            gp_eta_list.append(float(gp_eta))
            if layout.uses_correction:
                correction_gp_ell_list.append(float(correction_gp_ell))
                correction_gp_eta_list.append(float(correction_gp_eta))
        if layout.infer_disc_t0:
            disc_t0_list.append(float(disc_t0))
        if layout.infer_disc_L0:
            disc_L0_list.append(float(disc_L0))
        if layout.infer_disc_p:
            disc_p_list.append(float(disc_p))
        if layout.infer_disc_q:
            disc_q_list.append(float(disc_q))
        if layout.infer_disc_sigma_model:
            disc_sigma_model_list.append(float(disc_sigma_model))
    tc_arr = np.asarray(tc_list, dtype=np.float64).reshape(chains, draws)
    nu_arr = np.asarray(nu_list, dtype=np.float64).reshape(chains, draws)
    beta_arr = np.asarray(beta_list, dtype=np.float64).reshape(chains, draws)
    out: dict[str, np.ndarray] = {}
    if layout.infer_Tc:
        out["T_c"] = tc_arr
    if layout.infer_nu:
        out["nu"] = nu_arr
    if layout.infer_beta:
        out["beta"] = beta_arr
    gamma = 2.0 * nu_arr - 2.0 * beta_arr  # SPATIAL_DIMENSION=2
    if layout.infer_nu or layout.infer_beta:
        out["gamma"] = gamma
    if layout.infer_gp_hyperparams:
        out["gp_ell"] = np.asarray(gp_ell_list, dtype=np.float64).reshape(chains, draws)
        out["gp_eta"] = np.asarray(gp_eta_list, dtype=np.float64).reshape(chains, draws)
        if layout.uses_correction:
            out["correction_gp_ell"] = np.asarray(
                correction_gp_ell_list, dtype=np.float64
            ).reshape(chains, draws)
            out["correction_gp_eta"] = np.asarray(
                correction_gp_eta_list, dtype=np.float64
            ).reshape(chains, draws)
    if layout.infer_disc_t0:
        out["disc_t0"] = np.asarray(disc_t0_list, dtype=np.float64).reshape(chains, draws)
    if layout.infer_disc_L0:
        out["disc_L0"] = np.asarray(disc_L0_list, dtype=np.float64).reshape(chains, draws)
    if layout.infer_disc_p:
        out["disc_p"] = np.asarray(disc_p_list, dtype=np.float64).reshape(chains, draws)
    if layout.infer_disc_q:
        out["disc_q"] = np.asarray(disc_q_list, dtype=np.float64).reshape(chains, draws)
    if layout.infer_disc_sigma_model:
        out["disc_sigma_model"] = np.asarray(
            disc_sigma_model_list, dtype=np.float64
        ).reshape(chains, draws)
    return out


def _idata_from_posterior_vars(posterior_vars: dict[str, np.ndarray]) -> az.InferenceData:
    """Build ArviZ ``InferenceData`` from chain/draw arrays."""
    if not posterior_vars:
        raise ValueError("No posterior variables to store")
    sample = next(iter(posterior_vars.values()))
    n_chain, n_draw = sample.shape[:2]
    return az.from_dict(
        posterior=posterior_vars,
        dims={name: ["chain", "draw"] for name in posterior_vars},
        coords={
            "chain": np.arange(n_chain),
            "draw": np.arange(n_draw),
        },
    )


def sample_fss_posterior_jax(
    observables: pd.DataFrame,
    config: FssProfileConfig,
    *,
    infer_Tc: bool,
    infer_nu: bool,
    infer_beta: bool,
    infer_gp_hyperparams: bool = False,
    gp_ell_prior_sigma: float = DEFAULT_GP_ELL_PRIOR_SIGMA,
    draws: int,
    tune: int,
    chains: int,
    target_accept: float = 0.9,
    seed: int = 1,
    reparametrize_Tc: bool = False,
    tc_reparam_ref: float | None = None,
    tc_delta_scale: float = TC_DELTA_SCALE,
    tc_prior_lower: float = TC_PRIOR_LOWER,
    tc_prior_upper: float = TC_PRIOR_UPPER,
    tc_init: float | None = None,
    fixed_Tc: float = TC_EXACT,
    fixed_nu: float = NU_EXACT,
    fixed_beta: float = BETA_EXACT,
    init_positions: Sequence[jnp.ndarray] | None = None,
    progress_bar: bool = False,
) -> az.InferenceData:
    """Run Blackjax NUTS (vmap'd chains) and return ArviZ ``InferenceData``."""
    import blackjax
    from blackjax.util import run_inference_algorithm

    configure_jax()
    tc0 = float(fixed_Tc if tc_init is None else tc_init)
    posterior = compile_fss_log_posterior(
        observables,
        config,
        infer_Tc=infer_Tc,
        infer_nu=infer_nu,
        infer_beta=infer_beta,
        infer_gp_hyperparams=infer_gp_hyperparams,
        gp_ell_prior_sigma=gp_ell_prior_sigma,
        reparametrize_Tc=reparametrize_Tc,
        tc_reparam_ref=tc_reparam_ref,
        tc_delta_scale=tc_delta_scale,
        tc_prior_lower=tc_prior_lower,
        tc_prior_upper=tc_prior_upper,
        fixed_Tc=fixed_Tc,
        fixed_nu=fixed_nu,
        fixed_beta=fixed_beta,
    )
    layout = posterior.layout
    if layout.ndim == 0:
        raise ValueError(
            "JAX sampler requires at least one inferred parameter "
            "(exponent and/or GP hyperparameter)"
        )

    if init_positions is None:
        init_positions = [
            posterior.initial_position(T_c=tc0, nu=fixed_nu, beta=fixed_beta)
            for _ in range(chains)
        ]
    if len(init_positions) != chains:
        raise ValueError(f"Expected {chains} init positions, got {len(init_positions)}")

    init_stack = jnp.stack(
        [jnp.asarray(p, dtype=jnp.float64) for p in init_positions], axis=0
    )

    def _adapt_and_sample(
        rng_key: jax.Array,
        position: jax.Array,
        *,
        show_progress: bool,
    ) -> jax.Array:
        adapt_key, sample_key = jax.random.split(rng_key)
        warmup = blackjax.window_adaptation(
            blackjax.nuts,
            posterior.logdensity,
            target_acceptance_rate=target_accept,
            progress_bar=show_progress,
        )
        adapt_result, _ = warmup.run(adapt_key, position, num_steps=tune)
        nuts = blackjax.nuts(posterior.logdensity, **adapt_result.parameters)
        _, history = run_inference_algorithm(
            sample_key,
            nuts,
            num_steps=draws,
            initial_state=adapt_result.state,
            progress_bar=show_progress,
            transform=lambda state, _info: state.position,
        )
        return history

    keys = jax.random.split(jax.random.key(seed), chains)
    if progress_bar:
        if chains > 1:
            print(
                f"  NUTS: running {chains} chains sequentially "
                f"(progress bars; use progress_bar=False for parallel vmap)"
            )
        histories: list[jax.Array] = []
        for chain_idx in range(chains):
            if chains > 1:
                print(f"  Chain {chain_idx + 1}/{chains}")
            histories.append(
                _adapt_and_sample(
                    keys[chain_idx],
                    init_stack[chain_idx],
                    show_progress=True,
                )
            )
        chain_positions = jnp.stack(histories, axis=0)
    else:
        chain_positions = jax.jit(
            jax.vmap(
                lambda key, position: _adapt_and_sample(
                    key, position, show_progress=False
                )
            )
        )(keys, init_stack)
    chain_positions.block_until_ready()

    positions = np.asarray(chain_positions, dtype=np.float64)
    posterior_vars = _unpack_chain_positions(layout, jnp.asarray(positions))
    idata = _idata_from_posterior_vars(posterior_vars)
    idata.posterior.attrs["sampler_backend"] = "jax"
    idata.posterior.attrs["jax_sampler"] = "nuts"
    idata.posterior.attrs["infer_gp_hyperparams"] = str(infer_gp_hyperparams)
    idata.posterior.attrs["tc_prior_lower"] = str(tc_prior_lower)
    idata.posterior.attrs["tc_prior_upper"] = str(tc_prior_upper)
    idata.posterior.attrs["tc_init"] = str(tc0)
    idata.posterior.attrs["jax_device_summary"] = jax_device_summary()
    return idata


def _make_laps_sample_init(
    layout: FssJaxInferenceLayout,
    *,
    infer_Tc: bool,
    infer_nu: bool,
    infer_beta: bool,
    infer_gp_hyperparams: bool,
    mle: tuple[float, float, float] | None = None,
) -> Callable[[jax.Array], jnp.ndarray]:
    """Per-chain init with optional jitter around exact exponents or profile MLE."""
    if mle is not None:
        tc0, nu0, beta0 = mle
    else:
        tc0, nu0, beta0 = layout.fixed_Tc, layout.fixed_nu, layout.fixed_beta

    jitter = _LAPS_INIT_JITTER

    def sample_init(key: jax.Array) -> jnp.ndarray:
        keys = jax.random.split(key, 16)
        key_tc, key_nu, key_beta = keys[0], keys[1], keys[2]
        key_gp_ell, key_gp_eta, key_corr_ell, key_corr_eta = keys[3:7]
        key_t0, key_L0, key_p, key_q, key_sm = keys[7:12]
        tc = (
            tc0 + jax.random.normal(key_tc) * jitter["T_c"]
            if infer_Tc
            else jnp.asarray(tc0, dtype=jnp.float64)
        )
        nu = (
            nu0 + jax.random.normal(key_nu) * jitter["nu"]
            if infer_nu
            else jnp.asarray(nu0, dtype=jnp.float64)
        )
        beta = (
            beta0 + jax.random.normal(key_beta) * jitter["beta"]
            if infer_beta
            else jnp.asarray(beta0, dtype=jnp.float64)
        )
        gp_ell = jnp.asarray(layout.fixed_gp_ell, dtype=jnp.float64)
        gp_eta = jnp.asarray(layout.fixed_gp_eta, dtype=jnp.float64)
        correction_gp_ell = jnp.asarray(layout.fixed_correction_gp_ell, dtype=jnp.float64)
        correction_gp_eta = jnp.asarray(layout.fixed_correction_gp_eta, dtype=jnp.float64)
        if infer_gp_hyperparams:
            gp_ell = layout.fixed_gp_ell * jnp.exp(
                jax.random.normal(key_gp_ell) * jitter["gp_ell"]
            )
            gp_eta = layout.fixed_gp_eta * jnp.exp(
                jax.random.normal(key_gp_eta) * jitter["gp_eta"]
            )
            if layout.uses_correction:
                correction_gp_ell = layout.fixed_correction_gp_ell * jnp.exp(
                    jax.random.normal(key_corr_ell) * jitter["correction_gp_ell"]
                )
                correction_gp_eta = layout.fixed_correction_gp_eta * jnp.exp(
                    jax.random.normal(key_corr_eta) * jitter["correction_gp_eta"]
                )
        disc_t0 = jnp.asarray(layout.fixed_disc_t0, dtype=jnp.float64)
        disc_L0 = jnp.asarray(layout.fixed_disc_L0, dtype=jnp.float64)
        disc_p = jnp.asarray(layout.fixed_disc_p, dtype=jnp.float64)
        disc_q = jnp.asarray(layout.fixed_disc_q, dtype=jnp.float64)
        disc_sigma_model = jnp.asarray(layout.fixed_disc_sigma_model, dtype=jnp.float64)
        if layout.infer_disc_t0:
            disc_t0 = jnp.clip(
                layout.fixed_disc_t0 + jax.random.normal(key_t0) * jitter["disc_t0"],
                DISCREPANCY_T0_PRIOR_LOWER,
                DISCREPANCY_T0_PRIOR_UPPER,
            )
        if layout.infer_disc_L0:
            disc_L0 = jnp.clip(
                layout.fixed_disc_L0 + jax.random.normal(key_L0) * jitter["disc_L0"],
                DISCREPANCY_L0_PRIOR_LOWER,
                DISCREPANCY_L0_PRIOR_UPPER,
            )
        if layout.infer_disc_p:
            disc_p = jnp.clip(
                layout.fixed_disc_p + jax.random.normal(key_p) * jitter["disc_p"],
                DISCREPANCY_P_PRIOR_LOWER,
                DISCREPANCY_P_PRIOR_UPPER,
            )
        if layout.infer_disc_q:
            disc_q = jnp.clip(
                layout.fixed_disc_q + jax.random.normal(key_q) * jitter["disc_q"],
                DISCREPANCY_Q_PRIOR_LOWER,
                DISCREPANCY_Q_PRIOR_UPPER,
            )
        if layout.infer_disc_sigma_model:
            disc_sigma_model = jnp.maximum(
                layout.fixed_disc_sigma_model
                * jnp.exp(jax.random.normal(key_sm) * jitter["disc_sigma_model"]),
                1e-6,
            )
        return layout.initial_unconstrained_jax(
            T_c=tc,
            nu=nu,
            beta=beta,
            gp_ell=gp_ell,
            gp_eta=gp_eta,
            correction_gp_ell=correction_gp_ell,
            correction_gp_eta=correction_gp_eta,
            disc_t0=disc_t0,
            disc_L0=disc_L0,
            disc_p=disc_p,
            disc_q=disc_q,
            disc_sigma_model=disc_sigma_model,
        )

    return sample_init


def sample_fss_posterior_laps(
    observables: pd.DataFrame,
    config: FssProfileConfig,
    *,
    infer_Tc: bool,
    infer_nu: bool,
    infer_beta: bool,
    infer_gp_hyperparams: bool = False,
    infer_discrepancy: bool | None = None,
    infer_disc_t0: bool | None = None,
    infer_disc_L0: bool | None = None,
    infer_disc_p: bool | None = None,
    infer_disc_q: bool | None = None,
    infer_disc_sigma_model: bool | None = None,
    gp_ell_prior_sigma: float = DEFAULT_GP_ELL_PRIOR_SIGMA,
    tune: int,
    draws: int,
    chains: int = LAPS_DEFAULT_CHAINS,
    seed: int = 1,
    reparametrize_Tc: bool = False,
    tc_reparam_ref: float | None = None,
    tc_delta_scale: float = TC_DELTA_SCALE,
    tc_prior_lower: float = TC_PRIOR_LOWER,
    tc_prior_upper: float = TC_PRIOR_UPPER,
    tc_init: float | None = None,
    fixed_Tc: float = TC_EXACT,
    fixed_nu: float = NU_EXACT,
    fixed_beta: float = BETA_EXACT,
    fixed_disc_t0: float = DISCREPANCY_T0_INIT,
    fixed_disc_L0: float = DISCREPANCY_L0_INIT,
    fixed_disc_p: float = DISCREPANCY_P_INIT,
    fixed_disc_q: float = DISCREPANCY_Q_INIT,
    fixed_disc_sigma_model: float = DISCREPANCY_SIGMA_MODEL_INIT,
    mle_init: tuple[float, float, float] | None = None,
    early_stop: bool = True,
    steps_per_sample: int = 15,
    progress_bar: bool = False,
) -> az.InferenceData:
    """Run Blackjax LAPS (parallel chains) and return ArviZ ``InferenceData``.

    Phase 1 (``tune`` steps) uses an unadjusted kernel for fast equilibration;
    phase 2 (``draws`` steps) refines with an adjusted kernel. Each chain
    contributes one posterior draw (shape ``chains × 1``).
    """
    from blackjax.adaptation.laps import laps

    configure_jax()
    fixed_Tc_eff = float(fixed_Tc if tc_init is None else tc_init)
    posterior = compile_fss_log_posterior(
        observables,
        config,
        infer_Tc=infer_Tc,
        infer_nu=infer_nu,
        infer_beta=infer_beta,
        infer_gp_hyperparams=infer_gp_hyperparams,
        infer_discrepancy=infer_discrepancy,
        infer_disc_t0=infer_disc_t0,
        infer_disc_L0=infer_disc_L0,
        infer_disc_p=infer_disc_p,
        infer_disc_q=infer_disc_q,
        infer_disc_sigma_model=infer_disc_sigma_model,
        gp_ell_prior_sigma=gp_ell_prior_sigma,
        reparametrize_Tc=reparametrize_Tc,
        tc_reparam_ref=tc_reparam_ref,
        tc_delta_scale=tc_delta_scale,
        tc_prior_lower=tc_prior_lower,
        tc_prior_upper=tc_prior_upper,
        fixed_Tc=fixed_Tc_eff,
        fixed_nu=fixed_nu,
        fixed_beta=fixed_beta,
        fixed_disc_t0=fixed_disc_t0,
        fixed_disc_L0=fixed_disc_L0,
        fixed_disc_p=fixed_disc_p,
        fixed_disc_q=fixed_disc_q,
        fixed_disc_sigma_model=fixed_disc_sigma_model,
    )
    layout = posterior.layout
    if layout.ndim == 0:
        raise ValueError(
            "JAX sampler requires at least one inferred parameter "
            "(exponent and/or GP hyperparameter)"
        )

    sample_init = _make_laps_sample_init(
        layout,
        infer_Tc=infer_Tc,
        infer_nu=infer_nu,
        infer_beta=infer_beta,
        infer_gp_hyperparams=infer_gp_hyperparams,
        mle=mle_init,
    )
    mesh = jax.sharding.Mesh(jax.devices()[:1], "chains")

    if progress_bar:
        print(
            f"  LAPS: {chains} parallel chains, "
            f"phase1={tune}, phase2={draws}, steps_per_sample={steps_per_sample} "
            "(no step progress bar; timing wall-clock only)"
        )

    t0 = time.perf_counter()
    _, _, acc_prob, final_state = laps(
        logdensity_fn=posterior.logdensity,
        sample_init=sample_init,
        ndims=layout.ndim,
        num_steps1=tune,
        num_steps2=draws,
        num_chains=chains,
        mesh=mesh,
        rng_key=jax.random.key(seed),
        early_stop=early_stop,
        diagonal_preconditioning=True,
        steps_per_sample=steps_per_sample,
        diagnostics=False,
        superchain_size=1,
    )
    final_state.position.block_until_ready()
    if progress_bar:
        elapsed = time.perf_counter() - t0
        print(f"  LAPS finished in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    positions = jnp.asarray(final_state.position, dtype=jnp.float64)[:, None, :]
    posterior_vars = _unpack_chain_positions(layout, positions)
    idata = _idata_from_posterior_vars(posterior_vars)
    idata.posterior.attrs["sampler_backend"] = "laps"
    idata.posterior.attrs["jax_sampler"] = "laps"
    idata.posterior.attrs["infer_gp_hyperparams"] = str(infer_gp_hyperparams)
    idata.posterior.attrs["laps_num_steps1"] = str(tune)
    idata.posterior.attrs["laps_num_steps2"] = str(draws)
    idata.posterior.attrs["laps_chains"] = str(chains)
    idata.posterior.attrs["laps_acc_prob"] = str(acc_prob)
    idata.posterior.attrs["laps_steps_per_sample"] = str(steps_per_sample)
    idata.posterior.attrs["tc_prior_lower"] = str(tc_prior_lower)
    idata.posterior.attrs["tc_prior_upper"] = str(tc_prior_upper)
    idata.posterior.attrs["tc_init"] = str(fixed_Tc_eff)
    idata.posterior.attrs["jax_device_summary"] = jax_device_summary()
    return idata


def profile_config_from_fit_kwargs(
    *,
    use_m: bool,
    use_m2: bool,
    use_m4: bool,
    use_binder: bool,
    use_chi: bool,
    correction_m: bool,
    correction_m2: bool,
    correction_m4: bool,
    correction_binder: bool,
    correction_chi: bool,
    use_log_m: bool,
    gp_ell_factor: float | None,
    gp_eta: float | None,
    correction_gp_ell_factor: float | None,
    correction_gp_eta: float | None,
    gp_kernel: str | None,
    obs_sigma_scale: float | None,
    infer_Tc: bool = True,
    infer_nu: bool = True,
    infer_beta: bool = True,
    reparametrize_Tc: bool = False,
    tc_delta_scale: float | None = None,
    likelihood_backend: str = "jax",
    discrepancy_m: bool = False,
    discrepancy_m2: bool = False,
    discrepancy_m4: bool = False,
    discrepancy_binder: bool = False,
    discrepancy_chi: bool = False,
    discrepancy_form: str | None = None,
    gp_ell: float | None = None,
    correction_gp_ell: float | None = None,
) -> FssProfileConfig:
    """Mirror :func:`fit._profile_config_from_fit_kwargs` for JAX sampling."""
    from .gp_kernels import normalize_gp_kernel

    kwargs: dict[str, object] = {
        "use_m": use_m,
        "use_m2": use_m2,
        "use_m4": use_m4,
        "use_binder": use_binder,
        "use_chi": use_chi,
        "correction_m": correction_m,
        "correction_m2": correction_m2,
        "correction_m4": correction_m4,
        "correction_binder": correction_binder,
        "correction_chi": correction_chi,
        "discrepancy_m": discrepancy_m,
        "discrepancy_m2": discrepancy_m2,
        "discrepancy_m4": discrepancy_m4,
        "discrepancy_binder": discrepancy_binder,
        "discrepancy_chi": discrepancy_chi,
        "use_log_m": use_log_m,
        "infer_Tc": infer_Tc,
        "infer_nu": infer_nu,
        "infer_beta": infer_beta,
        "reparametrize_Tc": reparametrize_Tc,
        "tc_delta_scale": TC_DELTA_SCALE if tc_delta_scale is None else tc_delta_scale,
        "likelihood_backend": likelihood_backend,
    }
    if discrepancy_form is not None:
        kwargs["discrepancy_form"] = discrepancy_form
    if gp_ell is not None:
        kwargs["gp_ell"] = gp_ell
    if correction_gp_ell is not None:
        kwargs["correction_gp_ell"] = correction_gp_ell
    if gp_ell_factor is not None:
        kwargs["gp_ell_factor"] = gp_ell_factor
    if gp_eta is not None:
        kwargs["gp_eta"] = gp_eta
    if correction_gp_ell_factor is not None:
        kwargs["correction_gp_ell_factor"] = correction_gp_ell_factor
    if correction_gp_eta is not None:
        kwargs["correction_gp_eta"] = correction_gp_eta
    if gp_kernel is not None:
        kwargs["gp_kernel"] = normalize_gp_kernel(gp_kernel)
    if obs_sigma_scale is not None:
        kwargs["obs_sigma_scale"] = obs_sigma_scale
    return FssProfileConfig(**kwargs)
