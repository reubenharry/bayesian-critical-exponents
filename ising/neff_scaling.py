"""Dynamical finite-size scaling model for Monte Carlo n_eff(L, t).

For fixed (or linearly scaled) sweep budget N:

    τ_int(L, t) ∼ A L^{z_dyn} f(z_coll),  z_coll = t L^{1/ν},  t = (T - T_c)/T_c

and n_eff ≈ N / (2 τ_int), so

    log(n_eff / N) + z_dyn log L ≈ g(z_coll)

with g a 1D GP on the collapse coordinate.  T_c and ν are fixed at the exact
2D Ising values by default.

Inference (Blackjax NUTS):
  - Fixed: GP length-scale ell and amplitude eta (FSS-style heuristics)
  - Sampled: dynamical exponent z_dyn > 0 and shared observation noise σ
  - Marginalised: latent GP g (analytic GP marginal likelihood)

Priors:
  - z_dyn ~ Uniform(z_dyn_min, z_dyn_max)
  - σ ~ HalfNormal(σ_scale)  with σ_scale = noise_frac * eta  (noise level prior)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from .constants import NU_EXACT, TC_EXACT
from .gp_jax import gp_log_marginal_likelihood as gp_log_marginal_likelihood_jax
from .gp_kernels import DEFAULT_GP_KERNEL, GpKernelKind, normalize_gp_kernel
from .gp_utils import gp_log_marginal_likelihood as gp_lml_np
from .gp_utils import gp_posterior_predictive
from .jax_config import configure_jax
from .scaling_function import provisional_z

NeffChannel = Literal["binder", "m", "m2", "m4", "susceptibility", "log_m"]

_CHANNEL_COLUMNS: dict[NeffChannel, str] = {
    "binder": "n_eff_binder",
    "m": "n_eff",
    "m2": "n_eff_m2",
    "m4": "n_eff_m4",
    "susceptibility": "n_eff_susceptibility",
    "log_m": "n_eff_log",
}

DEFAULT_GP_ELL_FACTOR = 0.15
DEFAULT_NOISE_FRAC = 0.1
_LOG_SQRT_2_OVER_PI = float(0.5 * np.log(2.0 / np.pi))


@dataclass(frozen=True)
class NeffScalingFit:
    """Fitted dynamical scaling surrogate for n_eff.

    ``z_dyn`` / ``noise_sigma`` are posterior means.  ``ell`` and ``eta`` are
    fixed.  The latent GP is the analytic conditional at those means.
    """

    z_dyn: float
    length_scale: float
    amplitude: float
    noise_sigma: float
    kernel: GpKernelKind
    T_c: float
    nu: float
    channel: NeffChannel
    z_coll_train: np.ndarray
    y_collapse_train: np.ndarray
    L_train: np.ndarray
    T_train: np.ndarray
    n_eff_train: np.ndarray
    n_sweeps_train: np.ndarray
    log_marginal_likelihood: float
    z_dyn_samples: np.ndarray
    z_dyn_mean: float
    z_dyn_std: float
    z_dyn_hdi_low: float
    z_dyn_hdi_high: float
    z_dyn_map: float
    noise_sigma_samples: np.ndarray
    noise_sigma_mean: float
    noise_sigma_std: float
    noise_sigma_hdi_low: float
    noise_sigma_hdi_high: float
    noise_sigma_map: float
    noise_prior_scale: float
    sampler: str

    @property
    def z_train(self) -> np.ndarray:
        """Alias for collapse coordinates (legacy name)."""
        return self.z_coll_train


def neff_column(channel: NeffChannel = "binder") -> str:
    try:
        return _CHANNEL_COLUMNS[channel]
    except KeyError as exc:
        raise ValueError(
            f"Unknown n_eff channel {channel!r}; expected one of "
            f"{sorted(_CHANNEL_COLUMNS)}"
        ) from exc


def collapse_response(
    n_eff: np.ndarray,
    L: np.ndarray,
    n_sweeps: np.ndarray,
    z_dyn: float,
) -> np.ndarray:
    """Collapsed target: log(n_eff / N) + z_dyn log L."""
    n_eff = np.asarray(n_eff, dtype=np.float64).ravel()
    L = np.asarray(L, dtype=np.float64).ravel()
    n_sweeps = np.asarray(n_sweeps, dtype=np.float64).ravel()
    return np.log(np.maximum(n_eff, 1.0) / np.maximum(n_sweeps, 1.0)) + float(
        z_dyn
    ) * np.log(np.maximum(L, 1.0))


def collapse_coordinates(
    fit: NeffScalingFit,
    L: np.ndarray,
    T: np.ndarray,
    *,
    n_eff: np.ndarray | None = None,
    n_sweeps: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (z_coll, optional collapsed response if n_eff/n_sweeps given)."""
    z = provisional_z(L, T, T_c=fit.T_c, nu=fit.nu)
    if n_eff is None or n_sweeps is None:
        return z, None
    y = collapse_response(n_eff, L, n_sweeps, fit.z_dyn)
    return z, y


def _extract_arrays(
    df: pd.DataFrame,
    *,
    channel: NeffChannel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    col = neff_column(channel)
    required = {"L", "T", col, "n_sweeps"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns for n_eff fit: {sorted(missing)}")
    L = df["L"].to_numpy(dtype=np.float64)
    T = df["T"].to_numpy(dtype=np.float64)
    n_eff = df[col].to_numpy(dtype=np.float64)
    n_sweeps = df["n_sweeps"].to_numpy(dtype=np.float64)
    mask = np.isfinite(L) & np.isfinite(T) & np.isfinite(n_eff) & np.isfinite(n_sweeps)
    mask &= (n_eff > 0) & (n_sweeps > 0) & (L > 0)
    if not np.any(mask):
        raise ValueError("No finite positive (L, T, n_eff, n_sweeps) rows to fit")
    return L[mask], T[mask], n_eff[mask], n_sweeps[mask]


def _provisional_z_dyn(
    L: np.ndarray,
    n_eff: np.ndarray,
    n_sweeps: np.ndarray,
) -> float:
    log_L = np.log(L)
    log_rate = np.log(np.maximum(n_eff, 1.0) / np.maximum(n_sweeps, 1.0))
    if np.unique(L).size >= 2:
        slope = float(np.polyfit(log_L, log_rate, 1)[0])
        return float(np.clip(-slope, 1e-3, 2.5))
    return 0.2


def fixed_gp_kernel_hyperparams(
    L: np.ndarray,
    T: np.ndarray,
    n_eff: np.ndarray,
    n_sweeps: np.ndarray,
    *,
    T_c: float,
    nu: float,
    gp_ell_factor: float = DEFAULT_GP_ELL_FACTOR,
    length_scale: float | None = None,
    amplitude: float | None = None,
) -> tuple[float, float]:
    """Fixed ell / eta (not sampled)."""
    z_coll = provisional_z(L, T, T_c=T_c, nu=nu)
    z_span = float(np.ptp(z_coll)) if z_coll.size > 1 else 1.0
    if not np.isfinite(z_span) or z_span <= 0.0:
        z_span = 1.0
    ell = float(length_scale) if length_scale is not None else gp_ell_factor * z_span

    z_dyn0 = _provisional_z_dyn(L, n_eff, n_sweeps)
    y0 = collapse_response(n_eff, L, n_sweeps, z_dyn0)
    amp = float(amplitude) if amplitude is not None else max(float(np.std(y0)), 1e-2)
    return ell, amp


def _hdi(samples: np.ndarray, *, prob: float = 0.94) -> tuple[float, float]:
    s = np.sort(np.asarray(samples, dtype=np.float64).ravel())
    n = s.size
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        return float(s[0]), float(s[0])
    k = max(int(np.ceil(prob * n)), 1)
    if k >= n:
        return float(s[0]), float(s[-1])
    widths = s[k - 1 :] - s[: n - k + 1]
    i = int(np.argmin(widths))
    return float(s[i]), float(s[i + k - 1])


def _logit_from_bounded(value: float, lower: float, upper: float) -> float:
    p = (float(value) - lower) / (upper - lower)
    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    return float(np.log(p / (1.0 - p)))


def _sample_z_dyn_noise_nuts(
    *,
    z_coll: np.ndarray,
    L: np.ndarray,
    n_eff: np.ndarray,
    n_sweeps: np.ndarray,
    length_scale: float,
    amplitude: float,
    noise_prior_scale: float,
    noise_init: float,
    kernel: GpKernelKind,
    z_dyn_min: float,
    z_dyn_max: float,
    z_dyn_init: float,
    chains: int,
    tune: int,
    draws: int,
    seed: int,
    target_accept: float,
    jitter: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Blackjax NUTS samples of (z_dyn, σ); each array shape chains × draws."""
    import blackjax
    from blackjax.util import run_inference_algorithm

    configure_jax()

    z_coll_j = jnp.asarray(z_coll, dtype=jnp.float64)
    L_j = jnp.asarray(L, dtype=jnp.float64)
    n_eff_j = jnp.asarray(n_eff, dtype=jnp.float64)
    n_sweeps_j = jnp.asarray(n_sweeps, dtype=jnp.float64)
    log_L_j = jnp.log(jnp.maximum(L_j, 1.0))
    log_rate_j = jnp.log(jnp.maximum(n_eff_j, 1.0) / jnp.maximum(n_sweeps_j, 1.0))
    lower = float(z_dyn_min)
    upper = float(z_dyn_max)
    ell = float(length_scale)
    amp = float(amplitude)
    sigma_scale = float(noise_prior_scale)
    n_obs = int(z_coll_j.shape[0])

    def logdensity(theta: jax.Array) -> jax.Array:
        u_z = theta[0]
        u_log_sigma = theta[1]
        p = jax.nn.sigmoid(u_z)
        z_dyn = lower + (upper - lower) * p
        sigma = jnp.exp(u_log_sigma)
        y = log_rate_j + z_dyn * log_L_j
        sigma_vec = jnp.full((n_obs,), sigma, dtype=jnp.float64)
        lml = gp_log_marginal_likelihood_jax(
            z_coll_j,
            y,
            sigma_vec,
            length_scale=ell,
            amplitude=amp,
            kernel=kernel,
            jitter=jitter,
        )
        # Uniform prior on z_dyn ∈ (lower, upper).
        log_jac_z = jnp.log(upper - lower) - jax.nn.softplus(-u_z) - jax.nn.softplus(u_z)
        # HalfNormal(σ; scale) + log|dσ/du| = u_log_sigma.
        log_prior_sigma = (
            _LOG_SQRT_2_OVER_PI
            - jnp.log(sigma_scale)
            - 0.5 * (sigma / sigma_scale) ** 2
            + u_log_sigma
        )
        return lml + log_jac_z + log_prior_sigma

    init = jnp.array(
        [
            _logit_from_bounded(z_dyn_init, lower, upper),
            np.log(max(float(noise_init), 1e-6)),
        ],
        dtype=jnp.float64,
    )
    init_stack = jnp.stack([init for _ in range(chains)], axis=0)

    def _adapt_and_sample(rng_key: jax.Array, position: jax.Array) -> jax.Array:
        adapt_key, sample_key = jax.random.split(rng_key)
        warmup = blackjax.window_adaptation(
            blackjax.nuts,
            logdensity,
            target_acceptance_rate=target_accept,
            progress_bar=False,
        )
        adapt_result, _ = warmup.run(adapt_key, position, num_steps=tune)
        nuts = blackjax.nuts(logdensity, **adapt_result.parameters)
        _, history = run_inference_algorithm(
            sample_key,
            nuts,
            num_steps=draws,
            initial_state=adapt_result.state,
            progress_bar=False,
            transform=lambda state, _info: state.position,
        )
        return history

    keys = jax.random.split(jax.random.key(seed), chains)
    history = jax.jit(
        jax.vmap(lambda key, position: _adapt_and_sample(key, position))
    )(keys, init_stack)
    history.block_until_ready()
    theta = np.asarray(history, dtype=np.float64)  # chains × draws × 2
    u_z = theta[..., 0]
    u_log_sigma = theta[..., 1]
    p = 1.0 / (1.0 + np.exp(-u_z))
    z_dyn = lower + (upper - lower) * p
    sigma = np.exp(u_log_sigma)
    return z_dyn, sigma


def fit_neff_scaling(
    df: pd.DataFrame,
    *,
    channel: NeffChannel = "binder",
    T_c: float = TC_EXACT,
    nu: float = NU_EXACT,
    kernel: GpKernelKind | str = DEFAULT_GP_KERNEL,
    jitter: float = 1e-6,
    gp_ell_factor: float = DEFAULT_GP_ELL_FACTOR,
    noise_frac: float = DEFAULT_NOISE_FRAC,
    length_scale: float | None = None,
    amplitude: float | None = None,
    noise_prior_scale: float | None = None,
    z_dyn_min: float = 1e-3,
    z_dyn_max: float = 2.0,
    chains: int = 4,
    tune: int = 500,
    draws: int = 1000,
    seed: int = 0,
    target_accept: float = 0.9,
) -> NeffScalingFit:
    """NUTS on (z_dyn, σ); ell/eta fixed; latent GP analytically marginalised."""
    kernel_kind = normalize_gp_kernel(kernel) if isinstance(kernel, str) else kernel
    L, T, n_eff, n_sweeps = _extract_arrays(df, channel=channel)
    z_coll = provisional_z(L, T, T_c=T_c, nu=nu)
    ell, amp = fixed_gp_kernel_hyperparams(
        L,
        T,
        n_eff,
        n_sweeps,
        T_c=T_c,
        nu=nu,
        gp_ell_factor=gp_ell_factor,
        length_scale=length_scale,
        amplitude=amplitude,
    )
    sigma_scale = (
        float(noise_prior_scale)
        if noise_prior_scale is not None
        else max(noise_frac * amp, 1e-3)
    )
    z_dyn_init = _provisional_z_dyn(L, n_eff, n_sweeps)
    noise_init = sigma_scale

    z_dyn_chains, sigma_chains = _sample_z_dyn_noise_nuts(
        z_coll=z_coll,
        L=L,
        n_eff=n_eff,
        n_sweeps=n_sweeps,
        length_scale=ell,
        amplitude=amp,
        noise_prior_scale=sigma_scale,
        noise_init=noise_init,
        kernel=kernel_kind,
        z_dyn_min=z_dyn_min,
        z_dyn_max=z_dyn_max,
        z_dyn_init=z_dyn_init,
        chains=chains,
        tune=tune,
        draws=draws,
        seed=seed,
        target_accept=target_accept,
        jitter=jitter,
    )
    z_samples = z_dyn_chains.reshape(-1)
    s_samples = sigma_chains.reshape(-1)
    z_dyn_mean = float(np.mean(z_samples))
    z_dyn_std = float(np.std(z_samples, ddof=1))
    z_hdi_low, z_hdi_high = _hdi(z_samples, prob=0.94)
    noise_mean = float(np.mean(s_samples))
    noise_std = float(np.std(s_samples, ddof=1))
    n_hdi_low, n_hdi_high = _hdi(s_samples, prob=0.94)

    # Joint MAP proxy on a thinned draw set (max LML + log prior).
    thin = max(z_samples.size // 400, 1)
    idx = np.arange(0, z_samples.size, thin)
    z_thin = z_samples[idx]
    s_thin = s_samples[idx]
    log_prior = (
        _LOG_SQRT_2_OVER_PI
        - np.log(sigma_scale)
        - 0.5 * (s_thin / sigma_scale) ** 2
    )
    lml = np.empty(z_thin.size, dtype=np.float64)
    for i, (z_dyn, sigma) in enumerate(zip(z_thin, s_thin, strict=True)):
        y = collapse_response(n_eff, L, n_sweeps, float(z_dyn))
        sigma_arr = np.full(z_coll.size, float(sigma), dtype=np.float64)
        lml[i] = gp_lml_np(
            z_coll,
            y,
            sigma_arr,
            length_scale=ell,
            amplitude=amp,
            kernel=kernel_kind,
            jitter=jitter,
        )
    map_local = int(np.argmax(lml + log_prior))
    z_dyn_map = float(z_thin[map_local])
    noise_map = float(s_thin[map_local])

    y_collapse = collapse_response(n_eff, L, n_sweeps, z_dyn_mean)
    sigma_arr = np.full(z_coll.size, noise_mean, dtype=np.float64)
    lml_at_mean = float(
        gp_lml_np(
            z_coll,
            y_collapse,
            sigma_arr,
            length_scale=ell,
            amplitude=amp,
            kernel=kernel_kind,
            jitter=jitter,
        )
    )

    return NeffScalingFit(
        z_dyn=z_dyn_mean,
        length_scale=ell,
        amplitude=amp,
        noise_sigma=noise_mean,
        kernel=kernel_kind,
        T_c=float(T_c),
        nu=float(nu),
        channel=channel,
        z_coll_train=z_coll.copy(),
        y_collapse_train=y_collapse.copy(),
        L_train=L.copy(),
        T_train=T.copy(),
        n_eff_train=n_eff.copy(),
        n_sweeps_train=n_sweeps.copy(),
        log_marginal_likelihood=lml_at_mean,
        z_dyn_samples=z_samples.copy(),
        z_dyn_mean=z_dyn_mean,
        z_dyn_std=z_dyn_std,
        z_dyn_hdi_low=z_hdi_low,
        z_dyn_hdi_high=z_hdi_high,
        z_dyn_map=z_dyn_map,
        noise_sigma_samples=s_samples.copy(),
        noise_sigma_mean=noise_mean,
        noise_sigma_std=noise_std,
        noise_sigma_hdi_low=n_hdi_low,
        noise_sigma_hdi_high=n_hdi_high,
        noise_sigma_map=noise_map,
        noise_prior_scale=sigma_scale,
        sampler="blackjax_nuts",
    )


def predict_collapse_gp(
    fit: NeffScalingFit,
    z_new: np.ndarray,
    *,
    jitter: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior mean and std of g(z_coll) = log(n_eff/N) + z_dyn log L."""
    sigma_train = np.full(fit.y_collapse_train.size, fit.noise_sigma, dtype=np.float64)
    return gp_posterior_predictive(
        fit.z_coll_train,
        fit.y_collapse_train,
        sigma_train,
        z_new,
        length_scale=fit.length_scale,
        amplitude=fit.amplitude,
        kernel=fit.kernel,
        jitter=jitter,
    )


def predict_neff(
    fit: NeffScalingFit,
    L: np.ndarray,
    T: np.ndarray,
    *,
    n_sweeps: float | np.ndarray,
    jitter: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict n_eff mean and approximate 1σ (log→linear) at (L, T) for given N."""
    L = np.asarray(L, dtype=np.float64).ravel()
    T = np.asarray(T, dtype=np.float64).ravel()
    n_sweeps_arr = np.broadcast_to(
        np.asarray(n_sweeps, dtype=np.float64).ravel(), L.shape
    ).copy()
    z = provisional_z(L, T, T_c=fit.T_c, nu=fit.nu)
    g_mean, g_std = predict_collapse_gp(fit, z, jitter=jitter)
    log_rate_mean = g_mean - fit.z_dyn * np.log(np.maximum(L, 1.0))
    n_eff_mean = n_sweeps_arr * np.exp(log_rate_mean)
    n_eff_std = n_eff_mean * g_std
    return n_eff_mean, n_eff_std


# ---------------------------------------------------------------------------
# Online AL planners (score-time ESS; never use full-chain oracle n_eff)
# ---------------------------------------------------------------------------

DEFAULT_COLD_START_NEFF_RATE = 0.12  # n_eff / N when GP cannot be fit yet
DEFAULT_ONLINE_MIN_POINTS = 6
COST_L_REF = 64.0


@dataclass(frozen=True)
class NeffRatePlanner:
    """Score-time n_eff planner for budget AL."""

    kind: Literal["gp", "cold_start"]
    fit: NeffScalingFit | None = None
    cold_rate: float = DEFAULT_COLD_START_NEFF_RATE
    n_train: int = 0

    def predict_n_eff_full(
        self,
        L: float | np.ndarray,
        T: float | np.ndarray,
        *,
        n_sweeps_full: float,
    ) -> float:
        """Predicted Binder n_eff at full sweep budget ``n_sweeps_full``."""
        if self.kind == "cold_start" or self.fit is None:
            return max(float(self.cold_rate) * float(n_sweeps_full), 1.0)
        mean, _ = predict_neff(
            self.fit,
            np.asarray([L], dtype=np.float64),
            np.asarray([T], dtype=np.float64),
            n_sweeps=float(n_sweeps_full),
        )
        return max(float(mean[0]), 1.0)


def write_neff_fit_summary(fit: NeffScalingFit, path: Path) -> None:
    """Compact text summary of an online / offline NeffScalingFit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"channel={fit.channel}",
        f"n_points={fit.z_coll_train.size}",
        f"sampler={fit.sampler}",
        f"z_dyn_mean={fit.z_dyn_mean:.6f}",
        f"z_dyn_std={fit.z_dyn_std:.6f}",
        f"z_dyn_map={fit.z_dyn_map:.6f}",
        f"noise_sigma_mean={fit.noise_sigma_mean:.6f}",
        f"noise_sigma_std={fit.noise_sigma_std:.6f}",
        f"length_scale_fixed={fit.length_scale:.6g}",
        f"amplitude_fixed={fit.amplitude:.6g}",
        f"lml_at_mean_params={fit.log_marginal_likelihood:.6f}",
        f"T_c={fit.T_c:.10g}",
        f"nu={fit.nu:.10g}",
        f"kernel={fit.kernel}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_neff_train_csv(df: pd.DataFrame, path: Path, *, channel: NeffChannel = "binder") -> None:
    """Persist prefix rows used for an online n_eff fit (for stepper diagnostics)."""
    col = neff_column(channel)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if df is None or df.empty or col not in df.columns:
        pd.DataFrame(columns=["L", "T", col, "n_sweeps"]).to_csv(path, index=False)
        return
    cols = ["L", "T", col]
    if "n_sweeps" in df.columns:
        cols.append("n_sweeps")
    out = df.loc[:, cols].copy()
    out.to_csv(path, index=False)


def parse_neff_fit_summary(path: Path) -> dict[str, str]:
    """Parse ``neff_scaling_fit.txt`` into a string dict."""
    path = Path(path)
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, _, val = line.partition("=")
        out[key.strip()] = val.strip()
    return out


def neff_fit_at_fixed_params(
    df: pd.DataFrame,
    *,
    z_dyn: float,
    noise_sigma: float,
    length_scale: float | None = None,
    amplitude: float | None = None,
    channel: NeffChannel = "binder",
    T_c: float = TC_EXACT,
    nu: float = NU_EXACT,
    kernel: GpKernelKind | str = DEFAULT_GP_KERNEL,
    gp_ell_factor: float = DEFAULT_GP_ELL_FACTOR,
    jitter: float = 1e-6,
) -> NeffScalingFit:
    """Build a NeffScalingFit at fixed (z_dyn, σ) without NUTS (stepper / replay)."""
    kernel_kind = normalize_gp_kernel(kernel) if isinstance(kernel, str) else kernel
    L, T, n_eff, n_sweeps = _extract_arrays(df, channel=channel)
    z_coll = provisional_z(L, T, T_c=T_c, nu=nu)
    ell, amp = fixed_gp_kernel_hyperparams(
        L,
        T,
        n_eff,
        n_sweeps,
        T_c=T_c,
        nu=nu,
        gp_ell_factor=gp_ell_factor,
        length_scale=length_scale,
        amplitude=amplitude,
    )
    z_dyn_f = float(z_dyn)
    noise_f = float(noise_sigma)
    y_collapse = collapse_response(n_eff, L, n_sweeps, z_dyn_f)
    sigma_arr = np.full(z_coll.size, noise_f, dtype=np.float64)
    lml = float(
        gp_lml_np(
            z_coll,
            y_collapse,
            sigma_arr,
            length_scale=ell,
            amplitude=amp,
            kernel=kernel_kind,
            jitter=jitter,
        )
    )
    empty = np.array([], dtype=np.float64)
    return NeffScalingFit(
        z_dyn=z_dyn_f,
        length_scale=ell,
        amplitude=amp,
        noise_sigma=noise_f,
        kernel=kernel_kind,
        T_c=float(T_c),
        nu=float(nu),
        channel=channel,
        z_coll_train=z_coll.copy(),
        y_collapse_train=y_collapse.copy(),
        L_train=L.copy(),
        T_train=T.copy(),
        n_eff_train=n_eff.copy(),
        n_sweeps_train=n_sweeps.copy(),
        log_marginal_likelihood=lml,
        z_dyn_samples=empty,
        z_dyn_mean=z_dyn_f,
        z_dyn_std=0.0,
        z_dyn_hdi_low=z_dyn_f,
        z_dyn_hdi_high=z_dyn_f,
        z_dyn_map=z_dyn_f,
        noise_sigma_samples=empty,
        noise_sigma_mean=noise_f,
        noise_sigma_std=0.0,
        noise_sigma_hdi_low=noise_f,
        noise_sigma_hdi_high=noise_f,
        noise_sigma_map=noise_f,
        noise_prior_scale=noise_f,
        sampler="fixed_params",
    )


def build_online_neff_planner(
    current: pd.DataFrame,
    *,
    channel: NeffChannel = "binder",
    min_points: int = DEFAULT_ONLINE_MIN_POINTS,
    cold_rate: float = DEFAULT_COLD_START_NEFF_RATE,
    seed: int = 0,
    chains: int = 2,
    tune: int = 200,
    draws: int = 400,
) -> NeffRatePlanner:
    """Fit n_eff GP from materialized prefixes, or cold-start if too few rows."""
    col = neff_column(channel)
    if current is None or current.empty or col not in current.columns:
        return NeffRatePlanner(kind="cold_start", cold_rate=cold_rate, n_train=0)

    work = current.copy()
    if "n_sweeps" not in work.columns:
        raise ValueError("current observables need an n_sweeps column for online n_eff fit")
    mask = (
        np.isfinite(work[col].to_numpy(dtype=np.float64))
        & (work[col].to_numpy(dtype=np.float64) > 0)
        & (work["n_sweeps"].to_numpy(dtype=np.float64) > 0)
    )
    work = work.loc[mask].reset_index(drop=True)
    n_train = int(len(work))
    if n_train < int(min_points):
        return NeffRatePlanner(
            kind="cold_start", cold_rate=cold_rate, n_train=n_train
        )

    fit = fit_neff_scaling(
        work,
        channel=channel,
        chains=chains,
        tune=tune,
        draws=draws,
        seed=seed,
    )
    return NeffRatePlanner(kind="gp", fit=fit, cold_rate=cold_rate, n_train=n_train)


def chunk_compute_cost(L: float | int, *, L_ref: float = COST_L_REF) -> float:
    """Dimensionless computational cost of one chunk at lattice size L (~ L^2)."""
    return (float(L) / float(L_ref)) ** 2
