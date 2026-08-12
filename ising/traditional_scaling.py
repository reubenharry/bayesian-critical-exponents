"""Classical finite-size scaling analysis (Binder crossings + log-log power laws).

Independent of the GP / PyMC profile-likelihood pipeline. Estimates:

1. ``T_c`` from pairwise Binder-cumulant crossings ``U_4(L, T)``.
2. ``1/ν`` from the size dependence of crossing temperatures.
3. ``β/ν`` from ``|m| ~ L^{-β/ν}`` at ``T ≈ T_c``.
4. ``β`` via ``β = (β/ν) · ν`` once ``ν`` is known.

Also reports ``γ/ν`` from ``χ ~ L^{γ/ν}`` at ``T ≈ T_c`` and checks the
hyperscaling relation ``γ/ν = 2 + 2(β/ν)`` (``d = 2``).
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import make_smoothing_spline

from .constants import BETA_EXACT, GAMMA_EXACT, NU_EXACT, SPATIAL_DIMENSION, TC_EXACT
from .datasets import DatasetSpec, get_dataset, observables_path, resolve_paths

MIN_MAGNETIZATION = 1e-12
MIN_CHI = 1e-12


@dataclass(frozen=True)
class BinderCrossing:
    """Binder cumulant crossing temperature for one size pair."""

    L_small: int
    L_large: int
    T_cross: float


@dataclass(frozen=True)
class LogLogFit:
    """Weighted least-squares fit of ``log y = intercept + slope · log x``."""

    slope: float
    intercept: float
    slope_stderr: float
    intercept_stderr: float
    rvalue: float
    n_points: int


COLLAPSE_CHANNEL_SPECS: dict[str, tuple[str, str, str, int]] = {
    "m": ("magnetization", "magnetization_std", "n_eff", 1),
    "m2": ("m2", "m2_std", "n_eff_m2", 2),
    "m4": ("m4", "m4_std", "n_eff_m4", 4),
}


@dataclass(frozen=True)
class CollapseNuResult:
    """ν estimate from binned collapse χ² at fixed ``T_c`` and ``beta``."""

    nu_hat: float
    nu_stderr: float
    chi2_min: float
    chi2_exact: float
    nu_grid: np.ndarray
    delta_chi2: np.ndarray
    n_dof: int
    channels: tuple[str, ...]
    T_c: float
    beta: float

    @property
    def delta_chi2_exact(self) -> float:
        return self.chi2_exact - self.chi2_min


@dataclass(frozen=True)
class TraditionalScalingResult:
    """Summary of classical FSS estimates."""

    dataset: str
    T_c_binder: float
    T_c_binder_stderr: float
    nu_from_binder: float
    nu_from_binder_stderr: float
    beta_over_nu: float
    beta_over_nu_stderr: float
    gamma_over_nu: float
    gamma_over_nu_stderr: float
    beta: float
    beta_stderr: float
    crossings: tuple[BinderCrossing, ...]
    tc_binder_fit: LogLogFit | None
    magnetization_fit: LogLogFit
    susceptibility_fit: LogLogFit
    hyperscaling_residual: float

    @property
    def nu(self) -> float:
        return self.nu_from_binder

    @property
    def T_c(self) -> float:
        return self.T_c_binder


def load_observables_table(path: Path) -> pd.DataFrame:
    """Load observables CSV sorted by ``(L, T)``."""
    df = pd.read_csv(path)
    required = {"L", "T", "magnetization", "magnetization_std", "n_eff", "binder_cumulant"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Observables CSV missing columns: {sorted(missing)}")
    return df.sort_values(["L", "T"]).reset_index(drop=True)


def _sigma_from_std_n_eff(std: np.ndarray, n_eff: np.ndarray) -> np.ndarray:
    return std / np.sqrt(np.maximum(n_eff, 1.0))


def _interp_crossing_temperature(
    T_a: np.ndarray,
    y_a: np.ndarray,
    T_b: np.ndarray,
    y_b: np.ndarray,
) -> float | None:
    """Linear interpolation of the first sign change in ``y_a - y_b``."""
    T_min = max(float(np.min(T_a)), float(np.min(T_b)))
    T_max = min(float(np.max(T_a)), float(np.max(T_b)))
    if T_max <= T_min:
        return None

    n_grid = max(400, 4 * (len(T_a) + len(T_b)))
    T_grid = np.linspace(T_min, T_max, n_grid)
    ya = np.interp(T_grid, T_a, y_a)
    yb = np.interp(T_grid, T_b, y_b)
    diff = ya - yb
    sign = np.sign(diff)
    for i in range(len(sign) - 1):
        if sign[i] == 0.0:
            return float(T_grid[i])
        if sign[i] * sign[i + 1] < 0.0:
            t0, t1 = T_grid[i], T_grid[i + 1]
            d0, d1 = diff[i], diff[i + 1]
            return float(t0 - d0 * (t1 - t0) / (d1 - d0))
    return None


def binder_crossing_temperature(
    df: pd.DataFrame,
    L_small: int,
    L_large: int,
) -> float | None:
    """``T`` where ``U_4(L_small, T) = U_4(L_large, T)``."""
    if L_small == L_large:
        raise ValueError("L_small and L_large must differ")
    ga = df[df["L"] == L_small].sort_values("T")
    gb = df[df["L"] == L_large].sort_values("T")
    if ga.empty or gb.empty:
        return None
    return _interp_crossing_temperature(
        ga["T"].to_numpy(dtype=float),
        ga["binder_cumulant"].to_numpy(dtype=float),
        gb["T"].to_numpy(dtype=float),
        gb["binder_cumulant"].to_numpy(dtype=float),
    )


def all_binder_crossings(df: pd.DataFrame) -> list[BinderCrossing]:
    """All pairwise Binder crossings for distinct lattice sizes in ``df``."""
    sizes = sorted(int(L) for L in df["L"].unique())
    crossings: list[BinderCrossing] = []
    for i, L_small in enumerate(sizes):
        for L_large in sizes[i + 1 :]:
            T_cross = binder_crossing_temperature(df, L_small, L_large)
            if T_cross is not None:
                crossings.append(BinderCrossing(L_small, L_large, T_cross))
    return crossings


def _weighted_loglog_fit(
    x: np.ndarray,
    y: np.ndarray,
    sigma_y: np.ndarray | None = None,
) -> LogLogFit:
    """Fit ``log y = intercept + slope log x`` (optionally weighted)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size or x.size < 2:
        raise ValueError("Need at least two points for a log-log fit")
    if np.any(x <= 0.0) or np.any(y <= 0.0):
        raise ValueError("log-log fit requires strictly positive x and y")

    log_x = np.log(x)
    log_y = np.log(y)
    if sigma_y is None:
        res = stats.linregress(log_x, log_y)
        return LogLogFit(
            slope=float(res.slope),
            intercept=float(res.intercept),
            slope_stderr=float(res.stderr),
            intercept_stderr=float(res.intercept_stderr),
            rvalue=float(res.rvalue),
            n_points=int(x.size),
        )

    sigma_y = np.asarray(sigma_y, dtype=np.float64).ravel()
    rel_sigma = np.maximum(sigma_y / y, 1e-12)
    weights = 1.0 / rel_sigma**2
    coeffs, cov = np.polyfit(log_x, log_y, 1, w=weights, cov=True)
    slope, intercept = float(coeffs[0]), float(coeffs[1])
    fitted = intercept + slope * log_x
    ss_res = float(np.sum(weights * (log_y - fitted) ** 2))
    ss_tot = float(np.sum(weights * (log_y - np.average(log_y, weights=weights)) ** 2))
    rvalue = float(np.sqrt(max(0.0, 1.0 - ss_res / max(ss_tot, 1e-30))))
    return LogLogFit(
        slope=slope,
        intercept=intercept,
        slope_stderr=float(np.sqrt(cov[0, 0])),
        intercept_stderr=float(np.sqrt(cov[1, 1])),
        rvalue=rvalue,
        n_points=int(x.size),
    )


def _value_at_temperature(
    df: pd.DataFrame,
    L: int,
    T_target: float,
    *,
    value_col: str,
) -> float:
    g = df[df["L"] == L].sort_values("T")
    T = g["T"].to_numpy(dtype=float)
    y = g[value_col].to_numpy(dtype=float)
    return float(np.interp(T_target, T, y))


def _sigma_at_temperature(
    df: pd.DataFrame,
    L: int,
    T_target: float,
    *,
    value_col: str,
    std_col: str,
    n_eff_col: str,
) -> float:
    g = df[df["L"] == L].sort_values("T")
    T = g["T"].to_numpy(dtype=float)
    sigma = _sigma_from_std_n_eff(
        g[std_col].to_numpy(dtype=float),
        g[n_eff_col].to_numpy(dtype=float),
    )
    return float(np.interp(T_target, T, sigma))


def collapse_equivalence_groups(
    df: pd.DataFrame,
    spec: DatasetSpec,
) -> np.ndarray:
    """Group rows that should share the same scaling variable ``z`` at true ``nu``.

    For ``z_values`` grids (e.g. ``critical_fss``), rows with the same
    ``seed % len(z_values)`` were simulated at the same target ``z``.
    Otherwise group by rounded temperature (Cartesian ``L × T`` grids).
    """
    if spec.z_values is not None and "seed" in df.columns:
        n_z = len(spec.z_values)
        return (df["seed"].to_numpy(dtype=int) % n_z).astype(int)
    t_round = np.round(df["T"].to_numpy(dtype=np.float64), 4)
    _, groups = np.unique(t_round, return_inverse=True)
    return groups.astype(int)


def binned_collapse_chi2(
    nu: float,
    df: pd.DataFrame,
    group_ids: np.ndarray,
    *,
    T_c: float,
    beta: float,
    channels: tuple[str, ...] = ("m",),
) -> tuple[float, int]:
    """Weighted χ² measuring how well scaled observables collapse across ``L``.

    At each equivalence group (same target ``z`` or same ``T``), compute
    ``Phi = observable × L^{k beta/nu}`` and penalize scatter across lattice
    sizes relative to MC standard errors.
    """
    L = df["L"].to_numpy(dtype=np.float64)
    T = df["T"].to_numpy(dtype=np.float64)
    nu = float(nu)
    chi2 = 0.0
    n_dof = 0
    for name in channels:
        if name not in COLLAPSE_CHANNEL_SPECS:
            raise ValueError(f"Unknown collapse channel {name!r}")
        v_col, s_col, n_col, moment = COLLAPSE_CHANNEL_SPECS[name]
        exp = moment * beta / nu
        phi = df[v_col].to_numpy(dtype=np.float64) * L**exp
        sigma = (
            df[s_col].to_numpy(dtype=np.float64)
            / np.sqrt(np.maximum(df[n_col].to_numpy(dtype=np.float64), 1.0))
        ) * L**exp
        sigma = np.maximum(sigma, 1e-12)
        for g in np.unique(group_ids):
            mask = group_ids == g
            if int(mask.sum()) < 2:
                continue
            p = phi[mask]
            s = sigma[mask]
            weights = 1.0 / s**2
            mean = float(np.average(p, weights=weights))
            chi2 += float(np.sum(((p - mean) / s) ** 2))
            n_dof += int(mask.sum()) - 1
    return chi2, n_dof


def profile_nu_collapse_chi2(
    nu_grid: np.ndarray,
    df: pd.DataFrame,
    spec: DatasetSpec,
    *,
    T_c: float = TC_EXACT,
    beta: float = BETA_EXACT,
    channels: tuple[str, ...] = ("m", "m2", "m4"),
) -> tuple[np.ndarray, np.ndarray, int]:
    """Evaluate binned collapse χ² on a grid of trial ``nu`` values."""
    group_ids = collapse_equivalence_groups(df, spec)
    nu_grid = np.asarray(nu_grid, dtype=np.float64).ravel()
    chi2 = np.empty(nu_grid.size, dtype=np.float64)
    n_dof = 0
    for i, nu in enumerate(nu_grid):
        chi2[i], n_dof = binned_collapse_chi2(
            nu,
            df,
            group_ids,
            T_c=T_c,
            beta=beta,
            channels=channels,
        )
    return nu_grid, chi2, n_dof


def _chi2_curvature_stderr(
    fn: Callable[[float], float],
    x_hat: float,
    chi2_min: float,
    *,
    eps: float = 1e-4,
) -> float:
    """Asymptotic 1σ uncertainty from local curvature (Δχ² = 1)."""
    d2 = (fn(x_hat + eps) - 2.0 * chi2_min + fn(x_hat - eps)) / eps**2
    if d2 <= 0.0 or not np.isfinite(d2):
        return float("nan")
    return float(np.sqrt(2.0 / d2))


def estimate_nu_from_collapse_chi2(
    df: pd.DataFrame,
    spec: DatasetSpec,
    *,
    T_c: float = TC_EXACT,
    beta: float = BETA_EXACT,
    channels: tuple[str, ...] = ("m", "m2", "m4"),
    nu_bounds: tuple[float, float] = (0.85, 1.15),
    n_grid: int = 161,
    nu_exact: float = NU_EXACT,
) -> CollapseNuResult:
    """MLE for ``nu`` by minimizing binned collapse χ² at fixed ``T_c``, ``beta``."""
    from scipy.optimize import minimize_scalar

    group_ids = collapse_equivalence_groups(df, spec)

    def chi2(nu: float) -> float:
        value, _ = binned_collapse_chi2(
            nu,
            df,
            group_ids,
            T_c=T_c,
            beta=beta,
            channels=channels,
        )
        return value

    lo, hi = nu_bounds
    nu_grid = np.linspace(lo, hi, n_grid)
    chi2_grid = np.array([chi2(nu) for nu in nu_grid], dtype=np.float64)
    opt = minimize_scalar(chi2, bounds=(lo, hi), method="bounded")
    nu_hat = float(opt.x)
    chi2_min = float(opt.fun)
    chi2_exact = chi2(float(nu_exact))
    _, n_dof = binned_collapse_chi2(
        nu_hat,
        df,
        group_ids,
        T_c=T_c,
        beta=beta,
        channels=channels,
    )
    nu_stderr = _chi2_curvature_stderr(chi2, nu_hat, chi2_min)
    return CollapseNuResult(
        nu_hat=nu_hat,
        nu_stderr=nu_stderr,
        chi2_min=chi2_min,
        chi2_exact=chi2_exact,
        nu_grid=nu_grid,
        delta_chi2=chi2_grid - chi2_min,
        n_dof=n_dof,
        channels=channels,
        T_c=T_c,
        beta=beta,
    )


def plot_nu_collapse_chi2_profile(
    result: CollapseNuResult,
    path: Path | None = None,
    *,
    nu_exact: float = NU_EXACT,
    title: str | None = None,
) -> plt.Figure:
    """Plot Δχ²(ν) and annotate the local slope (curvature) at the minimum."""
    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    ax.plot(result.nu_grid, result.delta_chi2, color="#4c72b0", linewidth=2.0)
    ax.axvline(nu_exact, color="black", linestyle="--", linewidth=1.5, label=rf"exact $\nu={nu_exact:g}$")
    ax.axvline(
        result.nu_hat,
        color="#ffb000",
        linewidth=1.5,
        label=rf"MLE $\nu={result.nu_hat:.4f}$",
    )
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.7, label=r"$\Delta\chi^2=1$ (1$\sigma$)")
    ax.axhline(4.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.45, label=r"$\Delta\chi^2=4$ (2$\sigma$)")

    if np.isfinite(result.nu_stderr) and result.nu_stderr > 0.0:
        ax.axvspan(
            result.nu_hat - result.nu_stderr,
            result.nu_hat + result.nu_stderr,
            color="#ffb000",
            alpha=0.12,
            label=rf"curvature 1$\sigma$ ($\pm{result.nu_stderr:.4g}$)",
        )

    channel_label = ", ".join(
        {"m": r"$|m|$", "m2": r"$m^2$", "m4": r"$m^4$"}.get(c, c)
        for c in result.channels
    )
    ax.set_xlabel(r"$\nu$")
    ax.set_ylabel(r"$\Delta\chi^2 = \chi^2(\nu) - \chi^2_{\min}$")
    ax.set_title(
        title
        or (
            rf"Binned collapse $\chi^2$ profile ({channel_label}); "
            rf"$T_c={result.T_c:.5f}$, $\beta={result.beta:g}$ fixed"
        )
    )
    ax.annotate(
        rf"$\chi^2_{{\min}}={result.chi2_min:.1f}$, dof={result.n_dof}\n"
        rf"$\Delta\chi^2(\nu={nu_exact:g})={result.delta_chi2_exact:.1f}$",
        xy=(0.02, 0.98),
        xycoords="axes fraction",
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9),
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    return fig


def magnetization_collapse_arrays(
    df: pd.DataFrame,
    T_c: float,
    nu: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collapsed magnetization ``Phi_m = |m| L^(beta/nu)`` vs ``z = t L^(1/nu)``."""
    L = df["L"].to_numpy(dtype=np.float64)
    T = df["T"].to_numpy(dtype=np.float64)
    m = np.maximum(df["magnetization"].to_numpy(dtype=np.float64), MIN_MAGNETIZATION)
    sigma_m = _sigma_from_std_n_eff(
        df["magnetization_std"].to_numpy(dtype=np.float64),
        df["n_eff"].to_numpy(dtype=np.float64),
    )
    t = (T - T_c) / T_c
    z = t * L ** (1.0 / nu)
    phi = m * L ** (beta / nu)
    sigma_phi = sigma_m * L ** (beta / nu)
    return z, phi, sigma_phi, L


def _aggregate_collapse_clusters(
    z: np.ndarray,
    phi: np.ndarray,
    sigma_phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted-average collapsed points that share the same z-grid location."""
    z = np.asarray(z, dtype=np.float64).ravel()
    phi = np.asarray(phi, dtype=np.float64).ravel()
    sigma_phi = np.maximum(np.asarray(sigma_phi, dtype=np.float64).ravel(), 1e-12)
    order = np.argsort(z)
    z = z[order]
    phi = phi[order]
    sigma_phi = sigma_phi[order]

    gaps = np.diff(z)
    if gaps.size == 0:
        return z.copy(), phi.copy(), sigma_phi.copy()
    median_gap = float(np.median(gaps))
    between_cluster = gaps[gaps > 3.0 * median_gap]
    if between_cluster.size == 0:
        split_thresh = max(5.0 * median_gap, 1e-6)
    else:
        split_thresh = 0.5 * float(np.median(between_cluster))

    bin_id = np.zeros(z.size, dtype=int)
    current_bin = 0
    for i in range(1, z.size):
        if z[i] - z[i - 1] > split_thresh:
            current_bin += 1
        bin_id[i] = current_bin

    z_bins: list[float] = []
    phi_bins: list[float] = []
    sigma_bins: list[float] = []
    for b in range(int(bin_id.max()) + 1):
        mask = bin_id == b
        weights = 1.0 / sigma_phi[mask] ** 2
        z_bins.append(float(np.average(z[mask], weights=weights)))
        phi_bins.append(float(np.average(phi[mask], weights=weights)))
        sigma_bins.append(float(1.0 / np.sqrt(np.sum(weights))))
    return (
        np.asarray(z_bins, dtype=np.float64),
        np.asarray(phi_bins, dtype=np.float64),
        np.asarray(sigma_bins, dtype=np.float64),
    )


def fit_single_scaling_function_spline(
    z: np.ndarray,
    phi: np.ndarray,
    sigma_phi: np.ndarray,
    *,
    smoothing: float | None = None,
    n_eval: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth ``Phi(z)`` curve through cluster-averaged collapsed magnetization.

    Raw collapsed points sit on a discrete z-grid (one cluster per temperature,
    one point per L). Fitting a spline directly to all points with MC error
    weights over-interpolates between clusters; we first average within each
    cluster, then fit a penalized smoothing spline through the bin centres.
    """
    z_bins, phi_bins, sigma_bins = _aggregate_collapse_clusters(z, phi, sigma_phi)
    if z_bins.size < 4:
        raise ValueError("Need at least four z-clusters to fit a scaling-function spline")
    weights = 1.0 / np.maximum(sigma_bins, 1e-12) ** 2
    lam = smoothing if smoothing is not None else 0.05 * float(z_bins.size)
    spline = make_smoothing_spline(z_bins, phi_bins, w=weights, lam=lam)
    z_line = np.linspace(float(z_bins.min()), float(z_bins.max()), n_eval)
    phi_line = spline(z_line)
    return z_line, np.asarray(phi_line, dtype=np.float64)


def estimate_tc_from_binder_crossings(
    crossings: list[BinderCrossing],
) -> tuple[float, float]:
    """Mean and sample stderr of crossing temperatures."""
    if not crossings:
        raise ValueError("No Binder crossings found")
    temps = np.array([c.T_cross for c in crossings], dtype=np.float64)
    mean = float(np.mean(temps))
    stderr = float(np.std(temps, ddof=1) / np.sqrt(temps.size)) if temps.size > 1 else 0.0
    return mean, stderr


def estimate_nu_from_binder_crossings(
    crossings: list[BinderCrossing],
    T_c: float,
) -> tuple[float, float, LogLogFit | None]:
    """Fit ``T_cross - T_c ≈ b · L_small^{-1/ν}`` over Binder crossings."""
    shifts: list[float] = []
    L_vals: list[float] = []
    for cross in crossings:
        shift = cross.T_cross - T_c
        if shift > 0.0:
            shifts.append(shift)
            L_vals.append(float(cross.L_small))

    if len(shifts) < 2:
        return float("nan"), float("nan"), None

    x = np.asarray(L_vals, dtype=np.float64)
    y = np.asarray(shifts, dtype=np.float64)
    fit = _weighted_loglog_fit(x, y)
    nu = -1.0 / fit.slope
    nu_stderr = float(abs(fit.slope_stderr / (fit.slope**2)))
    return nu, nu_stderr, fit


def estimate_beta_over_nu_at_tc(
    df: pd.DataFrame,
    T_c: float,
) -> LogLogFit:
    """``|m| ~ L^{-β/ν}`` via log-log regression at ``T = T_c``."""
    sizes = sorted(int(L) for L in df["L"].unique())
    m_vals = []
    sigmas = []
    for L in sizes:
        m = _value_at_temperature(df, L, T_c, value_col="magnetization")
        m_vals.append(max(m, MIN_MAGNETIZATION))
        sigmas.append(
            _sigma_at_temperature(
                df,
                L,
                T_c,
                value_col="magnetization",
                std_col="magnetization_std",
                n_eff_col="n_eff",
            )
        )
    fit = _weighted_loglog_fit(
        np.asarray(sizes, dtype=float),
        np.asarray(m_vals, dtype=float),
        np.asarray(sigmas, dtype=float),
    )
    return LogLogFit(
        slope=-float(fit.slope),
        intercept=float(fit.intercept),
        slope_stderr=float(fit.slope_stderr),
        intercept_stderr=float(fit.intercept_stderr),
        rvalue=float(fit.rvalue),
        n_points=fit.n_points,
    )


def estimate_gamma_over_nu_at_tc(
    df: pd.DataFrame,
    T_c: float,
) -> LogLogFit:
    """``χ ~ L^{γ/ν}`` via log-log regression at ``T = T_c``."""
    if "susceptibility" not in df.columns:
        raise ValueError("Susceptibility column required for γ/ν estimate")
    sizes = sorted(int(L) for L in df["L"].unique())
    chi_vals = []
    sigmas = []
    for L in sizes:
        chi = _value_at_temperature(df, L, T_c, value_col="susceptibility")
        chi_vals.append(max(chi, MIN_CHI))
        sigmas.append(
            _sigma_at_temperature(
                df,
                L,
                T_c,
                value_col="susceptibility",
                std_col="susceptibility_std",
                n_eff_col="n_eff_susceptibility",
            )
        )
    return _weighted_loglog_fit(
        np.asarray(sizes, dtype=float),
        np.asarray(chi_vals, dtype=float),
        np.asarray(sigmas, dtype=float),
    )


def run_traditional_scaling_analysis(
    dataset: str,
    *,
    data: Path | None = None,
) -> TraditionalScalingResult:
    """Run Binder + log-log classical analysis on a named dataset."""
    path = data or observables_path(dataset)
    df = load_observables_table(path)
    crossings = all_binder_crossings(df)
    if not crossings:
        raise ValueError(f"No Binder crossings found in {path}")

    T_c, T_c_stderr = estimate_tc_from_binder_crossings(crossings)
    nu, nu_stderr, tc_shift_fit = estimate_nu_from_binder_crossings(crossings, T_c)
    m_fit = estimate_beta_over_nu_at_tc(df, T_c)
    beta_over_nu = m_fit.slope
    beta_over_nu_stderr = m_fit.slope_stderr

    chi_fit = estimate_gamma_over_nu_at_tc(df, T_c)
    gamma_over_nu = chi_fit.slope
    gamma_over_nu_stderr = chi_fit.slope_stderr

    beta = beta_over_nu * nu
    # Do not invent a combined SE via delta-method product of independent fits.
    # Quote β as the point product; uncertainty lives on (β/ν) and ν separately.
    beta_stderr = float("nan")

    hyperscaling_pred = SPATIAL_DIMENSION - 2.0 * beta_over_nu
    hyperscaling_residual = gamma_over_nu - hyperscaling_pred

    return TraditionalScalingResult(
        dataset=dataset,
        T_c_binder=T_c,
        T_c_binder_stderr=T_c_stderr,
        nu_from_binder=nu,
        nu_from_binder_stderr=nu_stderr,
        beta_over_nu=beta_over_nu,
        beta_over_nu_stderr=beta_over_nu_stderr,
        gamma_over_nu=gamma_over_nu,
        gamma_over_nu_stderr=gamma_over_nu_stderr,
        beta=beta,
        beta_stderr=beta_stderr,
        crossings=tuple(crossings),
        tc_binder_fit=tc_shift_fit,
        magnetization_fit=m_fit,
        susceptibility_fit=chi_fit,
        hyperscaling_residual=hyperscaling_residual,
    )


def format_traditional_scaling_summary(result: TraditionalScalingResult) -> str:
    """Human-readable summary with comparison to exact 2D Ising exponents."""
    lines = [
        f"Traditional FSS analysis ({result.dataset!r})",
        f"  Binder crossings ({len(result.crossings)} pairs):",
    ]
    for cross in result.crossings:
        lines.append(
            f"    L={cross.L_small} vs L={cross.L_large}: "
            f"T_cross={cross.T_cross:.6f}"
        )
    lines.extend(
        [
            f"  T_c (mean crossing) = {result.T_c_binder:.6f} ± {result.T_c_binder_stderr:.6f}"
            f"  (exact {TC_EXACT:.6f})",
            f"  ν (Binder shift)    = {result.nu_from_binder:.5f} ± "
            f"{result.nu_from_binder_stderr:.5f}  (exact {NU_EXACT:.5f})",
            f"  β/ν (|m| vs L)      = {result.beta_over_nu:.5f} ± "
            f"{result.beta_over_nu_stderr:.5f}  (exact {BETA_EXACT/NU_EXACT:.5f})",
            f"  β = (β/ν)·ν        = {result.beta:.5f}"
            f"  (exact {BETA_EXACT:.5f}; no combined SE)",
            f"  γ/ν (χ vs L)        = {result.gamma_over_nu:.5f} ± "
            f"{result.gamma_over_nu_stderr:.5f}  (exact {GAMMA_EXACT/NU_EXACT:.5f})",
            f"  hyperscaling check: γ/ν - (d - 2β/ν) = {result.hyperscaling_residual:+.5f}"
            f"  (d={SPATIAL_DIMENSION})",
        ]
    )
    return "\n".join(lines)


def _plot_binder_crossings(
    df: pd.DataFrame,
    result: TraditionalScalingResult,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for L in sorted(df["L"].unique()):
        g = df[df["L"] == L].sort_values("T")
        ax.plot(
            g["T"],
            g["binder_cumulant"],
            "o-",
            label=f"L={int(L)}",
            markersize=5,
        )
    for cross in result.crossings:
        ax.axvline(cross.T_cross, color="gray", linestyle=":", alpha=0.5, linewidth=1.0)
    ax.axvline(
        result.T_c_binder,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=rf"$T_c$ (mean cross) = {result.T_c_binder:.5f}",
    )
    ax.axvline(TC_EXACT, color="red", linestyle="-.", linewidth=1.2, label="exact $T_c$")
    ax.set_xlabel(r"$T$")
    ax.set_ylabel(r"$U_4 = 1 - \langle m^4\rangle / (3\langle m^2\rangle^2)$")
    ax.set_title("Binder cumulant crossings")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_loglog_at_tc(
    df: pd.DataFrame,
    result: TraditionalScalingResult,
    *,
    value_col: str,
    std_col: str,
    n_eff_col: str,
    ylabel: str,
    exponent_label: str,
    fit: LogLogFit,
    path: Path,
) -> None:
    sizes = sorted(int(L) for L in df["L"].unique())
    y = np.array(
        [
            max(
                _value_at_temperature(df, L, result.T_c_binder, value_col=value_col),
                MIN_MAGNETIZATION,
            )
            for L in sizes
        ],
        dtype=float,
    )
    sigma = np.array(
        [
            _sigma_at_temperature(
                df,
                L,
                result.T_c_binder,
                value_col=value_col,
                std_col=std_col,
                n_eff_col=n_eff_col,
            )
            for L in sizes
        ],
        dtype=float,
    )
    x = np.asarray(sizes, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    ax.errorbar(x, y, yerr=sigma, fmt="o", capsize=3, label="data at $T_c$")
    if "β/ν" in exponent_label:
        slope_plot = fit.slope
        fit_line = np.exp(fit.intercept - fit.slope * np.log(x))
    else:
        slope_plot = fit.slope
        fit_line = np.exp(fit.intercept + fit.slope * np.log(x))
    ax.plot(x, fit_line, "k--", label=rf"fit slope = {slope_plot:.4f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(ylabel)
    ax.set_title(rf"{ylabel} at $T_c={result.T_c_binder:.5f}$")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_magnetization_scaling_function(
    df: pd.DataFrame,
    *,
    T_c: float,
    nu: float,
    beta: float,
    path: Path,
    title_suffix: str = "",
) -> None:
    """Collapsed ``|m|`` with a single fitted universal ``Phi_m(z)`` curve."""
    z, phi, sigma_phi, L = magnetization_collapse_arrays(df, T_c, nu, beta)
    z_line, phi_line = fit_single_scaling_function_spline(z, phi, sigma_phi)

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for Lval in sorted(int(v) for v in np.unique(L)):
        mask = L == Lval
        ax.errorbar(
            z[mask],
            phi[mask],
            yerr=sigma_phi[mask],
            fmt="o",
            capsize=2,
            markersize=5,
            label=f"L={Lval}",
            alpha=0.9,
        )
    ax.plot(
        z_line,
        phi_line,
        color="black",
        linewidth=2.0,
        label=r"fitted $\Phi_m(z)$",
        zorder=0,
    )
    ax.axvline(0.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)
    ax.set_xlabel(r"$z = t\, L^{1/\nu}$")
    ax.set_ylabel(r"$\Phi_m = |m|\, L^{\beta/\nu}$")
    suffix = f" — {title_suffix}" if title_suffix else ""
    ax.set_title(
        rf"Magnetization scaling function{suffix} "
        rf"($T_c={T_c:.5f}$, $\nu={nu:.4g}$, $\beta={beta:.4g}$)"
    )
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_binder_shift_fit(
    result: TraditionalScalingResult,
    path: Path,
) -> None:
    if result.tc_binder_fit is None:
        return
    shifts = []
    L_vals = []
    for cross in result.crossings:
        shift = cross.T_cross - result.T_c_binder
        if shift > 0.0:
            shifts.append(shift)
            L_vals.append(float(cross.L_small))
    if len(shifts) < 2:
        return

    x = np.asarray(L_vals, dtype=float)
    y = np.asarray(shifts, dtype=float)
    fit = result.tc_binder_fit
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = np.exp(fit.intercept + fit.slope * np.log(x_line))

    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    ax.loglog(x, y, "o", label=r"$T_{\mathrm{cross}} - T_c$")
    ax.loglog(x_line, y_line, "k--", label=rf"slope $= -1/\nu$ ($\nu={result.nu_from_binder:.3f}$)")
    ax.set_xlabel(r"$L_{\mathrm{small}}$")
    ax.set_ylabel(r"$T_{\mathrm{cross}} - T_c$")
    ax.set_title("Binder crossing shift vs lattice size")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_traditional_scaling(
    df: pd.DataFrame,
    result: TraditionalScalingResult,
    out_dir: Path,
) -> list[Path]:
    """Write diagnostic figures for classical FSS analysis."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        out_dir / "traditional_binder_crossings.png",
        out_dir / "traditional_magnetization_loglog.png",
        out_dir / "traditional_susceptibility_loglog.png",
        out_dir / "traditional_binder_shift.png",
        out_dir / "traditional_scaling_function_fit.png",
        out_dir / "traditional_scaling_function_exact.png",
    ]
    _plot_binder_crossings(df, result, paths[0])
    _plot_loglog_at_tc(
        df,
        result,
        value_col="magnetization",
        std_col="magnetization_std",
        n_eff_col="n_eff",
        ylabel=r"$|m|$",
        exponent_label=r"$\beta/\nu$",
        fit=result.magnetization_fit,
        path=paths[1],
    )
    _plot_loglog_at_tc(
        df,
        result,
        value_col="susceptibility",
        std_col="susceptibility_std",
        n_eff_col="n_eff_susceptibility",
        ylabel=r"$\chi$",
        exponent_label=r"$\gamma/\nu$",
        fit=result.susceptibility_fit,
        path=paths[2],
    )
    _plot_binder_shift_fit(result, paths[3])
    _plot_magnetization_scaling_function(
        df,
        T_c=result.T_c_binder,
        nu=result.nu_from_binder,
        beta=result.beta,
        path=paths[4],
        title_suffix="classical estimates",
    )
    _plot_magnetization_scaling_function(
        df,
        T_c=TC_EXACT,
        nu=NU_EXACT,
        beta=BETA_EXACT,
        path=paths[5],
        title_suffix="exact exponents",
    )
    return paths


def run_traditional_scaling_diagnostics(
    dataset: str,
    *,
    data: Path | None = None,
    out: Path | None = None,
) -> tuple[TraditionalScalingResult, list[Path]]:
    """Analyze, print summary, and write plots."""
    path = data or observables_path(dataset)
    df = load_observables_table(path)
    result = run_traditional_scaling_analysis(dataset, data=path)
    print(format_traditional_scaling_summary(result))
    _, _, plots_path = resolve_paths(dataset, model="fss")
    out_dir = out or (plots_path / "traditional_scaling")
    plot_paths = plot_traditional_scaling(df, result, out_dir)
    for plot_path in plot_paths:
        print(f"Wrote {plot_path}")
    return result, plot_paths


def main(argv: list[str] | None = None) -> TraditionalScalingResult:
    parser = argparse.ArgumentParser(
        description="Classical Binder + log-log FSS analysis for Ising observables."
    )
    parser.add_argument("--dataset", default="critical_large_z")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Plot output directory (default: plots/<dataset>/traditional_scaling)",
    )
    args = parser.parse_args(argv)
    result, _ = run_traditional_scaling_diagnostics(
        args.dataset,
        data=args.data,
        out=args.out,
    )
    return result


if __name__ == "__main__":
    main()
