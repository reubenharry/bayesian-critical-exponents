"""Build the FSS PyMC model used by the ising pipeline."""

from __future__ import annotations

import pandas as pd

from ._deps import pm as _pm
from .model_fss import build_fss_model
from .scaling_function import ScalingFunctionConfig


def build_pipeline_model(
    observables: pd.DataFrame,
    *,
    use_m: bool = True,
    use_m2: bool = False,
    use_m4: bool = False,
    use_binder: bool = False,
    use_chi: bool = False,
    correction_m: bool = True,
    correction_m2: bool = True,
    correction_m4: bool = True,
    correction_binder: bool = True,
    correction_chi: bool = True,
    discrepancy_m: bool = False,
    discrepancy_m2: bool = False,
    discrepancy_m4: bool = False,
    discrepancy_binder: bool = False,
    discrepancy_chi: bool = False,
    infer_Tc: bool = False,
    infer_nu: bool = True,
    infer_beta: bool = False,
    infer_omega: bool = False,
    infer_gp_hyperparams: bool = False,
    gp_ell_factor: float | None = None,
    gp_eta: float | None = None,
    correction_gp_ell_factor: float | None = None,
    correction_gp_eta: float | None = None,
    gp_kernel: str | None = None,
    obs_sigma_scale: float | None = None,
    use_log_m: bool = True,
    reparametrize_Tc: bool = False,
    tc_reparam_ref: float | None = None,
    tc_delta_scale: float | None = None,
    **_ignored: object,
) -> _pm.Model:
    fss_kwargs: dict[str, object] = dict(
        use_m=use_m,
        use_m2=use_m2,
        use_m4=use_m4,
        use_binder=use_binder,
        use_chi=use_chi,
        scaling_config=ScalingFunctionConfig(backend="gp"),
        correction_m=correction_m,
        correction_m2=correction_m2,
        correction_m4=correction_m4,
        correction_binder=correction_binder,
        correction_chi=correction_chi,
        discrepancy_m=discrepancy_m,
        discrepancy_m2=discrepancy_m2,
        discrepancy_m4=discrepancy_m4,
        discrepancy_binder=discrepancy_binder,
        discrepancy_chi=discrepancy_chi,
        infer_Tc=infer_Tc,
        infer_nu=infer_nu,
        infer_beta=infer_beta,
        infer_omega=infer_omega,
        infer_gp_hyperparams=infer_gp_hyperparams,
        use_log_m=use_log_m,
        reparametrize_Tc=reparametrize_Tc,
    )
    if tc_reparam_ref is not None:
        fss_kwargs["tc_reparam_ref"] = tc_reparam_ref
    if tc_delta_scale is not None:
        fss_kwargs["tc_delta_scale"] = tc_delta_scale
    if gp_ell_factor is not None:
        fss_kwargs["gp_ell_factor"] = gp_ell_factor
    if gp_eta is not None:
        fss_kwargs["gp_eta"] = gp_eta
    if correction_gp_ell_factor is not None:
        fss_kwargs["correction_gp_ell_factor"] = correction_gp_ell_factor
    if correction_gp_eta is not None:
        fss_kwargs["correction_gp_eta"] = correction_gp_eta
    if gp_kernel is not None:
        fss_kwargs["gp_kernel"] = gp_kernel
    if obs_sigma_scale is not None:
        fss_kwargs["obs_sigma_scale"] = obs_sigma_scale
    return build_fss_model(observables, **fss_kwargs)  # type: ignore[arg-type]


# JAX branch entry point: same signature, backend-specific return later.
build_model = build_pipeline_model
