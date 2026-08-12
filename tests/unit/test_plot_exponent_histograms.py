"""Tests for posterior exponent histogram plotting."""

from __future__ import annotations

from pathlib import Path

import arviz as az
import matplotlib

matplotlib.use("Agg")

import numpy as np
from ising.constants import NU_EXACT, TC_EXACT
from ising.plot_posterior import (
    _posterior_exponent_panel_specs,
    _trace_var_names,
    plot_exponent_histograms,
)


def _mock_idata(*, include_tc: bool = True, include_nu: bool = True) -> az.InferenceData:
    rng = np.random.default_rng(0)
    n_chain, n_draw = 2, 200
    posterior: dict[str, np.ndarray] = {
        "gamma": rng.normal(1.75, 0.02, size=(n_chain, n_draw)),
    }
    if include_tc:
        posterior["T_c"] = rng.normal(TC_EXACT, 1e-4, size=(n_chain, n_draw))
    if include_nu:
        posterior["nu"] = rng.normal(NU_EXACT, 0.005, size=(n_chain, n_draw))
    return az.from_dict(
        posterior=posterior,
        dims={name: ["chain", "draw"] for name in posterior},
        coords={
            "chain": np.arange(n_chain),
            "draw": np.arange(n_draw),
        },
    )


def test_plot_exponent_histograms_includes_tc_nu_heatmap(tmp_path: Path):
    idata = _mock_idata()
    path = plot_exponent_histograms(idata, tmp_path)
    assert path is not None
    assert path.is_file()


def test_plot_exponent_histograms_no_heatmap_when_tc_fixed(tmp_path: Path):
    idata = _mock_idata(include_tc=False)
    path = plot_exponent_histograms(idata, tmp_path)
    assert path is not None
    assert path.is_file()


def test_plot_exponent_histograms_includes_gp_hyperparams(tmp_path: Path):
    rng = np.random.default_rng(1)
    n_chain, n_draw = 4, 50
    posterior = {
        "T_c": rng.normal(TC_EXACT, 1e-4, size=(n_chain, n_draw)),
        "gp_ell": rng.lognormal(mean=-1.0, sigma=0.2, size=(n_chain, n_draw)),
        "gp_eta": np.abs(rng.normal(0.5, 0.05, size=(n_chain, n_draw))),
        "correction_gp_ell": rng.lognormal(mean=-1.2, sigma=0.2, size=(n_chain, n_draw)),
        "correction_gp_eta": np.abs(rng.normal(0.4, 0.05, size=(n_chain, n_draw))),
    }
    idata = az.from_dict(
        posterior=posterior,
        dims={name: ["chain", "draw"] for name in posterior},
        coords={
            "chain": np.arange(n_chain),
            "draw": np.arange(n_draw),
        },
        attrs={
            "model": "fss",
            "fss_correction": "True",
            "infer_gp_hyperparams": "True",
            "gp_ell_prior_mu": "-1.0",
            "correction_gp_ell_prior_mu": "-1.2",
            "gp_ell_fixed": "0.35",
            "gp_eta_fixed": "0.5",
        },
    )
    specs = _posterior_exponent_panel_specs(idata)
    spec_names = {name for name, *_ in specs}
    assert {"T_c", "gp_ell", "gp_eta", "correction_gp_ell", "correction_gp_eta"}.issubset(
        spec_names
    )
    assert "gp_ell" in _trace_var_names(idata)
    path = plot_exponent_histograms(idata, tmp_path)
    assert path is not None
    assert path.is_file()
