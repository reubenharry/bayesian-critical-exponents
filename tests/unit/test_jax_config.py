"""Tests for JAX platform bootstrap."""

from __future__ import annotations

import importlib
import sys

import pytest

jax_config = pytest.importorskip(
    "ising.jax_config",
    reason="analysis extra not installed",
)


def test_resolve_jax_platform_falls_back_to_cpu_without_gpus(monkeypatch) -> None:
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.setattr(jax_config, "_cuda_devices_visible", lambda: False)
    assert jax_config.resolve_jax_platform(None) == "cpu"
    assert jax_config.resolve_jax_platform("auto") == "cpu"


def test_resolve_jax_platform_keeps_auto_when_gpus_visible(monkeypatch) -> None:
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.setattr(jax_config, "_cuda_devices_visible", lambda: True)
    assert jax_config.resolve_jax_platform(None) == "auto"


def test_bootstrap_forces_cpu_before_jax_import(monkeypatch) -> None:
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(jax_config, "_cuda_devices_visible", lambda: False)

    if "jax" in sys.modules:
        pytest.skip("jax already imported in this process")

    resolved = jax_config.bootstrap_jax_platform()
    assert resolved == "cpu"
    assert jax_config.os.environ["JAX_PLATFORM_NAME"] == "cpu"
    assert jax_config.os.environ["JAX_PLATFORMS"] == "cpu"
    assert jax_config.os.environ["CUDA_VISIBLE_DEVICES"] == ""

    import jax

    assert jax.default_backend() == "cpu"

    # Reloading jax_config in the same process is not supported; leave jax imported.
    importlib.invalidate_caches()
