"""Central JAX configuration (x64 numerics, device / GPU selection)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Literal

JaxPlatform = Literal["auto", "cpu", "gpu", "tpu"]

_DEFAULT_PLATFORM: JaxPlatform = "auto"
_CONFIGURED = False
_BOOTSTRAPPED = False


def _cuda_devices_visible() -> bool:
    """Return True when ``nvidia-smi`` reports at least one GPU."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return False
    try:
        proc = subprocess.run(
            [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0 and bool(proc.stdout.strip())


def force_jax_cpu(*, hide_gpus: bool = True) -> None:
    """Force JAX onto CPU even when ``jax[cuda]`` is installed.

    Must be called **before** ``import jax`` in the process. Optionally hides
    GPUs from CUDA so the plugin does not probe a broken driver at import time.
    """
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    os.environ["JAX_PLATFORMS"] = "cpu"
    if hide_gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


def resolve_jax_platform(platform: JaxPlatform | None = None) -> JaxPlatform:
    """Pick the JAX platform, falling back to CPU when no GPU is visible."""
    if platform is not None and platform != "auto":
        return platform

    env = os.environ.get("JAX_PLATFORM_NAME", "").strip().lower()
    if env in ("cpu", "gpu", "tpu"):
        return env  # type: ignore[return-value]

    env_plats = os.environ.get("JAX_PLATFORMS", "").strip().lower()
    if env_plats in ("cpu", "gpu", "tpu"):
        return env_plats  # type: ignore[return-value]
    if env_plats == "cuda":
        return "gpu"

    if _cuda_devices_visible():
        return "auto"
    return "cpu"


def apply_jax_platform(platform: JaxPlatform) -> None:
    """Set process env before the first ``import jax``."""
    if platform == "cpu":
        force_jax_cpu(hide_gpus=True)
    elif platform == "gpu":
        os.environ["JAX_PLATFORM_NAME"] = "gpu"
        os.environ["JAX_PLATFORMS"] = "cuda"
    elif platform == "tpu":
        os.environ["JAX_PLATFORM_NAME"] = "tpu"
        os.environ["JAX_PLATFORMS"] = "tpu"


def bootstrap_jax_platform(platform: JaxPlatform | None = None) -> JaxPlatform:
    """Apply platform env vars once, before JAX is first imported."""
    global _BOOTSTRAPPED
    resolved = resolve_jax_platform(platform)
    if "jax" not in sys.modules and resolved != "auto":
        apply_jax_platform(resolved)
    _BOOTSTRAPPED = True
    return resolved


def ensure_jax_cpu_backend() -> None:
    """Verify JAX is on CPU; raise with a clear message if not."""
    import jax

    configure_jax(platform="cpu")
    backend = jax.default_backend()
    if backend != "cpu":
        raise RuntimeError(
            f"Expected JAX CPU backend but got {backend!r}. "
            "Restart the kernel and import jax_config before JAX."
        )


def configure_jax(
    *,
    enable_x64: bool = True,
    platform: JaxPlatform | None = None,
) -> JaxPlatform:
    """Apply process-wide JAX settings once (safe to call repeatedly)."""
    global _CONFIGURED
    resolved = resolve_jax_platform(platform)
    if "jax" not in sys.modules and resolved != "auto":
        apply_jax_platform(resolved)

    import jax

    if enable_x64:
        jax.config.update("jax_enable_x64", True)

    _CONFIGURED = True
    return resolved


def jax_device_summary() -> str:
    """Human-readable active JAX backend and device list."""
    import jax

    configure_jax()
    backend = jax.default_backend()
    devices = ", ".join(str(d) for d in jax.devices())
    return f"backend={backend}, devices=[{devices}]"


def default_jax_platform() -> JaxPlatform:
    return resolve_jax_platform(None)


# Prefer CPU on login nodes / broken CUDA before jax[cuda] probes the driver.
bootstrap_jax_platform()
