"""NUMBA compilation helpers for PyMC / PyTensor FSS models."""

from __future__ import annotations

from functools import lru_cache

from pytensor.compile.mode import Mode
from pytensor.link.numba import NumbaLinker


@lru_cache(maxsize=1)
def numba_compile_mode() -> Mode:
    """Shared PyTensor compile mode using the Numba linker."""
    return Mode(linker=NumbaLinker())


def pymc_sample_compile_kwargs() -> dict[str, Mode]:
    """``compile_kwargs`` for :func:`pymc.sample` to evaluate logp with Numba."""
    return {"mode": numba_compile_mode()}
