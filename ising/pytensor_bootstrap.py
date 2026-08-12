"""PyTensor import-time configuration for pip/uv installs without system BLAS."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

_BLAS_TEST_CODE = """\
extern "C" double ddot_(int*, double*, int*, double*, int*);
int main() {
    int Nx = 5, Sx = 1;
    double x[5] = {0, 1, 2, 3, 4};
    double r = ddot_(&Nx, x, &Sx, x, &Sx);
    return ((r - 30.) > 1e-6 || (r - 30.) < -1e-6) ? -1 : 0;
}
"""


def _sanitize_pytensor_flags(flags: str) -> str:
    """Remove malformed ``blas__ldflags`` entries from ``PYTENSOR_FLAGS``."""
    if "blas__ldflags" not in flags:
        return flags

    # Commas inside ldflags break PyTensor's flag parser (e.g. -Wl,-rpath,path).
    if "-Wl" in flags or re.search(r",rpath", flags):
        parts = [
            part
            for part in flags.split(",")
            if not part.strip().startswith("blas__ldflags")
        ]
        return ",".join(parts)

    return flags


def _try_blas_ldflags(ldflags: str, *, cxx: str = "g++") -> bool:
    """Return True if *ldflags* link a working Fortran BLAS ``ddot_``."""
    if not ldflags.strip():
        return True

    with tempfile.TemporaryDirectory(prefix="pytensor_blas_") as tmp:
        src = Path(tmp) / "test_blas.cpp"
        out = Path(tmp) / "test_blas"
        src.write_text(_BLAS_TEST_CODE, encoding="utf-8")
        try:
            subprocess.run(
                [cxx, str(src), *ldflags.split(), "-o", str(out)],
                check=True,
                capture_output=True,
            )
            subprocess.run([str(out)], check=True, capture_output=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False
    return True


def _blas_ldflag_candidates() -> list[str]:
    """Ordered BLAS linker flags to try (fast system libs first)."""
    return [
        "-lopenblas",
        "-lopenblas -lgfortran -lpthread",
        "-lblas -lgfortran",
    ]


def _resolve_blas_ldflags() -> str:
    """Pick ldflags for a working Fortran BLAS, or empty string for NumPy fallback."""
    for candidate in _blas_ldflag_candidates():
        if _try_blas_ldflags(candidate):
            return candidate
    return ""


def bootstrap_pytensor() -> None:
    flags = os.environ.get("PYTENSOR_FLAGS", "")
    cleaned = _sanitize_pytensor_flags(flags)

    if "blas__ldflags" not in cleaned:
        ldflags = _resolve_blas_ldflags()
        entry = f"blas__ldflags={ldflags}"
        cleaned = f"{cleaned},{entry}" if cleaned else entry

    if cleaned != flags:
        if cleaned:
            os.environ["PYTENSOR_FLAGS"] = cleaned
        else:
            os.environ.pop("PYTENSOR_FLAGS", None)


bootstrap_pytensor()
