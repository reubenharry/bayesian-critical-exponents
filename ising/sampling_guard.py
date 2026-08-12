"""Abort PyMC sampling when projected runtime exceeds a budget."""

from __future__ import annotations

import time


class SamplingTimeGuard:
    """Raise ``KeyboardInterrupt`` if estimated remaining sampling time is too large."""

    def __init__(
        self,
        *,
        chains: int,
        tune: int,
        draws: int,
        max_eta_seconds: float = 3600.0,
        min_completed: int = 20,
        log_every: int = 50,
    ) -> None:
        self.total = chains * (tune + draws)
        self.max_eta_seconds = max_eta_seconds
        self.min_completed = min_completed
        # Early iterations (especially during tuning) give unreliable ETAs.
        self._eta_check_start = chains * tune + min_completed
        self.log_every = log_every
        self.completed = 0
        self.start = time.perf_counter()
        self._last_log = 0

    def __call__(self, *, trace, draw) -> None:  # noqa: ARG002
        del trace
        self.completed += 1
        if self.completed < self._eta_check_start:
            return

        elapsed = time.perf_counter() - self.start
        rate = self.completed / elapsed
        remaining = self.total - self.completed
        eta = remaining / rate

        if (
            self.log_every > 0
            and self.completed - self._last_log >= self.log_every
        ):
            self._last_log = self.completed
            print(
                f"[sampling] {self.completed}/{self.total} iterations, "
                f"ETA {eta / 60.0:.1f} min",
                flush=True,
            )

        if eta > self.max_eta_seconds:
            raise KeyboardInterrupt(
                f"Aborting fit: estimated remaining time "
                f"{eta / 3600.0:.2f} h exceeds limit "
                f"{self.max_eta_seconds / 3600.0:.2f} h "
                f"({self.completed}/{self.total} iterations completed)"
            )
