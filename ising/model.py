"""Backward-compatible re-exports; use :mod:`observables` for new code."""

from __future__ import annotations

from .observables import resolve_log_m_observables

__all__ = ["resolve_log_m_observables"]
